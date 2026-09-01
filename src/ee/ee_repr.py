#!/usr/bin/env python3
"""nero 双臂 关节 → 末端位姿 表示层。批量 FK + rot6D/rotvec 互转。

坐标约定(全部沿用 fk_nero.py, 别自己另起一套)
  · 世界系(base="default", 含底座位姿), 两臂共用同一个世界系 —— 双臂任务必须如此,
    否则左右臂的位置不可比。
  · TCP = TCP_FINGERTIP = 0.138(指尖对称轴最末端), 与用户给的取姿方法逐字一致:
        T = fk(q_rad, side="right", tcp=TCP_FINGERTIP)
  · rot6D = 旋转矩阵**前两列**展平 [R[:,0], R[:,1]](Zhou et al. 2019 的连续表示),
    恢复用 Gram-Schmidt。用列不用行, 全流程统一。

20 维绝对末端状态布局(observation.state / parquet 里的 action 都是这个):
   [0:3]   左臂 位置 xyz (m, 世界系)
   [3:6]   左臂 R 第 0 列
   [6:9]   左臂 R 第 1 列
   [9]     左臂 夹爪开度 (m, 原样搬运)
   [10:13] 右臂 位置
   [13:16] 右臂 R 第 0 列
   [16:19] 右臂 R 第 1 列
   [19]    右臂 夹爪开度

14 维相对动作布局(由 EEDeltaActions 在训练管线里现算, 不落盘):
   [0:3]   左臂 Δ位置 = p_i - p_0            (世界系, chunk 锚点 = 首帧 state)
   [3:6]   左臂 rotvec(R_i @ R_0^T)          (世界系相对旋转, 轴角)
   [6]     左臂 夹爪开度 (绝对)
   [7:10]  右臂 Δ位置
   [10:13] 右臂 rotvec
   [13]    右臂 夹爪开度 (绝对)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "urdf_export"))

import fk_nero  # noqa: E402
from fk_nero import TCP_FINGERTIP, TCP_GRIP_CENTER, fk  # noqa: E402  (re-export)

STATE_DIM = 20        # 绝对末端: (3 pos + 6 rot6d + 1 grip) x 2 臂
REL_ACTION_DIM = 14   # 相对动作: (3 dpos + 3 rotvec + 1 grip) x 2 臂

STATE_NAMES = [
    "left_ee_x", "left_ee_y", "left_ee_z",
    "left_rot6d_0", "left_rot6d_1", "left_rot6d_2",
    "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",
    "left_gripper_width",
    "right_ee_x", "right_ee_y", "right_ee_z",
    "right_rot6d_0", "right_rot6d_1", "right_rot6d_2",
    "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",
    "right_gripper_width",
]
REL_ACTION_NAMES = [
    "left_dx", "left_dy", "left_dz",
    "left_drot_x", "left_drot_y", "left_drot_z",
    "left_gripper_width",
    "right_dx", "right_dy", "right_dz",
    "right_drot_x", "right_drot_y", "right_drot_z",
    "right_gripper_width",
]

# 原始 16 维关节布局里的切片
J_LEFT = slice(0, 7)
G_LEFT = 7
J_RIGHT = slice(8, 15)
G_RIGHT = 15


# ---------------------------------------------------------------- 批量 FK
def _chain(side: str, link: str | None = None):
    """从 fk_nero 的 URDF 索引里抽出 link→root 的关节链(已 reverse 成 root→link)。"""
    link = link or f"{side}_gripper_base"
    chain, cur = [], link
    while cur in fk_nero._BY_CHILD:  # noqa: SLF001  (故意复用, 保证与 fk() 同源)
        j = fk_nero._BY_CHILD[cur]
        chain.append(j)
        cur = j.find("parent").get("link")
    steps = []
    for j in reversed(chain):
        ax = j.find("axis")
        steps.append((
            fk_nero._origin_T(j.find("origin")),  # noqa: SLF001
            None if ax is None else np.array([float(v) for v in ax.get("xyz").split()]),
            j.get("type"),
            j.get("name"),
        ))
    return steps


_CHAIN_CACHE: dict[str, list] = {}


def _axis_rot_batch(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """(N,) 角度 → (N,3,3)。与 fk_nero._axis_rot 同公式(Rodrigues)。"""
    a = axis / (np.linalg.norm(axis) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s = np.sin(ang)[:, None, None]
    c = (1.0 - np.cos(ang))[:, None, None]
    return np.eye(3)[None] + s * K[None] + c * (K @ K)[None]


def fk_batch(q: np.ndarray, side: str, tcp: float = TCP_FINGERTIP,
             base: str | None = "default") -> np.ndarray:
    """q: (N,7) 弧度 → (N,4,4)。与 fk_nero.fk 逐元素等价(见 selftest)。"""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[None]
    n = q.shape[0]
    steps = _CHAIN_CACHE.setdefault(side, _chain(side))
    qidx = {f"{side}_joint{i + 1}": i for i in range(7)}

    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for T0, axis, jtype, name in steps:
        Tj = np.broadcast_to(T0, (n, 4, 4)).copy()
        if axis is not None and jtype in ("revolute", "continuous"):
            # fk() 里 qmap.get(name, 0.0): 不在 q 里的关节角一律 0
            ang = q[:, qidx[name]] if name in qidx else np.zeros(n)
            Rq = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            Rq[:, :3, :3] = _axis_rot_batch(axis, ang)
            Tj = Tj @ Rq
        elif axis is not None and jtype == "prismatic":
            d = q[:, qidx[name]] if name in qidx else np.zeros(n)
            Tq = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            Tq[:, :3, 3] = axis[None] * d[:, None]
            Tj = Tj @ Tq
        T = T @ Tj
    if tcp:
        off = np.eye(4)
        off[2, 3] = tcp
        T = T @ off[None]
    if base is None:
        return T
    B = fk_nero.base_T() if base == "default" else np.asarray(base, float)
    return B[None] @ T


# ------------------------------------------------------- rot6D / rotvec
def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(...,3,3) → (...,6): 前两列展平。"""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_mat(r6: np.ndarray) -> np.ndarray:
    """(...,6) → (...,3,3): Gram-Schmidt 正交化(rot6D 的标准恢复)。"""
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """(...,3,3) → (...,3) 轴角。数值稳定版, 支持 |θ| 接近 0 与 π。"""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    tr = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=-1)  # = 2 sinθ · axis
    out = np.zeros((R.shape[0], 3))
    small = theta < 1e-6
    out[small] = 0.5 * w[small]                       # sinθ≈θ 时的一阶近似
    mid = ~small & (theta < math.pi - 1e-4)
    out[mid] = (theta[mid] / (2.0 * np.sin(theta[mid])))[:, None] * w[mid]
    near_pi = theta >= math.pi - 1e-4                 # sinθ→0: 从 (R+I)/2 ≈ aaᵀ 取轴
    if near_pi.any():
        A = 0.5 * (R[near_pi] + np.eye(3)[None])
        k = np.argmax(np.diagonal(A, axis1=1, axis2=2), axis=1)
        axis = A[np.arange(A.shape[0]), :, k]
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        # θ=π 处 v 与 -v 等价, 符号无法定; 用 w 的残余符号定向, w≈0 时取 +。
        sgn = np.sign(np.einsum("ij,ij->i", axis, w[near_pi]))
        sgn[sgn == 0] = 1.0
        out[near_pi] = (theta[near_pi] * sgn)[:, None] * axis
    return out.reshape(*shp, 3)


def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    """(...,3) 轴角 → (...,3,3)。"""
    v = np.asarray(v, dtype=np.float64)
    shp = v.shape[:-1]
    v = v.reshape(-1, 3)
    th = np.linalg.norm(v, axis=-1)
    a = np.where(th[:, None] > 1e-12, v / np.where(th[:, None] == 0, 1, th[:, None]), 0.0)
    K = np.zeros((v.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -a[:, 2], a[:, 1]
    K[:, 1, 0], K[:, 1, 2] = a[:, 2], -a[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -a[:, 1], a[:, 0]
    R = np.eye(3)[None] + np.sin(th)[:, None, None] * K + \
        (1 - np.cos(th))[:, None, None] * (K @ K)
    return R.reshape(*shp, 3, 3)


# ------------------------------------------------- 16 维关节 → 20 维末端
def joints16_to_ee20(s16: np.ndarray, tcp: float = TCP_FINGERTIP) -> np.ndarray:
    """(N,16) 关节+爪宽 → (N,20) 绝对末端(pos + rot6d + grip) x 2 臂。"""
    s16 = np.asarray(s16, dtype=np.float64)
    if s16.ndim == 1:
        s16 = s16[None]
    Tl = fk_batch(s16[:, J_LEFT], "left", tcp=tcp)
    Tr = fk_batch(s16[:, J_RIGHT], "right", tcp=tcp)
    return np.concatenate([
        Tl[:, :3, 3], mat_to_rot6d(Tl[:, :3, :3]), s16[:, [G_LEFT]],
        Tr[:, :3, 3], mat_to_rot6d(Tr[:, :3, :3]), s16[:, [G_RIGHT]],
    ], axis=-1)


def ee20_split(s20: np.ndarray):
    """(N,20) → (pos_l, R_l, grip_l, pos_r, R_r, grip_r)。"""
    s20 = np.asarray(s20, dtype=np.float64)
    return (s20[..., 0:3], rot6d_to_mat(s20[..., 3:9]), s20[..., 9],
            s20[..., 10:13], rot6d_to_mat(s20[..., 13:19]), s20[..., 19])


def ee20_to_rel14(state20: np.ndarray, actions20: np.ndarray,
                  frame: str = "world") -> np.ndarray:
    """chunk 锚定的相对动作。state20 (...,20) 是锚点(首帧), actions20 (...,H,20)。

    frame="world" (默认, 基座/世界系):
        Δp = p_i − p_0            Δr = rotvec(R_i @ R_0ᵀ)
    frame="tool" (工具系, 末端自身坐标系):
        Δp = R_0ᵀ (p_i − p_0)     Δr = rotvec(R_0ᵀ @ R_i)
    两者爪宽都保持绝对。默认 world —— 工具系下推理时要用估计的 R̂_0 把 Δp 转回世界系,
    姿态估计误差会整条旋歪平移指令(0.16m 位移 × 3° ≈ 8mm 侧偏), 世界系没这条串扰。
    """
    if frame not in ("world", "tool"):
        raise ValueError(f"frame 只能是 world / tool, 实得 {frame!r}")
    state20 = np.asarray(state20, dtype=np.float64)
    actions20 = np.asarray(actions20, dtype=np.float64)
    p0l, R0l, _, p0r, R0r, _ = ee20_split(state20)
    pil, Ril, gl, pir, Rir, gr = ee20_split(actions20)
    if actions20.ndim > state20.ndim:
        p0l, p0r = p0l[..., None, :], p0r[..., None, :]
        R0l, R0r = R0l[..., None, :, :], R0r[..., None, :, :]
    dl, dr = pil - p0l, pir - p0r
    if frame == "world":
        wl = mat_to_rotvec(Ril @ np.swapaxes(R0l, -1, -2))
        wr = mat_to_rotvec(Rir @ np.swapaxes(R0r, -1, -2))
    else:
        dl = (np.swapaxes(R0l, -1, -2) @ dl[..., None])[..., 0]
        dr = (np.swapaxes(R0r, -1, -2) @ dr[..., None])[..., 0]
        wl = mat_to_rotvec(np.swapaxes(R0l, -1, -2) @ Ril)
        wr = mat_to_rotvec(np.swapaxes(R0r, -1, -2) @ Rir)
    return np.concatenate([dl, wl, gl[..., None], dr, wr, gr[..., None]], axis=-1)


def rel14_to_ee20(state20: np.ndarray, rel14: np.ndarray,
                  frame: str = "world") -> np.ndarray:
    """ee20_to_rel14 的逆:相对动作 → 绝对末端位姿(推理侧用)。frame 必须与正变换一致。"""
    if frame not in ("world", "tool"):
        raise ValueError(f"frame 只能是 world / tool, 实得 {frame!r}")
    state20 = np.asarray(state20, dtype=np.float64)
    rel14 = np.asarray(rel14, dtype=np.float64)
    p0l, R0l, _, p0r, R0r, _ = ee20_split(state20)
    if rel14.ndim > state20.ndim:
        p0l, p0r = p0l[..., None, :], p0r[..., None, :]
        R0l, R0r = R0l[..., None, :, :], R0r[..., None, :, :]
    dl, dr = rel14[..., 0:3], rel14[..., 7:10]
    if frame == "world":
        Rl = rotvec_to_mat(rel14[..., 3:6]) @ R0l
        Rr = rotvec_to_mat(rel14[..., 10:13]) @ R0r
    else:
        dl = (R0l @ dl[..., None])[..., 0]
        dr = (R0r @ dr[..., None])[..., 0]
        Rl = R0l @ rotvec_to_mat(rel14[..., 3:6])
        Rr = R0r @ rotvec_to_mat(rel14[..., 10:13])
    return np.concatenate([
        p0l + dl, mat_to_rot6d(Rl), rel14[..., 6:7],
        p0r + dr, mat_to_rot6d(Rr), rel14[..., 13:14],
    ], axis=-1)


# ---------------------------------------------------------------- selftest
def selftest(n: int = 64, seed: int = 0) -> bool:
    """批量 FK 必须与 fk_nero.fk 逐元素一致; rot6d/rotvec 必须可逆。"""
    rng = np.random.default_rng(seed)
    lim = fk_nero.limits("right")
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(("✓" if cond else "✗"), name, " ", detail, flush=True)
        ok = ok and bool(cond)

    for side in ("left", "right"):
        q = rng.uniform(lim[:, 0], lim[:, 1], size=(n, 7))
        Tb = fk_batch(q, side)
        Ts = np.stack([fk(q[i], side=side, tcp=TCP_FINGERTIP) for i in range(n)])
        chk(f"fk_batch({side}) ≡ fk_nero.fk", float(np.abs(Tb - Ts).max()) < 1e-9,
            f"max|Δ|={np.abs(Tb - Ts).max():.2e}")

    q = rng.uniform(lim[:, 0], lim[:, 1], size=(n, 7))
    R = fk_batch(q, "right")[:, :3, :3]
    chk("rot6d 往返", float(np.abs(rot6d_to_mat(mat_to_rot6d(R)) - R).max()) < 1e-12,
        f"max|Δ|={np.abs(rot6d_to_mat(mat_to_rot6d(R)) - R).max():.2e}")
    chk("rotvec 往返", float(np.abs(rotvec_to_mat(mat_to_rotvec(R)) - R).max()) < 1e-9,
        f"max|Δ|={np.abs(rotvec_to_mat(mat_to_rotvec(R)) - R).max():.2e}")
    # 小角度 & 近 π 的边界
    for th in (0.0, 1e-9, 1e-3, math.pi - 1e-6, math.pi - 1e-9):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        Rt = rotvec_to_mat((th * ax)[None])
        back = mat_to_rotvec(Rt)[0]
        err = min(np.linalg.norm(back - th * ax), np.linalg.norm(back + th * ax))
        chk(f"rotvec 边界 θ={th:.3e}", err < 1e-5, f"err={err:.2e}")

    # rel14 往返: 随机 state + chunk
    s20 = joints16_to_ee20(np.concatenate([
        rng.uniform(lim[:, 0], lim[:, 1], size=(8, 7)), rng.uniform(0, .12, (8, 1)),
        rng.uniform(lim[:, 0], lim[:, 1], size=(8, 7)), rng.uniform(0, .12, (8, 1))], 1))
    a20 = joints16_to_ee20(np.concatenate([
        rng.uniform(lim[:, 0], lim[:, 1], size=(8 * 5, 7)), rng.uniform(0, .12, (8 * 5, 1)),
        rng.uniform(lim[:, 0], lim[:, 1], size=(8 * 5, 7)), rng.uniform(0, .12, (8 * 5, 1))], 1)
    ).reshape(8, 5, 20)
    for frame in ("world", "tool"):
        rel = ee20_to_rel14(s20, a20, frame=frame)
        back = rel14_to_ee20(s20, rel, frame=frame)
        chk(f"rel14 往返 ≡ 原绝对位姿 (frame={frame})",
            float(np.abs(back - a20).max()) < 1e-8,
            f"max|Δ|={np.abs(back - a20).max():.2e}")
    # 两个帧必须真的不同(否则说明 frame 开关没生效)
    dw = ee20_to_rel14(s20, a20, frame="world")
    dt = ee20_to_rel14(s20, a20, frame="tool")
    chk("world / tool 两帧确实不同", float(np.abs(dw - dt).max()) > 1e-3,
        f"max|Δ|={np.abs(dw - dt).max():.3f}")
    print("\nSELFTEST PASS" if ok else "\nSELFTEST FAILED", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
