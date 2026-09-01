#!/usr/bin/env python3
"""NERO 7 自由度臂逆运动学:末端位姿 → 关节角。阻尼最小二乘 + 零空间拉回种子。

为什么必须有零空间项(本任务的核心风险)
--------------------------------------
末端位姿只有 6 自由度,臂有 7 个关节 → 有 1 维冗余(肘部可以绕"肩-腕轴"自转而末端不动)。
封盖任务 71% 的 chunk 里有一条臂**本该完全不动**,模型对它输出 Δ≈0,还原出的目标位姿
就是"当前位姿";如果 IK 随手挑一个零空间解,**末端没动而肘部会甩**。关节版直接命令绝对
关节角,没这个问题;换末端位姿表示就必须自己把这条堵上。

三道防线(与 训练前须知_ee.md §7.2 一一对应):
  a. 零空间正则把解拉回种子(种子 = 上一次的关节指令 / 当前实测关节角)  ← 本文件
  b. 静止臂死区: 整段 Δ 低于阈值就冻结关节指令, 根本不调 IK          ← 本文件 arm_is_idle
  c. 上机前离线回放对齐录制关节角                                    ← replay_ee_ik.py

坐标/TCP 口径与 ee_repr.py / fk_nero.py 完全同源(世界系, TCP=指尖 0.138), 不另起一套。

自测: python ik_nero.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ee_repr as E  # noqa: E402
import fk_nero  # noqa: E402
from ee_repr import TCP_FINGERTIP  # noqa: E402

# ------------------------------------------------- 实测关节包络(放宽 URDF 软限位)
# URDF 里的限位偏保守, 真机能走到它外面 2~3.5°。示教录到的构型是**物理上真达到过**的,
# 按 URDF 裁剪会让那些位姿在数学上不可达(实测离线回放 0.86% 的步因此解不出来, 40/40 都是
# 卡在同一关节)。所以 IK 用 union(URDF, 数据集实测 min/max) ± slack 作为有效限位。
# 包络由 measure_joint_envelope.py 量出并落盘, 可审计、可被真机固件限位覆盖。
# 想强制回到纯 URDF 限位: 环境变量 NERO_IK_URDF_LIMITS_ONLY=1。
# 包络文件: 先找与本文件同源的 ../assets(本地/服务器都适用), 再退回服务器绝对路径。
_ENV_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "assets" / "joint_envelope.json",
    Path("<EE_PACK>/assets/joint_envelope.json"),
]
ENVELOPE_PATH = next((p for p in _ENV_CANDIDATES if p.exists()), _ENV_CANDIDATES[0])
_ENV_CACHE: dict = {}
_ENV_NOTED = False


def effective_limits(side: str) -> np.ndarray:
    """(7,2) 有效关节限位。有包络文件就用放宽后的, 否则退回纯 URDF 限位。"""
    global _ENV_NOTED
    if os.environ.get("NERO_IK_URDF_LIMITS_ONLY") == "1":
        return fk_nero.limits(side)
    if side not in _ENV_CACHE:
        try:
            d = json.loads(ENVELOPE_PATH.read_text())["arms"][side]
            _ENV_CACHE[side] = np.stack([np.asarray(d["effective_lower"], float),
                                         np.asarray(d["effective_upper"], float)], axis=1)
            if not _ENV_NOTED:
                u = fk_nero.limits(side)
                w = np.degrees(np.maximum(u[:, 0] - _ENV_CACHE[side][:, 0],
                                          _ENV_CACHE[side][:, 1] - u[:, 1])).max()
                print(f"[ik_nero] 用实测包络放宽限位 (最大 {w:.2f}°) ← {ENVELOPE_PATH}",
                      flush=True)
                _ENV_NOTED = True
        except (OSError, KeyError, ValueError):
            _ENV_CACHE[side] = fk_nero.limits(side)
            if not _ENV_NOTED:
                print(f"[ik_nero] ⚠ 无包络文件, 退回纯 URDF 限位 ({ENVELOPE_PATH})", flush=True)
                _ENV_NOTED = True
    return _ENV_CACHE[side]



# --------------------------------------------------------- FK + 几何 Jacobian
def fk_with_jacobian(q: np.ndarray, side: str, tcp: float = TCP_FINGERTIP):
    """q: (7,) 弧度 → (T_world 4x4, J 6x7)。

    J 是**世界系**的几何 Jacobian(上 3 行平移 / 下 3 行角速度), 与相对动作所在的
    坐标系一致。第 i 列: J_v = z_i × (p_ee − o_i), J_w = z_i, 其中 z_i/o_i 是第 i 个
    关节轴在世界系下的方向与原点。
    """
    q = np.asarray(q, dtype=np.float64).reshape(7)
    steps = E._CHAIN_CACHE.setdefault(side, E._chain(side))  # noqa: SLF001 复用同一份链
    qidx = {f"{side}_joint{i + 1}": i for i in range(7)}

    T = fk_nero.base_T()                      # 世界系起点(含底座位姿)
    axes = [np.zeros(3)] * 7
    origins = [np.zeros(3)] * 7
    for T0, axis, jtype, name in steps:
        T = T @ T0
        if axis is None:
            continue
        if jtype in ("revolute", "continuous"):
            if name in qidx:
                a = axis / (np.linalg.norm(axis) or 1.0)
                axes[qidx[name]] = T[:3, :3] @ a          # 轴方向在 T0 之后的帧里
                origins[qidx[name]] = T[:3, 3].copy()
                ang = q[qidx[name]]
            else:
                ang = 0.0
            Rq = np.eye(4)
            Rq[:3, :3] = fk_nero._axis_rot(axis, ang)     # noqa: SLF001 同源公式
            T = T @ Rq
        elif jtype == "prismatic":
            d = q[qidx[name]] if name in qidx else 0.0
            Tq = np.eye(4)
            Tq[:3, 3] = np.asarray([float(v) for v in axis]) * d
            T = T @ Tq
    if tcp:
        off = np.eye(4)
        off[2, 3] = tcp
        T = T @ off

    p_ee = T[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        J[0:3, i] = np.cross(axes[i], p_ee - origins[i])
        J[3:6, i] = axes[i]
    return T, J


# ------------------------------------------------------------------- 单臂 IK
def solve_arm_ik(T_target: np.ndarray, q_seed: np.ndarray, side: str, *,
                 q_ref: np.ndarray | None = None,   # 零空间锚点, 默认 = q_seed
                 tcp: float = TCP_FINGERTIP,
                 damping: float = 1e-2,       # λ: 阻尼, 越大越稳越慢
                 ns_gain: float = 0.30,       # 零空间拉向 q_ref 的强度(0 = 不拉)
                 w_rot: float = 1.0,          # 旋转误差权重(1 rad ≈ 1 m)
                 max_iter: int = 100,
                 tol_pos: float = 1e-4,       # 0.1 mm
                 tol_rot: float = 1e-3,       # ≈0.057°
                 max_step: float = 0.20,      # 任务项单步最大关节增量(rad), 防跳
                 ns_max_step: float = 0.05,   # 零空间项单步上限, 必须 << max_step
                 respect_limits: bool = True,
                 # ★ 关节限位规避(2026-08-26 加)。7 自由度冗余臂只有"拉向 q_ref"一项
                 #   零空间约束时, 关节会慢慢漂到机械死点 —— 末端位姿照样解得出(ok=True),
                 #   但那个关节被 np.clip 按在限位上, 真机顶上机械硬限位后堵转、过流、失能。
                 #   实测左臂 J6 就是这样卡死的。这里用冗余自由度把接近限位的关节推回来,
                 #   走零空间投影 ⇒ 完全不影响末端位姿。
                 limit_avoid: float = 0.6,    # 规避强度(0 = 关闭)
                 limit_act: float = 0.75):    # 激活阈值: |偏离中心|/半行程 超过它才发力
    """单臂 IK。返回 (q, info)。

    **q_seed 与 q_ref 是两件事, 别混:**
      · q_seed = 迭代起点(传上一步的解 / 当前实测关节角), 保证解连续、迭代快。
      · q_ref  = 零空间锚点, 冗余那 1 个自由度被拉向它。默认取 q_seed。
    ⚠ 逐步跟踪时**必须传一个固定的 q_ref**(如本 chunk 起始的实测关节角), 不能让它
      跟着 q_seed 一起走 —— 否则拉力目标随漂移一起移动, 冗余维会自由随机游走。实测:
      末端沿 2cm 小圆走一圈回到原位姿, 锚点跟着种子走时关节漂 65~155 mrad(3.8°~8.9°),
      固定锚点则量级骤降(见 logs/ik_selftest.log 的闭环漂移两条)。

    info: dict(ok, iters, pos_err, rot_err, ns_dist, polished, at_limit)
          ok=False 时**不要**下发; at_limit 非空 = 卡关节限位, 该分支不可达。
    """
    q = np.asarray(q_seed, dtype=np.float64).reshape(7).copy()
    seed = q.copy() if q_ref is None else np.asarray(q_ref, dtype=np.float64).reshape(7)
    lim = effective_limits(side) if respect_limits else None
    p_t, R_t = T_target[:3, 3], T_target[:3, :3]
    I7 = np.eye(7)
    pos_err = rot_err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        T, J = fk_with_jacobian(q, side, tcp=tcp)
        e_pos = p_t - T[:3, 3]
        e_rot = E.mat_to_rotvec((R_t @ T[:3, :3].T)[None])[0]   # 世界系, 与动作同约定
        pos_err = float(np.linalg.norm(e_pos))
        rot_err = float(np.linalg.norm(e_rot))
        if pos_err < tol_pos and rot_err < tol_rot:
            break
        e = np.concatenate([e_pos, w_rot * e_rot])
        Jw = np.vstack([J[0:3], w_rot * J[3:6]])
        # 任务项用**阻尼**伪逆(近奇异时稳)。
        JJt = Jw @ Jw.T + (damping ** 2) * np.eye(6)
        Jp = Jw.T @ np.linalg.solve(JJt, np.eye(6))
        # 零空间项必须用**精确**投影算子 N = I − J⁺J(SVD 伪逆), 不能用 I − Jp·J:
        # 带阻尼的 Jp 不是精确伪逆 → (I − Jp·J) 不是精确投影 → 拉回种子的力会漏进末端
        # 任务, 形成 0.2~5mm 的残差地板, 迭代再多也下不去(实测见 logs/ik_selftest.log 之前
        # 那版 32/40 的失败)。用 J⁺ 后 J·(N·v) ≡ 0, 泄漏从根上消掉。
        N = I7 - np.linalg.pinv(Jw) @ Jw
        # 两项**分别**限幅: 零空间只是姿态偏好, 不能挤占任务项的步长预算。合在一起限幅时
        # (旧版)零空间拉力大的步里任务项会被压缩, 迭代走不动、还容易被推到关节限位卡死 ——
        # 实测 3 集回放 40/4668 步解不出来、末端残差最坏 22.5mm 就是这么来的。
        dq_task = Jp @ e
        nt = np.linalg.norm(dq_task)
        if nt > max_step:
            dq_task *= max_step / nt
        v_ns = ns_gain * (seed - q)
        # ── 关节限位规避 ─────────────────────────────────────────────────────
        # u = 归一化偏离中心程度, ±1 = 正好在限位上。只有 |u| > limit_act 才发力,
        # 力度按 ((|u|−act)/(1−act))² 平方增长 —— 远离限位时完全不打扰 q_ref 拉力,
        # 逼近限位时迅速压过它, 把关节推回可动范围内。
        if lim is not None and limit_avoid > 0.0:
            rng = np.maximum(lim[:, 1] - lim[:, 0], 1e-6)
            mid = 0.5 * (lim[:, 0] + lim[:, 1])
            u = 2.0 * (q - mid) / rng
            w = np.clip((np.abs(u) - limit_act) / max(1.0 - limit_act, 1e-6), 0.0, 1.0) ** 2
            v_ns = v_ns - limit_avoid * w * np.sign(u) * (0.5 * rng)
        dq_ns = N @ v_ns
        nn = np.linalg.norm(dq_ns)
        if nn > ns_max_step:
            dq_ns *= ns_max_step / nn
        q = q + dq_task + dq_ns
        if lim is not None:
            q = np.clip(q, lim[:, 0], lim[:, 1])
    ok = bool(pos_err < tol_pos and rot_err < tol_rot)
    polished = False
    if not ok and ns_gain > 0.0:
        # 收尾补救: 关掉零空间纯打末端任务, 从当前 q 接着解。零空间项虽然已无泄漏,
        # 但它仍会把解往种子拽 —— 目标离种子较远时, 这一拽可能把解推到关节限位上卡住。
        # 这一步只影响"本来就没收敛"的 case, 不改变已收敛解的构型。
        q2, inf2 = solve_arm_ik(T_target, q, side, q_ref=seed, tcp=tcp, damping=damping,
                                ns_gain=0.0, w_rot=w_rot, max_iter=max_iter,
                                tol_pos=tol_pos, tol_rot=tol_rot, max_step=max_step,
                                respect_limits=respect_limits)
        if inf2["pos_err"] < pos_err:
            q, pos_err, rot_err, ok, polished = q2, inf2["pos_err"], inf2["rot_err"], \
                inf2["ok"], True
            it += inf2["iters"]
    # ⚠ 这里原本写的是 `if respect_limits and not ok:` —— 只在**没收敛**时才检查限位。
    #   但真正伤机器的恰恰是"末端位姿解出来了(ok=True), 却有关节被 clip 按在限位上"
    #   这种情况: 调用方看到 ok=True 就照单下发, 真机顶上机械硬限位 → 堵转 → 过流 → 失能。
    #   实测左臂 J6 就是这么卡死的。现在无论收敛与否都算, 让调用方看得见。
    at_limit = []
    if respect_limits:
        lm = effective_limits(side)
        # 1e-6 太严(浮点上几乎只有正好被 clip 才算)。放宽到 0.5°: 贴着限位就该预警。
        at_limit = [i for i in range(7)
                    if min(abs(q[i] - lm[i, 0]), abs(q[i] - lm[i, 1])) < 8.7e-3]
    return q, {"ok": ok, "iters": it, "pos_err": pos_err, "rot_err": rot_err,
               "ns_dist": float(np.abs(q - seed).max()), "polished": polished,
               # 非空 = 解卡在关节限位上: 该目标从这个分支不可达, 返回 ok=False 是**正确**的,
               # 调用侧应当拒绝下发而不是硬送。
               "at_limit": at_limit}


# ------------------------------------------------------------- 静止臂死区
def arm_is_idle(rel_chunk: np.ndarray, arm: str, *,
                dp_thresh: float = 3e-3, dth_thresh_deg: float = 0.3) -> bool:
    """整段 chunk 里这条臂是否"本该不动"。

    rel_chunk: (H,14) 相对动作(EEDeltaActions 的输出)。arm: "left" / "right"。
    阈值来自数据(见 ee_pack/logs/idle_arm.log 与 arm_motion.log): 静止臂整段行程
    **恰好 0.000 m**, 干活臂单 stage 行程 0.16~0.39 m —— 1~10 mm 的阈值能干净分开,
    默认 3 mm / 0.3°。
    """
    o = 0 if arm == "left" else 7
    dp = np.linalg.norm(rel_chunk[..., o:o + 3], axis=-1).max()
    dth = np.degrees(np.linalg.norm(rel_chunk[..., o + 3:o + 6], axis=-1)).max()
    return bool(dp < dp_thresh and dth < dth_thresh_deg)


# ------------------------------------------- 20 维绝对末端位姿 → 16 维关节指令
def ee20_to_T(state20: np.ndarray, arm: str, tcp: float = TCP_FINGERTIP) -> np.ndarray:
    """(20,) 绝对末端位姿 → 该臂的 4x4(世界系)。"""
    s = np.asarray(state20, dtype=np.float64).reshape(-1)
    o = 0 if arm == "left" else 10
    T = np.eye(4)
    T[:3, 3] = s[o:o + 3]
    T[:3, :3] = E.rot6d_to_mat(s[o + 3:o + 9][None])[0]
    return T


def ee20_to_joints16(target20: np.ndarray, seed_q16: np.ndarray, *,
                     ref_q16: np.ndarray | None = None,
                     freeze_left: bool = False, freeze_right: bool = False,
                     tcp: float = TCP_FINGERTIP, **ik_kw):
    """一帧的 20 维绝对末端位姿 → 16 维关节指令(左7+左爪+右7+右爪)。

    seed_q16: 迭代起点 —— 上一步下发的关节指令(或当前实测)。
    ref_q16 : 零空间锚点 —— **本 chunk 起始的实测关节角**, 整个 chunk 里保持不变。
              不传就退化成 seed(冗余维会随机游走, 见 solve_arm_ik 的告警)。
    freeze_*: True 时该臂关节**原样沿用 seed**、不调 IK(静止臂死区, 见 arm_is_idle)。
    夹爪开度一律直接搬运, 不进 IK。
    返回 (q16, info) —— info["left"/"right"] 是各臂 IK info(冻结时为 None)。
    """
    t20 = np.asarray(target20, dtype=np.float64).reshape(-1)
    q16 = np.asarray(seed_q16, dtype=np.float64).reshape(16).copy()
    r16 = None if ref_q16 is None else np.asarray(ref_q16, dtype=np.float64).reshape(16)
    info = {}
    for arm, jsl, gi, gt, frz in (("left", E.J_LEFT, E.G_LEFT, 9, freeze_left),
                                  ("right", E.J_RIGHT, E.G_RIGHT, 19, freeze_right)):
        if frz:
            info[arm] = None
        else:
            q, inf = solve_arm_ik(ee20_to_T(t20, arm, tcp), q16[jsl], arm,
                                  q_ref=None if r16 is None else r16[jsl],
                                  tcp=tcp, **ik_kw)
            q16[jsl] = q
            info[arm] = inf
        q16[gi] = t20[gt]        # 夹爪开度直接搬运, 不进 IK
    return q16, info


# ------------------------------------------------------------------ selftest
def selftest(n: int = 40, seed: int = 0) -> bool:
    import time
    rng = np.random.default_rng(seed)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(("✓" if cond else "✗"), name, " ", detail, flush=True)
        ok = ok and bool(cond)

    for side in ("left", "right"):
        lim = effective_limits(side)
        # ① FK 必须与 ee_repr / fk_nero 完全一致(同一份链, 不能有第二套约定)
        q = rng.uniform(lim[:, 0], lim[:, 1], size=(n, 7))
        Tj = np.stack([fk_with_jacobian(q[i], side)[0] for i in range(n)])
        Tb = E.fk_batch(q, side)
        chk(f"fk_with_jacobian({side}) ≡ ee_repr.fk_batch",
            float(np.abs(Tj - Tb).max()) < 1e-12, f"max|Δ|={np.abs(Tj - Tb).max():.2e}")

        # ② 解析 Jacobian 必须等于有限差分
        worst = 0.0
        for i in range(8):
            q0 = q[i]
            _, J = fk_with_jacobian(q0, side)
            Jn = np.zeros((6, 7))
            h = 1e-7
            T0 = fk_with_jacobian(q0, side)[0]
            for k in range(7):
                qp = q0.copy(); qp[k] += h
                Tp = fk_with_jacobian(qp, side)[0]
                Jn[0:3, k] = (Tp[:3, 3] - T0[:3, 3]) / h
                Jn[3:6, k] = E.mat_to_rotvec((Tp[:3, :3] @ T0[:3, :3].T)[None])[0] / h
            worst = max(worst, float(np.abs(J - Jn).max()))
        chk(f"解析 Jacobian({side}) ≡ 有限差分", worst < 1e-4, f"max|Δ|={worst:.2e}")

        # ③ 硬 case: 全关节域随机目标 + 种子偏 0.15rad(**不是**部署场景, 见 ③b)。
        #    这里允许少量失败, 但失败必须是"卡关节限位"(该分支不可达, 拒解才对), 不能是
        #    收敛到一半的假失败。
        conv = 0
        perr = rerr = 0.0
        bad_no_limit = []
        t0 = time.perf_counter()
        for i in range(n):
            T_t = E.fk_batch(q[i][None], side)[0]
            qs = np.clip(q[i] + rng.normal(0, 0.15, 7), lim[:, 0], lim[:, 1])
            qi, inf = solve_arm_ik(T_t, qs, side)
            conv += inf["ok"]
            if inf["ok"]:
                perr = max(perr, inf["pos_err"]); rerr = max(rerr, inf["rot_err"])
            elif not inf["at_limit"]:
                bad_no_limit.append((i, inf))
        dt = (time.perf_counter() - t0) / n * 1000
        chk(f"硬case 收敛率({side}) ≥90%", conv >= 0.9 * n,
            f"{conv}/{n}  收敛者最差 pos={perr*1000:.4f}mm rot={np.degrees(rerr):.5f}°"
            f"  单次 {dt:.2f}ms")
        chk(f"硬case 的失败都是卡限位(拒解正确)({side})", not bad_no_limit,
            f"非限位失败 {len(bad_no_limit)} 例" if bad_no_limit else "")

        # ③b 部署场景: 沿平滑轨迹跟踪, 种子 = 上一步的解(相邻帧只差几 mm)。这才是真机上
        #     IK 的实际用法, 必须 100% 收敛。轨迹**整条都在限位内**(不 clip): 用 clip 会让
        #     直线贴着关节限位走, 那是合成产物, 示教数据不会这样(§真实数据由 replay 验)。
        mid, span = lim.mean(1), lim[:, 1] - lim[:, 0]
        conv2 = ntraj = 0
        perr2 = rerr2 = 0.0
        t0 = time.perf_counter()
        for i in range(6):
            for _ in range(20):                      # 重采样直到整条轨迹不越限
                q0 = mid + (rng.random(7) - 0.5) * 0.3 * span
                dirn = rng.normal(size=7)
                dirn /= np.linalg.norm(dirn)
                traj = q0[None] + np.linspace(0, 0.25, 26)[:, None] * dirn[None]
                if (traj >= lim[:, 0]).all() and (traj <= lim[:, 1]).all():
                    break
            else:
                continue
            sd = traj[0].copy()
            for k in range(1, len(traj)):
                T_t = E.fk_batch(traj[k][None], side)[0]
                qi, inf = solve_arm_ik(T_t, sd, side)
                conv2 += inf["ok"]; ntraj += 1
                perr2 = max(perr2, inf["pos_err"]); rerr2 = max(rerr2, inf["rot_err"])
                sd = qi
        dt2 = (time.perf_counter() - t0) / max(ntraj, 1) * 1000
        chk(f"部署场景(轨迹跟踪, 种子=上一步解)收敛率({side})", conv2 == ntraj and ntraj > 0,
            f"{conv2}/{ntraj}  最差 pos={perr2*1000:.4f}mm rot={np.degrees(rerr2):.5f}°"
            f"  单次 {dt2:.2f}ms")

        # ④ 零空间: 目标 = 种子自身的位姿 时, 解必须**原地不动**(这才治得住甩肘)
        drift = 0.0
        for i in range(n):
            T_t = E.fk_batch(q[i][None], side)[0]
            qi, _ = solve_arm_ik(T_t, q[i], side)          # 种子就是真解
            drift = max(drift, float(np.abs(qi - q[i]).max()))
        chk(f"零空间不甩肘({side}): 目标≡种子位姿 → 关节不动",
            drift < 1e-6, f"最大关节漂移={drift:.2e} rad")

        # ⑤ 零空间投影必须**精确**: J·(N·v) ≡ 0, 否则拉回种子的力会漏成残差地板
        leak_exact = leak_damped = 0.0
        for i in range(12):
            _, J = fk_with_jacobian(q[i], side)
            v = rng.normal(size=7)
            N = np.eye(7) - np.linalg.pinv(J) @ J
            leak_exact = max(leak_exact, float(np.linalg.norm(J @ (N @ v))))
            Jp = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.eye(6))
            Nd = np.eye(7) - Jp @ J
            leak_damped = max(leak_damped, float(np.linalg.norm(J @ (Nd @ v))))
        chk(f"零空间投影无泄漏({side}): ‖J·N·v‖≈0", leak_exact < 1e-9,
            f"精确投影 {leak_exact:.2e}  (带阻伪逆那版是 {leak_damped:.2e}, 差 "
            f"{leak_damped/max(leak_exact,1e-18):.0e} 倍)")

        # ⑥ 闭环漂移(直击部署真实失效模式): 让末端沿一个小圆走一圈**回到出发位姿**,
        #    每步用上一步的解做种子。末端回到原点了, 关节**也应该**回到原处;冗余维如果
        #    每步偷偷漂一点, 一圈下来就是可见的甩肘。这条比"随机种子比距离"稳定得多。
        drift_ns = drift_no = 0.0
        for i in range(4):
            for _ in range(20):
                q0 = mid + (rng.random(7) - 0.5) * 0.2 * span
                T0 = E.fk_batch(q0[None], side)[0]
                th = np.linspace(0, 2 * np.pi, 33)
                Ts = np.repeat(T0[None], len(th), 0).copy()
                Ts[:, 0, 3] += 0.02 * (np.cos(th) - 1.0)   # 世界系 XY 平面上 2cm 小圆
                Ts[:, 1, 3] += 0.02 * np.sin(th)
                trial = [solve_arm_ik(Ts[k], q0, side)[1]["ok"] for k in (8, 16, 24)]
                if all(trial):
                    break
            else:
                continue
            for mode in ("fixed_ref", "moving_ref"):
                sd = q0.copy()
                bad = False
                for k in range(len(Ts)):
                    # fixed_ref: 锚点固定在起始构型(部署应当这样传)
                    # moving_ref: 锚点跟着种子走(= 不传 q_ref, 反面教材)
                    sd, inf = solve_arm_ik(Ts[k], sd, side,
                                           q_ref=q0 if mode == "fixed_ref" else None)
                    bad = bad or not inf["ok"]
                if bad:
                    continue
                d = float(np.abs(sd - q0).max())      # Ts[-1] ≡ T0, 关节该回到 q0
                if mode == "fixed_ref":
                    drift_ns = max(drift_ns, d)
                else:
                    drift_no = max(drift_no, d)
        chk(f"闭环漂移: 固定锚点 << 锚点跟着种子走({side})", drift_ns < 0.5 * drift_no,
            f"固定锚点 {drift_ns*1000:.3f} mrad  vs  跟着种子 {drift_no*1000:.3f} mrad")
        chk(f"闭环漂移绝对量可接受({side}) <5 mrad", drift_ns < 5e-3,
            f"固定锚点 {drift_ns*1000:.3f} mrad")

    # ⑥ 死区判定: 造一个"左臂全零 / 右臂有动作"的 chunk
    rel = np.zeros((50, 14))
    rel[:, 7:10] = np.linspace(0, 0.25, 50)[:, None]      # 右臂平移 25cm
    rel[:, 10:13] = np.linspace(0, 0.4, 50)[:, None]      # 右臂转 ~0.4rad
    chk("死区: 左臂判为静止", arm_is_idle(rel, "left"))
    chk("死区: 右臂判为活动", not arm_is_idle(rel, "right"))

    # ⑦ 端到端: 20 维目标 + 种子 → 16 维关节, 且冻结开关生效
    lim = effective_limits("right")
    q16 = np.concatenate([rng.uniform(lim[:, 0], lim[:, 1], 7), [0.001],
                          rng.uniform(lim[:, 0], lim[:, 1], 7), [0.002]])
    t20 = E.joints16_to_ee20(q16[None])[0]
    out, inf = ee20_to_joints16(t20, q16)
    chk("端到端 ee20→关节16(目标≡种子位姿)", float(np.abs(out - q16).max()) < 1e-6,
        f"max|Δ|={np.abs(out - q16).max():.2e}")
    out2, inf2 = ee20_to_joints16(t20, q16, freeze_left=True)
    chk("freeze_left 生效(不调 IK, 关节沿用种子)",
        inf2["left"] is None and np.array_equal(out2[E.J_LEFT], q16[E.J_LEFT]))
    print("\nIK SELFTEST PASS" if ok else "\nIK SELFTEST FAILED", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
