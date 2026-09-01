#!/usr/bin/env python3
"""末端位姿表示法选型实测:在真数据上量各候选「相对动作」表示的归一化健康度。

为什么要量而不是拍脑袋:π0.5 训练前会对 state/actions 做 z-归一化(除以逐维 std)。
某一维如果近常数(std≈0), 归一化后幅值会炸到 1e5~1e7, loss 一起飞 —— 这就是 hezi
封盖数据踩过的「夹爪常数维」坑#2。相对旋转用 rot6D 时, 近单位旋转的 6 个分量恒等于
[1,0,0,0,1,0], 天然是常数维高危; 用轴角(rotvec)则天然零中心。到底炸不炸, 只有量。

用法(CPU 即可, 不需要 GPU / norm stats):
  python probe_ee_repr.py [--stride 5] [--horizon 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ee_repr as E  # noqa: E402

SRCS = [
    Path("<DATA_ROOT>/local/nero_hezi_closing_refined_merged_v2"),
    Path("<DATA_ROOT>/local/nero_hezi_closing_refined_supplement_v1"),
]


def load_episodes(src: Path):
    info = json.loads((src / "meta/info.json").read_text())
    for e in range(info["total_episodes"]):
        t = pq.read_table(src / f"data/chunk-000/episode_{e:06d}.parquet",
                          columns=["observation.state", "action"])
        d = t.to_pydict()
        s = np.asarray([np.asarray(x) for x in d["observation.state"]], dtype=np.float64)
        a = np.asarray([np.asarray(x) for x in d["action"]], dtype=np.float64)
        yield src.name, e, s, a


def stat_block(name: str, x: np.ndarray, names: list[str]) -> dict:
    """逐维 mean/std/min/max, 并算 z-归一化后的最大幅值(openpi 的归一化口径)。"""
    mu, sd = x.mean(0), x.std(0)
    z = np.abs((x - mu) / np.maximum(sd, 1e-9)).max(0)   # openpi 用 std, eps 极小
    print(f"\n—— {name}  (N={len(x)}, dim={x.shape[1]})")
    print(f"{'dim':>22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'max|z|':>9}")
    for i, nm in enumerate(names):
        flag = "  ⚠std" if sd[i] < 1e-3 else ""
        print(f"{nm:>22} {mu[i]:>10.4f} {sd[i]:>10.5f} {x[:,i].min():>10.4f} "
              f"{x[:,i].max():>10.4f} {z[i]:>9.2f}{flag}")
    print(f"  → 最小 std = {sd.min():.3e} (维 {names[int(sd.argmin())]}) | "
          f"归一化后 max|z| = {z.max():.2f} (维 {names[int(z.argmax())]})")
    return {"dim": int(x.shape[1]), "min_std": float(sd.min()), "max_z": float(z.max()),
            "min_std_dim": names[int(sd.argmin())], "max_z_dim": names[int(z.argmax())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="chunk 起点采样步长")
    ap.add_argument("--horizon", type=int, default=50, help="pi0.5 action_horizon")
    args = ap.parse_args()

    if not E.selftest():
        sys.exit("✘ ee_repr selftest 未通过, 停止")

    H = args.horizon
    ee_all, rel_w_rotvec, rel_w_rot6d, rel_tool, joint_delta = [], [], [], [], []
    per_step_dp, per_step_dth, chunk_dp, chunk_dth = [], [], [], []
    n_eps = n_frames = 0

    for src in SRCS:
        for srcname, e, s16, a16 in load_episodes(src):
            n_eps += 1
            n_frames += len(s16)
            if not np.allclose(s16, a16, atol=1e-6):
                d = np.abs(s16 - a16).max()
                print(f"  · 注意 {srcname}/ep{e}: action≠state, max|Δ|={d:.2e}")
            ee = E.joints16_to_ee20(s16)          # (n,20) 绝对末端
            ee_all.append(ee)
            n = len(ee)
            pl, Rl, _, pr, Rr, _ = E.ee20_split(ee)

            # 逐步运动量(相邻帧)
            per_step_dp.append(np.linalg.norm(np.diff(pl, axis=0), axis=-1))
            per_step_dp.append(np.linalg.norm(np.diff(pr, axis=0), axis=-1))
            per_step_dth.append(np.linalg.norm(
                E.mat_to_rotvec(Rl[1:] @ np.swapaxes(Rl[:-1], -1, -2)), axis=-1))
            per_step_dth.append(np.linalg.norm(
                E.mat_to_rotvec(Rr[1:] @ np.swapaxes(Rr[:-1], -1, -2)), axis=-1))

            # chunk 锚定(和 openpi DeltaActions 同口径: 全 H 步都相对首帧 state)
            starts = np.arange(0, n, args.stride)
            idx = np.minimum(starts[:, None] + np.arange(H)[None, :], n - 1)  # 末尾重复补齐
            a20 = ee[idx]                                    # (M,H,20)
            s20 = ee[starts]                                 # (M,20)
            rel_w_rotvec.append(E.ee20_to_rel14(s20, a20).reshape(-1, 14))

            p0l, R0l, _, p0r, R0r, _ = E.ee20_split(s20)
            pil, Ril, gl, pir, Rir, gr = E.ee20_split(a20)
            R0lb, R0rb = R0l[:, None], R0r[:, None]
            Rrel_l = Ril @ np.swapaxes(R0lb, -1, -2)
            Rrel_r = Rir @ np.swapaxes(R0rb, -1, -2)
            chunk_dp.append(np.linalg.norm(pil - p0l[:, None], axis=-1).ravel())
            chunk_dp.append(np.linalg.norm(pir - p0r[:, None], axis=-1).ravel())
            chunk_dth.append(np.linalg.norm(E.mat_to_rotvec(Rrel_l), axis=-1).ravel())
            chunk_dth.append(np.linalg.norm(E.mat_to_rotvec(Rrel_r), axis=-1).ravel())

            # 候选 B: 世界系相对旋转用 rot6D (20 维)
            rel_w_rot6d.append(np.concatenate([
                pil - p0l[:, None], E.mat_to_rot6d(Rrel_l), gl[..., None],
                pir - p0r[:, None], E.mat_to_rot6d(Rrel_r), gr[..., None],
            ], -1).reshape(-1, 20))

            # 候选 C: 工具系(末端自身坐标系)相对位姿, 轴角 (14 维)
            rel_tool.append(np.concatenate([
                np.einsum("mij,mhj->mhi", np.swapaxes(R0l, -1, -2), pil - p0l[:, None]),
                E.mat_to_rotvec(np.swapaxes(R0lb, -1, -2) @ Ril), gl[..., None],
                np.einsum("mij,mhj->mhi", np.swapaxes(R0r, -1, -2), pir - p0r[:, None]),
                E.mat_to_rotvec(np.swapaxes(R0rb, -1, -2) @ Rir), gr[..., None],
            ], -1).reshape(-1, 14))

            # 对照: 现役 16 维关节 delta(爪绝对)
            jd = a16[idx].copy()
            jd[..., E.J_LEFT] -= s16[starts][:, None, E.J_LEFT]
            jd[..., E.J_RIGHT] -= s16[starts][:, None, E.J_RIGHT]
            joint_delta.append(jd.reshape(-1, 16))
            print(f"  ep{n_eps-1:03d} {srcname[:34]:34s} {n:5d}帧", flush=True)

    print(f"\n=== 共 {n_eps} 集 / {n_frames} 帧 ===")
    ee_all = np.concatenate(ee_all)
    print("\n### 1) FK 合理性(绝对末端, 世界系, TCP=指尖 0.138)")
    for arm, sl in (("左臂", slice(0, 3)), ("右臂", slice(10, 13))):
        p = ee_all[:, sl]
        print(f"  {arm} 位置 x[{p[:,0].min():.3f},{p[:,0].max():.3f}] "
              f"y[{p[:,1].min():.3f},{p[:,1].max():.3f}] z[{p[:,2].min():.3f},{p[:,2].max():.3f}] "
              f"| 到底座距离 [{np.linalg.norm(p,axis=1).min():.3f},{np.linalg.norm(p,axis=1).max():.3f}]")
    for arm, sl in (("左臂", slice(3, 9)), ("右臂", slice(13, 19))):
        R = E.rot6d_to_mat(ee_all[:, sl])
        det = np.linalg.det(R)
        print(f"  {arm} rot6d→R 正交性 det∈[{det.min():.6f},{det.max():.6f}] "
              f"(应≡1) | RᵀR-I 最大偏差 {np.abs(R.transpose(0,2,1)@R - np.eye(3)).max():.2e}")

    psp = np.concatenate(per_step_dp)
    pst = np.concatenate(per_step_dth)
    cdp = np.concatenate(chunk_dp)
    cdt = np.concatenate(chunk_dth)
    print("\n### 2) 运动幅度(决定相对量的量纲)")
    q = [50, 90, 99, 100]
    print(f"  相邻帧(1/15s)  |Δp| 分位 {np.percentile(psp,q).round(5)} m")
    print(f"  相邻帧(1/15s)  |Δθ| 分位 {np.degrees(np.percentile(pst,q)).round(3)} °")
    print(f"  整 chunk({H}步={H/15:.2f}s) |Δp| 分位 {np.percentile(cdp,q).round(4)} m")
    print(f"  整 chunk({H}步={H/15:.2f}s) |Δθ| 分位 {np.degrees(np.percentile(cdt,q)).round(2)} °")

    print("\n### 3) 归一化健康度对照(max|z| 是「归一化后幅值」, 铁律要 <20)")
    res = {}
    res["state20_abs_ee"] = stat_block("输入 state:20 维绝对末端(pos+rot6d+爪)",
                                      ee_all, E.STATE_NAMES)
    res["A_world_rotvec14"] = stat_block(
        "候选A 动作:14 维 世界系 Δpos+rotvec+爪绝对",
        np.concatenate(rel_w_rotvec), E.REL_ACTION_NAMES)
    r6names = ["left_dx", "left_dy", "left_dz"] + [f"left_drot6d_{i}" for i in range(6)] + \
              ["left_gripper_width", "right_dx", "right_dy", "right_dz"] + \
              [f"right_drot6d_{i}" for i in range(6)] + ["right_gripper_width"]
    res["B_world_rot6d20"] = stat_block(
        "候选B 动作:20 维 世界系 Δpos+rot6d(相对旋转)+爪绝对",
        np.concatenate(rel_w_rot6d), r6names)
    res["C_tool_rotvec14"] = stat_block(
        "候选C 动作:14 维 工具系 Δpos+rotvec+爪绝对",
        np.concatenate(rel_tool), E.REL_ACTION_NAMES)
    jn = [f"left_j{i+1}" for i in range(7)] + ["left_gripper_width"] + \
         [f"right_j{i+1}" for i in range(7)] + ["right_gripper_width"]
    res["ref_joint_delta16"] = stat_block(
        "对照 现役:16 维关节 delta(爪绝对)", np.concatenate(joint_delta), jn)

    print("\n### 4) 结论表")
    print(f"{'方案':>34} {'维度':>5} {'最小std':>11} {'max|z|':>9}  最危维")
    for k, v in res.items():
        print(f"{k:>34} {v['dim']:>5} {v['min_std']:>11.3e} {v['max_z']:>9.2f}  {v['min_std_dim']}")


if __name__ == "__main__":
    main()
