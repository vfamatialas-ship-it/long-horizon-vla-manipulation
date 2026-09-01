#!/usr/bin/env python3
"""检查 nero 双臂 LeRobot 数据集完整性 —— 重点: 关节是否在变化(防"冻结废数据")。

用法:
  python3 check_dataset_motion.py                     # 自动挑 data_depth 里【最新】采集目录
  python3 check_dataset_motion.py <数据集目录>          # 指定目录
  python3 check_dataset_motion.py --glob 'xxx_*'       # 换匹配再挑最新
输出: 逐维 range/std、逐 episode 左右臂关节+夹爪变化、冻结统计、总结论。
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DEPTH = Path("<PROJECT_ROOT>/lerobot_data/lerobot_hezi/data_depth")


def newest_dataset(pattern: str):
    cands = [Path(p) for p in glob.glob(str(DATA_DEPTH / pattern)) if "ee_pose" not in p]
    cands = [c for c in cands if (c / "meta" / "info.json").exists()]
    return max(cands, key=lambda c: c.stat().st_mtime) if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None, help="数据集目录; 不给=自动挑最新")
    ap.add_argument("--glob", default="nero_box_packing_closing_refined_v1_*", help="自动挑最新时的匹配")
    ap.add_argument("--frozen-rad", type=float, default=1e-3, help="关节 range 低于此=判冻结(rad)")
    args = ap.parse_args()

    root = Path(args.dataset) if args.dataset else newest_dataset(args.glob)
    if root is None or not (root / "meta" / "info.json").exists():
        print(f"❌ 找不到数据集(dir={root})"); sys.exit(1)

    info = json.loads((root / "meta" / "info.json").read_text())
    print(f"数据集: {root}")
    print(f"episodes={info.get('total_episodes')} frames={info.get('total_frames')} "
          f"fps={info.get('fps')} robot={info.get('robot_type')}")
    feats = info["features"]
    print("图像/深度特征:", [k for k in feats if "image" in k or "depth" in k])
    print(f"深度: {'有' if any('depth' in k for k in feats) else '无'}")

    KEY = "observation.state"
    if KEY not in feats:
        print(f"❌ 没有 {KEY} 特征"); sys.exit(1)
    names = feats[KEY].get("names") or [f"d{i}" for i in range(feats[KEY]["shape"][0])]
    ndim = len(names)
    grip_idx = [i for i, n in enumerate(names) if "gripper" in n.lower()]
    left_j = [i for i in range(0, ndim // 2) if i not in grip_idx]
    right_j = [i for i in range(ndim // 2, ndim) if i not in grip_idx]

    pqs = sorted(glob.glob(str(root / "data" / "chunk-*" / "episode_*.parquet")))
    inc = len(glob.glob(str(root / "incomplete" / "**" / "*.parquet"), recursive=True))
    print(f"\ndata/ 完成={len(pqs)}   incomplete/ 未完成={inc}")
    if not pqs:
        print("❌ 没有已完成的 episode(还没按 s/f 保存, 或都在 incomplete/)"); sys.exit(1)

    allst, per = [], []
    for pq in pqs:
        df = pd.read_parquet(pq)
        st = np.stack(df[KEY].to_numpy()).astype(float)
        allst.append(st); per.append((Path(pq).name, len(df), st.max(0) - st.min(0)))
    allst = np.concatenate(allst, 0)
    rng, std = allst.max(0) - allst.min(0), allst.std(0)

    print(f"\n===== 全体 {allst.shape[0]} 帧 逐维 range/std =====")
    for i, n in enumerate(names):
        tag = "爪" if i in grip_idx else ("左" if i < ndim // 2 else "右")
        if i not in grip_idx and rng[i] < args.frozen_rad:
            flag = " ❄️冻结"
        elif i in grip_idx and float(rng[i]) == 0.0:
            flag = " ⚠️精确0(疑似反馈冻结)"
        else:
            flag = ""
        print(f"  [{i:2d}][{tag}] {n:20s} range={rng[i]:8.4f} std={std[i]:8.5f}{flag}")

    print(f"\n===== 逐 episode =====")
    fL = fR = 0
    for name, nfr, r in per:
        lm, rm = r[left_j].max(), r[right_j].max()
        lf = " ❄️左冻" if lm < args.frozen_rad else ""
        rf = " ❄️右冻" if rm < args.frozen_rad else ""
        fL += lm < args.frozen_rad; fR += rm < args.frozen_rad
        grip = "  ".join(f"{names[g].split('_')[0]}爪Δ={r[g]:.4f}" for g in grip_idx)
        print(f"  {name} 帧={nfr:4d}  左臂Δ={lm:6.3f}{lf}  右臂Δ={rm:6.3f}{rf}  {grip}")

    print(f"\n===== 结论 =====")
    print(f"  左臂冻结 {fL}/{len(per)}   右臂冻结 {fR}/{len(per)}")
    for g in grip_idx:
        note = "  ⚠️精确 0.0000(反馈冻结特征; 若本该开合=没录上)" if float(rng[g]) == 0.0 else ""
        print(f"  {names[g]}: 全程 range={rng[g]:.4f}{note}")
    ok = fL == 0 and fR == 0
    print("\n  " + ("✅ 两臂关节都在动、无冻结集 —— 关节信息完整可用。" if ok
                    else "❌ 有臂冻结集 —— 同上次废数据, 别用, 先修反馈再重采。"))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
