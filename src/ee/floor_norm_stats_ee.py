#!/usr/bin/env python3
"""对末端位姿版 norm_stats.json 的近常数维做 std 兜底, 防 z-score (x-mean)/(std+1e-6) 爆炸。

本任务(封盖/合盖)两个夹爪整程几乎不动 —— 实测 100 集里左爪全程 range 只有 0.57mm、
右爪 p99 4.6mm, std≈1e-4。不兜底时归一化幅值贴着「max|z|<20」铁律的红线(实测 19.9),
换一批数据立刻炸。末端位姿/旋转各维 std 都 ≥0.019, 不会被兜底波及。

  state   20 维: [左 pos3, 左 rot6d, 左爪宽, 右 pos3, 右 rot6d, 右爪宽]
  actions 14 维: [左 Δpos3, 左 rotvec3, 左爪宽, 右 Δpos3, 右 rotvec3, 右爪宽]

幂等: 首次运行会把原始存成 .prefloor, 之后一律以 .prefloor 为基准重算, 不会二次抬高。
用法: floor_norm_stats_ee.py <norm_stats.json> [THRESH=0.01] [FLOOR=0.05]
"""
import json
import pathlib
import sys

import numpy as np

NAMES = {
    20: ["Lx", "Ly", "Lz", "Lr0", "Lr1", "Lr2", "Lr3", "Lr4", "Lr5", "Lgrip",
         "Rx", "Ry", "Rz", "Rr0", "Rr1", "Rr2", "Rr3", "Rr4", "Rr5", "Rgrip"],
    14: ["Ldx", "Ldy", "Ldz", "Lwx", "Lwy", "Lwz", "Lgrip",
         "Rdx", "Rdy", "Rdz", "Rwx", "Rwy", "Rwz", "Rgrip"],
}

p = pathlib.Path(sys.argv[1])
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
FLOOR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

bak = p.with_name(p.name + ".prefloor")
src_text = bak.read_text() if bak.exists() else p.read_text()
if not bak.exists():
    bak.write_text(src_text)
    print("已备份原始 →", bak)

d = json.loads(src_text)
ns = d["norm_stats"]
np.set_printoptions(precision=5, suppress=True, linewidth=220)
for key in ("state", "actions"):
    std = np.array(ns[key]["std"], dtype=np.float64)
    mean = np.array(ns[key]["mean"], dtype=np.float64)
    mn = np.array(ns[key]["min"], dtype=np.float64) if "min" in ns[key] else None
    mx = np.array(ns[key]["max"], dtype=np.float64) if "max" in ns[key] else None
    names = NAMES.get(len(std), [f"d{i}" for i in range(len(std))])
    before = std.copy()
    floored = std < THRESH
    std[floored] = FLOOR
    ns[key]["std"] = std.tolist()
    print(f"\n[{key}] {len(std)} 维")
    print(f"  std 前: {before}")
    print(f"  std 后: {std}")
    print(f"  兜底维(std<{THRESH} → {FLOOR}): "
          f"{[f'{names[i]}({before[i]:.2g})' for i in range(len(std)) if floored[i]]}")
    if mn is not None and mx is not None:
        z_before = np.maximum(np.abs(mn - mean), np.abs(mx - mean)) / (before + 1e-6)
        z_after = np.maximum(np.abs(mn - mean), np.abs(mx - mean)) / (std + 1e-6)
        print(f"  归一化后 max|z|: 兜底前 {z_before.max():.2f}(维 {names[int(z_before.argmax())]})"
              f"  →  兜底后 {z_after.max():.2f}(维 {names[int(z_after.argmax())]})")
        if z_after.max() >= 20.0:
            sys.exit(f"✘ [{key}] 兜底后 max|z|={z_after.max():.2f} 仍 ≥20, 别训, 先查数据")

p.write_text(json.dumps(d, indent=2))
print("\nwritten:", p)
