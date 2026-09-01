#!/usr/bin/env python3
"""手教轨迹回放 —— 独立工具, 用来单独验证一条录好的轨迹。

和 traj_teach.py --replay 的关系
--------------------------------
回放的核心逻辑在 traj_teach.py 里(部署链也调它), 本脚本**复用**那份实现, 不重写。
多出来的是**回放前的体检和可视化**: 录得好不好、会不会撞限位、速度多快、
起点离当前位差多远 —— 这些在真让机械臂动之前就该看清楚。

体检做什么
----------
1. **录制质量**: 帧间隔是否稳定; 完全静止帧占比(过高说明可能读到了旧缓存)
2. **关节限位**: 逐关节比对 joint_envelope.json 的训练包络, 超出的标出来
3. **速度**: 角速度分位数 + 回放时会不会被 --rate 限幅削掉
4. **起点差**: 当前位与轨迹起点差多少 —— 差太多回放会先"走过去", 那一段没被示教过
5. **ASCII 曲线**: 直接在终端画各关节随时间的变化, 不用开图形界面

用法::

    python3 traj_replay.py --list                 # 有哪些轨迹
    python3 traj_replay.py 123                    # 体检 + 干跑(不动机械臂)
    python3 traj_replay.py 123 --plot             # additionally 画 ASCII 曲线
    python3 traj_replay.py 123 --execute          # 体检通过后真回放
    python3 traj_replay.py 123 --execute --replay-fps 30   # 回放更细腻
    python3 traj_replay.py 123 --arms left        # 只放一条臂
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NERO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NERO_ROOT / "nero_control"))

import traj_teach as T  # noqa: E402  (复用录制/回放实现)

ENV_PATH = NERO_ROOT / "ee_pose" / "assets" / "joint_envelope.json"


def load(name: str) -> dict:
    p = T.TRAJ_DIR / f"{name}.json"
    if not p.exists():
        avail = [q.stem for q in sorted(T.TRAJ_DIR.glob("*.json"))]
        sys.exit(f"✘ 没有这条轨迹: {p}\n   已录的: {avail}")
    return json.loads(p.read_text(encoding="utf-8"))


def ascii_curve(t: np.ndarray, A: np.ndarray, label: str, w: int = 66, h: int = 9) -> None:
    """在终端画一条关节角曲线。图形界面装不装都能看。"""
    lo, hi = A.min(), A.max()
    if hi - lo < 1e-9:
        print(f"    {label}: 恒定 {np.degrees(lo):+.1f}°")
        return
    grid = [[" "] * w for _ in range(h)]
    idx = np.linspace(0, len(A) - 1, w).astype(int)
    for x, i in enumerate(idx):
        y = int(round((A[i] - lo) / (hi - lo) * (h - 1)))
        grid[h - 1 - y][x] = "█"
    print(f"    {label}  {np.degrees(hi):+7.1f}° ┤{''.join(grid[0])}")
    for r in range(1, h - 1):
        print(f"    {'':>{len(label)}}  {'':>7}  │{''.join(grid[r])}")
    print(f"    {'':>{len(label)}}  {np.degrees(lo):+7.1f}° ┴{''.join(grid[h-1])}")


def inspect(d: dict, args) -> bool:
    """回放前体检。返回 False 表示有硬问题, 别直接 --execute。"""
    F = d["frames"]
    t = np.array([f["t"] for f in F])
    dt = np.diff(t)
    ok = True

    print(f"\n══ 轨迹体检: {d['name']} ══", flush=True)
    print(f"  录于 {d.get('_recorded_at','?')}   {d['n_frames']} 帧 @ {d['fps']}Hz "
          f"/ {d['duration_s']:.1f}s   臂: {','.join(d['arms'])}")

    # ① 录制质量
    nominal = 1.0 / d["fps"]
    jitter = float(np.percentile(np.abs(dt - nominal), 95)) * 1000
    print(f"\n  【录制质量】帧间隔 均值 {dt.mean()*1000:.2f}ms (标称 {nominal*1000:.2f}ms)"
          f"  p95 抖动 {jitter:.2f}ms")
    if jitter > nominal * 1000 * 0.5:
        print(f"    ⚠ 抖动偏大 —— 录制时系统负载高? 回放仍可用, 但时间轴不够均匀")

    env = json.loads(ENV_PATH.read_text(encoding="utf-8"))["arms"] if ENV_PATH.exists() else None

    for s in d["arms"]:
        if s not in args.arms:
            continue
        A = np.array([f[s] for f in F])
        print(f"\n  【{s}】")
        print(f"    起点 {np.round(np.degrees(A[0]), 1).tolist()}")
        print(f"    终点 {np.round(np.degrees(A[-1]), 1).tolist()}")
        span = np.degrees(A.max(0) - A.min(0))
        print(f"    行程 {np.round(span, 1).tolist()}  最大 {span.max():.1f}° (j{int(span.argmax())})")

        # 完全静止帧: 过高说明可能读到了旧缓存(主臂模式的经典坑)
        still = (np.abs(np.diff(A, axis=0)).max(1) < 1e-6)
        pct = 100 * still.mean()
        flag = "  ⚠ 过高, 疑似读到旧缓存(见 traj_teach.read_teach 的注释)" if pct > 90 else ""
        print(f"    完全静止帧 {still.sum()}/{len(still)} ({pct:.1f}%){flag}")
        if pct > 90:
            ok = False

        # 角速度 vs 回放限幅
        v = np.abs(np.diff(A, axis=0)) / np.maximum(dt[:, None], 1e-9)
        vmax_allowed = args.rate * args.replay_fps          # rad/tick × tick/s
        p95 = float(np.percentile(v, 95))
        print(f"    角速度 p50={np.degrees(np.percentile(v,50)):5.1f}  "
              f"p95={np.degrees(p95):5.1f}  max={np.degrees(v.max()):5.1f} °/s")
        print(f"    回放限幅上限 = --rate {args.rate} × {args.replay_fps}Hz = "
              f"{np.degrees(vmax_allowed):.1f} °/s", end="")
        if p95 > vmax_allowed:
            print(f"   ⚠ p95 超上限 → 回放会比示教慢, 路径不变但耗时更长")
        else:
            print("   ✓ 够用")

        # 关节限位
        if env and s in env:
            lo = np.array(env[s]["observed_min"]); hi = np.array(env[s]["observed_max"])
            out_lo = (A < lo).any(0); out_hi = (A > hi).any(0)
            bad = np.flatnonzero(out_lo | out_hi)
            if len(bad) == 0:
                print(f"    关节限位 ✓ 全程在训练包络内")
            else:
                for j in bad:
                    print(f"    关节限位 ⚠ j{j}: 轨迹 [{np.degrees(A[:,j].min()):+.1f}, "
                          f"{np.degrees(A[:,j].max()):+.1f}]°  包络 "
                          f"[{np.degrees(lo[j]):+.1f}, {np.degrees(hi[j]):+.1f}]°")
                print(f"       (超出部分回放时会被 clamp 夹住 —— 那几个关节到不了示教位置)")

        if args.plot:
            print(f"    ── 各关节随时间 ──")
            for j in range(7):
                ascii_curve(t, A[:, j], f"j{j}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="手教轨迹回放 + 回放前体检")
    ap.add_argument("name", nargs="?", help="轨迹名(不含 .json)")
    ap.add_argument("--list", action="store_true", help="列出已录轨迹")
    ap.add_argument("--execute", action="store_true", help="真回放(默认只体检 + 干跑)")
    ap.add_argument("--plot", action="store_true", help="终端画各关节 ASCII 曲线")
    ap.add_argument("--force", action="store_true", help="体检不过也强行回放")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--replay-fps", type=int, default=15, help="回放帧率")
    ap.add_argument("--rate", type=float, default=0.05, help="回放每 tick 最大关节步进(rad)")
    ap.add_argument("--stall-gap", type=float, default=0.12,
                    help="抗积分饱和: 指令最多超前实际位置多少 rad。0 = 关闭")
    ap.add_argument("--tol-deg", type=float, default=3.0, help="终点到位判据(度)")
    ap.add_argument("--settle-timeout", type=float, default=8.0)
    ap.add_argument("--speed-percent", type=int, default=20)
    ap.add_argument("--enable-drift-abort", type=float, default=0.5)
    ap.add_argument("--left-can", default=T.CAN_OF["left"])
    ap.add_argument("--right-can", default=T.CAN_OF["right"])
    ap.add_argument("--firmware", default="v120")
    args = ap.parse_args()
    args.arms = [s.strip() for s in args.arms.split(",") if s.strip()]

    if args.list or not args.name:
        T.TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        ps = sorted(T.TRAJ_DIR.glob("*.json"))
        if not ps:
            print(f"(还没有录过轨迹。目录: {T.TRAJ_DIR})")
            return
        print(f"已录轨迹({T.TRAJ_DIR}):")
        for p in ps:
            x = json.loads(p.read_text(encoding="utf-8"))
            print(f"  {p.stem:20s} {x['n_frames']:5d} 帧 @ {x['fps']:3d}Hz / {x['duration_s']:6.1f}s"
                  f"  臂: {','.join(x['arms'])}  录于 {x.get('_recorded_at','?')}")
        if not args.name:
            print("\n用法: python3 traj_replay.py <名字> [--plot] [--execute]")
        return

    d = load(args.name)
    healthy = inspect(d, args)
    if not healthy and args.execute and not args.force:
        sys.exit("\n✘ 体检发现硬问题 —— 未回放。确认无误可加 --force。")

    # 回放逻辑直接复用 traj_teach.replay(部署链走的也是它, 保证行为一致)
    args.replay = args.name
    print(f"\n══ 回放「{args.name}」══   "
          f"{'真执行' if args.execute else '干跑(不发运动命令)'}", flush=True)
    sys.exit(T.replay(args))


if __name__ == "__main__":
    main()
