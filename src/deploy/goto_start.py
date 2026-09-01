#!/usr/bin/env python3
"""回到【最初始位置】—— 整轮开跑前的摆位。**只给人手动用,任何自动流程都不调它。**

和 goto_home.py 的分工
----------------------
  goto_home.py  →  home_pose.json   阶段间归位点。**四个 rollout 客户端每段结束
                                     自动调用**, 让下一段从与训练一致的位形起步。
  goto_start.py →  start_pose.json  最初始位置。**没有任何脚本会自动调用它** ——
                                     只有你想把两臂摆回"整轮开始前的样子"时手动跑。

实现直接复用 goto_home 的闭环归位(插值 + 速率限幅 + 抗积分饱和 + 到位校验 +
全程不失能), 只把目标文件换掉, 不重写一遍逻辑。

用法::

    python3 goto_start.py             # 干跑, 只看要转多少
    python3 goto_start.py --execute   # 真回到最初始位置
    python3 goto_start.py --capture   # 把当前摆位存为新的最初始位置
    python3 goto_start.py --execute --arms left    # 只动一条臂
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import goto_home as G  # noqa: E402


def main() -> None:
    # 把目标文件从 home_pose.json 换成 start_pose.json —— 其余逻辑完全复用
    G.HOME_FILE = HERE / "start_pose.json"

    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(
        description="回到最初始位置(start_pose.json)。只供手动使用。")
    ap.add_argument("--capture", action="store_true", help="把当前摆位存为最初始位置(只读)")
    ap.add_argument("--execute", action="store_true", help="真执行(默认干跑不发运动命令)")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--rate", type=float, default=0.05, help="每 tick 最大关节步进(rad)")
    ap.add_argument("--stall-gap", type=float, default=0.12,
                    help="抗积分饱和: 指令最多超前实际位置多少 rad。0 = 关闭")
    ap.add_argument("--settle", type=float, default=0.6, help="到点后保持钉位的秒数")
    ap.add_argument("--tol-deg", type=float, default=3.0, help="到位判据(度)")
    # 与 goto_home 一致: 回到最初始位置时夹爪也闭合。goto() 的实现是共用的,
    # 这里只需要把同名参数补齐, 否则 send_grip 取不到 args.grip 会 AttributeError。
    ap.add_argument("--grip", type=float, default=0.0,
                    help="夹爪目标开度(米)。默认 0 = 完全闭合")
    ap.add_argument("--grip-force", type=float, default=1.0, help="夹爪力(N)")
    ap.add_argument("--no-grip", action="store_true", help="不动夹爪(旧行为)")
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--speed-percent", type=int, default=20)
    ap.add_argument("--enable-drift-abort", type=float, default=0.5)
    ap.add_argument("--left-can", default=G.CAN_OF["left"])
    ap.add_argument("--right-can", default=G.CAN_OF["right"])
    ap.add_argument("--firmware", default="v120")
    args = ap.parse_args()
    args.arms = [s.strip() for s in args.arms.split(",") if s.strip()]

    print(f"\n══ 回到最初始位置 ══   {G.HOME_FILE.name}   臂: {', '.join(args.arms)}", flush=True)
    if args.capture:
        G.capture(args)
        return
    sys.exit(G.goto(args))


if __name__ == "__main__":
    main()
