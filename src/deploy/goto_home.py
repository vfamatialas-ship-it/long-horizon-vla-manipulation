#!/usr/bin/env python3
"""双臂归位 —— 大阶段之间把两臂送回同一个摆位。

为什么需要它
------------
四个大阶段用的是四个不同的模型,动作语义各不相同。E0 结束时手臂停在"刚把盒子
放进打包箱"的位形,而 E1 的模型是从"你摆好的初始位"开始训练的 —— 直接接续等于
让 E1 从一个它训练时从没见过的位形起步,这正是串跑效果远差于单跑的主要原因。

在两段之间插一段**关节空间的插值轨迹**,把两臂送回同一个起点,后面每一段拿到的
初始条件就和单独跑时一致了。

安全设计(和 rollout 客户端同一套口径)
--------------------------------------
· 归位点从 home_pose.json 读,已核对落在 joint_envelope.json 的训练包络内
· 关节空间**线性插值** + 逐 tick 速率限幅(--rate),不会跳变
· **抗积分饱和**: 指令不得超前实际位置 --stall-gap 以上。顶到东西就停在那里轻轻
  顶着, 而不是越顶越狠 —— 防的是左臂 J6 那次 9A 过流跳闸
· 到位判据是**实际位置**而非指令位置; 超时(--timeout)就停下并如实报告,不假装成功
· 全程不失能。该硬件下电后抱闸不保持,臂会砸下来

用法::

    # 读当前摆位存为新的归位点(只读,不动机械臂)
    python3 goto_home.py --capture

    # 干跑: 打印轨迹但不发运动命令
    python3 goto_home.py

    # 真执行
    python3 goto_home.py --execute

    # 只归一条臂
    python3 goto_home.py --execute --arms left
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NERO_ROOT = HERE.parent
sys.path.insert(0, str(NERO_ROOT / "nero_control"))

HOME_FILE = HERE / "home_pose.json"
FPS = 15
PERIOD = 1.0 / FPS
def _resolve_can_of() -> dict:
    """按 USB-CAN 适配器的**序列号**认臂, 而不是写死 can0/can1。

    ⚠ 为什么不能写死: 接口号是内核按枚举顺序给的, 重插 USB / 重启换个口就会对调。
    实测 2026-08-29 重新接线后 can0 变成了左臂(之前是右臂) —— 写死的话
    goto_home 会把两条臂的目标位姿各发给对方, 那是很危险的一次运动。

    序列号来自 rollout_boxpick_common.CAN_SERIALS(rollout 客户端一直这么认)。
    解析不到就退回写死的默认值, 并打印一行提示。
    """
    fallback = {"left": "can1", "right": "can0"}
    try:
        import rollout_boxpick_common as _C          # noqa: PLC0415
        import run_pi05_deploy_hezi_ee as _D         # noqa: PLC0415
    except Exception:                                # noqa: BLE001
        return fallback
    out, miss = {}, []
    for side, ser in _C.CAN_SERIALS.items():
        got = _D.resolve_can_by_serial(ser)
        if got:
            out[side] = got
        else:
            miss.append(side)
            out[side] = fallback[side]
    if miss:
        print(f"  ⚠ {', '.join(miss)} 的 CAN 未按序列号解析到, 用默认值 "
              f"{ {s: fallback[s] for s in miss} }", flush=True)
    if out != fallback:
        print(f"  · CAN 按序列号解析: {out}", flush=True)
    return out


CAN_OF = _resolve_can_of()


def open_arm(side: str, chan: str, fw: str = "v120"):
    from single_nero_driver import NeroArm, NeroArmConfig  # noqa: PLC0415
    return NeroArm(NeroArmConfig(name=side, can_channel=chan, firmware=fw,
                                 can_interface="socketcan", bitrate=1_000_000, timeout=1.0))


def refresh_feedback(arm) -> None:
    """打开 CAN 反馈推送。

    ⚠ 必做。臂进过主臂/拖动模式后 enable_can_push 会被关掉, 此时 read_joints()
    直接抛 "joint feedback unavailable" —— 第一版漏了这步, 归位脚本一上来就崩。
    rollout 客户端在 safe_enable() 前也是先调这个(见 run_pi05_rollout_*.py 的
    refresh_feedback)。
    """
    try:
        arm.set_can_control_mode(enable_can_push=True)
        time.sleep(0.25)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ 打开反馈推送失败: {exc}", flush=True)


def read_fresh(arm, n: int = 6) -> np.ndarray:
    """多读几拍再取值 —— CAN 反馈是异步刷新的,单次读可能拿到上一帧。

    读不到就重开一次反馈推送再试, 而不是直接崩 —— 反馈偶尔会因为模式切换掉一拍。
    """
    q = None
    for k in range(n):
        try:
            q = np.asarray(arm.read_joints(), dtype=np.float64)[:7]
        except RuntimeError:
            if k == 0:
                refresh_feedback(arm)
                continue
            raise
        time.sleep(0.12)
    if q is None:
        raise RuntimeError("读不到关节反馈 —— 检查 CAN 是否 up、臂是否在主臂/拖动模式")
    return q


def capture(args) -> None:
    """把当前摆位存成归位点。只读,不发任何运动命令。"""
    out = {"_comment": "大阶段之间的归位点。由 goto_home.py --capture 从真机当前摆位读取。",
           "_captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "_note": "两臂 7 关节(rad) + 夹爪(m)。"}
    for side in args.arms:
        arm = open_arm(side, getattr(args, f"{side}_can"), args.firmware)
        try:
            refresh_feedback(arm)
            q = read_fresh(arm)
            w = float(arm.read_gripper_width())
            out[side] = {"joints": [round(float(v), 4) for v in q], "gripper": round(w, 4)}
            print(f"  {side:5s} {np.round(np.degrees(q), 2).tolist()}°  爪 {w:.4f} m", flush=True)
        finally:
            arm.close()
    # 保留没重新采集的那条臂
    if HOME_FILE.exists():
        old = json.loads(HOME_FILE.read_text(encoding="utf-8"))
        for s in ("left", "right"):
            if s not in out and s in old:
                out[s] = old[s]
    HOME_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n✓ 已写入 {HOME_FILE}", flush=True)


def assert_alive(arm, side: str, chan: str | None = None) -> bool:
    """确认这条臂真的活着, 而不是驱动在返回掉电前的缓存值。

    ⚠ 为什么必须查: 臂断电后 read_joints() 仍会返回**最后一帧**, 归位循环照跑不误、
    误差算成 0, 最后打印 "✓ 到位" —— 一条没通电的臂被判成归位成功。

    两道判据, 任一过即认为活着:
      ① get_arm_status() 不是 None
      ② CAN 接口的 rx_packets 在半秒内还在涨

    ⚠ 只用 ①**不够**: 实测臂在示教模式(ctrl_mode=0x06)时 get_arm_status() 会返回
      None, 而关节角读得又准又新。只认 ① 会把好臂judge成死臂, 拒绝归位。
      ② 才是真正的「线上有没有数据」, 与驱动的状态缓存无关。
    """
    try:
        if arm.arm.get_arm_status() is not None:
            return True
    except Exception:  # noqa: BLE001
        pass

    if chan:
        f = Path(f"/sys/class/net/{chan}/statistics/rx_packets")
        try:
            a = int(f.read_text())
            time.sleep(0.5)
            b = int(f.read_text())
            if b > a:
                return True
            print(f"  🛑 {side} ({chan}) 半秒内一帧没收到 —— 该臂没上电 / CAN 线掉了。",
                  flush=True)
            print(f"      不归位(否则会拿掉电前的缓存值算出\"✓ 到位\"这种假结果)。",
                  flush=True)
            print(f"      体检: ./check_arms.sh", flush=True)
            return False
        except OSError:
            pass

    print(f"  ⚠ {side} 读不到 arm_status, 也查不到 {chan} 的收包计数 —— 无法确认死活。",
          flush=True)
    print(f"      体检: ./check_arms.sh", flush=True)
    return False


def send_grip(arms: dict, args, tag: str = "") -> None:
    """把两臂夹爪送到 args.grip(默认 0 = 完全闭合)。

    为什么归位要顺带闭爪: 阶段之间臂要穿过箱子上方, 张开的爪比闭合的爪宽一大截,
    最容易蹭到的就是它。到位判据只看 7 个关节, 爪张着也算"归位成功", 所以不显式
    闭一下就会带着张开的爪进下一段。

    夹爪是宽度指令, 不走关节那套限幅闭环 —— 发一次即可, 不需要逐拍推进。
    """
    if getattr(args, "no_grip", False) or not args.execute:
        return
    w = float(max(args.grip, 0.0))
    for side, arm in arms.items():
        try:
            arm.set_gripper_width(w, force_n=float(args.grip_force))
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ {side} 夹爪指令失败({exc}) —— 关节归位不受影响", flush=True)
    if tag:
        print(f"  夹爪 → {w*1000:.0f}mm ({tag})", flush=True)


def goto(args) -> int:
    home = json.loads(HOME_FILE.read_text(encoding="utf-8"))
    arms, targets, starts = {}, {}, {}
    for side in args.arms:
        if side not in home:
            print(f"✘ {HOME_FILE} 里没有 {side} 的归位点 —— 先跑 --capture", flush=True)
            return 2
        arms[side] = open_arm(side, getattr(args, f"{side}_can"), args.firmware)
        refresh_feedback(arms[side])
        if not assert_alive(arms[side], side, getattr(args, f"{side}_can")):
            for a in arms.values():
                a.close()
            return 4
        targets[side] = np.asarray(home[side]["joints"], dtype=np.float64)[:7]

    try:
        for side, arm in arms.items():
            starts[side] = read_fresh(arm)
            d = np.degrees(np.abs(targets[side] - starts[side]))
            print(f"  {side:5s} 当前 {np.round(np.degrees(starts[side]),1).tolist()}", flush=True)
            print(f"  {side:5s} 目标 {np.round(np.degrees(targets[side]),1).tolist()}", flush=True)
            print(f"  {side:5s} 最大需转 {d.max():.1f}°  (j{int(d.argmax())})", flush=True)
            if not getattr(args, "no_grip", False):
                try:
                    print(f"  {side:5s} 夹爪 当前 {arm.read_gripper_width()*1000:.1f}mm "
                          f"→ 目标 {args.grip*1000:.0f}mm", flush=True)
                except Exception:  # noqa: BLE001
                    pass

        # 预估用时(仅供显示)。真正的结束条件是"实际位置进容差", 见下面的闭环。
        need = max(float(np.abs(targets[s] - starts[s]).max()) for s in arms)
        eta = need / max(args.rate, 1e-6) / FPS
        # 超时按需转角自适应: 转得多就给更长时间, 但至少 15s。
        args.timeout = max(args.timeout, eta * 3.0 + 10.0)
        print(f"\n  预计 ≈{eta:.1f}s  (限幅 {args.rate} rad/tick, 抗饱和 {args.stall_gap} rad, "
              f"超时 {args.timeout:.0f}s)", flush=True)
        print(f"  结束条件: 两臂实际位置都进 ±{args.tol_deg}° 并稳住 {args.settle}s", flush=True)
        if not args.execute:
            print("  (干跑: 不发运动命令。加 --execute 真执行)", flush=True)
            return 0

        # ── 使能(以当前实读位为伺服目标, 防冲) ────────────────────────────────
        # ⚠ 不使能的话 move_joints 发下去电机不动 —— 第一版就漏了这步。
        #   hold_joints=当前位 是防冲关键: 使能瞬间伺服目标 == 实际位置, 不会弹跳。
        for side, arm in arms.items():
            print(f"  使能 {side}(speed={args.speed_percent}%) …", flush=True)
            arm.enable(speed_percent=args.speed_percent,
                       hold_joints=[float(v) for v in starts[side]])
        time.sleep(0.3)
        for side, arm in arms.items():       # 防冲检查: 使能后不该自己跑掉
            q = np.asarray(arm.read_joints(), dtype=np.float64)[:7]
            d = float(np.abs(q - starts[side]).max())
            if d > args.enable_drift_abort:
                print(f"  🛑 {side} 使能后漂移 {d:.3f} rad > {args.enable_drift_abort}"
                      f" — 停止归位(未下电, 臂保持使能)", flush=True)
                return 3
            print(f"  ✓ {side} 使能通过防冲检测(漂移 {d:.4f} rad)", flush=True)

        # 闭爪与关节运动同时进行 —— 爪先收好, 免得张着爪穿过箱子上方
        send_grip(arms, args, "使能后")

        # ── 走到位为止, 不是"跑固定拍数就收工" ──────────────────────────────
        # ⚠ 第一版按 n = 需转角/限幅×1.3 算固定拍数, 跑完就退出 —— 结果一次归不到位,
        #   要跑第二次才行。原因: 抗积分饱和把指令钳在实际位置 ±stall_gap 内, 臂走得
        #   慢时指令也被拖慢, 1.3 倍余量根本不够。
        #   现在改成闭环: **每拍都朝目标推进, 直到实际位置进容差**(或超时)。
        #   插值系数不再按固定总拍数算, 而是按"还剩多远"自适应 —— 走不动时不会
        #   把目标点甩在前面, 走得动时也不会拖沓。
        cmd = {s: starts[s].copy() for s in arms}
        t_end = time.monotonic() + args.timeout
        tol = np.radians(args.tol_deg)
        arrived_since = None
        k = 0
        while True:
            tick = time.monotonic()
            now = {}
            for side, arm in arms.items():
                q_now = np.asarray(arm.read_joints(), dtype=np.float64)[:7]
                now[side] = q_now
                # 目标方向上每拍推进 rate, 直接朝 target 走(不再用全局插值系数)
                step = np.clip(targets[side] - cmd[side], -args.rate, args.rate)
                nxt = cmd[side] + step
                if args.stall_gap > 0:      # 抗积分饱和: 指令别跑到实际位置前面太多
                    nxt = np.clip(nxt, q_now - args.stall_gap, q_now + args.stall_gap)
                cmd[side] = nxt
                arm.move_joints([float(v) for v in nxt])
            k += 1

            worst = max(float(np.abs(now[s] - targets[s]).max()) for s in arms)
            if worst <= tol:
                if arrived_since is None:
                    arrived_since = time.monotonic()
                    print(f"\n  ✓ 已进容差(最大 {np.degrees(worst):.2f}°), 保持钉位 "
                          f"{args.settle}s …", flush=True)
                elif time.monotonic() - arrived_since >= args.settle:
                    break                      # 稳住了才算真到位
            else:
                arrived_since = None
            if k % 15 == 0:
                print(f"    {k/FPS:5.1f}s  最大剩余 {np.degrees(worst):6.2f}°", flush=True)
            if time.monotonic() > t_end:
                print(f"\n  ⚠ 超过 --timeout {args.timeout}s (最大剩余 "
                      f"{np.degrees(worst):.2f}°), 停止归位", flush=True)
                break
            dt = time.monotonic() - tick
            if dt < PERIOD:
                time.sleep(PERIOD - dt)

        # 到位后补发一次: 途中若被夹爪自身的力控回弹/或指令丢包, 这一次兜底
        send_grip(arms, args, "到位后")
        if args.execute and not getattr(args, "no_grip", False):
            time.sleep(0.5)

        rc = 0
        print("", flush=True)
        for side, arm in arms.items():
            q = read_fresh(arm, 4)
            err = np.degrees(np.abs(q - targets[side]))
            good = err.max() <= args.tol_deg
            rc |= 0 if good else 1
            print(f"  {'✓' if good else '⚠'} {side:5s} 到位误差 最大 {err.max():.2f}° "
                  f"(j{int(err.argmax())})  判据 ≤{args.tol_deg}°", flush=True)
            if not good:
                print(f"      逐关节 {np.round(err,2).tolist()}", flush=True)
                print("      多半是被挡住了(抗饱和会主动停止顶) —— 检查现场后重试", flush=True)
            if args.execute and not getattr(args, "no_grip", False):
                try:
                    gw = arm.read_gripper_width()
                    gok = abs(gw - args.grip) <= 0.003          # 3mm 容差
                    print(f"      {'✓' if gok else '⚠'} 夹爪 {gw*1000:.1f}mm "
                          f"(目标 {args.grip*1000:.0f}mm)"
                          + ("" if gok else "  —— 可能夹到东西了, 检查现场"), flush=True)
                    rc |= 0 if gok else 1
                except Exception:  # noqa: BLE001
                    pass
        return rc
    finally:
        for arm in arms.values():
            arm.close()      # 只关 CAN, **不失能**: 该硬件下电抱闸不保持


def main() -> None:
    ap = argparse.ArgumentParser(description="双臂归位到 home_pose.json 的摆位")
    ap.add_argument("--capture", action="store_true", help="读当前摆位存为归位点(只读)")
    ap.add_argument("--execute", action="store_true", help="真执行(默认干跑不发运动命令)")
    ap.add_argument("--arms", default="left,right", help="要处理的臂, 逗号分隔")
    ap.add_argument("--rate", type=float, default=0.05,
                    help="每 tick 最大关节步进(rad)。归位是自由空间运动, 比 rollout 保守些")
    ap.add_argument("--stall-gap", type=float, default=0.12,
                    help="抗积分饱和: 指令最多超前实际位置多少 rad。0 = 关闭")
    ap.add_argument("--settle", type=float, default=0.6, help="到点后保持钉位的秒数")
    ap.add_argument("--tol-deg", type=float, default=3.0, help="到位判据(度)")
    ap.add_argument("--grip", type=float, default=0.0,
                    help="归位时夹爪目标开度(米)。默认 0 = 完全闭合")
    ap.add_argument("--grip-force", type=float, default=1.0, help="夹爪力(N)")
    ap.add_argument("--no-grip", action="store_true", help="不动夹爪(旧行为)")
    ap.add_argument("--timeout", type=float, default=40.0, help="归位总超时(秒)")
    ap.add_argument("--speed-percent", type=int, default=20,
                    help="使能时的机械臂速度百分比")
    ap.add_argument("--enable-drift-abort", type=float, default=0.5,
                    help="使能后允许的最大漂移(rad), 超过就停止归位")
    ap.add_argument("--left-can", default=CAN_OF["left"])
    ap.add_argument("--right-can", default=CAN_OF["right"])
    ap.add_argument("--firmware", default="v120")
    args = ap.parse_args()
    args.arms = [s.strip() for s in args.arms.split(",") if s.strip()]

    print(f"\n══ 双臂归位 ══   {HOME_FILE.name}   臂: {', '.join(args.arms)}", flush=True)
    if args.capture:
        capture(args)
        return
    sys.exit(goto(args))


if __name__ == "__main__":
    main()
