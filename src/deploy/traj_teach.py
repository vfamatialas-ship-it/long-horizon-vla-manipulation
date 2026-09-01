#!/usr/bin/env python3
"""过渡轨迹的手教录制与回放 —— 用在四个大阶段之间。

为什么要它
----------
goto_home.py 走的是**关节空间直线插值**: 起点到终点各关节按比例同步转。这条路径
在自由空间没问题, 但阶段之间往往要绕开箱子、避免两臂互撞 —— 直线插值不知道这些,
容易蹭到东西(蹭到就触发抗积分饱和停住, 于是"归不到位")。

手教一遍再回放, 路径就是你亲手示范的那条, 该绕的地方会绕。

主臂模式下怎么读关节(这是本脚本最容易踩的坑)
------------------------------------------------
``set_leader_mode()`` 会把 ``enable_can_push`` 关掉, 此时:
  · ``get_joint_angles()`` 返回的是**进入拖动前的旧缓存值** —— 看着有数, 其实不动
  · 手动拖动的实时位是作为 **leader 关节广播帧**发出的, 要用 ``read_leader_joints()``

``NeroArm.read_joints()`` 只在普通反馈**返回 None** 时才回退到 leader 帧; 而"返回旧
缓存"这种情况它不会回退。所以录制时必须**显式优先取 leader 帧**, 见 read_teach()。

夹爪(2026-08-30 加)
--------------------
拖动示教只让**手臂**进 leader 模式(``enter_drag_teach`` 只调 ``set_leader_mode``),
**夹爪不受影响** —— 它仍带电、手掰不动。所以爪宽没法"被拖出来", 只能录制时用键盘控:

    o = 张开(--grip-open)    c = 闭合(--grip-close)    [ / ] = 每次减/加 5mm

每帧同时记两个值:
    <side>_grip       你下发的目标宽度(米)  ← 回放照它复现
    <side>_grip_meas  夹爪实测宽度(米)      ← 用来核对到底夹到没有

⚠ 夹爪实测要求它**已回零**(read_gripper_status 的 homed=True)。没回零时 0x2A8 反馈
不上报, 实测恒为 0 —— 这不影响下发的目标宽度被录下来, 但 _meas 列会是常数。

老轨迹(没有 grip 字段)照常能读能回放, 只是回放时不动夹爪。

录制帧率
--------
默认 60Hz(--fps)。比控制环的 15Hz 高 4 倍, 回放时插值更细、复现更准。
回放按 --replay-fps(默认 15, 与控制环一致)重采样。

用法::

    # 1) 录一条 "E1 结束 → E2 开始" 的过渡轨迹
    python3 traj_teach.py --record e1_to_e2

    # 1b) 只录右臂, 录制中用 o/c 控夹爪(抓取类任务必须这么录)
    python3 traj_teach.py --record 右臂抓橙色物体_1 --arms right

    # 2) 看看录到了什么(不动机械臂)
    python3 traj_teach.py --replay e1_to_e2

    # 3) 真回放
    python3 traj_teach.py --replay e1_to_e2 --execute

    # 4) 列出已录的轨迹
    python3 traj_teach.py --list

录制流程: 脚本切主臂模式 → 你手动把两臂摆过去(其间按 o/c 控夹爪)→ 按回车结束
          → 自动切回从臂并钉位。
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
sys.path.insert(0, str(NERO_ROOT / "lerobot_data" / "lerobot_tools"))

from collect_dual_nero import KeyPad  # noqa: E402  (单键非阻塞读, 与采集脚本同一个)

GRIP_MAX = 0.105        # 夹爪工作范围上限(米), 与各 rollout 客户端一致

# 录制落盘目录。2026-08-30 起改到 lerobot_data/HY/replay(用户要求, 与采集数据放一处);
# 老的 deploy/trajectories/ 仍然**可读** —— 那三条(123/1234/hy601)还被
# run_pi05_rollout_hezi_ee_traj.py 的前缀回放用着, 不能搬走。
TRAJ_DIR = Path("<PROJECT_ROOT>/lerobot_data/HY/replay")
LEGACY_TRAJ_DIR = HERE / "trajectories"


def traj_dirs() -> list[Path]:
    """查找顺序: 新目录优先, 老目录兜底。"""
    out = [TRAJ_DIR]
    if LEGACY_TRAJ_DIR.resolve() != TRAJ_DIR.resolve():
        out.append(LEGACY_TRAJ_DIR)
    return out


def find_traj(name: str) -> Path | None:
    """按名字在所有目录里找轨迹文件, 找不到返回 None。"""
    for d in traj_dirs():
        q = d / f"{name}.json"
        if q.exists():
            return q
    return None


def next_free_name(name: str) -> str:
    """给 '<前缀>_<数字>' 形式的名字找下一个没被占用的编号 —— 连录多条时用。

    '右臂抓橙色物体_1' 已存在 → 返回 '右臂抓橙色物体_2'。
    名字不带 _数字 后缀时, 从 _2 开始试。
    """
    import re as _re
    m = _re.match(r"^(.*)_(\d+)$", name)
    stem, i = (m.group(1), int(m.group(2))) if m else (name, 1)
    while find_traj(f"{stem}_{i}") is not None:
        i += 1
    return f"{stem}_{i}"
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
    """打开 CAN 反馈推送。进过主臂/拖动模式后它会被关掉, 不开就读不到关节。"""
    try:
        arm.set_can_control_mode(enable_can_push=True)
        time.sleep(0.25)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ 打开反馈推送失败: {exc}", flush=True)


def read_teach(arm) -> np.ndarray | None:
    """主臂模式下读**实时**关节角。

    ⚠ 顺序不能反: 先试 leader 广播帧, 拿不到才退回普通反馈。
    反过来写(先普通后 leader)会读到进入拖动前的旧缓存 —— 录出来的轨迹是一条直线,
    因为每一帧都是同一个旧值。
    """
    try:
        lj = arm.read_leader_joints()
        if lj is not None:
            return np.asarray(lj, dtype=np.float64)[:7]
    except Exception:  # noqa: BLE001
        pass
    try:
        return np.asarray(arm.read_joints(), dtype=np.float64)[:7]
    except Exception:  # noqa: BLE001
        return None


def read_grip(arm) -> float:
    """普通(从臂)模式下的爪宽 —— 走 0x2A8 反馈帧。读不到返回 nan。"""
    try:
        return float(arm.read_gripper_width())
    except Exception:  # noqa: BLE001
        return float("nan")


def read_grip_teach(arm) -> float:
    """**拖动示教模式**下读手掰出来的爪宽(米)。读不到返回 nan。

    ⚠ 顺序不能反, 而且不能只读 0x2A8 —— 这是本文件最容易踩的第二个坑,
      与关节角那个(read_teach)是同一个机制:

      leader 模式下 enable_can_push=DISABLE, 手臂**不发**反馈帧(0x2A5~0x2A9),
      所以 0x2A8 根本不上总线, read_gripper_width() 永远读不到 / 恒 0。
      手掰出来的爪宽是作为**主臂控制帧 0x159 广播**出去的(主臂就是靠这个驱动从臂),
      要用 get_gripper_ctrl_states() 取。

      实测(2026-08-30, 右臂 can1 拖动中): can1 上只有 151/155/156/157/159/170,
      一个 2A* 都没有; 0x159 前 4 字节是 µm 级爪宽, 掰爪时从 0 连续变到 2889。
    """
    g = getattr(arm, "gripper", None)
    if g is not None:
        try:
            st = g.get_gripper_ctrl_states()
            if st is not None:
                return float(st.msg.value)      # 解析器已按 1e-6 换成米
        except Exception:  # noqa: BLE001
            pass
    return read_grip(arm)                        # 退回 0x2A8(从臂模式下才有)


def grip_homed(arm) -> bool | None:
    """夹爪是否已回零。None = 拿不到状态。"""
    try:
        st = arm.read_gripper_status()
        return None if st is None else bool(st.get("homed"))
    except Exception:  # noqa: BLE001
        return None


def grip_feed_alive(arm, wait_s: float = 2.5) -> tuple[bool, float, str]:
    """判夹爪 0x2A8 反馈通道死活。返回 (活着, 帧率Hz, 说明)。

    ⚠ 判据是 **hz + timestamp 在推进**, 不是"读到的值变不变" ——
      爪不动时值本来就恒定, 拿值判活会把好通道误判成死的(踩过这个坑)。
    ⚠ 也必须**给解析器留时间**: 刚 open 完立刻读一定是 None(实测约 50ms 才有第一帧)。

    典型健康值: hz ≈ 180。
    """
    g = getattr(arm, "gripper", None)
    if g is None:
        return False, 0.0, "没有夹爪执行器"
    # 拖动模式看 0x159(主臂广播), 从臂模式看 0x2A8(反馈) —— 哪个活用哪个
    getters = (("0x159主臂广播", g.get_gripper_ctrl_states),
               ("0x2A8反馈", g.get_gripper_status))
    t0 = time.monotonic()
    st, tag = None, ""
    while time.monotonic() - t0 < wait_s:
        for tag, fn in getters:
            try:
                st = fn()
            except Exception:  # noqa: BLE001
                st = None
            if st is not None and float(getattr(st, "hz", 0) or 0) > 0:
                break
        if st is not None and float(getattr(st, "hz", 0) or 0) > 0:
            break
        time.sleep(0.05)
    if st is None:
        return False, 0.0, f"{wait_s:.1f}s 内一帧没收到(0x159 与 0x2A8 都没上总线)"
    hz = float(getattr(st, "hz", 0) or 0)
    ts0 = getattr(st, "timestamp", None)
    time.sleep(0.35)
    st2 = g.get_gripper_status()
    ts1 = getattr(st2, "timestamp", None) if st2 is not None else None
    if ts0 is not None and ts1 is not None and ts1 == ts0:
        return False, hz, "时间戳不推进 —— 收到的是旧缓存, 通道已停"
    return True, hz, f"{hz:.0f}Hz via {tag}"


def send_grip(arm, width_m: float, force_n: float) -> bool:
    try:
        arm.set_gripper_width(float(width_m), force_n=float(force_n))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ⚠ 夹爪下发失败: {exc}", flush=True)
        return False


def read_settled(arm, n: int = 6) -> np.ndarray:
    q = None
    for k in range(n):
        try:
            q = np.asarray(arm.read_joints(), dtype=np.float64)[:7]
        except RuntimeError:
            if k == 0:
                refresh_feedback(arm)
                continue
            raise
        time.sleep(0.1)
    if q is None:
        raise RuntimeError("读不到关节反馈 —— 检查 CAN 是否 up")
    return q


# ────────────────────────────────────────────────────────────── 录制
def record(args) -> int:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAJ_DIR / f"{args.record}.json"
    dup = find_traj(args.record)
    if dup is not None and not args.overwrite:
        nxt = next_free_name(args.record)
        print(f"✘ 轨迹 [{args.record}] 已存在: {dup}", flush=True)
        print(f"   连录下一条就用:  --record {nxt}", flush=True)
        print(f"   非要覆盖这条加:  --overwrite", flush=True)
        return 2

    arms = {}
    for side in args.arms:
        arms[side] = open_arm(side, getattr(args, f"{side}_can"), args.firmware)
        refresh_feedback(arms[side])

    period = 1.0 / args.fps
    frames: list[dict] = []
    try:
        q0 = {s: read_settled(a) for s, a in arms.items()}
        for s in args.arms:
            print(f"  {s:5s} 起点 {np.round(np.degrees(q0[s]), 1).tolist()}", flush=True)

        print(f"\n⚠ 即将切【主臂模式】—— 切换后两臂会**卸力可自由拖动**, 请先用手托住!", flush=True)
        try:
            if input("  输入 TEACH 开始录制(其它=取消): ").strip() != "TEACH":
                return 1
        except (EOFError, KeyboardInterrupt):
            return 1

        for side, arm in arms.items():
            arm.enter_drag_teach()
        time.sleep(0.4)

        # ★★ 关键一步: 把反馈推送重新打开, 否则夹爪反馈是哑的 ★★
        #
        # set_leader_mode() 内部做了**两件独立的事**(见 SDK driver.py:1209):
        #     ① enable_can_push = DISABLE  + _set_mode()      ← 掐掉所有反馈帧
        #     ② _set_leader_follower_config(0xFA)             ← 这个才是零力拖动
        # ① 一关, 夹爪的 0x2A8 状态帧跟着一起没了 —— 于是 read_gripper_width()
        # 永远返回 0 / 抛异常, 表现为"录不到夹爪"。
        # 零力拖动由 ② 管, 与 ① 无关, 所以这里把 ① 打开不影响手能不能拖。
        if args.teach_feedback:
            for side, arm in arms.items():
                try:
                    arm.set_can_control_mode(enable_can_push=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠ {side} 恢复反馈推送失败: {exc}", flush=True)
            time.sleep(0.4)

        # ── 夹爪反馈体检 ──
        # 判据是**帧率与时间戳在推进**, 不是"读到的值变不变" —— 爪不动时值本来
        # 就恒定, 拿值判活会把好通道误判成死的。健康时 hz≈180。
        grip_live, grip_cmd = {}, {}
        for side, arm in arms.items():
            live, hz, why = grip_feed_alive(arm)
            w = read_grip(arm)
            w0 = float(np.clip(w, 0.0, GRIP_MAX)) if not np.isnan(w) else args.grip_open
            grip_cmd[side] = w0
            homed = grip_homed(arm)
            grip_live[side] = live
            if not live:
                print(f"  {side:5s} ✘ 夹爪反馈通道不通({why})", flush=True)
                print(f"        → 本条只能靠键盘 o/c 记指令。若刚进拖动就这样, "
                      f"检查 --teach-feedback 是否被关掉。", flush=True)
            else:
                print(f"  {side:5s} ✓ 夹爪反馈活着({why}), 当前 {w0*1000:5.1f}mm"
                      f"{'' if homed else '  ⚠ 未回零(homed=False)'}", flush=True)
                if not homed:
                    print(f"        → 未回零时读数可能不跟手 —— 开录后先掰两下看数字动不动,"
                          f"\n          不动就 Ctrl-C 放弃, 先把爪回零再录。", flush=True)
                else:
                    print(f"        → **直接用手掰爪, 逐帧录实测宽度**", flush=True)

        any_live = any(grip_live.values())
        print(f"\n▶ 录制中 @ {args.fps}Hz  —— 手动把两臂摆到目标位形", flush=True)
        if any_live:
            print(f"   夹爪: 直接用手掰, 实测宽度逐帧录下; 也可用 o/c/[/] 下发指令", flush=True)
        else:
            print(f"   夹爪: ⚠ 读不到实测 —— 只能按键盘 "
                  f"o=张到{args.grip_open*1000:.0f}mm  c=合到{args.grip_close*1000:.0f}mm  "
                  f"[ / ]=每次 ∓5mm", flush=True)
        print(f"   完成后按【回车】结束(Ctrl-C 放弃)", flush=True)

        t0 = time.monotonic()
        stale = 0
        with KeyPad() as keypad:
            keypad.drain()
            while True:
                tick = time.monotonic()

                # ── 键盘: 夹爪 / 结束 ──
                key = keypad.get(0.0)
                if key in ("\r", "\n"):
                    break
                if key:
                    k = key.lower()
                    tgt = None
                    if k == "o":
                        tgt = args.grip_open
                    elif k == "c":
                        tgt = args.grip_close
                    elif k == "]":
                        tgt = min(GRIP_MAX, max(grip_cmd.values()) + 0.005)
                    elif k == "[":
                        tgt = max(0.0, min(grip_cmd.values()) - 0.005)
                    if tgt is not None:
                        for side, arm in arms.items():
                            grip_cmd[side] = float(np.clip(tgt, 0.0, GRIP_MAX))
                            send_grip(arm, grip_cmd[side], args.grip_force)
                        print(f"\n    爪 → {tgt*1000:.0f}mm", flush=True)

                row = {"t": round(tick - t0, 5)}
                ok = True
                for side, arm in arms.items():
                    q = read_teach(arm)
                    if q is None:
                        ok = False
                        break
                    row[side] = [round(float(v), 5) for v in q]
                    # 夹爪。<side>_grip = 回放时要复现的宽度:
                    #   反馈活着 → 用**实测**(手掰出来的就是它, 这才是真正的示教)
                    #   反馈哑了 → 退回键盘下发的指令值
                    w = read_grip_teach(arm)      # ★ 拖动模式必须走 0x159
                    ok_w = not np.isnan(w)
                    row[f"{side}_grip_meas"] = round(float(w), 5) if ok_w else None
                    row[f"{side}_grip_cmd"] = round(float(grip_cmd[side]), 5)
                    if grip_live[side] and ok_w:
                        row[f"{side}_grip"] = round(float(np.clip(w, 0.0, GRIP_MAX)), 5)
                        row[f"{side}_grip_src"] = "meas"
                    else:
                        row[f"{side}_grip"] = round(float(grip_cmd[side]), 5)
                        row[f"{side}_grip_src"] = "key"
                if ok:
                    frames.append(row)
                    stale = 0
                else:
                    stale += 1
                    if stale > args.fps:      # 连续 1 秒读不到 → 别再闷头录
                        print("\n  ⚠ 连续 1s 读不到关节 —— 停止录制", flush=True)
                        break
                if len(frames) % max(args.fps, 1) == 0 and frames:
                    g = "/".join(f"{row[f'{x}_grip']*1000:.0f}" for x in arms)
                    src = row.get(f"{list(arms)[0]}_grip_src", "?")
                    print(f"    {row['t']:6.1f}s  {len(frames)} 帧  爪 {g}mm({src})",
                          end="\r", flush=True)
                if tick - t0 > args.max_seconds:
                    print(f"\n  ⚠ 到达 --max-seconds {args.max_seconds}s, 停止", flush=True)
                    break
                dt = time.monotonic() - tick
                if dt < period:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n  已放弃录制。", flush=True)
        frames = []
    finally:
        # 切回从臂并**以当前手摆位为伺服目标**钉住 —— 不这样做, 一进从臂会伺服到
        # SDK 里缓存的上一条 move_j 旧目标, 表现为"松手后手臂自己跑一段"。
        for side, arm in arms.items():
            try:
                refresh_feedback(arm)
                q_now = read_settled(arm, 4)
                arm.exit_drag_teach(hold_joints=[float(v) for v in q_now])
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ {side} 退出拖动模式失败: {exc}", flush=True)
        for arm in arms.values():
            arm.close()

    if len(frames) < 5:
        print(f"\n✘ 只录到 {len(frames)} 帧, 太短, 不保存。", flush=True)
        return 3

    dur = frames[-1]["t"]
    has_grip = any(f"{s_}_grip" in frames[0] for s_ in args.arms)
    srcs = sorted({f.get(f"{s_}_grip_src") for s_ in args.arms for f in frames} - {None})
    data = {
        "_comment": "手教过渡轨迹。由 traj_teach.py --record 录制, --replay 回放。",
        "_recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": args.record,
        "fps": args.fps,
        "arms": args.arms,
        # has_gripper: 老轨迹(2026-08-30 之前)没有夹爪列, 读的一方据此判断
        "has_gripper": bool(has_grip),
        # grip_source: "meas"=手掰实测(真示教) / "key"=键盘下发 / 两者都有则混合
        "grip_source": "+".join(srcs) if srcs else None,
        "grip_open": args.grip_open,
        "grip_close": args.grip_close,
        "n_frames": len(frames),
        "duration_s": round(dur, 3),
        "frames": frames,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n✓ 已录 {len(frames)} 帧 / {dur:.1f}s @ {args.fps}Hz → {path}", flush=True)
    for s_ in args.arms:
        A = np.array([f[s_] for f in frames])
        span = np.degrees(A.max(0) - A.min(0))
        print(f"  {s_:5s} 各关节行程(度): {np.round(span, 1).tolist()}", flush=True)
        if has_grip:
            G = np.array([f[f"{s_}_grip"] for f in frames], dtype=float)
            src = frames[0].get(f"{s_}_grip_src", "?")
            rng = (G.max() - G.min()) * 1000
            print(f"  {s_:5s} 夹爪({'手掰实测' if src == 'meas' else '键盘指令'}): "
                  f"{G.min()*1000:.1f}~{G.max()*1000:.1f}mm, 幅度 {rng:.1f}mm", flush=True)
            if rng < 1.0:
                print(f"        ✘✘ 爪宽全程几乎没变(幅度 {rng:.1f}mm) —— "
                      f"这条**训抓取用不了**!", flush=True)
                if src == "meas":
                    print(f"           反馈是活的但值不动 → 爪没被掰动, 或没回零。", flush=True)
                else:
                    print(f"           反馈是哑的且没按 o/c → 先修反馈(见 --teach-feedback), "
                          f"或录制时按 o/c。", flush=True)
    return 0


# ────────────────────────────────────────────────────────────── 回放
def replay(args) -> int:
    path = find_traj(args.replay) or (TRAJ_DIR / f"{args.replay}.json")
    if not path.exists():
        print(f"✘ 没有这条轨迹: {path}\n   已录的: "
              f"{[q.stem for d in traj_dirs() for q in sorted(d.glob('*.json'))]}", flush=True)
        return 2
    data = json.loads(path.read_text(encoding="utf-8"))
    sides = [s for s in data["arms"] if s in args.arms]
    F = data["frames"]
    src_t = np.array([f["t"] for f in F])
    traj = {s: np.array([f[s] for f in F]) for s in sides}

    # 按回放帧率重采样(录制 60Hz → 回放 15Hz, 逐关节线性插值)
    n_out = max(int(round(data["duration_s"] * args.replay_fps)), 2)
    out_t = np.linspace(src_t[0], src_t[-1], n_out)
    res = {s: np.stack([np.interp(out_t, src_t, traj[s][:, j]) for j in range(7)], axis=1)
           for s in sides}

    # 夹爪时间线。用 **前值保持** 而不是线性插值 —— 夹爪是离散的开/合指令,
    # 插值会造出"半开"的中间态, 回放时爪会缓慢蠕动而不是干脆地开合。
    grip = {}
    if data.get("has_gripper") and args.replay_grip:
        for s in sides:
            key = f"{s}_grip"
            if key not in F[0]:
                continue
            g_src = np.array([f[key] for f in F])
            idx = np.searchsorted(src_t, out_t, side="right") - 1
            grip[s] = g_src[np.clip(idx, 0, len(g_src) - 1)]

    print(f"  轨迹 {data['name']}: {data['n_frames']} 帧 @ {data['fps']}Hz / "
          f"{data['duration_s']:.1f}s", flush=True)
    print(f"  重采样 → {n_out} 帧 @ {args.replay_fps}Hz", flush=True)
    if grip:
        for s in grip:
            n_chg = int((np.abs(np.diff(grip[s])) > 1e-6).sum())
            print(f"  {s:5s} 夹爪会复现: {grip[s].min()*1000:.0f}~{grip[s].max()*1000:.0f}mm, "
                  f"{n_chg} 次变化", flush=True)
    elif data.get("has_gripper"):
        print("  (--no-replay-grip: 不复现夹爪)", flush=True)
    else:
        print("  ⚠ 这条轨迹没录夹爪(2026-08-30 之前录的) —— 回放不会动夹爪", flush=True)

    arms = {}
    for side in sides:
        arms[side] = open_arm(side, getattr(args, f"{side}_can"), args.firmware)
        refresh_feedback(arms[side])
    try:
        starts = {s: read_settled(a) for s, a in arms.items()}
        for s in sides:
            gap = np.degrees(np.abs(starts[s] - res[s][0]).max())
            print(f"  {s:5s} 当前位与轨迹起点相差 {gap:.1f}°"
                  + ("  ⚠ 偏差较大, 回放会先走过去" if gap > 10 else ""), flush=True)

        if not args.execute:
            print("  (干跑: 不发运动命令。加 --execute 真回放)", flush=True)
            return 0

        for side, arm in arms.items():
            print(f"  使能 {side}(speed={args.speed_percent}%) …", flush=True)
            arm.enable(speed_percent=args.speed_percent,
                       hold_joints=[float(v) for v in starts[side]])
        time.sleep(0.3)
        for side, arm in arms.items():
            d = float(np.abs(read_settled(arm, 3) - starts[side]).max())
            if d > args.enable_drift_abort:
                print(f"  🛑 {side} 使能后漂移 {d:.3f} rad — 停止回放(未下电)", flush=True)
                return 3

        period = 1.0 / args.replay_fps
        cmd = {s: starts[s].copy() for s in sides}
        last_g = {s: None for s in sides}
        for k in range(n_out):
            tick = time.monotonic()
            for side in grip:                      # 夹爪: 变了才发, 免得 15Hz 刷爆总线
                g = float(grip[side][k])
                if last_g[side] is None or abs(g - last_g[side]) > 1e-4:
                    send_grip(arms[side], g, args.grip_force)
                    last_g[side] = g
            for side, arm in arms.items():
                q_now = np.asarray(arm.read_joints(), dtype=np.float64)[:7]
                step = np.clip(res[side][k] - cmd[side], -args.rate, args.rate)
                nxt = cmd[side] + step
                if args.stall_gap > 0:   # 抗积分饱和: 顶到东西就停着轻轻顶, 不越顶越狠
                    nxt = np.clip(nxt, q_now - args.stall_gap, q_now + args.stall_gap)
                cmd[side] = nxt
                arm.move_joints([float(v) for v in nxt])
            if (k + 1) % args.replay_fps == 0:
                print(f"    {(k+1)/args.replay_fps:5.1f}s  {k+1}/{n_out}", end="\r", flush=True)
            dt = time.monotonic() - tick
            if dt < period:
                time.sleep(period - dt)

        # 收尾: 朝轨迹终点再收敛一会儿, 免得因限幅/抗饱和差最后几度
        tgt = {s: res[s][-1] for s in sides}
        t_end = time.monotonic() + args.settle_timeout
        while time.monotonic() < t_end:
            tick = time.monotonic()
            worst = 0.0
            for side, arm in arms.items():
                q_now = np.asarray(arm.read_joints(), dtype=np.float64)[:7]
                worst = max(worst, float(np.abs(q_now - tgt[side]).max()))
                step = np.clip(tgt[side] - cmd[side], -args.rate, args.rate)
                nxt = cmd[side] + step
                if args.stall_gap > 0:
                    nxt = np.clip(nxt, q_now - args.stall_gap, q_now + args.stall_gap)
                cmd[side] = nxt
                arm.move_joints([float(v) for v in nxt])
            if worst <= np.radians(args.tol_deg):
                break
            dt = time.monotonic() - tick
            if dt < period:
                time.sleep(period - dt)

        rc = 0
        print("", flush=True)
        for side, arm in arms.items():
            q = read_settled(arm, 4)
            err = np.degrees(np.abs(q - tgt[side]))
            good = err.max() <= args.tol_deg
            rc |= 0 if good else 1
            print(f"  {'✓' if good else '⚠'} {side:5s} 终点误差 最大 {err.max():.2f}° "
                  f"(j{int(err.argmax())})  判据 ≤{args.tol_deg}°", flush=True)
        return rc
    finally:
        for arm in arms.values():
            arm.close()      # 只关 CAN, **不失能**: 该硬件下电抱闸不保持


def main() -> None:
    global TRAJ_DIR          # --out-dir 会改它
    ap = argparse.ArgumentParser(description="过渡轨迹手教录制 / 回放")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", metavar="NAME", help="录制一条轨迹并以该名字保存")
    g.add_argument("--replay", metavar="NAME", help="回放指定轨迹")
    g.add_argument("--list", action="store_true", help="列出已录的轨迹")
    ap.add_argument("--execute", action="store_true", help="回放时真发运动命令(默认干跑)")
    ap.add_argument("--overwrite", action="store_true", help="录制时允许覆盖同名轨迹")
    ap.add_argument("--out-dir", default="",
                    help=f"改落盘目录(默认 {TRAJ_DIR})")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--fps", type=int, default=60, help="录制帧率(越高回放越精准)")
    ap.add_argument("--replay-fps", type=int, default=15, help="回放帧率, 与控制环一致")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="单条录制时长上限")
    ap.add_argument("--grip-open", type=float, default=0.090,
                    help="录制/回放中按 o 张到多宽(米)")
    ap.add_argument("--grip-close", type=float, default=0.000,
                    help="录制/回放中按 c 合到多窄(米)")
    ap.add_argument("--grip-force", type=float, default=1.0, help="夹持力(N)")
    # 默认 **关**。2026-08-30 实测: 拖动中的爪宽走 0x159 主臂广播, leader 模式下本来
    # 就在总线上, 不需要重开推送; 而这个开关发的 0x151 模式帧会把 leader 广播打断
    # (实测发完 can1 上 0x159/0x2A8 全没了)。留着只为排查用。
    ap.add_argument("--teach-feedback", action=argparse.BooleanOptionalAction, default=False,
                    help="进拖动后重开 CAN 反馈推送(默认关)。**一般不要开** —— "
                         "它会发模式帧, 实测会打断主臂广播, 反而读不到爪宽")
    ap.add_argument("--replay-grip", action=argparse.BooleanOptionalAction, default=True,
                    help="回放时复现录下来的夹爪时间线(默认开; 老轨迹没有 grip 字段则自动跳过)")
    ap.add_argument("--rate", type=float, default=0.05, help="回放每 tick 最大关节步进(rad)")
    ap.add_argument("--stall-gap", type=float, default=0.12,
                    help="抗积分饱和: 指令最多超前实际位置多少 rad。0 = 关闭")
    ap.add_argument("--tol-deg", type=float, default=3.0, help="终点到位判据(度)")
    ap.add_argument("--settle-timeout", type=float, default=8.0, help="终点收敛的额外时间")
    ap.add_argument("--speed-percent", type=int, default=20)
    ap.add_argument("--enable-drift-abort", type=float, default=0.5)
    ap.add_argument("--left-can", default=CAN_OF["left"])
    ap.add_argument("--right-can", default=CAN_OF["right"])
    ap.add_argument("--firmware", default="v120")
    args = ap.parse_args()
    args.arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    if args.out_dir:
        TRAJ_DIR = Path(args.out_dir).expanduser().resolve()

    if args.list:
        TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        ps = [q for d in traj_dirs() for q in sorted(d.glob("*.json"))]
        if not ps:
            print(f"(还没有录过轨迹。目录: {', '.join(str(d) for d in traj_dirs())})", flush=True)
            return
        print(f"已录轨迹(录制落盘 → {TRAJ_DIR}):", flush=True)
        for p in ps:
            d = json.loads(p.read_text(encoding="utf-8"))
            where = "" if p.parent.resolve() == TRAJ_DIR.resolve() else "  [旧目录]"
            print(f"  {p.stem:24s} {d['n_frames']:5d} 帧 @ {d['fps']}Hz "
                  f"/ {d['duration_s']:6.1f}s  臂: {','.join(d['arms'])}  "
                  f"录于 {d.get('_recorded_at','?')}{where}", flush=True)
        return

    if args.record:
        print(f"\n══ 录制过渡轨迹「{args.record}」══  {args.fps}Hz  臂: {','.join(args.arms)}",
              flush=True)
        sys.exit(record(args))
    print(f"\n══ 回放过渡轨迹「{args.replay}」══  臂: {','.join(args.arms)}", flush=True)
    sys.exit(replay(args))


if __name__ == "__main__":
    main()
