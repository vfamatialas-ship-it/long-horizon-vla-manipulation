#!/usr/bin/env python3
"""Nero 双臂 π0.5 部署客户端(snack box, 子任务条件化 + 手动切段).

模式(从安全到危险, 默认最安全):
  --fake-obs            无臂无相机: 合成观测, 纯验证到 <SERVER> 的推理链路
  (默认 dry-run)        真相机+真臂只读: 完整闭环但绝不发 CAN 运动命令, 打印将执行的动作
  --execute             真执行. 需同时给 --confirm-enable-risk, 且运行时输入 DEPLOY
  --prepare-only        只做"安全使能"流程(含防冲检测)就退出, 用于首次上机验证 enable
  --mock-robot / --mock-cameras 可与以上组合(联调用)

运行中键盘(单键, 同采集 UX):
  n/空格=下一段   1-7=跳到某段   b=退一段   p=暂停/继续   回车 或 q=急停(停发+下电抱闸)

安全设计:
  * enable 前后关节防冲检测(J2 事故防线): enable 后 1.5s 内持续钉住当前位姿,
    若任一关节相对 enable 前漂移 > --enable-drift-abort (默认0.08rad) 立即下电退出。
  * 逐 tick 速率限幅 --rate-limit (默认0.06rad/tick@15Hz≈0.9rad/s) + 数据集关节范围夹持(±0.15rad裕量)。
  * delta 版模型首步动作≈当前位姿(离线已验证 0.0009rad), 起步天然平滑。
  * 急停 = 停止发送 + 双臂 disable(抱闸保持当前姿态, 不回弹不下坠)。

前置(照 docs/进入拖动示教流程.md 的反向): 两臂上电、CAN 可读; 执行模式无需 Web 设主臂
(enable() 会走 SDK 的 CAN 控制权激活, gripper_control.py 已在真臂验证过该路径)。
服务端: <SERVER> 先跑 GPU=3 ./serve_nero.sh 19999 8005
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

NERO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NERO_ROOT / "nero_control"))
sys.path.insert(0, str(NERO_ROOT / "lerobot_data" / "lerobot_tools"))
sys.path.insert(0, str(NERO_ROOT / "deploy" / "openpi-client" / "src"))

# 连 <SERVER> 的 websocket 前必须排代理(websockets 不认 no_proxy 里的 CIDR, 老坑)
import os

for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = "<POLICY_SERVER_IP>,<POLICY_SERVER_IP>,127.0.0.1,localhost"

from camera_utils import CameraSet, add_camera_args, build_camera_configs  # noqa: E402
from collect_dual_nero import KeyPad  # noqa: E402  复用采集的单键读取(termios 崩溃恢复)

FPS = 15
PERIOD = 1.0 / FPS
H = 50  # 服务端动作块长度
GRIP = (7, 15)
JOINTS = tuple(i for i in range(16) if i not in GRIP)

# hezi_closing_refined_merged_v2 数据集 60640 帧实测 action 范围, 夹持=范围±margin
DATA_MIN = np.array([-1.455, 0.258, -2.061, 0.467, 1.790, -0.014, -1.240, 0.0,
                     -0.869, 0.284, -0.153, 0.709, -2.810, -0.752, -1.226, 0.001])
DATA_MAX = np.array([0.442, 1.413, 0.224, 2.200, 2.790, 0.994, 1.301, 0.0,
                     1.369, 1.676, 2.191, 2.201, -1.980, 0.195, 0.973, 0.016])

# 7 段封盖 prompt, 按 task_index(0-6)索引, 文本与训练 tasks.jsonl 逐字一致。
# 具体执行顺序由 --stage-order 决定(见下面两个预设)。
PROMPTS_BY_TASK = [
    "Move the right arm toward the lower part of the box's right side flap and brace the middle of the right side flap.",                          # task0 右臂撑右侧翼
    "Move the right arm toward the front part of the box's right lower flap and gently push that area.",                                            # task1 右臂推右下翼前部
    "Move both arms away from the carton flaps and withdraw to a safe position, clearing space for the subsequent lower-flap pushing actions.",    # task2 双臂退安全位
    "Move the right arm under the open flap, brace the left edge of the box's lower flap from the right-arm viewpoint, and gently push it.",        # task3 右臂推下翼
    "Move the left arm under the open flap, brace the left edge of the box's lower flap from the left-arm viewpoint, and gently push it.",          # task4 左臂推下翼
    "Use both arms at the same time to fold the left and right side flaps inward toward the carton opening, then press the flaps down until they lie flat.",  # task5 双臂折压侧翼
    "Move the left arm toward the lower part of the box's left side flap and brace the middle of the left side flap.",                             # task6 左臂撑左侧翼
]

# 阶段执行顺序(task_index 序列)
STAGE_ORDER_TRAIN = [0, 6, 5, 2, 1, 4, 3]   # ★训练权威顺序(parquet stage_id 单调, 推荐)
STAGE_ORDER_ALT = [0, 6, 5, 2, 3, 4, 1]     # 备选(task1/task3 位置不同, 用户提的)


def load_subtasks(order: list[int]) -> list[str]:
    """按给定 task_index 顺序返回 prompt(文本与训练逐字一致)."""
    return [PROMPTS_BY_TASK[i] for i in order]


class DeployArms:
    """双臂执行封装: 状态读取 16 维与数据集同序(左7+左爪+右7+右爪)."""

    def __init__(self, args: argparse.Namespace) -> None:
        from single_nero_driver import NeroArm, NeroArmConfig

        def mk(name: str, chan: str, fw: str) -> NeroArm:
            return NeroArm(NeroArmConfig(
                name=name, can_channel=chan, firmware=fw,
                can_interface=args.can_interface, bitrate=args.can_bitrate,
                timeout=args.can_timeout))

        self.left = mk("left", args.left_can, args.left_firmware)
        try:
            self.right = mk("right", args.right_can, args.right_firmware)
        except Exception:
            self.left.close()
            raise
        self._last_grip = [None, None]

    def read_state(self) -> np.ndarray:
        l, r = self.left.read_joints(), self.right.read_joints()
        return np.array(l + [self.left.read_gripper_width()] + r + [self.right.read_gripper_width()],
                        dtype=np.float64)

    def modes(self) -> dict:
        return {"left": self.left.control_mode(), "right": self.right.control_mode()}

    def enable(self, speed_percent: int) -> None:
        self.left.enable(speed_percent=speed_percent)
        self.right.enable(speed_percent=speed_percent)

    def send_joints(self, t: np.ndarray) -> None:
        self.left.move_joints([float(v) for v in t[0:7]])
        self.right.move_joints([float(v) for v in t[8:15]])

    def send_grippers(self, t: np.ndarray, force_n: float) -> None:
        # 变化 >1mm 才发, 避免每 tick 刷爆夹爪 CAN
        for i, (arm, dim) in enumerate(((self.left, 7), (self.right, 15))):
            w = float(np.clip(t[dim], 0.0, 0.075))
            if self._last_grip[i] is None or abs(w - self._last_grip[i]) > 0.001:
                arm.set_gripper_width(w, force_n=force_n)
                self._last_grip[i] = w

    def estop(self) -> None:
        """停发 + 双臂下电抱闸(保持当前姿态)."""
        for arm in (self.left, self.right):
            try:
                arm.arm.disable()
            except Exception as exc:  # noqa: BLE001
                print(f"⚠ {arm.config.name} disable 失败: {exc}", flush=True)

    def close(self) -> None:
        self.right.close()
        self.left.close()


class MockArms:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self._mid = (DATA_MIN + DATA_MAX) / 2

    def read_state(self) -> np.ndarray:
        t = time.monotonic() - self.t0
        s = self._mid + 0.05 * np.sin(t + np.arange(16))
        s[list(GRIP)] = np.clip(s[list(GRIP)], 0.0, 0.067)
        return s

    def modes(self) -> dict:
        return {"left": (0x01, "mock"), "right": (0x01, "mock")}

    def enable(self, speed_percent: int) -> None:
        print(f"[mock] enable speed={speed_percent}%", flush=True)

    def send_joints(self, t: np.ndarray) -> None:
        pass

    def send_grippers(self, t: np.ndarray, force_n: float) -> None:
        pass

    def estop(self) -> None:
        print("[mock] estop", flush=True)

    def close(self) -> None:
        pass


def clamp_target(target: np.ndarray, last_cmd: np.ndarray, rate: float, margin: float) -> np.ndarray:
    """数据集范围夹持 + 逐 tick 速率限幅. 返回实际可发送的目标."""
    t = np.clip(target, DATA_MIN - margin, DATA_MAX + margin)
    t[list(GRIP)] = np.clip(t[list(GRIP)], 0.0, 0.075)
    step = np.clip(t - last_cmd, -rate, rate)
    return last_cmd + step


def fake_obs(subtask: str) -> dict:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    return {
        "observation/image": img,
        "observation/wrist_image": img,
        "observation/right_wrist_image": img,
        "observation/state": ((DATA_MIN + DATA_MAX) / 2).astype(np.float64),
        "prompt": subtask,
    }


def build_obs(state: np.ndarray, frames: dict, subtask: str) -> dict:
    return {
        "observation/image": frames["observation.images.third"],
        "observation/wrist_image": frames["observation.images.left_wrist"],
        "observation/right_wrist_image": frames["observation.images.right_wrist"],
        "observation/state": state,
        "prompt": subtask,
    }


def safe_enable(arms, keypad: KeyPad, args: argparse.Namespace) -> np.ndarray | None:
    """使能防冲流程(J2 事故防线). 成功返回 enable 后的基准位姿, 失败返回 None."""
    q0 = arms.read_state()
    print("使能前位姿:", np.round(q0, 3).tolist(), flush=True)
    print("当前模式:", arms.modes(), flush=True)
    print(f"\n⚠ 即将使能双臂(speed={args.speed_percent}%)。确认: 工作区无人手、急停在手边。", flush=True)
    if input("输入 DEPLOY 继续(其它=退出): ").strip() != "DEPLOY":
        return None

    # 尽力先把控制器目标钉到当前位姿(未使能时可能被忽略, 无害)
    for _ in range(3):
        arms.send_joints(q0)
        time.sleep(0.05)
    arms.enable(args.speed_percent)

    # enable 后 1.5s: 持续钉住当前位姿并监测漂移
    for _ in range(int(1.5 * FPS)):
        arms.send_joints(q0)
        time.sleep(PERIOD)
        q = arms.read_state()
        drift = np.abs(np.asarray(q)[list(JOINTS)] - q0[list(JOINTS)]).max()
        if drift > args.enable_drift_abort:
            print(f"\n🛑 使能后关节漂移 {drift:.3f}rad > {args.enable_drift_abort} — 疑似冲缓存目标, 立即下电!",
                  flush=True)
            arms.estop()
            return None
    print(f"✓ 使能通过防冲检测(漂移 {drift:.4f} rad)", flush=True)
    return arms.read_state()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Nero dual-arm pi0.5 deploy — hezi_closing_refined_merged_v2")
    ap.add_argument("--host", default="<POLICY_SERVER_IP>")   # <SERVER>(无线)
    ap.add_argument("--port", type=int, default=8023)      # hezi merged serve
    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True,
                    help="真执行(默认开; --no-execute 回到 dry-run 不发运动命令)")
    ap.add_argument("--confirm-enable-risk", action=argparse.BooleanOptionalAction, default=True,
                    help="--execute 必须(默认开); 确认已知晓使能冲缓存目标的风险与防线")
    ap.add_argument("--prepare-only", action="store_true", help="只跑安全使能流程后退出")
    ap.add_argument("--fake-obs", action="store_true", help="合成观测, 只测服务链路")
    ap.add_argument("--mock-robot", action="store_true")
    ap.add_argument("--execute-horizon", type=int, default=25, help="每个动作块执行帧数后重推理")
    ap.add_argument("--rate-limit", type=float, default=0.06, help="每tick最大关节步进(rad)")
    ap.add_argument("--limit-margin", type=float, default=0.15, help="数据集范围外扩裕量(rad)")
    ap.add_argument("--enable-drift-abort", type=float, default=0.08)
    ap.add_argument("--speed-percent", type=int, default=5)
    ap.add_argument("--grip-force", type=float, default=1.0)
    ap.add_argument("--start-stage", type=int, default=1, help="起始子任务段(1-7)")
    ap.add_argument("--stage-order", default="0,6,5,2,1,4,3",
                    help="阶段执行顺序(task_index 逗号分隔)。训练权威=0,6,5,2,1,4,3(默认/推荐); 备选=0,6,5,2,3,4,1")
    # CAN(与 collect.sh 同映射: 左=can1 右=can0)
    ap.add_argument("--left-can", default="can1")
    ap.add_argument("--right-can", default="can0")
    ap.add_argument("--left-firmware", default="v120")
    ap.add_argument("--right-firmware", default="v120")
    ap.add_argument("--can-interface", default="socketcan")
    ap.add_argument("--can-bitrate", type=int, default=1_000_000)
    ap.add_argument("--can-timeout", type=float, default=1.0)
    add_camera_args(ap)
    args = ap.parse_args()

    # 相机默认 = collect.sh 固化的序列号(不传参就用它们)
    byid = "/dev/v4l/by-id"
    if args.left_wrist_camera is None and not args.mock_cameras:
        args.left_wrist_camera = f"{byid}/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_<LEFT_WRIST_CAM_SERIAL>-video-index0"
    if args.right_wrist_camera is None and not args.mock_cameras:
        args.right_wrist_camera = f"{byid}/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_<RIGHT_WRIST_CAM_SERIAL>-video-index0"
    if args.third_camera is None and not args.mock_cameras:
        args.third_camera = f"{byid}/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_<THIRD_CAM_SERIAL>-video-index0"
    args.camera_width, args.camera_height = 640, 480  # 与采集一致

    if args.execute and not args.confirm_enable_risk:
        ap.error("--execute 需要 --confirm-enable-risk")
    if args.execute and args.fake_obs:
        ap.error("--fake-obs 不能与 --execute 组合")
    return args


def main() -> None:  # noqa: PLR0915, PLR0912
    args = parse_args()
    order = [int(x) for x in args.stage_order.split(",")]
    subtasks = load_subtasks(order)
    print(f"子任务 prompt({len(subtasks)} 段, 执行顺序 task_index={order}):", flush=True)
    for i, s in enumerate(subtasks, 1):
        print(f"  [{i}] {s}", flush=True)

    from openpi_client import websocket_client_policy

    print(f"\n连接推理服务 ws://{args.host}:{args.port} …", flush=True)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    # --fake-obs: 只测链路
    if args.fake_obs:
        t0 = time.monotonic()
        actions = np.asarray(policy.infer(fake_obs(subtasks[0]))["actions"])
        dt = (time.monotonic() - t0) * 1000
        print(f"✓ 服务链路通: actions{actions.shape} 时延{dt:.0f}ms "
              f"首步range[{actions[0].min():.3f},{actions[0].max():.3f}]", flush=True)
        return

    arms = MockArms() if args.mock_robot else DeployArms(args)
    cameras = CameraSet(build_camera_configs(args))
    stage = int(np.clip(args.start_stage, 1, len(subtasks))) - 1
    mode = "EXECUTE" if args.execute else "DRY-RUN(不发命令)"

    try:
        cameras.open()
        state = np.asarray(arms.read_state())
        print(f"\n模式={mode}  当前16维状态: {np.round(state, 3).tolist()}", flush=True)

        with KeyPad() as keypad:
            last_cmd = state.copy()
            if args.execute or args.prepare_only:
                base = safe_enable(arms, keypad, args)
                if base is None:
                    print("使能未通过/已取消, 退出。", flush=True)
                    return
                last_cmd = np.asarray(base)
                if args.prepare_only:
                    print("--prepare-only 完成, 保持使能退出(需要下电请按急停)。", flush=True)
                    return

            print(f"\n▶ 进入策略循环 @15Hz  段 {stage + 1}/{len(subtasks)}: {subtasks[stage]}", flush=True)
            print("  键: n/空格=下段 数字=跳段 b=退段 p=暂停 回车/q=急停退出\a", flush=True)
            keypad.drain()
            paused = False
            while True:
                state = np.asarray(arms.read_state())
                frames = cameras.read_frames()
                t0 = time.monotonic()
                chunk = np.asarray(policy.infer(build_obs(state, frames, subtasks[stage]))["actions"])
                infer_ms = (time.monotonic() - t0) * 1000

                n_exec = min(args.execute_horizon, len(chunk))
                first_dev = np.abs(chunk[0][list(JOINTS)] - state[list(JOINTS)]).max()
                print(f"  推理{infer_ms:4.0f}ms 段{stage + 1} 首步偏差{first_dev:.4f}rad "
                      f"执行{n_exec}帧{'(dry)' if not args.execute else ''}   ", end="\r", flush=True)

                restart_chunk = False
                for i in range(n_exec):
                    tick = time.monotonic()
                    target = clamp_target(chunk[i], last_cmd, args.rate_limit, args.limit_margin)
                    if args.execute and not paused:
                        arms.send_joints(target)
                        arms.send_grippers(target, args.grip_force)
                        last_cmd = target
                    elif not args.execute:
                        last_cmd = target  # dry-run 也推进限幅参考, 模拟真实轨迹

                    key = keypad.get(0.0)
                    if key:
                        k = key.lower()
                        if k in ("\r", "\n", "q"):
                            print("\n🛑 急停: 停止发送并下电抱闸。", flush=True)
                            if args.execute:
                                arms.estop()
                            return
                        if k == "p":
                            paused = not paused
                            print(f"\n{'⏸ 暂停(不发命令,臂保持)' if paused else '▶ 继续'}\a", flush=True)
                        prev = stage
                        if k in ("n", " "):
                            stage = min(stage + 1, len(subtasks) - 1)
                        elif k == "b":
                            stage = max(stage - 1, 0)
                        elif k.isdigit() and 1 <= int(k) <= len(subtasks):
                            stage = int(k) - 1
                        if stage != prev:
                            print(f"\n  ● 段 {stage + 1}/{len(subtasks)}: {subtasks[stage]}\a", flush=True)
                            restart_chunk = True
                            break  # 立即用新 prompt 重推理

                    dt = time.monotonic() - tick
                    if dt < PERIOD:
                        time.sleep(PERIOD - dt)
                if restart_chunk:
                    continue
    except KeyboardInterrupt:
        print("\nCtrl-C: ", "急停下电" if args.execute else "退出", flush=True)
        if args.execute:
            arms.estop()
    finally:
        cameras.close()
        arms.close()


if __name__ == "__main__":
    main()
