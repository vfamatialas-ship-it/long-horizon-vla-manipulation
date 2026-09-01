#!/usr/bin/env python3
"""Nero pi0.5 **rollout 采集器** 共用实现 —— 抓盒装箱(两子任务: 抓取 → 放置)。

这是"部署 + 边推理边录数据集"合体: 策略在线推理驱动机械臂, 同时把每一帧
(观测/动作/子任务/成败/失败类型)按 LeRobot v2.1 存下来, 供 RLinf(RECAP) 当 rollout 数据用。

与人工遥操作采集(lerobot_data/Rlinf/tools/collect_leftbox_*.py)的区别:
  遥操作: 动作来自人手, action ≡ state(拖动示教)
  rollout: 动作来自**策略输出**, action = 发给机械臂的目标(限幅后), state = 实读反馈
RECAP 假设的 rollout 正是后者 —— 同一个策略有时成功有时失败, 失败在轨迹中途才发生,
所以 V 会在"策略犯错的那一刻"掉下来, 优势才有 credit assignment 的意义。

═══ 双臂角色 ═══
执行臂跑策略; **另一条臂由你在外部(Web UI)设成主臂(leader/零力)模式**, 摆在不同初始位置,
本脚本【全程不碰它】, 只读它的关节角写进 state/action(保持 16 维 schema)。
  leftbox  rollout: 左臂执行, 右臂当主臂
  rightbox rollout: 右臂执行, 左臂当主臂
⚠ 主臂模式下普通反馈是"进入拖动前的旧缓存值"(冻结), 必须读 leader 广播帧。
  启动时会做只读核对(check_leader_broadcast), 广播不通会警告 —— 否则录出的另一条臂
  是个假常数, 文件看着正常实则废数据(本项目踩过的坑)。

═══ 两个子任务与按键 ═══
  子任务1 抓取 → 空格/s = 本段成功, 切到子任务2(换 prompt)
                f = 本段失败 → 选失败类型 → **整条立即结束**(不进入子任务2)
  子任务2 放置 → 空格/s = 本段成功 → 整条结束(成功)
                f = 本段失败 → 选失败类型 → 整条结束
  其它: g=丢弃整条  p=暂停/继续  回车/q=急停退出
失败类型按子任务分别给菜单(抓取: 没对准/位置太低; 放置: 抬得不够高/没放进打包箱; 都含"其他")。

═══ 设备自动检测 ═══
相机和 CAN 全部**按设备内部序列号**解析(照抄 lerobot_data/HY/tools/ 采集脚本的做法),
USB 插拔重枚举后不会错位, 不用手填 /dev/videoN 和 canX。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

NERO_ROOT = Path(__file__).resolve().parents[1]
for _p in (NERO_ROOT / "nero_control",
           NERO_ROOT / "lerobot_data" / "lerobot_tools",
           NERO_ROOT / "lerobot_data" / "lerobot_tools" / "lerobot_v21" / "src",
           NERO_ROOT / "deploy" / "openpi-client" / "src",
           Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = "<POLICY_SERVER_IP>,<POLICY_SERVER_IP>,127.0.0.1,localhost"

from openpi_client import image_tools  # noqa: E402  (与服务端 ResizeImages 同一函数)
from camera_utils import CameraSet, add_camera_args, build_camera_configs  # noqa: E402
from collect_dual_nero import KeyPad  # noqa: E402

FPS = 15
PERIOD = 1.0 / FPS
GRIP_DIMS = (7, 15)                                   # 16 维里两个夹爪宽度所在维
JOINT_DIMS = tuple(i for i in range(16) if i not in GRIP_DIMS)
STATE_NAMES = [f"left_j{i}" for i in range(1, 8)] + ["left_gripper_width"] \
    + [f"right_j{i}" for i in range(1, 8)] + ["right_gripper_width"]
_IMG_FEAT = {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]}

# ── 设备序列号(固定不变, 插拔不影响) ───────────────────────────────────────
CAN_SERIALS = {"left": "<LEFT_CAN_SERIAL>", "right": "<RIGHT_CAN_SERIAL>"}
CAM_SERIALS = {"third_view": "<THIRD_CAM_SERIAL>", "left_wrist": "<LEFT_WRIST_CAM_SERIAL>", "right_wrist": "<RIGHT_WRIST_CAM_SERIAL>"}
# 夹爪宽度上限(分侧, 实测工作范围): 右爪到 ~0.105, 左爪到 ~0.10
GRIP_MAX = {"left": 0.100, "right": 0.105}
# 事件码(与人工采集 collect_leftbox_common.py 的取值完全一致, 保证两批数据可合并)
EVENT = {"NONE": 0, "EPISODE_STARTED": 1, "SUBTASK_STARTED": 2, "SUBTASK_DONE": 3}


# ══════════════════════ 设备按序列号解析 ══════════════════════
def _udev_serial_of_syspath(dev_path: str) -> str | None:
    """读 sysfs 路径(如 /sys/class/net/can0)对应设备的 ID_SERIAL_SHORT。"""
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-p", os.path.realpath(dev_path)],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if line.startswith("ID_SERIAL_SHORT="):
            return line.split("=", 1)[1].strip()
    return None


def resolve_can_by_serial(serial: str) -> str | None:
    """按 CAN 适配器序列号找当前 canX。找不到返回 None(调用方决定报错还是兜底)。"""
    for path in sorted(glob.glob("/sys/class/net/can*")):
        if _udev_serial_of_syspath(path) == serial:
            return os.path.basename(path)
    return None


def resolve_video_by_serial(serial: str) -> str | None:
    """按相机 USB 序列号找当前 /dev/videoN。

    同一台相机会占多个 videoN 节点(视频流/元数据), 按编号从小到大取第一个匹配的 ——
    与 lerobot_data/HY/tools/collect_*.py 的做法一致(实测取到的就是可取流的那个)。
    """
    def _num(p: str) -> int:
        s = p.removeprefix("/dev/video")
        return int(s) if s.isdigit() else 10_000

    for path in sorted(glob.glob("/dev/video*"), key=_num):
        try:
            out = subprocess.check_output(
                ["udevadm", "info", "-q", "property", "-n", path],
                text=True, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            continue
        for line in out.splitlines():
            if line.startswith("ID_SERIAL_SHORT=") and line.split("=", 1)[1].strip() == serial:
                return path
    return None


def autodetect_devices(args: argparse.Namespace) -> None:
    """把 CAN 与三路相机按序列号解析进 args(命令行显式指定的不覆盖)。"""
    print("── 设备自动检测(按内部序列号) ──", flush=True)
    for side in ("left", "right"):
        attr = f"{side}_can"
        if getattr(args, attr) in (None, "auto"):
            got = resolve_can_by_serial(CAN_SERIALS[side])
            if got is None:
                raise RuntimeError(
                    f"{side} 臂 CAN 适配器未找到(serial={CAN_SERIALS[side]})。"
                    f"\n  现有 CAN: {[os.path.basename(p) for p in glob.glob('/sys/class/net/can*')]}"
                    f"\n  检查臂是否上电/USB-CAN 是否插好; 也可 --{side}-can canX 手动指定。")
            setattr(args, attr, got)
        print(f"  {side:>5} 臂 CAN → {getattr(args, attr)}", flush=True)

    for cam, attr in (("third_view", "third_camera"), ("left_wrist", "left_wrist_camera"),
                      ("right_wrist", "right_wrist_camera")):
        if getattr(args, attr, None) is None and not args.mock_cameras:
            got = resolve_video_by_serial(CAM_SERIALS[cam])
            if got is None:
                raise RuntimeError(
                    f"相机 {cam} 未找到(serial={CAM_SERIALS[cam]})。"
                    f"\n  检查 USB 是否插好; 也可 --{attr.replace('_', '-')} /dev/videoN 手动指定。")
            setattr(args, attr, got)
        if getattr(args, attr, None):
            print(f"  相机 {cam:>11} → {getattr(args, attr)}", flush=True)


# ══════════════════════ 任务定义 ══════════════════════
@dataclass(frozen=True)
class Subtask:
    subtask_id: int          # 1=抓取 2=放置(与人工采集的 prompt_index 对齐)
    name: str
    zh: str
    en: str                  # 送给策略的 prompt
    failures: tuple          # ((code, 中文说明), ...) 本子任务的失败菜单


@dataclass(frozen=True)
class TaskSpec:
    exec_side: str           # "left" | "right" —— 跑策略的臂
    robot_type: str
    port: int
    subtasks: tuple
    data_min: np.ndarray     # 16 维训练集实测范围(clamp 安全限位)
    data_max: np.ndarray

    @property
    def leader_side(self) -> str:
        return "right" if self.exec_side == "left" else "left"

    @property
    def exec_slice(self) -> slice:
        """执行臂在 16 维里的区间(7关节+爪)。"""
        return slice(0, 8) if self.exec_side == "left" else slice(8, 16)

    @property
    def leader_slice(self) -> slice:
        return slice(8, 16) if self.exec_side == "left" else slice(0, 8)

    @property
    def exec_joint_dims(self) -> list:
        return list(range(0, 7)) if self.exec_side == "left" else list(range(8, 15))

    @property
    def exec_grip_dim(self) -> int:
        return 7 if self.exec_side == "left" else 15


_GRASP_FAILURES = (
    ("gripper_misaligned", "夹爪没对准盒子"),
    ("gripper_too_low", "夹爪位置太低"),
    ("other", "其他"),
)
_PLACE_FAILURES = (
    ("lift_not_high_enough", "夹爪抬起得不够高"),
    ("box_not_in_carton", "盒子没被放进打包箱"),
    ("other", "其他"),
)


def make_task(exec_side: str, port: int, data_min, data_max) -> TaskSpec:
    zh_side = "左臂" if exec_side == "left" else "右臂"
    en_side = "left" if exec_side == "left" else "right"
    area = "left-side" if exec_side == "left" else "right-side"
    return TaskSpec(
        exec_side=exec_side,
        robot_type="dual_nero_left_box_pick",   # 与人工采集数据集同 robot_type, 便于合并
        port=port,
        subtasks=(
            Subtask(1, "grasp", f"{zh_side}抓取{'左' if exec_side == 'left' else '右'}侧区域的盒子",
                    f"Use the {en_side} arm to grasp a box from the {area} area.", _GRASP_FAILURES),
            Subtask(2, "place", f"{zh_side}将抓到的盒子放进打包盒的空处",
                    f"Use the {en_side} arm to place the grasped box into an empty spot "
                    f"of the middle packing carton.", _PLACE_FAILURES),
        ),
        data_min=np.asarray(data_min, dtype=np.float64),
        data_max=np.asarray(data_max, dtype=np.float64),
    )


# ══════════════════════ 机械臂封装 ══════════════════════
class RolloutArms:
    """双臂封装 —— 执行臂跑策略, 另一臂由你外部设成主臂(leader), 本类**只读不碰**。

    按 exec_side 参数化(不是左右各抄一份): leader 相关逻辑左右完全对称, 抄两份反而容易
    改一边忘一边。所有"哪条臂"的判断都集中在 self.ex / self.ld 两个引用上。
    """

    def __init__(self, args: argparse.Namespace, task: TaskSpec) -> None:
        from single_nero_driver import NeroArm, NeroArmConfig

        self.task = task

        def mk(name, chan, fw):
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
        self.ex = self.left if task.exec_side == "left" else self.right     # 执行臂
        self.ld = self.right if task.exec_side == "left" else self.left     # 主臂(只读)
        self.ex_gmax = GRIP_MAX[task.exec_side]
        self._last_grip = None
        self._last_leader = [0.0] * 7

    def setup_arm_modes(self) -> None:
        print(f"{self.task.leader_side}臂: 主臂(leader)模式由你外部设置, 本脚本不碰它, "
              f"只控{self.task.exec_side}臂、读双臂关节。", flush=True)

    def _read_side(self, arm, is_leader: bool):
        """返回 (7关节, 爪宽)。主臂读 leader 广播帧(新鲜手摆位), 普通反馈是冻结旧值。"""
        if is_leader:
            j = arm.read_leader_joints()
            if j is None:
                try:
                    j = arm.read_joints()
                except Exception:  # noqa: BLE001
                    j = list(self._last_leader)
            self._last_leader = list(j)
        else:
            j = arm.read_joints()
        try:
            g = arm.read_gripper_width()
        except Exception:  # noqa: BLE001
            g = 0.0
        return list(j), float(g)

    def read_state(self) -> np.ndarray:
        lj, lg = self._read_side(self.left, self.task.leader_side == "left")
        rj, rg = self._read_side(self.right, self.task.leader_side == "right")
        return np.array(lj + [lg] + rj + [rg], dtype=np.float64)

    def refresh_feedback(self) -> None:
        """确保【执行臂】can_push 开着。不碰主臂 —— set_can_control_mode 会破坏 leader 状态。"""
        try:
            self.ex.set_can_control_mode(enable_can_push=True)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ {self.ex.config.name} refresh_feedback 失败: {exc}", flush=True)

    def modes(self) -> dict:
        return {"left": self.left.control_mode(), "right": self.right.control_mode()}

    def enable(self, speed_percent: int, hold_joints=None) -> None:
        """只使能执行臂。hold_joints(16维当前实读): 使能前把目标寄存器写成当前位,
        防止伺服弹向 SDK 缓存的上一条 move_j 终点(病因: "使能后先运动一段")。"""
        h = None if hold_joints is None else np.asarray(hold_joints, dtype=float)
        eh = None if h is None else [float(v) for v in h[self.task.exec_joint_dims]]
        self.ex.enable(speed_percent=speed_percent, hold_joints=eh)
        self._reenable_gripper()

    def _reenable_gripper(self) -> None:
        """复位失能时夹爪也断电 → 使能后发一次当前宽度让它复能, 并清缓存使下一帧必发。"""
        try:
            w = float(np.clip(self.ex.read_gripper_width(), 0.0, self.ex_gmax))
            self.ex.set_gripper_width(w, force_n=1.0)
        except Exception:  # noqa: BLE001
            pass
        self._last_grip = None

    def send_joints(self, t: np.ndarray) -> None:
        self.ex.move_joints([float(v) for v in np.asarray(t)[self.task.exec_joint_dims]])

    def send_grippers(self, t: np.ndarray, force_n: float, squeeze: float = 0.0) -> None:
        """变化 >1mm 才发, 避免每 tick 刷爆夹爪 CAN。主臂夹爪不发。

        ★ squeeze(夹持偏置, 米): 实际命令宽度 = 策略目标 − squeeze。
          夹爪是**宽度模式**: 走到目标宽度就停, 不再用力。示教数据里的宽度往往只是
          "刚贴上物体", 于是 rollout 时物体会滑落。命令得再窄一点, 夹爪才会持续挤压,
          挤压力由 force_n 封顶 —— 两者缺一不可(只加力: 位置到了不使劲;
          只加偏置: 力上限 1N 挤不动)。
          对"全开"影响可忽略(0.100 → 0.097), 所以不区分开合, 统一减。
        """
        w_cmd = float(np.clip(np.asarray(t)[self.task.exec_grip_dim], 0.0, self.ex_gmax))
        w = float(np.clip(w_cmd - squeeze, 0.0, self.ex_gmax))
        if self._last_grip is None or abs(w - self._last_grip) > 0.001:
            self.ex.set_gripper_width(w, force_n=force_n)
            self._last_grip = w

    def grip_feedback(self) -> str:
        """执行臂夹爪实测宽度/受力, 用于调参时看夹得紧不紧(只在打状态行时调, 不每 tick 读)。"""
        try:
            st = self.ex.read_gripper_status()
            if st:
                return f"爪{st.get('pos_mm', 0):.0f}mm/{st.get('force_N', 0):.1f}N"
        except Exception:  # noqa: BLE001
            pass
        return ""

    def estop(self) -> None:
        """急停下电 —— 只下电执行臂; 主臂在你设的零力模式本就安全, 不碰。"""
        try:
            self.ex.arm.disable()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ {self.ex.config.name} disable 失败: {exc}", flush=True)

    def disable_for_reset(self) -> None:
        """复位: 失能执行臂的关节电机+夹爪电机(motors off) —— 松垂可手搬;
        can_push 不关 → 反馈保持新鲜, 手摆后能读到真实位。失能后无重力补偿会松垂, 手托住。"""
        try:
            self.ex.disable()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ {self.ex.config.name} disable 失败: {exc}", flush=True)

    def close(self) -> None:
        self.right.close()
        self.left.close()


class MockArms:
    """无臂干跑: 造 16 维正弦状态, 验证键位/录制/标注链路。"""

    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.t0 = time.monotonic()
        self._mid = (task.data_min + task.data_max) / 2

    def setup_arm_modes(self) -> None:
        print("[mock] 无真臂", flush=True)

    def read_state(self) -> np.ndarray:
        s = self._mid + 0.05 * np.sin(time.monotonic() - self.t0 + np.arange(16))
        for d in GRIP_DIMS:
            s[d] = float(np.clip(s[d], 0.0, 0.1))
        return s

    def refresh_feedback(self) -> None:
        pass

    def modes(self) -> dict:
        return {"left": (0x01, "mock"), "right": (0x01, "mock")}

    def enable(self, speed_percent: int, hold_joints=None) -> None:
        print(f"[mock] enable speed={speed_percent}%", flush=True)

    def send_joints(self, t) -> None:
        pass

    def send_grippers(self, t, force_n, squeeze=0.0) -> None:
        pass

    def grip_feedback(self) -> str:
        return ""

    def estop(self) -> None:
        print("[mock] estop", flush=True)

    def disable_for_reset(self) -> None:
        print("[mock] disable_for_reset", flush=True)

    def close(self) -> None:
        pass


# ══════════════════════ 观测 / 限幅 ══════════════════════
_ZERO_IMG = np.zeros((480, 640, 3), dtype=np.uint8)


_RS = 224   # 服务端 ResizeImages 的目标尺寸


def build_obs(state: np.ndarray, frames: dict, prompt: str, exec_side: str,
              preresize: bool = True) -> dict:
    """三路图像槽 + 16 维 state + prompt。

    执行臂的腕相机进 observation/wrist_image; 第三个槽零填(服务端 mask 掉) ——
    与 run_pi05_deploy_left_box.py / run_pi05_deploy_right_box.py 完全一致:
      leftbox  配置 LeRobotNeroDataConfig(use_right_wrist=False): 右腕槽零填+mask
      rightbox 配置 LeRobotNeroRightArm16DataConfig: 只用 third + 右腕(多送的键会被 repack 忽略)

    ★ preresize(默认开): 客户端先用 **服务端同一个函数** image_tools.resize_with_pad
      把图缩到 224x224 再发。服务端再缩一次是恒等操作 → 喂给模型的图**逐比特相同**,
      但 obs 负载 2.76MB → 0.45MB。无线网下实测推理时延 ~1000ms → ~165ms。
      时延直接决定"卡顿": 每 execute_horizon 帧要停下来等一次推理。
      注意: 录进数据集的仍是 480x640 原图(record_frame 用的是 frames, 不是这里的缩放图)。
    """
    wrist = frames["observation.images.left_wrist"] if exec_side == "left" \
        else frames["observation.images.right_wrist"]

    def rs(img):
        return image_tools.resize_with_pad(np.asarray(img), _RS, _RS) if preresize else img

    return {
        "observation/image": rs(frames["observation.images.third"]),
        "observation/wrist_image": rs(wrist),
        "observation/right_wrist_image": rs(_ZERO_IMG),
        "observation/state": state,
        "prompt": prompt,
    }


def clamp_target(target: np.ndarray, last_cmd: np.ndarray, task: TaskSpec,
                 rate: float, margin: float) -> np.ndarray:
    """训练范围钳位 + 逐 tick 变化率限幅(防策略给出跳变目标把臂甩出去)。"""
    t = np.clip(np.asarray(target, dtype=np.float64), task.data_min - margin, task.data_max + margin)
    for d in GRIP_DIMS:
        t[d] = np.clip(t[d], 0.0, 0.105)
    step = np.clip(t - last_cmd, -rate, rate)
    return last_cmd + step


# ══════════════════════ 数据集 ══════════════════════
def create_dataset(args, task: TaskSpec):
    """LeRobot v2.1 数据集。features 与人工采集(collect_leftbox_common.py)**逐字段一致**,
    这样 rollout 数据能和遥操作数据合并, 直接喂 RLinf 的切分/打标脚本。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    i64 = {"dtype": "int64", "shape": (1,), "names": None}
    s1 = {"dtype": "string", "shape": (1,), "names": None}
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
        "observation.images.third_view": dict(_IMG_FEAT),
        "observation.images.left_wrist": dict(_IMG_FEAT),
        "observation.images.right_wrist": dict(_IMG_FEAT),
        "subtask_instance_id": dict(i64),
        "active_arm": dict(s1),
        "prompt_index": dict(i64),
        "prompt_text": dict(s1),
        "prompt_text_zh": dict(s1),
        "subtask_start": dict(i64),
        "subtask_end": dict(i64),
        "event_code": dict(i64),
        "event_name": dict(s1),
        "episode_success": dict(i64),
        "failure_type": dict(s1),
        "subtask_success": dict(i64),
    }
    meta_dir = args.root / "meta"
    if args.root.exists() and (meta_dir / "info.json").exists():
        # ⚠ 只有 info.json、缺 tasks/episodes/episodes_stats = **建好但一条没存过**。
        #   这种半成品不能走续采: load_metadata() 读不到 tasks.jsonl 会抛 FileNotFoundError,
        #   而 LeRobot 的 except 分支是"去 HuggingFace 拉 meta" —— 本地数据集没有远端,
        #   于是卡 20 秒后 ConnectError: No route to host。
        need = ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl")
        missing = [f for f in need if not (meta_dir / f).exists()]
        if missing:
            # ★ 删之前三重确认真的一条数据都没有 —— 若 info.json 写着 >0 条而 meta 只是损坏,
            #   那是有真数据的破损数据集, 删掉找不回来。
            import json as _json, shutil
            try:
                declared = _json.load(open(meta_dir / "info.json", encoding="utf-8")
                                      ).get("total_episodes", -1)
            except Exception:  # noqa: BLE001
                declared = -1
            n_pq = len(glob.glob(str(args.root / "data" / "**" / "*.parquet"), recursive=True))
            n_mp4 = len(glob.glob(str(args.root / "videos" / "**" / "*.mp4"), recursive=True))
            if declared == 0 and n_pq == 0 and n_mp4 == 0:
                print(f"(落点是个空数据集: meta 缺 {', '.join(missing)} —— "
                      f"建好但一条没存过, 清掉重建)", flush=True)
                shutil.rmtree(args.root)
            else:
                raise RuntimeError(
                    f"落点的 meta 不完整, 但里面有数据, 不敢自动清理:\n"
                    f"  {args.root}\n"
                    f"  info.json 声明 {declared} 条; 实有 parquet {n_pq} 个 / mp4 {n_mp4} 个\n"
                    f"  meta 缺: {', '.join(missing)}\n"
                    f"  → 这是个破损数据集。先备份, 再手动决定修复还是丢弃; 或换个 --root 另存。"
                )
        else:
            LeRobotDatasetMetadata(args.repo_id, root=args.root)
            ds = LeRobotDataset(args.repo_id, root=args.root)
            print(f"▶ 续采: 已有 {ds.meta.total_episodes} 条, 接着编号。", flush=True)
            return ds
    if args.root.exists():
        # 目录在但没有 meta/info.json —— LeRobotDataset.create() 内部 mkdir(exist_ok=False)
        # 撞上已存在目录会 FileExistsError。预先 mkdir 好落点(很自然的操作)正好掉进这缺口。
        # rmdir 对非空目录会抛错 → 天然不会误删数据, 不用 rmtree。
        try:
            args.root.rmdir()
            print(f"(落点是个空目录, 已让位给 LeRobot 新建: {args.root})", flush=True)
        except OSError:
            raise RuntimeError(
                f"落点已存在且非空, 但没有 meta/info.json, 不能续采也不能新建:\n"
                f"  {args.root}\n"
                f"  里面有: {sorted(q.name for q in args.root.iterdir())[:8]}\n"
                f"  → 换个 --root, 或先把这个目录挪走/清空。"
            ) from None
    args.root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features, root=args.root,
                                 robot_type=task.robot_type, use_videos=True,
                                 image_writer_threads=args.image_writer_threads)


def record_frame(ds, task: TaskSpec, sub: Subtask, state, action, frames,
                 frame_idx: int, is_first_of_episode: bool, is_first_of_subtask: bool) -> None:
    """写一帧。成败/失败类型是占位, 段结束时由 label_segment 回填, 整条结束时回填 episode_success。"""
    if is_first_of_episode:
        code, name = EVENT["EPISODE_STARTED"], "EPISODE_STARTED"
    elif is_first_of_subtask:
        code, name = EVENT["SUBTASK_STARTED"], "SUBTASK_STARTED"
    else:
        code, name = EVENT["NONE"], "NONE"
    ds.add_frame({
        "observation.state": np.asarray(state, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "observation.images.third_view": frames["observation.images.third"],
        "observation.images.left_wrist": frames["observation.images.left_wrist"],
        "observation.images.right_wrist": frames["observation.images.right_wrist"],
        "subtask_instance_id": np.array([sub.subtask_id], dtype=np.int64),
        "active_arm": task.exec_side.upper(),
        "prompt_index": np.array([sub.subtask_id], dtype=np.int64),
        "prompt_text": sub.en,
        "prompt_text_zh": sub.zh,
        "subtask_start": np.array([1 if is_first_of_subtask else 0], dtype=np.int64),
        "subtask_end": np.array([0], dtype=np.int64),      # 段末由 label_segment 置 1
        "event_code": np.array([code], dtype=np.int64),
        "event_name": name,
        "episode_success": np.array([-1], dtype=np.int64),  # 占位
        "failure_type": "pending",                          # 占位
        "subtask_success": np.array([-1], dtype=np.int64),  # 占位
    }, task=sub.en, timestamp=frame_idx / FPS)


def label_segment(ds, start: int, end: int, success: bool, failure_type: str) -> None:
    """把 [start,end) 这一段(某个子任务)的逐帧标签回填进 episode_buffer。"""
    buf = ds.episode_buffer
    for j in range(start, end):
        buf["subtask_success"][j] = np.array([1 if success else 0], dtype=np.int64)
        buf["failure_type"][j] = failure_type
        buf["event_code"][j] = np.array([EVENT["NONE"]], dtype=np.int64)
        buf["event_name"][j] = "NONE"
    if end > start:
        buf["subtask_end"][end - 1] = np.array([1], dtype=np.int64)
        buf["event_code"][end - 1] = np.array([EVENT["SUBTASK_DONE"]], dtype=np.int64)
        buf["event_name"][end - 1] = "SUBTASK_DONE"
        buf["subtask_start"][start] = np.array([1], dtype=np.int64)
        if start == 0:
            buf["event_code"][0] = np.array([EVENT["EPISODE_STARTED"]], dtype=np.int64)
            buf["event_name"][0] = "EPISODE_STARTED"
        else:
            buf["event_code"][start] = np.array([EVENT["SUBTASK_STARTED"]], dtype=np.int64)
            buf["event_name"][start] = "SUBTASK_STARTED"


def choose_failure_type(sub: Subtask, keypad=None) -> str:
    """按当前子任务给对应的失败菜单(抓取/放置的失败模式不同)。
    序号选预设, 也可直接敲中英文当自定义原因记录。

    ⚠ 必须先把终端从 cbreak 恢复成行模式 —— KeyPad 关了回显, 否则你打的字看不见、
    也退不了格。读完再切回 cbreak, 不影响后续单键操作。
    """
    print(f"\n  【{sub.zh}】失败类型:", flush=True)
    for i, (code, zh) in enumerate(sub.failures, 1):
        print(f"    {i}. {code:<18} {zh}", flush=True)
    print(f"  可输入: 序号(1~{len(sub.failures)}) / 任意文字作自定义原因(中英文均可) / 回车=other",
          flush=True)

    if keypad is not None:
        keypad.restore()            # 回到行模式 → 输入可见、可退格
    try:
        v = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        v = ""
    finally:
        if keypad is not None:
            keypad.__enter__()      # 切回 cbreak(restore 已置空 _old, 可重入)
            keypad.drain()

    if not v:
        print("  → 记为 other", flush=True)
        return "other"
    if v.isdigit() and 1 <= int(v) <= len(sub.failures):
        code = sub.failures[int(v) - 1][0]
        print(f"  → {code}", flush=True)
        return code
    custom = " ".join(v.split())[:120]      # 压掉换行/多余空白, 限长防污染字段
    print(f"  → 自定义原因已记录: {custom}", flush=True)
    return custom


def save_episode(ds, args, task: TaskSpec, records: list) -> int:
    """records: [(subtask_id, start, end, success, ftype)]。episode_success = 所有段都成功。"""
    size = int(ds.episode_buffer["size"])
    all_ok = bool(records) and all(r[3] for r in records) and len(records) == len(task.subtasks)
    ds.episode_buffer["episode_success"] = [np.array([1 if all_ok else 0], np.int64)
                                            for _ in range(size)]
    ep_index = ds.meta.total_episodes
    ds.save_episode()
    ann = args.root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    with open(ann / "episode_outcomes.jsonl", "a") as f:
        f.write(json.dumps({
            "episode_index": ep_index, "frames": size, "episode_success": all_ok,
            "exec_arm": task.exec_side, "source": "rollout",
            "subtasks": [{"subtask_id": r[0], "start": r[1], "end": r[2],
                          "success": r[3], "failure_type": r[4]} for r in records],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False) + "\n")
    if all_ok:
        tag = "全部成功"
    else:
        bad = [f"子任务{r[0]}/{r[4]}" for r in records if not r[3]]
        tag = "失败(" + ",".join(bad) + ")" if bad else "未完成"
    print(f"\n✓ 已存 episode {ep_index:06d}  帧{size}  段{len(records)}  {tag}", flush=True)
    return ep_index


# ══════════════════════ 安全 ══════════════════════
def check_leader_broadcast(arms, task: TaskSpec, args, samples: int = 20) -> bool:
    """**只读**核对主臂真的在 leader 模式广播关节角 —— 不设置、不切换任何模式。

    为什么要查: read_leader_joints() 拿不到广播帧时返回 None → read_state 静默回退到普通反馈,
    而 leader 模式下普通反馈是**进入拖动前的旧缓存值**(冻结)。于是录出来的主臂是个假常数,
    文件齐全、时间戳在走、看着正常, 实则废数据。
    """
    if args.mock_robot:
        return True
    got, last = 0, None
    for _ in range(samples):
        try:
            j = arms.ld.read_leader_joints()
        except Exception:  # noqa: BLE001
            j = None
        if j is not None:
            got += 1
            last = j
        time.sleep(0.05)
    side = task.leader_side
    if got == 0:
        print(f"\n🛑 {side}臂【没有】leader 广播帧({samples} 次全空)。", flush=True)
        print(f"   说明主臂(leader)模式没生效 —— 此时读到的是进入拖动前的**冻结旧值**,", flush=True)
        print(f"   录下来的{side}臂 state/action 会是假常数(废数据)。", flush=True)
        print(f"   请先在 Web UI 把{side}臂设成主臂(零力)模式再来。", flush=True)
        return input("   仍要继续录制吗? 输入 YES 继续(其它=退出): ").strip() == "YES"
    print(f"✓ {side}臂 leader 广播正常({got}/{samples} 帧), 当前手摆位 "
          f"{np.round(np.asarray(last), 3).tolist()}", flush=True)
    if got < samples * 0.5:
        print(f"  ⚠ 广播命中率偏低({got}/{samples}), 录制中可能间歇回退到冻结值 —— 留意。",
              flush=True)
    return True


def safe_enable(arms, task: TaskSpec, args):
    """使能防冲: 以当前实读位为伺服目标使能, 1.5 秒内监控漂移, 超阈值立即下电。"""
    check = task.exec_joint_dims
    arms.refresh_feedback()
    time.sleep(0.3)
    q0 = np.asarray(arms.read_state())        # 失能态反馈新鲜 → 你手摆的真实位
    print("\n使能前位姿(新鲜):", np.round(q0, 3).tolist(), flush=True)
    print(f"⚠ 使能{task.exec_side}臂(speed={args.speed_percent}%)。"
          f"确认工作区无人手、急停在手边。", flush=True)
    arms.enable(args.speed_percent, hold_joints=q0)
    drift = 0.0
    for _ in range(int(1.5 * FPS)):
        arms.send_joints(q0)
        time.sleep(PERIOD)
        q = np.asarray(arms.read_state())
        drift = float(np.abs(q[check] - q0[check]).max())
        if drift > args.enable_drift_abort:
            print(f"\n🛑 使能后漂移 {drift:.3f}rad > {args.enable_drift_abort} — 立即下电!",
                  flush=True)
            arms.estop()
            return None
    print(f"✓ 使能通过防冲检测(漂移 {drift:.4f} rad)", flush=True)
    return np.asarray(arms.read_state())


# ══════════════════════ 参数 ══════════════════════
def build_parser(exec_side: str, default_port: int, default_repo: str) -> argparse.ArgumentParser:
    zh = "左臂" if exec_side == "left" else "右臂"
    other = "右臂" if exec_side == "left" else "左臂"
    ap = argparse.ArgumentParser(
        description=f"Nero pi0.5 rollout 采集器 —— {zh}抓盒装箱(两子任务), {other}主臂只读")
    ap.add_argument("--host", default="<POLICY_SERVER_IP>")
    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--root", default=None, help="数据集目录(默认按时间戳建在 Rlinf/data 下)")
    ap.add_argument("--repo-id", default=default_repo)
    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True,
                    help="真执行(默认开; --no-execute 回到 dry-run 只推理只录不发运动)")
    ap.add_argument("--mock-robot", action="store_true", help="无臂干跑, 验证键位/录制链路")
    ap.add_argument("--execute-horizon", type=int, default=25,
                    help="每次推理执行多少帧后带新观测重推理(动作块 50 帧)")
    ap.add_argument("--rate-limit", type=float, default=0.06,
                    help="每 tick 关节指令最大变化(rad)。v2 数据逐帧变化 p99≈0.058, 0.06 恰好覆盖")
    ap.add_argument("--limit-margin", type=float, default=0.15, help="训练范围外放宽多少(rad)")
    ap.add_argument("--enable-drift-abort", type=float, default=1.5,
                    help="使能防冲阈值(rad)。退拖动瞬间会抖, 设小了会误触发")
    # 机械臂内部伺服速度上限(arm.set_speed_percent)。驱动默认 40, 部署脚本用 20。
    # 旧 rollout 脚本用 5 过于保守 —— 臂跟不上 rate-limit 给出的目标, 表现为"动得很慢"。
    # 真正的安全边界是 --rate-limit(目标每 tick 只前进 0.06rad), speed 只决定臂能否跟上目标。
    ap.add_argument("--speed-percent", type=int, default=20,
                    help="伺服速度上限%%(默认20; 嫌慢可到30~40, 嫌快降回5~10)")
    # 夹爪力上限(N)。底层 int16 毫牛编码, 硬上限约 32.7N。全项目此前一直用默认 1.0,
    # 偏小 —— 物体易滑落。3.0 是温和起点, 夹不住往上加, 夹变形往下减。
    ap.add_argument("--grip-force", type=float, default=3.0,
                    help="夹爪力上限N(默认3.0; 滑落就加到5~8, 夹坏就降回1~2)")
    # 夹持偏置(米): 实际命令宽度 = 策略目标 − 此值, 让夹爪主动挤压而不只是贴着。
    ap.add_argument("--grip-squeeze", type=float, default=0.003,
                    help="夹持偏置m(默认0.003=3mm; 还滑就到0.005~0.008, 夹变形就调到0)")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--disable-at-end", action="store_true",
                    help="每条结束后失能(默认不失能: 该硬件 disable 抱闸不保持, 臂会砸下来)")
    ap.add_argument("--no-resync-anchor", dest="resync_anchor", action="store_false",
                    help="关闭'每块重新锚定到实际位置'(默认开)。关掉会让指令与实际脱节跨块累积, "
                         "臂跟不上时出现'原地前进撤回跳动'")
    ap.add_argument("--no-preresize", action="store_true",
                    help="关闭客户端预缩放(改回送 480x640 原图)。默认开启, 关掉会让推理慢很多")
    ap.add_argument("--left-can", default="auto", help="auto=按序列号自动检测; 或 canX")
    ap.add_argument("--right-can", default="auto", help="auto=按序列号自动检测; 或 canX")
    ap.add_argument("--left-firmware", default="v120")
    ap.add_argument("--right-firmware", default="v120")
    ap.add_argument("--can-interface", default="socketcan")
    ap.add_argument("--can-bitrate", type=int, default=1_000_000)
    ap.add_argument("--can-timeout", type=float, default=1.0)
    add_camera_args(ap)
    return ap


def finalize_args(args, exec_side: str, default_repo: str):
    if args.root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.root = NERO_ROOT / "lerobot_data" / "Rlinf" / "data" / f"nero_{exec_side}box_rollout_{stamp}"
    args.root = Path(args.root)
    args.camera_width, args.camera_height = 640, 480
    autodetect_devices(args)
    return args


# ══════════════════════ 主循环 ══════════════════════
def run(task: TaskSpec, args) -> None:  # noqa: PLR0915, PLR0912
    mode = "EXECUTE(真执行)" if args.execute else "DRY-RUN(只推理只录, 不发运动)"
    print(f"\n任务: {task.exec_side}臂执行 / {task.leader_side}臂主臂只读"
          f"  子任务 {len(task.subtasks)} 个  模式={mode}", flush=True)
    print(f"数据集: {args.root}", flush=True)
    print(f"夹爪: 力上限 {args.grip_force}N  夹持偏置 {args.grip_squeeze*1000:.0f}mm"
          f"(实际命令宽度 = 策略目标 − 偏置)", flush=True)

    from openpi_client import websocket_client_policy
    print(f"连接推理服务 ws://{args.host}:{args.port} …", flush=True)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    ds = create_dataset(args, task)
    arms = MockArms(task) if args.mock_robot else RolloutArms(args, task)
    if args.execute:
        arms.setup_arm_modes()
    cameras = CameraSet(build_camera_configs(args))

    n_saved = 0
    try:
        cameras.open()
        if not check_leader_broadcast(arms, task, args):
            print("已退出(未录制)。", flush=True)
            return

        with KeyPad() as keypad:
            print(f"\n就绪。空格=开始一条 rollout   回车/q=退出", flush=True)
            keypad.drain()
            while True:                                        # ── 会话循环(IDLE) ──
                k = (keypad.get(0.2) or "").lower()
                if k in ("\r", "\n", "q"):
                    break
                if k != " ":
                    continue

                if args.execute:
                    base = safe_enable(arms, task, args)
                    if base is None:
                        continue
                    last_cmd = base
                else:
                    last_cmd = np.asarray(arms.read_state())

                print(f"\n▶ RECORDING 第 {ds.meta.total_episodes:06d} 条", flush=True)
                keypad.drain()
                records = []            # [(subtask_id, start, end, success, ftype)]
                frame_idx = 0
                paused = False
                aborted = False
                failed_early = False
                state = np.asarray(arms.read_state())
                frames = cameras.read_frames()

                for sub in task.subtasks:
                    seg_start = frame_idx
                    nxt = "→ 切子任务2" if sub.subtask_id < len(task.subtasks) else "→ 结束整条"
                    print(f"\n  ● 子任务 {sub.subtask_id}/{len(task.subtasks)}: {sub.zh}", flush=True)
                    print(f"    空格/s = 本段成功 {nxt}    f = 本段失败(选类型后整条结束)", flush=True)
                    print(f"    g=丢弃整条  p=暂停  回车/q=急停退出", flush=True)
                    label = None
                    while label is None:
                        # ★ 每个动作块开始前, 把限幅基准重新锚定到【机械臂实际位置】。
                        #   不锚定的话: 臂跟不上指令时 last_cmd 会一路跑到实际位置前面,
                        #   而新 chunk[0] 是按【实际观测】预测的(在后面) → 限幅从 last_cmd
                        #   往回走 → 块内前进、块边界倒退, 表现为"原地前进撤回跳动"。
                        #   锚定后每块都从真实位起步, 误差不会跨块累积。
                        lag = float(np.abs(last_cmd[task.exec_joint_dims]
                                           - state[task.exec_joint_dims]).max())
                        if args.resync_anchor:
                            last_cmd = state.copy()
                        _t0 = time.monotonic()
                        chunk = np.asarray(policy.infer(build_obs(
                            state, frames, sub.en, task.exec_side,
                            preresize=not args.no_preresize))["actions"])
                        infer_ms = (time.monotonic() - _t0) * 1000
                        for i in range(min(args.execute_horizon, len(chunk))):
                            tick = time.monotonic()
                            target = clamp_target(chunk[i], last_cmd, task,
                                                  args.rate_limit, args.limit_margin)
                            # 主臂不执行: 它那 8 维动作 = 当前实读(保持 action≡state 语义)
                            ls = task.leader_slice
                            target[ls] = state[ls]
                            if args.execute and not paused:
                                arms.send_joints(target)
                                arms.send_grippers(target, args.grip_force, args.grip_squeeze)
                                last_cmd = target
                            elif not args.execute:
                                last_cmd = target
                            record_frame(ds, task, sub, state, target, frames, frame_idx,
                                         is_first_of_episode=(frame_idx == 0),
                                         is_first_of_subtask=(frame_idx == seg_start))
                            frame_idx += 1

                            key = (keypad.get(0.0) or "").lower()
                            if key:
                                if key in ("\r", "\n", "q"):
                                    print("\n🛑 急停退出。", flush=True)
                                    if args.execute:
                                        arms.estop()
                                    ds.clear_episode_buffer()
                                    raise KeyboardInterrupt
                                if key in (" ", "s"):
                                    label = "s"
                                    break
                                if key == "f":
                                    label = "f"
                                    break
                                if key == "g":
                                    label = "g"
                                    break
                                if key == "p":
                                    paused = not paused
                                    print(f"\n    {'⏸ 暂停(不发命令)' if paused else '▶ 继续'}",
                                          flush=True)
                            state = np.asarray(arms.read_state())
                            frames = cameras.read_frames()
                            dt = time.monotonic() - tick
                            if dt < PERIOD:
                                time.sleep(PERIOD - dt)
                        gfb = arms.grip_feedback() if args.execute else ""
                        # lag = 上一块结束时"指令 vs 实际"的最大脱节(rad)。持续 >0.1 说明
                        # 臂跟不上 → 调大 --speed-percent 或调小 --rate-limit。
                        print(f"    本段 {frame_idx - seg_start} 帧  推理{infer_ms:4.0f}ms  "
                              f"脱节{lag:.3f}rad  {gfb}  (空格/s=成功 f=失败)   ",
                              end="\r", flush=True)

                    if label == "g":
                        aborted = True
                        break
                    success = (label == "s")
                    ftype = "none" if success else choose_failure_type(sub, keypad)
                    label_segment(ds, seg_start, frame_idx, success, ftype)
                    records.append((sub.subtask_id, seg_start, frame_idx, success, ftype))
                    print(f"    子任务{sub.subtask_id} 标为 "
                          f"{'成功' if success else '失败 / ' + ftype}", flush=True)
                    if not success:
                        # 第一个子任务就失败 → 不进入第二个, 整条到此结束(按你的流程)
                        failed_early = True
                        break

                if aborted:
                    ds.clear_episode_buffer()
                    print("\n已丢弃整条。", flush=True)
                else:
                    save_episode(ds, args, task, records)
                    n_saved += 1
                    if failed_early and len(records) < len(task.subtasks):
                        print("  (子任务失败, 未进入后续子任务 —— 请复位后从头再来)", flush=True)

                keypad.drain()
                if args.execute:
                    # ⚠ 不再默认失能: 该硬件 disable 后抱闸不保持, 臂会直接砸下来。
                    #   改为「保持使能 + 钉住当前位姿」—— 臂停在原地不下落, 人可从容复位。
                    #   真要下电请物理支撑后按 d, 或加 --disable-at-end。
                    #   (主臂本来就是零力模式, 不受影响, 仍可直接手搬。)
                    try:
                        q_hold = np.asarray(arms.read_state(), dtype=np.float64).copy()
                        arms.send_joints(q_hold)
                        print("\n本条结束。已【保持使能并钉住当前位姿】, 臂不会下落。", flush=True)
                        print(f"   手推着复位到初始位 + 摆好盒子;"
                              f"{task.leader_side}臂在主臂模式可直接手搬换位置。", flush=True)
                        print("   空格 = 开下一条    d = 真要失能(⚠ 松垂, 先托住!)    回车/q = 退出",
                              flush=True)
                    except Exception as _exc:  # noqa: BLE001
                        print("\n  ⚠ 钉位失败(%s); 未下电, 臂仍使能" % _exc, flush=True)
                    if getattr(args, "disable_at_end", False):
                        arms.disable_for_reset()
                        print("   (--disable-at-end: 已失能, 注意臂会松垂)", flush=True)
                    else:
                        while True:
                            _k = (keypad.get(0.2) or "").lower()
                            if _k == "d":
                                arms.disable_for_reset()
                                print(f"   🔧 {task.exec_side}臂已失能(松垂, 托住了吗?), "
                                      f"复位好按 空格 开下一条", flush=True)
                                continue
                            if _k == " ":
                                break
                            if _k in ("\r", "\n", "q"):
                                raise KeyboardInterrupt
                else:
                    print("\n就绪。空格=下一条  回车/q=退出", flush=True)
    except KeyboardInterrupt:
        print("\n退出中…", flush=True)
        if args.execute:
            arms.estop()
    finally:
        cameras.close()
        arms.close()
        print(f"\n本次保存 {n_saved} 条 → {args.root}", flush=True)
