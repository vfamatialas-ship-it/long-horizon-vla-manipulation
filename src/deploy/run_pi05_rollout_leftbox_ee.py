#!/usr/bin/env python3
"""Nero pi0.5 rollout 采集 —— 【左臂抓盒装箱·末端位姿版】(2 子任务), 边部署边录。

模型: config `pi05_nero_left_box_pick_ee_v1` / lbp_ee_run1 / step 19999
  ckpt: <CKPT_ROOT_EE>/pi05_nero_left_box_pick_ee_v1/lbp_ee_run1/19999
  这是**补训版**: v1 从 pi05_base 训 20000 步(loss 0.0841→0.0077), v1b 在 v1 的 19999
  存档上再训 20000 步(lr 重启到 5e-5), 最终 loss **0.0049**, 比 v1 又降 36%。
  数据 local/nero_left_box_pick_ee_v1(100 集 / 38087 帧 / 15Hz)。

⚠⚠ 这是【末端位姿版】, 不是关节版 —— 与 run_pi05_rollout_leftbox_v2.py 的根本区别:
      leftbox_v2 : 16 维双臂关节角进出, 动作直接 move_joints() 发下去
      本脚本     : **10 维绝对末端位姿**进出(x,y,z + rot6D + 爪宽), 必须【接 IK】
                   解算成关节角才能驱动机械臂。
  骨架取自 run_pi05_rollout_trash_ee.py(同为单臂 EE + IK, 已打通链路), 不抄关节版。
  IK 的两个坑沿用那边的规避:
    ① 零空间必须用精确投影 I−J⁺J(近似投影会让冗余维乱走)
    ② **q_seed 与 q_ref 必须分开**: seed=上一步的解(保连续), ref=本 chunk 起始的实测
       关节角且【整块固定】。让 ref 跟着 seed 走时实测闭环漂移达 65~155 mrad。

═══ 与 trash_ee 的差别: 2 个子任务, 不是单 prompt ═══
  ① 左臂从左侧区域抓一个盒子
  ② 把抓到的盒子放进中间打包箱的空位
  ⚠ tasks.jsonl 里 task_index **顺序是反的**(0=放置, 1=抓取), 但执行必须先抓后放,
    所以 SUBTASKS 按执行序排, 每项自带它在训练集里的 task_index。
  打标沿用 leftbox_v2 口径: **逐段标** —— 空格/s=本段成功并切下一段, f=本段失败(整条结束)。

═══ 单臂 ═══
只有左臂执行。右臂不参与 —— 本脚本【完全不碰右臂】, 连读都不读
(数据集 state/action 就是 10 维, 没有右臂的位置)。

═══ 相机 ═══
config 的 repack 只取两路 RGB:
  observation.images.third_view  → observation/image      (base_0_rgb)
  observation.images.left_wrist  → observation/wrist_image (left_wrist_0_rgb)
  第三槽零填 + mask=False —— 右腕相机不开。

═══ 使用 ═══
0. 起服务(<SERVER>, GPU0):
     GPU=0 PORT=8032 STEP=19999 bash ~/项目/nero/deploy/ee_server/serve_leftbox_ee.sh
1. 清代理: export no_proxy=<POLICY_SERVER_IP> ; unset http_proxy https_proxy all_proxy
2. 干跑验键位:  python3 run_pi05_rollout_leftbox_ee.py --mock-robot --mock-cameras
3. 真机 dry-run: python3 run_pi05_rollout_leftbox_ee.py            (只推理只录, 不发运动)
4. 真执行:      python3 run_pi05_rollout_leftbox_ee.py --execute

键位: 空格/s = 本段成功并切下一段    f = 本段失败(选类型, 整条结束)
      g=丢弃整条   p=暂停/继续   回车/q=急停(停发+左臂下电抱闸)
相机与 CAN 按设备序列号自动检测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

NERO_ROOT = Path(__file__).resolve().parents[1]
for _p in (NERO_ROOT / "nero_control",
           NERO_ROOT / "lerobot_data" / "lerobot_tools",
           NERO_ROOT / "lerobot_data" / "lerobot_tools" / "lerobot_v21" / "src",
           NERO_ROOT / "deploy" / "openpi-client" / "src",
           NERO_ROOT / "ee_pose" / "tools",          # ee_repr / ik_nero
           Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = "127.0.0.1,localhost,<POLICY_SERVER_IP>,<POLICY_SERVER_IP>"

from openpi_client import image_tools  # noqa: E402
from camera_utils import CameraSet, add_camera_args, build_camera_configs  # noqa: E402
from collect_dual_nero import KeyPad  # noqa: E402
import ee_repr as E  # noqa: E402  (rot6d_to_mat / mat_to_rot6d / TCP 常量)
import ik_nero as K  # noqa: E402  (solve_arm_ik: DLS + 精确零空间投影)
import rollout_dualarm_common as C  # noqa: E402  (只借设备解析/常量, 不用它的双臂封装)

# 退出码约定(run_chain_4stage 按它决定下一步):
#   0   本段正常跑完
#   20  你按了 f 判失败 → chain 立即停整条链, 并统一编码视频
#   130 Ctrl-C
EXIT_USER_FAIL = 20
# 模块级可变标志: 主循环在深层嵌套里置位, main() 末尾据此决定退出码。
_user_failed = [False]

FPS = 15
PERIOD = 1.0 / FPS
SIDE = "left"                       # 本任务只有左臂
GRIP_MAX = 0.105                    # 左爪硬件上限(数据实测最大 0.1001)
# 夹持偏置的开关判据 —— 对比对象是**策略宽度的游程极值**, 不是上一 tick。
# 合拢若很慢(每 tick 只走零点几 mm), 逐 tick 比会因为差值小于阈值而永远判不出"在合拢"。
GRIP_CLOSE_EPS_M = 0.0005   # 比合拢过的最窄再窄 0.5mm 就算"还在往里合"
GRIP_REOPEN_M = 0.002       # 比最窄处宽 2mm 才算"要张开放手" —— 大于抖动, 远小于真张开
_RS = 224                           # 服务端 ResizeImages 目标尺寸
TCP = E.TCP_FINGERTIP

# 两个子任务, **按执行顺序**排(先抓后放)。文本逐字取自训练集 meta/tasks.jsonl —
# prompt_from_task=True 时模型靠它区分该做哪一段, 改一个字都可能失配。
# ⚠ tasks.jsonl 里的 task_index 顺序是反的(0=放置, 1=抓取), 所以显式带上原索引。
SUBTASKS = (
    {"task_index": 1, "zh": "从左侧区域抓一个盒子",
     "en": "Use the left arm to grasp a box from the left-side area."},
    {"task_index": 0, "zh": "把盒子放进中间打包箱的空位",
     "en": "Use the left arm to place the grasped box into an empty spot "
           "of the middle packing carton."},
)

# nero_left_box_pick_ee_v1 实测 10 维范围(2026-08-26 从 100 集 / 38087 帧 parquet 实算)。
# 维度: ee_x, ee_y, ee_z, rot6d_0..5, gripper_width
STATE_MIN = np.array([-0.4790, -0.3192, 0.7168, -0.7522, -0.5758, -0.1146,
                      -0.3709, -0.9998, -0.6537, 0.0000])
STATE_MAX = np.array([0.0784, 0.0335, 1.2376, 0.9996, 0.9957, 1.0000,
                      1.0000, 0.6983, 0.4124, 0.1001])
# 位置逐帧变化 p50/p90/p99/max = 0.0018/0.0144/0.0278/0.2145
# → 限幅取 p99 量级(0.03); max 0.21 是毛刺, 不能照它放宽。

FAILURE_TYPES = (
    ("grasp_missed", "没抓到盒子/抓空"),
    ("grasp_slipped", "抓到后滑落"),
    ("wrong_box", "抓错盒子(不是最左侧的)"),
    ("place_outside", "没放进打包箱/掉在外面"),
    ("place_wrong_slot", "放错格子/压到已有的盒子"),
    ("collision", "碰撞(碰到箱体/其它盒子)"),
    ("ik_or_limit", "IK 解不出或卡限位"),
    ("other", "其他"),
)
_IMG_FEAT = {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]}
EVENT = C.EVENT


# ══════════════════════ 左臂封装(只碰左臂) ══════════════════════
class LeftArm:
    """单臂封装。state/action 都是 10 维末端位姿, 关节角只在 IK 前后出现。"""

    def __init__(self, args: argparse.Namespace) -> None:
        from single_nero_driver import NeroArm, NeroArmConfig
        self.arm = NeroArm(NeroArmConfig(
            name="left", can_channel=args.left_can, firmware=args.left_firmware,
            can_interface=args.can_interface, bitrate=args.can_bitrate,
            timeout=args.can_timeout))
        self._last_grip = None
        self._grip_ref = None       # 策略宽度的游程极值(最近一个拐点)
        self._closing = False       # 是否处于"合拢/夹住保持"阶段(决定加不加偏置)

    def read_joints(self) -> np.ndarray:
        return np.asarray(self.arm.read_joints(), dtype=np.float64)

    def read_grip(self) -> float:
        try:
            return float(self.arm.read_gripper_width())
        except Exception:  # noqa: BLE001
            return 0.0

    def read_state10(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (10维末端位姿, 7维关节角)。位姿由 FK 从实测关节角算出 —— 与训练数据
        的生成口径一致(数据集的 observation.state 也是这么来的)。"""
        q = self.read_joints()
        T = E.fk_batch(q[None], side=SIDE, tcp=TCP)[0]
        s10 = np.empty(10, dtype=np.float64)
        s10[0:3] = T[:3, 3]
        s10[3:9] = E.mat_to_rot6d(T[:3, :3][None])[0]
        s10[9] = self.read_grip()
        return s10, q

    def refresh_feedback(self) -> None:
        try:
            self.arm.set_can_control_mode(enable_can_push=True)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ refresh_feedback 失败: {exc}", flush=True)

    def enable(self, speed_percent: int, hold_joints=None) -> None:
        hj = None if hold_joints is None else [float(v) for v in np.asarray(hold_joints)[:7]]
        self.arm.enable(speed_percent=speed_percent, hold_joints=hj)
        try:    # 复位失能时夹爪也断电, 发一次当前宽度复能
            w = float(np.clip(self.read_grip(), 0.0, GRIP_MAX))
            self.arm.set_gripper_width(w, force_n=1.0)
        except Exception:  # noqa: BLE001
            pass
        self._last_grip = None
        self._grip_ref = None       # 策略宽度的游程极值(最近一个拐点)
        self._closing = False       # 是否处于"合拢/夹住保持"阶段(决定加不加偏置)

    def send_joints(self, q7) -> None:
        self.arm.move_joints([float(v) for v in np.asarray(q7)[:7]])

    def send_grip(self, width: float, force_n: float, squeeze: float = 0.0) -> None:
        """夹爪是宽度模式: 走到目标宽度就停、不再用力。示教里的宽度常只是"刚贴上物体",
        rollout 时东西会滑 → 命令得比策略目标再窄 squeeze 米, 夹爪才会持续挤压,
        挤压力由 force_n 封顶。

        ★ 只在【抓到东西之后】加偏置, 分三态:
          · 张开/接近途中  → 不加。免得白白把开口收窄, 反而蹭到盒子。
          · 合拢 + 夹住保持 → 加。这才是要多收那 1.5mm 的时刻。
          · 明显张开(要放手)→ 立刻撤。不撤的话这点偏置会拖住松手,
                              把"放进打包箱"变成"放不下"。
        """
        w_cmd = float(np.clip(float(width), 0.0, GRIP_MAX))
        if self._grip_ref is None:
            self._grip_ref = w_cmd
        elif w_cmd < self._grip_ref - GRIP_CLOSE_EPS_M:      # 比合拢过的最窄还窄 → 在合
            self._closing, self._grip_ref = True, w_cmd
        elif w_cmd > self._grip_ref + GRIP_REOPEN_M:         # 明显变宽 → 在张开/放手
            self._closing, self._grip_ref = False, w_cmd
        # 其余 = 停在原处保持; 状态不变(合拢后的保持仍算 closing)
        w = (max(0.0, w_cmd - squeeze) if (self._closing and squeeze > 0.0) else w_cmd)
        if self._last_grip is None or abs(w - self._last_grip) > 0.001:
            self.arm.set_gripper_width(w, force_n=force_n)
            self._last_grip = w

    def grip_feedback(self) -> str:
        try:
            st = self.arm.read_gripper_status()
            if st:
                return f"爪{st.get('pos_mm', 0):.0f}mm/{st.get('force_N', 0):.1f}N"
        except Exception:  # noqa: BLE001
            pass
        return ""

    def estop(self) -> None:
        try:
            self.arm.arm.disable()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ disable 失败: {exc}", flush=True)

    def disable_for_reset(self) -> None:
        """失能关节+夹爪(松垂可手搬); can_push 不关 → 反馈保持新鲜, 手摆后读得到真值。"""
        try:
            self.arm.disable()
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ disable 失败: {exc}", flush=True)

    def close(self) -> None:
        self.arm.close()


class MockArm:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self._q = np.array([-0.166, 0.570, 1.443, 1.932, -2.327, -0.491, 0.360])

    def read_joints(self):
        return self._q + 0.02 * np.sin(time.monotonic() - self.t0 + np.arange(7))

    def read_grip(self):
        return 0.03

    def read_state10(self):
        q = self.read_joints()
        T = E.fk_batch(q[None], side=SIDE, tcp=TCP)[0]
        s = np.empty(10)
        s[0:3] = T[:3, 3]; s[3:9] = E.mat_to_rot6d(T[:3, :3][None])[0]; s[9] = 0.03
        return s, q

    def refresh_feedback(self): pass
    def enable(self, speed_percent, hold_joints=None): print(f"[mock] enable {speed_percent}%", flush=True)
    def send_joints(self, q7): pass
    def send_grip(self, width, force_n, squeeze=0.0): pass
    def grip_feedback(self): return ""
    def estop(self): print("[mock] estop", flush=True)
    def disable_for_reset(self): print("[mock] disable_for_reset", flush=True)
    def close(self): pass


# ══════════════════════ 观测 / IK ══════════════════════
def build_obs(state10: np.ndarray, frames: dict, prompt: str, preresize: bool = True) -> dict:
    """两路 RGB —— 与 config 的 repack 一一对应(third→image, left_wrist→wrist_image)。

    preresize: 客户端先用服务端同一个函数缩到 224(再缩一次是恒等操作), 负载 2.76MB→0.45MB。
    录进数据集的仍是 480x640 原图。
    """
    def rs(img):
        return image_tools.resize_with_pad(np.asarray(img), _RS, _RS) if preresize else img

    return {
        "observation/image": rs(frames["observation.images.third"]),
        "observation/wrist_image": rs(frames["observation.images.left_wrist"]),
        "observation/state": np.asarray(state10, dtype=np.float64),
        "prompt": prompt,        # ← 逐段切换, 模型靠它区分抓/放
    }


def clamp_pose(a10: np.ndarray, last10: np.ndarray, rate_pos: float, margin: float) -> np.ndarray:
    """末端位姿限幅: 训练范围钳位 + 位置项逐 tick 变化率限幅。

    只限位置(前3维)和爪宽; 旋转 6 维不做速率限幅 —— rot6D 分量本身不是欧氏量,
    逐分量限幅会把旋转矩阵拧歪。旋转的连续性由 IK 的 max_step 兜底。
    """
    t = np.asarray(a10, dtype=np.float64).copy()
    t = np.clip(t, STATE_MIN - margin, STATE_MAX + margin)
    t[9] = np.clip(t[9], 0.0, GRIP_MAX)
    dp = np.clip(t[0:3] - last10[0:3], -rate_pos, rate_pos)
    t[0:3] = last10[0:3] + dp
    return t


def pose10_to_T(s10: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(s10)[0:3]
    T[:3, :3] = E.rot6d_to_mat(np.asarray(s10)[3:9][None])[0]
    return T


# ══════════════════════ 数据集 ══════════════════════
def create_dataset(args):
    """LeRobot v2.1。state/action 存 10 维末端位姿(与训练数据同构), 同时把 IK 解出的
    关节角和残差一并记下来 —— 出问题时能区分"策略给的位姿不好"还是"IK 没解好"。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    i64 = {"dtype": "int64", "shape": (1,), "names": None}
    s1 = {"dtype": "string", "shape": (1,), "names": None}
    f32 = {"dtype": "float32", "shape": (1,), "names": None}
    names10 = ["ee_x", "ee_y", "ee_z"] + [f"ee_rot6d_{i}" for i in range(6)] + ["gripper_width"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (10,), "names": names10},
        "action": {"dtype": "float32", "shape": (10,), "names": names10},
        "observation.images.third_view": dict(_IMG_FEAT),
        "observation.images.left_wrist": dict(_IMG_FEAT),
        "joint_cmd": {"dtype": "float32", "shape": (7,),
                      "names": [f"left_j{i}" for i in range(1, 8)]},
        "ik_pos_err": dict(f32),
        "ik_ok": dict(i64),
        "active_arm": dict(s1),
        "prompt_text": dict(s1),
        "prompt_text_zh": dict(s1),
        "train_task_index": dict(i64),   # 训练集 tasks.jsonl 里的原索引(0=放置 1=抓取)
                                         # ⚠ 不能叫 task_index —— 那是 LeRobot 保留列
        "subtask_instance_id": dict(i64),
        "subtask_start": dict(i64),
        "subtask_end": dict(i64),
        "subtask_success": dict(i64),
        "episode_success": dict(i64),
        "failure_type": dict(s1),
    }
    if args.root.exists() and (args.root / "meta" / "info.json").exists():
        LeRobotDatasetMetadata(args.repo_id, root=args.root)
        ds = LeRobotDataset(args.repo_id, root=args.root)
        print(f"▶ 续采: 已有 {ds.meta.total_episodes} 条, 接着编号。", flush=True)
        return ds
    args.root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features, root=args.root,
                                 robot_type="left_nero_box_pick_ee", use_videos=True,
                                 image_writer_threads=args.image_writer_threads,
                                 batch_encoding_size=getattr(args, 'batch_encoding_size', 1))


def record_frame(ds, state10, action10, q_cmd, frames, frame_idx, ik_err, ik_ok,
                 sub: dict, sid: int, first_of_sub: bool) -> None:
    ds.add_frame({
        "observation.state": np.asarray(state10, dtype=np.float32),
        "action": np.asarray(action10, dtype=np.float32),
        "observation.images.third_view": frames["observation.images.third"],
        "observation.images.left_wrist": frames["observation.images.left_wrist"],
        "joint_cmd": np.asarray(q_cmd, dtype=np.float32)[:7],
        "ik_pos_err": np.array([float(ik_err)], dtype=np.float32),
        "ik_ok": np.array([1 if ik_ok else 0], dtype=np.int64),
        "active_arm": "RIGHT",
        "prompt_text": sub["en"],
        "prompt_text_zh": sub["zh"],
        # train_task_index = 训练集原索引(0=放置 1=抓取); subtask_instance_id = 执行序(1,2)
        # LeRobot 自己的 task_index 由 task= 参数生成, 不在这里给。
        "train_task_index": np.array([sub["task_index"]], dtype=np.int64),
        "subtask_instance_id": np.array([sid + 1], dtype=np.int64),
        "subtask_start": np.array([1 if first_of_sub else 0], dtype=np.int64),
        "subtask_end": np.array([0], dtype=np.int64),         # 切段时回填
        "subtask_success": np.array([-1], dtype=np.int64),    # 切段时回填
        "episode_success": np.array([-1], dtype=np.int64),    # 占位, 保存时回填
        "failure_type": "pending",                             # 占位
    }, task=sub["en"], timestamp=frame_idx / FPS)


def choose_failure() -> str:
    print("\n  失败类型(序号, 回车=other):", flush=True)
    for i, (_c, zh) in enumerate(FAILURE_TYPES, 1):
        print(f"    {i}. {zh}", flush=True)
    v = input("  > ").strip()
    if v.isdigit() and 1 <= int(v) <= len(FAILURE_TYPES):
        return FAILURE_TYPES[int(v) - 1][0]
    return "other"


def save_episode(ds, args, success: bool, failure_type: str, ik_stats: dict,
                 seg_bounds: list) -> int:
    """整条打标 + 回填段边界与逐段成败。

    seg_bounds: [(start, end, ok), ...] 按执行序。逐段标是 leftbox_v2 的口径 ——
    它比"只标整条"多出定位能力: 失败发生在抓取段还是放置段, RECAP 的优势分配用得上。
    ⚠ 老的 collect_leftbox_common.py 有个 bug: 定义了 close_active() 却从没调用,
      导致 subtask_success 一直停在占位值 1。这里显式逐帧回填, 不走那条路。"""
    size = int(ds.episode_buffer["size"])
    ds.episode_buffer["episode_success"] = [np.array([1 if success else 0], np.int64)
                                            for _ in range(size)]
    ds.episode_buffer["failure_type"] = [failure_type for _ in range(size)]
    for (a, b, ok) in seg_bounds:
        for j in range(a, min(b, size)):
            ds.episode_buffer["subtask_success"][j] = np.array([1 if ok else 0], np.int64)
        if b > a:
            ds.episode_buffer["subtask_end"][min(b - 1, size - 1)] = np.array([1], np.int64)
    ep = ds.meta.total_episodes
    ds.save_episode()
    ann = args.root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    with open(ann / "episode_outcomes.jsonl", "a") as f:
        f.write(json.dumps({
            "episode_index": ep, "frames": size,
            "episode_success": bool(success), "failure_type": failure_type,
            "exec_arm": "left", "source": "rollout_ee",
            "segments": [{"subtask_id": i + 1,
                          "task_index": SUBTASKS[i]["task_index"] if i < len(SUBTASKS) else -1,
                          "prompt": SUBTASKS[i]["en"] if i < len(SUBTASKS) else "",
                          "start": a, "end": b, "success": bool(ok)}
                         for i, (a, b, ok) in enumerate(seg_bounds)],
            "ik_fail_frames": ik_stats.get("fail", 0),
            "ik_pos_err_mean_mm": round(ik_stats.get("err_mean", 0.0) * 1000, 3),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False) + "\n")
    print(f"\n✓ 已存 episode {ep:06d}  帧{size}  "
          f"{'成功' if success else '失败/' + failure_type}  "
          f"IK未收敛 {ik_stats.get('fail', 0)} 帧", flush=True)
    return ep


# ══════════════════════ 安全 ══════════════════════
def safe_enable(arm, args):
    """左臂使能防冲: 以当前实读关节角为伺服目标使能, 1.5 秒内监控漂移。"""
    arm.refresh_feedback()
    time.sleep(0.3)
    q0 = arm.read_joints()
    print("\n使能前关节角(新鲜):", np.round(q0, 3).tolist(), flush=True)
    print(f"⚠ 使能左臂(speed={args.speed_percent}%)。确认工作区无人手、急停在手边。", flush=True)
    arm.enable(args.speed_percent, hold_joints=q0)
    drift = 0.0
    for _ in range(int(1.5 * FPS)):
        arm.send_joints(q0)
        time.sleep(PERIOD)
        drift = float(np.abs(arm.read_joints() - q0).max())
        if drift > args.enable_drift_abort:
            print(f"\n🛑 使能后漂移 {drift:.3f}rad > {args.enable_drift_abort} — 立即下电!", flush=True)
            arm.estop()
            return None
    print(f"✓ 使能通过防冲检测(漂移 {drift:.4f} rad)", flush=True)
    return arm.read_joints()


# ══════════════════════ 参数 ══════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description="Nero pi0.5 rollout 采集 —— 左臂抓盒装箱(末端位姿版, 2 子任务)")
    ap.add_argument("--host", default="<POLICY_SERVER_IP>")
    ap.add_argument("--port", type=int, default=8029,
                    help="leftbox EE serve 端口")
    ap.add_argument("--root", default=None)
    ap.add_argument("--repo-id", default="local/nero_leftbox_ee_rollout")
    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True,
                    help="真执行(默认开; --no-execute 回到 dry-run 只推理只录不发运动)")
    ap.add_argument("--mock-robot", action="store_true")
    ap.add_argument("--disable-at-end", action="store_true",
                    help="每条结束后失能(默认不失能: 该硬件 disable 抱闸不保持, 臂会砸下来)")
    # 整块 50 帧用满(原 40/45)。网络抖时推理要 0.5~15s, 用满一块 = 每次推理换来
    # 3.33s 连续运动(原 2.67/3.00s), 占空比更高、动作更连贯。
    ap.add_argument("--execute-horizon", type=int, default=50,
                    help="每次推理执行多少帧后带新观测重推理")
    # 0.03 m/tick 只有训练 p99(26.6mm)的 1.13 倍, 快动作会被削。
    # 0.06 = 0.9 m/s, 是 p99 的 2.3 倍, 留足余量又不至于失控。
    ap.add_argument("--rate-limit-pos", type=float, default=0.06,
                    help="末端位置每 tick 最大变化(m)。数据实测逐帧 p99=0.0278, max 0.21 是毛刺")
    # ★ 抗积分饱和: 关节指令最多超前实际位置多少弧度。位置伺服里 力矩 ∝ 位置误差,
    #   夹住它就是夹住堵转力矩 —— 顶到东西会"轻轻顶着"而不是越顶越狠到过流跳闸。
    #   0.12 rad ≈ 6.9°(比 E2/E3 的 0.10 略松, 因为抓盒是自由空间运动)。0 = 关闭。
    ap.add_argument("--stall-gap", type=float, default=0.12)
    ap.add_argument("--limit-margin", type=float, default=0.05,
                    help="训练范围外放宽多少(位姿单位)")
    ap.add_argument("--enable-drift-abort", type=float, default=1.5)
    ap.add_argument("--speed-percent", type=int, default=20)
    # 夹爪: force 用项目基线 1.0(hezi 那几个脚本一直是这个值);
    #       squeeze 1.5mm 且**只在夹住后生效**(张开/接近不加, 要放手时立刻撤)。
    ap.add_argument("--grip-force", type=float, default=1.0,
                    help="夹爪力上限N(默认1.0基线; 滑落再加到3~5)")
    ap.add_argument("--grip-squeeze", type=float, default=0.0015,
                    help="夹持偏置m(默认0.0015=1.5mm, 只在夹住后生效; 还滑加到0.003, 夹坏调0)")
    ap.add_argument("--defer-label", action="store_true",
                    help="跑完不问 s/f/g, 自动保存; 成功/失败留到整条链结束统一标注")
    ap.add_argument("--auto-start", action="store_true",
                    help="不等「空格=开始」, 直接开跑(串跑用; 单跑时保留人工确认)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="执行节拍倍速。1.0=按训练的 15Hz 回放。"
                         "⚠ 只改节拍不改限幅; >1 会让自动切换略微晚切, 建议 ≤1.5")
    ap.add_argument("--auto-switch", action="store_true",
                    help="子任务自动切换: 用进度头判断该不该切, 不用人按空格。"
                         "空格仍可手动覆盖")
    ap.add_argument("--switch-expert", type=int, default=None,
                    help="切换器的专家编号。默认按本客户端固定值 1")
    ap.add_argument("--switch-device", default="cuda",
                    help="切换器跑在哪。⚠ CPU 上 SigLIP2 约 2.8Hz, 撑不住 15Hz")
    ap.add_argument("--no-home", action="store_true",
                    help="本条结束后不归位(旧行为)。默认送回 home_pose.json 的摆位")
    ap.add_argument("--home-rate", type=float, default=0.05,
                    help="归位轨迹每 tick 关节步进(rad)")
    ap.add_argument("--once", action="store_true",
                    help="录完一条就退出 —— 串跑时让空格走完最后一个子任务即推进到下一大阶段")
    ap.add_argument("--no-encode-on-exit", action="store_true",
                    help="本段结束不编码视频 —— 串跑时由 run_chain_4stage 在\n                         整条链结束(或按 f 中止)时统一编。单跑不设时照常在退出时编。")
    ap.add_argument("--defer-encode", action="store_true",
                    help="录制期间不编码视频(只落图像帧), 由 run_chain_4stage --finalize 最后统一编码")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--no-preresize", action="store_true")
    ap.add_argument("--no-resync-anchor", dest="resync_anchor", action="store_false",
                    help="关闭'每块锚定实际位置'(默认开)")
    # IK 参数(默认值 = ik_nero selftest 22/22 通过的那组)
    ap.add_argument("--ik-tol-pos", type=float, default=2e-3,
                    help="IK 位置容差(m)。ik_nero 默认 0.1mm 是给离线回放的(目标由FK生成必然可达), "
                         "真机策略给的位姿未必可达, 太严会把好解判成失败")
    ap.add_argument("--ik-damping", type=float, default=1e-2)
    ap.add_argument("--ik-ns-gain", type=float, default=0.30)
    ap.add_argument("--left-can", default="auto")
    ap.add_argument("--left-firmware", default="v120")
    ap.add_argument("--can-interface", default="socketcan")
    ap.add_argument("--can-bitrate", type=int, default=1_000_000)
    ap.add_argument("--can-timeout", type=float, default=1.0)
    add_camera_args(ap)
    args = ap.parse_args()

    # 局域网绕开系统代理 —— 必须在任何联网之前。不加的话连推理服务会报
    # InvalidMessage: did not receive a valid HTTP response(其实是被代理吃了)。
    from proxy_bypass import bypass_proxy  # noqa: PLC0415
    bypass_proxy(getattr(args, "host", "") or "")

    if args.root is None:
        # ── 全过程串跑: 统一落在 wholeprocess/<本次运行ID>/<阶段>/ 下, 不与单阶段旧采集混淆。
        #    WP_RUN 由 run_chain 统一下发; 单独跑某一阶段时自动生成, 行为与以前一致。
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = os.environ.get("WP_RUN") or f"run_{stamp}"
        args.root = NERO_ROOT / "lerobot_data" / "Rlinf" / "data" / "wholeprocess" / run_id / "e1_leftbox"
        args.repo_id = f"local/wholeprocess_{run_id}_e1_leftbox"
    args.batch_encoding_size = 1000000000 if args.defer_encode else 1
    args.root = Path(args.root)
    args.camera_width, args.camera_height = 640, 480

    # ── 设备自动检测(按内部序列号) ──
    print("── 设备自动检测(按内部序列号) ──", flush=True)
    if args.left_can in (None, "auto"):
        got = C.resolve_can_by_serial(C.CAN_SERIALS["left"])
        if got is None:
            raise RuntimeError(f"左臂 CAN 未找到(serial={C.CAN_SERIALS['left']})。"
                               f"检查臂上电/CAN 是否 up; 或 --left-can canX 手动指定。")
        args.left_can = got
    print(f"  左臂 CAN → {args.left_can}", flush=True)
    # 本任务只用 third + 左腕两路(config 的 repack 就这两个), 右腕不开
    for cam, attr in (("third_view", "third_camera"), ("left_wrist", "left_wrist_camera")):
        if getattr(args, attr, None) is None and not args.mock_cameras:
            got = C.resolve_video_by_serial(C.CAM_SERIALS[cam])
            if got is None:
                raise RuntimeError(f"相机 {cam} 未找到(serial={C.CAM_SERIALS[cam]})。")
            setattr(args, attr, got)
        if getattr(args, attr, None):
            print(f"  相机 {cam:>11} → {getattr(args, attr)}", flush=True)
    # ★ 自动切换要三路相机: 进度头是按 n_view=3 训的(expert 0/1 的数据集都存了三路),
    #   少一路会在 switcher._forward 里 KeyError, 喂零图又是分布外。所以开了
    #   --auto-switch 就把这一路也打开 —— **只给切换器用, 策略的观测一点没变**
    #   (repack 仍只取 third + 本臂腕部)。
    if getattr(args, "auto_switch", False) and not args.mock_cameras:
        got = C.resolve_video_by_serial(C.CAM_SERIALS["right_wrist"])
        if got is None:
            raise RuntimeError(
                f"--auto-switch 需要 right_wrist 相机(serial={C.CAM_SERIALS['right_wrist']}), 但没找到。"
                f" 插上它, 或去掉 --auto-switch 用手动空格。")
        args.right_wrist_camera = got
        print(f"  相机 right_wrist → {got}   {'(仅供自动切换)'}", flush=True)
    else:
        args.right_wrist_camera = None
    return args


# ══════════════════════ 主循环 ══════════════════════
# 本段结束后的过渡轨迹名(trajectories/<名字>.json)。
# 对应 E1 左臂抓盒 → E2 stage34。录了同名轨迹就优先回放它, 没录则回退 goto_home.py 直线插值归位。
# ⚠ 这个常量之前漏了定义 —— goto_home_inline() 一调就 NameError, 整段过渡直接崩。
DEFAULT_TRAJ = "transition_e1_to_e2"


def goto_home_inline(args) -> None:
    """本段结束后的过渡动作。优先回放**手教轨迹**, 没有就退回直线插值归位。

    为什么优先手教轨迹:
    goto_home.py 是关节空间直线插值 —— 起点到终点各关节按比例同步转。自由空间没
    问题, 但阶段之间常要绕开箱子、避免两臂互撞, 直线插值不知道这些, 容易蹭到东西
    (蹭到就触发抗积分饱和停住, 表现为"归不到位")。手教一遍再回放, 路径就是你亲手
    示范的那条。

    轨迹名由 --transition-traj 指定, 默认按阶段自动取(见各文件的 DEFAULT_TRAJ)。
    没录过就自动回退 goto_home.py, 不会因为缺文件就不动。
    """
    if getattr(args, "no_home", False) or not args.execute:
        return
    import subprocess as _sp  # noqa: PLC0415
    here = Path(__file__).resolve().parent
    name = getattr(args, "transition_traj", None) or DEFAULT_TRAJ
    traj = here / "trajectories" / f"{name}.json"
    tt, gh = here / "traj_teach.py", here / "goto_home.py"

    if traj.exists() and tt.exists():
        print(f"\n══ 过渡: 回放手教轨迹「{name}」══", flush=True)
        cmd = [sys.executable, str(tt), "--replay", name, "--execute",
               "--rate", str(getattr(args, "home_rate", 0.05))]
    elif gh.exists():
        why = "没录过手教轨迹" if not traj.exists() else "traj_teach.py 缺失"
        print(f"\n══ 过渡: 直线插值归位({why}, 回退 goto_home)══", flush=True)
        cmd = [sys.executable, str(gh), "--execute",
               "--rate", str(getattr(args, "home_rate", 0.05))]
    else:
        print("  ⚠ 既没有手教轨迹也没有 goto_home.py, 跳过过渡", flush=True)
        return
    # ★ 必须确认归位真的做完了。原来这里 _sp.call() 不看返回码 —— 归位没到位也照样
    #   进下一段, 而下一段的模型是从初始位形训练的, 起点不对整段都白跑。
    #   goto_home.py 的退出码: 0=到位  1=有关节没进容差  2=缺归位点  3=使能防冲失败
    #   4=臂没上电。只有 1 值得重试(多半是被挡了一下或速率限幅没跑够超时)。
    for attempt in (1, 2):
        try:
            rc = _sp.call(cmd, cwd=str(here))
        except KeyboardInterrupt:
            print("  ⚠ 过渡动作被中断 —— 下一段的起点可能不对", flush=True)
            return
        if rc == 0:
            if attempt > 1:
                print("  ✓ 第 2 次归位到位", flush=True)
            return
        if rc != 1 or attempt == 2:
            print(f"  ⚠ 过渡未完成(退出码 {rc})。下一段的模型是从初始位形训练的, "
                  f"起点不对整段都会偏 —— 建议停下来看现场。", flush=True)
            return
        print(f"  ⚠ 归位没进容差(退出码 {rc}), 再试一次 …", flush=True)


def encode_pending_videos(ds, args) -> None:
    """无论本次怎么结束(正常/Ctrl-C/异常), 都把推迟的视频编码掉。

    ⚠ 为什么必须放在 finally:
    --defer-encode 把 batch_encoding_size 设成天文数字, save_episode() 只落
    parquet + 图像帧, 视频留到最后统一编。原来那个"最后"只在 run_chain_4stage
    正常跑完时才发生 —— 一旦中途 Ctrl-C 或某段崩了, **视频就永远没编**, 数据集
    里只有一堆 images/ 散帧, 后面训练读不了。
    放在客户端的 finally 里, 三条退出路径都会走到, 视频一定落地。
    """
    if ds is None or not getattr(args, "defer_encode", False):
        return
    try:
        n = int(ds.meta.total_episodes)
    except Exception:  # noqa: BLE001
        return
    if n <= 0:
        return
    try:
        print(f"\n══ 编码视频({n} 条)══  中断也会编, 请等它跑完", flush=True)
        ds.batch_encode_videos(0, n)
        print("  ✓ 视频编码完成", flush=True)
    except KeyboardInterrupt:
        print("  ⚠ 编码被再次中断 —— 数据仍在, 可用 "
              "run_chain_4stage.py --finalize-only <RUN_ID> 补", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✘ 编码失败: {exc}\n"
              "     可用 run_chain_4stage.py --finalize-only <RUN_ID> 补", flush=True)


def main() -> None:  # noqa: PLR0915, PLR0912
    args = parse_args()
    mode = "EXECUTE(真执行)" if args.execute else "DRY-RUN(只推理只录, 不发运动)"
    print(f"\n任务: 左臂抓盒装箱【末端位姿版】 {len(SUBTASKS)} 子任务  模式={mode}", flush=True)
    for i, sub in enumerate(SUBTASKS):
        print(f"  第{i+1}段 (train task_index={sub['task_index']}): {sub['zh']}", flush=True)
        print(f"      {sub['en']}", flush=True)
    print(f"数据集: {args.root}", flush=True)
    print(f"夹爪: 力 {args.grip_force}N  偏置 {args.grip_squeeze*1000:.1f}mm"
          f"{'(仅夹住后生效, 张开即撤)' if args.grip_squeeze > 0 else '(关)'} | "
          f"IK 容差 {args.ik_tol_pos*1000:.1f}mm", flush=True)

    from openpi_client import websocket_client_policy
    print(f"连接推理服务 ws://{args.host}:{args.port} …", flush=True)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    ds = create_dataset(args)
    arm = MockArm() if args.mock_robot else LeftArm(args)
    cameras = CameraSet(build_camera_configs(args))

    n_saved = 0
    try:
        cameras.open()
        # ── 子任务自动切换 ────────────────────────────────────────────────────
        # 进度头只吃当前帧, 判断「这个子任务做到哪了」。切换时刻由它给, 人按空格仍可覆盖。
        # ⚠ 它要跟**相机帧**走(15Hz), 不跟策略推理走 —— 见 switcher_service.py 的说明。
        sw = None
        if getattr(args, "auto_switch", False):
            from switcher_client import SwitcherClient  # noqa: PLC0415
            sw = SwitcherClient(expert=(args.switch_expert if args.switch_expert is not None
                                        else 1),
                                device=args.switch_device)
        with KeyPad() as keypad:
            if getattr(args, "auto_start", False):
                print("\n(--auto-start: 直接开跑, 不等空格)", flush=True)
                keypad.drain()
            else:
                print("\n就绪。空格=开始一条 rollout   回车/q=退出", flush=True)
                keypad.drain()
            while True:
                k = " " if getattr(args, "auto_start", False) else (keypad.get(0.2) or "").lower()
                if k in ("\r", "\n", "q"):
                    break
                if k != " ":
                    continue

                if args.execute:
                    q_base = safe_enable(arm, args)
                    if q_base is None:
                        continue
                else:
                    q_base = arm.read_joints()

                print(f"\n▶ RECORDING 第 {ds.meta.total_episodes:06d} 条", flush=True)
                print("  空格/s=本段成功并切下一段  f=本段失败(整条结束)  g=丢弃  "
                      "p=暂停  回车/q=急停", flush=True)
                keypad.drain()
                frame_idx = 0
                paused = False
                label = None
                ik_fail = 0
                n_stall = 0
                ik_errs = []
                sid = 0                 # 当前子任务下标(执行序: 0=抓 1=放)
                last_sid = -1
                seg_bounds = []         # [(start, end, success)]
                seg_start = 0
                first_of_sub = True
                state10, q_now = arm.read_state10()
                last10 = state10.copy()
                q_cmd = q_base.copy()
                frames = cameras.read_frames()

                while label is None and sid < len(SUBTASKS):
                    sub = SUBTASKS[sid]
                    if sid != last_sid:
                        print(f"\n● 第{sid+1}/{len(SUBTASKS)}段: {sub['zh']}", flush=True)
                        last_sid = sid
                        first_of_sub = True
                    # 每块开始: 锚定到实际位置(不锚定时指令会跑到实际前面, 下一块又往回拉,
                    # 表现为"原地前进撤回跳动" —— 关节版实测确认过)
                    lag = float(np.linalg.norm(last10[0:3] - state10[0:3]))
                    if args.resync_anchor:
                        last10 = state10.copy()
                        q_cmd = q_now.copy()
                    # q_ref 整块固定 = 本块起始实测关节角(见文件头第②个坑)
                    q_ref = q_now.copy()

                    _t0 = time.monotonic()
                    chunk = np.asarray(policy.infer(build_obs(
                        state10, frames, sub["en"],
                        preresize=not args.no_preresize))["actions"])
                    infer_ms = (time.monotonic() - _t0) * 1000

                    for i in range(min(args.execute_horizon, len(chunk))):
                        tick = time.monotonic()
                        a10 = clamp_pose(chunk[i], last10, args.rate_limit_pos, args.limit_margin)
                        # 末端位姿 → IK → 关节角
                        T = pose10_to_T(a10)
                        q_sol, info = K.solve_arm_ik(
                            T, q_seed=q_cmd, side=SIDE, q_ref=q_ref, tcp=TCP,
                            damping=args.ik_damping, ns_gain=args.ik_ns_gain,
                            tol_pos=args.ik_tol_pos)
                        # ik_nero.solve_arm_ik 返回 info 的键是 ok / pos_err / rot_err / iters
                        ok = bool(info["ok"])
                        perr = float(info["pos_err"])
                        ik_errs.append(perr)
                        if not ok:
                            ik_fail += 1

                        # ── 关节层保护(2026-08-26 加)────────────────────────────
                        # 原来这里把 IK 解直接下发, 关节层**没有任何限幅**: 手臂被挡住时
                        # 实际位置不动而指令一路往前推, 位置误差累积 → 力矩 ∝ 误差 → 过流
                        # 跳闸失能(左臂 J6 实测 9.0A)。E2/E3 已有该保护, 这里补齐。
                        q_now = np.asarray(arm.read_joints(), dtype=np.float64)[:7]
                        if args.stall_gap > 0:
                            q_raw = np.asarray(q_sol, dtype=np.float64)[:7]
                            q_sol = np.clip(q_raw, q_now - args.stall_gap, q_now + args.stall_gap)
                            if not np.allclose(q_sol, q_raw, atol=1e-9):
                                n_stall += 1
                        if args.execute and not paused and ok:
                            arm.send_joints(q_sol)
                            arm.send_grip(a10[9], args.grip_force, args.grip_squeeze)
                        if ok:
                            q_cmd = np.asarray(q_sol)[:7]
                            last10 = a10
                        record_frame(ds, state10, a10, q_cmd, frames, frame_idx, perr, ok,
                                     sub, sid, first_of_sub)
                        frame_idx += 1
                        first_of_sub = False

                        key = (keypad.get(0.0) or "").lower()
                        # 自动切换: 每帧都要 step(15Hz), 漏帧等于变相加速、判据会偏。
                        # 人按了键就按人的来 —— 手动优先, 这一步不合成。
                        if sw is not None:
                            try:
                                _r = sw.step(frames, state10)
                                if _r["switched"] and not key:
                                    print("\n  [自动切换] E1 左臂抓盒 第%d段完成%s"
                                          % (_r["subtask"] - (0 if _r["finished"] else 1),
                                             "  (本专家全部完成)" if _r["finished"] else ""),
                                          flush=True)
                                    key = " "          # ← 等价于人按空格, 后面分支原样跑
                            except Exception as _e:  # noqa: BLE001
                                print(f"\n  ⚠ 自动切换出错({_e}) —— 退回手动按空格", flush=True)
                                sw = None

                        if key:
                            if key in ("\r", "\n", "q"):
                                print("\n🛑 急停: 停发运动指令, **保持使能并钉住当前位姿**(该硬件下电抱闸不保持, 会砸下来)。", flush=True)
                                if args.execute:
                                    try:
                                        _qh = np.asarray(arm.read_joints(), dtype=np.float64).copy()
                                        arm.send_joints(_qh)
                                    except Exception as _e:  # noqa: BLE001
                                        print(f"  ⚠ 钉位失败({_e}); 仍未下电, 臂保持使能", flush=True)
                                if getattr(ds, "image_writer", None) is not None:
                                    ds.image_writer.wait_until_done()   # 防 rmtree 撞上异步写图线程
                                ds.clear_episode_buffer()
                                raise KeyboardInterrupt
                            if key in (" ", "s"):
                                # 本段成功 → 切下一段(最后一段则整条成功)
                                seg_bounds.append((seg_start, frame_idx, True))
                                seg_start = frame_idx
                                sid += 1
                                if sid >= len(SUBTASKS):
                                    label = "s"
                                break
                            if key in ("f", "g"):
                                if key == "f":
                                    seg_bounds.append((seg_start, frame_idx, False))
                                label = key
                                if key == "f":
                                    _user_failed[0] = True
                                break
                            if key == "p":
                                paused = not paused
                                print(f"\n  {'⏸ 暂停(不发命令)' if paused else '▶ 继续'}", flush=True)
                        state10, q_now = arm.read_state10()
                        frames = cameras.read_frames()
                        dt = time.monotonic() - tick
                        if dt < PERIOD / max(args.speed, 1e-6):
                            time.sleep(PERIOD / max(args.speed, 1e-6) - dt)
                    gfb = arm.grip_feedback() if args.execute else ""
                    print(f"  {frame_idx}帧 推理{infer_ms:4.0f}ms IK残差{np.mean(ik_errs[-40:])*1000:.1f}mm "
                          f"未收敛{ik_fail} 脱节{lag*1000:.0f}mm {gfb}  (s=成功 f=失败)   ",
                          end="\r", flush=True)

                stats = {"fail": ik_fail,
                         "err_mean": float(np.mean(ik_errs)) if ik_errs else 0.0}
                if label == "g":
                    if getattr(ds, "image_writer", None) is not None:
                        ds.image_writer.wait_until_done()   # 防 rmtree 撞上异步写图线程
                    ds.clear_episode_buffer()
                    print("\n已丢弃整条。", flush=True)
                else:
                    success = (label == "s")
                    ftype = "none" if success else choose_failure()
                    save_episode(ds, args, success, ftype, stats, seg_bounds)
                    n_saved += 1

                keypad.drain()
                if getattr(args, "once", False):
                    # 串跑: 空格走完最后一个子任务 = 本大阶段结束 → 退出本客户端,
                    # 由 run_chain_4stage 推进到下一大阶段。放在 if args.execute 之外,
                    # 只读模式(不加 --execute)同样生效。
                    print("\n✓ --once: 本阶段完成, 退出以推进到下一大阶段。", flush=True)
                    goto_home_inline(args)
                    break
                if args.execute:
                    # ⚠ 不再默认失能: 该硬件 disable 后抱闸不保持, 臂会直接砸下来。
                    #   改为「保持使能 + 钉住当前位姿」—— 臂停在原地不下落, 人可从容复位。
                    #   真要下电请物理支撑后按 d, 或加 --disable-at-end。
                    try:
                        q_hold = np.asarray(arm.read_joints(), dtype=np.float64).copy()
                        arm.send_joints(q_hold)
                        print("\n本条结束。已【保持使能并钉住当前位姿】, 臂不会下落。", flush=True)
                        print("   手推着复位到初始位 + 摆好盒子和打包箱, 按 空格 开下一条。", flush=True)
                        print("   d = 真要失能(⚠ 松垂, 先托住!)   回车/q = 退出", flush=True)
                    except Exception as _exc:  # noqa: BLE001
                        print("\n  ⚠ 钉位失败(%s); 未下电, 臂仍使能" % _exc, flush=True)
                    if getattr(args, "disable_at_end", False):
                        arm.disable_for_reset()
                        print("   (--disable-at-end: 已失能, 注意臂会松垂)", flush=True)
                    else:
                        # 只在这里等一次: 按 d 才失能, 空格直接进下一条
                        while True:
                            _k = (keypad.get(0.2) or "").lower()
                            if _k == "d":
                                arm.disable_for_reset()
                                print("   🔧 已失能(松垂, 托住了吗?), 复位好按 空格 开下一条", flush=True)
                                continue
                            if _k == " ":
                                break
                            if _k in ("\r", "\n", "q"):
                                raise KeyboardInterrupt
                else:
                    print("\n就绪。空格=下一条  回车/q=退出", flush=True)
    except KeyboardInterrupt:
        print("\n退出中… (保持使能并钉住当前位姿, 不下电)", flush=True)
        if args.execute:
            try:
                _qh = np.asarray(arm.read_joints(), dtype=np.float64).copy()
                arm.send_joints(_qh)
            except Exception:  # noqa: BLE001
                pass
    finally:
        # ★ 先编码再关设备 —— 编码只用 CPU 和磁盘, 与相机/CAN 无关,
        #   但要保证无论怎么退出都执行到。
        try:
            if not getattr(args, "no_encode_on_exit", False):
                encode_pending_videos(locals().get("ds"), args)
            else:
                print("\n(--no-encode-on-exit: 视频留到整条链结束统一编)", flush=True)
        except Exception:  # noqa: BLE001
            pass
        cameras.close()
        arm.close()
        print(f"\n本次保存 {n_saved} 条 → {args.root}", flush=True)
    if _user_failed[0]:
        # 把"你按了 f"这个信号传给 run_chain_4stage: 它据此停链 + 统一编码。
        print("\n(按 f 判失败 → 退出码 %d, 整条链将停止并保存视频)" % EXIT_USER_FAIL,
              flush=True)
        sys.exit(EXIT_USER_FAIL)


if __name__ == "__main__":
    main()
