#!/usr/bin/env python3
"""RLinf 左臂抓盒【对照数据】采集 —— 共享实现(成功版 / 失败版共用, 由 3 个薄入口 + 3 份 YAML 驱动)。

设计要点:
- 成功版: 2 子任务(①抓取 ②放置), Space 逐子任务推进, S=全成功保存 / F=选失败类型保存 / G=丢弃;
- 失败版(按子任务分开, 单子任务): 故意采失败模式, F 时从配置的失败类型里选标签;
- 示教采集 action≡state(与 SFT left_box 数据同口径), 15Hz, 16 维(双臂关节+夹爪);
- CAN 按【适配器序列号】自动解析 canX(插拔不变), 相机按【USB 序列号】自动解析 /dev/videoN;
- 每帧写 subtask_instance_id / active_arm / prompt / episode_success / failure_type /
  subtask_success(保存时按段回填), 与 RLinf 的 add_is_success_column 口径对齐。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RLINF_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DATA_ROOT = RLINF_ROOT.parent
NERO_ROOT = LEROBOT_DATA_ROOT.parent
TOOLS_ROOT = LEROBOT_DATA_ROOT / "lerobot_tools"

sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(NERO_ROOT))

import collect_dual_nero_rgbd as base  # noqa: E402  (KeyPad, prepare_dataset_root)
from camera_rgbd_utils import CameraConfig, CameraSet, camera_features  # noqa: E402

LEFT_JOINT_NAMES = [f"left_j{i}" for i in range(1, 8)]
RIGHT_JOINT_NAMES = [f"right_j{i}" for i in range(1, 8)]
DUAL_STATE_NAMES = LEFT_JOINT_NAMES + ["left_gripper_width"] + RIGHT_JOINT_NAMES + ["right_gripper_width"]

STATE_IDLE = "IDLE"
STATE_RECORDING = "RECORDING"
STATE_WAIT_SAVE = "WAIT_SAVE"

EVENT_NONE = 0
EVENT_EPISODE_STARTED = 1
EVENT_SUBTASK_STARTED = 2
EVENT_SUBTASK_DONE = 3
EVENT_EPISODE_SUCCESS = 4
EVENT_EPISODE_FAILED = 5
EVENT_EPISODE_DISCARDED = 6
EVENT_NAMES = {
    EVENT_NONE: "NONE",
    EVENT_EPISODE_STARTED: "EPISODE_STARTED",
    EVENT_SUBTASK_STARTED: "SUBTASK_STARTED",
    EVENT_SUBTASK_DONE: "SUBTASK_DONE",
    EVENT_EPISODE_SUCCESS: "EPISODE_SUCCESS",
    EVENT_EPISODE_FAILED: "EPISODE_FAILED",
    EVENT_EPISODE_DISCARDED: "EPISODE_DISCARDED",
}


# ---------------------------------------------------------------- 配置
@dataclass(frozen=True)
class SubtaskDef:
    subtask_id: int
    name: str
    zh: str
    en: str


@dataclass
class TaskConfig:
    raw: dict[str, Any]
    subtasks: list[SubtaskDef]
    failure_types: list[str]
    active_arm: str = "LEFT"

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: Path) -> TaskConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    subtasks = [
        SubtaskDef(
            subtask_id=int(item["subtask_id"]),
            name=str(item["name"]),
            zh=str(item["zh"]),
            en=str(item["en"]),
        )
        for item in raw["subtasks"]
    ]
    failure_types = [str(v) for v in raw.get("failure_types", ["none", "other"])]
    active_arm = str(raw.get("task", {}).get("active_arm", "LEFT")).upper()
    return TaskConfig(raw=raw, subtasks=subtasks, failure_types=failure_types,
                      active_arm=active_arm)


# ---------------------------------------------------------------- 硬件解析(CAN/相机按序列号)
def _udev_serial_of(dev_path: str) -> str | None:
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
    for p in sorted(glob.glob("/sys/class/net/can*")):
        if _udev_serial_of(p) == serial:
            return os.path.basename(p)
    return None


def resolve_video_by_serial(serial: str) -> str | None:
    # 注意: /dev/videoN 是设备节点, 必须用 udevadm -n; -p 只认 /sys 路径(会报 Unknown device)
    for p in sorted(glob.glob("/dev/video*"), key=lambda x: int(x[len("/dev/video"):])):
        try:
            out = subprocess.check_output(["udevadm", "info", "-q", "property", "-n", p],
                                          text=True, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            continue
        for line in out.splitlines():
            if line.startswith("ID_SERIAL_SHORT=") and line.split("=", 1)[1].strip() == serial:
                return p
    return None


def ensure_can_up(can: str, bitrate: int) -> None:
    try:
        state = subprocess.check_output(["ip", "-details", "link", "show", can],
                                        text=True, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return
    if "state UP" not in state:
        print(f"  CAN {can} 未 UP → 尝试拉起...", flush=True)
        subprocess.run(["sudo", "-n", "ip", "link", "set", can, "up", "type", "can",
                        "bitrate", str(bitrate)], capture_output=True, text=True)
        time.sleep(0.3)


def apply_exposure(device: str, ctrl: dict[str, Any]) -> None:
    if shutil.which("v4l2-ctl") is None:
        return
    args = ["v4l2-ctl", "-d", device, "-c", "auto_exposure=1",
            "-c", "exposure_dynamic_framerate=0"]
    for k, v in ctrl.items():
        args += ["-c", f"{k}={v}"]
    subprocess.run(args, capture_output=True, text=True)


# ---------------------------------------------------------------- 特征/数据集
def int_feature(value: int) -> np.ndarray:
    return np.asarray([int(value)], dtype=np.int64)


def dataset_features(camera_configs: list[CameraConfig]) -> dict[str, dict[str, Any]]:
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": DUAL_STATE_NAMES},
        "action": {"dtype": "float32", "shape": (16,), "names": DUAL_STATE_NAMES},
    }
    features.update(camera_features(camera_configs))
    features.update({
        "subtask_instance_id": {"dtype": "int64", "shape": (1,), "names": None},
        "active_arm": {"dtype": "string", "shape": (1,), "names": None},
        "prompt_index": {"dtype": "int64", "shape": (1,), "names": None},
        "prompt_text": {"dtype": "string", "shape": (1,), "names": None},
        "prompt_text_zh": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_start": {"dtype": "int64", "shape": (1,), "names": None},
        "subtask_end": {"dtype": "int64", "shape": (1,), "names": None},
        "event_code": {"dtype": "int64", "shape": (1,), "names": None},
        "event_name": {"dtype": "string", "shape": (1,), "names": None},
        "episode_success": {"dtype": "int64", "shape": (1,), "names": None},
        "failure_type": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_success": {"dtype": "int64", "shape": (1,), "names": None},
    })
    return features


def create_dataset(args: argparse.Namespace, cfg: TaskConfig):
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"Expected LeRobot codebase v2.1, got {CODEBASE_VERSION}")
    if args.overwrite and args.root.exists():
        shutil.rmtree(args.root)
    if args.resume and not args.overwrite and args.root.exists() and not base._dataset_is_empty(args.root):
        meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
        wanted = dataset_features(args.camera_configs)
        for key, spec in wanted.items():
            if key not in meta.features:
                raise RuntimeError(f"续采失败:已有数据缺少特征 {key}。换 --repo-id 或 --root 新采。")
            if tuple(meta.features[key]["shape"]) != tuple(spec["shape"]):
                raise RuntimeError(f"续采失败:{key} shape {tuple(meta.features[key]['shape'])} != {tuple(spec['shape'])}")
        if int(round(meta.fps)) != int(args.fps):
            raise RuntimeError(f"续采失败:已有 fps={meta.fps}, 当前 fps={args.fps}。")
        ds = LeRobotDataset.__new__(LeRobotDataset)
        ds.meta = meta
        ds.repo_id = meta.repo_id
        ds.root = meta.root
        ds.revision = None
        ds.tolerance_s = 1e-4
        ds.image_writer = None
        ds.batch_encoding_size = 1
        ds.episodes_since_last_encoding = 0
        ds.episodes = None
        ds.hf_dataset = ds.create_hf_dataset()
        ds.image_transforms = None
        ds.delta_timestamps = None
        ds.delta_indices = None
        ds.episode_data_index = None
        ds.video_backend = None
        if args.image_writer_threads:
            ds.start_image_writer(0, args.image_writer_threads)
        ds.episode_buffer = ds.create_episode_buffer()
        print(f"▶ 续采模式:已有 {meta.total_episodes} 条, 新 episode 从该编号继续。", flush=True)
        return ds

    base.prepare_dataset_root(args.root, overwrite=args.overwrite)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=dataset_features(args.camera_configs),
        root=args.root,
        robot_type=args.robot_type,
        use_videos=any(c.storage == "video" for c in args.camera_configs),
        image_writer_threads=args.image_writer_threads,
    )


# ---------------------------------------------------------------- 状态机
@dataclass
class SegmentRecord:
    subtask_instance_id: int
    name: str
    prompt_zh: str
    start_frame: int
    end_frame: int | None = None
    success: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "subtask_instance_id": self.subtask_instance_id,
            "name": self.name,
            "prompt_zh": self.prompt_zh,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "success": self.success,
        }


class StageManager:
    def __init__(self, cfg: TaskConfig, *, min_subtask_seconds: float, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.min_subtask_seconds = float(min_subtask_seconds)
        self.dry_run = dry_run
        self.episode_index = 0
        self.state = STATE_IDLE
        self.pos = 0
        self.pending_start = False
        self._pending_after_frame = None
        self._active: SegmentRecord | None = None
        self.segments: list[SegmentRecord] = []
        self.subtask_started_s = time.monotonic()
        self.last_frame_index = -1

    @property
    def subtask(self) -> SubtaskDef | None:
        if self.state != STATE_RECORDING or self.pos >= len(self.cfg.subtasks):
            return None
        return self.cfg.subtasks[self.pos]

    @property
    def subtask_instance_id(self) -> int:
        return self.pos + 1 if self.state == STATE_RECORDING else 0

    @property
    def wait_save(self) -> bool:
        return self.state == STATE_WAIT_SAVE

    def begin_episode(self, episode_index: int) -> None:
        self.episode_index = int(episode_index)
        self.state = STATE_RECORDING
        self.pos = 0
        self.pending_start = True
        self._pending_after_frame = None
        self._active = None
        self.segments = []
        self.last_frame_index = -1
        self.subtask_started_s = time.monotonic()

    def frame_context(self, frame_index: int, override: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.state == STATE_WAIT_SAVE:
            return dict(subtask_instance_id=0, active_arm="NONE", prompt_index=0,
                        prompt_text="", prompt_text_zh="", subtask_start=0, subtask_end=0,
                        event_code=EVENT_NONE, event_name="NONE",
                        episode_success=-1, failure_type="pending", subtask_success=-1)
        sub = self.subtask
        assert sub is not None
        ctx = dict(
            subtask_instance_id=self.subtask_instance_id,
            active_arm=self.cfg.active_arm,
            prompt_index=sub.subtask_id,
            prompt_text=sub.en,
            prompt_text_zh=sub.zh,
            subtask_start=0,
            subtask_end=0,
            event_code=EVENT_NONE,
            event_name="NONE",
            episode_success=-1,
            failure_type="pending",
            subtask_success=-1,
        )
        if self.pending_start:
            ctx["subtask_start"] = 1
            ctx["event_code"] = EVENT_EPISODE_STARTED if self.last_frame_index < 0 else EVENT_SUBTASK_STARTED
            ctx["event_name"] = EVENT_NAMES[ctx["event_code"]]
        if override:
            ctx.update(override)
        return ctx

    def _begin_segment(self, frame_index: int, sub: SubtaskDef) -> None:
        self._active = SegmentRecord(self.subtask_instance_id, sub.name, sub.zh, frame_index)

    def _finish_segment(self, frame_index: int, success: bool) -> None:
        if self._active is None:
            return
        self._active.end_frame = max(frame_index, self._active.start_frame)
        self._active.success = bool(success)
        self.segments.append(self._active)
        self._active = None

    def after_frame(self, frame_index: int, ctx: dict[str, Any]) -> None:
        self.last_frame_index = frame_index
        if ctx["subtask_start"] and self._active is None and self.state == STATE_RECORDING:
            sub = self.subtask
            if sub is not None:
                self._begin_segment(frame_index, sub)
        self.pending_start = False
        if self._pending_after_frame is not None:
            fn = self._pending_after_frame
            self._pending_after_frame = None
            fn(frame_index)

    def handle_space(self, frame_index: int, now_s: float | None = None) -> tuple[dict[str, Any] | None, str | None]:
        if self.state == STATE_IDLE:
            return None, "当前还没有开始episode。"
        if self.state == STATE_WAIT_SAVE:
            return None, "全部子任务已完成；按 S 保存成功、F 保存失败、G 丢弃。"
        if self.pending_start and self._active is None:
            return None, "当前子任务刚开始，还没有采到起始帧；稍等一帧后再按 Space。"
        elapsed = (now_s if now_s is not None else time.monotonic()) - self.subtask_started_s
        if not self.dry_run and elapsed < self.min_subtask_seconds:
            return None, f"子任务刚开始 {elapsed:.1f}s，忽略这次 Space，防止误连按。"
        sub = self.subtask
        assert sub is not None
        ctx = dict(subtask_end=1, event_code=EVENT_SUBTASK_DONE,
                   event_name=EVENT_NAMES[EVENT_SUBTASK_DONE])

        def advance(done_frame: int) -> None:
            self._finish_segment(done_frame, success=True)
            self.pos += 1
            if self.pos >= len(self.cfg.subtasks):
                self.state = STATE_WAIT_SAVE
                print(f"  ✓ 子任务 {sub.zh} 完成；按 S 成功保存 / F 失败保存 / G 丢弃。", flush=True)
            else:
                self.pending_start = True
                self.subtask_started_s = time.monotonic()
                nxt = self.cfg.subtasks[self.pos]
                print(f"  → 进入子任务 {self.pos + 1}/{len(self.cfg.subtasks)}: {nxt.zh}", flush=True)

        self._pending_after_frame = advance
        return ctx, None

    def close_active(self, frame_index: int, *, success: bool) -> None:
        self._finish_segment(max(frame_index, 0), success=success)

    def status_lines(self, frame_count: int, elapsed_s: float) -> list[str]:
        sub = self.subtask
        zh = sub.zh if sub is not None else ""
        return [
            f"Episode: {self.episode_index:06d}   状态: {self.state}",
            f"子任务: {self.subtask_instance_id}/{len(self.cfg.subtasks)}  {zh}",
            f"帧数: {frame_count}   时长: {elapsed_s:.1f}s",
            "按键: Space=本子任务完成并推进  S=成功保存  F=失败保存  G=丢弃  Q=退出",
        ]


# ---------------------------------------------------------------- 帧字段
def frame_fields(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "subtask_instance_id": int_feature(ctx["subtask_instance_id"]),
        "active_arm": ctx["active_arm"],
        "prompt_index": int_feature(ctx["prompt_index"]),
        "prompt_text": ctx["prompt_text"],
        "prompt_text_zh": ctx["prompt_text_zh"],
        "subtask_start": int_feature(ctx["subtask_start"]),
        "subtask_end": int_feature(ctx["subtask_end"]),
        "event_code": int_feature(ctx["event_code"]),
        "event_name": ctx["event_name"],
        "episode_success": int_feature(ctx["episode_success"]),
        "failure_type": ctx["failure_type"],
        "subtask_success": int_feature(ctx["subtask_success"]),
    }


# ---------------------------------------------------------------- 双臂/相机
class MockDualNeroArms:
    def __init__(self) -> None:
        self.start_time = time.monotonic()

    def enable(self, *, speed_percent: int = 40, timeout: float = 5.0) -> None:
        del speed_percent, timeout

    def enter_drag_teach(self) -> None:
        print("[mock dual] enter_drag_teach")

    def exit_drag_teach(self) -> None:
        print("[mock dual] exit_drag_teach")

    def close(self) -> None:
        print("[mock dual] close")

    def read_state(self, *, use_gripper: bool = True) -> list[float]:
        t = time.monotonic() - self.start_time
        left = [0.2 * np.sin(t + i * 0.2) for i in range(7)]
        right = [0.05 * np.sin(i * 0.2) for i in range(7)]
        if use_gripper:
            return left + [0.035 + 0.015 * np.sin(t)] + right + [0.025]
        return left + right


def build_arms(args: argparse.Namespace):
    if args.mock:
        return MockDualNeroArms()
    from nero_control.dual_nero_driver import DualNeroArms, NeroArmConfig

    return DualNeroArms(
        NeroArmConfig(name="left", can_channel=args.left_can, firmware=args.left_firmware,
                      can_interface=args.can_interface, bitrate=args.can_bitrate,
                      timeout=args.can_timeout),
        NeroArmConfig(name="right", can_channel=args.right_can, firmware=args.right_firmware,
                      can_interface=args.can_interface, bitrate=args.can_bitrate,
                      timeout=args.can_timeout),
    )


def read_state_frame(arms, args: argparse.Namespace, last_state: np.ndarray | None) -> tuple[np.ndarray, bool]:
    try:
        state = list(arms.read_state(use_gripper=True))
        if len(state) != 16:
            raise RuntimeError(f"dual arms: expected 16 state values, got {len(state)}")
        return np.asarray(state, dtype=np.float32), True
    except Exception as exc:  # noqa: BLE001
        if last_state is not None:
            return last_state.copy(), False
        raise RuntimeError(f"dual arm feedback unavailable: {exc}") from exc


# ---------------------------------------------------------------- 交互
def choose_failure_type(failure_types: list[str]) -> str:
    choices = [ft for ft in failure_types if ft != "none"]
    print("\n请选择失败类型，输入序号；直接回车=manual_abort")
    for i, ft in enumerate(choices, 1):
        print(f"  {i}. {ft}")
    value = input("> ").strip()
    if not value:
        return "manual_abort"
    if value.isdigit() and 1 <= int(value) <= len(choices):
        return choices[int(value) - 1]
    if value in choices:
        return value
    print(f"  未识别 {value!r}，记录为 other。", flush=True)
    return "other"


def append_jsonl(path: Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 采集主循环
def record_episode(args: argparse.Namespace, cfg: TaskConfig, dataset, arms,
                   cameras: CameraSet, keys, episode_index: int):
    manager = StageManager(cfg, min_subtask_seconds=args.min_subtask_seconds,
                           dry_run=args.mock)
    manager.begin_episode(episode_index)
    period_s = 1.0 / args.fps
    max_frames = max(1, int(round(args.episode_seconds * args.fps)))
    start_s = time.monotonic()
    last_state: np.ndarray | None = None
    stale = 0
    max_stale = max(1, int(round(args.max_stale_state_seconds * args.fps)))
    frame_index = 0
    action = "timeout_failure"
    first_s_pending = False

    while frame_index < max_frames:
        target_s = start_s + frame_index * period_s
        now_s = time.monotonic()
        if now_s < target_s:
            time.sleep(target_s - now_s)

        key = (keys.get(0.0) or "").lower()
        override = None
        if key == " ":
            override, warning = manager.handle_space(frame_index, now_s=time.monotonic())
            if warning:
                print(f"\n  ⚠ {warning}", flush=True)
        elif key == "s":
            if manager.wait_save:
                action = "save_success"
                break
            if not first_s_pending:
                first_s_pending = True
                print("\n  ⚠ 还有子任务未完成；再按一次 S 才把本条保存为不完整成功数据。", flush=True)
            else:
                action = "save_success"
                break
        elif key == "f":
            action = "save_failure"
            break
        elif key == "g":
            action = "discard"
            break
        elif key == "q":
            print("\n当前episode还在录制。按 S保存成功，F保存失败，G丢弃；再按 Q 强制结束。", flush=True)
            k2 = (keys.get(0.0) or "").lower()
            for _ in range(100):
                if k2 in {"s", "f", "g", "q"}:
                    break
                time.sleep(0.05)
                k2 = (keys.get(0.0) or "").lower()
            action = {"s": "save_success", "f": "save_failure", "g": "discard", "q": "force_quit"}[k2 or "q"]
            break

        ctx = manager.frame_context(frame_index, override)
        state, ok = read_state_frame(arms, args, last_state)
        if ok:
            stale = 0
        else:
            stale += 1
            if stale == 1 or stale % args.fps == 0:
                print(f"\n  ⚠ 机器人状态读取失败，复用上一帧 ({stale}/{max_stale})", flush=True)
            if stale > max_stale:
                raise RuntimeError("机器人状态连续读取失败过久，已停止本条采集。")
        last_state = state
        frames = cameras.read_frames()
        frame = {"observation.state": state, "action": state.copy()}
        frame.update(frame_fields(ctx))
        frame.update(frames)
        dataset.add_frame(frame, task=ctx["prompt_text"], timestamp=frame_index / args.fps)
        manager.after_frame(frame_index, ctx)
        frame_index += 1

        if frame_index == 1 or frame_index % args.fps == 0 or ctx["event_code"] != EVENT_NONE:
            print("\n" + "\n".join(manager.status_lines(frame_index, time.monotonic() - start_s)), flush=True)

        if manager.wait_save:
            while True:
                k2 = (keys.get(0.3) or "").lower()
                if k2 == "s":
                    action = "save_success"
                    break
                if k2 == "f":
                    action = "save_failure"
                    break
                if k2 == "g":
                    action = "discard"
                    break
                if k2 == "q":
                    action = "force_quit"
                    break
            break
    else:
        print("\n  ⚠ 达到episode时长上限，按失败保存(timeout)。", flush=True)
        action = "save_failure"

    return manager, frame_index, action


def label_subtask_success(dataset, segments: list[SegmentRecord]) -> None:
    size = int(dataset.episode_buffer["size"])
    arr = dataset.episode_buffer["subtask_success"]
    for seg in segments:
        end = size if seg.end_frame is None else min(seg.end_frame + 1, size)
        for i in range(seg.start_frame, end):
            arr[i] = int_feature(1 if seg.success else 0)


def set_outcome_fields(dataset, *, success: bool, failure_type: str) -> None:
    size = int(dataset.episode_buffer["size"])
    dataset.episode_buffer["episode_success"] = [int_feature(1 if success else 0) for _ in range(size)]
    dataset.episode_buffer["failure_type"] = [failure_type for _ in range(size)]


def save_episode_result(args: argparse.Namespace, dataset, manager: StageManager,
                        *, success: bool, failure_type: str) -> None:
    set_outcome_fields(dataset, success=success, failure_type=failure_type)
    label_subtask_success(dataset, manager.segments)
    print("  保存中: LeRobot Parquet/视频/元数据编码...", flush=True)
    dataset.save_episode()
    annotations = args.root / "annotations"
    append_jsonl(
        annotations / "episode_outcomes.jsonl",
        {
            "episode_index": manager.episode_index,
            "episode_success": bool(success),
            "failure_type": failure_type,
            "frames": int(dataset.episode_buffer["size"]) if False else int(manager.last_frame_index + 1),
            "subtasks": [seg.as_dict() for seg in manager.segments],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    write_sidecars(args, manager.cfg)
    print(f"  ✓ 已存 episode {manager.episode_index:06d}  帧 {manager.last_frame_index + 1}  "
          f"成功={success} 失败类型={failure_type}", flush=True)


def discard_episode(dataset) -> None:
    if dataset.episode_buffer is not None and int(dataset.episode_buffer["size"]) > 0:
        dataset.clear_episode_buffer()
    print("  ✗ 已丢弃本条。", flush=True)


def write_sidecars(args: argparse.Namespace, cfg: TaskConfig) -> None:
    annotations = args.root / "annotations"
    append_jsonl(
        annotations / "collection_snapshot.jsonl",
        {
            "repo_id": args.repo_id,
            "root": str(args.root),
            "subtasks": [dict(subtask_id=s.subtask_id, name=s.name, zh=s.zh, en=s.en)
                         for s in cfg.subtasks],
            "failure_types": cfg.failure_types,
            "left_can_serial": cfg.get("can.left_serial"),
            "right_can_serial": cfg.get("can.right_serial"),
            "camera_serials": cfg.get("cameras.rgb.serials"),
        },
    )


# ---------------------------------------------------------------- 参数
def parse_args(default_config: Path) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RLinf 左臂抓盒对照数据采集(成功/失败)")
    ap.add_argument("--config", type=Path, default=default_config)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--timestamp-root", action=argparse.BooleanOptionalAction, default=None,
                    help="默认按配置(开)=每次运行生成新时间戳文件夹; --no-timestamp-root 可续采指定 --root")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--episode-seconds", type=float, default=None)
    ap.add_argument("--num-episodes", type=int, default=0, help="0=不限, 采集到手动退出")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--image-writer-threads", type=int, default=None)
    ap.add_argument("--min-subtask-seconds", type=float, default=0.8)
    ap.add_argument("--max-stale-state-seconds", type=float, default=0.5)
    ap.add_argument("--dry-run-state-machine", action="store_true", help="只测状态机, 不碰硬件")

    ap.add_argument("--left-can", default=None, help="默认 auto=按左臂适配器序列号解析")
    ap.add_argument("--right-can", default=None, help="默认 auto=按右臂适配器序列号解析")
    ap.add_argument("--left-firmware", default=None)
    ap.add_argument("--right-firmware", default=None)
    ap.add_argument("--can-interface", default=None)
    ap.add_argument("--can-bitrate", type=int, default=None)
    ap.add_argument("--can-timeout", type=float, default=None)
    ap.add_argument("--speed-percent", type=int, default=None)
    ap.add_argument("--enable", dest="enable", action="store_true")
    ap.add_argument("--no-enable", dest="enable", action="store_false")
    ap.set_defaults(enable=None)
    ap.add_argument("--confirm-enable-risk", action="store_true")
    ap.add_argument("--drag-teach", dest="drag_teach", action="store_true")
    ap.add_argument("--no-drag-teach", dest="drag_teach", action="store_false")
    ap.set_defaults(drag_teach=None)
    ap.add_argument("--confirm-drag-teach-risk", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-cameras", action="store_true")

    ap.add_argument("--left-wrist-camera", default=None, help="默认 auto=按左腕相机序列号解析")
    ap.add_argument("--right-wrist-camera", default=None, help="默认 auto=按右腕相机序列号解析")
    ap.add_argument("--third-camera", default=None, help="默认 auto=按第三视角相机序列号解析")
    ap.add_argument("--camera-width", type=int, default=None)
    ap.add_argument("--camera-height", type=int, default=None)
    ap.add_argument("--camera-fps", type=int, default=None)
    ap.add_argument("--camera-fourcc", default=None)
    ap.add_argument("--camera-storage", choices=("video", "image"), default=None)
    ap.add_argument("--apply-v4l2", type=int, default=None, help="1=按配置应用曝光(默认1)")
    return ap.parse_args()


def apply_config_defaults(args: argparse.Namespace, cfg: TaskConfig) -> argparse.Namespace:
    if args.dry_run_state_machine:
        args.mock = True
        args.mock_cameras = True
    args.repo_id = args.repo_id or cfg.get("dataset.repo_id")
    args.root = Path(args.root).expanduser() if args.root else Path(cfg.get("dataset.root")).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    want_timestamp = bool(cfg.get("dataset.timestamp_root", False)) if args.timestamp_root is None \
        else bool(args.timestamp_root)
    if want_timestamp and not args.dry_run_state_machine:
        args.root = args.root.parent / f"{args.root.name}_{stamp}"
        args.repo_id = f"local/{args.root.name}"
        args.resume = False
    args.fps = args.fps or int(cfg.get("dataset.fps", 15))
    args.episode_seconds = args.episode_seconds or float(cfg.get("dataset.episode_seconds", 180))
    args.resume = bool(cfg.get("dataset.resume", False)) if args.resume is None else bool(args.resume)
    args.image_writer_threads = args.image_writer_threads if args.image_writer_threads is not None \
        else int(cfg.get("dataset.image_writer_threads", 4))
    args.robot_type = str(cfg.get("dataset.robot_type", "dual_nero_left_box_pick"))

    args.can_bitrate = args.can_bitrate or int(cfg.get("can.bitrate", 1_000_000))
    args.can_interface = args.can_interface or cfg.get("can.interface", "socketcan")
    args.can_timeout = args.can_timeout or float(cfg.get("can.timeout", 1.0))
    args.left_firmware = args.left_firmware or cfg.get("can.left_firmware", "v120")
    args.right_firmware = args.right_firmware or cfg.get("can.right_firmware", "v120")
    args.speed_percent = args.speed_percent or int(cfg.get("safety.speed_percent", 5))
    args.enable = bool(cfg.get("safety.enable", False)) if args.enable is None else bool(args.enable)
    args.drag_teach = bool(cfg.get("safety.drag_teach", False)) if args.drag_teach is None else bool(args.drag_teach)
    args.apply_v4l2 = int(cfg.get("cameras.rgb.apply_v4l2", 1)) if args.apply_v4l2 is None else args.apply_v4l2

    if args.enable and not args.confirm_enable_risk:
        raise ValueError("--enable requires --confirm-enable-risk")
    if args.drag_teach and args.enable:
        raise ValueError("不要同时使用 --drag-teach 和 --enable。")
    if args.drag_teach and not args.confirm_drag_teach_risk:
        raise ValueError("--drag-teach requires --confirm-drag-teach-risk")

    # ---- CAN: 按序列号解析(除非命令行显式指定) ----
    left_serial = cfg.get("can.left_serial")
    right_serial = cfg.get("can.right_serial")
    if not args.mock:
        args.left_can = args.left_can or resolve_can_by_serial(left_serial) or None
        args.right_can = args.right_can or resolve_can_by_serial(right_serial) or None
        if args.left_can is None or args.right_can is None:
            raise RuntimeError(
                f"CAN 适配器未按序列号找到(左={left_serial}, 右={right_serial})。"
                f"检查适配器是否插好/上电。可用 --left-can/--right-can 手动指定。")
        if args.left_can == args.right_can:
            raise RuntimeError(f"左右臂解析到了同一个 CAN({args.left_can}) —— 检查适配器序列号。")
        ensure_can_up(args.left_can, args.can_bitrate)
        ensure_can_up(args.right_can, args.can_bitrate)
    else:
        args.left_can = args.left_can or "can1"
        args.right_can = args.right_can or "can0"

    # ---- 相机: 按序列号解析 ----
    ser = cfg.get("cameras.rgb.serials", {})
    def cam_or(arg_serial: str | None, name: str) -> str | None:
        if args.mock_cameras:
            return None
        if arg_serial is not None:
            return arg_serial
        want = ser.get(name)
        if not want:
            raise RuntimeError(f"配置缺 cameras.rgb.serials.{name}")
        dev = resolve_video_by_serial(want)
        if dev is None:
            raise RuntimeError(f"相机 {name}(serial={want}) 未找到对应 /dev/videoN, 检查 USB 连接。")
        return dev

    args.left_wrist_camera = cam_or(args.left_wrist_camera, "left_wrist")
    args.right_wrist_camera = cam_or(args.right_wrist_camera, "right_wrist")
    args.third_camera = cam_or(args.third_camera, "third_view")

    args.camera_width = args.camera_width or int(cfg.get("cameras.rgb.width", 640))
    args.camera_height = args.camera_height or int(cfg.get("cameras.rgb.height", 480))
    args.camera_fps = args.camera_fps or int(cfg.get("cameras.rgb.fps", 15))
    args.camera_fourcc = args.camera_fourcc or cfg.get("cameras.rgb.fourcc", "MJPG")
    args.camera_storage = args.camera_storage or cfg.get("cameras.rgb.storage", "video")

    if args.fps <= 0 or args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("fps/camera width/height must be positive")

    def mk_camera(name: str, device: str | None) -> CameraConfig:
        return CameraConfig(name=name, device=device, width=args.camera_width,
                            height=args.camera_height, fps=args.camera_fps,
                            fourcc=args.camera_fourcc, storage=args.camera_storage,
                            mock=args.mock_cameras)

    args.camera_configs = [
        mk_camera("left_wrist", args.left_wrist_camera),
        mk_camera("right_wrist", args.right_wrist_camera),
        mk_camera("third_view", args.third_camera),
    ]
    return args


def print_startup(args: argparse.Namespace, cfg: TaskConfig) -> None:
    print("============ RLinf 左臂抓盒对照数据采集 ============")
    print(f"配置: {args.config}")
    print(f"Dataset: {args.root}")
    print(f"子任务({len(cfg.subtasks)} 个):")
    for s in cfg.subtasks:
        print(f"  {s.subtask_id}. [{s.name}] {s.zh}")
        print(f"     en: {s.en}")
    print(f"失败类型: {cfg.failure_types}")
    print(f"FPS: {args.fps}Hz   单条上限: {args.episode_seconds:.0f}s")
    print("相机(按序列号解析):")
    for c in args.camera_configs:
        print(f"  {c.name}: {c.device}  {c.width}x{c.height}@{c.fps} {c.storage}")
    print(f"CAN: 左={args.left_can} 右={args.right_can}")
    print(f"拖动示教: {'是(零力, 双臂)' if args.drag_teach else '否(右臂主臂模式请由外部设置)'}")
    print("====================================================")


def run_state_machine_dry_run(cfg: TaskConfig, args: argparse.Namespace) -> None:
    manager = StageManager(cfg, min_subtask_seconds=0.0, dry_run=True)
    manager.begin_episode(0)
    frame = 0
    trace = [STATE_IDLE]

    def tick(override: dict[str, Any] | None = None) -> None:
        nonlocal frame
        ctx = manager.frame_context(frame, override)
        manager.after_frame(frame, ctx)
        frame += 1

    tick()
    while not manager.wait_save:
        ctx, warning = manager.handle_space(frame, now_s=time.monotonic() + 10.0)
        if warning:
            raise RuntimeError(f"dry-run 非法警告: {warning}")
        tick(ctx)
        if not manager.wait_save:
            trace.append(manager.subtask.name if manager.subtask else "?")
            tick()
    trace.append(STATE_WAIT_SAVE)
    if len(manager.segments) != len(cfg.subtasks):
        raise RuntimeError(f"dry-run失败: segments={len(manager.segments)} != {len(cfg.subtasks)}")
    print("dry-run 状态机通过:")
    print("  " + " -> ".join(trace) + " -> S/F/G")
    for seg in manager.segments:
        print(f"  inst={seg.subtask_instance_id} {seg.name} frames={seg.start_frame}-{seg.end_frame}")


def run_collection(args: argparse.Namespace, cfg: TaskConfig) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("采集需要交互式终端；状态机测试用 --dry-run-state-machine。")
    dataset = create_dataset(args, cfg)
    print("初始化 CAN/机器人接口...", flush=True)
    arms = build_arms(args)
    cameras = CameraSet(args.camera_configs)
    saved = 0
    try:
        print("打开三路 RGB 相机...", flush=True)
        cameras.open()
        print("  ✓ RGB 相机已打开", flush=True)
        if args.apply_v4l2 and not args.mock_cameras:
            exp = cfg.get("cameras.rgb.exposure", {})
            for c in args.camera_configs:
                if c.name in exp and c.device:
                    apply_exposure(c.device, exp[c.name])
                    print(f"  ✓ {c.name} 曝光已应用", flush=True)
        if args.enable:
            print("使能机械臂...", flush=True)
            arms.enable(speed_percent=args.speed_percent, timeout=5.0)
        elif args.drag_teach:
            print("进入拖动示教(零力)...", flush=True)
            arms.enter_drag_teach()

        while args.num_episodes <= 0 or saved < args.num_episodes:
            with base.KeyPad() as keys:
                ep_index = int(dataset.episode_buffer["episode_index"])
                print(f"\n=== IDLE: 准备采集 episode {ep_index:06d} ===")
                print("摆好机械臂/盒子后按 Space 开始；Q 结束程序。", flush=True)
                keys.drain()
                start_key = None
                while True:
                    k = (keys.get(0.3) or "").lower()
                    if k == " ":
                        start_key = " "
                        break
                    if k == "q":
                        start_key = "q"
                        break
                if start_key == "q":
                    break
                manager, frames, action = record_episode(args, cfg, dataset, arms, cameras,
                                                         keys, ep_index)

            if action == "force_quit":
                print("  ⚠ 强制结束；当前缓存未保存。", flush=True)
                break
            if action == "discard":
                discard_episode(dataset)
                continue
            if action == "save_failure":
                failure_type = choose_failure_type(cfg.failure_types)
                save_episode_result(args, dataset, manager, success=False, failure_type=failure_type)
                saved += 1
                continue
            if action == "save_success":
                all_ok = all(seg.success for seg in manager.segments) and \
                    len(manager.segments) == len(cfg.subtasks)
                save_episode_result(args, dataset, manager, success=all_ok,
                                    failure_type="none")
                saved += 1
                continue
    finally:
        try:
            if args.drag_teach:
                arms.exit_drag_teach()
        except Exception:  # noqa: BLE001
            pass
        cameras.close()
        arms.close()
    return saved


def run(default_config: Path) -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECTTIMEOUT", "1")
    base.add_local_import_paths()   # 关键: 把 third_party/lerobot_v21/src 加进 sys.path(import lerobot 靠它)
    args = parse_args(default_config)
    cfg = load_config(args.config)
    args = apply_config_defaults(args, cfg)

    if args.dry_run_state_machine:
        run_state_machine_dry_run(cfg, args)
        return

    print_startup(args, cfg)
    saved = run_collection(args, cfg)
    print(f"\nDone. Saved {saved} episodes.")
    print(f"Dataset: {args.root}")
