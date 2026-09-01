#!/usr/bin/env python3
"""Nero Stage5/6 flap-closing collection for LeRobot v2.1.

This is a copy of collect_box_packing_closing_refined.py retargeted at the
Stage5/Stage6 flap-closing segment. The carton is already rotated into the
Stage5 pose with both flaps open; this collector records the 7 micro subtasks
that brace the front/rear flaps, close the left flap, then close the right
flap. The original collectors are unchanged.

Seven micro subtasks (stage id 5x = macro Stage5, 6x = macro Stage6):
  51 right arm lifts over the flaps to the carton top surface
  52 right gripper opens and braces the front and rear flaps
  53 left arm lifts the left flap to near vertical
  54 left arm folds the left flap closed
  55 right arm retracts
  61 right arm lifts the right flap to near vertical
  62 right arm folds the right flap closed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HEZI_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DATA_ROOT = HEZI_ROOT.parent
NERO_ROOT = LEROBOT_DATA_ROOT.parent
TOOLS_ROOT = LEROBOT_DATA_ROOT / "lerobot_tools"
DEFAULT_CONFIG = HEZI_ROOT / "configs" / "box_packing_stage56_task.yaml"

sys.path.insert(0, str(TOOLS_ROOT))
import collect_dual_nero_rgbd as base_collector  # noqa: E402
from camera_rgbd_utils import (  # noqa: E402
    CameraConfig,
    CameraSet,
    DepthCameraConfig,
    DepthCameraSet,
    camera_features,
    depth_camera_features,
)


STAGE_IDLE = 0
STAGE_WAIT_SAVE = 7

EVENT_NONE = 0
EVENT_EPISODE_STARTED = 1
EVENT_SUBTASK_STARTED = 2
EVENT_MOUSE_BOX_PLACED = 3
EVENT_EARPHONE_BOX_PLACED = 4
EVENT_STAGE_COMPLETED = 5
EVENT_EPISODE_SUCCESS = 6
EVENT_EPISODE_FAILED = 7
EVENT_EPISODE_DISCARDED = 8

EVENT_NAMES = {
    EVENT_NONE: "NONE",
    EVENT_EPISODE_STARTED: "EPISODE_STARTED",
    EVENT_SUBTASK_STARTED: "SUBTASK_STARTED",
    EVENT_MOUSE_BOX_PLACED: "MOUSE_BOX_PLACED",
    EVENT_EARPHONE_BOX_PLACED: "EARPHONE_BOX_PLACED",
    EVENT_STAGE_COMPLETED: "STAGE_COMPLETED",
    EVENT_EPISODE_SUCCESS: "EPISODE_SUCCESS",
    EVENT_EPISODE_FAILED: "EPISODE_FAILED",
    EVENT_EPISODE_DISCARDED: "EPISODE_DISCARDED",
}


@dataclass(frozen=True)
class PromptDef:
    prompt_index: int
    zh: str
    en: str


@dataclass(frozen=True)
class StageDef:
    stage_id: int
    name: str
    prompt_index: int
    repeat: int
    display: str


@dataclass
class SegmentRecord:
    episode_index: int
    stage_id: int
    stage_name: str
    prompt_index: int
    prompt_text: str
    subtask_instance_id: int
    start_frame: int
    end_frame: int | None = None
    completion_event: str = ""
    success: bool | None = None
    planned_mouse_box_count: int = 0
    planned_earphone_box_count: int = 0
    planned_subtask_count: int = 0
    collection_plan: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "prompt_index": self.prompt_index,
            "prompt_text": self.prompt_text,
            "subtask_instance_id": self.subtask_instance_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "completion_event": self.completion_event,
            "success": self.success,
            "planned_mouse_box_count": self.planned_mouse_box_count,
            "planned_earphone_box_count": self.planned_earphone_box_count,
            "planned_subtask_count": self.planned_subtask_count,
            "collection_plan": self.collection_plan,
        }


@dataclass
class FrameContext:
    stage_id: int
    stage_name: str
    prompt_index: int
    prompt_text: str
    prompt_text_zh: str
    subtask_instance_id: int
    subtask_start: int = 0
    subtask_end: int = 0
    event_code: int = EVENT_NONE
    event_name: str = "NONE"
    keyframe_flag: int = 0
    mouse_box_success_count: int = 0
    earphone_box_success_count: int = 0
    successful_place_count: int = 0
    episode_success: int = -1
    failure_type: str = "pending"
    planned_mouse_box_count: int = 0
    planned_earphone_box_count: int = 0
    planned_subtask_count: int = 0
    collection_plan: str = ""


@dataclass
class KeyResult:
    action: str | None = None
    frame_context: FrameContext | None = None
    warning: str | None = None


@dataclass
class EpisodeResult:
    action: str
    frames: int
    episode_index: int
    temp_dir: Path
    segments: list[dict[str, Any]]
    events: list[dict[str, Any]]
    failure_type: str = "none"
    force_quit: bool = False
    planned_mouse_box_count: int = 0
    planned_earphone_box_count: int = 0
    planned_subtask_count: int = 0
    collection_plan: str = ""


@dataclass
class LongTaskConfig:
    raw: dict[str, Any]
    stages: dict[int, StageDef]
    prompts: dict[int, PromptDef]
    event_names: dict[int, str]
    failure_types: list[str]


def load_config(path: Path) -> LongTaskConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = {
        int(item["prompt_index"]): PromptDef(
            prompt_index=int(item["prompt_index"]),
            zh=str(item["zh"]),
            en=str(item["en"]),
        )
        for item in raw["prompts"]
    }
    stages = {
        int(item["id"]): StageDef(
            stage_id=int(item["id"]),
            name=str(item["name"]),
            prompt_index=int(item["prompt_index"]),
            repeat=int(item.get("repeat", 1)),
            display=str(item.get("display", item["name"])),
        )
        for item in raw["stages"]
    }
    events = {int(k): str(v) for k, v in raw.get("events", EVENT_NAMES).items()}
    failure_types = [str(v) for v in raw.get("failure_types", ["none", "other"])]
    return LongTaskConfig(raw=raw, stages=stages, prompts=prompts, event_names=events, failure_types=failure_types)


def apply_repeat_overrides(cfg: LongTaskConfig, *, mouse_boxes: int | None, earphone_boxes: int | None) -> LongTaskConfig:
    if mouse_boxes is None and earphone_boxes is None:
        return cfg
    if 1 not in cfg.stages or 2 not in cfg.stages:
        print("  [WARN] 当前是Stage5/6合页采集配置，没有Stage 1/2；忽略 --mouse-boxes/--earphone-boxes。", flush=True)
        return cfg
    stages = dict(cfg.stages)
    raw = dict(cfg.raw)
    counts = dict(raw.get("counts", {}))
    if mouse_boxes is not None:
        if mouse_boxes <= 0:
            raise ValueError("--mouse-boxes must be positive")
        stages[1] = replace(stages[1], repeat=int(mouse_boxes))
        counts["mouse_boxes"] = int(mouse_boxes)
    if earphone_boxes is not None:
        if earphone_boxes <= 0:
            raise ValueError("--earphone-boxes must be positive")
        stages[2] = replace(stages[2], repeat=int(earphone_boxes))
        counts["earphone_boxes"] = int(earphone_boxes)
    raw["counts"] = counts
    raw["stages"] = [
        {
            "id": stage.stage_id,
            "name": stage.name,
            "prompt_index": stage.prompt_index,
            "repeat": stage.repeat,
            "display": stage.display,
        }
        for stage in stages.values()
    ]
    return LongTaskConfig(raw=raw, stages=stages, prompts=cfg.prompts, event_names=cfg.event_names, failure_types=cfg.failure_types)


def as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def int_feature(value: int) -> np.ndarray:
    return np.asarray([int(value)], dtype=np.int64)


def custom_task_features() -> dict[str, dict[str, Any]]:
    return {
        "stage_id": {"dtype": "int64", "shape": (1,), "names": None},
        "stage_name": {"dtype": "string", "shape": (1,), "names": None},
        "prompt_index": {"dtype": "int64", "shape": (1,), "names": None},
        "prompt_text": {"dtype": "string", "shape": (1,), "names": None},
        "prompt_text_zh": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_instance_id": {"dtype": "int64", "shape": (1,), "names": None},
        "subtask_start": {"dtype": "int64", "shape": (1,), "names": None},
        "subtask_end": {"dtype": "int64", "shape": (1,), "names": None},
        "event_code": {"dtype": "int64", "shape": (1,), "names": None},
        "event_name": {"dtype": "string", "shape": (1,), "names": None},
        "keyframe_flag": {"dtype": "int64", "shape": (1,), "names": None},
        "mouse_box_success_count": {"dtype": "int64", "shape": (1,), "names": None},
        "earphone_box_success_count": {"dtype": "int64", "shape": (1,), "names": None},
        "successful_place_count": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_success": {"dtype": "int64", "shape": (1,), "names": None},
        "failure_type": {"dtype": "string", "shape": (1,), "names": None},
        "planned_mouse_box_count": {"dtype": "int64", "shape": (1,), "names": None},
        "planned_earphone_box_count": {"dtype": "int64", "shape": (1,), "names": None},
        "planned_subtask_count": {"dtype": "int64", "shape": (1,), "names": None},
        "collection_plan": {"dtype": "string", "shape": (1,), "names": None},
    }


def dataset_features(camera_configs: list, depth_camera_configs: list, *, use_gripper: bool) -> dict[str, dict[str, Any]]:
    names = base_collector.state_names(use_gripper=use_gripper)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    features.update(camera_features(camera_configs))
    features.update(depth_camera_features(depth_camera_configs))
    features.update(custom_task_features())
    return features


def quarantine_uncommitted_episode_artifacts(root: Path, meta: Any, episode_index: int) -> None:
    """Move leftovers from a failed save before reusing the same episode index."""

    candidates: list[Path] = []
    data_path = root / meta.get_data_file_path(episode_index)
    if data_path.exists():
        candidates.append(data_path)
    for key in meta.video_keys:
        video_path = root / meta.get_video_file_path(episode_index, key)
        if video_path.exists():
            candidates.append(video_path)
        image_dir = root / "images" / key / f"episode_{episode_index:06d}"
        if image_dir.exists():
            candidates.append(image_dir)
    raw_dir = root / "depth_raw"
    if raw_dir.exists():
        for path in raw_dir.glob(f"*/episode_{episode_index:06d}"):
            candidates.append(path)
    if not candidates:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    quarantine = root / "incomplete" / f"orphaned_artifacts_episode_{episode_index:06d}_{stamp}"
    print(
        f"⚠ 发现未提交的 episode_{episode_index:06d} 残留，移动到 {quarantine}",
        flush=True,
    )
    for src in candidates:
        dst = quarantine / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def create_dataset(args: argparse.Namespace):
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"Expected LeRobot codebase v2.1, got {CODEBASE_VERSION}")

    if args.resume and not args.overwrite and args.root.exists() and not base_collector._dataset_is_empty(args.root):
        meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
        quarantine_uncommitted_episode_artifacts(args.root, meta, meta.total_episodes)
        wanted = dataset_features(args.camera_configs, args.depth_camera_configs, use_gripper=args.use_gripper)
        for key, spec in wanted.items():
            if key not in meta.features:
                raise RuntimeError(f"续采失败:已有数据缺少特征 {key}。换 --repo-id 或 --root 新采。")
            if tuple(meta.features[key]["shape"]) != tuple(spec["shape"]):
                raise RuntimeError(
                    f"续采失败:{key} shape {tuple(meta.features[key]['shape'])} != {tuple(spec['shape'])}。"
                )
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
        threads = args.image_writer_threads if (args.camera_configs or args.depth_camera_configs) else 0
        if threads:
            ds.start_image_writer(0, threads)
        ds.episode_buffer = ds.create_episode_buffer()
        print(f"▶ 续采模式:已有 {meta.total_episodes} 条,新 episode 从该编号继续。", flush=True)
        return ds

    base_collector.prepare_dataset_root(args.root, overwrite=args.overwrite)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=dataset_features(args.camera_configs, args.depth_camera_configs, use_gripper=args.use_gripper),
        root=args.root,
        robot_type=args.robot_type,
        use_videos=any(cfg.storage == "video" for cfg in args.camera_configs + args.depth_camera_configs),
        image_writer_threads=args.image_writer_threads if (args.camera_configs or args.depth_camera_configs) else 0,
    )


def _cfg_get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_camera_configs(args: argparse.Namespace) -> list[CameraConfig]:
    return [
        CameraConfig(
            name="left_wrist",
            device=None if args.mock_cameras else args.left_wrist_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc=args.camera_fourcc,
            storage=args.camera_storage,
            mock=args.mock_cameras,
        ),
        CameraConfig(
            name="right_wrist",
            device=None if args.mock_cameras else args.right_wrist_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc=args.camera_fourcc,
            storage=args.camera_storage,
            mock=args.mock_cameras,
        ),
        CameraConfig(
            name="third_view",
            device=None if args.mock_cameras else args.third_camera,
            width=args.third_camera_width or args.camera_width,
            height=args.third_camera_height or args.camera_height,
            fps=args.third_camera_fps or args.camera_fps,
            fourcc=args.camera_fourcc,
            storage=args.camera_storage,
            mock=args.mock_cameras,
        ),
    ]


def build_depth_camera_configs(args: argparse.Namespace) -> list[DepthCameraConfig]:
    if not args.use_depth:
        return []
    devices = {
        "third_view": args.third_depth_serial if args.depth_backend == "orbbec_v1" else args.third_depth_index,
    }
    if args.depth_backend == "v4l2":
        devices["third_view"] = args.third_depth_device
    configs: list[DepthCameraConfig] = []
    for name in args.depth_camera_names:
        if name != "third_view":
            raise ValueError("本长程任务当前只支持第三视角深度: --depth-camera-names third_view")
        configs.append(
            DepthCameraConfig(
                name=name,
                device=None if args.mock_cameras else devices[name],
                width=args.depth_width,
                height=args.depth_height,
                fps=args.depth_fps,
                storage=args.depth_storage,
                backend=args.depth_backend,
                min_mm=args.depth_min_mm,
                max_mm=args.depth_max_mm,
                colorize=args.depth_colorize,
                open_timeout=args.depth_open_timeout,
                bridge_path=args.orbbec_v1_bridge,
                mock=args.mock_cameras or args.depth_backend == "mock",
            )
        )
    return configs


class KeyboardController:
    def __init__(self) -> None:
        self._pad = base_collector.KeyPad()
        self.enabled = self._pad.enabled

    def __enter__(self) -> "KeyboardController":
        self._pad.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._pad.__exit__(*exc)

    def get(self, timeout: float | None = 0.0) -> str | None:
        key = self._pad.get(timeout)
        if key is None:
            return None
        if key == " ":
            return "SPACE"
        key = key.lower()
        if key in {"1", "2", "3"}:
            return key
        if key in {"n", "s", "f", "g", "q"}:
            return key.upper()
        return None

    def drain(self) -> None:
        self._pad.drain()


class StageManager:
    def __init__(self, cfg: LongTaskConfig, *, min_stage_seconds: float = 0.8, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.min_stage_seconds = float(min_stage_seconds)
        self.dry_run = dry_run
        self.episode_index = 0
        self.stage_id = STAGE_IDLE
        self.subtask_instance_id = 0
        self.mouse_box_success_count = 0
        self.earphone_box_success_count = 0
        self.successful_place_count = 0
        self.stage_completed = False
        self.pending_start_next_frame = False
        self._pending_after_frame = None
        self._active_segment: SegmentRecord | None = None
        self.segments: list[SegmentRecord] = []
        self.events: list[dict[str, Any]] = []
        self.last_frame_index = -1
        self.stage_entered_s = time.monotonic()

    @property
    def recording(self) -> bool:
        return self.stage_id not in (STAGE_IDLE, STAGE_WAIT_SAVE)

    @property
    def wait_save(self) -> bool:
        return self.stage_id == STAGE_WAIT_SAVE

    def stage(self, stage_id: int | None = None) -> StageDef:
        sid = self.stage_id if stage_id is None else stage_id
        return self.cfg.stages[sid]

    def prompt(self, prompt_index: int | None = None) -> PromptDef:
        idx = self.stage().prompt_index if prompt_index is None else prompt_index
        return self.cfg.prompts[idx]

    @property
    def planned_mouse_box_count(self) -> int:
        stage = self.cfg.stages.get(1)
        return 0 if stage is None else stage.repeat

    @property
    def planned_earphone_box_count(self) -> int:
        stage = self.cfg.stages.get(2)
        return 0 if stage is None else stage.repeat

    @property
    def planned_subtask_count(self) -> int:
        return expected_segment_count(self.cfg)

    @property
    def collection_plan(self) -> str:
        return "stage56_flap_closing"

    def _apply_plan_fields(self, ctx: FrameContext) -> FrameContext:
        ctx.planned_mouse_box_count = self.planned_mouse_box_count
        ctx.planned_earphone_box_count = self.planned_earphone_box_count
        ctx.planned_subtask_count = self.planned_subtask_count
        ctx.collection_plan = self.collection_plan
        return ctx

    def begin_episode(self, episode_index: int) -> None:
        self.episode_index = int(episode_index)
        self.stage_id = min(self.cfg.stages)
        self.subtask_instance_id = 1
        self.mouse_box_success_count = 0
        self.earphone_box_success_count = 0
        self.successful_place_count = 0
        self.stage_completed = False
        self.pending_start_next_frame = True
        self._pending_after_frame = None
        self._active_segment = None
        self.segments = []
        self.events = []
        self.last_frame_index = -1
        self.stage_entered_s = time.monotonic()

    def _begin_segment(self, frame_index: int) -> None:
        stage = self.stage()
        prompt = self.prompt(stage.prompt_index)
        self._active_segment = SegmentRecord(
            episode_index=self.episode_index,
            stage_id=stage.stage_id,
            stage_name=stage.name,
            prompt_index=prompt.prompt_index,
            prompt_text=prompt.en,
            subtask_instance_id=self.subtask_instance_id,
            start_frame=frame_index,
            planned_mouse_box_count=self.planned_mouse_box_count,
            planned_earphone_box_count=self.planned_earphone_box_count,
            planned_subtask_count=self.planned_subtask_count,
            collection_plan=self.collection_plan,
        )

    def _finish_segment(self, frame_index: int, event_name: str, success: bool) -> None:
        if self._active_segment is None:
            return
        self._active_segment.end_frame = frame_index
        self._active_segment.completion_event = event_name
        self._active_segment.success = bool(success)
        self.segments.append(self._active_segment)
        self._active_segment = None

    def _base_context(self) -> FrameContext:
        if self.stage_id == STAGE_WAIT_SAVE:
            return self._apply_plan_fields(FrameContext(
                stage_id=STAGE_WAIT_SAVE,
                stage_name="WAIT_SAVE",
                prompt_index=0,
                prompt_text="",
                prompt_text_zh="",
                subtask_instance_id=0,
                mouse_box_success_count=self.mouse_box_success_count,
                earphone_box_success_count=self.earphone_box_success_count,
                successful_place_count=self.successful_place_count,
            ))
        stage = self.stage()
        prompt = self.prompt(stage.prompt_index)
        return self._apply_plan_fields(FrameContext(
            stage_id=stage.stage_id,
            stage_name=stage.name,
            prompt_index=prompt.prompt_index,
            prompt_text=prompt.en,
            prompt_text_zh=prompt.zh,
            subtask_instance_id=self.subtask_instance_id,
            mouse_box_success_count=self.mouse_box_success_count,
            earphone_box_success_count=self.earphone_box_success_count,
            successful_place_count=self.successful_place_count,
        ))

    def frame_context(self, frame_index: int, override: FrameContext | None = None) -> FrameContext:
        if override is not None:
            return override
        ctx = self._base_context()
        if self.pending_start_next_frame and self.recording:
            ctx.subtask_start = 1
            ctx.keyframe_flag = 1
            if self.last_frame_index < 0:
                ctx.event_code = EVENT_EPISODE_STARTED
            else:
                ctx.event_code = EVENT_SUBTASK_STARTED
            ctx.event_name = self.cfg.event_names.get(ctx.event_code, EVENT_NAMES[ctx.event_code])
        return self._apply_plan_fields(ctx)

    def after_frame(self, frame_index: int, ctx: FrameContext) -> None:
        self.last_frame_index = frame_index
        if ctx.subtask_start and self._active_segment is None and self.recording:
            self._begin_segment(frame_index)
        if ctx.event_code != EVENT_NONE:
            self.events.append(
                {
                    "episode_index": self.episode_index,
                    "frame_index": frame_index,
                    "stage_id": ctx.stage_id,
                    "stage_name": ctx.stage_name,
                    "prompt_index": ctx.prompt_index,
                    "subtask_instance_id": ctx.subtask_instance_id,
                    "event_code": ctx.event_code,
                    "event_name": ctx.event_name,
                    "timestamp": frame_index,
                    "planned_mouse_box_count": ctx.planned_mouse_box_count,
                    "planned_earphone_box_count": ctx.planned_earphone_box_count,
                    "planned_subtask_count": ctx.planned_subtask_count,
                    "collection_plan": ctx.collection_plan,
                }
            )
        self.pending_start_next_frame = False
        if self._pending_after_frame is not None:
            fn = self._pending_after_frame
            self._pending_after_frame = None
            fn(frame_index)

    def handle_n(self, frame_index: int) -> KeyResult:
        if self.stage_id not in (1, 2) or self.stage_completed:
            return KeyResult(warning="N仅用于重复抓盒阶段；当前阶段不会改变。")
        if self.pending_start_next_frame and self._active_segment is None:
            return KeyResult(warning="当前子任务刚开始，还没有采到起始帧；稍等一帧后再按N。")
        old_stage = self.stage()
        old_prompt = self.prompt(old_stage.prompt_index)
        old_instance = self.subtask_instance_id
        if self.stage_id == 1:
            if self.mouse_box_success_count >= old_stage.repeat:
                return KeyResult(warning=f"鼠标盒已经记录满{old_stage.repeat}个，等待按Space进入Stage 2。")
            self.mouse_box_success_count += 1
            self.successful_place_count += 1
            event_code = EVENT_MOUSE_BOX_PLACED
        else:
            if self.earphone_box_success_count >= old_stage.repeat:
                return KeyResult(warning=f"耳机盒已经记录满{old_stage.repeat}个，等待按Space进入Stage 3。")
            self.earphone_box_success_count += 1
            self.successful_place_count += 1
            event_code = EVENT_EARPHONE_BOX_PLACED

        ctx = self._apply_plan_fields(FrameContext(
            stage_id=old_stage.stage_id,
            stage_name=old_stage.name,
            prompt_index=old_prompt.prompt_index,
            prompt_text=old_prompt.en,
            prompt_text_zh=old_prompt.zh,
            subtask_instance_id=old_instance,
            subtask_end=1,
            event_code=event_code,
            event_name=self.cfg.event_names[event_code],
            keyframe_flag=1,
            mouse_box_success_count=self.mouse_box_success_count,
            earphone_box_success_count=self.earphone_box_success_count,
            successful_place_count=self.successful_place_count,
        ))

        def finish_and_maybe_start_next(done_frame: int) -> None:
            self._finish_segment(done_frame, self.cfg.event_names[event_code], success=True)
            completed = (
                self.mouse_box_success_count if old_stage.stage_id == 1 else self.earphone_box_success_count
            )
            if completed < old_stage.repeat:
                self.subtask_instance_id += 1
                self.pending_start_next_frame = True
                print(f"  → {old_stage.display}: 第{self.subtask_instance_id}个", flush=True)
            else:
                self.stage_completed = True
                print(f"  ✓ {old_stage.name} 已完成，按Space进入下一阶段。", flush=True)

        self._pending_after_frame = finish_and_maybe_start_next
        return KeyResult(frame_context=ctx)

    def handle_space(self, frame_index: int, now_s: float | None = None) -> KeyResult:
        if self.stage_id == STAGE_IDLE:
            return KeyResult(action="start")
        if self.stage_id == STAGE_WAIT_SAVE:
            return KeyResult(warning="当前已在WAIT_SAVE；请按S保存、F保存失败或G丢弃。")
        stage = self.stage()
        if self.stage_id >= 3 and self.pending_start_next_frame and self._active_segment is None:
            return KeyResult(warning="当前阶段刚开始，还没有采到起始帧；稍等一帧后再按Space。")
        elapsed = (now_s if now_s is not None else time.monotonic()) - self.stage_entered_s
        if not self.dry_run and self.stage_id >= 3 and elapsed < self.min_stage_seconds:
            return KeyResult(warning=f"阶段刚开始 {elapsed:.1f}s，忽略这次Space，防止误连按跳阶段。")

        prompt = self.prompt(stage.prompt_index)
        ctx = self._apply_plan_fields(FrameContext(
            stage_id=stage.stage_id,
            stage_name=stage.name,
            prompt_index=prompt.prompt_index,
            prompt_text=prompt.en,
            prompt_text_zh=prompt.zh,
            subtask_instance_id=self.subtask_instance_id,
            subtask_end=1 if self.stage_id >= 3 and self._active_segment is not None else 0,
            event_code=EVENT_STAGE_COMPLETED,
            event_name=self.cfg.event_names[EVENT_STAGE_COMPLETED],
            keyframe_flag=1,
            mouse_box_success_count=self.mouse_box_success_count,
            earphone_box_success_count=self.earphone_box_success_count,
            successful_place_count=self.successful_place_count,
        ))

        def advance(done_frame: int) -> None:
            if stage.stage_id >= 3:
                self._finish_segment(done_frame, self.cfg.event_names[EVENT_STAGE_COMPLETED], success=True)
            next_stage_ids = [sid for sid in sorted(self.cfg.stages) if sid > stage.stage_id]
            if next_stage_ids:
                self.stage_id = next_stage_ids[0]
                self.subtask_instance_id = 1
                self.stage_completed = False
                self.pending_start_next_frame = True
                self.stage_entered_s = time.monotonic()
                nxt = self.stage()
                print(f"  → 进入Stage {nxt.stage_id}: {nxt.name}", flush=True)
                if nxt.stage_id in (1, 2):
                    print(f"  → {nxt.display}: 第1个", flush=True)
            else:
                self.stage_id = STAGE_WAIT_SAVE
                self.stage_completed = True
                print("  ✓ 最后一个Stage5/6合页子任务已完成，进入WAIT_SAVE；按S成功保存，F失败保存，G丢弃。", flush=True)

        self._pending_after_frame = advance
        return KeyResult(frame_context=ctx)

    def close_active_for_outcome(self, frame_index: int, *, success: bool, completion_event: str) -> None:
        if self._active_segment is not None:
            self._finish_segment(max(frame_index, self._active_segment.start_frame), completion_event, success=success)

    def status_lines(self, frame_count: int, elapsed_s: float, *, status: str) -> list[str]:
        if self.stage_id in (STAGE_IDLE, STAGE_WAIT_SAVE):
            stage_name = "IDLE" if self.stage_id == STAGE_IDLE else "WAIT_SAVE"
            prompt = ""
            repeat = 1
        else:
            stage = self.stage()
            prompt = self.prompt(stage.prompt_index).zh
            stage_name = f"{stage.stage_id} - {stage.name}"
            repeat = stage.repeat
        return [
            f"Episode: {self.episode_index:06d}",
            f"Stage: {stage_name}",
            f"Prompt: {prompt}",
            f"Prompt instance: {self.subtask_instance_id} / {repeat}",
            f"Selected plan: {self.collection_plan}, subtask_instances={self.planned_subtask_count}",
            f"Stage5/6 flap subtasks completed by Space: {len(self.segments)} / {self.planned_subtask_count}",
            f"Frames: {frame_count}   Duration: {elapsed_s:.1f}s",
            f"State: {status}",
            "Available keys: SPACE=start/advance  S=save  F=fail  G=discard  Q=quit",
        ]


class EpisodeWorkspace:
    def __init__(self, dataset_root: Path, incomplete_name: str, episode_index: int) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = dataset_root / incomplete_name / f"session_{stamp}_episode_{episode_index:06d}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.events_path = self.path / "events.jsonl"
        self.info_path = self.path / "episode_info.json"

    def write_info(self, payload: dict[str, Any]) -> None:
        self.info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def write_event(self, payload: dict[str, Any]) -> None:
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def discard(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)


def frame_fields(ctx: FrameContext) -> dict[str, Any]:
    return {
        "stage_id": int_feature(ctx.stage_id),
        "stage_name": ctx.stage_name,
        "prompt_index": int_feature(ctx.prompt_index),
        "prompt_text": ctx.prompt_text,
        "prompt_text_zh": ctx.prompt_text_zh,
        "subtask_instance_id": int_feature(ctx.subtask_instance_id),
        "subtask_start": int_feature(ctx.subtask_start),
        "subtask_end": int_feature(ctx.subtask_end),
        "event_code": int_feature(ctx.event_code),
        "event_name": ctx.event_name,
        "keyframe_flag": int_feature(ctx.keyframe_flag),
        "mouse_box_success_count": int_feature(ctx.mouse_box_success_count),
        "earphone_box_success_count": int_feature(ctx.earphone_box_success_count),
        "successful_place_count": int_feature(ctx.successful_place_count),
        "episode_success": int_feature(ctx.episode_success),
        "failure_type": ctx.failure_type,
        "planned_mouse_box_count": int_feature(ctx.planned_mouse_box_count),
        "planned_earphone_box_count": int_feature(ctx.planned_earphone_box_count),
        "planned_subtask_count": int_feature(ctx.planned_subtask_count),
        "collection_plan": ctx.collection_plan,
    }


def set_episode_outcome_fields(dataset, *, success: bool, failure_type: str) -> None:
    size = int(dataset.episode_buffer["size"])
    dataset.episode_buffer["episode_success"] = [int_feature(1 if success else 0) for _ in range(size)]
    dataset.episode_buffer["failure_type"] = [failure_type for _ in range(size)]


def assert_episode_has_robot_motion(dataset, *, min_motion_rad: float) -> None:
    states = dataset.episode_buffer.get("observation.state", [])
    if not states:
        raise RuntimeError("完整性检查失败: episode没有 observation.state。")
    arr = np.asarray([np.asarray(v, dtype=np.float64).reshape(-1) for v in states])
    if arr.ndim != 2 or arr.shape[1] < 14:
        raise RuntimeError(f"完整性检查失败: observation.state 维度异常: {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError("完整性检查失败: observation.state 包含 NaN/Inf。")
    joint_ranges = np.nanmax(arr[:, :14], axis=0) - np.nanmin(arr[:, :14], axis=0)
    max_motion = float(np.nanmax(joint_ranges))
    moving_dims = int(np.sum(joint_ranges > 1e-4))
    if max_motion < float(min_motion_rad):
        raise RuntimeError(
            "完整性检查失败: 机械臂关节轨迹几乎没有变化，疑似没有采到真实关节反馈。"
            f" max_joint_range={max_motion:.6f}rad < {float(min_motion_rad):.6f}rad, "
            f"moving_joint_dims={moving_dims}/14"
        )


def append_jsonl(path: Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset_sidecars(args: argparse.Namespace, cfg: LongTaskConfig) -> None:
    annotations = args.root / "annotations"
    mappings = {
        "stage_ids": {
            "0": "IDLE",
            **{str(stage.stage_id): stage.name for stage in cfg.stages.values()},
            "7": "WAIT_SAVE",
        },
        "prompts": {
            str(idx): {"zh": prompt.zh, "en": prompt.en}
            for idx, prompt in sorted(cfg.prompts.items())
        },
        "event_codes": {str(k): v for k, v in sorted(cfg.event_names.items())},
        "failure_types": cfg.failure_types,
        "action_source": _cfg_get(cfg.raw, "task.action_source", ""),
        "action_note": _cfg_get(cfg.raw, "task.action_note", ""),
    }
    write_json(annotations / "field_mappings.json", mappings)
    write_json(annotations / "collection_config_snapshot.json", cfg.raw)


def move_raw_depth_from_temp(temp_dir: Path, dataset_root: Path, episode_index: int) -> None:
    src_root = temp_dir / "depth_raw"
    if not src_root.exists():
        return
    for src in src_root.glob("*/episode_*"):
        dst = dataset_root / "depth_raw" / src.parent.name / f"episode_{episode_index:06d}"
        if dst.exists():
            raise RuntimeError(f"raw depth destination already exists, refusing overwrite: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def raw_depth_episode_manifests(args: argparse.Namespace, episode_index: int) -> list[dict[str, Any]]:
    raw_root = args.root / "depth_raw"
    if not raw_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    cfg_by_name = {cfg.name: cfg for cfg in args.depth_camera_configs}
    for ep_dir in sorted(raw_root.glob(f"*/episode_{episode_index:06d}")):
        camera_name = ep_dir.parent.name
        files = sorted([p for p in ep_dir.iterdir() if p.is_file()])
        if not files:
            continue
        first = files[0]
        shape = None
        dtype = ""
        if first.suffix.lower() == ".png":
            import cv2

            image = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
            if image is not None:
                shape = list(image.shape)
                dtype = str(image.dtype)
        elif first.suffix.lower() == ".npy":
            image = np.load(first, mmap_mode="r")
            shape = list(image.shape)
            dtype = str(image.dtype)

        height = int(shape[0]) if shape else None
        width = int(shape[1]) if shape and len(shape) >= 2 else None
        cfg = cfg_by_name.get(camera_name)
        video_shape = [3, int(cfg.height), int(cfg.width)] if cfg is not None else None
        rows.append(
            {
                "episode_index": episode_index,
                "camera_name": camera_name,
                "feature_key": f"observation.depth_images.{camera_name}",
                "raw_depth_dir": str(ep_dir.relative_to(args.root)),
                "path_pattern": str((ep_dir / "frame_%06d.*").relative_to(args.root)),
                "frame_count": len(files),
                "raw_dtype": dtype,
                "raw_shape_hw": [height, width],
                "raw_unit": "millimeter",
                "raw_note": "Raw uint16 depth sidecar is stored at the sensor's native depth resolution; the LeRobot depth video feature may be resized/colorized according to dataset metadata.",
                "video_feature_shape_chw": video_shape,
            }
        )
    return rows


def expected_segment_count(cfg: LongTaskConfig) -> int:
    return sum(stage.repeat for stage in cfg.stages.values())


def success_key_sequence(cfg: LongTaskConfig) -> list[str]:
    # First SPACE starts the episode outside this helper; one SPACE completes each micro subtask.
    return ["SPACE"] * len(cfg.stages)


def integrity_check(args: argparse.Namespace, cfg: LongTaskConfig, result: EpisodeResult, *, success: bool) -> None:
    parquet = list(args.root.rglob(f"episode_{result.episode_index:06d}.parquet"))
    if not parquet:
        raise RuntimeError(f"完整性检查失败: 未找到 episode_{result.episode_index:06d}.parquet")
    if args.raw_depth_format != "none":
        for row in raw_depth_episode_manifests(args, result.episode_index):
            if row["frame_count"] != result.frames:
                raise RuntimeError(
                    f"完整性检查失败: {row['camera_name']} raw depth 帧数 "
                    f"{row['frame_count']} != episode帧数 {result.frames}"
                )
    if success:
        expected_segments = expected_segment_count(cfg)
        if len(result.segments) != expected_segments:
            raise RuntimeError(
                f"完整性检查失败: 成功episode应有{expected_segments}个Stage5/6合页子任务片段，实际 {len(result.segments)}"
            )
    print(f"  ✓ 完整性检查通过: episode={result.episode_index:06d}, frames={result.frames}", flush=True)


def print_block(lines: list[str]) -> None:
    print("\n" + "\n".join(lines), flush=True)


def choose_failure_type(failure_types: list[str]) -> str:
    choices = [ft for ft in failure_types if ft != "none"]
    print("\n请选择失败类型，输入序号或名称；直接回车=manual_abort")
    for i, ft in enumerate(choices, 1):
        print(f"  {i}. {ft}")
    value = input("> ").strip()
    if not value:
        return "manual_abort"
    if value.isdigit() and 1 <= int(value) <= len(choices):
        return choices[int(value) - 1]
    if value in choices:
        return value
    print(f"  未识别失败类型 {value!r}，记录为 other。", flush=True)
    return "other"


def read_state_frame(arms, args: argparse.Namespace, last_state: np.ndarray | None) -> tuple[np.ndarray, bool]:
    return base_collector._read_state_resilient(arms, use_gripper=args.use_gripper, last_state=last_state)


def choose_box_count(keys: KeyboardController, label: str, default: int) -> int | None:
    print(f"{label}数量：按 1/2/3 选择；按 Space 使用默认 {default}；按 Q 结束程序。")
    while True:
        key = keys.get(0.3)
        if key in {"1", "2", "3"}:
            value = int(key)
            print(f"  → {label}: {value} 个", flush=True)
            return value
        if key == "SPACE":
            print(f"  → {label}: {default} 个", flush=True)
            return default
        if key == "Q":
            return None


def wait_idle_start(
    keys: KeyboardController,
    episode_index: int,
    cfg: LongTaskConfig,
    *,
    ask_box_counts: bool,
) -> tuple[str, LongTaskConfig | None]:
    print(f"\n=== IDLE: 准备采集 episode {episode_index:06d} ===")
    print("请把纸箱摆到Stage5起始位姿(已旋转90度、左右合页都张开)，双臂回到统一初始姿态。")
    keys.drain()
    ep_cfg = cfg
    if ask_box_counts:
        if 1 not in cfg.stages or 2 not in cfg.stages:
            print("  [WARN] 当前配置没有抓盒阶段，跳过左右盒子数量选择。", flush=True)
            ask_box_counts = False
    if ask_box_counts:
        mouse = choose_box_count(keys, "右臂鼠标盒", cfg.stages[1].repeat)
        if mouse is None:
            return "quit", None
        ear = choose_box_count(keys, "左臂耳机盒", cfg.stages[2].repeat)
        if ear is None:
            return "quit", None
        ep_cfg = apply_repeat_overrides(cfg, mouse_boxes=mouse, earphone_boxes=ear)
    print(
        f"本条计划: Stage5/6合页子任务实例 {expected_segment_count(ep_cfg)} 个。"
    )
    print("按 Space 开始；按 Q 结束程序。")
    while True:
        key = keys.get(0.3)
        if key == "SPACE":
            stage = ep_cfg.stages[min(ep_cfg.stages)]
            print(f"  → 开始episode，进入Stage {stage.stage_id}: {stage.name}")
            return "start", ep_cfg
        if key == "Q":
            return "quit", None


def confirm_quit_or_save(keys: KeyboardController) -> str:
    print("\n当前episode还在录制。按 S保存成功，F保存失败，G丢弃；再次按 Q 强制结束。")
    while True:
        key = keys.get(0.3)
        if key in {"S", "F", "G", "Q"}:
            return {"S": "save_success", "F": "save_failure", "G": "discard", "Q": "force_quit"}[key]


def record_episode(
    args: argparse.Namespace,
    cfg: LongTaskConfig,
    dataset,
    arms,
    cameras: CameraSet,
    depth_cameras: DepthCameraSet,
    keys: KeyboardController,
) -> EpisodeResult:
    ep_index = int(dataset.episode_buffer["episode_index"])
    workspace = EpisodeWorkspace(args.root, args.incomplete_dir_name, ep_index)
    workspace.write_info(
        {
            "episode_index": ep_index,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "repo_id": args.repo_id,
            "root": str(args.root),
            "config": str(args.config),
            "status": "recording",
            "planned_mouse_box_count": 0,
            "planned_earphone_box_count": 0,
            "planned_subtask_count": expected_segment_count(cfg),
            "collection_plan": "stage56_flap_closing",
        }
    )
    manager = StageManager(cfg, min_stage_seconds=args.min_stage_seconds)
    manager.begin_episode(ep_index)
    raw_depth_writer = base_collector.RawDepthSidecarWriter(
        workspace.path,
        args.raw_depth_format,
        args.raw_depth_png_compression,
        args.raw_depth_writer_queue_size,
    )

    period_s = 1.0 / args.fps
    max_frames = max(1, int(round(args.episode_seconds * args.fps)))
    start_s = time.monotonic()
    last_state: np.ndarray | None = None
    stale_state_frames = 0
    max_stale_state_frames = max(1, int(round(args.max_stale_state_seconds * args.fps)))
    frame_index = 0
    early_success_confirm = False
    action = "timeout_failure"
    try:
        while frame_index < max_frames:
            target_s = start_s + frame_index * period_s
            now_s = time.monotonic()
            if now_s < target_s:
                time.sleep(target_s - now_s)

            key = keys.get(0.0)
            override: FrameContext | None = None
            if key == "SPACE":
                result = manager.handle_space(frame_index, now_s=time.monotonic())
                if result.warning:
                    print(f"\n  ⚠ {result.warning}", flush=True)
                override = result.frame_context
            elif key == "N":
                result = manager.handle_n(frame_index)
                if result.warning:
                    print(f"\n  ⚠ {result.warning}", flush=True)
                override = result.frame_context
            elif key == "S":
                if not manager.wait_save and not early_success_confirm:
                    early_success_confirm = True
                    print("\n  ⚠ Stage 6尚未完成；再次按S才会把本条保存为不完整成功数据。", flush=True)
                    continue
                action = "save_success"
                break
            elif key == "F":
                action = "save_failure"
                break
            elif key == "G":
                action = "discard"
                break
            elif key == "Q":
                action = confirm_quit_or_save(keys)
                break

            ctx = manager.frame_context(frame_index, override)
            state, ok = read_state_frame(arms, args, last_state)
            if ok:
                stale_state_frames = 0
            else:
                stale_state_frames += 1
                if stale_state_frames == 1 or stale_state_frames % max(1, args.fps) == 0:
                    print(
                        f"\n  ⚠ 机器人状态读取失败，正在复用上一帧 "
                        f"({stale_state_frames}/{max_stale_state_frames})",
                        flush=True,
                    )
                if stale_state_frames > max_stale_state_frames:
                    raise RuntimeError(
                        "机器人状态连续读取失败过久，已停止本条采集，避免保存固定关节数据。"
                    )
            last_state = state
            frame = {
                "observation.state": state,
                "action": state.copy(),
            }
            frame.update(frame_fields(ctx))
            frame.update(cameras.read_frames())
            frame.update(depth_cameras.read_frames())
            raw_depths = depth_cameras.read_raw_depths() if raw_depth_writer.enabled else None
            dataset.add_frame(frame, task=ctx.prompt_text or _cfg_get(cfg.raw, "task.instruction", ""), timestamp=frame_index / args.fps)
            if raw_depths is not None:
                raw_depth_writer.submit(ep_index, frame_index, raw_depths)
            if ctx.event_code != EVENT_NONE:
                workspace.write_event(
                    {
                        "episode_index": ep_index,
                        "frame_index": frame_index,
                        "timestamp_s": frame_index / args.fps,
                        "event_code": ctx.event_code,
                        "event_name": ctx.event_name,
                        "stage_id": ctx.stage_id,
                        "stage_name": ctx.stage_name,
                        "prompt_index": ctx.prompt_index,
                        "subtask_instance_id": ctx.subtask_instance_id,
                        "planned_mouse_box_count": ctx.planned_mouse_box_count,
                        "planned_earphone_box_count": ctx.planned_earphone_box_count,
                        "planned_subtask_count": ctx.planned_subtask_count,
                        "collection_plan": ctx.collection_plan,
                    }
                )
            if not ok:
                print("\n  ⚠ 本帧CAN读取失败，复用上一帧状态。", flush=True)

            manager.after_frame(frame_index, ctx)
            frame_index += 1

            if frame_index == 1 or frame_index % args.fps == 0 or ctx.event_code != EVENT_NONE:
                elapsed = time.monotonic() - start_s
                print_block(manager.status_lines(frame_index, elapsed, status="WAIT_SAVE" if manager.wait_save else "RECORDING"))

            if manager.wait_save:
                while True:
                    key2 = keys.get(0.3)
                    if key2 == "S":
                        action = "save_success"
                        break
                    if key2 == "F":
                        action = "save_failure"
                        break
                    if key2 == "G":
                        action = "discard"
                        break
                    if key2 == "Q":
                        action = "force_quit"
                        break
                break
        else:
            print("\n  ⚠ 达到episode时长上限，按失败类型timeout保存。", flush=True)
            action = "save_failure"

        raw_depth_writer.flush()
    finally:
        raw_depth_writer.close()

    frames = int(dataset.episode_buffer["size"])
    if action == "save_success":
        manager.close_active_for_outcome(frames - 1, success=True, completion_event=EVENT_NAMES[EVENT_EPISODE_SUCCESS])
        failure_type = "none"
    elif action == "save_failure":
        manager.close_active_for_outcome(frames - 1, success=False, completion_event=EVENT_NAMES[EVENT_EPISODE_FAILED])
        failure_type = "pending"
    else:
        manager.close_active_for_outcome(frames - 1, success=False, completion_event=EVENT_NAMES[EVENT_EPISODE_DISCARDED])
        failure_type = "manual_abort"

    return EpisodeResult(
        action=action,
        frames=frames,
        episode_index=ep_index,
        temp_dir=workspace.path,
        segments=[seg.as_dict() for seg in manager.segments],
        events=manager.events,
        failure_type=failure_type,
        force_quit=(action == "force_quit"),
        planned_mouse_box_count=manager.planned_mouse_box_count,
        planned_earphone_box_count=manager.planned_earphone_box_count,
        planned_subtask_count=manager.planned_subtask_count,
        collection_plan=manager.collection_plan,
    )


def save_episode_result(args: argparse.Namespace, cfg: LongTaskConfig, dataset, result: EpisodeResult, *, success: bool, failure_type: str) -> None:
    if success:
        assert_episode_has_robot_motion(dataset, min_motion_rad=args.min_state_motion_rad)
    set_episode_outcome_fields(dataset, success=success, failure_type=failure_type)
    print("  保存中: LeRobot Parquet/视频/元数据编码...", flush=True)
    dataset.save_episode()
    move_raw_depth_from_temp(result.temp_dir, args.root, result.episode_index)
    raw_depth_rows = raw_depth_episode_manifests(args, result.episode_index)
    annotations = args.root / "annotations"
    append_jsonl(annotations / "subtask_segments.jsonl", result.segments)
    event_rows = [dict(row, timestamp_s=row["frame_index"] / args.fps) for row in result.events]
    append_jsonl(annotations / "episode_events.jsonl", event_rows)
    if raw_depth_rows:
        append_jsonl(annotations / "raw_depth_episodes.jsonl", raw_depth_rows)
    append_jsonl(
        annotations / "episode_outcomes_detailed.jsonl",
        {
            "episode_index": result.episode_index,
            "episode_success": bool(success),
            "failure_type": failure_type,
            "frames": result.frames,
            "segment_count": len(result.segments),
            "planned_mouse_box_count": result.planned_mouse_box_count,
            "planned_earphone_box_count": result.planned_earphone_box_count,
            "planned_subtask_count": result.planned_subtask_count,
            "collection_plan": result.collection_plan,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    write_dataset_sidecars(args, cfg)
    if result.temp_dir.exists():
        shutil.rmtree(result.temp_dir)
    integrity_check(args, cfg, result, success=success)


def discard_episode(dataset, result: EpisodeResult) -> None:
    if dataset.episode_buffer is not None and int(dataset.episode_buffer["size"]) > 0:
        dataset.clear_episode_buffer()
    if result.temp_dir.exists():
        shutil.rmtree(result.temp_dir)
    print("  ✗ 已丢弃本条；未增加episode_index，未写正式元数据。", flush=True)


def run_collection(args: argparse.Namespace, cfg: LongTaskConfig, dataset, arms, cameras, depth_cameras) -> int:
    interactive = sys.stdin.isatty()
    if not interactive:
        raise RuntimeError("长程采集需要交互式终端；状态机dry-run请用 --dry-run-state-machine。")
    saved = 0

    while args.num_episodes <= 0 or saved < args.num_episodes:
        with KeyboardController() as keys:
            ep_index = int(dataset.episode_buffer["episode_index"])
            idle, ep_cfg = wait_idle_start(keys, ep_index, cfg, ask_box_counts=args.ask_box_counts)
            if idle == "quit":
                break
            assert ep_cfg is not None
            result = record_episode(args, ep_cfg, dataset, arms, cameras, depth_cameras, keys)

        if result.force_quit:
            print("  ⚠ 强制结束；当前LeRobot缓存未保存，incomplete目录保留供检查。", flush=True)
            break
        if result.action == "discard":
            discard_episode(dataset, result)
            continue
        if result.action == "save_failure":
            failure_type = choose_failure_type(ep_cfg.failure_types)
            save_episode_result(args, ep_cfg, dataset, result, success=False, failure_type=failure_type)
            saved += 1
            continue
        if result.action == "save_success":
            save_episode_result(args, ep_cfg, dataset, result, success=True, failure_type="none")
            saved += 1
            continue
    return saved


def run_state_machine_dry_run(cfg: LongTaskConfig) -> None:
    manager = StageManager(cfg, min_stage_seconds=0.0, dry_run=True)
    frame = 0
    now = time.monotonic()
    trace: list[str] = ["IDLE"]

    def tick() -> None:
        nonlocal frame
        ctx = manager.frame_context(frame)
        manager.after_frame(frame, ctx)
        frame += 1

    manager.begin_episode(0)
    tick()
    trace.append(f"Stage {manager.stage_id}")

    sequence = success_key_sequence(cfg)
    for key in sequence:
        if key == "N":
            result = manager.handle_n(frame)
        else:
            now += 1.0
            result = manager.handle_space(frame, now_s=now)
        if result.warning:
            raise RuntimeError(f"dry-run状态机出现非法警告: {result.warning}")
        ctx = manager.frame_context(frame, result.frame_context)
        manager.after_frame(frame, ctx)
        if key == "SPACE":
            if manager.wait_save:
                trace.append("WAIT_SAVE")
            else:
                trace.append(f"Stage {manager.stage_id}")
        frame += 1
        if manager.pending_start_next_frame and not manager.wait_save:
            tick()

    if not manager.wait_save:
        raise RuntimeError("dry-run失败: 最终没有进入WAIT_SAVE")
    expected_segments = expected_segment_count(cfg)
    if len(manager.segments) != expected_segments:
        raise RuntimeError(f"dry-run失败: 子任务片段数 {len(manager.segments)} != {expected_segments}")
    print("dry-run状态机通过:")
    print("  " + " -> ".join(trace) + " -> S/F/G -> IDLE")
    print("子任务片段:")
    for seg in manager.segments:
        print(
            f"  ep={seg.episode_index} stage={seg.stage_id} prompt={seg.prompt_index} "
            f"inst={seg.subtask_instance_id} frames={seg.start_frame}-{seg.end_frame} "
            f"event={seg.completion_event}"
        )


def run_dry_run_save_episode(args: argparse.Namespace, cfg: LongTaskConfig) -> None:
    dataset = create_dataset(args)
    write_dataset_sidecars(args, cfg)
    arms = base_collector.MockDualNeroArms()
    cameras = CameraSet(args.camera_configs)
    depth_cameras = DepthCameraSet(args.depth_camera_configs)
    ep_index = int(dataset.episode_buffer["episode_index"])
    workspace = EpisodeWorkspace(args.root, args.incomplete_dir_name, ep_index)
    manager = StageManager(cfg, min_stage_seconds=0.0, dry_run=True)
    manager.begin_episode(ep_index)
    last_state: np.ndarray | None = None
    frame = 0

    def add_one(ctx: FrameContext | None = None) -> None:
        nonlocal frame, last_state
        ctx2 = manager.frame_context(frame, ctx)
        state, _ok = read_state_frame(arms, args, last_state)
        last_state = state
        row = {"observation.state": state, "action": state.copy()}
        row.update(frame_fields(ctx2))
        row.update(cameras.read_frames())
        row.update(depth_cameras.read_frames())
        dataset.add_frame(row, task=ctx2.prompt_text, timestamp=frame / args.fps)
        if ctx2.event_code != EVENT_NONE:
            workspace.write_event(
                {
                    "episode_index": ep_index,
                    "frame_index": frame,
                    "timestamp_s": frame / args.fps,
                    "event_code": ctx2.event_code,
                    "event_name": ctx2.event_name,
                    "stage_id": ctx2.stage_id,
                    "stage_name": ctx2.stage_name,
                    "prompt_index": ctx2.prompt_index,
                    "subtask_instance_id": ctx2.subtask_instance_id,
                    "planned_mouse_box_count": ctx2.planned_mouse_box_count,
                    "planned_earphone_box_count": ctx2.planned_earphone_box_count,
                    "planned_subtask_count": ctx2.planned_subtask_count,
                    "collection_plan": ctx2.collection_plan,
                }
            )
        manager.after_frame(frame, ctx2)
        frame += 1

    def maybe_start_tick() -> None:
        if manager.pending_start_next_frame and not manager.wait_save:
            add_one()

    cameras.open()
    depth_cameras.open()
    try:
        add_one()
        for key in success_key_sequence(cfg):
            if key == "N":
                res = manager.handle_n(frame)
            else:
                res = manager.handle_space(frame, now_s=time.monotonic() + 10.0)
            if res.warning:
                raise RuntimeError(f"dry-run-save非法警告: {res.warning}")
            add_one(res.frame_context)
            maybe_start_tick()
        if not manager.wait_save:
            raise RuntimeError("dry-run-save未进入WAIT_SAVE")
        result = EpisodeResult(
            action="save_success",
            frames=int(dataset.episode_buffer["size"]),
            episode_index=ep_index,
            temp_dir=workspace.path,
            segments=[seg.as_dict() for seg in manager.segments],
            events=manager.events,
            planned_mouse_box_count=manager.planned_mouse_box_count,
            planned_earphone_box_count=manager.planned_earphone_box_count,
            planned_subtask_count=manager.planned_subtask_count,
            collection_plan=manager.collection_plan,
        )
        save_episode_result(args, cfg, dataset, result, success=True, failure_type="none")
        print(f"dry-run-save通过: {args.root}")
    finally:
        depth_cameras.close()
        cameras.close()
        arms.close()


def check_depth_only(args: argparse.Namespace) -> None:
    import numpy as np

    depth_cameras = DepthCameraSet(args.depth_camera_configs)
    try:
        depth_cameras.open()
        frames = depth_cameras.read_frames()
        raws = depth_cameras.read_raw_depths()
        for cfg in args.depth_camera_configs:
            raw = np.asarray(raws[cfg.name])
            frame = np.asarray(frames[cfg.feature_key])
            valid = raw > 0
            if valid.any():
                extra = f"valid={int(valid.sum())}/{raw.size} range={int(raw[valid].min())}-{int(raw[valid].max())}mm"
            else:
                extra = f"valid=0/{raw.size}"
            print(f"  ✓ {cfg.name}: {frame.shape}/{frame.dtype}, raw={raw.shape}/{raw.dtype}, {extra}")
    finally:
        depth_cameras.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Nero long box-packing episodes as LeRobot v2.1 data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run-state-machine", action="store_true")
    parser.add_argument("--dry-run-save-episode", action="store_true")
    parser.add_argument("--check-depth-only", action="store_true")
    parser.add_argument("--mouse-boxes", type=int, default=None, help="Override Stage 1 repeat count")
    parser.add_argument("--earphone-boxes", type=int, default=None, help="Override Stage 2 repeat count")
    parser.add_argument(
        "--ask-box-counts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compatibility option from the full box-packing collector; ignored by this Stage5/6 flap-closing collector.",
    )
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--timestamp-root",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Create a fresh timestamped dataset directory for each program run.",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=None)
    parser.add_argument("--min-stage-seconds", type=float, default=0.8)
    parser.add_argument(
        "--min-state-motion-rad",
        type=float,
        default=0.02,
        help="Minimum max joint range required before saving a successful episode.",
    )
    parser.add_argument(
        "--max-stale-state-seconds",
        type=float,
        default=0.5,
        help="Abort recording if robot state reads fail and reuse the previous state for longer than this.",
    )

    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--left-firmware", choices=("default", "v111", "v112", "v120"), default=None)
    parser.add_argument("--right-firmware", choices=("default", "v111", "v112", "v120"), default=None)
    parser.add_argument("--can-interface", default=None)
    parser.add_argument("--can-bitrate", type=int, default=None)
    parser.add_argument("--can-timeout", type=float, default=None)
    parser.add_argument("--speed-percent", type=int, default=None)
    parser.add_argument("--enable", dest="enable", action="store_true")
    parser.add_argument("--no-enable", dest="enable", action="store_false")
    parser.set_defaults(enable=None)
    parser.add_argument("--confirm-enable-risk", action="store_true")
    parser.add_argument("--drag-teach", dest="drag_teach", action="store_true")
    parser.add_argument("--no-drag-teach", dest="drag_teach", action="store_false")
    parser.set_defaults(drag_teach=None)
    parser.add_argument("--confirm-drag-teach-risk", action="store_true")
    parser.add_argument("--mode-settle-seconds", type=float, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--use-gripper", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--left-wrist-camera", default=None)
    parser.add_argument("--right-wrist-camera", default=None)
    parser.add_argument("--third-camera", default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--third-camera-width", type=int, default=None)
    parser.add_argument("--third-camera-height", type=int, default=None)
    parser.add_argument("--third-camera-fps", type=int, default=None)
    parser.add_argument("--camera-fourcc", default=None)
    parser.add_argument("--camera-storage", choices=("video", "image"), default=None)
    parser.add_argument("--mock-cameras", action="store_true")

    parser.add_argument("--use-depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--depth-camera-names", nargs="+", default=None)
    parser.add_argument("--depth-backend", choices=("orbbec_v1", "obsensor", "v4l2", "openni2", "openni2_astra", "mock"), default=None)
    parser.add_argument("--third-depth-index", type=int, default=None)
    parser.add_argument("--third-depth-serial", default=None)
    parser.add_argument("--third-depth-device", default=None)
    parser.add_argument("--depth-width", type=int, default=None)
    parser.add_argument("--depth-height", type=int, default=None)
    parser.add_argument("--depth-fps", type=int, default=None)
    parser.add_argument("--depth-storage", choices=("video", "image"), default=None)
    parser.add_argument("--depth-min-mm", type=float, default=None)
    parser.add_argument("--depth-max-mm", type=float, default=None)
    parser.add_argument("--depth-colorize", choices=("gray", "turbo"), default=None)
    parser.add_argument("--depth-open-timeout", type=float, default=None)
    parser.add_argument("--orbbec-v1-bridge", default=None)
    parser.add_argument("--raw-depth-format", choices=("none", "png", "npy", "both"), default=None)
    parser.add_argument("--raw-depth-png-compression", type=int, default=None)
    parser.add_argument("--raw-depth-writer-queue-size", type=int, default=128)
    return parser.parse_args()


def apply_config_defaults(args: argparse.Namespace, cfg: LongTaskConfig) -> argparse.Namespace:
    raw = cfg.raw
    repo_id_from_cli = args.repo_id is not None
    if args.dry_run_save_episode:
        args.mock = True
        args.mock_cameras = True
        args.overwrite = True
        if args.root is None:
            args.root = Path("/tmp/nero_stage56_dryrun_episode")
        if args.repo_id is None:
            args.repo_id = "local/nero_stage56_dryrun_episode"
        if args.fps is None:
            args.fps = 5
        if args.camera_width is None:
            args.camera_width = 64
        if args.camera_height is None:
            args.camera_height = 48
        if args.third_camera_width is None:
            args.third_camera_width = 64
        if args.third_camera_height is None:
            args.third_camera_height = 48
        if args.camera_storage is None:
            args.camera_storage = "image"
        if args.depth_backend is None:
            args.depth_backend = "mock"
        if args.depth_width is None:
            args.depth_width = 32
        if args.depth_height is None:
            args.depth_height = 24
        if args.depth_storage is None:
            args.depth_storage = "image"
        if args.raw_depth_format is None:
            args.raw_depth_format = "none"

    args.repo_id = args.repo_id or _cfg_get(raw, "dataset.repo_id")
    args.root = as_path(args.root or _cfg_get(raw, "dataset.root"))
    timestamp_root = bool(_cfg_get(raw, "dataset.timestamp_root", False)) if args.timestamp_root is None else bool(args.timestamp_root)
    if timestamp_root and not (args.dry_run_state_machine or args.dry_run_save_episode or args.check_depth_only):
        base_root = args.root
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.root = base_root.parent / f"{base_root.name}_{stamp}"
        if not repo_id_from_cli:
            args.repo_id = f"local/{args.root.name}"
        args.resume = False
        args.overwrite = False
    args.fps = args.fps or int(_cfg_get(raw, "dataset.fps", 15))
    args.episode_seconds = args.episode_seconds or float(_cfg_get(raw, "dataset.episode_seconds", 300))
    args.resume = bool(_cfg_get(raw, "dataset.resume", True)) if args.resume is None else bool(args.resume)
    args.image_writer_threads = args.image_writer_threads if args.image_writer_threads is not None else int(_cfg_get(raw, "dataset.image_writer_threads", 4))
    args.robot_type = str(_cfg_get(raw, "dataset.robot_type", "dual_nero_box_packing"))
    args.incomplete_dir_name = str(_cfg_get(raw, "dataset.incomplete_dir_name", "incomplete"))

    args.left_can = args.left_can or _cfg_get(raw, "can.default_left_can", "can0")
    args.right_can = args.right_can or _cfg_get(raw, "can.default_right_can", "can1")
    args.left_firmware = args.left_firmware or _cfg_get(raw, "can.left_firmware", "v120")
    args.right_firmware = args.right_firmware or _cfg_get(raw, "can.right_firmware", "v120")
    args.can_interface = args.can_interface or _cfg_get(raw, "can.interface", "socketcan")
    args.can_bitrate = args.can_bitrate or int(_cfg_get(raw, "can.bitrate", 1_000_000))
    args.can_timeout = args.can_timeout or float(_cfg_get(raw, "can.timeout", 1.0))
    args.speed_percent = args.speed_percent or int(_cfg_get(raw, "safety.speed_percent", 5))
    args.enable = bool(_cfg_get(raw, "safety.enable", False)) if args.enable is None else bool(args.enable)
    args.drag_teach = bool(_cfg_get(raw, "safety.drag_teach", False)) if args.drag_teach is None else bool(args.drag_teach)
    args.mode_settle_seconds = args.mode_settle_seconds or float(_cfg_get(raw, "safety.mode_settle_seconds", 0.5))

    args.left_wrist_camera = args.left_wrist_camera or _cfg_get(raw, "cameras.rgb.left_wrist", "/dev/video2")
    args.right_wrist_camera = args.right_wrist_camera or _cfg_get(raw, "cameras.rgb.right_wrist", "/dev/video6")
    args.third_camera = args.third_camera or _cfg_get(raw, "cameras.rgb.third_view", "/dev/video4")
    args.camera_width = args.camera_width or int(_cfg_get(raw, "cameras.rgb.width", 1280))
    args.camera_height = args.camera_height or int(_cfg_get(raw, "cameras.rgb.height", 960))
    args.camera_fps = args.camera_fps or int(_cfg_get(raw, "cameras.rgb.fps", 15))
    args.camera_fourcc = args.camera_fourcc or _cfg_get(raw, "cameras.rgb.fourcc", "MJPG")
    args.camera_storage = args.camera_storage or _cfg_get(raw, "cameras.rgb.storage", "video")

    args.use_depth = bool(_cfg_get(raw, "cameras.depth.enabled", True)) if args.use_depth is None else bool(args.use_depth)
    args.depth_camera_names = args.depth_camera_names or list(_cfg_get(raw, "cameras.depth.camera_names", ["third_view"]))
    args.depth_backend = args.depth_backend or _cfg_get(raw, "cameras.depth.backend", "orbbec_v1")
    args.third_depth_index = args.third_depth_index if args.third_depth_index is not None else 2
    args.third_depth_serial = args.third_depth_serial or _cfg_get(raw, "cameras.depth.third_view_serial", "<THIRD_CAM_SERIAL>")
    args.third_depth_device = args.third_depth_device or _cfg_get(raw, "cameras.depth.third_view_device", None)
    args.depth_width = args.depth_width or int(_cfg_get(raw, "cameras.depth.width", 640))
    args.depth_height = args.depth_height or int(_cfg_get(raw, "cameras.depth.height", 480))
    args.depth_fps = args.depth_fps or int(_cfg_get(raw, "cameras.depth.fps", 15))
    args.depth_storage = args.depth_storage or _cfg_get(raw, "cameras.depth.storage", "video")
    args.depth_min_mm = args.depth_min_mm if args.depth_min_mm is not None else float(_cfg_get(raw, "cameras.depth.min_mm", 250))
    args.depth_max_mm = args.depth_max_mm if args.depth_max_mm is not None else float(_cfg_get(raw, "cameras.depth.max_mm", 1800))
    args.depth_colorize = args.depth_colorize or _cfg_get(raw, "cameras.depth.colorize", "gray")
    args.depth_open_timeout = args.depth_open_timeout or float(_cfg_get(raw, "cameras.depth.open_timeout", 10))
    args.orbbec_v1_bridge = args.orbbec_v1_bridge or str(TOOLS_ROOT / "orbbec_v1_depth_bridge")
    args.raw_depth_format = args.raw_depth_format or _cfg_get(raw, "cameras.depth.raw_format", "png")
    args.raw_depth_png_compression = args.raw_depth_png_compression if args.raw_depth_png_compression is not None else int(_cfg_get(raw, "cameras.depth.raw_png_compression", 1))

    if args.fps <= 0 or args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("fps/camera width/height must be positive")
    if args.use_depth and (args.depth_width <= 0 or args.depth_height <= 0 or args.depth_fps <= 0):
        raise ValueError("depth width/height/fps must be positive")
    if args.enable and not args.confirm_enable_risk:
        raise ValueError("--enable requires --confirm-enable-risk")
    if args.drag_teach and args.enable:
        raise ValueError("不要同时使用 --drag-teach 和 --enable；主臂示教采集应保持 --no-enable。")
    if args.drag_teach and not args.confirm_drag_teach_risk:
        raise ValueError("--drag-teach requires --confirm-drag-teach-risk")

    args.camera_configs = build_camera_configs(args)
    args.depth_camera_configs = build_depth_camera_configs(args)
    return args


def print_startup(args: argparse.Namespace, cfg: LongTaskConfig) -> None:
    print("============ Nero Stage5/6 合页封盖 7子任务采集 ============")
    print(f"配置: {args.config}")
    print(f"Dataset root: {args.root}")
    print(f"Repo id: {args.repo_id}")
    print(f"FPS/最长时长: {args.fps}Hz / {args.episode_seconds:.0f}s")
    print("RGB:")
    for c in args.camera_configs:
        print(f"  {c.feature_key}: device={c.device}, {c.width}x{c.height}@{c.fps}, {c.storage}")
    print("Depth:")
    for c in args.depth_camera_configs:
        print(f"  {c.feature_key}: backend={c.backend}, device={c.device}, {c.width}x{c.height}@{c.fps}, raw={args.raw_depth_format}")
    print(f"CAN: left={args.left_can} right={args.right_can}")
    print(f"成功保存前关节运动检查: max_joint_range >= {args.min_state_motion_rad:.4f} rad")
    print(f"{len(cfg.prompts)}个微动作prompt:")
    for idx, prompt in sorted(cfg.prompts.items()):
        print(f"  P{idx}: {prompt.zh}")
    print("按键: Space=开始/完成当前子任务并推进  S=成功保存  F=失败保存  G=丢弃  Q=退出")
    print("====================================================")


def main() -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECTTIMEOUT", "1")
    base_collector.add_local_import_paths()
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_repeat_overrides(cfg, mouse_boxes=args.mouse_boxes, earphone_boxes=args.earphone_boxes)
    args = apply_config_defaults(args, cfg)

    if args.dry_run_state_machine:
        run_state_machine_dry_run(cfg)
        return

    if args.dry_run_save_episode:
        print_startup(args, cfg)
        run_dry_run_save_episode(args, cfg)
        return

    print_startup(args, cfg)
    if args.check_depth_only:
        check_depth_only(args)
        return

    dataset = create_dataset(args)
    write_dataset_sidecars(args, cfg)
    arms = base_collector.build_arms(args)
    cameras = CameraSet(args.camera_configs)
    depth_cameras = DepthCameraSet(args.depth_camera_configs)
    try:
        cameras.open()
        depth_cameras.open()
        if args.enable:
            arms.enable(speed_percent=args.speed_percent, timeout=5.0)
        if args.drag_teach:
            print("进入零力/主臂示教模式，并重新开启 CAN 主动反馈...", flush=True)
            arms.enter_drag_teach()
            time.sleep(args.mode_settle_seconds)
        saved = run_collection(args, cfg, dataset, arms, cameras, depth_cameras)
        print(f"\nDone. Saved {saved} episodes.")
        print(f"Dataset: {args.root}")
    finally:
        depth_cameras.close()
        cameras.close()
        arms.close()


if __name__ == "__main__":
    main()
