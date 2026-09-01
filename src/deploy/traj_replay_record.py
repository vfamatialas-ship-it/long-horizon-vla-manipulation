#!/usr/bin/env python3
"""手教轨迹**边回放边采集** —— 回放一条录好的轨迹, 同时录相机与关节角。

和另外两个脚本的分工
--------------------
  traj_teach.py         录轨迹(主臂/拖动示教模式), 只存关节角, 不碰相机
  traj_replay.py        回放 + 回放前体检, 只动机械臂, **不录任何数据**
  traj_replay_record.py 本脚本: 回放的同时把三路相机 + 双臂关节录成 LeRobot 数据集

回放的运动逻辑直接沿用 run_pi05_rollout_hezi_ee_traj.replay_prefix_traj 那一套
(重采样 → 速率限幅 → 抗积分饱和), 那份已经在真机上跑过 664 帧验证过。

数据格式怎么和模型 rollout 区分开
--------------------------------
这批数据是**人手示教的回放**, 不是模型推理跑出来的, 混进 rollout 目录里以后没法
分辨。所以从四个层面都打上标记:

  目录      lerobot_data/Rlinf/data/trajreplay/<轨迹名>_<时间戳>/
            (模型 rollout 在 .../wholeprocess/run_<时间戳>/ 下)
  repo_id   local/trajreplay_<轨迹名>_<时间戳>
  robot_type dual_nero_traj_replay          (rollout 是 dual_nero_hezi_closing_ee)
  逐帧字段  data_source = "traj_replay"     ← 就算目录被搬走也还认得出
            traj_name   = <轨迹名>
            traj_fps    = 原始录制帧率

其余字段与 rollout **完全同构**(16 维关节 + 20 维末端位姿 + 三路相机 + prompt),
所以两边的数据可以直接合并训练, 需要时用 data_source 一句话筛掉或筛出。

action 记**下发的关节指令**, observation.state 记实测关节角 —— 与 rollout 同口径。

用法::

    python3 traj_replay_record.py hy601                    # 干跑, 不动臂不录
    python3 traj_replay_record.py hy601 --execute          # 真回放 + 真录
    python3 traj_replay_record.py hy601 --execute --replay-fps 30
    python3 traj_replay_record.py hy601 --execute --prompt "close the left flap"
    python3 traj_replay_record.py hy601 --execute --repeat 3   # 连录 3 条
    python3 traj_replay_record.py --list

键位(回放中)::

    回车 / q   急停(停发指令, 保持使能钉住当前位姿, 不下电)
    f          判本条失败, 立刻停并保存
    g          丢弃本条
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NERO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NERO_ROOT / "nero_control"))
sys.path.insert(0, str(NERO_ROOT / "ee_pose" / "tools"))
sys.path.insert(0, str(NERO_ROOT / "lerobot_data" / "lerobot_tools" / "lerobot_v21" / "src"))
sys.path.insert(0, str(NERO_ROOT / "lerobot_data" / "lerobot_tools"))

import run_pi05_deploy_hezi_ee as D          # noqa: E402  (臂/相机/限幅/末端位姿)
import rollout_boxpick_common as C           # noqa: E402  (KeyPad)

TRAJ_DIR = HERE / "trajectories"
DATA_ROOT = NERO_ROOT / "lerobot_data" / "Rlinf" / "data" / "trajreplay"

FPS = D.FPS
STATE_NAMES = [f"{s}_j{i}" for s in ("left", "right") for i in range(7)]
STATE_NAMES = ([f"left_j{i}" for i in range(7)] + ["left_gripper"] +
               [f"right_j{i}" for i in range(7)] + ["right_gripper"])
EE20_NAMES = ([f"left_{k}" for k in
               ("x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12", "grip")] +
              [f"right_{k}" for k in
               ("x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12", "grip")])
_IMG = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]}


# ── 数据集 ──────────────────────────────────────────────────────────────────
def create_dataset(args):
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"需要 LeRobot v2.1, 实得 {CODEBASE_VERSION}")
    i64 = {"dtype": "int64", "shape": (1,), "names": None}
    f32 = {"dtype": "float32", "shape": (1,), "names": None}
    s1 = {"dtype": "string", "shape": (1,), "names": None}
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
        "observation.state_ee20": {"dtype": "float32", "shape": (20,), "names": EE20_NAMES},
        "action_ee20": {"dtype": "float32", "shape": (20,), "names": EE20_NAMES},
        "observation.images.third_view": _IMG,
        "observation.images.left_wrist": _IMG,
        "observation.images.right_wrist": _IMG,
        "subtask_instance_id": i64, "prompt_index": i64,
        "prompt_text": s1, "prompt_text_zh": s1,
        "subtask_start": i64, "subtask_end": i64,
        # ── 与模型 rollout 的区分标记(rollout 里没有这三列)──────────────
        "data_source": s1,      # 恒为 "traj_replay"
        "traj_name": s1,        # 哪条手教轨迹
        "traj_fps": i64,        # 原始录制帧率(通常 100)
        # ── 与 rollout 同构的其余列, 便于合并 ───────────────────────────
        "ik_pos_err": f32, "ik_ok": i64,     # 回放不做 IK: 恒 0.0 / 1
        "idle_left": i64, "idle_right": i64,
        "episode_success": i64, "failure_type": s1,
    }
    args.root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=features, root=args.root,
        robot_type="dual_nero_traj_replay",     # ★ 与 rollout 的 robot_type 不同
        use_videos=True, image_writer_threads=4,
        batch_encoding_size=10**9 if args.defer_encode else 1)


def record_frame(ds, args, prompt, q_now, target, frames, frame_idx, first_of_ep, traj_fps):
    ds.add_frame({
        "observation.state": np.asarray(q_now, dtype=np.float32),
        "action": np.asarray(target, dtype=np.float32),
        "observation.state_ee20": np.asarray(D.state20_from_joints(q_now), dtype=np.float32),
        "action_ee20": np.asarray(D.state20_from_joints(target), dtype=np.float32),
        "observation.images.third_view": frames["observation.images.third"],
        "observation.images.left_wrist": frames["observation.images.left_wrist"],
        "observation.images.right_wrist": frames["observation.images.right_wrist"],
        "subtask_instance_id": np.array([1], dtype=np.int64),
        "prompt_index": np.array([1], dtype=np.int64),
        "prompt_text": prompt, "prompt_text_zh": prompt,
        "subtask_start": np.array([1 if first_of_ep else 0], dtype=np.int64),
        "subtask_end": np.array([0], dtype=np.int64),
        "data_source": "traj_replay",
        "traj_name": args.name,
        "traj_fps": np.array([int(traj_fps)], dtype=np.int64),
        "ik_pos_err": np.array([0.0], dtype=np.float32),
        "ik_ok": np.array([1], dtype=np.int64),
        "idle_left": np.array([0], dtype=np.int64),
        "idle_right": np.array([0], dtype=np.int64),
        "episode_success": np.array([-1], dtype=np.int64),   # 结束时回填
        "failure_type": "pending",
    }, task=prompt, timestamp=frame_idx / FPS)


def finish_episode(ds, success: bool, failure_type: str, n: int) -> None:
    buf = ds.episode_buffer
    size = int(buf["size"])
    for j in range(size):
        buf["episode_success"][j] = np.array([1 if success else 0], dtype=np.int64)
        buf["failure_type"][j] = failure_type
    if size:
        buf["subtask_end"][size - 1] = np.array([1], dtype=np.int64)
    ep = ds.meta.total_episodes
    ds.save_episode()
    print(f"\n✓ 已存 episode {ep:06d}  {size} 帧  "
          f"{'成功' if success else '失败/' + failure_type}", flush=True)


def discard_episode(ds) -> None:
    if int(ds.episode_buffer["size"]) > 0:
        if getattr(ds, "image_writer", None) is not None:
            ds.image_writer.wait_until_done()   # 防 rmtree 撞上异步写图线程
        ds.clear_episode_buffer()
    print("  ✗ 已丢弃本条", flush=True)


# ── 回放 ────────────────────────────────────────────────────────────────────
def load_traj(name: str) -> dict:
    p = TRAJ_DIR / f"{name}.json"
    if not p.exists():
        avail = [q.stem for q in sorted(TRAJ_DIR.glob("*.json"))]
        sys.exit(f"✘ 没有这条轨迹: {p}\n   已录的: {avail}")
    return json.loads(p.read_text(encoding="utf-8"))


def resample(d: dict, replay_fps: int) -> tuple[dict, int]:
    """把原始录制(通常 100Hz)重采样到回放/录制帧率。"""
    F = d["frames"]
    src_t = np.array([f["t"] for f in F])
    n_out = max(int(round(d["duration_s"] * replay_fps)), 2)
    out_t = np.linspace(src_t[0], src_t[-1], n_out)
    res = {}
    for side in ("left", "right"):
        if side not in d["arms"]:
            continue
        A = np.array([f[side] for f in F])
        res[side] = np.stack([np.interp(out_t, src_t, A[:, j]) for j in range(7)], axis=1)
    return res, n_out


def replay_once(ds, arms, cameras, args, d, res, n_out, keypad) -> tuple[str, int]:
    """回放一遍并录制。返回 (结局, 帧数)。结局: done/abort/f/g。"""
    q0 = np.asarray(arms.read_state(), dtype=np.float64)[:16]
    cmd16 = q0.copy()
    period = 1.0 / args.replay_fps
    outcome = "done"
    frame_idx = 0

    print(f"\n▶ 回放「{args.name}」 {n_out} 帧 @ {args.replay_fps}Hz "
          f"({n_out / args.replay_fps:.1f}s)", flush=True)
    print("   回车/q=急停   f=判失败并停   g=丢弃", flush=True)

    for k in range(n_out):
        tick = time.monotonic()
        q_now = np.asarray(arms.read_state(), dtype=np.float64)[:16]
        target = cmd16.copy()
        for side, sl in (("left", slice(0, 7)), ("right", slice(8, 15))):
            if side not in res:
                continue
            step = np.clip(res[side][k] - cmd16[sl], -args.rate_limit, args.rate_limit)
            nxt = cmd16[sl] + step
            if args.stall_gap > 0:      # 抗积分饱和: 指令别跑到实际位置前面太多
                nxt = np.clip(nxt, q_now[sl] - args.stall_gap, q_now[sl] + args.stall_gap)
            target[sl] = nxt
        cmd16 = target

        frames = cameras.read_frames()
        record_frame(ds, args, args.prompt, q_now, target, frames, frame_idx,
                     frame_idx == 0, d["fps"])
        frame_idx += 1

        if args.execute:
            arms.send_joints(target)
            arms.send_grippers(target, args.grip_force)

        key = (keypad.get(0.0) or "").lower()
        if key in ("\r", "\n", "q"):
            print("\n🛑 急停: 停发运动指令, 保持使能钉住当前位姿(不下电)。", flush=True)
            if args.execute:
                try:
                    arms.send_joints(np.asarray(arms.read_state(), dtype=np.float64)[:16])
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠ 钉位失败({exc}); 仍未下电, 臂保持使能", flush=True)
            outcome = "abort"
            break
        if key in ("f", "g"):
            outcome = key
            break

        if (k + 1) % args.replay_fps == 0:
            print(f"     {(k+1)/args.replay_fps:5.1f}s / {n_out/args.replay_fps:.1f}s",
                  end="\r", flush=True)
        dt = time.monotonic() - tick
        if dt < period:
            time.sleep(period - dt)

    print(f"\n   回放结束: {frame_idx} 帧", flush=True)
    return outcome, frame_idx


def encode_pending(ds, args) -> None:
    if ds is None or not args.defer_encode:
        return
    try:
        n = int(ds.meta.total_episodes)
    except Exception:  # noqa: BLE001
        return
    if n <= 0:
        return
    print(f"\n══ 编码视频({n} 条)══  中断也会编, 请等它跑完", flush=True)
    try:
        ds.batch_encode_videos(0, n)
        print("  ✓ 视频编码完成", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✘ 编码失败: {exc}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="手教轨迹边回放边采集(相机 + 关节角)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="轨迹名(不含 .json)")
    ap.add_argument("--list", action="store_true", help="列出已录轨迹")
    ap.add_argument("--execute", action="store_true",
                    help="真回放 + 真录(默认干跑: 不发运动指令、不建数据集)")
    ap.add_argument("--repeat", type=int, default=1, help="连录几条")
    ap.add_argument("--prompt", default=None,
                    help="写进数据集的任务描述。不给则用 'replay teaching trajectory <名字>'")
    ap.add_argument("--arms", default="left,right")
    ap.add_argument("--replay-fps", type=int, default=FPS,
                    help=f"回放/录制帧率。默认 {FPS} = 数据集 fps, 改了时间轴就对不上")
    ap.add_argument("--rate-limit", type=float, default=0.05,
                    help="每 tick 最大关节步进(rad)")
    ap.add_argument("--stall-gap", type=float, default=0.12,
                    help="抗积分饱和: 指令最多超前实际位置多少 rad。0 = 关闭")
    ap.add_argument("--grip-force", type=float, default=1.0)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--defer-encode", action=argparse.BooleanOptionalAction, default=True,
                    help="录完统一编码视频(默认开)。--no-defer-encode 则每条录完就编")
    ap.add_argument("--mock-robot", action="store_true")
    ap.add_argument("--mock-cameras", action="store_true")
    ap.add_argument("--speed-percent", type=int, default=20)
    ap.add_argument("--enable-drift-abort", type=float, default=1.5)
    ap.add_argument("--confirm-enable-risk", action=argparse.BooleanOptionalAction,
                    default=True)
    # 设备(与 rollout 客户端同名同默认值, 直接喂给 D.DeployArms / D.CameraSet)
    ap.add_argument("--left-can", default="auto")
    ap.add_argument("--right-can", default="auto")
    ap.add_argument("--left-firmware", default="v120")
    ap.add_argument("--right-firmware", default="v120")
    ap.add_argument("--can-interface", default="socketcan")
    ap.add_argument("--can-bitrate", type=int, default=1_000_000)
    ap.add_argument("--can-timeout", type=float, default=1.0)
    ap.add_argument("--third-camera", default=None)
    ap.add_argument("--left-wrist-camera", default=None)
    ap.add_argument("--right-wrist-camera", default=None)
    ap.add_argument("--camera-width", type=int, default=640)
    ap.add_argument("--camera-height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=15)
    ap.add_argument("--camera-fourcc", default="MJPG")
    ap.add_argument("--camera-storage", default="video")
    ap.add_argument("--third-camera-width", type=int, default=None)
    ap.add_argument("--third-camera-height", type=int, default=None)
    ap.add_argument("--third-camera-fps", type=int, default=None)
    args = ap.parse_args()
    args.arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    return args


def main() -> None:
    args = parse_args()

    if args.list or not args.name:
        TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        ps = sorted(TRAJ_DIR.glob("*.json"))
        if not ps:
            print(f"(还没有录过轨迹。目录: {TRAJ_DIR})")
            return
        print(f"已录轨迹({TRAJ_DIR}):")
        for p in ps:
            x = json.loads(p.read_text(encoding="utf-8"))
            print(f"  {p.stem:14s} {x['n_frames']:6d} 帧 @{x['fps']:3d}Hz / "
                  f"{x['duration_s']:6.1f}s  臂: {','.join(x['arms'])}")
        if not args.name:
            print("\n用法: python3 traj_replay_record.py <名字> --execute")
        return

    d = load_traj(args.name)
    res, n_out = resample(d, args.replay_fps)
    res = {k: v for k, v in res.items() if k in args.arms}
    if not res:
        sys.exit(f"✘ 轨迹里没有 {args.arms} 的数据(它有 {d['arms']})")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.root is None:
        args.root = DATA_ROOT / f"{args.name}_{stamp}"
    if args.repo_id is None:
        args.repo_id = f"local/trajreplay_{args.name}_{stamp}"
    if args.prompt is None:
        args.prompt = f"replay teaching trajectory {args.name}"

    print(f"\n══ 边回放边采集 ══")
    print(f"  轨迹    {args.name}   {d['n_frames']} 帧 @{d['fps']}Hz / {d['duration_s']:.1f}s"
          f"  臂: {','.join(d['arms'])}")
    print(f"  回放    {n_out} 帧 @{args.replay_fps}Hz ({n_out/args.replay_fps:.1f}s)"
          f"   限幅 {args.rate_limit} rad/tick  抗饱和 {args.stall_gap} rad")
    print(f"  录制    {args.root}")
    print(f"  repo_id {args.repo_id}")
    print(f"  标记    data_source=traj_replay  traj_name={args.name}  "
          f"robot_type=dual_nero_traj_replay")
    print(f"  prompt  {args.prompt}")
    print(f"  条数    {args.repeat}")
    if args.replay_fps != FPS:
        print(f"  ⚠ --replay-fps {args.replay_fps} ≠ 数据集 fps {FPS} —— "
              f"时间轴会对不上, 除非你清楚在做什么", flush=True)
    if not args.execute:
        print("\n  (干跑: 不发运动指令, 不建数据集。加 --execute 真跑)", flush=True)
        return

    D.autodetect_devices(args)
    arms = D.MockArms() if args.mock_robot else D.DeployArms(args)
    cameras = D.CameraSet(D.build_camera_configs(args))
    ds = None
    n_saved = 0
    keypad = None
    try:
        cameras.open()
        ds = create_dataset(args)
        keypad = C.KeyPad()
        with keypad:
            q0 = D.safe_enable(arms, keypad, args)
            if q0 is None:
                print("  已取消。", flush=True)
                return
            for take in range(args.repeat):
                print(f"\n──── 第 {take+1}/{args.repeat} 条 ────", flush=True)
                outcome, n = replay_once(ds, arms, cameras, args, d, res, n_out, keypad)
                if outcome == "g" or n == 0:
                    discard_episode(ds)
                elif outcome == "f":
                    finish_episode(ds, False, "user_marked_fail", n); n_saved += 1
                elif outcome == "abort":
                    finish_episode(ds, False, "aborted", n); n_saved += 1
                    print("  (急停 → 本条按失败存, 不再继续)", flush=True)
                    break
                else:
                    finish_episode(ds, True, "none", n); n_saved += 1
                if take + 1 < args.repeat:
                    print("\n  把现场复位到轨迹起点, 按空格继续下一条 (q=结束)", flush=True)
                    keypad.drain()
                    k = ""
                    while k not in (" ", "q", "\r", "\n"):
                        k = (keypad.get(0.2) or "").lower()
                    if k != " ":
                        break
    finally:
        # ★ 先编码再关设备 —— 编码只用 CPU 和磁盘, 但必须保证走到
        try:
            encode_pending(ds, args)
        except Exception:  # noqa: BLE001
            pass
        try:
            cameras.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            arms.close()
        except Exception:  # noqa: BLE001
            pass
        print(f"\n本次保存 {n_saved} 条 → {args.root}", flush=True)


if __name__ == "__main__":
    main()
