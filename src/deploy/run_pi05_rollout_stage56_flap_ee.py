#!/usr/bin/env python3
"""Nero pi0.5 rollout 采集 —— stage5/6 合页封盖【末端位姿版·新数据】(7 段, 双臂, 20维位姿→IK→关节)。

模型: pi05_nero_stage56_flap_closing_ee_v2 / stage56_ee_run1 / step 19999 (69 集版, loss 0.0224)
  serve 默认 8030 (已挂在 GPU3)。
  100 集版 pi05_nero_stage56_flap_closing_ee_v100 仍在训练(GPU2), 训完可另起服务再用 --port 指过去。
录制内容(与人工采集口径一致, 便于合并/复训):
  observation.state / action : **16 维双臂关节**(左7+左爪+右7+右爪)  ← 你要的双臂关节信息
  三路相机: third_view / left_wrist / right_wrist
  另存 observation.state_ee20 / action_ee20 : 20 维绝对末端位姿(模型的真实输入/输出空间)
  逐帧还记 subtask_id / prompt / IK 残差 / IK 是否收敛 / 该臂是否 idle

标注: **只在整条结束时标一次成功/失败**(按你的要求, 不逐段标)。
  s = 成功保存    f = 失败保存(再选失败类型)    g = 丢弃整条

═══ 使用 ═══
  0. 起服务: GPU=0 PORT=8026 STEP=19999 bash <WORKSPACE>/serve_ee.sh   (在 <SERVER>)
  1. 清代理: export no_proxy=<POLICY_SERVER_IP> ; unset http_proxy https_proxy all_proxy ...
  2. 干跑:   python3 run_pi05_rollout_hezi_ee.py --mock-robot
  3. 只读:   python3 run_pi05_rollout_hezi_ee.py --host <POLICY_SERVER_IP>
  4. 真执行: python3 run_pi05_rollout_hezi_ee.py --host <POLICY_SERVER_IP> --execute --confirm-enable-risk

键位: 空格=开始一条 / 推进到下一段    b=退一段   p=暂停/继续
      s=整条成功保存   f=整条失败保存(选类型)   g=丢弃整条   回车/q=急停退出
相机与 CAN 按序列号自动检测。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pi05_deploy_hezi_ee as D  # noqa: E402  (复用: 设备检测/IK链路/限幅/相机/臂)
import run_pi05_deploy_hezi_merged as HZ  # noqa: E402  (7 段 prompt)
import rollout_boxpick_common as C  # noqa: E402  (复用: KeyPad/数据集样板)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ee_pose" / "tools"))
import ee_repr as E  # noqa: E402

# 退出码约定(run_chain_4stage 按它决定下一步):
#   0   本段正常跑完
#   20  你按了 f 判失败 → chain 立即停整条链, 并统一编码视频
#   130 Ctrl-C
EXIT_USER_FAIL = 20
# 模块级可变标志: 主循环在深层嵌套里置位, main() 末尾据此决定退出码。
_user_failed = [False]

FPS = 15
PERIOD = 1.0 / FPS
STATE_NAMES = ([f"left_j{i}" for i in range(1, 8)] + ["left_gripper_width"]
               + [f"right_j{i}" for i in range(1, 8)] + ["right_gripper_width"])
EE20_NAMES = (["left_x", "left_y", "left_z"] + [f"left_rot6d_{i}" for i in range(6)] + ["left_grip"]
              + ["right_x", "right_y", "right_z"] + [f"right_rot6d_{i}" for i in range(6)] + ["right_grip"])
_IMG = {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]}
FAILURES = (
    ("flap_not_closed", "盖子没合上/合不到位"),
    ("flap_bounced", "盖子弹回"),
    ("arm_collision", "机械臂碰撞(碰到箱体/对臂)"),
    ("wrong_stage_order", "段序错乱/该动的臂没动"),
    ("ik_or_limit", "IK解不出或卡限位"),
    ("box_moved", "箱子被推走/移位"),
    ("other", "其他"),
)


# ── stage5/6 合页封盖的 7 段 prompt(本文件自带, **不再借用 stage34 的**) ────────
# ⚠ 这里踩过一个大坑: 本文件是从 run_pi05_rollout_hezi_ee.py 复制来的, 那句
#     subtasks = HZ.load_subtasks(HZ.STAGE_ORDER_TRAIN)
#   忘了改 —— HZ 是 run_pi05_deploy_hezi_merged, 里面装的是 **stage34(折压侧翼)**
#   的 7 段提示词。结果: 跑的是 stage56 的模型(8030), 送进去的却是 stage34 的
#   prompt, 模型收到的是完全不相干的指令。
#   下面这份逐字取自 nero_stage56_flap_closing_ee_v2 的 prompt_text 列,
#   顺序按 parquet 里 stage_id 51→52→53→54→55→61→62 的真实执行序
#   (对应 tasks.jsonl 的 task_index [1,5,6,4,3,0,2] —— task_index 不是执行顺序)。
STAGE56_PROMPTS = [
    "Raise the right arm, move it over the flaps, and bring it close to the top surface of the carton.",
    "Open the right gripper and lower it so that the two fingers brace the front and rear flaps.",
    "Use the left arm to lift the left flap from the left side until it is nearly vertical, while the right arm keeps bracing the front and rear flaps.",
    "Use the left arm to fold the left flap inward and close it, while the right arm keeps bracing the front and rear flaps.",
    "Retract the right arm from above the carton and clear the flap area.",
    "Use the right arm to lift the right flap from the right side until it is nearly vertical.",
    "Use the right arm to fold the right flap inward and close it.",
]
STAGE56_STAGE_IDS = [51, 52, 53, 54, 55, 61, 62]


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
        "ik_pos_err": f32, "ik_ok": i64,
        "idle_left": i64, "idle_right": i64,
        "episode_success": i64, "failure_type": s1,
    }
    args.root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features,
                                 root=args.root, robot_type="dual_nero_hezi_closing_ee",
                                 use_videos=True, image_writer_threads=4,
                                 batch_encoding_size=getattr(args, 'batch_encoding_size', 1))


def record_frame(ds, sid: int, prompt: str, state16, action16, state20, action20,
                 frames, frame_idx: int, ik_err: float, ik_ok: bool,
                 idle_l: bool, idle_r: bool, first_of_ep: bool, first_of_sub: bool) -> None:
    ds.add_frame({
        "observation.state": np.asarray(state16, dtype=np.float32),
        "action": np.asarray(action16, dtype=np.float32),
        "observation.state_ee20": np.asarray(state20, dtype=np.float32),
        "action_ee20": np.asarray(action20, dtype=np.float32),
        "observation.images.third_view": frames["observation.images.third"],
        "observation.images.left_wrist": frames["observation.images.left_wrist"],
        "observation.images.right_wrist": frames["observation.images.right_wrist"],
        "subtask_instance_id": np.array([sid], dtype=np.int64),
        "prompt_index": np.array([sid], dtype=np.int64),
        "prompt_text": prompt, "prompt_text_zh": prompt,
        "subtask_start": np.array([1 if first_of_sub else 0], dtype=np.int64),
        "subtask_end": np.array([0], dtype=np.int64),
        "ik_pos_err": np.array([float(ik_err)], dtype=np.float32),
        "ik_ok": np.array([1 if ik_ok else 0], dtype=np.int64),
        "idle_left": np.array([1 if idle_l else 0], dtype=np.int64),
        "idle_right": np.array([1 if idle_r else 0], dtype=np.int64),
        "episode_success": np.array([-1], dtype=np.int64),   # 结束时回填
        "failure_type": "pending",
    }, task=prompt, timestamp=frame_idx / FPS)


def finish_episode(ds, args, success: bool, failure_type: str, seg_bounds: list) -> None:
    """整条结束: 回填成败 + 段边界, 落盘, 写 annotations。"""
    buf = ds.episode_buffer
    n = int(buf["size"])
    for j in range(n):
        buf["episode_success"][j] = np.array([1 if success else 0], dtype=np.int64)
        buf["failure_type"][j] = failure_type
    for (s, e) in seg_bounds:
        if e > s:
            buf["subtask_end"][min(e - 1, n - 1)] = np.array([1], dtype=np.int64)
    ep = ds.meta.total_episodes
    ds.save_episode()
    ann = args.root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    with open(ann / "episode_outcomes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"episode_index": ep, "frames": n,
                            "episode_success": bool(success), "failure_type": failure_type,
                            "segments": [{"subtask_id": i + 1, "start": s, "end": e}
                                         for i, (s, e) in enumerate(seg_bounds)],
                            "saved_at": datetime.now().isoformat(timespec="seconds")},
                           ensure_ascii=False) + "\n")
    print(f"\n✓ 已存 episode {ep:06d}  帧{n}  {'成功' if success else '失败/' + failure_type}", flush=True)


def choose_failure(keypad=None) -> str:
    """选失败类型。序号 1~N 选预设; 也可直接敲任意文字(中文/英文)当自定义原因记录。

    ⚠ 必须先把终端从 cbreak 恢复成行模式, 否则 KeyPad 关了回显, 你打的字看不见。
    读完再切回 cbreak, 不影响后续单键操作。
    """
    print("\n  失败类型:", flush=True)
    for i, (code, zh) in enumerate(FAILURES, 1):
        print(f"    {i}. {code:22s} {zh}")
    print("  可输入: 序号(1~%d) / 任意文字作自定义原因(中英文均可) / 直接回车=other"
          % len(FAILURES), flush=True)

    if keypad is not None:
        keypad.restore()          # 回到行模式 → 输入可见、可退格
    try:
        v = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        v = ""
    finally:
        if keypad is not None:
            keypad.__enter__()    # 切回 cbreak(restore 已把 _old 置空, 可重入)
            keypad.drain()

    if not v:
        print("  → 记为 other", flush=True)
        return "other"
    if v.isdigit() and 1 <= int(v) <= len(FAILURES):
        code = FAILURES[int(v) - 1][0]
        print(f"  → {code}", flush=True)
        return code
    custom = " ".join(v.split())[:120]      # 压掉换行/多余空白, 限长防污染字段
    print(f"  → 自定义原因已记录: {custom}", flush=True)
    return custom


# 本段结束后的过渡轨迹名(trajectories/<名字>.json)。
# 对应 E3 stage56 → 收尾。录了同名轨迹就优先回放它, 没录则回退 goto_home.py 直线插值归位。
# ⚠ 这个常量之前漏了定义 —— goto_home_inline() 一调就 NameError, 整段过渡直接崩。
DEFAULT_TRAJ = "transition_e3_end"


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


def main() -> None:
    ap = argparse.ArgumentParser(description="hezi 封盖末端位姿版 rollout 采集(7段, 整条标注)")
    ap.add_argument("--host", default="<POLICY_SERVER_IP>")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--repo-id", default="local/nero_stage56_flap_ee_rollout")
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
                    help="切换器的专家编号。默认按本客户端固定值 3")
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
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True,
                    help="真执行(默认开; --no-execute 回到只推理只录不发运动)")
    ap.add_argument("--confirm-enable-risk", action=argparse.BooleanOptionalAction, default=True,
                    help="--execute 必须(默认开)")
    ap.add_argument("--mock-robot", action="store_true")
    ap.add_argument("--mock-cameras", action="store_true")
    ap.add_argument("--disable-at-end", action="store_true",
                    help="每条结束后下电(默认不下电: 该硬件 disable 抱闸不保持, 臂会砸下来)")
    # 整块 50 帧用满(原 40/45)。网络抖时推理要 0.5~15s, 用满一块 = 每次推理换来
    # 3.33s 连续运动(原 2.67/3.00s), 占空比更高、动作更连贯。
    ap.add_argument("--execute-horizon", type=int, default=50)
    # 0.06 太紧: stage34 训练实测逐帧 |Δq| 的 p99 = 0.052(段32)/0.073(段41)/0.080(段43),
    # 卡 0.06 会把每段最关键的发力段削掉 —— 表现为"同一段反复重推、位移不收敛"。
    # 0.12 覆盖所有段的 p99, 也与 run_pi05_deploy_hezi_ee.py 的默认值一致。
    # 训练 p99 = 0.050 rad/tick(stage34) / 0.041(stage56)。
    # 0.18 rad/tick = 2.7 rad/s, 是 p99 的 3.6 倍。
    ap.add_argument("--rate-limit", type=float, default=0.18)
    # 0.15 rad = 8.6° 太松: 允许指令超出训练包络 8.6°, 等于把臂往死里顶。
    # 收到 0.02 rad(1.1°), 只留数值裕量, 不再越界。
    ap.add_argument("--limit-margin", type=float, default=0.02)
    # ★ 抗积分饱和: 指令最多超前实际位置多少弧度。位置伺服里 力矩 ∝ 位置误差,
    #   夹住它就等于夹住堵转力矩 —— 顶到箱子会"轻轻顶着"而不是越顶越狠到跳闸。
    #   0.10 rad ≈ 5.7°。设 0 关闭该保护。
    ap.add_argument("--stall-gap", type=float, default=0.10)
    ap.add_argument("--grip-force", type=float, default=1.0)
    # 10% → 20%, 与 E0/E1 一致。
    ap.add_argument("--speed-percent", type=int, default=20)
    ap.add_argument("--enable-drift-abort", type=float, default=1.5)
    ap.add_argument("--ik-ns-gain", type=float, default=0.30)
    ap.add_argument("--ik-damping", type=float, default=1e-2)
    ap.add_argument("--ik-max-iter", type=int, default=100)
    ap.add_argument("--ik-tol-pos", type=float, default=5e-3)
    ap.add_argument("--ik-tol-rot", type=float, default=0.5 * np.pi / 180)
    # 静止臂死区阈值 —— 真机实测(ee_trace.npz, 7段):
    #   该动的臂 整段位移 0.10~0.23 m / 转角 9~26°
    #   不该动的臂        0.0007~0.023 m / 0.1~2°   ← 模型的噪声漂移
    # 原默认 3mm/0.3° 卡在噪声区间里, 漏掉 5~23mm 的漂移 → "不该动的臂自己动了 1.6~6.8°"。
    # 提到 50mm/5° 后两类干净分开(23mm < 50mm < 100mm)。
    ap.add_argument("--idle-dp", type=float, default=5e-2, help="静止臂死区: 整段位移阈值(m)")
    ap.add_argument("--idle-dth", type=float, default=5.0, help="静止臂死区: 整段转角阈值(度)")
    ap.add_argument("--no-idle-freeze", action="store_true",
                    help="关掉静止臂死区(诊断'某臂不动'时用: 强制两臂都走 IK)")
    ap.add_argument("--left-can", default="auto")
    ap.add_argument("--right-can", default="auto")
    # DeployArms 需要的 CAN 参数(默认值与 run_pi05_deploy_hezi_ee.py 一致)
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
    # build_camera_configs 需要的第三视角覆盖项(None = 沿用通用值)
    ap.add_argument("--third-camera-width", type=int, default=None)
    ap.add_argument("--third-camera-height", type=int, default=None)
    ap.add_argument("--third-camera-fps", type=int, default=None)
    args = ap.parse_args()
    # 局域网绕开系统代理 —— 必须在任何联网之前。不加的话连推理服务会报
    # InvalidMessage: did not receive a valid HTTP response(其实是被代理吃了)。
    from proxy_bypass import bypass_proxy  # noqa: PLC0415
    bypass_proxy(getattr(args, "host", "") or "")

    if args.execute and not args.confirm_enable_risk:
        raise SystemExit("--execute 需要 --confirm-enable-risk")
    if args.root is None:
        # ── 全过程串跑: 统一落在 wholeprocess/<本次运行ID>/<阶段>/ 下, 不与单阶段旧采集混淆。
        #    WP_RUN 由 run_chain 统一下发; 单独跑某一阶段时自动生成, 行为与以前一致。
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = os.environ.get("WP_RUN") or f"run_{stamp}"
        args.root = Path("<PROJECT_ROOT>/lerobot_data/Rlinf/data") / "wholeprocess" / run_id / "e3_stage56"
        args.repo_id = f"local/wholeprocess_{run_id}_e3_stage56"
    if args.no_idle_freeze:
        args.idle_dp, args.idle_dth = -1.0, -1.0   # 永远判不成静止 → 两臂都走 IK
    args.batch_encoding_size = 1000000000 if args.defer_encode else 1
    D.autodetect_devices(args)

    from openpi_client import websocket_client_policy
    # 无线偶发丢包会让首次连接抛 "No route to host"/超时, 这里重试几次再放弃
    policy = None
    for attempt in range(1, 6):
        try:
            policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
            break
        except OSError as exc:
            print(f"  连接 {args.host}:{args.port} 失败({attempt}/5): {exc} — 2s 后重试", flush=True)
            time.sleep(2)
    if policy is None:
        raise SystemExit(f"✘ 连不上 {args.host}:{args.port}。检查: ① serve 是否在跑 "
                         f"② 有线用 20.198 / 无线用 10.198 ③ 是否 unset 了代理")
    subtasks = STAGE56_PROMPTS      # ★ 本任务自己的 prompt, 不是 stage34 的
    ds = create_dataset(args)
    arms = D.MockArms() if args.mock_robot else D.DeployArms(args)
    cameras = D.CameraSet(D.build_camera_configs(args))
    n_saved = 0
    print(f"\n{'='*88}\n末端位姿版 rollout 采集 · {len(subtasks)} 段 · {'真执行' if args.execute else '只读dry-run'}"
          f"\n数据集: {args.root}\n{'='*88}")
    print("键位: 空格=开始/推进下一段  b=退一段  p=暂停  s=整条成功保存  f=整条失败保存  g=丢弃  回车/q=急停\n")
    try:
        cameras.open()
        # ── 子任务自动切换 ────────────────────────────────────────────────────
        # 进度头只吃当前帧, 判断「这个子任务做到哪了」。切换时刻由它给, 人按空格仍可覆盖。
        # ⚠ 它要跟**相机帧**走(15Hz), 不跟策略推理走 —— 见 switcher_service.py 的说明。
        sw = None
        if getattr(args, "auto_switch", False):
            from switcher_client import SwitcherClient  # noqa: PLC0415
            sw = SwitcherClient(expert=(args.switch_expert if args.switch_expert is not None
                                        else 3),
                                device=args.switch_device)
        with C.KeyPad() as keypad:
            while True:
                q16 = D.wait_state_fresh(arms)
                if q16 is None:
                    print("🛑 关节反馈不完整"); break
                print(f"\n就绪(第 {ds.meta.total_episodes:06d} 条)。摆好机械臂+箱子, 空格=开始, 回车/q=退出", flush=True)
                keypad.drain()
                k = " "
                if not getattr(args, "auto_start", False):
                    k = ""
                    while k not in (" ", "\r", "\n", "q"):
                        k = (keypad.get(0.2) or "").lower()
                else:
                    print("(--auto-start: 直接开跑, 不等空格)", flush=True)
                if k in ("\r", "\n", "q"):
                    break
                if args.execute:
                    base = D.safe_enable(arms, keypad, args)
                    if base is None:
                        continue
                    last_cmd = base.copy()
                else:
                    last_cmd = np.asarray(arms.read_state(), dtype=np.float64)

                frame_idx = 0
                sid = 0
                last_sid = -1
                seg_bounds = []
                seg_start = 0
                action = "timeout"
                first_of_ep = True
                paused = False
                while sid < len(subtasks):
                    prompt = subtasks[sid]
                    if sid != last_sid:
                        print(f"\n● 第{sid+1}/{len(subtasks)}段: {prompt[:56]}", flush=True)
                        print("   (本段会持续推理执行; 按 空格 才切下一段, b 退一段)", flush=True)
                        last_sid = sid
                    else:
                        print(f"  ↻ 第{sid+1}段续", flush=True)
                    q16 = np.asarray(arms.read_state(), dtype=np.float64)
                    frames = cameras.read_frames()
                    s20 = D.state20_from_joints(q16)
                    t0 = time.monotonic()
                    chunk20 = np.asarray(policy.infer(D.build_obs(s20, frames, prompt))["actions"],
                                         dtype=np.float64)
                    ec = D.EEChunk(s20, chunk20, q16, args)
                    n_exec = min(args.execute_horizon, len(chunk20))
                    print(f"  推理{(time.monotonic()-t0)*1000:.0f}ms  {ec.summary()}  执行{n_exec}帧"
                          f"{'' if args.execute else '(dry)'}", flush=True)
                    D.reset_clamp_stat()          # 本块的限位/限速统计清零
                    ee_before = s20.copy()        # 用于块末对比"要求 vs 实走"
                    first_of_sub = True
                    advance = None
                    for i in range(n_exec):
                        tick = time.monotonic()
                        q_ik, note = ec.solve(i, last_cmd)
                        # 先读实际位置 —— clamp_target 要用它做抗积分饱和(防堵转过流)
                        state_now = np.asarray(arms.read_state(), dtype=np.float64)
                        target = D.clamp_target(q_ik, last_cmd, args.rate_limit,
                                                args.limit_margin,
                                                q_actual=state_now,
                                                stall_gap=args.stall_gap)
                        frames = cameras.read_frames()
                        record_frame(ds, sid + 1, prompt, state_now, target,
                                     D.state20_from_joints(state_now), chunk20[i], frames,
                                     frame_idx, ec.max_pos_err, note == "",
                                     ec.idle["left"], ec.idle["right"], first_of_ep, first_of_sub)
                        frame_idx += 1
                        first_of_ep = first_of_sub = False
                        if args.execute and not paused:
                            arms.send_joints(target)
                            arms.send_grippers(target, args.grip_force)
                        last_cmd = target
                        key = (keypad.get(0.0) or "").lower()
                        # 自动切换: 每帧都要 step(15Hz), 漏帧等于变相加速、判据会偏。
                        # 人按了键就按人的来 —— 手动优先, 这一步不合成。
                        if sw is not None:
                            try:
                                _r = sw.step(frames, D.state20_from_joints(state_now))
                                if _r["switched"] and not key:
                                    print("\n  [自动切换] E3 stage56 第%d段完成%s"
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
                                        _qh = np.asarray(arms.read_state(), dtype=np.float64)[:16].copy()
                                        arms.send_joints(_qh)
                                    except Exception as _e:  # noqa: BLE001
                                        print(f"  ⚠ 钉位失败({_e}); 仍未下电, 臂保持使能", flush=True)
                                advance = "abort"; break
                            if key == " ":
                                advance = "next"; break
                            if key == "b":
                                advance = "back"; break
                            if key in ("s", "f", "g"):
                                if key == "f":
                                    _user_failed[0] = True
                                advance = key; break
                            if key == "p":
                                paused = not paused
                                print(f"  {'⏸暂停' if paused else '▶继续'}", flush=True)
                        dt = time.monotonic() - tick
                        if dt < PERIOD / max(args.speed, 1e-6):
                            time.sleep(PERIOD / max(args.speed, 1e-6) - dt)
                    # ── 块末诊断: 模型要求走多远 vs 实际走了多远 ────────────────
                    #    两者持续对不上 = 手臂跟不上 chunk, 表现就是"同一段反复重推、
                    #    位移不收敛"。再配合限位/限速计数, 能直接分辨是被夹住了还是被拖慢了。
                    try:
                        ee_after = D.state20_from_joints(
                            np.asarray(arms.read_state(), dtype=np.float64))
                        moved = {a: float(np.linalg.norm(ee_after[o:o + 3] - ee_before[o:o + 3]))
                                 for a, o in (("left", 0), ("right", 10))}
                        parts = []
                        for a, zh in (("left", "左"), ("right", "右")):
                            if ec.idle[a]:
                                continue
                            want, got = ec.dp[a], moved[a]
                            pct = 100 * got / want if want > 1e-6 else 100.0
                            parts.append(f"{zh}要{want*100:.1f}cm/实走{got*100:.1f}cm({pct:.0f}%)")
                        if parts:
                            print("    ↳ " + "  ".join(parts) + D.clamp_stat_note(), flush=True)
                    except Exception as _e:  # noqa: BLE001
                        print(f"    ↳ (跟随度诊断失败: {_e})", flush=True)

                    if advance == "abort":
                        action = "abort"; break
                    if advance in ("s", "f", "g"):
                        action = advance; break
                    if advance in ("next", "back"):
                        # 只有你按了空格/b 才切段并记录段边界
                        seg_bounds.append((seg_start, frame_idx))
                        seg_start = frame_idx
                        sid = sid + 1 if advance == "next" else max(0, sid - 1)
                    # advance is None: 这一块 45 帧执行完但你没按键 → 留在本段, 重新推理接着做
                else:
                    if getattr(args, "defer_label", False):
                        # 串跑: 全部子任务跑完 = 本大阶段完成 → **自动存, 不打断**。
                        # 成功/失败的标注推迟到整条链结束时统一问一次, 由
                        # run_chain_4stage --label 回填进 parquet 与 annotations。
                        # 这样从头到尾只有空格一个键, 不会每段被 s/f/g 截停。
                        print("\n  ✓ 全部子任务跑完 → 自动保存(标注留到最后统一做)", flush=True)
                        action = "s"
                    else:
                        print("\n  ✓ 7 段跑完。s=成功保存  f=失败保存  g=丢弃", flush=True)
                        keypad.drain()
                        k2 = ""
                        while k2 not in ("s", "f", "g", "q", "\r", "\n"):
                            k2 = (keypad.get(0.2) or "").lower()
                        action = "abort" if k2 in ("q", "\r", "\n") else k2
                    if action == "f":
                        _user_failed[0] = True

                if seg_start < frame_idx:
                    seg_bounds.append((seg_start, frame_idx))
                if action == "abort" or action == "g" or frame_idx == 0:
                    if int(ds.episode_buffer["size"]) > 0:
                        if getattr(ds, "image_writer", None) is not None:
                            ds.image_writer.wait_until_done()   # 防 rmtree 撞上异步写图线程
                        ds.clear_episode_buffer()
                    print("  ✗ 已丢弃本条", flush=True)
                    if action == "abort":
                        break
                    continue
                if action == "s":
                    finish_episode(ds, args, True, "none", seg_bounds); n_saved += 1
                elif action == "f":
                    finish_episode(ds, args, False, choose_failure(keypad), seg_bounds); n_saved += 1
                if getattr(args, "once", False):
                    # 串跑: 空格走完最后一个子任务 = 本大阶段结束 → 退出本客户端,
                    # 由 run_chain_4stage 推进到下一大阶段。放在 if args.execute 之外,
                    # 只读模式(不加 --execute)同样生效。
                    print("\n✓ --once: 本阶段完成, 退出以推进到下一大阶段。", flush=True)
                    goto_home_inline(args)
                    break
                if args.execute:
                    # ⚠ 不再 estop(): 该硬件 disable 后抱闸不保持, 臂会直接砸下来。
                    # 改为「保持使能 + 钉住当前位姿」, 臂停在原地不下落, 人可从容复位。
                    # 真要下电请物理支撑后手动执行, 或加 --disable-at-end。
                    try:
                        q_hold = arms.read_state()[:16].copy()
                        arms.send_joints(q_hold)
                        print("  (保持使能并钉住当前位姿, 臂不会下落; 复位后按空格开下一条)", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ⚠ 钉位失败({exc}); 未下电, 臂仍使能", flush=True)
                    if getattr(args, "disable_at_end", False):
                        arms.estop()
                        print("  (--disable-at-end: 已下电, 注意臂可能下落)", flush=True)
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
        arms.close()
        print(f"\n本次保存 {n_saved} 条 → {args.root}", flush=True)
    if _user_failed[0]:
        # 把"你按了 f"这个信号传给 run_chain_4stage: 它据此停链 + 统一编码。
        print("\n(按 f 判失败 → 退出码 %d, 整条链将停止并保存视频)" % EXIT_USER_FAIL,
              flush=True)
        sys.exit(EXIT_USER_FAIL)


if __name__ == "__main__":
    main()
