#!/usr/bin/env python3
"""Nero 双臂 pi0.5 部署客户端 —— hezi 封盖【末端位姿版】(20 维绝对位姿输入 / 相对位姿输出 + IK)。

模型: pi05_nero_hezi_closing_ee_v1 / exp hezi_ee_run1 / step 19999 (最终 loss 0.0207)
  ckpt: <CKPT_ROOT_EE>/pi05_nero_hezi_closing_ee_v1/hezi_ee_run1/19999
  数据: 100 集(70+30 合并)/87649 帧/15Hz, 7 段封盖 stage prompt, 三路 RGB 全开无 mask

与关节版(8023)的差别 —— 只在 state/action 表示, 相机和 prompt 完全一样:
  输入 state : **20 维绝对末端位姿** = 每臂 (3 位置 + rot6D + 1 爪宽), 世界系, TCP=指尖 0.138
               客户端读 16 维关节 → FK → 20 维送出去
  服务端输出 : **20 维绝对末端位姿**(EEAbsoluteActions 已把 14 维相对量还原成绝对)
  客户端     : 绝对末端位姿 → **IK** → 16 维关节目标 → 限幅 → 下发
位姿口径(与训练逐字一致, 改这块要回 ee_pose/tools/ee_repr.py 对):
    from fk_nero import fk, TCP_FINGERTIP        # 0.138 = 指尖对称轴最末端
    T = fk(q_rad, side="right", tcp=TCP_FINGERTIP)   # 世界系 4x4

IK(ee_pose/tools/ik_nero.py, selftest 22/22): 阻尼最小二乘 + **精确零空间投影**。两个坑:
  · 零空间必须用 I − J⁺J(SVD 伪逆); 用带阻尼的 I − Jp·J 会漏进末端任务, 形成残差地板。
  · **q_seed(迭代起点) 与 q_ref(零空间锚点) 必须分开** —— q_ref 固定为本 chunk 起始的实测
    关节角, 不能跟着 seed 走, 否则冗余维随机游走(闭环一圈漂 65~155 mrad, 固定后 3.2 mrad)。
静止臂死区: 整段 Δ 低于阈值的臂**冻结关节指令、根本不调 IK**。本任务 71% 的 chunk 有一条臂
  整段不动, 这条既省一半 IK 调用, 又保证那条臂严格不动(见 ee_pose/tools/replay_ee_ik.py)。

相机与 CAN 全部**按序列号自动检测**(参照 HY 采集脚本的 resolve_*_by_serial), 免疫插拔重枚举:
  第三视角 <THIRD_CAM_SERIAL> / 左腕 <LEFT_WRIST_CAM_SERIAL> / 右腕 <RIGHT_WRIST_CAM_SERIAL>
  左臂 CAN <LEFT_CAN_SERIAL> / 右臂 CAN <RIGHT_CAN_SERIAL>

模式(从安全到危险, 默认最安全):
  --fake-obs   无臂无相机: 合成观测, 纯验证推理链路(会打印 IK 还原结果)
  (默认 dry-run) 真相机+真臂只读: 完整闭环含 IK, 绝不发 CAN 运动命令
  --execute    真执行(需 --confirm-enable-risk, 运行时输入 DEPLOY)
  --prepare-only 只做安全使能流程后退出
运行中键盘: n/空格=下一段  1-7=跳段  b=退一段  p=暂停/继续  回车 或 q=急停(停发+下电抱闸)
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NERO_ROOT = Path(__file__).resolve().parents[1]
# 把本文件所在目录放进 sys.path: 同级的 run_pi05_deploy_hezi_merged 要 import 得到,
# 从任何 cwd 跑都行(不加的话只有在 deploy/ 目录下跑才不报 ModuleNotFoundError)。
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(NERO_ROOT / "nero_control"))
sys.path.insert(0, str(NERO_ROOT / "lerobot_data" / "lerobot_tools"))
sys.path.insert(0, str(NERO_ROOT / "deploy" / "openpi-client" / "src"))
sys.path.insert(0, str(NERO_ROOT / "ee_pose" / "tools"))      # ee_repr / ik_nero

for _k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = "<POLICY_SERVER_IP>,<POLICY_SERVER_IP>,127.0.0.1,localhost"

import ee_repr as E  # noqa: E402  (批量 FK + rot6d/rotvec + 20↔14 互转)
import ik_nero as K  # noqa: E402  (DLS IK + 精确零空间投影 + 静止臂死区)
from camera_utils import CameraSet, add_camera_args, build_camera_configs  # noqa: E402
from collect_dual_nero import KeyPad  # noqa: E402

import run_pi05_deploy_hezi_merged as HZ  # noqa: E402  (复用: 臂控制层 / 7 段 prompt)
from run_pi05_deploy_hezi_merged import DeployArms, MockArms, FPS, PERIOD, GRIP, JOINTS  # noqa: E402

H = 50           # 服务端动作块长度
EE_DIM = 20      # 服务端返回的绝对末端位姿维度

# ── 设备序列号(固定不变, 与 HY 采集配置/rollout 脚本三处一致) ──────────────────
CAM_SERIALS = {"third": "<THIRD_CAM_SERIAL>", "left_wrist": "<LEFT_WRIST_CAM_SERIAL>", "right_wrist": "<RIGHT_WRIST_CAM_SERIAL>"}
CAN_SERIALS = {"left": "<LEFT_CAN_SERIAL>", "right": "<RIGHT_CAN_SERIAL>"}
CAN_FALLBACK = {"left": "can1", "right": "can0"}


# ── 按序列号自动检测(照抄 HY 采集脚本的做法, 免疫 USB/CAN 重枚举) ──────────────
def _udev_prop(args_: list[str]) -> str | None:
    try:
        return subprocess.check_output(["udevadm", "info", "-q", "property", *args_],
                                       text=True, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return None


def _serial_of(out: str | None) -> str | None:
    if not out:
        return None
    for line in out.splitlines():
        if line.startswith("ID_SERIAL_SHORT="):
            return line.split("=", 1)[1].strip()
    return None


def resolve_can_by_serial(serial: str) -> str | None:
    for path in sorted(glob.glob("/sys/class/net/can*")):
        if _serial_of(_udev_prop(["-p", os.path.realpath(path)])) == serial:
            return os.path.basename(path)
    return None


def resolve_video_by_serial(serial: str) -> str | None:
    def key(p):
        n = p.removeprefix("/dev/video")
        return int(n) if n.isdigit() else 10_000
    for path in sorted(glob.glob("/dev/video*"), key=key):
        if _serial_of(_udev_prop(["-n", path])) == serial:
            return path
    return None


def autodetect_devices(args: argparse.Namespace) -> None:
    """把没显式指定的相机/CAN 按序列号解析出来; 解析不到就明确报出来, 不静默用错设备。"""
    print("按序列号自动检测设备 …", flush=True)
    missing = []
    for name, attr in (("third", "third_camera"), ("left_wrist", "left_wrist_camera"),
                       ("right_wrist", "right_wrist_camera")):
        if getattr(args, attr) is not None or args.mock_cameras:
            continue
        dev = resolve_video_by_serial(CAM_SERIALS[name])
        setattr(args, attr, dev)
        print(f"  相机 {name:11s} serial={CAM_SERIALS[name]} → {dev or '未找到'}", flush=True)
        if dev is None:
            missing.append(f"相机 {name}({CAM_SERIALS[name]})")
    for side in ("left", "right"):
        attr = f"{side}_can"
        if getattr(args, attr) != "auto":
            continue
        can = resolve_can_by_serial(CAN_SERIALS[side])
        setattr(args, attr, can or CAN_FALLBACK[side])
        print(f"  CAN  {side:11s} serial={CAN_SERIALS[side]} → "
              f"{can or CAN_FALLBACK[side] + '(未找到, 用兜底)'}", flush=True)
        if can is None and not args.mock_robot:
            missing.append(f"CAN {side}({CAN_SERIALS[side]})")
    if missing and not (args.mock_cameras and args.mock_robot):
        print(f"\n⚠ 以下设备按序列号没找到: {', '.join(missing)}", flush=True)
        print("  检查 USB/CAN 是否插好、can 接口是否 up(ip link set canX up type can bitrate 1000000)。",
              flush=True)


# ── 关节安全范围: 从实测包络文件取(与 IK 用的是同一份, 不另写死一套) ────────────
def joint_limits_16() -> tuple[np.ndarray, np.ndarray]:
    """(16,) 下限/上限。7 关节取实测包络的 observed_min/max, 夹爪取训练集实测范围。"""
    lo, hi = np.zeros(16), np.zeros(16)
    import json
    env = json.loads(K.ENVELOPE_PATH.read_text())["arms"]
    for side, sl in (("left", slice(0, 7)), ("right", slice(8, 15))):
        lo[sl] = env[side]["observed_min"]
        hi[sl] = env[side]["observed_max"]
    # 夹爪: 本任务两爪整程几乎不动(左 0~0.00057 / 右 0.00021~0.02038), 给到 0.075 上限即可
    lo[7], hi[7] = 0.0, 0.075
    lo[15], hi[15] = 0.0, 0.075
    return lo, hi


Q_MIN, Q_MAX = joint_limits_16()


# clamp_target 的两处裁剪原本完全静默 —— 手臂"走不到模型要的位置"时看不出是被
# 关节限位夹住了还是被速率限幅拖住了。这里把它们记下来, 由调用方汇总打印。
CLAMP_STAT = {"lim": 0, "rate": 0, "stall": 0,
              "lim_joints": set(), "stall_joints": set(), "ticks": 0}


def reset_clamp_stat() -> None:
    CLAMP_STAT.update(lim=0, rate=0, stall=0, ticks=0)
    CLAMP_STAT["lim_joints"] = set()
    CLAMP_STAT["stall_joints"] = set()


def clamp_stat_note() -> str:
    """一行摘要; 没触发任何裁剪就返回空串。"""
    n = CLAMP_STAT["ticks"]
    if not n or not (CLAMP_STAT["lim"] or CLAMP_STAT["rate"] or CLAMP_STAT["stall"]):
        return ""
    out = []
    if CLAMP_STAT["stall"]:
        js = ",".join(f"j{j}" for j in sorted(CLAMP_STAT["stall_joints"]))
        out.append(f"顶住{CLAMP_STAT['stall']}/{n}帧[{js}]")
    if CLAMP_STAT["lim"]:
        js = ",".join(f"j{j}" for j in sorted(CLAMP_STAT["lim_joints"]))
        out.append(f"卡限位{CLAMP_STAT['lim']}/{n}帧[{js}]")
    if CLAMP_STAT["rate"]:
        out.append(f"限速{CLAMP_STAT['rate']}/{n}帧")
    return "  ⚠ " + " ".join(out)


def clamp_target(target: np.ndarray, last_cmd: np.ndarray, rate: float, margin: float,
                 q_actual: np.ndarray | None = None, stall_gap: float = 0.0) -> np.ndarray:
    """关节空间的安全限幅(IK 之后做)。

    三道闸, 依次:
      ① 实测包络夹持(±margin)
      ② **抗积分饱和**: 指令不得超前实际位置 stall_gap 以上  ← 防过流的关键
      ③ 逐 tick 速率限幅(rate)

    ② 是 2026-08-26 加的。病因: 位置伺服里 力矩 ∝ (指令 − 实际)。手臂被纸箱挡住
    时实际位置不再前进, 而指令还在按 chunk 一路往前推, 位置误差越积越大 → 电流
    飙到 9A → 驱动器过流保护跳闸失能 → 那条臂彻底不动, 表现为"同一段反复重推、
    位移永不收敛"。实测左臂 J6 就是这么卡死的(52.1° 被挡, 指令仍推向 57~65°)。
    把误差夹在 stall_gap 内, 等于把堵转力矩夹在一个安全值 —— 手臂顶到东西就自然
    停在那里"轻轻顶着", 而不是越顶越狠直到跳闸。
    """
    t = np.clip(target, Q_MIN - margin, Q_MAX + margin)
    t[list(GRIP)] = np.clip(t[list(GRIP)], 0.0, 0.075)
    hit_lim = ~np.isclose(t, target, atol=1e-9)
    # 夹爪(j7/j15)被夹到 0~0.075 是正常工作范围, 不是"手臂走不到" —— 排除掉,
    # 否则每帧都报卡限位, 真正的臂关节受限反而被淹没。
    hit_lim[list(GRIP)] = False

    stalled = np.zeros(16, dtype=bool)
    if q_actual is not None and stall_gap > 0:
        qa = np.asarray(q_actual, dtype=np.float64)
        lo, hi = qa - stall_gap, qa + stall_gap
        t2 = np.clip(t, lo, hi)
        t2[list(GRIP)] = t[list(GRIP)]          # 夹爪本来就靠位置差产生夹持力, 不夹
        stalled = ~np.isclose(t2, t, atol=1e-9)
        stalled[list(GRIP)] = False
        t = t2

    raw_step = t - last_cmd
    step = np.clip(raw_step, -rate, rate)
    CLAMP_STAT["ticks"] += 1
    if hit_lim.any():
        CLAMP_STAT["lim"] += 1
        CLAMP_STAT["lim_joints"].update(np.flatnonzero(hit_lim).tolist())
    if stalled.any():
        CLAMP_STAT["stall"] += 1
        CLAMP_STAT["stall_joints"].update(np.flatnonzero(stalled).tolist())
    if not np.isclose(step, raw_step, atol=1e-9).all():
        CLAMP_STAT["rate"] += 1
    return last_cmd + step


# ── 观测 ────────────────────────────────────────────────────────────────────
def state20_from_joints(q16: np.ndarray) -> np.ndarray:
    """16 维关节 → 20 维绝对末端位姿(世界系, TCP=指尖 0.138)。与训练侧同一份实现。"""
    return E.joints16_to_ee20(np.asarray(q16, dtype=np.float64)[None])[0]


def fake_obs(subtask: str) -> dict:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    q = (Q_MIN + Q_MAX) / 2
    return {
        "observation/image": img,
        "observation/wrist_image": img,
        "observation/right_wrist_image": img,
        "observation/state": state20_from_joints(q),
        "prompt": subtask,
    }


_RS = 224          # 服务端 ResizeImages 的目标边长


def build_obs(state20: np.ndarray, frames: dict, subtask: str,
              preresize: bool = True) -> dict:
    """三路图 + 20 维末端位姿 + prompt。

    ★ preresize(默认开): 发之前先用**服务端同一个函数** image_tools.resize_with_pad
      缩到 224×224。服务端再缩一次是恒等操作 → 喂给模型的图**逐比特相同**,
      但 obs 负载 2.77MB → 0.45MB。

      为什么要紧: 无线网下这一条直接决定"卡顿"。每 execute_horizon 帧要停下来等一次
      推理, 2.77MB 在 WiFi 上实测约 780ms(rtt 均值 44ms/抖动 43ms), 缩完约 165ms。
      单臂客户端(rollout_boxpick_common.build_obs)一直是这么做的, 双臂这两个漏了。

    ⚠ 录进数据集的仍是 480×640 原图 —— record_frame 用的是 frames, 不是这里的缩放图。
    """
    def rs(img):
        if not preresize:
            return img
        from openpi_client import image_tools  # noqa: PLC0415
        return image_tools.resize_with_pad(np.asarray(img), _RS, _RS)

    return {
        "observation/image": rs(frames["observation.images.third"]),
        "observation/wrist_image": rs(frames["observation.images.left_wrist"]),
        "observation/right_wrist_image": rs(frames["observation.images.right_wrist"]),
        "observation/state": state20,
        "prompt": subtask,
    }


# ── 末端位姿 → 关节: IK 执行层 ───────────────────────────────────────────────
class EEChunk:
    """把服务端返回的一块 20 维绝对末端位姿, 逐步解成 16 维关节指令。

    · q_ref(零空间锚点) = 本 chunk 起始的**实测**关节角, 整块不变 —— 不能跟着 seed 走。
    · 静止臂死区: 整块 Δ 低于阈值的臂冻结关节、不调 IK。
    · IK 解不出来(ok=False)时**不下发那条臂**, 保持上一条指令 —— 宁可不动也不送坏解。
    """

    def __init__(self, state20: np.ndarray, chunk20: np.ndarray, q_ref16: np.ndarray,
                 args: argparse.Namespace) -> None:
        self.chunk20 = np.asarray(chunk20, dtype=np.float64)
        self.q_ref = np.asarray(q_ref16, dtype=np.float64).copy()
        self.args = args
        # 从(锚点, 绝对目标)反算相对量, 只为判静止臂 —— 与训练侧 EEDeltaActions 同一口径
        rel = E.ee20_to_rel14(np.asarray(state20, dtype=np.float64), self.chunk20)
        self.idle = {
            "left": K.arm_is_idle(rel, "left", dp_thresh=args.idle_dp, dth_thresh_deg=args.idle_dth),
            "right": K.arm_is_idle(rel, "right", dp_thresh=args.idle_dp, dth_thresh_deg=args.idle_dth),
        }
        self.dp = {a: float(np.linalg.norm(rel[..., o:o + 3], axis=-1).max())
                   for a, o in (("left", 0), ("right", 7))}
        self.n_fail = 0
        self.max_pos_err = 0.0
        self.n_at_limit = 0            # 本块里有多少步出现关节顶限位
        self.at_limit_joints = set()   # {(arm, joint_idx)}

    def solve(self, i: int, last_cmd: np.ndarray) -> tuple[np.ndarray, str]:
        """第 i 步 → (16 维关节指令, 一行状态摘要)。

        ⚠ 关于「IK 未收敛」的处理(2026-08-25 改):
        旧行为是把该臂指令冻结回 last_cmd。问题在于**这会自锁**: 冻结后下一步又拿
        同一个 last_cmd 当种子去解, 解不出继续冻, 种子永远不变 → 臂再也动不了,
        整条 rollout 卡死。实测左臂就是这么锁住的。
        现在默认放行 IK 的**尽力解**(求解器最后一次迭代值), 让种子能往前走、自己
        爬出奇异区。安全性由下游 clamp_target() 兜底: 关节限位裁剪 + 每 tick 步长
        限幅(--rate-limit), 不会跳变。要回到旧行为加 --freeze-on-ik-fail。
        """
        q16, info = K.ee20_to_joints16(
            self.chunk20[i], last_cmd, ref_q16=self.q_ref,
            freeze_left=self.idle["left"], freeze_right=self.idle["right"],
            ns_gain=self.args.ik_ns_gain, damping=self.args.ik_damping,
            max_iter=self.args.ik_max_iter,
            tol_pos=self.args.ik_tol_pos, tol_rot=self.args.ik_tol_rot)
        freeze = getattr(self.args, "freeze_on_ik_fail", False)
        notes = []
        for arm, sl in (("left", E.J_LEFT), ("right", E.J_RIGHT)):
            inf = info[arm]
            if inf is None:
                continue
            # ★ 关节顶到限位(2026-08-26 加)。实测: 该位形下零空间只有 1 维、方向是
            #   [J1,J3], **J6 分量恰为 0** —— 冗余自由度动不了 J6, 它的角度由末端位姿
            #   唯一确定。所以 IK 层面无解, 只能在这里拦: 一旦某关节被 clip 在限位上,
            #   就把该关节的指令冻回上一条(不再往限位方向推), 其余关节照常跟随。
            #   不拦的话真机会顶上机械硬限位 → 堵转 → 过流 → 失能(左臂 J6 实测 9.0A)。
            if inf.get("at_limit"):
                lc = np.asarray(last_cmd, dtype=np.float64)
                idx = np.arange(sl.start, sl.stop) if isinstance(sl, slice) else np.asarray(sl)
                for j in inf["at_limit"]:
                    q16[idx[j]] = lc[idx[j]]
                self.n_at_limit += 1
                self.at_limit_joints.update((arm, int(j)) for j in inf["at_limit"])
            self.max_pos_err = max(self.max_pos_err, inf["pos_err"])
            if not inf["ok"]:
                self.n_fail += 1
                tag = f"{arm}IK未收敛({inf['pos_err']*1000:.1f}mm" \
                      f"{',卡限位' if inf['at_limit'] else ''})"
                if freeze:
                    q16[sl] = np.asarray(last_cmd, dtype=np.float64)[sl]
                    notes.append(tag + "→保持(--freeze-on-ik-fail)")
                else:
                    bad = ~np.isfinite(q16[sl])
                    if bad.any():   # 只有真出 NaN/Inf 才退回上一指令
                        q16[sl] = np.where(bad, np.asarray(last_cmd, dtype=np.float64)[sl], q16[sl])
                        notes.append(tag + "→含NaN, 该维回退")
                    else:
                        notes.append(tag + "→放行尽力解")
        return q16, ("  ".join(notes) if notes else "")

    def summary(self) -> str:
        f = lambda a: ("静止" if self.idle[a] else f"{self.dp[a]*100:.1f}cm")
        s = (f"左{f('left')}/右{f('right')}  IK残差{self.max_pos_err*1000:.2f}mm"
             + (f"  未收敛{self.n_fail}次" if self.n_fail else ""))
        if self.n_at_limit:
            js = ",".join(f"{a[0].upper()}J{j+1}" for a, j in sorted(self.at_limit_joints))
            s += f"  🛑顶限位{self.n_at_limit}步[{js}]"
        return s


def safe_enable(arms, keypad: KeyPad, args: argparse.Namespace) -> np.ndarray | None:
    """双臂使能防冲(J2 事故防线)。返回基准 16 维关节位姿, 失败 None。"""
    q0 = wait_state_fresh(arms)
    if q0 is None:
        print("🛑 CAN 关节反馈始终不完整(有维恒为精确 0.0) — 不使能。", flush=True)
        return None
    print("使能前位姿:", np.round(q0, 3).tolist(), flush=True)
    print("当前模式:", arms.modes(), flush=True)
    print(f"\n⚠ 即将使能双臂(speed={args.speed_percent}%)。确认: 工作区无人手、急停在手边。", flush=True)
    # 串跑时 run_chain_4stage 已经统一问过一次 GO, 这里不该每段再让人手打 DEPLOY。
    # NERO_SKIP_DEPLOY_PROMPT=1 由 chain 下发; 单独手跑时不设, 行为与以前完全一致。
    if os.environ.get("NERO_SKIP_DEPLOY_PROMPT") == "1":
        print("输入 DEPLOY 继续(其它=退出): [已由上层确认, 自动继续]", flush=True)
    elif input("输入 DEPLOY 继续(其它=退出): ").strip() != "DEPLOY":
        return None
    for _ in range(3):
        arms.send_joints(q0)
        time.sleep(0.05)
    arms.enable(args.speed_percent)
    drift = 0.0
    for _ in range(int(1.5 * FPS)):
        arms.send_joints(q0)
        time.sleep(PERIOD)
        q = np.asarray(arms.read_state())
        drift = float(np.abs(q[list(JOINTS)] - q0[list(JOINTS)]).max())
        if drift > args.enable_drift_abort:
            print(f"\n🛑 使能后关节漂移 {drift:.3f}rad > {args.enable_drift_abort} — 立即下电!", flush=True)
            arms.estop()
            return None
    print(f"✓ 使能通过防冲检测(漂移 {drift:.4f} rad)", flush=True)
    return np.asarray(arms.read_state())


def wait_state_fresh(arms, timeout: float = 8.0) -> np.ndarray | None:
    """等 CAN 关节反馈全部上来。刚 connect 时部分关节会读成**精确 0.0**, 拿它当保持目标
    会让使能后从真实位姿冲向 0(实测最大 1.86rad)。判据: 14 个关节没有一维恒为 0.0。"""
    t_end = time.monotonic() + timeout
    prev = None
    while time.monotonic() < t_end:
        try:
            s = np.asarray(arms.read_state(), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            print(f"  等反馈: 读取失败 {exc}", flush=True)
            time.sleep(0.1)
            continue
        zeros = [i for i in JOINTS if s[i] == 0.0]
        if not zeros:
            if prev is not None:
                return s
            prev = s
        else:
            print(f"  等 CAN 反馈… 维 {zeros} 仍为精确 0.0", flush=True)
        time.sleep(0.1)
    return None


def check_state_in_range(q16: np.ndarray, margin: float = 0.35) -> None:
    out = [i for i in range(16) if not (Q_MIN[i] - margin <= q16[i] <= Q_MAX[i] + margin)]
    if out:
        print(f"  ⚠ 起始位姿有 {len(out)} 维在训练包络外: "
              + "; ".join(f"dim{i}({q16[i]:+.3f}∉[{Q_MIN[i]:.3f},{Q_MAX[i]:.3f}]±{margin})"
                          for i in out), flush=True)
        print("    模型没在这些区域训过, 首步动作可能不合理 —— 建议先摆回常见起始位姿。", flush=True)
    else:
        print("  ✓ 起始位姿 16 维全部落在训练包络内", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Nero 双臂 pi0.5 部署 — hezi 封盖(末端位姿版 + IK)")
    ap.add_argument("--host", default="<POLICY_SERVER_IP>", help="<SERVER>(有线 20.198 / 无线 10.198)")
    ap.add_argument("--port", type=int, default=8026, help="末端位姿版 serve 端口")
    ap.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True,
                    help="真执行(默认开; --no-execute 回到 dry-run 不发运动命令)")
    ap.add_argument("--confirm-enable-risk", action=argparse.BooleanOptionalAction, default=True,
                    help="--execute 必须(默认开)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--fake-obs", action="store_true")
    ap.add_argument("--mock-robot", action="store_true")
    ap.add_argument("--execute-horizon", type=int, default=30,
                    help="每次推理后执行多少帧再重推理(动作块共 50 帧)")
    ap.add_argument("--rate-limit", type=float, default=0.12,
                    help="关节指令每 tick 最多变多少(rad)。0.12≈103°/s; 封盖示教逐帧 p99 见须知")
    ap.add_argument("--limit-margin", type=float, default=0.15, help="实测关节包络外的裕量")
    ap.add_argument("--enable-drift-abort", type=float, default=0.8,
                    help="使能防冲阈值(rad, 默认0.8≈46°)。放宽的代价: 小于该值的跳变不再拦, "
                         "务必人手放急停键")
    ap.add_argument("--speed-percent", type=int, default=20)
    ap.add_argument("--grip-force", type=float, default=1.0)
    ap.add_argument("--freeze-on-ik-fail", action="store_true",
                    help="IK未收敛时把该臂冻结回上一指令(旧行为)。⚠ 会自锁: 种子不再变化, "
                         "臂可能永久卡住。默认关闭, 改为放行尽力解+下游限幅兜底")
    ap.add_argument("--start-stage", type=int, default=1, help="从第几段开始(1..7)")
    # IK 参数(默认值即 ik_nero selftest 22/22 通过的那组, 一般不用动)
    ap.add_argument("--ik-ns-gain", type=float, default=0.30, help="零空间拉回 q_ref 的强度")
    ap.add_argument("--ik-damping", type=float, default=1e-2)
    ap.add_argument("--ik-max-iter", type=int, default=100)
    # 真机口径容差: 模型输出的位姿不保证严格落在可达流形上, 残差 1~2mm 属正常。
    # ik_nero 的默认 0.1mm 是给离线回放用的(目标由 FK 生成必然可达), 真机太严会把
    # 每一步都判失败 → 关节全程"保持"、模型等于没在跑。抓取任务 5mm 足够。
    ap.add_argument("--ik-tol-pos", type=float, default=5e-3, help="IK 位置容差(m), 默认 5mm")
    ap.add_argument("--ik-tol-rot", type=float, default=0.5 * 3.141592653589793 / 180,
                    help="IK 旋转容差(rad), 默认 0.5°")
    ap.add_argument("--idle-dp", type=float, default=3e-3, help="静止臂死区: 位移阈值(m)")
    ap.add_argument("--idle-dth", type=float, default=0.3, help="静止臂死区: 转角阈值(度)")
    ap.add_argument("--no-idle-freeze", action="store_true", help="关掉静止臂死区(排障用)")
    # CAN: auto = 按适配器序列号解析
    ap.add_argument("--left-can", default="auto")
    ap.add_argument("--right-can", default="auto")
    ap.add_argument("--left-firmware", default="v120")
    ap.add_argument("--right-firmware", default="v120")
    ap.add_argument("--can-interface", default="socketcan")
    ap.add_argument("--can-bitrate", type=int, default=1_000_000)
    ap.add_argument("--can-timeout", type=float, default=1.0)
    add_camera_args(ap)
    args = ap.parse_args()

    args.camera_width, args.camera_height = 640, 480
    autodetect_devices(args)
    if args.no_idle_freeze:
        args.idle_dp, args.idle_dth = -1.0, -1.0   # 永远判不成静止 → 两臂都走 IK

    if args.execute and not args.confirm_enable_risk:
        ap.error("--execute 需要 --confirm-enable-risk")
    if args.execute and args.fake_obs:
        ap.error("--fake-obs 不能与 --execute 组合")
    return args


def main() -> None:  # noqa: PLR0915, PLR0912
    args = parse_args()
    subtasks = HZ.load_subtasks(HZ.STAGE_ORDER_TRAIN)
    print(f"\n7 段封盖 stage prompt(训练权威顺序 {HZ.STAGE_ORDER_TRAIN}), 从第 {args.start_stage} 段开始",
          flush=True)
    print(f"服务端: ws://{args.host}:{args.port}   末端位姿版(20 维绝对位姿 → IK → 关节)", flush=True)

    from openpi_client import websocket_client_policy

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    if args.fake_obs:
        t0 = time.monotonic()
        ch = np.asarray(policy.infer(fake_obs(subtasks[0]))["actions"])
        dt = (time.monotonic() - t0) * 1000
        print(f"✓ 服务链路通: actions{ch.shape} 时延{dt:.0f}ms", flush=True)
        assert ch.shape[1] == EE_DIM, f"期望 {EE_DIM} 维绝对末端位姿, 实得 {ch.shape[1]}"
        q = (Q_MIN + Q_MAX) / 2
        s20 = state20_from_joints(q)
        ec = EEChunk(s20, ch, q, args)
        q_cmd, note = ec.solve(0, q)
        print(f"  首步位姿 位置左{np.round(ch[0][0:3],3).tolist()} 右{np.round(ch[0][10:13],3).tolist()}",
              flush=True)
        print(f"  IK 还原: {ec.summary()}  {note}", flush=True)
        print(f"  → 16 维关节指令 {np.round(q_cmd, 4).tolist()}", flush=True)
        return

    arms = MockArms() if args.mock_robot else DeployArms(args)
    cameras = CameraSet(build_camera_configs(args))
    mode = "EXECUTE" if args.execute else "DRY-RUN(不发命令)"

    try:
        cameras.open()
        print("等 CAN 关节反馈就绪(冷启动时部分维会先读成 0) …", flush=True)
        q16 = wait_state_fresh(arms)
        if q16 is None:
            print("🛑 CAN 关节反馈始终不完整 — 退出。", flush=True)
            return
        print(f"\n模式={mode}  当前 16 维关节: {np.round(q16, 3).tolist()}", flush=True)
        check_state_in_range(q16)
        s20 = state20_from_joints(q16)
        print(f"  → 20 维末端位姿 左{np.round(s20[0:3],3).tolist()} 右{np.round(s20[10:13],3).tolist()}"
              f"  爪 {s20[9]:.4f}/{s20[19]:.4f}", flush=True)

        with KeyPad() as keypad:
            last_cmd = q16.copy()
            if args.execute or args.prepare_only:
                base = safe_enable(arms, keypad, args)
                if base is None:
                    print("使能未通过/已取消, 退出。", flush=True)
                    return
                last_cmd = np.asarray(base)
                if args.prepare_only:
                    print("--prepare-only 完成, 保持使能退出。", flush=True)
                    return

            sid = max(0, min(len(subtasks) - 1, args.start_stage - 1))
            print(f"\n▶ 进入策略循环 @{FPS}Hz", flush=True)
            print("  键: n/空格=下一段  1-7=跳段  b=退一段  p=暂停/继续  回车/q=急停\a", flush=True)
            keypad.drain()
            paused = False
            while True:
                prompt = subtasks[sid]
                print(f"\n● 第 {sid+1}/{len(subtasks)} 段: {prompt[:58]}", flush=True)
                q16 = np.asarray(arms.read_state())
                frames = cameras.read_frames()
                s20 = state20_from_joints(q16)
                t0 = time.monotonic()
                chunk20 = np.asarray(policy.infer(build_obs(s20, frames, prompt))["actions"])
                infer_ms = (time.monotonic() - t0) * 1000
                if chunk20.shape[1] != EE_DIM:
                    print(f"🛑 服务端返回 {chunk20.shape[1]} 维, 期望 {EE_DIM} —— 端口/模型接错了?",
                          flush=True)
                    if args.execute:
                        arms.estop()
                    return

                ec = EEChunk(s20, chunk20, q16, args)          # q_ref = 本块起始实测关节角
                n_exec = min(args.execute_horizon, len(chunk20))
                t_ik = time.monotonic()
                print(f"  推理{infer_ms:4.0f}ms  {ec.summary()}  执行{n_exec}帧"
                      f"{'(dry)' if not args.execute else ''}", flush=True)

                stop = False
                for i in range(n_exec):
                    tick = time.monotonic()
                    q_ik, note = ec.solve(i, last_cmd)
                    target = clamp_target(q_ik, last_cmd, args.rate_limit, args.limit_margin)
                    if args.execute and not paused:
                        arms.send_joints(target)
                        arms.send_grippers(target, args.grip_force)
                        last_cmd = target
                    elif not args.execute:
                        last_cmd = target
                    if note:
                        print(f"    步{i}: {note}", flush=True)

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
                        elif k in ("n", " "):
                            sid = min(sid + 1, len(subtasks) - 1); stop = True
                        elif k == "b":
                            sid = max(sid - 1, 0); stop = True
                        elif k.isdigit() and 1 <= int(k) <= len(subtasks):
                            sid = int(k) - 1; stop = True
                        if stop:
                            break
                    dt = time.monotonic() - tick
                    if dt < PERIOD:
                        time.sleep(PERIOD - dt)
                if not stop:
                    ik_ms = (time.monotonic() - t_ik) * 1000 / max(n_exec, 1)
                    print(f"    本块完成, 单步 IK 均耗时 {ik_ms:.2f}ms", flush=True)
    except KeyboardInterrupt:
        print("\nCtrl-C: ", "急停下电" if args.execute else "退出", flush=True)
        if args.execute:
            arms.estop()
    finally:
        cameras.close()
        arms.close()


if __name__ == "__main__":
    main()
