#!/usr/bin/env python3
"""Nero 四阶段长程串跑编排 —— 右臂抓盒 → 左臂抓盒 → stage34 封盖 → stage56 合页。

它**不重新实现** rollout,而是按顺序调起四个已经验证过的单阶段客户端,并统一:

  1. **运行 ID** —— 通过环境变量 ``WP_RUN`` 下发,四段数据落到同一个
     ``lerobot_data/Rlinf/data/wholeprocess/<run_id>/{e0_rightbox,e1_leftbox,e2_hezi34,e3_stage56}/``
  2. **阶段编排** —— 按 switcher.py 的设计,专家之间用**确定性计数器**调度
     (不需要分类器);switcher 只管专家**内部**的子任务推进。
  3. **阶段间闸门** —— 默认每段跑完停下来等人确认(见 ``--auto-advance``)。
     四个模型动作语义不同,直接接续会跳变,首次串跑务必人工确认。

为什么是"调起子进程"而不是"合并成一个大循环"
----------------------------------------------
四个客户端的机械臂控制、IK、相机、急停逻辑各不相同(单臂 10 维 vs 双臂 20 维),
合并会把四份已验证的代码搅成一份没验证过的。子进程方案的代价是每段要重连一次
机械臂和相机(约 10~15 秒),换来的是**任何一段崩了都不影响其它段**,而且
单段仍可独立跑 —— 这是首次打通时更重要的性质。

关于 switcher(自动子任务切换)
--------------------------------
本脚本默认**不启用** switcher,四段都沿用客户端自带的"人按空格推进"。
原因见 --with-switcher 的说明:现有客户端把子任务推进权写死在键盘分支里,
接 switcher 需要改动四个客户端的主循环,那是下一步的事,不该和"先把四段
串起来跑通"混在一起做。

用法::

    # 干跑:不连机械臂/相机,只验编排与路径
    python3 run_chain_4stage.py --dry-run

    # 真机只读(推理+录制,不发运动命令)
    python3 run_chain_4stage.py --host <POLICY_SERVER_IP>

    # 真执行
    python3 run_chain_4stage.py --host <POLICY_SERVER_IP> --execute

    # 只跑其中几段(调试用)
    python3 run_chain_4stage.py --stages 0,1 --execute

    # 接着上次的运行 ID 补跑后面几段
    WP_RUN=run_20260826_150000 python3 run_chain_4stage.py --stages 2,3 --execute
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NERO_ROOT = HERE.parent
DATA_ROOT = NERO_ROOT / "lerobot_data" / "Rlinf" / "data" / "wholeprocess"

# ── 四个阶段 ────────────────────────────────────────────────────────────────
# expert  : switcher.py 的专家编号(0=右臂 1=左臂 2=封箱中段 3=封箱末段),与本表顺序一致
# port    : 服务端口。⚠ 以**服务器上实际挂的**为准,不是各 serve_*.sh 的默认值
# extra   : 该客户端特有的必需参数
STAGES = [
    dict(idx=0, expert=0, tag="e0_rightbox", fail_hint="e0_grasp_failed", name="右臂抓盒",
         script="run_pi05_rollout_rightbox_ee.py", port=8031, n_sub=2,
         model="pi05_nero_right_box_pick_ee_v1b / rbp_ee_v1b_run1/19999", extra=[],
         home_after=False),   # E0→E1 连续装箱, 中间不归位
    dict(idx=1, expert=1, tag="e1_leftbox", fail_hint="e1_grasp_failed", name="左臂抓盒",
         script="run_pi05_rollout_leftbox_ee.py", port=8029, n_sub=2,
         model="pi05_nero_left_box_pick_ee_v1 / lbp_ee_run1/19999", extra=[],
         home_after=True),    # ★ 两个盒子都装完 → 归位, 再进封盖阶段。
                              #   E2/E3 都是从初始位形训练的, 不归位就等于让它们
                              #   从一个训练集里没出现过的位形起步。
    dict(idx=2, expert=2, tag="e2_hezi34", fail_hint="flap_not_closed", name="stage34 封盖",
         script="run_pi05_rollout_hezi_ee.py", port=8026, n_sub=7,
         model="pi05_nero_hezi_closing_ee_v1 / hezi_ee_run1/19999",
         extra=["--confirm-enable-risk"], home_after=True),   # ★ stage34 结束时两臂还抵在扇面上,
                                                              #   而 stage56 是从初始位训练的 → 必须归位
    dict(idx=3, expert=3, tag="e3_stage56", fail_hint="flap_not_closed", name="stage56 合页",
         script="run_pi05_rollout_stage56_flap_ee.py", port=8032, n_sub=7,
         model="pi05_nero_stage56_flap_closing_ee_v100 / run1/19999",
         extra=["--confirm-enable-risk"], home_after=False),  # 整轮结束不自动归位
]

EXIT_USER_FAIL = 20     # 与各 rollout 客户端约定: 你按 f 判失败

C_OK, C_WARN, C_ERR, C_DIM, C_HL, C_END = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1;36m", "\033[0m")


def rule(ch="─", n=76):
    print(C_DIM + ch * n + C_END, flush=True)


def preflight(args, stages) -> bool:
    """开跑前的静态检查。任一条不过就别开始 —— 跑到一半才发现端口不通最浪费时间。"""
    ok = True
    print(f"\n{C_HL}══ 预检 ══{C_END}", flush=True)

    for st in stages:
        p = HERE / st["script"]
        mark = f"{C_OK}✓{C_END}" if p.exists() else f"{C_ERR}✘{C_END}"
        if not p.exists():
            ok = False
        print(f"  {mark} 客户端  E{st['idx']} {st['script']}", flush=True)

    if not args.dry_run:
        import socket
        import time as _t
        # WiFi 实测 rtt mdev 可达 33ms、瞬时 100ms+, 单次 3s 超时会被抖动误判。
        # 重试 3 次是为了区分「真没起服」和「网络抖了一下」—— 前者要停,后者不该拦住整条链。
        for st in stages:
            last = None
            for attempt in range(3):
                try:
                    t0 = _t.time()
                    with socket.create_connection((args.host, st["port"]), timeout=6):
                        ms = 1000 * (_t.time() - t0)
                        note = f"  {C_DIM}(第{attempt+1}次, {ms:.0f}ms){C_END}" if attempt else ""
                        print(f"  {C_OK}✓{C_END} 服务端  E{st['idx']} {args.host}:{st['port']}  "
                              f"{C_DIM}{st['model']}{C_END}{note}", flush=True)
                        last = None
                        break
                except OSError as e:
                    last = e
                    if attempt < 2:
                        _t.sleep(1.0)
            if last is not None:
                ok = False
                print(f"  {C_ERR}✘{C_END} 服务端  E{st['idx']} {args.host}:{st['port']} "
                      f"连不上 ({last}) —— 已重试 3 次\n"
                      f"      起服: GPU=<卡> PORT={st['port']} STEP=19999 "
                      f"bash deploy/ee_server/serve_*.sh", flush=True)
    else:
        print(f"  {C_DIM}(--dry-run: 跳过服务端连通性检查){C_END}", flush=True)

    return ok


def resolve_devices() -> list:
    """把 CAN 与相机按序列号解析一次, 变成可以直接下发给各段的参数。

    ⚠ 为什么值得做: 每段客户端都会自己扫一遍序列号(udev 查询 + 逐个开相机试),
    实测每段 10~15 秒。四段就是一分钟, 全花在重复做同一件事上。
    客户端本来就是「显式给了就不扫」, 所以 chain 解析一次再传进去即可, 不用改客户端。

    ⚠ CAN 必须按**适配器序列号**认, 不能写死 can0/can1 —— 接口号是内核按枚举顺序给的,
    重插 USB 就会对调。实测 2026-08-29 重新接线后 can0/can1 正好换了个位置。
    """
    import sys as _sys  # noqa: PLC0415
    _sys.path.insert(0, str(HERE))
    try:
        import rollout_boxpick_common as C   # noqa: PLC0415
        import run_pi05_deploy_hezi_ee as D  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        print(f"  {C_WARN}⚠ 设备预解析失败({e}) —— 各段自己扫{C_END}", flush=True)
        return []
    out, miss = [], []
    for side, ser in C.CAN_SERIALS.items():
        got = D.resolve_can_by_serial(ser)
        (out.extend([f"--{side.replace('_','-')}-can", got]) if got else miss.append(f"{side} CAN"))
    for cam, ser in C.CAM_SERIALS.items():
        got = D.resolve_video_by_serial(ser)
        flag = {"third_view": "--third-camera", "left_wrist": "--left-wrist-camera",
                "right_wrist": "--right-wrist-camera"}[cam]
        (out.extend([flag, got]) if got else miss.append(f"{cam} 相机"))
    if miss:
        print(f"  {C_WARN}⚠ 没解析到: {', '.join(miss)} —— 那几项交给各段自己扫{C_END}",
              flush=True)
    return out


def run_stage(st, args, run_id) -> int:
    """跑一个阶段。返回子进程退出码。"""
    cmd = [sys.executable, str(HERE / st["script"])]
    if args.dry_run:
        cmd += ["--mock-robot", "--mock-cameras"]
    else:
        cmd += ["--host", args.host, "--port", str(st["port"])]
        if args.execute:
            cmd.append("--execute")
        cmd += st["extra"]
    # ★ 串跑语义: 空格走完最后一个子任务 = 本大阶段结束 → 客户端退出 → 本脚本推进到下一段。
    #   这样从头到尾**只有空格一个键**: 小阶段推进用它, 大阶段推进也是它。
    if not args.multi_episode:
        cmd.append("--once")
    # ★ 视频不在段内编码 —— 每条之后那次 AV1 编码(几秒到几十秒)会打断连贯性,
    #   挪到四段全跑完后由 --finalize 统一做。
    if not args.encode_inline:
        cmd.append("--defer-encode")
    # ★ 每段跑完不再问 s/f/g —— 自动存, 成功/失败留到整条链结束统一标一次。
    #   否则"空格一路到底"会在每个大阶段末尾被 s/f/g 截停。
    if not args.label_per_stage:
        cmd.append("--defer-label")
    # ★ 视频不在每段结束时编 —— 整条链跑完(或你按 f 中止)才统一编一次。
    if not args.encode_inline:
        cmd.append("--no-encode-on-exit")
    # 归位由**客户端自己**在本段结束时做(goto_home_inline), chain 不再重复调一次 ——
    # 否则串跑时会连着归位两遍(第二遍虽然已在位、只花 0.7s, 但白等且容易看懵)。
    # 写在客户端的好处是: 单独跑某一阶段也会归位, 不依赖 chain。
    # 归位只在 STAGES 里标了 home_after=True 的那一段之后做。
    # 当前只有 E1(左臂抓盒)之后归 —— E0→E1 是连续装两个盒子, 中间归位反而打断;
    # 两个盒子都装完再回初始位, 让 E2/E3 封盖从与训练一致的位形起步。
    if args.no_home or not st.get("home_after", False):
        cmd.append("--no-home")
    else:
        cmd += ["--home-rate", str(args.home_rate)]
    # 自动切换: expert 编号取自 STAGES 表, 与切换包的 0=右臂 1=左臂 2=封箱中段
    # 3=封箱末段 完全一致, 不需要额外映射。
    if args.auto_switch:
        cmd += ["--auto-switch", "--switch-expert", str(st["expert"]),
                "--switch-device", args.switch_device]
    # 串跑时不再每段等人按空格 —— 阶段闸门已由 --auto-advance 跳过, 这是最后一处按键。
    if not args.manual_start:
        cmd.append("--auto-start")
    if abs(args.speed - 1.0) > 1e-9:
        cmd += ["--speed", str(args.speed)]
    # 设备只解析一次(见 resolve_devices), 每段省 10~15 秒重复扫描。
    # ⚠ 单臂客户端没有 --left-can / --right-wrist-camera 这类参数, 逐个过滤掉,
    #   否则 argparse 会以 "unrecognized arguments" 直接退出。
    if not args.rescan_devices and not args.dry_run:
        import subprocess as _sp2  # noqa: PLC0415
        try:
            helptxt = _sp2.run([sys.executable, str(HERE / st["script"]), "--help"],
                               capture_output=True, text=True, timeout=90).stdout
        except Exception:  # noqa: BLE001
            helptxt = ""
        dev = args._devices
        for i in range(0, len(dev), 2):
            if dev[i] in helptxt:
                cmd += [dev[i], dev[i + 1]]
    cmd += args.passthrough

    env = dict(os.environ, WP_RUN=run_id)          # ★ 四段共用同一个运行 ID
    # E2/E3 的 safe_enable 原本每段都要手打 DEPLOY。串跑开头已经统一确认过一次 GO,
    # 再让人逐段打字既慢又打断"空格一路到底"。单独手跑不设这个变量, 老行为不变。
    env["NERO_SKIP_DEPLOY_PROMPT"] = "1"
    out = DATA_ROOT / run_id / st["tag"]

    rule("═")
    print(f"{C_HL}▶ E{st['idx']} · {st['name']}{C_END}   "
          f"({st['n_sub']} 个子任务)", flush=True)
    print(f"  模型   {C_DIM}{st['model']}{C_END}", flush=True)
    print(f"  端口   {args.host}:{st['port']}" if not args.dry_run else "  端口   (dry-run)", flush=True)
    print(f"  输出   {C_DIM}{out}{C_END}", flush=True)
    print(f"  命令   {C_DIM}{' '.join(cmd[1:])}{C_END}", flush=True)
    rule("═")
    print(f"\n{C_WARN}键位: 空格/s=本段成功切下一段  f=本段失败(整条结束)  "
          f"g=丢弃  p=暂停  回车/q=急停{C_END}\n", flush=True)

    try:
        return subprocess.call(cmd, env=env, cwd=str(HERE))
    except KeyboardInterrupt:
        print(f"\n{C_WARN}⚠ 该阶段被 Ctrl-C 中断{C_END}", flush=True)
        return 130


def go_home(args, why: str) -> int:
    """大阶段之间把两臂送回同一摆位(goto_home.py)。

    这是串跑效果远差于单跑的主因: 四个大阶段用四个不同模型, E0 结束时手臂停在
    "刚把盒子放进打包箱"的位形, 而 E1 的模型是从**你摆好的初始位**开始训练的 ——
    直接接续等于让 E1 从一个训练时没见过的位形起步。插一段归位轨迹后, 每一段拿到
    的初始条件就和单独跑时一致了。
    """
    if args.no_home:
        return 0
    cmd = [sys.executable, str(HERE / "goto_home.py")]
    if args.execute and not args.dry_run:
        cmd.append("--execute")
    cmd += ["--rate", str(args.home_rate)]
    print(f"\n{C_HL}══ 归位({why})══{C_END}", flush=True)
    if args.dry_run:
        print(f"  {C_DIM}(--dry-run: 跳过){C_END}", flush=True)
        return 0
    try:
        rc = subprocess.call(cmd, cwd=str(HERE))
    except KeyboardInterrupt:
        return 130
    if rc != 0:
        print(f"  {C_WARN}⚠ 归位未完全到位(退出码 {rc}) —— 现场确认后再继续{C_END}", flush=True)
    return rc


def gate(st_next, args) -> bool:
    """阶段间闸门。返回 False 表示中止整条链。"""
    if args.auto_advance:
        print(f"\n{C_WARN}--auto-advance: 不停,直接进 E{st_next['idx']}{C_END}", flush=True)
        return True
    print(f"\n{C_HL}══ 阶段间确认 ══{C_END}", flush=True)
    print(f"  下一段: E{st_next['idx']} · {st_next['name']}", flush=True)
    print(f"  {C_WARN}确认机械臂已在安全位、现场无人、可以继续。{C_END}", flush=True)
    try:
        ans = input("  回车=继续   s=跳过这一段   q=结束整条链: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if ans == "q":
        return False
    if ans == "s":
        st_next["_skip"] = True
    return True


WHOLE_FAILURES = [
    ("none", "成功"),
    ("e0_grasp_failed", "E0 右臂没抓到 / 抓偏"),
    ("e1_grasp_failed", "E1 左臂没抓到 / 抓偏"),
    ("place_failed", "没放进打包箱 / 掉在外面"),
    ("flap_not_closed", "扇面没合上 / 合不到位"),
    ("arm_collision", "机械臂碰撞"),
    ("joint_at_limit", "关节顶限位 / 电机跳闸"),
    ("network_stall", "网络卡顿导致中断"),
    ("aborted", "人为中止"),
    ("other", "其他"),
]


def label_run(run_id: str, stages, aborted: bool, failed_st=None) -> None:
    """整条链结束时统一标注一次, 回填进四段的 parquet 与 annotations。

    录制期各段用 --defer-label 自动存, episode_success 落的是占位 -1、
    failure_type 是 "pending" —— 这里一次性改写。好处是全程只按空格,
    不会每个大阶段末尾被 s/f/g 截停; 而标注本身是整条长程任务级别的,
    本来也该整体判一次而不是逐段判。
    """
    rule("═")
    print(f"{C_HL}══ 整条链标注 ══{C_END}   运行 ID {run_id}", flush=True)
    if aborted:
        print(f"  {C_WARN}(本次中止){C_END}", flush=True)
    if failed_st is not None:
        print(f"  {C_WARN}(你在 E{failed_st['idx']} {failed_st['name']} 按了 f 判失败){C_END}",
              flush=True)
    for i, (_c, zh) in enumerate(WHOLE_FAILURES, 1):
        print(f"    {i}. {zh}")
    try:
        # 已经按过 f 了就别再默认"成功" —— 回车落到**那一段最常见的失败原因**上,
        # 省得为了标一次去数 10 行。当然想选别的照样输编号。
        dflt = 1
        if failed_st is not None:
            hint = failed_st.get("fail_hint")
            dflt = next((i for i, (c, _) in enumerate(WHOLE_FAILURES, 1) if c == hint), 2)
        v = input(f"  整条任务的结果(1~{len(WHOLE_FAILURES)}, 回车={dflt} "
                  f"{WHOLE_FAILURES[dflt-1][1]}): ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n  {C_WARN}跳过标注 —— 之后可补:"
              f" python3 run_chain_4stage.py --label-only {run_id}{C_END}", flush=True)
        return
    idx = int(v) - 1 if v.isdigit() and 1 <= int(v) <= len(WHOLE_FAILURES) else dflt - 1
    code, zh = WHOLE_FAILURES[idx]
    success = (code == "none")
    print(f"  → {'成功' if success else '失败/' + code}  ({zh})", flush=True)

    import pandas as pd  # noqa: PLC0415
    for st in stages:
        root = DATA_ROOT / run_id / st["tag"]
        pqs = sorted(root.glob("data/**/*.parquet"))
        if not pqs:
            print(f"  {C_DIM}— E{st['idx']} {st['name']}: 无数据{C_END}", flush=True)
            continue
        n = 0
        for f in pqs:
            try:
                df = pd.read_parquet(f)
                if "episode_success" in df:
                    df["episode_success"] = np.int64(1 if success else 0)
                if "failure_type" in df:
                    df["failure_type"] = code
                df.to_parquet(f, index=False)
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {C_ERR}✘ {f.name}: {e}{C_END}", flush=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        with open(root / "annotations" / "whole_process_outcome.json", "w",
                  encoding="utf-8") as fh:
            json.dump({"run_id": run_id, "stage": st["tag"],
                       "episode_success": success, "failure_type": code,
                       "failure_zh": zh, "aborted": aborted}, fh, ensure_ascii=False, indent=2)
        print(f"  {C_OK}✓{C_END} E{st['idx']} {st['name']}: 回填 {n} 个 parquet", flush=True)
    rule("═")


def finalize(run_id: str, stages) -> None:
    """把四段推迟下来的视频统一编码。

    录制时各客户端用 batch_encoding_size=天文数字, 所以 save_episode 只落
    parquet + 图像帧, 不碰视频编码。这里逐个数据集调 batch_encode_videos()
    补上 —— 一次把四段的编码集中做完, 而不是散在每条 rollout 之间打断节奏。
    """
    sys.path.insert(0, str(NERO_ROOT / "lerobot_data" / "lerobot_tools" / "lerobot_v21" / "src"))
    from lerobot.datasets.lerobot_dataset import (  # noqa: PLC0415
        CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata)

    rule("═")
    print(f"{C_HL}══ 统一编码视频 ══{C_END}   运行 ID {run_id}", flush=True)
    for st in stages:
        root = DATA_ROOT / run_id / st["tag"]
        if not (root / "meta" / "info.json").exists():
            print(f"  {C_DIM}— E{st['idx']} {st['name']}: 无数据, 跳过{C_END}", flush=True)
            continue
        repo_id = f"local/wholeprocess_{run_id}_{st['tag']}"
        try:
            # ⚠ 不能用 LeRobotDataset(repo_id, root=...) —— 它 __init__ 里会
            #   assert 所有 episode 文件都在本地, 而 get_episodes_file_paths()
            #   **把 mp4 也算进去**。我们正是为了"最后统一编码"才故意没编视频,
            #   断言必然失败 → 它就去 HuggingFace Hub 下载这个根本不存在的仓库,
            #   于是要么卡在网络上, 要么被 httpx 的代理解析直接掀翻:
            #       ValueError: Unknown scheme for proxy URL 'socks://127.0.0.1:7897/'
            #   这就是 20:09 之后每条 run 都只剩散图、一个 mp4 都没有的原因。
            #
            #   编码本身只用到 meta + root + fps, 一点网络都不需要。所以跳过
            #   __init__, 只建本地的 Metadata(meta/ 齐全时它不联网), 再调同一个
            #   batch_encode_videos。行为与客户端内部编码完全一致。
            ds = LeRobotDataset.__new__(LeRobotDataset)
            ds.root = root
            ds.repo_id = repo_id
            ds.revision = CODEBASE_VERSION
            ds.meta = LeRobotDatasetMetadata(repo_id, root=root)
            ds.episodes = None
            ds.image_writer = None
            n = ds.meta.total_episodes
            if n == 0:
                print(f"  {C_DIM}— E{st['idx']} {st['name']}: 0 条, 跳过{C_END}", flush=True)
                continue
            n_png = sum(1 for _ in root.rglob("*.png"))
            print(f"  {C_HL}▶{C_END} E{st['idx']} {st['name']}: {n} 条 / {n_png} 张图 编码中…",
                  flush=True)
            ds.batch_encode_videos(0, n)
            got = sorted(root.rglob("*.mp4"))
            print(f"  {C_OK}✓{C_END} E{st['idx']} {st['name']} 完成 —— {len(got)} 个 mp4, "
                  f"{sum(q.stat().st_size for q in got)/1e6:.1f} MB", flush=True)
        except Exception as e:  # noqa: BLE001
            # 别只印一行就算了 —— 上一版就是这样把真正的报错藏起来的。
            import traceback  # noqa: PLC0415
            print(f"  {C_ERR}✘{C_END} E{st['idx']} {st['name']} 编码失败: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print(f"      图还在 {root}/images, 可单独补:\n"
                  f"      python3 run_chain_4stage.py --finalize-only {run_id}", flush=True)
    rule("═")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nero 四阶段长程串跑(右臂→左臂→stage34→stage56)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="<POLICY_SERVER_IP>", help="policy 服务地址")
    ap.add_argument("--execute", action="store_true", help="真执行(默认只推理只录,不发运动)")
    ap.add_argument("--dry-run", action="store_true", help="不连机械臂/相机,只验编排与路径")
    ap.add_argument("--stages", default="0,1,2,3", help="要跑的阶段,逗号分隔,如 0,1")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="执行节拍倍速(1.0=训练的 15Hz)。⚠ 只改节拍不放宽限幅; "
                         ">1 会让自动切换略微晚切, 建议 ≤1.5")
    ap.add_argument("--manual-start", action="store_true",
                    help="每段开头仍要人按空格(默认串跑自动开始)")
    ap.add_argument("--rescan-devices", action="store_true",
                    help="每段各自扫一遍 CAN/相机(默认 chain 只扫一次再显式下发)")
    ap.add_argument("--auto-switch", action="store_true",
                    help="子任务自动切换: 各段用进度头判断该不该切, 不用人按空格。"
                         "空格仍可手动覆盖。expert 编号取 STAGES 表里的 expert 字段")
    ap.add_argument("--switch-device", default="cuda",
                    help="切换器跑在哪。⚠ CPU 上 SigLIP2 约 2.8Hz, 撑不住 15Hz")
    ap.add_argument("--keep-switcher", action="store_true",
                    help="整条链结束后不停切换服务(下次起省掉 816M 视觉塔的加载时间)")
    ap.add_argument("--auto-advance", action="store_true",
                    help="阶段间不停顿。⚠ 首次串跑不要用 —— 四个模型动作语义不同,"
                         "接续处会跳变")
    ap.add_argument("--multi-episode", action="store_true",
                    help="每个阶段可连录多条(不传 --once)。默认一段一条, 空格走完即进下一大阶段")
    ap.add_argument("--encode-inline", action="store_true",
                    help="每条结束就编码视频(旧行为)。默认推迟到全部跑完后统一编码")
    ap.add_argument("--no-home", action="store_true",
                    help="阶段之间不归位(旧行为)。默认每段结束后把两臂送回 home_pose.json")
    ap.add_argument("--home-rate", type=float, default=0.05,
                    help="归位轨迹的每 tick 关节步进(rad)")
    ap.add_argument("--label-per-stage", action="store_true",
                    help="每个阶段跑完就问 s/f/g(旧行为)。默认整条链结束后统一标一次")
    ap.add_argument("--no-finalize", action="store_true",
                    help="跑完不自动编码视频。之后可用 --finalize-only 补")
    ap.add_argument("--label-only", metavar="RUN_ID",
                    help="只对指定运行 ID 的四段补标注(成功/失败), 不跑任何阶段")
    ap.add_argument("--finalize-only", metavar="RUN_ID",
                    help="只对指定运行 ID 的四个数据集补编码视频, 不跑任何阶段")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="某段非零退出就中止整条链(默认继续问)")
    ap.add_argument("--with-switcher", action="store_true",
                    help="[未实现] 启用 switcher 自动推进子任务。现有客户端把推进权写死在"
                         "键盘分支里(sid += 1 由空格触发),接 switcher 需要改四个客户端的"
                         "主循环,且 E1 送 switcher 的 state 必须是 16 维关节而非 10 维末端"
                         "(progress_head.pt 里 E1 编码器烧死为 128x16)。先把四段串通再做。")
    ap.add_argument("passthrough", nargs="*", default=[],
                    help="透传给各客户端的额外参数")
    args = ap.parse_args()

    # 局域网绕开系统代理 —— 必须在任何联网之前。不加的话连推理服务会报
    # InvalidMessage: did not receive a valid HTTP response(其实是被代理吃了)。
    from proxy_bypass import bypass_proxy  # noqa: PLC0415
    bypass_proxy(getattr(args, "host", "") or "")

    if args.with_switcher:
        sys.exit(f"{C_ERR}--with-switcher 尚未实现,见 --help 里的说明。{C_END}")

    try:
        want = [int(x) for x in args.stages.split(",") if x.strip() != ""]
    except ValueError:
        sys.exit(f"{C_ERR}--stages 格式错误: {args.stages}{C_END}")
    stages = [s for s in STAGES if s["idx"] in want]
    if not stages:
        sys.exit(f"{C_ERR}--stages 没选中任何阶段{C_END}")


    if args.label_only:
        label_run(args.label_only, STAGES, aborted=False)
        return
    if args.finalize_only:
        finalize(args.finalize_only, STAGES)
        return

    run_id = os.environ.get("WP_RUN") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{C_HL}╔══ Nero 四阶段长程串跑 ══╗{C_END}")
    print(f"  运行 ID  {C_HL}{run_id}{C_END}   {C_DIM}(补跑用: export WP_RUN={run_id}){C_END}")
    print("  阶段     " + " → ".join(f"E{s['idx']} {s['name']}" for s in stages))
    mode = "干跑(mock)" if args.dry_run else ("真执行" if args.execute else "真机只读")
    print(f"  模式     {C_WARN if args.execute else C_DIM}{mode}{C_END}")
    print(f"  数据根   {C_DIM}{DATA_ROOT / run_id}{C_END}")

    args._devices = [] if (args.rescan_devices or args.dry_run) else resolve_devices()
    if args._devices:
        print(f"  设备     {C_DIM}" +
              "  ".join(f"{args._devices[i][2:]}={args._devices[i+1]}"
                        for i in range(0, len(args._devices), 2)) + f"{C_END}", flush=True)

    if not preflight(args, stages):
        sys.exit(f"\n{C_ERR}预检未过 —— 没有启动任何阶段。{C_END}")

    if args.execute and not args.dry_run:
        print(f"\n{C_ERR}⚠ 真执行模式。确认已悬吊 / 现场有人守急停。{C_END}")
        try:
            if input("  输入 GO 开始,其它任意键取消: ").strip() != "GO":
                sys.exit("已取消。")
        except (EOFError, KeyboardInterrupt):
            sys.exit("\n已取消。")

    results = []
    i = 0
    aborted = False
    user_failed = False
    failed_st = None
    while i < len(stages):
        st = stages[i]
        if i > 0 and not gate(st, args):
            print(f"{C_WARN}用户中止整条链。{C_END}", flush=True)
            aborted = True
            break
        if st.get("_skip"):
            print(f"{C_DIM}跳过 E{st['idx']} {st['name']}{C_END}", flush=True)
            results.append((st, None)); i += 1
            continue

        rc = run_stage(st, args, run_id)
        results = [(s, r) for s, r in results if s is not st]   # 重试时覆盖旧结果
        results.append((st, rc))

        # ── Ctrl-C(130): 你主动喊停 → 立刻结束整条链, 一句都不再问 ──────────
        #    以前这里还会弹 input(), 于是你得连按好几次 Ctrl-C 才退得掉。
        # rc=20: 你在某段按了 f 判失败 → 立刻停整条链, 并统一编码视频。
        if rc == EXIT_USER_FAIL:
            print(f"\n{C_WARN}⚠ 你在 E{st['idx']} {st['name']} 按了 f 判失败 "
                  f"→ 停止整条链, 保存已录的视频。{C_END}", flush=True)
            user_failed = True
            failed_st = st
            break

        if rc == 130:
            print(f"\n{C_WARN}⚠ E{st['idx']} {st['name']} 被 Ctrl-C 中断 → 立即结束整条链。{C_END}",
                  flush=True)
            aborted = True
            break

        if rc == 0:
            # 归位已由客户端在本段结束时做过(goto_home_inline), 这里不再重复。
            # 见 run_stage() 里 --home-rate / --no-home 的下发。
            i += 1
            continue

        # ── 真失败: 给三个选择, 默认可以直接重试本段 ────────────────────────
        print(f"\n{C_ERR}✘ E{st['idx']} {st['name']} 退出码 {rc}{C_END}", flush=True)
        if args.stop_on_fail:
            print(f"{C_ERR}--stop-on-fail: 中止整条链。{C_END}", flush=True)
            aborted = True
            break
        print(f"{C_WARN}⚠ 该段失败(常见原因: 网络抖动导致推理服务断开)。{C_END}", flush=True)
        print(f"  {C_HL}回车 / r{C_END} = 重跑本段 E{st['idx']} {st['name']}", flush=True)
        print(f"  {C_HL}n{C_END}        = 跳过本段, 进下一段", flush=True)
        print(f"  {C_HL}q{C_END}        = 结束整条链", flush=True)
        try:
            ans = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C_WARN}中止整条链。{C_END}", flush=True)
            aborted = True
            break
        if ans in ("", "r"):
            print(f"{C_HL}↻ 重跑 E{st['idx']} {st['name']}{C_END}", flush=True)
            continue          # i 不变 → 同一段再来一次
        if ans == "n":
            i += 1
            continue
        aborted = True
        break

    # ── 哪些段要保存(标注 + 编码) ──────────────────────────────────────────
    # rc=0            正常跑完 → 存
    # rc=EXIT_USER_FAIL 你按了 f 判失败 → **也存**。那一条正是"失败到什么样子"的
    #                 证据, 不存就等于白跑; 客户端按 f 时已经把 episode 落盘了。
    # 其它非零        真崩了(连不上/异常), 数据不完整, 不存。
    done = [st for st, rc in results if rc in (0, EXIT_USER_FAIL)]

    if done and not args.label_per_stage and not args.dry_run:
        try:
            label_run(run_id, done, aborted, failed_st)
        except KeyboardInterrupt:
            print(f"\n{C_WARN}跳过标注。可补: python3 run_chain_4stage.py "
                  f"--label-only {run_id}{C_END}", flush=True)

    # 被 Ctrl-C 中止时不编 —— 你喊停了就该立刻停, 不该再卡在几分钟的编码里。
    # 但按 f 是"这次失败了, 把录像留下来", 语义相反, 必须编。
    if user_failed and done:
        print(f"\n{C_WARN}按 f 判失败 —— 仍保存已录的视频({len(done)} 段)。{C_END}", flush=True)
    elif aborted and done:
        print(f"\n{C_WARN}已中止 → 跳过视频编码。之后可补:{C_END}\n"
              f"  python3 run_chain_4stage.py --finalize-only {run_id}", flush=True)
        done = []
    if done and not args.no_finalize and not args.encode_inline and not args.dry_run:
        try:
            finalize(run_id, done)
        except KeyboardInterrupt:
            print(f"\n{C_WARN}⚠ 编码被中断。之后可补: "
                  f"python3 run_chain_4stage.py --finalize-only {run_id}{C_END}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"\n{C_ERR}✘ 编码出错: {e}\n"
                  f"   可补: python3 run_chain_4stage.py --finalize-only {run_id}{C_END}", flush=True)
    elif not done:
        print(f"\n{C_DIM}(没有完整跑完的阶段, 跳过视频编码){C_END}", flush=True)

    if args.auto_switch and not args.keep_switcher:
        try:
            sys.path.insert(0, str(HERE))
            from switcher_client import stop_switcher_service  # noqa: PLC0415
            stop_switcher_service()
        except Exception as e:  # noqa: BLE001
            print(f"  {C_DIM}(停切换服务失败: {e}){C_END}", flush=True)

    rule("═")
    print(f"{C_HL}══ 汇总 ══{C_END}   运行 ID {run_id}")
    for st, rc in results:
        mark = (f"{C_DIM}跳过{C_END}" if rc is None else
                f"{C_OK}✓ 完成{C_END}" if rc == 0 else
                f"{C_WARN}⚑ 你判失败(已存){C_END}" if rc == EXIT_USER_FAIL else
                f"{C_ERR}✘ 退出码 {rc}{C_END}")
        out = DATA_ROOT / run_id / st["tag"]
        n = len(list((out / "data").rglob("*.parquet"))) if (out / "data").exists() else 0
        print(f"  E{st['idx']} {st['name']:<14} {mark:<22} {n} 条  {C_DIM}{out}{C_END}")
    print(f"\n  数据根目录: {DATA_ROOT / run_id}")
    rule("═")


if __name__ == "__main__":
    main()
