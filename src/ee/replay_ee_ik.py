#!/usr/bin/env python3
"""离线回放:关节 → FK → 相对动作 → 还原绝对位姿 → IK → 关节,和录制关节角对账。

这是「训练前须知_ee.md §7.2c」那道防线 —— **不跑这个别上真机**。它把模型排除在外
(用真值相对动作), 于是量到的误差纯粹来自 末端位姿表示 + IK 这条链, 和关节版当年
"离线回放 ALL PASS(关节 MAE 0.011 rad)"是同一件事。

严格照部署时序走(比真机更悲观):
  · 每 RE_INFER(默认 25) 帧一个 chunk, 用 chunk 起始那帧当锚点(与训练侧 chunk 锚定一致);
  · 零空间锚点 q_ref = chunk 起始的**实测**关节角, 整个 chunk 不变;
  · 迭代种子 = 上一步**自己算出来的**指令(开环, 不拿真机反馈纠偏)→ 误差会累积, 这样测出的
    是上界;
  · 静止臂死区: 整段 Δ 低于阈值就冻结该臂关节, 不调 IK。

用法:
  python replay_ee_ik.py                       # 默认 20 集 x 3 种模式对照
  python replay_ee_ik.py --episodes 100        # 全量
  python replay_ee_ik.py --modes full          # 只跑推荐配置
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ee_repr as E  # noqa: E402
import ik_nero as K  # noqa: E402

DATA = Path("<DATA_ROOT>/local/nero_hezi_closing_ee_v1")
H = 50            # pi0.5 action_horizon
RE_INFER = 25     # 部署里每 25 帧重推理一次(与 run_pi05_deploy_* 一致)

# 三种模式:推荐配置 / 拆掉死区 / 拆掉固定锚点 —— 用真数据量每道防线各值多少
MODES = {
    # 前三个: 每 chunk 把种子重置回录制关节角(乐观, 藏住跨 chunk 累积漂移)
    "full":        dict(deadband=True,  fixed_ref=True),
    "no_deadband": dict(deadband=False, fixed_ref=True),
    "no_ref":      dict(deadband=True,  fixed_ref=False),
    # 后两个: 种子跨 chunk 延续(真机最坏情形), 专门对比固定锚点值不值
    "cont":        dict(deadband=True,  fixed_ref=True,  continuous=True),
    "cont_no_ref": dict(deadband=True,  fixed_ref=False, continuous=True),
}


def episode_joints(e: int) -> np.ndarray:
    d = pq.read_table(DATA / f"data/chunk-000/episode_{e:06d}.parquet",
                      columns=["observation.state_joint16"]).to_pydict()
    return np.asarray([np.asarray(x) for x in d["observation.state_joint16"]], np.float64)


def replay_episode(q: np.ndarray, deadband: bool, fixed_ref: bool,
                   continuous: bool = False) -> dict:
    """continuous=True: 种子跨 chunk 延续、锚点取上一 chunk 末的**指令**(而非录制关节角)。
    这才是真机的最坏情形 —— 默认的每 chunk 重置回录制值会把跨 chunk 累积漂移藏起来。"""
    n = len(q)
    ee = E.joints16_to_ee20(q)                       # (n,20) 真值绝对末端
    jerr_L, jerr_R, perr, rerr = [], [], [], []
    idle_L_joint, idle_R_joint = [], []              # 静止臂的关节偏差(甩肘就看这个)
    step_cmd, step_rec = [], []                      # 逐步关节增量: 指令 vs 录制
    perr_bad, rerr_bad = [], []                      # 被拒解那些步的残差(不会下发)
    perr_at_bigj = []                                # 关节偏差大的那些步, 末端误差是多少
    out_env = 0                                      # 指令越出实测包络的次数
    n_ik = n_fail = n_frz_L = n_frz_R = n_step = 0
    envL, envR = K.effective_limits("left"), K.effective_limits("right")
    q_cmd = q[0].copy()
    # continuous 下还额外量"指令末端 vs 录制末端"的任务空间漂移 —— 相对动作从漂开的
    # 原点出发, 形状对但位置会偏; 真机靠每 25 帧带新观测重推理闭环, 离线测不到那一环,
    # 所以这个数是**上界**。
    task_drift = []

    for t in range(0, n - 1, RE_INFER):
        idx = np.minimum(t + np.arange(H), n - 1)
        rel = E.ee20_to_rel14(ee[t], ee[idx])        # 真值相对动作(模型该输出的东西)
        # 锚点必须是**当前实际位姿**, 与真机一致: 模型的 state 输入是 FK(实测关节),
        # EEAbsoluteActions 就是相对它还原目标。continuous 下指令会漂离录制值, 若仍拿
        # 录制位姿当锚点, 就造出真机不存在的错配(目标锚在录制、手臂在别处), 冻结的臂
        # 永远追不上 → 实测能虚报到 55°/107° 的假漂移。
        anchor = E.joints16_to_ee20(q_cmd[None])[0] if (continuous and t > 0) else ee[t]
        tgt = E.rel14_to_ee20(anchor, rel)           # 还原绝对位姿(推理侧 EEAbsoluteActions)
        frz_L = deadband and K.arm_is_idle(rel, "left")
        frz_R = deadband and K.arm_is_idle(rel, "right")
        # 真实静止判定(不受 deadband 开关影响), 用来挑出该盯的那条臂
        true_idle_L = K.arm_is_idle(rel, "left")
        true_idle_R = K.arm_is_idle(rel, "right")

        if not continuous or t == 0:
            q_cmd = q[t].copy()
        q_ref = q_cmd.copy() if fixed_ref else None
        for i in range(min(RE_INFER, n - t)):
            q_prev = q_cmd.copy()
            q_cmd, info = K.ee20_to_joints16(
                tgt[i], q_cmd, ref_q16=q_ref if fixed_ref else None,
                freeze_left=frz_L, freeze_right=frz_R)
            gt = q[min(t + i, n - 1)]
            if continuous:
                cur = E.joints16_to_ee20(q_cmd[None])[0]
                rec = ee[min(t + i, n - 1)]
                task_drift.append(float(max(
                    np.linalg.norm(cur[0:3] - rec[0:3]),
                    np.linalg.norm(cur[10:13] - rec[10:13]))))
            step_cmd.append(float(np.abs(np.concatenate(
                [q_cmd[E.J_LEFT] - q_prev[E.J_LEFT], q_cmd[E.J_RIGHT] - q_prev[E.J_RIGHT]])).max()))
            if t + i > 0:
                gp = q[min(t + i - 1, n - 1)]
                step_rec.append(float(np.abs(np.concatenate(
                    [gt[E.J_LEFT] - gp[E.J_LEFT], gt[E.J_RIGHT] - gp[E.J_RIGHT]])).max()))
            for sl, env in ((E.J_LEFT, envL), (E.J_RIGHT, envR)):
                a = q_cmd[sl]
                out_env += int(((a < env[:, 0] - 1e-9) | (a > env[:, 1] + 1e-9)).any())
            eL = float(np.abs(q_cmd[E.J_LEFT] - gt[E.J_LEFT]).max())
            eR = float(np.abs(q_cmd[E.J_RIGHT] - gt[E.J_RIGHT]).max())
            jerr_L.append(eL); jerr_R.append(eR)
            if true_idle_L:
                idle_L_joint.append(eL)
            if true_idle_R:
                idle_R_joint.append(eR)
            for arm, ej in (("left", eL), ("right", eR)):
                inf = info[arm]
                if inf is None:
                    continue
                n_ik += 1
                n_fail += (not inf["ok"])
                (perr if inf["ok"] else perr_bad).append(inf["pos_err"])
                (rerr if inf["ok"] else rerr_bad).append(inf["rot_err"])
                if ej > 0.1:                      # 关节差 >0.1rad 的步, 末端误差多少?
                    perr_at_bigj.append(inf["pos_err"])
            n_frz_L += frz_L; n_frz_R += frz_R
            n_step += 1
    return dict(jerr_L=np.asarray(jerr_L), jerr_R=np.asarray(jerr_R),
                perr=np.asarray(perr), rerr=np.asarray(rerr),
                perr_bad=np.asarray(perr_bad), rerr_bad=np.asarray(rerr_bad),
                idle_L=np.asarray(idle_L_joint), idle_R=np.asarray(idle_R_joint),
                step_cmd=np.asarray(step_cmd), step_rec=np.asarray(step_rec),
                perr_at_bigj=np.asarray(perr_at_bigj),
                task_drift=np.asarray(task_drift),
                n_ik=n_ik, n_fail=n_fail, n_frz_L=n_frz_L, n_frz_R=n_frz_R,
                n_step=n_step, out_env=out_env)


def merge(rs: list[dict]) -> dict:
    out = {}
    for k in ("jerr_L", "jerr_R", "perr", "rerr", "perr_bad", "rerr_bad",
              "idle_L", "idle_R", "step_cmd", "step_rec", "perr_at_bigj",
              "task_drift"):
        out[k] = np.concatenate([r[k] for r in rs]) if rs else np.zeros(0)
    for k in ("n_ik", "n_fail", "n_frz_L", "n_frz_R", "n_step", "out_env"):
        out[k] = sum(r[k] for r in rs)
    return out


def report(tag: str, m: dict) -> bool:
    jl, jr = m["jerr_L"], m["jerr_R"]
    ok = True

    def line(name, cond, detail):
        nonlocal ok
        print(("  ✓" if cond else "  ✗"), name, " ", detail, flush=True)
        ok = ok and bool(cond)

    cont = tag.startswith("cont")
    print(f"\n———— 模式 {tag} {'(诊断项, 不作判据)' if cont else '(★判据模式)'} ————")
    if cont:
        print("  ⚠ 本模式整集全程开环、一次都不重新观测 —— **不是部署的模型**。相对动作会把"
              "任何既有偏差\n     原样保留、永不纠正(现役关节 delta 版同理), 真机靠每 25 帧带"
              "新图像重推理闭环纠正。\n     这里的漂移数只用来量化「不重新观测会怎样」, 不代表"
              "真机误差。")
    print(f"  步数 {m['n_step']}  IK 调用 {m['n_ik']}  冻结(左/右) "
          f"{m['n_frz_L']}/{m['n_frz_R']} ({100*m['n_frz_L']/max(m['n_step'],1):.1f}%/"
          f"{100*m['n_frz_R']/max(m['n_step'],1):.1f}%)")
    # 被拒解的步(ok=False)**不会下发** —— 调用侧保持上一条指令。所以跟踪精度只统计
    # 会真正下发的那些步; 被拒率单独作为一条判据。
    rate = m["n_fail"] / max(m["n_ik"], 1)
    line("IK 被拒率 < 0.1%(被拒的步保持上一条指令, 不下发)", rate < 1e-3,
         f"{m['n_fail']}/{m['n_ik']} = {100*rate:.4f}%" +
         (f"  这些步残差 max={m['perr_bad'].max()*1000:.3f}mm/"
          f"{np.degrees(m['rerr_bad']).max():.4f}°" if len(m["perr_bad"]) else ""))
    if len(m["perr"]):
        line("已下发步 末端位置残差 < 0.5mm", m["perr"].max() < 5e-4,
             f"max={m['perr'].max()*1000:.4f}mm  p99={np.percentile(m['perr'],99)*1000:.4f}mm")
        line("已下发步 末端姿态残差 < 0.1°", np.degrees(m["rerr"]).max() < 0.1,
             f"max={np.degrees(m['rerr']).max():.5f}°")
    line("指令关节不越出实测包络", m["out_env"] == 0, f"越出 {m['out_env']} 次")
    if len(m["step_cmd"]) and len(m["step_rec"]):
        sc, sr = m["step_cmd"], m["step_rec"]
        line("逐步关节增量不超过示教本身(p99)",
             np.percentile(sc, 99) <= np.percentile(sr, 99) * 1.5,
             f"指令 p99={np.percentile(sc,99):.5f} / max={sc.max():.5f} rad  vs  "
             f"录制 p99={np.percentile(sr,99):.5f} / max={sr.max():.5f} rad")
    # 关节偏差是**零空间分支**差异, 不是跟踪失败 —— 用"大关节偏差步的末端误差仍很小"来证明。
    if len(m["perr_at_bigj"]):
        line("关节偏差>0.1rad 的步末端仍准(证明只是肘部分支不同)",
             m["perr_at_bigj"].max() < 5e-4,
             f"这些步末端 max={m['perr_at_bigj'].max()*1000:.4f}mm  n={len(m['perr_at_bigj'])}")
    # 关节 MAE 只作诊断: 录制的肘部构型只是无穷多合法零空间解之一, 不要求复现。
    print(f"  · 诊断(非判据) 关节偏差: 左 MAE={jl.mean():.5f} p99={np.percentile(jl,99):.5f} "
          f"max={jl.max():.5f} | 右 MAE={jr.mean():.5f} p99={np.percentile(jr,99):.5f} "
          f"max={jr.max():.5f} rad")
    if len(m["task_drift"]):
        td = m["task_drift"]
        line("任务空间漂移(指令末端 vs 录制末端) < 30mm", td.max() < 0.03,
             f"MAE={td.mean()*1000:.3f}mm  p99={np.percentile(td,99)*1000:.3f}mm  "
             f"max={td.max()*1000:.3f}mm  (离线开环上界, 真机每 25 帧带新观测闭环)")
    for nm, a in (("左", m["idle_L"]), ("右", m["idle_R"])):
        if len(a):
            line(f"{nm}臂**静止时**关节偏差 < 0.01 rad(甩肘哨兵)", a.max() < 0.01,
                 f"MAE={a.mean():.6f}  max={a.max():.6f} rad ({np.degrees(a.max()):.3f}°)"
                 f"  n={len(a)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    args = ap.parse_args()

    if not E.selftest() or not K.selftest():
        sys.exit("✘ ee_repr / ik_nero selftest 未通过, 停止")
    n_eps = min(args.episodes,
                json.loads((DATA / "meta/info.json").read_text())["total_episodes"])
    print(f"\n=== 离线 FK→IK 回放: 前 {n_eps} 集, 每 {RE_INFER} 帧一个 chunk "
          f"(H={H}), 开环累积 ===")

    all_ok = True
    for mode in args.modes:
        kw = MODES[mode]
        t0 = time.perf_counter()
        rs = []
        for e in range(n_eps):
            rs.append(replay_episode(episode_joints(e), **kw))
            if (e + 1) % 5 == 0:
                print(f"    {mode}: {e+1}/{n_eps} 集  ({time.perf_counter()-t0:.0f}s)",
                      flush=True)
        m = merge(rs)
        okm = report(mode, m)
        print(f"  用时 {time.perf_counter()-t0:.0f}s")
        if mode == "full":
            all_ok = okm          # 只有 full 决定通过与否; 其余是对照/诊断
    print("\nREPLAY PASS" if all_ok else "\nREPLAY FAILED", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
