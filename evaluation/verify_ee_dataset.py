#!/usr/bin/env python3
"""nero hezi 封盖「末端位姿版」数据集 + π0.5 训练管线验证。

用法:
  python verify_ee_dataset.py             # 数据集级(不需 norm stats)
  python verify_ee_dataset.py --pipeline  # 追加全管线批次(需 norm stats 已算好+已兜底)
运行前需 export HF_LEROBOT_HOME=<DATA_ROOT>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "<EE_PACK>/tools")
import ee_repr as E  # noqa: E402

CONFIG_NAME = "pi05_nero_hezi_closing_ee_v1"
REPO_ID = "local/nero_hezi_closing_ee_v1"
EXPECT_EPS = 100
EXPECT_FRAMES = 87649
EXPECT_TASKS = 7
SPLIT_EP = 70          # 0..69 来自 merged_v2, 70..99 来自 supplement_v1
DATA_ROOT = Path("<DATA_ROOT>/local/nero_hezi_closing_ee_v1")

ok = True


def check(name, cond, detail=""):
    global ok
    print(("✓" if cond else "✗"), name, " ", detail, flush=True)
    ok = ok and bool(cond)


def dataset_checks():
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID)
    m = ds.meta
    check("集数/帧数", m.total_episodes == EXPECT_EPS and m.total_frames == EXPECT_FRAMES,
          f"{m.total_episodes}集/{m.total_frames}帧 (期望 {EXPECT_EPS}/{EXPECT_FRAMES})")
    check("fps=15", int(m.fps) == 15, f"fps={m.fps}")
    tasks = list(m.tasks.values()) if hasattr(m.tasks, "values") else list(m.tasks)
    check("7 段 stage prompt", len(tasks) == EXPECT_TASKS, f"{len(tasks)}段")
    joined = " || ".join(str(t) for t in tasks).lower()
    check("prompt 是封盖语义(含 flap)", "flap" in joined)
    check("prompt 非空", all(len(str(t)) > 5 for t in tasks))
    feat = m.info["features"]

    def shape_of(k):   # lerobot 会把 info.json 里的 shape 解析成 tuple, 统一成 list 再比
        return list(feat.get(k, {}).get("shape") or [])

    check("state 声明 20 维", shape_of("observation.state") == [20],
          str(shape_of("observation.state")))
    check("action 声明 20 维(落盘是绝对末端位姿)", shape_of("action") == [20],
          str(shape_of("action")))
    check("state names 是末端位姿命名",
          list(feat["observation.state"]["names"]) == E.STATE_NAMES)
    check("原关节 16 维也在",
          shape_of("observation.state_joint16") == [16] and shape_of("action_joint16") == [16],
          f"{shape_of('observation.state_joint16')}/{shape_of('action_joint16')}")

    # 两半各抽 5 帧(0..69 来自 merged_v2, 70..99 来自 supplement_v1)
    n = m.total_frames
    idxs = [0, n // 8, n // 4, 3 * n // 8, n // 2 - 1,
            n // 2, 5 * n // 8, 3 * n // 4, 7 * n // 8, n - 1]
    seen_halves = set()
    for i in idxs:
        it = ds[int(i)]
        s = np.asarray(it["observation.state"], dtype=np.float64)
        a = np.asarray(it["action"], dtype=np.float64)
        j = np.asarray(it["observation.state_joint16"], dtype=np.float64)
        ep = int(np.asarray(it["episode_index"]))
        seen_halves.add(ep < SPLIT_EP)
        check(f"帧{i}(ep{ep}): state/action 20 维", s.shape == (20,) and a.shape == (20,),
              f"{s.shape}/{a.shape}")
        check(f"帧{i}: action ≡ state(拖动示教)", bool(np.abs(s - a).max() < 1e-5),
              f"max|Δ|={np.abs(s - a).max():.2e}")
        # 关节 → 末端 的转换正确性:用落盘的 16 维关节重算 FK, 必须还原出落盘的 20 维
        s_re = E.joints16_to_ee20(j[None])[0]
        check(f"帧{i}: FK(关节16) ≡ 落盘末端20", bool(np.abs(s_re - s).max() < 2e-5),
              f"max|Δ|={np.abs(s_re - s).max():.2e}")
        for arm, sl in (("左", slice(3, 9)), ("右", slice(13, 19))):
            R = E.rot6d_to_mat(s[sl][None])[0]
            err = float(np.abs(R.T @ R - np.eye(3)).max())
            check(f"帧{i}: {arm}臂 rot6d 正交", err < 1e-5, f"RᵀR-I={err:.2e}")
        check(f"帧{i}: 爪宽 0~0.12m",
              bool(min(s[9], s[19]) >= -1e-6 and max(s[9], s[19]) <= 0.12),
              f"[{min(s[9], s[19]):.5f},{max(s[9], s[19]):.5f}]")
        check(f"帧{i}: 末端位置量纲合理(|p|<1.5m)",
              bool(np.linalg.norm(s[0:3]) < 1.5 and np.linalg.norm(s[10:13]) < 1.5),
              f"|pL|={np.linalg.norm(s[0:3]):.3f} |pR|={np.linalg.norm(s[10:13]):.3f}")
        for k in ("observation.images.third_view", "observation.images.left_wrist",
                  "observation.images.right_wrist"):
            img = np.asarray(it[k])
            check(f"帧{i}: {k.split('.')[-1]}形状",
                  tuple(img.shape) in {(3, 480, 640), (480, 640, 3)}, str(tuple(img.shape)))
            check(f"帧{i}: {k.split('.')[-1]}非全零", float(np.abs(img).sum()) > 0)
    check("抽帧覆盖两份源(merged_v2 + supplement_v1)", seen_halves == {True, False},
          f"halves={seen_halves}")

    # 逐集: index 全局连续 / prompt 与 task_index 一致 / 无冻结集
    import json

    import pyarrow.parquet as pq
    tasks_ref = [json.loads(l)["task"] for l in (DATA_ROOT / "meta/tasks.jsonl").open()]
    nxt = 0
    bad_prompt = frozen = 0
    for e in range(EXPECT_EPS):
        t = pq.read_table(DATA_ROOT / f"data/chunk-000/episode_{e:06d}.parquet",
                          columns=["index", "episode_index", "task_index", "prompt_text",
                                   "observation.state_joint16"])
        d = t.to_pydict()
        if d["index"][0] != nxt or d["index"][-1] != nxt + t.num_rows - 1:
            check(f"ep{e}: index 连续", False, f"起 {d['index'][0]} 期望 {nxt}")
        nxt += t.num_rows
        if set(d["episode_index"]) != {e}:
            check(f"ep{e}: episode_index", False, str(set(d['episode_index'])))
        bad_prompt += sum(1 for i in range(t.num_rows)
                          if d["prompt_text"][i] != tasks_ref[d["task_index"][i]])
        j = np.asarray([np.asarray(x) for x in d["observation.state_joint16"]], float)
        rng = j.max(0) - j.min(0)
        if max(rng[0:7].max(), rng[8:15].max()) < 1e-6:
            frozen += 1
    check("index 全局连续且总数对", nxt == EXPECT_FRAMES, f"{nxt}")
    check("prompt_text ≡ tasks[task_index] (逐行)", bad_prompt == 0, f"{bad_prompt} 行不符")
    check("冻结集(全关节 range=0) 为 0", frozen == 0, f"{frozen} 集")


def math_crosscheck():
    """openpi 侧的相对动作数学 必须与 ee_pack/tools/ee_repr.py 那份等价。

    两份实现是有意分开的(训练管线不依赖 <NVME_ROOT> 的脚本路径), 这条断言就是把它们
    钉在一起 —— 改了任一边不跑这条, 训出来的动作语义可能和数据准备时不一致。
    """
    import openpi.policies.nero_ee_policy as P

    rng = np.random.default_rng(0)
    lim = E.fk_nero.limits("right")
    q = np.concatenate([rng.uniform(lim[:, 0], lim[:, 1], size=(6 * 9, 7)),
                        rng.uniform(0, .12, (6 * 9, 1)),
                        rng.uniform(lim[:, 0], lim[:, 1], size=(6 * 9, 7)),
                        rng.uniform(0, .12, (6 * 9, 1))], 1)
    ee = E.joints16_to_ee20(q)
    s20, a20 = ee[:6], ee[6:].reshape(6, 8, 20)

    # world / tool 两个坐标系都比一遍(config 里 ee_action_frame 可切, 两条路径都要有断言)
    for frame in ("world", "tool"):
        ref = E.ee20_to_rel14(s20, a20, frame=frame)
        got = np.stack([P.EEDeltaActions(frame=frame)(
            {"state": s20[i], "actions": a20[i].copy()})["actions"] for i in range(6)])
        check(f"EEDeltaActions ≡ ee_repr.ee20_to_rel14 (frame={frame})",
              float(np.abs(got - ref).max()) < 1e-5, f"max|Δ|={np.abs(got - ref).max():.2e}")

        # 逆变换: 相对 → 绝对, 必须还原原始绝对位姿(state 补零到 32 维, 模拟推理时的形状)
        back = np.stack([P.EEAbsoluteActions(frame=frame)(
            {"state": np.pad(s20[i], (0, 12)),
             "actions": np.pad(got[i].astype(np.float64), ((0, 0), (0, 18)))})["actions"]
            for i in range(6)])
        check(f"EEAbsoluteActions 还原 ≡ 原绝对末端位姿 (frame={frame})",
              float(np.abs(back - a20).max()) < 1e-6, f"max|Δ|={np.abs(back - a20).max():.2e}")

    # 训练/推理必须用同一帧: 拿 world 训出来的量用 tool 还原, 必须还原不出来(哨兵)
    rel_w = E.ee20_to_rel14(s20, a20, frame="world")
    bad = E.rel14_to_ee20(s20, rel_w, frame="tool")
    check("帧不匹配会还原错(证明 frame 真的生效)",
          float(np.abs(bad - a20).max()) > 1e-3, f"max|Δ|={np.abs(bad - a20).max():.3f}")

    # 与 config 实际配置的帧一致性: 训练侧和推理侧必须是同一个 frame
    import openpi.training.config as _cfgmod
    _c = _cfgmod.get_config(CONFIG_NAME)
    _dc = _c.data.create(_c.assets_dirs, _c.model)
    fin = [t.frame for t in _dc.data_transforms.inputs if isinstance(t, P.EEDeltaActions)]
    fout = [t.frame for t in _dc.data_transforms.outputs if isinstance(t, P.EEAbsoluteActions)]
    check("config 里训练侧/推理侧 frame 一致", fin == fout and len(fin) == 1,
          f"inputs={fin} outputs={fout}")

    # chunk 锚点必须是首帧 state: 令 a[0]=state, 则相对动作第 0 步的位置/旋转分量应为 0
    a0 = a20.copy()
    a0[:, 0] = s20
    r0 = E.ee20_to_rel14(s20, a0)[:, 0]
    posrot = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    check("相对动作第 0 步 ≡ 0(chunk 锚点 = 首帧 state)",
          float(np.abs(r0[:, posrot]).max()) < 1e-9, f"max|Δ|={np.abs(r0[:, posrot]).max():.2e}")


def pipeline_checks():
    import openpi.training.config as _config
    import openpi.training.data_loader as _dl

    cfg = _config.get_config(CONFIG_NAME)
    loader = _dl.create_data_loader(cfg, num_batches=2, shuffle=True)
    obs, actions = next(iter(loader))
    imgs = obs.images
    check("三路图像槽", set(imgs) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"},
          str(list(imgs)))
    for k in imgs:
        v = np.asarray(imgs[k])
        check(f"{k} (B,224,224,3)", tuple(v.shape[1:]) == (224, 224, 3), str(v.shape))
        check(f"{k} 是真图(非全零)", float(np.abs(v).sum()) > 0)
    mk = obs.image_masks
    check("三路mask全True", all(bool(np.all(mk[k])) for k in mk))

    st = np.asarray(obs.state)
    ac = np.asarray(actions)
    check("state (B,32) 且 20 维之后填 0",
          st.shape[-1] == 32 and float(np.abs(st[..., 20:]).max()) == 0.0, str(st.shape))
    check("actions (B,50,32) 且 14 维之后填 0",
          ac.shape[-2:] == (50, 32) and float(np.abs(ac[..., 14:]).max()) == 0.0, str(ac.shape))
    check("无NaN", not (np.isnan(st).any() or np.isnan(ac).any()))
    check("tokenized_prompt 存在", obs.tokenized_prompt is not None)
    # 铁律: 归一化后幅值有界。两个夹爪维近常数(左爪全程 range 0.57mm), 是 std≈0 除以 eps
    # 爆炸的高危维 —— 必须先跑 floor_norm_stats_ee.py 兜底, 这条就是盯它的。
    amax = float(np.abs(ac[..., :14]).max())
    smax = float(np.abs(st[..., :20]).max())
    check("归一化 actions 幅值<20(防爪常数维爆炸)", amax < 20.0, f"max|a|={amax:.2f}")
    check("归一化 state 幅值<20", smax < 20.0, f"max|s|={smax:.2f}")


if __name__ == "__main__":
    dataset_checks()
    math_crosscheck()
    if "--pipeline" in sys.argv:
        pipeline_checks()
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)
