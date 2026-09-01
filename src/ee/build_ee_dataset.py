#!/usr/bin/env python3
"""合并 hezi 封盖 70 集 + 30 集补充集, 并把 16 维关节改写成 20 维「绝对末端位姿」。

产出: local/nero_hezi_closing_ee_v1  (100 集 / 87649 帧 / 7 段 prompt / 三路 RGB)

三件非做不可的改写(漏一个整份数据就废) —— 沿用 merge_datasets.py 的教训
------------------------------------------------------------------------
1. **task_index 重映射**:两份数据 7 条 task 文本相同但索引被打乱。以 70 集(A)的
   映射为准, 按**文本**把 30 集(B)的 task_index 翻译过去; 每行再断言
   prompt_text == tasks_A[task_index], 对不上直接退出。
2. **episode_index / index 重编号**:B 的 30 集接到 70 之后成 70..99; `index` 是
   全数据集连续的全局帧号, 必须整体平移。
3. **episodes_stats 必须重算**:state/action 从 16 维变 20 维, 直接抄源 stats 会
   形状不符。图像 stats 抄源(重算要解全部视频, 没必要)。

末端位姿口径(与 ee_repr.py / 用户给的取姿方法逐字一致)
  T = fk(q_rad, side=..., tcp=TCP_FINGERTIP)   # 世界系 4x4, TCP=指尖 0.138
  20 维 = [左 pos3, 左 rot6d, 左爪宽, 右 pos3, 右 rot6d, 右爪宽]
  parquet 里 observation.state 与 action **都存绝对末端位姿**(拖动示教 action≡state,
  已逐集核过)。相对动作不落盘 —— 由训练管线的 EEDeltaActions 按 chunk 锚点现算,
  和现役关节 delta 配方同一套语义(见 nero_ee_policy.py)。原始 16 维关节另存
  observation.state_joint16 / action_joint16, 供部署侧 IK 与后续分析用。

视频: 用**硬链接**(两份源同在 <NVME_ROOT>, 零拷贝零额外占盘)。逐集逐路核帧数
      == parquet 行数, 不符即停。

用法:
  python build_ee_dataset.py [--out DIR] [--copy-videos] [--limit N] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ee_repr as E  # noqa: E402

SRC_A = Path("<DATA_ROOT>/local/nero_hezi_closing_refined_merged_v2")      # 70 集
SRC_B = Path("<DATA_ROOT>/local/nero_hezi_closing_refined_supplement_v1")  # 30 集
OUT_DEFAULT = Path("<DATA_ROOT>/local/nero_hezi_closing_ee_v1")

VIEWS = ["observation.images.left_wrist",
         "observation.images.right_wrist",
         "observation.images.third_view"]
STATE_KEY, ACTION_KEY = "observation.state", "action"
J_STATE_KEY, J_ACTION_KEY = "observation.state_joint16", "action_joint16"
JOINT16_NAMES = [f"left_j{i + 1}" for i in range(7)] + ["left_gripper_width"] + \
                [f"right_j{i + 1}" for i in range(7)] + ["right_gripper_width"]

ok = True


def fail(msg: str):
    sys.exit(f"✘ {msg}")


def nframes(p: Path) -> int:
    """先读容器 nb_frames(快); 拿不到再 -count_frames 解一遍(慢但准)。"""
    for extra in ([], ["-count_frames"]):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", *extra,
                            "-show_entries",
                            "stream=nb_read_frames" if extra else "stream=nb_frames",
                            "-of", "csv=p=0", str(p)], capture_output=True, text=True)
        try:
            return int(r.stdout.strip().rstrip(","))
        except ValueError:
            continue
    return -1


def col_stats(x: np.ndarray, n: int) -> dict:
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "count": [n]}


def fixed_list(dim: int) -> pa.DataType:
    """复刻源 schema 的 fixed_size_list<element: float>[dim]。"""
    return pa.list_(pa.field("element", pa.float32()), dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--copy-videos", action="store_true", help="拷贝而不是硬链接视频")
    ap.add_argument("--limit", type=int, default=0, help="只处理每份源的前 N 集(冒烟用)")
    ap.add_argument("--force", action="store_true", help="输出目录已存在时先删掉")
    args = ap.parse_args()
    OUT: Path = args.out

    if not E.selftest():
        fail("ee_repr selftest 未通过")
    print()

    if OUT.exists():
        if not args.force:
            fail(f"{OUT} 已存在。要重做加 --force(会先删)。")
        print(f"· --force: 删除已存在的 {OUT}")
        shutil.rmtree(OUT)

    tasks_a = [json.loads(l)["task"] for l in (SRC_A / "meta/tasks.jsonl").open()]
    tasks_b = [json.loads(l)["task"] for l in (SRC_B / "meta/tasks.jsonl").open()]
    if set(tasks_a) != set(tasks_b):
        fail("两份数据 task 文本集合不一致, 不能合并")
    remap = {i: tasks_a.index(t) for i, t in enumerate(tasks_b)}
    print(f"· task 重映射 B→A = {remap}")
    print(f"· 7 段 prompt 以 A(70集) 的索引为准\n")

    for sub in ("data/chunk-000", "meta"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for v in VIEWS:
        (OUT / "videos/chunk-000" / v).mkdir(parents=True, exist_ok=True)

    info_a = json.loads((SRC_A / "meta/info.json").read_text())
    n_a = info_a["total_episodes"]
    n_b = json.loads((SRC_B / "meta/info.json").read_text())["total_episodes"]
    if args.limit:
        n_a, n_b = min(n_a, args.limit), min(n_b, args.limit)

    out_ep = 0
    global_index = 0
    total_frames = 0
    ep_rows, ep_stats_rows = [], []
    max_fk_err = 0.0

    for src, n_eps, rm in ((SRC_A, n_a, None), (SRC_B, n_b, remap)):
        ep_meta = {json.loads(l)["episode_index"]: json.loads(l)
                   for l in (src / "meta/episodes.jsonl").open()}
        st_meta = {json.loads(l)["episode_index"]: json.loads(l)
                   for l in (src / "meta/episodes_stats.jsonl").open()}

        for e in range(n_eps):
            t = pq.read_table(src / f"data/chunk-000/episode_{e:06d}.parquet")
            d = t.to_pydict()
            n = t.num_rows

            s16 = np.asarray([np.asarray(x) for x in d[STATE_KEY]], dtype=np.float64)
            a16 = np.asarray([np.asarray(x) for x in d[ACTION_KEY]], dtype=np.float64)
            if s16.shape != (n, 16) or a16.shape != (n, 16):
                fail(f"ep{out_ep} 源 state/action 不是 16 维: {s16.shape}/{a16.shape}")
            if np.isnan(s16).any() or np.isnan(a16).any():
                fail(f"ep{out_ep} 源 state/action 含 NaN")

            s20 = E.joints16_to_ee20(s16)
            a20 = E.joints16_to_ee20(a16)
            if not np.isfinite(s20).all() or not np.isfinite(a20).all():
                fail(f"ep{out_ep} FK 结果含 NaN/Inf")
            # rot6d 正交性(FK 出来的一定正交, 这是防转换写错的哨兵)
            for arr in (s20, a20):
                R = E.rot6d_to_mat(arr[:, 3:9])
                err = np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)).max()
                R2 = E.rot6d_to_mat(arr[:, 13:19])
                err = max(err, np.abs(R2.transpose(0, 2, 1) @ R2 - np.eye(3)).max())
                max_fk_err = max(max_fk_err, float(err))
                if err > 1e-9:
                    fail(f"ep{out_ep} rot6d 非正交, 偏差 {err:.2e}")

            d[STATE_KEY] = s20.astype(np.float32).tolist()
            d[ACTION_KEY] = a20.astype(np.float32).tolist()
            d[J_STATE_KEY] = s16.astype(np.float32).tolist()
            d[J_ACTION_KEY] = a16.astype(np.float32).tolist()
            d["episode_index"] = [out_ep] * n
            d["index"] = list(range(global_index, global_index + n))
            if rm is not None:
                d["task_index"] = [rm[x] for x in d["task_index"]]

            # 铁律: 每一行 prompt_text 必须等于 tasks_a[task_index]
            bad = [i for i in range(n) if d["prompt_text"][i] != tasks_a[d["task_index"][i]]]
            if bad:
                fail(f"ep{out_ep} 有 {len(bad)} 行 prompt_text 与 task_index 对不上, "
                     f"首例 idx={bad[0]}")

            fields = []
            for f in t.schema:
                fields.append(pa.field(f.name, fixed_list(20)
                                       if f.name in (STATE_KEY, ACTION_KEY) else f.type))
            fields += [pa.field(J_STATE_KEY, fixed_list(16)),
                       pa.field(J_ACTION_KEY, fixed_list(16))]
            pq.write_table(pa.Table.from_pydict(d, schema=pa.schema(fields)),
                           OUT / f"data/chunk-000/episode_{out_ep:06d}.parquet")

            # 视频: 硬链接 + 逐路核帧数
            for v in VIEWS:
                s = src / f"videos/chunk-000/{v}/episode_{e:06d}.mp4"
                if not s.exists():
                    fail(f"缺视频 {s}")
                nf = nframes(s)
                if nf != n:
                    fail(f"ep{out_ep} {v} 帧数 {nf} ≠ parquet 行数 {n} ({s})")
                dst = OUT / "videos/chunk-000" / v / f"episode_{out_ep:06d}.mp4"
                if args.copy_videos:
                    shutil.copy2(s, dst)
                else:
                    try:
                        os.link(s, dst)
                    except OSError:
                        shutil.copy2(s, dst)

            em = dict(ep_meta[e]); em["episode_index"] = out_ep
            if em.get("length") != n:
                fail(f"ep{out_ep} episodes.jsonl length={em.get('length')} ≠ {n}")
            ep_texts = set(em.get("tasks") or [])
            row_texts = {tasks_a[i] for i in d["task_index"]}
            if ep_texts and not row_texts <= ep_texts:
                fail(f"ep{out_ep} 逐行 task_index 译出的文本不在 episodes.jsonl tasks 里\n"
                     f"   jsonl={sorted(ep_texts)}\n   rows ={sorted(row_texts)}")
            ep_rows.append(em)

            sm = dict(st_meta[e]); sm["episode_index"] = out_ep
            st = {k: v for k, v in sm["stats"].items()
                  if k != "observation.depth_images.third_view"}
            st[STATE_KEY] = col_stats(s20, n)
            st[ACTION_KEY] = col_stats(a20, n)
            st[J_STATE_KEY] = col_stats(s16, n)
            st[J_ACTION_KEY] = col_stats(a16, n)
            sm["stats"] = st
            ep_stats_rows.append(sm)

            total_frames += n
            global_index += n
            out_ep += 1
            print(f"  ep{out_ep-1:03d} ← {src.name[:38]:38s}/ep{e:02d}  {n:5d}帧", flush=True)

    with (OUT / "meta/episodes.jsonl").open("w") as f:
        for r in ep_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "meta/episodes_stats.jsonl").open("w") as f:
        for r in ep_stats_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "meta/tasks.jsonl").open("w") as f:
        for i, t in enumerate(tasks_a):
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")

    info = json.loads(json.dumps(info_a))  # 深拷贝
    info["features"].pop("observation.depth_images.third_view", None)
    info["features"][STATE_KEY] = {"dtype": "float32", "shape": [E.STATE_DIM],
                                  "names": E.STATE_NAMES}
    # action 落盘的是**绝对**末端位姿(与 state 同布局); 相对动作由训练管线现算。
    info["features"][ACTION_KEY] = {"dtype": "float32", "shape": [E.STATE_DIM],
                                    "names": E.STATE_NAMES}
    info["features"][J_STATE_KEY] = {"dtype": "float32", "shape": [16], "names": JOINT16_NAMES}
    info["features"][J_ACTION_KEY] = {"dtype": "float32", "shape": [16], "names": JOINT16_NAMES}
    # third_view 两份源都是去人手后的 H.264。腕部: A 是 av1、B 是 h264(混合) ——
    # lerobot 解码不读这个字段(靠 torchcodec/pyav 自探), 这里只作元数据; verify 会
    # 从两半各抽帧实解一次来兜底。
    info["features"]["observation.images.third_view"]["info"]["video.codec"] = "h264"
    info["robot_type"] = "dual_nero_box_packing_closing_ee_abs10x2"
    info["repo_id"] = f"local/{OUT.name}"
    info["total_episodes"] = out_ep
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks_a)
    info["total_videos"] = out_ep * len(VIEWS)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{out_ep}"}
    (OUT / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    print(f"\n· rot6d 正交性最大偏差 {max_fk_err:.2e} (阈 1e-9)")
    print(f"· 完成: {out_ep} 集 / {total_frames} 帧 / {len(tasks_a)} 任务 → {OUT}")
    print(f"· state/action = {E.STATE_DIM} 维绝对末端; 原关节存 {J_STATE_KEY}/{J_ACTION_KEY}")


if __name__ == "__main__":
    main()
