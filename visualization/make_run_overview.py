#!/usr/bin/env python3
"""把一次 wholeprocess 串跑的四段视频拼成一条总览片, 并按倍速播放。

为什么单独写一个而不是敲 ffmpeg
--------------------------------
这台机器**没有 ffmpeg/ffprobe 命令行**(lerobot 是通过 PyAV 编码的), 所以拼接和
变速都走 PyAV。顺带也就能在帧上直接画阶段标签和进度, 比纯 concat 好看。

倍速怎么实现
------------
不丢帧, 只把输出帧率乘上倍数(15Hz → 45Hz)。这样 3 倍速下每一帧都还在, 动作细节
不会被抽掉; 如果改成"每 3 帧取 1 帧"虽然文件更小, 但快速动作会跳。

用法::

    python3 make_run_overview.py run_20260827_184854
    python3 make_run_overview.py run_20260827_184854 --speed 5
    python3 make_run_overview.py run_20260827_184854 --camera left_wrist
    python3 make_run_overview.py run_20260827_184854 --grid      # 三路相机并排
    python3 make_run_overview.py /绝对/路径/到/run_xxx -o /tmp/out.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NERO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = NERO_ROOT / "lerobot_data" / "Rlinf" / "data" / "wholeprocess"

STAGES = [("e0_rightbox", "E0  右臂抓盒"), ("e1_leftbox", "E1  左臂抓盒"),
          ("e2_hezi34", "E2  stage34 封盖"), ("e3_stage56", "E3  stage56 合页")]
CAMS = ["third_view", "left_wrist", "right_wrist"]

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001, S112
                continue
    return ImageFont.load_default()


def vid_path(root: Path, tag: str, cam: str) -> Path | None:
    p = root / tag / "videos" / "chunk-000" / f"observation.images.{cam}" / "episode_000000.mp4"
    return p if p.exists() else None


def decode(path: Path):
    """逐帧产出 RGB ndarray。"""
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="rgb24")


def count_frames(path: Path) -> int:
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))


def annotate(img: Image.Image, label: str, sub: str, font, font_s) -> None:
    """左上角画阶段名, 左下角画进度。带半透明底条, 免得白背景上看不清。"""
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0, 150))
    d.text((10, 5), label, font=font, fill=(255, 255, 255, 255))
    if sub:
        d.rectangle([0, H - 26, W, H], fill=(0, 0, 0, 150))
        d.text((10, H - 23), sub, font=font_s, fill=(200, 220, 255, 255))


def main() -> None:
    ap = argparse.ArgumentParser(description="把一次串跑的四段视频拼成总览片并加速")
    ap.add_argument("run", help="运行 ID(如 run_20260827_184854)或绝对路径")
    ap.add_argument("--speed", type=float, default=3.0, help="倍速, 默认 3")
    ap.add_argument("--camera", default="third_view", choices=CAMS, help="用哪路相机")
    ap.add_argument("--grid", action="store_true", help="三路相机并排(缺的补黑)")
    ap.add_argument("-o", "--out", default=None, help="输出路径")
    ap.add_argument("--crf", type=int, default=26, help="画质, 越小越清晰(18~30)")
    ap.add_argument("--no-label", action="store_true", help="不画阶段标签")
    args = ap.parse_args()

    root = Path(args.run)
    if not root.is_absolute():
        root = DATA_ROOT / args.run
    if not root.is_dir():
        sys.exit(f"✘ 找不到: {root}")

    cams = CAMS if args.grid else [args.camera]
    plan = []          # [(tag, 中文名, {cam: path or None}, 帧数)]
    for tag, zh in STAGES:
        paths = {c: vid_path(root, tag, c) for c in cams}
        have = [p for p in paths.values() if p]
        if not have:
            print(f"  · {tag}: 没有 {'/'.join(cams)} 视频, 跳过")
            continue
        n = min(count_frames(p) for p in have)
        plan.append((tag, zh, paths, n))
        miss = [c for c, p in paths.items() if p is None]
        print(f"  ✓ {tag:<14} {n:5d} 帧 / {n/15:6.1f}s"
              + (f"   (缺 {', '.join(miss)}, 补黑)" if miss else ""))
    if not plan:
        sys.exit("✘ 一个视频都没找到")

    total = sum(n for *_, n in plan)
    out_fps = 15.0 * args.speed
    out = Path(args.out) if args.out else root / (
        f"overview_{'grid' if args.grid else args.camera}_{args.speed:g}x.mp4")

    W = 640 * len(cams)
    H = 480
    print(f"\n  合计 {total} 帧 / {total/15:.1f}s  →  {args.speed:g}x = "
          f"{total/out_fps:.1f}s @ {out_fps:g}fps   {W}x{H}")
    print(f"  输出 {out}\n")

    font = load_font(22)
    font_s = load_font(15)
    oc = av.open(str(out), "w")
    st = oc.add_stream("libx264", rate=int(round(out_fps)))
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": str(args.crf), "preset": "medium"}

    done = 0
    for tag, zh, paths, n in plan:
        gens = {c: (decode(p) if p else None) for c, p in paths.items()}
        black = np.zeros((480, 640, 3), np.uint8)
        for i in range(n):
            tiles = []
            for c in cams:
                g = gens[c]
                try:
                    tiles.append(next(g) if g else black)
                except StopIteration:
                    tiles.append(black)
            arr = np.hstack(tiles) if len(tiles) > 1 else tiles[0]
            img = Image.fromarray(arr)
            if not args.no_label:
                annotate(img, f"{zh}   ·   {args.speed:g}x",
                         f"{i/15:5.1f}s / {n/15:.1f}s        总进度 "
                         f"{(done+i)/total*100:4.1f}%   ({'+'.join(cams)})",
                         font, font_s)
            fr = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
            for pkt in st.encode(fr):
                oc.mux(pkt)
            if (i + 1) % 200 == 0:
                print(f"    {tag}  {i+1}/{n}", end="\r", flush=True)
        done += n
        print(f"    {tag}  {n}/{n}  ✓          ", flush=True)

    for pkt in st.encode():
        oc.mux(pkt)
    oc.close()
    mb = out.stat().st_size / 1e6
    print(f"\n✓ 完成  {out}   {mb:.1f} MB   {total/out_fps:.1f}s")


if __name__ == "__main__":
    main()
