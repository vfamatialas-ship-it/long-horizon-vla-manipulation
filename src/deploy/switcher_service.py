#!/usr/bin/env python3
"""子任务自动切换 —— **服务进程**。跑在 switcher_pkg/.venv 里(torch cu128 + transformers)。

为什么要拆成服务而不是直接 import
----------------------------------
rollout 客户端跑的是系统 python3(装着 lerobot / cv2 / pyrealsense2),而切换器要
torch cu128 + transformers —— 本机系统 torch 是 **2.13.0+cpu**,换掉它会连累 lerobot
的视频编码。所以切换器单独一个 venv、单独一个进程,客户端通过 **Unix socket** 调它。

代价是每步一次本地 socket 往返。图像在客户端就 resize 到 224×224 再发
(3×224×224×3 = 450KB/步 @15Hz = 6.8MB/s,本地 socket 毫无压力),
比发原图 2.7MB/步 省 6 倍。resize 本来在切换器内部也要做,挪到客户端不增加总开销。

为什么必须 15Hz
---------------
τ / 延迟 D / 连续 K 帧 / EMA 全部是**步数量纲**,按数据集 15fps 标定的。
实测(见 部署包/README.md):按 5Hz 调用,切换准确率从 76% 崩到 4%,还有 9/73 集跑不完。
所以 step() 要跟**相机帧**走,不能跟策略推理走。

协议
----
每条消息 = 4 字节大端长度 + msgpack。请求:
    {"cmd": "reset",  "expert": int, "sub": int}
    {"cmd": "step",   "expert": int, "images": {view: bytes}, "shape": [h,w,3], "state": [float]}
    {"cmd": "ping"}
响应即 SubtaskSwitcher.step() 的返回 dict(numpy 标量已转 float)。

用法::

    # 手动起(调试用; 客户端会自动拉起)
    switcher_pkg/.venv/bin/python switcher_service.py

    # 自检: 不连机器人, 用随机帧量单步延迟
    switcher_pkg/.venv/bin/python switcher_service.py --benchmark
"""
from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE / "switcher_pkg"
SOCK = "/tmp/nero_switcher.sock"
VIEWS = ("third_view", "right_wrist", "left_wrist")


def _send(conn, obj):
    import msgpack
    b = msgpack.packb(obj, use_bin_type=True)
    conn.sendall(struct.pack(">I", len(b)) + b)


def _recv(conn):
    import msgpack
    head = b""
    while len(head) < 4:
        c = conn.recv(4 - len(head))
        if not c:
            return None
        head += c
    n = struct.unpack(">I", head)[0]
    buf = bytearray()
    while len(buf) < n:
        c = conn.recv(min(65536, n - len(buf)))
        if not c:
            return None
        buf += c
    return msgpack.unpackb(bytes(buf), raw=False)


class Runner:
    """按 expert 缓存 SubtaskSwitcher 实例 —— 视觉塔只加载一次(816M, 加载要几秒)。"""

    def __init__(self, device: str, verbose: bool):
        sys.path.insert(0, str(PKG))
        self.device, self.verbose = device, verbose
        self._sw = {}
        import torch
        self.torch = torch
        print(f"  torch {torch.__version__}  cuda={torch.version.cuda}  "
              f"设备={device}", flush=True)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("✘ 要求 cuda 但 torch.cuda.is_available()=False —— "
                             "这个 venv 装的可能是 CPU 版 torch")

    def get(self, expert: int):
        if expert not in self._sw:
            from switcher import SubtaskSwitcher  # noqa: PLC0415
            t0 = time.time()
            self._sw[expert] = SubtaskSwitcher(expert=expert, pkg=PKG,
                                               device=self.device, verbose=self.verbose)
            print(f"  ✓ 载入 expert={expert} (E{expert+1})  {time.time()-t0:.1f}s",
                  flush=True)
        return self._sw[expert]

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "device": self.device,
                    "cuda": bool(self.torch.cuda.is_available())}
        if cmd == "reset":
            sw = self.get(int(req["expert"]))
            sw.reset(int(req.get("sub", 1)))
            return {"ok": True, "subtask": sw.sub, "prompt": sw.prompt,
                    "n_sub": sw.n_sub}
        if cmd == "step":
            sw = self.get(int(req["expert"]))
            h, w, c = req["shape"]
            imgs = {v: np.frombuffer(req["images"][v], np.uint8).reshape(h, w, c)
                    for v in VIEWS if v in req["images"]}
            # 进度头按 n_view=3 训的, 少一路会在 switcher._forward 里抛 KeyError,
            # 客户端只看到 "KeyError: 'left_wrist'" 完全不知道该怎么办。这里挑明:
            # 单臂客户端默认不开另一侧腕部相机, 要跑 --auto-switch 必须把它插上。
            need = {"third_view": "third", "right_wrist": "right", "left_wrist": "left"}
            want = [k for k, v in need.items() if v in sw.views]
            miss = [k for k in want if k not in imgs]
            if miss:
                return {"ok": False, "error":
                        f"缺相机 {miss} —— 进度头需要 {want} 三路(它就是按三路训的)。"
                        f"单臂客户端默认不开另一侧腕部, 请插上该相机; "
                        f"或去掉 --auto-switch 用手动空格。"}
            r = sw.step(imgs, np.asarray(req["state"], np.float32))
            # msgpack 不认 numpy 标量 / None 以外的东西, 统一成 python 原生类型
            return {k: (float(v) if isinstance(v, (np.floating, np.integer))
                        else (bool(v) if isinstance(v, np.bool_) else v))
                    for k, v in r.items()}
        return {"ok": False, "error": f"未知命令 {cmd!r}"}


def benchmark(runner: Runner, expert: int, n: int = 40) -> None:
    """量单步延迟 —— 必须 < 66.7ms 才撑得住 15Hz。"""
    sw = runner.get(expert)
    rng = np.random.default_rng(0)
    imgs = {v: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for v in VIEWS}
    st = np.zeros(sw.state_dim, np.float32)
    for _ in range(5):                      # 预热(首次含 CUDA 上下文与 cudnn 选核)
        sw.step(imgs, st)
    sw.reset()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sw.step(imgs, st)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    print(f"\n  expert={expert}  {n} 步")
    print(f"    中位 {np.median(ts):.1f} ms   p95 {np.percentile(ts,95):.1f} ms   "
          f"最大 {ts.max():.1f} ms")
    print(f"    → 最高 {1000/np.median(ts):.1f} Hz   "
          f"{'✓ 撑得住 15Hz' if np.median(ts) < 66.7 else '✘ 撑不住 15Hz(需 <66.7ms)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="子任务自动切换服务")
    ap.add_argument("--sock", default=SOCK)
    ap.add_argument("--device", default="cuda",
                    help="cuda / cpu。⚠ CPU 上 SigLIP2 约 2.8Hz, 撑不住 15Hz")
    ap.add_argument("--benchmark", action="store_true", help="只量延迟, 不起服务")
    ap.add_argument("--expert", type=int, default=2, help="--benchmark 用哪个专家")
    ap.add_argument("--preload", default="", help="启动时预载哪些专家, 如 0,1,2,3")
    ap.add_argument("--quiet", action="store_true", help="不打切换日志")
    args = ap.parse_args()

    runner = Runner(args.device, verbose=not args.quiet)

    if args.benchmark:
        benchmark(runner, args.expert)
        return

    for e in [int(x) for x in args.preload.split(",") if x.strip()]:
        runner.get(e)

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.sock)
    os.chmod(args.sock, 0o666)
    srv.listen(4)
    print(f"  ✓ 切换服务就绪: {args.sock}", flush=True)

    try:
        while True:
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP if False else socket.SOL_SOCKET,
                            socket.SO_RCVBUF, 1 << 20)
            try:
                while True:
                    req = _recv(conn)
                    if req is None:
                        break
                    try:
                        _send(conn, runner.handle(req))
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        _send(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n  服务退出", flush=True)
    finally:
        srv.close()
        if os.path.exists(args.sock):
            os.unlink(args.sock)


if __name__ == "__main__":
    main()
