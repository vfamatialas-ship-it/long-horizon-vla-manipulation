#!/usr/bin/env python3
"""子任务自动切换 —— **客户端侧的瘦封装**,给四个 rollout 客户端 import。

跑在系统 python3 里(只用 numpy + 标准库),真正的模型在 switcher_service.py 那个
独立进程里(venv,torch cu128)。见 switcher_service.py 顶部关于「为什么拆进程」。

用法::

    from switcher_client import SwitcherClient

    sw = SwitcherClient(expert=2)          # 起不来就抛异常, 不会静默降级
    sw.reset()
    ...
    r = sw.step(frames, state)             # frames = cameras.read_frames() 的返回
    if r["switched"]: ...                  # 该切下一个子任务了
    if r["finished"]: ...                  # 本专家全部子任务完成

⚠ step() 必须跟**相机帧**走(15Hz),不能跟策略推理走 —— 见 service 里的说明。
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VENV_PY = HERE / "switcher_pkg" / ".venv" / "bin" / "python"
SERVICE = HERE / "switcher_service.py"
SOCK = "/tmp/nero_switcher.sock"

# 流水线第一拍还没有结果可用, 用一个「什么都不发生」的占位
_IDLE = {"switched": False, "finished": False, "subtask": 1, "steps": 0,
         "min_steps": 0, "too_early": True, "signal": 0.0, "ema": 0.0,
         "tau": 0.0, "hits": 0, "pending": None, "stalled": False,
         "frozen": False, "forced": False, "prompt": ""}

# rollout 客户端 read_frames() 的键 → 服务端认的视角名
FRAME_KEY = {
    "third_view": ("observation.images.third", "observation.images.third_view"),
    "right_wrist": ("observation.images.right_wrist",),
    "left_wrist": ("observation.images.left_wrist",),
}


def _resize224(img: np.ndarray) -> np.ndarray:
    """缩到 224×224。优先用 cv2(客户端一定有), 没有就退回最近邻切片。

    在客户端做而不是服务端: 网络上少传 6 倍数据, 而且切换器内部本来也要缩到 224,
    总计算量不变。
    """
    if img.shape[0] == 224 and img.shape[1] == 224:
        return np.ascontiguousarray(img, dtype=np.uint8)
    try:
        import cv2  # noqa: PLC0415
        return np.ascontiguousarray(
            cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA), dtype=np.uint8)
    except Exception:  # noqa: BLE001
        h, w = img.shape[:2]
        yi = (np.arange(224) * h // 224).clip(0, h - 1)
        xi = (np.arange(224) * w // 224).clip(0, w - 1)
        return np.ascontiguousarray(img[yi][:, xi], dtype=np.uint8)


class SwitcherClient:
    """连本地切换服务。服务没起就自动拉起(用 venv 的 python)。"""

    def __init__(self, expert: int, sock: str = SOCK, autostart: bool = True,
                 device: str = "cuda", timeout: float = 5.0, boot_timeout: float = 180.0,
                 trace: str | None = None, pipelined: bool = True):
        self.expert, self.sock_path, self.timeout = int(expert), sock, timeout
        self._proc = None
        self._conn = None
        self._pending = False
        self.pipelined = bool(pipelined)
        if not self._try_connect():
            if not autostart:
                raise RuntimeError(f"切换服务没在跑, 且 autostart=False: {sock}")
            self._spawn(device, boot_timeout)
        r = self._rpc({"cmd": "ping"})
        if not r.get("cuda"):
            print("  ⚠ 切换服务跑在 CPU 上 —— SigLIP2 约 2.8Hz, 撑不住 15Hz, "
                  "切换准确率会大幅下降", flush=True)
        try:
            self._log = open(trace or "/tmp/nero_switch_trace.jsonl", "a",
                             encoding="utf-8")
            self._log.write(json.dumps(
                {"_run": time.strftime("%Y-%m-%d %H:%M:%S"), "expert": self.expert},
                ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            self._log = None
        self._n = 0
        info = self._rpc({"cmd": "reset", "expert": self.expert, "sub": 1})
        self.n_sub = int(info["n_sub"])
        print(f"  ✓ 自动切换: expert={self.expert} (E{self.expert+1})  "
              f"{self.n_sub} 个子任务  设备={r.get('device')}", flush=True)

    # ── 连接 ────────────────────────────────────────────────────────────────
    def _try_connect(self) -> bool:
        if not os.path.exists(self.sock_path):
            return False
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(self.timeout)
            c.connect(self.sock_path)
            self._conn = c
            return True
        except OSError:
            return False

    def _spawn(self, device: str, boot_timeout: float) -> None:
        if not VENV_PY.exists():
            raise RuntimeError(
                f"找不到切换器的 venv: {VENV_PY}\n"
                f"   它需要 torch cu128 + transformers(系统 torch 是 CPU 版, 不能用)。")
        print(f"  · 拉起切换服务({device}) …", flush=True)
        log = open("/tmp/nero_switcher.log", "ab", buffering=0)
        self._proc = subprocess.Popen(
            [str(VENV_PY), str(SERVICE), "--sock", self.sock_path,
             "--device", device, "--preload", str(self.expert)],
            stdout=log, stderr=log, cwd=str(HERE))
        t0 = time.time()
        while time.time() - t0 < boot_timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"切换服务启动即退出(码 {self._proc.returncode})。"
                    f"看日志: tail -40 /tmp/nero_switcher.log")
            if self._try_connect():
                print(f"  · 服务就绪 ({time.time()-t0:.1f}s)", flush=True)
                return
            time.sleep(0.3)
        raise RuntimeError(f"切换服务 {boot_timeout:.0f}s 内没起来。"
                           f"看日志: tail -40 /tmp/nero_switcher.log")

    def _send(self, req: dict) -> None:
        import msgpack  # noqa: PLC0415
        b = msgpack.packb(req, use_bin_type=True)
        self._conn.sendall(struct.pack(">I", len(b)) + b)

    def _recv_pending(self) -> dict:
        self._pending = False
        return self._recv_body()

    def _rpc(self, req: dict) -> dict:
        self._send(req)
        return self._recv_body()

    def _recv_body(self) -> dict:
        import msgpack  # noqa: PLC0415
        head = b""
        while len(head) < 4:
            c = self._conn.recv(4 - len(head))
            if not c:
                raise RuntimeError("切换服务断开(看 /tmp/nero_switcher.log)")
            head += c
        n = struct.unpack(">I", head)[0]
        buf = bytearray()
        while len(buf) < n:
            c = self._conn.recv(min(65536, n - len(buf)))
            if not c:
                raise RuntimeError("切换服务断开(看 /tmp/nero_switcher.log)")
            buf += c
        r = msgpack.unpackb(bytes(buf), raw=False)
        if isinstance(r, dict) and r.get("ok") is False and "error" in r:
            raise RuntimeError(f"切换服务报错: {r['error']}")
        return r

    # ── 对外 ────────────────────────────────────────────────────────────────
    def reset(self, sub: int = 1) -> dict:
        # ⚠ 在途请求必须先收掉, 否则 reset 的响应会跟它错位, 后面每次读到的都是上一条
        if self._pending:
            self._recv_pending()
        return self._rpc({"cmd": "reset", "expert": self.expert, "sub": int(sub)})

    def step(self, frames: dict, state) -> dict:
        """frames = rollout 客户端 cameras.read_frames() 的返回(键名自动适配)。

        每步的判据都写进 /tmp/nero_switch_trace.jsonl —— 事后能直接看出「为什么切」
        「为什么不切」, 不用靠猜。一步一行, 一次 rollout 几千行, 文件几 MB。
        """
        imgs = {}
        for view, cands in FRAME_KEY.items():
            for k in cands:
                if k in frames and frames[k] is not None:
                    imgs[view] = _resize224(np.asarray(frames[k])).tobytes()
                    break
        req = {"cmd": "step", "expert": self.expert, "images": imgs,
               "shape": [224, 224, 3],
               "state": [float(v) for v in np.asarray(state).ravel()]}
        if not self.pipelined:
            r = self._rpc(req)
            self._trace(r)
            return r

        # ── 流水线: 先收上一步的结果, 再把这一步发出去 ──────────────────────
        # 同步调用要占掉控制回路 ~36ms, 而 15Hz 的预算只有 66.7ms —— 光切换器就吃掉
        # 一半多。流水线把「等 GPU 算完」挪出临界路径: 本拍只做一次 send, 下一拍再 recv。
        # 代价是判据晚一拍(~66ms)。相比延迟补偿 D 本来就是 3~11 步, 这一拍可以忽略。
        # ⚠ 步数仍是 1:1 —— 每个动作步对应一次 step(), 步数量纲的标定没有被破坏。
        prev = self._recv_pending() if self._pending else _IDLE
        self._send(req)
        self._pending = True
        self._trace(prev)
        return prev

    def _trace(self, r: dict) -> None:
        if self._log is None:
            return
        self._n += 1
        try:
            keep = ("subtask", "steps", "min_steps", "too_early", "signal", "ema",
                    "tau", "hits", "pending", "switched", "stalled", "frozen",
                    "forced", "finished")
            rec = {"i": self._n, "e": self.expert}
            rec.update({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items() if k in keep})
            self._log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if r.get("switched") or r.get("forced"):
                self._log.flush()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            if self._conn:
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        # 服务是共享的(四个阶段接力用同一个), 客户端退出不杀它 —— 省下每段
        # 重新加载 816M 视觉塔的几秒。要停用 stop_switcher_service()。


def stop_switcher_service(sock: str = SOCK) -> None:
    """停掉后台的切换服务(chain 整条链结束时调)。"""
    import signal  # noqa: PLC0415
    try:
        out = subprocess.run(["pgrep", "-f", "switcher_service.py"],
                             capture_output=True, text=True).stdout.split()
        for pid in out:
            os.kill(int(pid), signal.SIGTERM)
        if out:
            print(f"  · 已停切换服务 (pid {', '.join(out)})", flush=True)
    except Exception:  # noqa: BLE001
        pass
    if os.path.exists(sock):
        try:
            os.unlink(sock)
        except OSError:
            pass


if __name__ == "__main__":
    # 自检: 起服务 + 发几帧假数据, 量往返延迟
    import argparse
    ap = argparse.ArgumentParser(description="切换客户端自检")
    ap.add_argument("--expert", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    sw = SwitcherClient(expert=a.expert, device=a.device)
    rng = np.random.default_rng(0)
    fr = {"observation.images.third": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
          "observation.images.right_wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
          "observation.images.left_wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)}
    st = np.zeros(20, np.float32)
    for _ in range(5):
        sw.step(fr, st)
    sw.reset()
    ts = []
    for _ in range(a.n):
        t0 = time.perf_counter(); sw.step(fr, st); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    print(f"\n  端到端(含 resize + socket 往返)  {a.n} 步")
    print(f"    中位 {np.median(ts):.1f} ms  p95 {np.percentile(ts,95):.1f} ms  "
          f"最大 {ts.max():.1f} ms")
    print(f"    → 最高 {1000/np.median(ts):.1f} Hz   "
          f"{'✓ 撑得住 15Hz' if np.median(ts) < 66.7 else '✘ 撑不住 15Hz'}")
