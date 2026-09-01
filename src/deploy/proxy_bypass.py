#!/usr/bin/env python3
"""让局域网地址绕开系统代理。**每个要连推理服务的脚本都必须在联网前调一次。**

为什么需要这个
--------------
这台机器全局挂着代理(``http_proxy``/``all_proxy`` → 127.0.0.1:7897), 而
``no_proxy`` 里只有 ``localhost,127.0.0.1,::1``。于是任何走局域网的连接都会被
塞进代理, 表现成各种看不出是代理问题的错误。今天一天就撞了三次:

  * 机械臂网页 <ROBOT_WEB_IP>      → HTTP 502
  * finalize 里的 HuggingFace  → ValueError: Unknown scheme for proxy URL 'socks://...'
  * 推理服务 <POLICY_SERVER_IP>:8026 → InvalidMessage: did not receive a valid HTTP response

三个都不长得像"代理问题", 所以每次都要重新查一遍。

为什么只能写字面 IP
-------------------
实测(``websockets`` 16.1.1):

    no_proxy=<LAN_IP>/16     ✘ 仍然走代理 —— **它不解析 CIDR**
    no_proxy=<POLICY_SERVER_IP>     ✓ 0.02s 连上

所以不能图省事写个网段了事, 必须把**具体那个 IP** 加进去。GNOME 的
ignore-hosts 里那条 ``<LAN_IP>/16`` 只对浏览器有效, 对 Python 一点用没有。
"""
from __future__ import annotations

import os

_LOCAL = ("localhost", "127.0.0.1", "::1")


def bypass_proxy(*hosts: str, verbose: bool = False) -> None:
    """把 hosts 逐个加进 no_proxy/NO_PROXY(字面量, 不是网段)。

    只加不删 —— 不动 http_proxy 等变量, 走外网的东西照常用代理。
    """
    want = [h for h in (*_LOCAL, *hosts) if h]
    for var in ("no_proxy", "NO_PROXY"):
        cur = [x.strip() for x in os.environ.get(var, "").split(",") if x.strip()]
        for h in want:
            if h not in cur:
                cur.append(h)
        os.environ[var] = ",".join(cur)
    # all_proxy 在这台机器上是 'socks://...' —— httpx 只认 socks5://, 拿到这个
    # 直接抛 ValueError。局域网既然已经绕开, 这个畸形值留着只会继续坑别的库。
    for var in ("all_proxy", "ALL_PROXY"):
        v = os.environ.get(var, "")
        if v.startswith("socks://"):
            os.environ[var] = v.replace("socks://", "socks5://", 1)
    if verbose:
        print(f"  (代理绕行: no_proxy += {', '.join(hosts)})", flush=True)
