#!/usr/bin/env python3
"""NERO 正运动学：7 个关节角 → 末端位姿。纯 numpy，不依赖 Isaac / ROS。

  python3 fk_nero.py                       # 自测：零位下的末端位姿
  python3 fk_nero.py 50.28 56.2 41.59 75.22 -131.46 -37.19 75.9    # 给 7 个角(度)

  from fk_nero import fk
  T = fk(q_rad, side="right")              # 4x4，**世界系**（含机器人底座位姿）
  T = fk(q_rad, side="right", base=None)   # 4x4，机器人基座系
  T = fk(q_rad, side="right", tcp=0.105)   # 末端取夹持中心而不是法兰

约定（都是这套仿真里在用的，换环境要核对）
  · 末端 link = {side}_gripper_base（法兰面）。两指的**夹持中心**还在它 +z
    方向 105mm 处，抓取用的是夹持中心，所以传 tcp=0.105。
  · 机器人底座在世界系：x=0, y=-0.5822, z=0, yaw=90°（实测：距桌前缘 13cm）。
    换场景要改 BASE_POSE，或者调用时传 base=你的 4x4。
  · 关节角单位弧度，顺序 joint1..joint7。
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).with_name("nero_dual.urdf")
BASE_POSE = dict(x=0.0, y=-0.5822, z=0.0, yaw_deg=90.0)
# 沿法兰 +z 的几个特征点（都在两指对称轴上，和开度无关，已由 URDF 实测）：
TCP_FINGERTIP = 0.138            # **指尖最末端**（手指网格在法兰系占 0.0615~0.138）
TCP_GRIP_CENTER = 0.105          # 夹持中心：指面中段，抓取管线一直用的控制点
TCP_FINGER_ROOT = 0.0615         # 指根


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) or 1.0)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _origin_T(o):
    T = np.eye(4)
    if o is None:
        return T
    T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
    T[:3, :3] = _rpy(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
    return T


def base_T(pose=None):
    p = pose or BASE_POSE
    y = math.radians(p["yaw_deg"])
    T = np.eye(4)
    T[:3, :3] = [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0],
                 [0, 0, 1]]
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


_ROOT = ET.parse(URDF).getroot()
_BY_CHILD = {j.find("child").get("link"): j for j in _ROOT.findall("joint")}


def fk(q, side="right", link=None, base="default", tcp=0.0):
    """q: 7 个关节角(弧度)。返回 4x4。

    base="default" 用 BASE_POSE；base=None 留在基座系；也可直接传 4x4。
    tcp>0 时在末端 z 方向再前进这么多（取夹持中心传 0.105）。
    """
    link = link or f"{side}_gripper_base"
    qmap = {f"{side}_joint{i + 1}": float(q[i]) for i in range(7)}
    chain, cur = [], link
    while cur in _BY_CHILD:
        chain.append(_BY_CHILD[cur])
        cur = _BY_CHILD[cur].find("parent").get("link")
    T = np.eye(4)
    for j in reversed(chain):
        Tj = _origin_T(j.find("origin"))
        ax = j.find("axis")
        if ax is not None and j.get("type") in ("revolute", "continuous"):
            Rq = np.eye(4)
            Rq[:3, :3] = _axis_rot([float(v) for v in ax.get("xyz").split()],
                                   qmap.get(j.get("name"), 0.0))
            Tj = Tj @ Rq
        elif ax is not None and j.get("type") == "prismatic":
            Tq = np.eye(4)
            Tq[:3, 3] = (np.array([float(v) for v in ax.get("xyz").split()])
                         * qmap.get(j.get("name"), 0.0))
            Tj = Tj @ Tq
        T = T @ Tj
    if tcp:
        off = np.eye(4)
        off[2, 3] = tcp
        T = T @ off
    if base is None:
        return T
    return (base_T() if base == "default" else np.asarray(base, float)) @ T


def limits(side="right"):
    """→ (7,2) 弧度上下限，顺序 joint1..7。"""
    out = []
    for i in range(1, 8):
        j = next(x for x in _ROOT.findall("joint")
                 if x.get("name") == f"{side}_joint{i}")
        l = j.find("limit")
        out.append([float(l.get("lower")), float(l.get("upper"))])
    return np.array(out)


if __name__ == "__main__":
    if len(sys.argv) == 8:
        q = np.radians([float(v) for v in sys.argv[1:8]])
    else:
        q = np.radians([-26.0, 61.4, 103.5, 115.2, -146.8, -22.4, 32.1])
        print("· 未给角度，用 v6 零位（实采起始均值）")
    print(f"· 关节角(度) {np.round(np.degrees(q), 2)}")
    for side in ("right", "left"):
        T = fk(q, side=side)
        print(f"  {side:5s} 法兰     {np.round(T[:3, 3], 4)}")
        print(f"        指尖     {np.round(fk(q, side=side, tcp=TCP_FINGERTIP)[:3, 3], 4)}"
              f"   ← 两指对称轴最末端")
        print(f"        夹持中心 {np.round(fk(q, side=side, tcp=TCP_GRIP_CENTER)[:3, 3], 4)}")
        print(f"        R=\n{np.round(T[:3, :3], 3)}")
    print(f"· 关节限位(度)\n{np.round(np.degrees(limits()), 1)}")
