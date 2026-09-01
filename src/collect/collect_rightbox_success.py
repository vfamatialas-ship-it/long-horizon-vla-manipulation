#!/usr/bin/env python3
"""RLinf 右臂抓盒【成功】对照采集 —— 两子任务: ①右臂抓取右侧区域的盒子 ②右臂将抓到的盒子放进打包盒的空处。

采集流程:
  Space 开始 → 子任务1 抓盒(抓到后 Space 推进)→ 子任务2 放盒(放好后 Space)→ S 保存成功 / F 保存失败(选类型)/ G 丢弃
CAN 按适配器序列号、相机按 USB 序列号自动解析(插拔不乱)。示教采集 action≡state, 15Hz。
用法: ./collect_rightbox_success.sh [--dry-run-state-machine | 其它透传参数]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_leftbox_common import run  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rightbox_success.yaml"

if __name__ == "__main__":
    run(DEFAULT_CONFIG)
