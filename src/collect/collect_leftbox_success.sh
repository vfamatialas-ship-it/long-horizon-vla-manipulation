#!/usr/bin/env bash
# RLinf 左臂抓盒【成功】对照采集入口(两子任务: 抓取 → 放置)。
# CAN 按适配器序列号、相机按 USB 序列号自动解析 —— 插拔 arm CAN/相机都不会乱。
#
# 用法:
#   ./collect_leftbox_success.sh                        # 默认不进拖动示教(右臂主臂模式由外部设置)
#   DRAG=1 ./collect_leftbox_success.sh                 # 让脚本进拖动示教(双臂零力, 示教采集)
#   ./collect_leftbox_success.sh --dry-run-state-machine  # 只测状态机, 不碰硬件
#   ./collect_leftbox_success.sh --left-can can1 --right-can can0   # 序列号解析失败时手动指定
set -euo pipefail
export OPENCV_VIDEOIO_V4L_SELECTTIMEOUT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${RLINF_ROOT}/configs/leftbox_success.yaml}"

DRAG_ARGS=(--no-enable --no-drag-teach)
if [[ "${DRAG:-0}" == 1 ]]; then
  DRAG_ARGS=(--drag-teach --confirm-drag-teach-risk)
fi

echo "=== RLinf 左臂抓盒【成功】采集(两子任务) ==="
echo "配置: ${CONFIG}   拖动示教: $([[ "${DRAG:-0}" == 1 ]] && echo 是 || echo 否)"
echo "CAN/相机按序列号自动解析(见启动打印)。"

exec python3 "${SCRIPT_DIR}/collect_leftbox_success.py" \
  --config "$CONFIG" \
  "${DRAG_ARGS[@]}" \
  "$@"
