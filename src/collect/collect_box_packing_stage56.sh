#!/usr/bin/env bash
# Nero 双臂 —— Stage5/6 合页封盖(7 个微动作子任务)LeRobot v2.1 采集入口。
# 关键: CAN 按【序列号】解析(插拔改 can 号也不会左右反), 相机 左2/右4/三6。
#
# 7 段(全部靠 Space 推进, 每段一次):
#   51 右臂举起跨过合页贴近箱子上表面
#   52 右夹爪张开抵住前后两个合页
#   53 左臂从左侧托起左合页至近垂直
#   54 左臂将左合页合上
#   55 右臂收回
#   61 右臂从右侧托起右合页至近垂直
#   62 右臂将右合页合上
#
# 用法:
#   ./collect_box_packing_stage56.sh
#   DRAG=1 ./collect_box_packing_stage56.sh     # 让脚本进拖动示教(默认不进, 靠外部/Web 设)
#   LEFT_CAM=/dev/video2 RIGHT_CAM=/dev/video4 THIRD_CAM=/dev/video6 ./...  # 相机号变了就覆盖
#   ./collect_box_packing_stage56.sh --dry-run-state-machine       # 透传给采集脚本的参数
set -euo pipefail
export OPENCV_VIDEOIO_V4L_SELECTTIMEOUT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEZI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${HEZI_ROOT}/configs/box_packing_stage56_task.yaml}"

# ---- 左右臂 CAN 适配器序列号(固定不变, 用它解析当前 canX) ----
LEFT_CAN_SERIAL="<LEFT_CAN_SERIAL>"   # 左臂(接网线那只)
RIGHT_CAN_SERIAL="<RIGHT_CAN_SERIAL>"  # 右臂

# ---- 相机: 左2 / 右4 / 三6(与 config 一致; 改了插口就用 env 覆盖) ----
LEFT_CAM="${LEFT_CAM:-/dev/video2}"    # 左腕
RIGHT_CAM="${RIGHT_CAM:-/dev/video4}"  # 右腕
THIRD_CAM="${THIRD_CAM:-/dev/video6}"  # 第三视角

# ---- RGB 曝光(v4l2, 按角色套到对应设备; 可用 env 覆盖) ----
LEFT_RGB_EXPOSURE="${LEFT_RGB_EXPOSURE:-200}";  LEFT_RGB_GAIN="${LEFT_RGB_GAIN:-8}";  LEFT_RGB_GAMMA="${LEFT_RGB_GAMMA:-110}"
RIGHT_RGB_EXPOSURE="${RIGHT_RGB_EXPOSURE:-240}"; RIGHT_RGB_GAIN="${RIGHT_RGB_GAIN:-8}"; RIGHT_RGB_GAMMA="${RIGHT_RGB_GAMMA:-130}"
THIRD_RGB_EXPOSURE="${THIRD_RGB_EXPOSURE:-240}"; THIRD_RGB_GAIN="${THIRD_RGB_GAIN:-8}"; THIRD_RGB_GAMMA="${THIRD_RGB_GAMMA:-140}"   # 顶视: 160/120 实测过暗(p99只到150,亮部空40%),8/24 调亮
APPLY_V4L2="${APPLY_V4L2:-1}"
USE_DEPTH="${USE_DEPTH:-0}"   # 默认【不采深度】; 想采深度(需 Orbbec)才 USE_DEPTH=1

# 按序列号找当前 canX
resolve_can() {
  local want="$1" c sn
  for c in $(ls /sys/class/net/ 2>/dev/null | grep -E '^can[0-9]+$'); do
    sn=$(udevadm info -q property -p "$(readlink -f "/sys/class/net/$c")" 2>/dev/null \
      | grep -m1 ID_SERIAL_SHORT | cut -d= -f2)
    [[ "$sn" == "$want" ]] && { echo "$c"; return 0; }
  done
  return 1
}

apply_camera_controls() {
  local tag="$1" device="$2" exposure="$3" gain="$4" gamma="$5"
  v4l2-ctl -d "$device" \
    -c auto_exposure=1 -c exposure_dynamic_framerate=0 \
    -c "exposure_time_absolute=${exposure}" -c "gain=${gain}" -c "gamma=${gamma}" >/dev/null
  echo "  ✓ ${tag}: exposure=${exposure} gain=${gain} gamma=${gamma}"
}

# mock/dry 模式下不强制真硬件
MOCK=0
for a in "$@"; do
  [[ "$a" == "--mock" || "$a" == "--mock-cameras" || "$a" == "--dry-run-state-machine" || "$a" == "--dry-run-save-episode" || "$a" == "--check-depth-only" ]] && MOCK=1
done

LEFT_CAN="$(resolve_can "$LEFT_CAN_SERIAL" || true)"
RIGHT_CAN="$(resolve_can "$RIGHT_CAN_SERIAL" || true)"
if [[ "$MOCK" -eq 1 ]]; then LEFT_CAN="${LEFT_CAN:-can0}"; RIGHT_CAN="${RIGHT_CAN:-can1}"; fi

echo "========= Nero Stage5/6 合页封盖采集 (7 子任务) ========="
echo "配置: ${CONFIG}"
echo "CAN(按序列号解析):"
echo "    左臂 serial=${LEFT_CAN_SERIAL} -> ${LEFT_CAN:-未找到}"
echo "    右臂 serial=${RIGHT_CAN_SERIAL} -> ${RIGHT_CAN:-未找到}"
echo "相机: 左腕=${LEFT_CAM}  右腕=${RIGHT_CAM}  第三视角=${THIRD_CAM}"
echo "拖动示教: $([[ "${DRAG:-0}" == 1 ]] && echo '脚本进 --drag-teach' || echo '不进(外部/Web 设置; --no-drag-teach)')"
echo "深度: $([[ "${USE_DEPTH}" == 1 ]] && echo '采(Orbbec)' || echo '【不采】(--no-use-depth)')"
echo "========================================================"

# 上机前硬件自检(mock 跳过)
if [[ "$MOCK" -eq 0 ]]; then
  miss=0
  for p in "$LEFT_CAM" "$RIGHT_CAM" "$THIRD_CAM"; do
    [[ -e "$p" ]] || { echo "  ✗ RGB 相机不存在: $p"; miss=1; }
  done
  for tc in "左臂:$LEFT_CAN" "右臂:$RIGHT_CAN"; do
    tag="${tc%%:*}"; c="${tc##*:}"
    if [[ -z "$c" ]]; then
      echo "  ✗ $tag CAN 适配器未按序列号找到(适配器没插/没上电?)"; miss=1
    elif ! ip -details link show "$c" 2>/dev/null | grep -q "state UP"; then
      echo "  ✗ $tag $c 未 UP → 先执行: sudo ip link set $c up type can bitrate 1000000"; miss=1
    fi
  done
  [[ "$miss" -ne 0 ]] && { echo "自检未通过, 已中止(未启动采集)。"; exit 1; }

  if [[ "$APPLY_V4L2" == 1 ]]; then
    echo "应用相机曝光:"
    apply_camera_controls "left_wrist"  "$LEFT_CAM"  "$LEFT_RGB_EXPOSURE"  "$LEFT_RGB_GAIN"  "$LEFT_RGB_GAMMA"
    apply_camera_controls "right_wrist" "$RIGHT_CAM" "$RIGHT_RGB_EXPOSURE" "$RIGHT_RGB_GAIN" "$RIGHT_RGB_GAMMA"
    apply_camera_controls "third_view"  "$THIRD_CAM" "$THIRD_RGB_EXPOSURE" "$THIRD_RGB_GAIN" "$THIRD_RGB_GAMMA"
  fi
  for p in "$LEFT_CAM" "$RIGHT_CAM" "$THIRD_CAM"; do
    if fuser "$p" &>/dev/null; then echo "⚠ $p 被其它进程占用(先关相机预览/其它采集)。"; exit 1; fi
  done
fi

# 拖动示教开关: DRAG=1 才让脚本自己进 leader 模式(否则不碰电机模式, 靠外部设置)
DRAG_ARGS=(--no-enable)
if [[ "${DRAG:-0}" == 1 ]]; then
  DRAG_ARGS+=(--drag-teach --confirm-drag-teach-risk)
else
  DRAG_ARGS+=(--no-drag-teach)
fi

DEPTH_ARGS=()
[[ "$USE_DEPTH" == 1 ]] || DEPTH_ARGS+=(--no-use-depth)   # 默认关深度

exec python3 "${SCRIPT_DIR}/collect_box_packing_stage56.py" \
  --config "$CONFIG" \
  --left-can "$LEFT_CAN" --right-can "$RIGHT_CAN" \
  --left-wrist-camera "$LEFT_CAM" --right-wrist-camera "$RIGHT_CAM" --third-camera "$THIRD_CAM" \
  "${DRAG_ARGS[@]}" "${DEPTH_ARGS[@]}" \
  "$@"
