#!/usr/bin/env bash
# 右臂抓盒装箱【关节版 · 补训版 v2b】部署服务。
#
# ⚠ 这是**关节版**, 用**主仓**即可 —— 不需要 openpi_supp fork
#   (fork 里的 EEAbsoluteActions / nero_ee_policy 是末端位姿版才要的)。
#   v2b 也**只在主仓注册**, 拿 fork 起会报 config 找不到。
#
# 模型 pi05_nero_right_box_pick_v2b / right_box_pick_v2b_run1
#   v2 从 pi05_base 训 20000 步(loss 0.0637) → v2b 在 v2 的 19999 上再训 20000 步,
#   最终 loss 0.0451。数据与 v2 同一份 local/nero_right_box_pick_v2(100 集/41034 帧)。
#   16 维双臂关节 delta, 两子任务(抓 → 放), prompt_from_task=True。
#
#   GPU=0 PORT=8033 STEP=19999 bash serve_rightbox_v2b.sh
# 停: kill $(cat <LOG_DIR>/serve_rightbox_v2b.pid)
set -uo pipefail
REPO=<OPENPI_REPO>
PY=<OPENPI_VENV>/bin/python
CFG=pi05_nero_right_box_pick_v2b
CKPT_ROOT=<CKPT_ROOT>/pi05_nero_right_box_pick_v2b/right_box_pick_v2b_run1
GPU="${GPU:-0}"; PORT="${PORT:-8033}"; STEP="${STEP:-19999}"   # 存档: 5000 10000 15000 19999
DIR="$CKPT_ROOT/$STEP"
mkdir -p <LOG_DIR>
LOG=<LOG_DIR>/serve_rightbox_v2b_${STEP}_$(date +%Y%m%d_%H%M%S).log
[ -d "$DIR/params" ] || { echo "✘ ckpt 不存在: $DIR (可选: $(ls $CKPT_ROOT 2>/dev/null | tr '\n' ' '))"; exit 1; }
if ss -ltn 2>/dev/null | grep -qw "$PORT"; then echo "✘ 端口 $PORT 已被占用"; exit 1; fi
read -r TOTAL USED < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
  -i "$GPU" | awk -F', ' '{print $1, $2}')
FREE=$((TOTAL - USED))
echo "GPU$GPU: 已用 ${USED}MiB / ${TOTAL}MiB → 空闲 ${FREE}MiB"
# ★ 32.6G 的 5090 上 MEM_FRACTION=0.2 只有 6.5G, 装不下 6.0G 权重会 RESOURCE_EXHAUSTED。
[ "$FREE" -ge 8000 ] || { echo "✘ 空闲显存不足 8G, 拒绝启动(换 GPU= 或先停别的)"; exit 1; }
echo "起服务: config=$CFG  step=$STEP  GPU=$GPU  port=$PORT"
echo "ckpt=$DIR"; echo "log=$LOG"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONUNBUFFERED=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src" \
setsid nohup "$PY" scripts/serve_policy.py --port "$PORT" \
  policy:checkpoint --policy.config="$CFG" --policy.dir="$DIR" \
  > "$LOG" 2>&1 &
echo $! > <LOG_DIR>/serve_rightbox_v2b.pid
echo "$LOG" > <LOG_DIR>/serve_rightbox_v2b.logpath
echo "pid=$(cat <LOG_DIR>/serve_rightbox_v2b.pid)"
echo
echo "客户端: python3 ~/项目/nero/deploy/run_pi05_rollout_rightbox_v2b.py --port $PORT"
