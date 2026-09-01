#!/usr/bin/env bash
# 右臂抓盒装箱【末端位姿版 v1b】部署服务。
# ★ 必须用 openpi_supp fork(含 EERightAbsoluteActions / nero_ee_right_policy), 主仓没有。
#
# v1b = 补训版: v1 从 pi05_base 训 20000 步(loss 0.0841→0.0077),
#       v1b 在 v1 的 19999 存档上再训 20000 步(lr 重启 5e-5), 最终 loss 0.0049。
# 存档: 5000 / 10000 / 15000 / 19999
#
#   GPU=0 PORT=8026 STEP=19999 bash serve_rightbox_ee.sh
# 停: kill $(cat <LOG_DIR>/serve_rightbox_ee.pid)
set -uo pipefail
FORK=<OPENPI_REPO>          # ★ fork, 不是主仓
REPO=<OPENPI_REPO>        # 借它的 scripts/ 与 client 包
PY=<OPENPI_VENV>/bin/python
CFG=pi05_nero_right_box_pick_ee_v1b
CKPT_ROOT=<CKPT_ROOT_EE>/pi05_nero_right_box_pick_ee_v1b/rbp_ee_v1b_run1
GPU="${GPU:-0}"; PORT="${PORT:-8031}"; STEP="${STEP:-19999}"
DIR="$CKPT_ROOT/$STEP"
mkdir -p <LOG_DIR>
LOG=<LOG_DIR>/serve_rightbox_ee_${STEP}_$(date +%Y%m%d_%H%M%S).log
[ -d "$DIR/params" ] || { echo "✘ ckpt 不存在: $DIR (可选: $(ls $CKPT_ROOT 2>/dev/null | tr '\n' ' '))"; exit 1; }
if ss -ltn 2>/dev/null | grep -qw "$PORT"; then echo "✘ 端口 $PORT 已被占用, 先停旧服务"; exit 1; fi
read -r TOTAL USED < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
  -i "$GPU" | awk -F', ' '{print $1, $2}')
FREE=$((TOTAL - USED))
echo "GPU$GPU: 已用 ${USED}MiB / ${TOTAL}MiB → 空闲 ${FREE}MiB"
[ "$FREE" -ge 8000 ] || { echo "✘ 空闲显存不足 8G, 拒绝启动(换 GPU= 或先停别的)"; exit 1; }
echo "起服务: config=$CFG  step=$STEP  GPU=$GPU  port=$PORT"
echo "ckpt=$DIR"; echo "fork=$FORK"; echo "log=$LOG"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONUNBUFFERED=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
PYTHONPATH="$FORK/src:$REPO/packages/openpi-client/src" \
setsid nohup "$PY" scripts/serve_policy.py --port "$PORT" \
  policy:checkpoint \
  --policy.config="$CFG" \
  --policy.dir="$DIR" \
  > "$LOG" 2>&1 &
echo $! > <LOG_DIR>/serve_rightbox_ee.pid
echo "$LOG" > <LOG_DIR>/serve_rightbox_ee.logpath
echo "pid=$(cat <LOG_DIR>/serve_rightbox_ee.pid)"
echo
echo "客户端: python3 ~/项目/nero/deploy/run_pi05_rollout_rightbox_ee.py --port $PORT"
