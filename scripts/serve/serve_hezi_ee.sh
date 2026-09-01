#!/usr/bin/env bash
# hezi 封盖【末端位姿版】推理服务 (pi05_nero_hezi_closing_ee_v1)。
# 输入 20 维绝对末端位姿, 输出 20 维绝对末端位姿(EEAbsoluteActions 已还原); 客户端接 IK。
#   GPU=0 PORT=8026 STEP=19999 bash serve_hezi_ee.sh
# 停: kill $(cat <LOG_DIR>/serve_hezi_ee.pid)
#
# ⚠ 必须用 openpi_supp 这个 fork —— pi05_nero_hezi_closing_ee_v1 和 nero_ee_policy
#   只注册在这个仓库里, <USER_B> 的主仓没有。
# ⚠ 起服务前预检目标卡空闲显存, 不足就拒绝(别挤死同卡在跑的东西)。
set -uo pipefail
REPO=<OPENPI_REPO>
PY=<OPENPI_VENV>/bin/python
CFG=pi05_nero_hezi_closing_ee_v1
CKPT_ROOT=<CKPT_ROOT_EE>/$CFG/hezi_ee_run1
GPU="${GPU:-0}"; PORT="${PORT:-8026}"; STEP="${STEP:-19999}"
MEMFRAC="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
NEED_FREE_MB="${NEED_FREE_MB:-12000}"
DIR="$CKPT_ROOT/$STEP"
mkdir -p <LOG_DIR>
LOG=<LOG_DIR>/serve_hezi_ee_${STEP}_$(date +%Y%m%d_%H%M%S).log

[ -d "$DIR/params" ] || { echo "✘ ckpt 不存在: $DIR"; echo "  可选: $(ls $CKPT_ROOT 2>/dev/null | tr '\n' ' ')"; exit 1; }
ss -ltn 2>/dev/null | grep -qw "$PORT" && { echo "✘ 端口 $PORT 已占用"; exit 1; }

read -r TOTAL USED < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
  | sed -n "$((GPU+1))p" | awk -F', ' '{print $1, $2}')
FREE=$((TOTAL - USED))
echo "GPU$GPU: 已用 ${USED}MiB / 共 ${TOTAL}MiB → 空闲 ${FREE}MiB (需要 ≥${NEED_FREE_MB})"
[ "$FREE" -lt "$NEED_FREE_MB" ] && { echo "✘ 空闲显存不足, 拒绝启动。换 GPU=<别的卡> 或等占卡的退出。"; exit 1; }

cd "$REPO"
echo "起服务: config=$CFG step=$STEP GPU=$GPU port=$PORT memfrac=$MEMFRAC"
echo "ckpt=$DIR"; echo "log=$LOG"
CUDA_VISIBLE_DEVICES="$GPU" \
XLA_PYTHON_CLIENT_MEM_FRACTION="$MEMFRAC" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TMPDIR=<TMP_DIR> \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src" \
setsid nohup "$PY" scripts/serve_policy.py --port "$PORT" \
  policy:checkpoint --policy.config="$CFG" --policy.dir="$DIR" \
  < /dev/null > "$LOG" 2>&1 &
PID=$!
echo "$PID" > <LOG_DIR>/serve_hezi_ee.pid
echo "$LOG"  > <LOG_DIR>/serve_hezi_ee.logpath
echo "pid=$PID"
