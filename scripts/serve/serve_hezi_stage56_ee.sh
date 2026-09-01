#!/usr/bin/env bash
# hezi 封盖【stage5+6 · 末端位姿版】部署服务。
# ★ 必须用 openpi_supp fork(含 EEAbsoluteActions / nero_ee_policy), 主仓没有这些变换。
#
# 与 serve_ee.sh(7 段封盖版)只差 CFG / CKPT_ROOT / 默认端口:
#   7 段版:   pi05_nero_hezi_closing_ee_v1  / hezi_ee_run1   / 19999 / 8026
#   stage56: pi05_nero_hezi_stage56_ee_v1  / stage56_ee_run1 / 14999 / 8029
# 端口错开 → 两个服务可以同时挂着, 方便对比。
#
#   GPU=0 PORT=8029 STEP=14999 bash serve_hezi_stage56_ee.sh
# 停: kill $(cat <LOG_DIR>/serve_stage56_ee.pid)
set -uo pipefail
FORK=<OPENPI_REPO>          # ★ fork, 不是主仓
REPO=<OPENPI_REPO>        # 借它的 scripts/ 与 client 包
PY=<OPENPI_VENV>/bin/python
CFG=pi05_nero_hezi_stage56_ee_v1
CKPT_ROOT=<CKPT_ROOT_EE>/pi05_nero_hezi_stage56_ee_v1/stage56_ee_run1
GPU="${GPU:-0}"; PORT="${PORT:-8029}"; STEP="${STEP:-14999}"   # 存档只剩 10000/12000/14999
DIR="$CKPT_ROOT/$STEP"
mkdir -p <LOG_DIR>
LOG=<LOG_DIR>/serve_stage56_ee_${STEP}_$(date +%Y%m%d_%H%M%S).log
[ -d "$DIR/params" ] || { echo "✘ ckpt 不存在: $DIR (可选: $(ls $CKPT_ROOT 2>/dev/null | tr '\n' ' '))"; exit 1; }
if ss -ltn 2>/dev/null | grep -qw "$PORT"; then echo "✘ 端口 $PORT 已被占用"; exit 1; fi
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
echo $! > <LOG_DIR>/serve_stage56_ee.pid
echo "$LOG" > <LOG_DIR>/serve_stage56_ee.logpath
echo "pid=$(cat <LOG_DIR>/serve_stage56_ee.pid)"
echo
echo "客户端: python3 ~/项目/nero/deploy/run_pi05_rollout_hezi_stage56_ee.py --port $PORT"
