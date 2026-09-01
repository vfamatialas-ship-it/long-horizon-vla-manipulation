#!/usr/bin/env bash
# hezi 封盖【stage5+6 · 末端位姿版 · 100 集版】部署服务。
# ★ 必须用 openpi_supp fork(含 EEAbsoluteActions / nero_ee_policy), 主仓没有这些变换。
#
# 与 69 集版 serve_stage56_flap_ee.sh 只差 CFG / CKPT_ROOT / 默认端口:
#   69 集版 : pi05_nero_stage56_flap_closing_ee_v2   / stage56_ee_run1 / 19999 / 8030  loss 0.0224
#   100集版 : pi05_nero_stage56_flap_closing_ee_v100 / run1            / 19999 / 8032  loss 0.0267
# 端口错开 → 两个同时挂着可直接对比。100 集版 loss 略高是正常的:
# 数据多 45%、场景更杂, 训练 loss 高一点但泛化通常更好, 别拿 loss 直接判优劣。
#
#   GPU=2 PORT=8032 STEP=19999 bash serve_stage56_flap_ee_v100.sh
# 停: kill $(cat <LOG_DIR>/serve_stage56_flap_ee_v100.pid)
set -uo pipefail
FORK=<OPENPI_REPO>          # ★ fork, 不是主仓
REPO=<OPENPI_REPO>        # 借它的 scripts/ 与 client 包
PY=<OPENPI_VENV>/bin/python
CFG=pi05_nero_stage56_flap_closing_ee_v100
CKPT_ROOT=<CKPT_ROOT_EE>/pi05_nero_stage56_flap_closing_ee_v100/run1
GPU="${GPU:-2}"; PORT="${PORT:-8032}"; STEP="${STEP:-19999}"   # 存档: 5000 10000 15000 19999
DIR="$CKPT_ROOT/$STEP"
mkdir -p <LOG_DIR>
LOG=<LOG_DIR>/serve_stage56_flap_ee_v100_${STEP}_$(date +%Y%m%d_%H%M%S).log
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
echo $! > <LOG_DIR>/serve_stage56_flap_ee_v100.pid
echo "$LOG" > <LOG_DIR>/serve_stage56_flap_ee_v100.logpath
echo "pid=$(cat <LOG_DIR>/serve_stage56_flap_ee_v100.pid)"
echo
echo "客户端: python3 ~/项目/nero/deploy/run_pi05_rollout_stage56_flap_ee_v100.py --port $PORT"
