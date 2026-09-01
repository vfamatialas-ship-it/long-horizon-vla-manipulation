#!/usr/bin/env bash
# hezi 封盖【关节版 · 7 段细化元动作】部署服务。
#
# ⚠ 这是**关节版**, 不是末端位姿版 —— 与 serve_hezi_ee.sh 的根本区别:
#     本脚本   : 16 维双臂关节 delta 动作直接下发 move_joints(), 客户端**不接 IK**
#     hezi_ee  : 20 维绝对末端位姿进出, 客户端必须接 IK 解成关节角
#   所以本服务用**主仓**即可(不需要 openpi_supp fork 的 EEAbsoluteActions 变换)。
#
# 模型 pi05_nero_hezi_closing_refined_merged_v2 / hezi_closing_refined_merged_run1
#   三路 RGB 全部进模型(third + left_wrist + right_wrist), 没有零填槽。
#   7 段 stage prompt, prompt_from_task=True —— 发错 prompt 不报错但明显掉效果。
#
#   GPU=3 PORT=8023 STEP=19999 bash serve_hezi_merged_v2.sh
# 停: kill $(cat <LOG_DIR>/serve_hezi_merged_v2.pid)
set -uo pipefail
REPO=<OPENPI_REPO>
PY=<OPENPI_VENV>/bin/python
CFG=pi05_nero_hezi_closing_refined_merged_v2
CKPT_ROOT=<CKPT_ROOT>/pi05_nero_hezi_closing_refined_merged_v2/hezi_closing_refined_merged_run1
GPU="${GPU:-3}"; PORT="${PORT:-8023}"; STEP="${STEP:-19999}"   # 存档: 5000 10000 15000 19999
DIR="$CKPT_ROOT/$STEP"
mkdir -p <LOG_DIR>
LOG=<LOG_DIR>/serve_hezi_merged_v2_${STEP}_$(date +%Y%m%d_%H%M%S).log
[ -d "$DIR/params" ] || { echo "✘ ckpt 不存在: $DIR (可选: $(ls $CKPT_ROOT 2>/dev/null | tr '\n' ' '))"; exit 1; }
if ss -ltn 2>/dev/null | grep -qw "$PORT"; then echo "✘ 端口 $PORT 已被占用"; exit 1; fi
read -r TOTAL USED < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
  -i "$GPU" | awk -F', ' '{print $1, $2}')
FREE=$((TOTAL - USED))
echo "GPU$GPU: 已用 ${USED}MiB / ${TOTAL}MiB → 空闲 ${FREE}MiB"
# ★ 32.6G 的 5090 上 MEM_FRACTION=0.2 只有 6.5G, 装不下 6.0G 权重会 RESOURCE_EXHAUSTED。
#   0.45 在 32G 卡上给到 ~14.7G, 48G 卡上 ~22G, 两种卡都够。
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
  policy:checkpoint \
  --policy.config="$CFG" \
  --policy.dir="$DIR" \
  > "$LOG" 2>&1 &
echo $! > <LOG_DIR>/serve_hezi_merged_v2.pid
echo "$LOG" > <LOG_DIR>/serve_hezi_merged_v2.logpath
echo "pid=$(cat <LOG_DIR>/serve_hezi_merged_v2.pid)"
echo
echo "客户端: python3 ~/项目/nero/deploy/run_pi05_rollout_hezi_dual.py --port $PORT"
