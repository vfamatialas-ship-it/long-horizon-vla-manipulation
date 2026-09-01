#!/usr/bin/env bash
# pi0.5 专项微调: hezi 封盖「末端位姿版」(70集+30集合并 = 100集/87649帧/15Hz)。
#   输入 20 维绝对末端位姿(每臂 3 位置 + rot6D + 1 爪宽), 输出 14 维相对位姿。
#   从 pi05_base 自身权重初始化(动作空间与关节版不同, 不续训)。20000 步 / 每 5000 存。
#
# 用法: GPU=<卡号> ./train_hezi_ee.sh [exp_name] [额外 train.py 参数...]
# 例:   GPU=2 ./train_hezi_ee.sh hezi_ee_run1
#       SMOKE=1 GPU=2 ./train_hezi_ee.sh          # 30 步冒烟测显存, 落 smoke 目录
#       RESUME=1 GPU=2 ./train_hezi_ee.sh hezi_ee_run1   # 断点续
#
# ⚠️ 启动前必须已完成: build_ee_dataset → compute_norm_stats → floor_norm_stats_ee
#    → verify_ee_dataset.py --pipeline 见 ALL PASS(尤其 max|a|/max|s| < 20)。
set -euo pipefail

GPU="${GPU:?必须指定 GPU=<卡号> (48G 卡; LoRA batch32 约 40G+)}"
EXP="${1:-hezi_ee_run1}"; shift || true
CONFIG="${CONFIG:-pi05_nero_hezi_closing_ee_v1}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"

REPO=<OPENPI_REPO>                 # fork, 不碰 <USER_B> 在跑的仓库
PY=<OPENPI_VENV>/bin/python
export HF_LEROBOT_HOME=<DATA_ROOT>      # 数据集在 nvme2t, 不在 ~/.cache
LOG_DIR=<EE_PACK>/logs
CKPT_BASE=<CKPT_ROOT_EE>/$CONFIG
mkdir -p "$LOG_DIR"

ARGS=(--exp-name="$EXP" --no-wandb-enabled)
if [ "$SMOKE" = 1 ]; then
    EXP=smoke
    ARGS=(--exp-name=smoke --no-wandb-enabled --num-train-steps=30 --save-interval=1000 --overwrite)
    LOG="$LOG_DIR/hezi_ee_smoke_$(date +%Y%m%d_%H%M%S).log"
else
    LOG="$LOG_DIR/${EXP}_$(date +%Y%m%d_%H%M%S).log"
    if [ "$RESUME" = 1 ]; then
        ARGS+=(--resume)
    elif [ -d "$CKPT_BASE/$EXP" ]; then
        echo "✘ $CKPT_BASE/$EXP 已存在。要续训用 RESUME=1, 要重来自己先删。"; exit 1
    fi
fi

echo "config   : $CONFIG"
echo "exp-name : $EXP"
echo "GPU      : $GPU"
echo "数据集   : HF_LEROBOT_HOME=$HF_LEROBOT_HOME / local/nero_hezi_closing_ee_v1"
echo "ckpt     : $CKPT_BASE/$EXP"
echo "log      : $LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    | sed -n "$((GPU+1))p" | sed 's/^/占用前 GPU/'

cd "$REPO"
# PYTHONUNBUFFERED=1: loss 行即时落盘(stdbuf 对 python3 无效, 别用)
setsid nohup env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${MEMFRAC:-0.85}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR=<WORKSPACE>/jax_cache \
    TMPDIR=<TMP_DIR> \
    HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src" \
    "$PY" scripts/train.py "$CONFIG" "${ARGS[@]}" "$@" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/hezi_ee.pid"
echo "$LOG"  > "$LOG_DIR/hezi_ee.logpath"
echo "已后台启动(setsid) pid=$PID"
echo "看进度: tail -f $LOG      (≤15 分钟内必须看到第一个 loss 数值, 否则不算健康)"
