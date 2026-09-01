#!/usr/bin/env bash
# 算末端位姿版 norm stats(CPU)。算完**必须**接 floor_norm_stats_ee.py 兜底夹爪常数维。
#
# 坑#1(踩过): compute_norm_stats 不锁卡时 JAX 会把 batch=32 分到所有可用 GPU 上,
#   可用卡数不整除 32 就报 "... divisible by 3 ... equal to 32"。一律 CPU 跑。
#   ~9 分钟量级。真正训练是单卡, 不受此影响, 别给训练也锁 CPU。
set -euo pipefail

CONFIG="${CONFIG:-pi05_nero_hezi_closing_ee_v1}"
REPO=<OPENPI_REPO>
PY=<OPENPI_VENV>/bin/python
ASSETS=<ASSETS_ROOT>/$CONFIG/local/nero_hezi_closing_ee_v1
LOG=<EE_PACK>/logs/norm_stats_$(date +%Y%m%d_%H%M%S).log

cd "$REPO"
echo "config=$CONFIG  →  $ASSETS"
echo "log=$LOG"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
HF_LEROBOT_HOME=<DATA_ROOT> \
TMPDIR=<TMP_DIR> \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src" \
"$PY" scripts/compute_norm_stats.py --config-name "$CONFIG" 2>&1 | tee "$LOG"

echo
echo "=== 兜底夹爪常数维(阈 0.01 → 0.05) ==="
"$PY" <EE_PACK>/tools/floor_norm_stats_ee.py "$ASSETS/norm_stats.json" 0.01 0.05
