#!/usr/bin/env bash
# 末端位姿版训练的**唯一**启动入口。把「没全准备好就别碰卡上的进程」这条纪律写进代码:
#
#   1. norm stats 存在 + 已兜底(.prefloor 存在)     ← 不满足直接退出, 不动 GPU
#   2. verify --pipeline 必须 ALL PASS               ← 不满足直接退出, 不动 GPU
#   3. 只有 1&2 都过, 才停 GPU3 上的 hezi 关节版 serve
#   4. 启动训练
#   5. 立刻挂上守候进程: 训练一结束(成功或失败)就把 serve 挂回去
#
# 用法: bash go_ee_train.sh [exp_name]
#       DRY=1 bash go_ee_train.sh      # 只做 1&2 的检查, 绝不动 GPU、绝不启动训练
set -uo pipefail

EXP="${1:-hezi_ee_run1}"
GPU="${GPU:-3}"
DRY="${DRY:-0}"
CONFIG=pi05_nero_hezi_closing_ee_v1
EE=<EE_PACK>
PY=<OPENPI_VENV>/bin/python
REPO=<OPENPI_REPO>
NS=<ASSETS_ROOT>/$CONFIG/local/nero_hezi_closing_ee_v1/norm_stats.json

step(){ echo; echo "════ $* ════"; }
die(){ echo "✘ $*"; echo "→ 未满足前置条件, **没有动 GPU$GPU 上的任何进程**, 训练未启动。"; exit 1; }

step "1/5 检查 norm stats 与兜底"
[ -f "$NS" ] || die "norm stats 不存在: $NS (先跑 scripts/compute_norm_stats_ee.sh)"
[ -f "$NS.prefloor" ] || die "没看到 $NS.prefloor —— 说明夹爪常数维还没兜底(跑 tools/floor_norm_stats_ee.py)"
"$PY" - "$NS" <<'EOF' || die "norm stats 维度/兜底检查未过"
import json, sys
# 注意: openpi 的 normalize.save 只写 mean/std/q01/q99, **没有 min/max**。
# 所以这里用 q01/q99 估一个 |z| 下界(真正的 max|z| 由 verify --pipeline 在真批次上断言)。
d = json.load(open(sys.argv[1]))["norm_stats"]
s, a = d["state"], d["actions"]
assert len(s["std"]) == 20, f"state 应 20 维, 实得 {len(s['std'])}"
assert len(a["std"]) == 14, f"actions 应 14 维, 实得 {len(a['std'])}"
for k, v in (("state", s), ("actions", a)):
    mu, sd, q01, q99 = v["mean"], v["std"], v["q01"], v["q99"]
    assert min(sd) >= 0.01, f"{k} 仍有 std<0.01 的维(兜底没生效?): min={min(sd):.2e}"
    z99 = max(max(abs(q01[i]-mu[i]), abs(q99[i]-mu[i])) / (sd[i]+1e-6)
              for i in range(len(sd)))
    assert z99 < 20, f"{k} 连 q01/q99 处的 |z|={z99:.2f} 就 ≥20 了"
    print(f"  ✓ {k}: {len(sd)} 维, 最小 std={min(sd):.4f}, q01/q99 处 |z|={z99:.2f}")
EOF

step "2/5 跑 verify --pipeline(CPU, 必须 ALL PASS)"
cd "$EE/scripts"
VLOG=$EE/logs/verify_pipeline_$(date +%Y%m%d_%H%M%S).log
HF_LEROBOT_HOME=<DATA_ROOT> \
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
TMPDIR=<TMP_DIR> \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src" \
"$PY" verify_ee_dataset.py --pipeline 2>&1 | tee "$VLOG"
grep -q "^ALL PASS" "$VLOG" || die "verify 未 ALL PASS(见 $VLOG)"
echo "  ✓ verify ALL PASS → $VLOG"

if [ "$DRY" = 1 ]; then
    echo; echo "DRY=1: 前置条件全过, 但按要求**没有动 GPU$GPU**、没有启动训练。"
    echo "去掉 DRY=1 再跑即可开训。"; exit 0
fi

step "3/5 停 GPU$GPU 上的 hezi 关节版 serve(前置条件已全过, 现在才动它)"
SPID=$(cat <LOG_DIR>/serve_hezi.pid 2>/dev/null || true)
if [ -n "${SPID:-}" ] && kill -0 "$SPID" 2>/dev/null; then
    echo "  kill $SPID (serve_hezi, port 8023)"
    kill "$SPID"                       # 只按 PID 杀, 不用 pkill -f 模糊匹配
    for i in $(seq 1 40); do
        kill -0 "$SPID" 2>/dev/null || break
        sleep 3
    done
    kill -0 "$SPID" 2>/dev/null && { echo "  15s 未退, 发 SIGKILL"; kill -9 "$SPID"; sleep 5; }
else
    echo "  serve_hezi.pid 里的进程已不在, 跳过"
fi
for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sed -n "$((GPU+1))p" | awk -F', ' '{print $2}')
    echo "  GPU$GPU 已用 ${used} MiB"
    [ "${used:-99999}" -lt 6000 ] && break
    sleep 10
done
[ "${used:-99999}" -lt 6000 ] || die "GPU$GPU 显存没降到 6G 以下(还剩 ${used} MiB), 不硬上"

step "4/5 启动训练"
GPU="$GPU" bash "$EE/scripts/train_hezi_ee.sh" "$EXP"

step "5/5 挂守候: 训练结束或失败后立刻恢复 serve"
GPU="$GPU" setsid nohup bash "$EE/scripts/after_ee_train.sh" > /dev/null 2>&1 &
echo "  守候已起, 日志 $EE/logs/after_ee_train.log"
echo
echo "盯 loss(≤15 分钟内必须见到第一个 loss 数值):"
echo "  tail -f \$(cat $EE/logs/hezi_ee.logpath)"
