#!/usr/bin/env bash
# 守候:末端位姿版训练**一结束就立刻**把 GPU3 上的 hezi 关节版推理服务挂回去。
# 无论训练是正常跑完还是中途失败/被 kill,都恢复服务 —— 卡不能空着,服务不能长时间下线。
# 按被停掉时的原样恢复: config=pi05_nero_hezi_closing_refined_merged_v2 / GPU3 / port 8023 / step 19999。
# 不自动删 checkpoint(删是不可逆的),等人看过最终列表再决定。
#
# 用法(训练启动后立刻起): setsid nohup bash after_ee_train.sh > /dev/null 2>&1 &
set -uo pipefail
LOGD=<EE_PACK>/logs
W=$LOGD/after_ee_train.log
GPU="${GPU:-3}"; PORT="${PORT:-8023}"; STEP="${STEP:-19999}"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$W"; }

say "守候开始:等末端位姿版训练结束 → 恢复 GPU$GPU 上的 serve(port $PORT / step $STEP)"
# 等训练进程退出(不用 pkill -f 之类的模糊匹配去杀东西, 只做只读探测)
while pgrep -f "[t]rain.py pi05_nero_hezi_closing_ee_v1" > /dev/null; do sleep 60; done
say "训练进程已退出"

CK=<CKPT_ROOT_EE>/pi05_nero_hezi_closing_ee_v1
RUNS=$(ls "$CK" 2>/dev/null | tr '\n' ' ')
say "run 目录: ${RUNS:-（无）}"
for r in $RUNS; do
    say "  $r 已存 checkpoint: $(ls "$CK/$r" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
done
if ! ls "$CK"/*/19999 > /dev/null 2>&1; then
    say "⚠ 没看到 19999 —— 训练可能是中途挂的。仍然恢复服务, 但结果要先查训练日志。"
fi

# 等显存真正释放, 否则 serve 抢不到
for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sed -n "$((GPU+1))p" | awk -F', ' '{print $2}')
    [ "${used:-99999}" -lt 3000 ] && { say "GPU$GPU 已释放(${used} MiB)"; break; }
    sleep 15
done

if ss -ltn 2>/dev/null | grep -qw "$PORT"; then
    say "✘ 端口 $PORT 已被占用, 不重复起(可能有人手动挂回来了)"
else
    say "起 hezi 关节版服务: GPU$GPU / port $PORT / step $STEP"
    GPU="$GPU" PORT="$PORT" STEP="$STEP" bash <WORKSPACE>/serve_hezi.sh >> "$W" 2>&1
    sleep 60
    if ss -ltn 2>/dev/null | grep -qw "$PORT"; then
        say "✔ 服务已监听 $PORT"
    else
        say "⚠ 60s 后 $PORT 还没监听, 见 $(cat <LOG_DIR>/serve_hezi.logpath 2>/dev/null)"
    fi
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    | sed -n "$((GPU+1))p" >> "$W"
say "守候结束"
