#!/usr/bin/env bash
# 双臂体检 —— "机械臂网页又打不开了" 时先跑这个, 它会告诉你到底是网络问题还是臂本身掉了。
#
# 三条独立通道各查一遍, 因为它们坏的方式不一样:
#   CAN   部署脚本真正用的通道。左臂=can1, 右臂=can0
#   网页  <ROBOT_WEB_IP> (有线 <ROBOT_LAN_IP>/24)。⚠ 必须绕代理, 否则永远 502
#   驱动  读 ctrl_mode/arm_status, 只读, 不发任何运动指令
#
# 判读:
#   CAN 静默 + 网页不通  → 臂断电/控制器挂了。软件修不了, 去断电重启那条臂
#   CAN 正常 + 网页不通  → 真是网络问题(代理没绕开 / 网线 / IP)
#   都正常但仍不能动     → 模式或抱闸问题, 跑 fix_left_brake.py --arm left
set -u
cd "$(dirname "$0")"
W=http://<ROBOT_WEB_IP>

echo "════ CAN(部署脚本用的通道) ════"
for pair in "left:can1" "right:can0"; do
  side=${pair%%:*}; c=${pair##*:}
  n=$(timeout 3 candump -n 100 "$c" 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then echo "  ✓ $side ($c)  3 秒收到 $n 帧 —— 臂在推送"
  else echo "  ✘ $side ($c)  一帧没有 —— 该臂没上电 / CAN 线掉了 / 控制器挂了"; fi
done

echo
echo "════ 网页 $W ════"
echo "  (代理必须绕开: no_proxy 里没有 <ROBOT_LAN_IP>/24, 走代理会得到 502)"
code=$(timeout 6 curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "$W/" 2>/dev/null)
if [ "$code" = "200" ]; then echo "  ✓ HTTP 200 —— 网页活着"
else
  echo "  ✘ HTTP ${code:-无响应}"
  printf "    ARP: "; ip neigh show <ROBOT_WEB_IP> 2>/dev/null | grep -q . \
    && ip neigh show <ROBOT_WEB_IP> || echo "无表项"
  printf "    物理链路 carrier="; cat /sys/class/net/enx6c1ff7c17dff/carrier 2>/dev/null || echo "?"
fi

echo
echo "════ 驱动层只读探测 ════"
timeout 40 python3 fix_left_brake.py --arm left 2>&1 | sed 's/^/  /'

echo
echo "════ 浏览器打不开时 ════"
echo "  Chrome/Edge 读的是 GNOME 代理设置, 其 ignore-hosts 已含 10.0.0.0/8, 本该直连。"
echo "  若仍不通, 用命令行确认是不是臂的问题:  curl -I --noproxy '*' $W/"
