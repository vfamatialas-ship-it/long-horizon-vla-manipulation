# Rollout media — 待补

每个占位说明**要放什么、多大、怎么生成**。GIF 建议 ≤10 MB,mp4 ≤25 MB。

| 文件名 | 内容 | 用在哪 |
|---|---|---|
| `full_chain_3x.gif` | 四段完整串跑, 3 倍速, 第三视角 | README 第一屏 ★最重要 |
| `success_full.gif` | 一条成功的完整 rollout | README Demo 表格左 |
| `failure_grasp_order.gif` | 抓取顺序错误的失败案例 | README Demo 表格右 / Failure Case 1 |
| `switch_timeline.png` | 子任务切换时刻 vs 真值的时间轴 | Experiments |

## 怎么生成

```bash
# 四段拼接 + 倍速(项目里已有工具)
python3 visualization/make_run_overview.py <RUN_ID> --speed 3
# → run_<时间戳>/overview_third_view_3x.mp4

# 转 GIF(控制体积)
ffmpeg -i overview_third_view_3x.mp4 -vf "fps=10,scale=640:-1" -loop 0 full_chain_3x.gif
```
