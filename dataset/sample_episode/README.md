# Sample episode — 待补

放**一条**完整 episode, 让人不下载 10GB 也能看懂 schema。建议:

```
sample_episode/
├── data/chunk-000/episode_000000.parquet     全部列, 一条
├── videos/.../episode_000000.mp4             三路, 可降到 320×240
└── meta/{info,episodes,tasks}.jsonl
```

从任一数据集导出一条即可, 注意**先脱敏 meta 里的绝对路径**。
