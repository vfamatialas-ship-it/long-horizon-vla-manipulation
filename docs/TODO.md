# 上传前 TODO

分成 **① 现在就能传**、**② 传之前必须补**、**③ 之后慢慢加** 三档。

---

## ① 现在就能传(代码与文档已就绪)

两个仓库的代码、配置、README 都写好并脱敏了,`git init` 就能推。

- `vfamatialas-ship-it/` → Profile README
- `long-horizon-vla-manipulation/` → 主项目
- `custom-robot-arm-control/` → 控制层

---

## ② 传之前必须补(缺了会显得空)

### 视频与图片 —— 优先级最高

招聘者第一屏看的就是这个。位置已经占好,见各 `assets/*/README.md`(里面写了生成命令)。

| 文件 | 用途 | 怎么做 |
|---|---|---|
| `assets/rollouts/full_chain_3x.gif` | **README 第一屏** | `make_run_overview.py <RUN_ID> --speed 3` 再转 GIF |
| `assets/rollouts/success_full.gif` | 成功案例 | 挑一条完整成功的 run |
| `assets/rollouts/failure_grasp_order.gif` | 失败案例 | 挑一条抓取顺序错的 |
| `assets/images/system_diagram.png` | 架构图 | 画一张(README 里有 ASCII 版可参考) |
| `assets/images/hardware_setup.png` | 工位照 | 拍一张双臂 + 三相机 + 纸箱 |

> 现有的 rollout 数据在本地 `盒子数据整理_最终的总体数据/rollout数据/`,
> 挑素材从那里找。

### 一条 sample episode

`dataset/sample_episode/` —— 让人不下 10 GB 也能看懂数据长什么样。
**导出时先脱敏 `meta/` 里的绝对路径。**

### 两个待确认的数字

README 里这两处我按现有材料写的,你核对一下:

- **"~4 分钟一轮"** —— 从 rollout 数据推的,建议用真实计时替换
- **成功率** —— 目前只写了子任务切换的 96%,**没有写整任务成功率**,因为我手上没有
  足够的真机成功/失败统计。有的话补进 Key Results,这是招聘者最想看的一个数

---

## ③ 之后慢慢加

### Hugging Face

| 放什么 | 说明 |
|---|---|
| 4 个数据集 | best 版本即可,README 里的 "link TBD" 换成真链接 |
| 4 个 best checkpoint | **只传 19999 那一档**,中间档不传 |

### 可以再开的仓库

- **subtask-progress-head** —— 子任务自动切换单独成仓。它有完整的训练代码、零人工标注的
  标签生成、消融和留出集指标,本身就是个完整的小研究。现在寄居在主项目里有点埋没
- **robot-data-pipeline** —— 去人手管线(YOLO+SAM2 / 绿手套 chroma-key 两条路线)+
  关节⇄末端位姿转换。工程味重,能体现数据处理能力
- **记忆机制的否定结果** —— 你有完整的消融记录。**发表否定结果本身是加分项**,
  说明做事严谨,不是只报喜

### Repo 3 的 WIP 部分

重力补偿和 PICO 遥操作现在标着 🔶。上机验证过了就把状态改成 ✅ 并补数据。

---

## ⚠ 明确排除、不要上传

| 内容 | 原因 |
|---|---|
| 模型权重(`params/` `train_state/`) | 单档 9 GB;best 档走 Hugging Face |
| 完整数据集 | ~10 GB;走 Hugging Face |
| 原始长视频、逐帧 PNG | 体积大且无展示价值;只放剪好的 GIF |
| `*.bak_*` 备份 | 原项目里有大量迭代备份,是噪声 |
| 日志 / cache / `__pycache__` | 已在 `.gitignore` 里 |
| **厂商 SDK `pyAgxArm`** | LGPL-3.0,含固件二进制,**不可再分发** |
| **LimX 低层 SDK** | 同上 |
| 内网 IP / 服务器路径 / 账号 | 已由 `_tools/sanitize.py` 换成占位符 |
| CAN 适配器与相机序列号 | 实验室设备指纹,已脱敏 |
| Repo 2(sim-real) | **是伙伴做的**,不列入个人作品集 |

---

## 脱敏说明

`_tools/sanitize.py` 把内网信息换成了占位符:

```
<POLICY_SERVER_IP>  <ROBOT_WEB_IP>  <DATA_ROOT>  <CKPT_ROOT>
<OPENPI_REPO>  <SERVER>  <USER>  <LEFT_CAN_SERIAL>  ...
```

**端口号保留了**(8026 等)—— 它们是配置约定,去掉反而看不懂四段怎么对应模型。
算法参数、阈值、实测数据全部保留,那才是这份代码的价值。

`_tools/` 本身不要上传(它的规则表里还留着原始字符串)。

重新扫描:

```bash
grep -rInE '192\.168|10\.90\.|/mnt/(nvme2t|hdd3)|<真实用户名>' <repo>/
```
