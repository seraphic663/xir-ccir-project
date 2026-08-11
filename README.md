# XIR / CCIR 项目公开材料

欢迎从这里开始看 XIR 项目。仓库以最新 CCIR v3.0 为主，中文材料优先，适合先看答辩和报告，再让 AI 帮你核查项目。

## 先克隆，再问 AI

```bash
git clone https://github.com/seraphic663/xir-ccir-project.git
cd xir-ccir-project
```

然后把这段话发给能读取当前目录的 AI：

```text
请先阅读 README.md、docs/project/FACTS.md、docs/project/SOURCES.md、docs/defense/v3.0/README.md 和 docs/reports/README.md，用中文给我概括项目目标、M10 的训练方法、A/B 榜结果、证据边界，以及仓库包含和不包含什么。之后如果我问某一页 PDF、某个数字或某段代码，请给出对应的仓库路径，并区分官方来源、代码证据、平台记录、推断和未知；不要先运行训练。
```

接下来可以直接问：

```text
请核查 CCIR v3.0 第 N 页：从 PDF 和对应 TeX 提取主张，再用报告、FACTS.md 和代码逐项核对，输出“主张—证据路径—证据等级—风险/未知”。
```

```text
请静态核查 M10 的数据切分、loss、训练步数和输入 SHA；不要启动训练，并说明哪些外部文件仍然缺失。
```

## 从哪里看

| 想看什么 | 入口 |
|---|---|
| 最新中文答辩 | [`docs/defense/v3.0/v3.0.pdf`](docs/defense/v3.0/v3.0.pdf) |
| 答辩源文件 | [`docs/defense/v3.0/source/`](docs/defense/v3.0/source/) |
| 中文两页打印稿 | [`docs/reports/chinese-two-page/`](docs/reports/chinese-two-page/) |
| 中文 LRAT 风格稿 | [`docs/reports/chinese-lrat-style/`](docs/reports/chinese-lrat-style/) |
| 英文报告 | [`docs/reports/english/`](docs/reports/english/) |
| 项目事实与来源 | [`docs/project/FACTS.md`](docs/project/FACTS.md)、[`SOURCES.md`](docs/project/SOURCES.md) |
| M10 基础代码 | [`code/m10-reproduction/`](code/m10-reproduction/) |
| 审计与评测脚本 | [`code/audit/`](code/audit/) |

当前快照（2026-08-11）共 90 个文件、约 3.4 MB：PDF 6 个、TeX 17 个、Markdown 15 个、Python 30 个、构建/脚本 12 个。真正用于阅读的主 PDF 有 4 个：v3.0 中文答辩、中文两页稿、中文 LRAT 风格稿和英文报告；另外 2 个是主题资源。

## 当前公开口径

- M10：原始 Qwen3-Embedding-0.6B、官方 LRAT 输入、query-disjoint 切分，全参数训练，1 epoch、11,764 steps、learning rate `2e-6`。
- A 榜记录：Total `40.6`、Recall `46.3`、Success `21.1`、AvgSteps `16.0`。
- B 榜结果和完整证据边界请以 [`docs/project/FACTS.md`](docs/project/FACTS.md) 及报告为准。

这些是项目材料中记录的结果。代码用于说明训练合同、输入身份和核查方法；仅凭 clone 不能自动得到同一个 checkpoint，也不能访问比赛平台。

## 仓库不包含什么

训练数据、trajectory/corpus、基础模型和训练后权重、checkpoint、optimizer/scheduler/RNG 状态、B 榜 ZIP、比赛上传 JSONL、缓存、训练日志、SSH/Hugging Face 凭据，以及内部工作流文件均未放入仓库。

因此，代码里的外部路径、下载说明和 SHA-256 只是身份核查信息，不代表这些文件已经随仓库提供。

## 如果要复现

先看 [`code/m10-reproduction/README.md`](code/m10-reproduction/README.md)，再运行 `reproduce_m10.py --dry-run` 检查外部输入和切分。完整训练需要官方输入、GPU 和较大磁盘；只想理解项目时，不需要运行训练。

官方 LRAT 代码、训练数据和基础模型链接见 [`docs/project/SOURCES.md`](docs/project/SOURCES.md)。
