# 项目事实与证据边界

快照日期：2026-08-11。本文是面向公开审阅的短版事实表，不替代比赛平台原始记录；若未来材料更新，应以新的、带日期的证据为准。

## 项目是什么

XIR / CCIR 项目研究如何利用 Agentic Search 轨迹中的 query、Search→Browse 转换、候选文档和后续 reasoning 信号，训练更适合 Agent 检索的 dense retriever。当前公开材料的核心实现围绕官方 LRAT Training pairs、trajectories、离线 corpus 和 Qwen3-Embedding-0.6B 展开。

项目本次工作的重点不是重新生成 Agent trajectory，也不是重新执行论文中的外部 judge 流程，而是把官方输入、provenance、query-disjoint 切分、加权对比学习、独立评测和交付身份连接成一条可审计链路。

## M10 的身份

| 项目 | 公开记录 |
|---|---|
| 基础模型 | `Qwen/Qwen3-Embedding-0.6B`，全参数训练 |
| 训练输入 | 官方 LRAT Training pairs、Agent trajectories 与离线 corpus；本仓库不包含这些文件 |
| 训练划分 | `94,113` 行 query-disjoint train；dev 与 locked test 的 query 组独立于 train |
| 训练合同 | 1 epoch、11,764 optimizer steps、per-device batch 1、gradient accumulation 4、group size 6 |
| 优化设置 | learning rate `2e-6`、temperature `0.02`、bf16、cross-device negatives |
| 表示方式 | last-token pooling、L2 normalization、1024 维 embedding |
| 正式 A 榜 | Total `40.6` / Recall `46.3` / Success `21.1` / AvgSteps `16.0` |
| B 榜材料 | 三个 Agent 的 Total `67.42 / 60.96 / 40.16`，报告平均 `56.18` |

## 结果怎么读

- A 榜数字是平台结果，不是本地 dev 指标。
- B 榜三个 Agent 的数字证明了当前材料记录的跨 Agent 可运行性，但没有匹配的逐 Agent baseline，因此不能扩写成“对所有 Agent 普遍提升”。
- dev1500 用于训练内选择和诊断；它不能替代 A 榜，也不能自动证明跨 Agent 泛化。
- 独立复现的目标是重新完成固定数据处理、训练合同和指标容差验收，不承诺浮点 checkpoint 与原提交权重逐字节一致。

## 生命周期边界

M01 是此前的正式参考；M07 是早停路线的正式模型；M10 是当前材料中的正式 A 榜最佳和 B 榜默认候选；M12 是完成过平台探索但未通过预声明候选门控的 search-index 加权实验。它们的训练来源、切分、评测和生命周期不同，不能仅凭模型编号或目录名称互相替代。

本公开仓库不承载 M10 的权重、checkpoint、B 榜复现 ZIP 或上传 JSONL。现有 B 榜交付包仍是独立的冻结对象；本仓库只引用公开方法和身份信息，不复制附件内容。

## 证据等级

1. `官方来源`：上游 LRAT 仓库、官方数据/模型页面和正式比赛定义。
2. `不可变身份`：输入或模型的 SHA-256、HF commit、Git commit、逐文件 manifest。
3. `本地可复核`：仓库中的代码、配置、测试、报告源文件和构建产物。
4. `平台结果`：A/B 榜网页、平台导出或用户提供的截图；截图证据不能自动扩写为完整原始评测明细。
5. `解释或推断`：由上述证据推导出的机制解释，必须标注为推断，不能写成已证明因果。

## 读者不应做的事

- 不要因为仓库里有训练入口，就声称本仓库已携带数据、模型或已经完成训练。
- 不要把论文中的数据生成流程、judge 或 Data Flywheel 过程写成当前比赛训练重新完成的步骤。
- 不要把历史诊断实验的 test 或 dev 结果当作独立最终泛化证据。
- 不要从报告中的绝对路径、服务器参数或 hash 反推出本仓库包含相应私有文件。

