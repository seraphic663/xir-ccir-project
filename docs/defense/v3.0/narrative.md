# v3.0 答辩叙事设计

## 一句话主线

Agent 会为多步 reasoning 生成中间 query，检索结果继续影响后续推理；因此本项目要回答的是：如何围绕固定 Training pairs，构建更适合这种使用方式、且来源可核验、query 不泄漏、训练可复核的 retriever。

## 讲述顺序

先说明传统检索与 Agentic Search 的使用差异，再提出三个具体问题：Training pairs 能否追溯到 trajectory 和 Corpus、重复 query 怎样隔离、训练轮数与学习率怎样选择。随后分别介绍来源核验、normalized-query 整组隔离、1 正例 + 5 负例的加权对比训练和固定 dev1500 的配置选择。最后用 A 榜回答端到端效果，用 B 榜回答当前三种 Agent 上的评测结果和交付边界。

## “改进”具体指什么

1. **监督信号与来源核验**：论文的行为逻辑是 Search→Browse 提供正例候选，同一候选集内未 Browse 的文档进入轨迹负例池，Browse 后 reasoning length 提供强度信息。本项目不重新生成 trajectory，也不重新计算这些标签和权重；我们把 Training pair 的 query、文档 ID、负例和权重回溯到原始轨迹与 Corpus，并记录 stable/ambiguous/mismatch。
2. **normalized query 整组隔离**：先做 strip、合并连续空白和 Unicode casefold，再按完整 normalized-query group 分配 train/dev/locked test，避免同一 query 的不同 source rows 被随机拆开。这个处理改善的是证据边界，不直接等价于排行榜得分提升。
3. **固定样本组装与加权跨卡训练**：每行形成 1 positive + 5 negatives 的六路候选组；query/passage 上限为 128/512；使用 Training pair 中的 `reweight_rate`，在两张 A40 上做 cross-device negatives 和 weighted contrastive learning。128/512 是计算预算，不能说成理论无损。
4. **先选训练轮数，再选学习率并验收交付**：先固定 LR=1e-6，分别训练 1、2、3 epoch 并提交 A 榜，确定不继续累计 epoch；再在相同 500-step 设置下比较四个学习率，`2e-6` 同时取得最高 R@1 和 MRR，随后进入完整一轮。locked test 不回流；训练完成后再做独立加载、冻结、manifest、公开模型、代码和复现包验收。

## 12 页结构

| 页 | 主张 | 作用 |
|---:|---|---|
| 1 | 交代题目、队伍和对象 | 让听众知道讲的是 Agent 轨迹驱动的 retriever |
| 2 | 团队与分工 | 满足通知中的团队介绍、成员角色和队伍标识要求 |
| 3 | Agent 的中间 query 改变检索器的使用方式 | 从使用差异提出研究问题 |
| 4 | 从 LRAT 到比赛训练的三个落地问题 | 建立问题—方案对应关系 |
| 5 | 轨迹行为如何对应 Training pairs，并完成来源核验 | 介绍监督来源和第一项工作 |
| 6 | 按 normalized query 整组隔离 | 介绍第二项工作和评测边界 |
| 7 | 1+5 训练样本组装与长度预算 | 介绍样本层的固定配方 |
| 8 | reweight_rate 如何进入 loss 与双卡更新 | 介绍优化层的真实计算方式 |
| 9 | 先用 A 榜确定训练轮数，再用 dev1500 选择学习率 | 介绍配置选择 |
| 10 | A 榜平台对比与内部受控对照 | 回答效果及归因边界 |
| 11 | B 榜跨 Agent 结果与复现验收 | 回答当前评测覆盖与交付证据 |
| 12 | 方案、结果与边界 | 收束贡献，主动说明不能声称的内容 |

## 证据分层

论文用于解释研究背景和 LRAT 行为信号；本项目源码、manifest 和实验记录用于证明我们的实现；A 榜与 B 榜平台记录用于证明外部效果。报告不把论文的 91,713 training pairs、InfoSeekQA 轨迹生成或论文的 LLM judge 过程写成我们本次重新完成的工作。

## 结尾应该留下的印象

我们围绕固定 Training pairs 完成了来源核验、query 隔离、样本组装、加权跨卡训练和配置选择；A 榜最终 Total 为 40.6，B 榜覆盖三种 Agent、平均 Total 为 56.18。A 榜内部对照显示，query-disjoint 数据边界和学习率都会改变结果；B 榜结果则说明同一 retriever 在三种 Agent 上都能完成端到端评测，但不能外推为对所有 Agent 的统一提升。
