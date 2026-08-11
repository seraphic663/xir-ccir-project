# 基于 Agent 轨迹的检索器训练

决赛答辩方案介绍 · v3.0

队伍：DefaultGroup（中国人民大学 CCIR 参赛队）；Team ID：362259；Record ID：998912；成员：刘琦（队长）、赵梓名（队员）；主要联系人：刘琦（13798122637）。根据组委会通知，本队在 B 榜评测中进入前三，晋级决赛答辩。

## 摘要

在 Agentic Search 中，Agent 会为多步 reasoning 生成中间 query，先查看 snippet，再决定是否 Browse 全文；检索结果随后进入下一轮 reasoning。LRAT 论文指出，Search→Browse、同轮未 Browse 候选和 post-Browse reasoning 可以提供与文档效用相关的监督。本项目不重新生成 Agent trajectory、不重新调用论文 judge，也不引入外部数据、教师模型、LoRA 或 Adapter，而是围绕官方 Training pairs、Agent trajectories、离线 Corpus 和指定的 Qwen3-Embedding-0.6B，建立来源可追溯、query 不泄漏、训练可执行、结果可验收的链路。最终 A 榜取得 Total 40.6、Recall 46.3、Success 21.1、Avg_Steps 16.0；B 榜三个 Agent 的平均 Total 为 56.18。

## 研究背景与总体链路

深度研究 Agent 会反复 Think、Search、Browse 和 Reason：Search 返回候选摘要，Browse 决定读取哪篇全文，全文内容再影响后续 query 和答案。因此 retriever 的目标不只是主题相关，而是返回能够推进当前局部推理的证据。我们的落地链路是：固定输入 → 轨迹/Training pair 来源核验 → normalized query 整组切分 → 1 个正例 + 5 个负例 → 加权对比训练 → A/B 榜评测 → 独立加载、冻结与复现验收。

论文中的 Browse 文档先作为候选正例，同一 Search 候选集合中未 Browse 的文档进入负例池，post-Browse reasoning 用于过滤无效浏览并提供效用强度信号；这些机制解释了为什么 Agent 轨迹适合作为面向 Agent 的训练监督。[1] 本项目实际使用组委会发布的 Training pairs：trajectory 和 Corpus 用于来源与文档一致性核验，不在本次工作中重新采集 trajectory、重算 label/weight 或完成论文式 Data Flywheel。

## 数据核验与 query-disjoint 切分

我们将 query、正例文档、负例文档和 `reweight_rate` 回溯到轨迹事件，并检查文档是否存在于固定 Corpus。96,504 条 Training pairs 中，96,366 条可稳定映射，138 条为字段一致的多候选映射，0 条无法匹配；1,989,015 个文档引用、959,042 个唯一文档 ID 均无缺失或文本不一致。

对 query 执行去首尾空格、合并连续空白和 Unicode casefold；相同 normalized query 的所有 source rows 作为一个 group，整组分配到 train、dev 或 locked test。最终 train 为 78,890 groups / 94,113 rows，dev 为 1,500 groups，locked test 为 500 groups，集合间 overlap 为 0。dev 只用于开发与选择，locked test 不回流。

## 训练方法与配置

每条 Training pair row 构造成 1 个 query、1 个 positive 和从该行 `neg` 列表随机抽取的 5 个 negatives，组成六路候选组；正例固定在候选位置 0。query / passage 上限为 128 / 512 tokens。模型从指定的 Qwen3-Embedding-0.6B 开始全参数更新，使用共享 encoder、last-token pooling、L2 normalization 和 1024 维向量。相似度为 $s(q,d)=q^{\mathsf T}d/\tau$，其中 $\tau=0.02$；设 query 对比交叉熵为 $\ell_j$、官方 `reweight_rate` 为 $w_j$，实现的加权目标为 $\mathcal{L}=\sum_j w_j\ell_j/\sum_j w_j$。权重只改变 query 对 loss 的贡献，不改变正负标签。

两张 A40 各处理 1 个 candidate group，cross-device gather 后另一 query 的候选文档作为 in-batch negatives；gradient accumulation=4 只改变 optimizer update 粒度，effective query batch 为 8，不合并四个 micro-step 的候选池。最终训练使用 1 epoch、11,764 steps、per-device batch 1、accumulation 4、bf16、group size 6 和 learning rate `2e-6`。

## 配置选择、实验结果与边界

我们先在完整 96,504 条 Training pairs 上固定 LR=`1e-6`，分别训练 1、2、3 epoch 并提交 A 榜；Total 分别为 40.1、38.2、38.7，因此选择 1 epoch。随后固定 query-disjoint train（94,113 rows），在 dev1500 上对四个学习率做 500-step 选择：`3e-7`、`5e-7`、`1e-6`、`2e-6` 的 R@1/MRR 分别为 0.4913/0.6419、0.5073/0.6571、0.5433/0.6831、0.5640/0.6993，最终选择 `2e-6`。

正式 A 榜对照为：Baseline Qwen3 Embedding 28.5/28.5/10.6/17.8，赛事官方 LRAT baseline 34.0/36.1/15.5/16.8，query-disjoint（`1e-6`）39.5/45.0/19.8/15.9，最终候选（`2e-6`）40.6/46.3/21.1/16.0；字段依次为 Total、Recall、Success、Avg_Steps。最终候选相对两个外部/平台参照行分别提高 12.1 和 6.6 个 Total 点；在同一 query-disjoint 配方中，调整 LR 后由 39.5 提升至 40.6。以上是平台结果行之间的比较，不能归因给单一变量。

B 榜在 DeepSeek-V4-Flash-0731、Qwen3.5-35B-A3B、gpt-oss-120b 三种 Agent 上的 Total 分别为 67.42、60.96、40.16，平均 56.18，最高与最低相差 27.26。它支持跨 Agent 的实测可运行性，但由于没有匹配的逐 Agent baseline，正文只报告绝对表现，不外推为对所有 Agent 的普遍提升。

## 交付与结论

最终模型完成独立 CPU 进程加载、query/document 推理 smoke、冻结 manifest、Hugging Face 公开 revision 回读以及 GitHub/复现包校验；具体哈希、commit 和逐文件证据保留在《复现验收附录》中。我们的核心贡献不是重新制造监督，而是把固定官方监督组织成一条可审计、可隔离、可训练、可复核的工程链路：来源核验回答“信号从哪里来”，query-disjoint 保护开发边界，固定候选与加权 loss 记录训练语义，分阶段实验确定 epoch 与 LR，A/B 榜给出正式效果及其跨 Agent 限定。

## 参考资料

[1] Yuqi Zhou et al., “Learning to Retrieve from Agent Trajectories,” arXiv:2604.04949, 2026；[2] BC-Plus A Leaderboard；[3] CCIR Cup 2026 赛事说明页。
