# 支撑核查代码

这些脚本用于解释和复核报告中的数据链路，不是一个带数据即可无条件运行的完整工程。它们需要外部 trajectory archive、Training pairs、tokenizer、模型或评测输入；这些文件没有随仓库公开。

| 文件 | 作用 |
|---|---|
| `build_trajectory_provenance.py` | 将 pair 与原始 trajectory 做可审计映射，并记录稳定身份与歧义 |
| `prepare_early_stop_split.py` | 按 normalized query 构造 query-disjoint train/dev/locked-test 划分 |
| `evaluate_qwen3_pairs.py` | 对 query/document pair 运行 embedding 与召回指标评测 |
| `compare_paired_evals.py` | 对成对 query 结果进行差异比较和不确定性分析 |

优先使用 `code/m10-reproduction/code/reproduce_m10.py --dry-run` 做公开 M10 合同检查；这些审计脚本适合在需要解释某个主张时阅读或在拥有合法外部输入后运行。

