# 来源与核查入口

## 官方上游

| 资源 | 地址 | 用途 |
|---|---|---|
| LRAT 代码 | <https://github.com/Yuqi-Zhou/LRAT> | Agent、Search/Browse、训练与评测的上游参考实现 |
| LRAT Training pairs | <https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train> | 官方训练 pairs 与相关输入说明 |
| 基础模型 | <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> | 原始 Qwen3-Embedding-0.6B |
| 项目公开 M10 复现参考 | <https://github.com/seraphic663/ccir-lrat-retriever> | 既有的独立 B 榜复现仓库；与本仓库的材料归档边界不同 |

## 本仓库内部的核查顺序

先看 `README.md` 和 `docs/project/FACTS.md`，再按具体问题选择：

- 结果和口径：`docs/reports/chinese-two-page/`、`docs/reports/english/`。
- 答辩页面：`docs/defense/v3.0/v3.0.pdf` 与 `docs/defense/v3.0/source/`。
- M10 训练合同：`code/m10-reproduction/README.md`、`DATA_SOURCES.json`、`code/config.yaml`。
- 代码实现：`code/m10-reproduction/code/`。
- provenance、切分和评测核查：`code/audit/`。

## 证据使用规则

回答具体事实时，优先给出仓库相对路径和文件名；如果主张依赖上游或平台，必须同时给出外部来源，并说明本仓库是否能独立复核。对于动态信息，先标注本文快照日期，再建议重新查询官方页面。

本仓库故意不包含训练数据、模型权重、checkpoint、B 榜 ZIP、比赛上传 JSONL、缓存、日志和私密凭据。因此，任何需要这些文件的复现步骤都只能验证“输入合同和代码路径”，不能在 clone 后离线完成。

