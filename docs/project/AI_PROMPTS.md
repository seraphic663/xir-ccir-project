# AI 审阅 Prompt 模板

下面的模板假定 AI 已经能够读取本仓库文件。每次提问都应要求它给出文件路径、证据等级和未知项。

## 1. 项目总览

```text
请阅读 README.md、docs/project/FACTS.md、docs/project/SOURCES.md、docs/defense/v3.0/README.md 和 docs/reports/README.md。用一页中文说明：项目要解决什么问题、M10 做了什么、哪些结果来自 A/B 榜、仓库中实际包含什么、明确不包含什么。每个关键结论后标注仓库文件路径；不要把推断写成事实。
```

## 2. 核查答辩某一页

```text
请核查 CCIR v3.0 第 N 页。先从 docs/defense/v3.0/v3.0.pdf 和对应的 source/src/frames/ 文件提取主张，再到 docs/reports/、docs/project/FACTS.md 和 code/ 中寻找支持。输出“主张—证据—证据等级—是否存在口径风险”四列表格；如果只能由平台结果或用户截图支持，请明确写出。
```

## 3. 核查训练方法

```text
请只基于 code/m10-reproduction/ 和 code/audit/ 核查 M10 的数据切分、样本组装、loss、pooling、cross-device negatives、训练步数和输入身份校验。不要假设数据或模型在仓库中。指出代码能证明什么、不能证明什么，以及需要哪些外部输入才能运行 dry-run。
```

## 4. 核查一条结果主张

```text
我要核查这条主张：<粘贴主张>。请搜索整个仓库，但优先使用 README.md、docs/project/FACTS.md、报告源文件和测试。输出：1) 精确支持位置；2) 是官方、不可变、本地、平台还是推断证据；3) 是否有冲突版本；4) 仍需外部验证的最小步骤。不要因为多个文件重复同一个数字就把它升级为独立证据。
```

## 5. 复现前检查

```text
请检查我提供的外部输入是否满足 code/m10-reproduction/README.md 和 DATA_SOURCES.json 的身份要求。只做静态核查和命令审阅，不启动训练、不下载大文件、不覆盖任何输出。最后给出 dry-run 命令、预计资源和失败时最可能的三类原因。
```

