# CCIR 答辩材料 v3.0

这是当前公开材料中的最新 v3.0 答辩版本，按“研究背景 → 具体问题 → LRAT 监督信号 → 本次训练方法 → 配置选择 → A/B 榜结果 → 总结”的论文式叙事组织。

- `v3.0.pdf`：最终中文答辩 PDF。
- `source/src/v3.0.tex`：Beamer 入口；页面源文件位于 `source/src/frames/`。
- `source/theme/`、`source/assets/`：主题和资源。
- `source/scripts/`：Linux/Windows 构建与 watcher 脚本。
- `narrative.md`：v3.0 的核心主线、实际改进和逐页讲述目的。

中文两页稿、英文报告和中文 LRAT 风格稿统一放在 [`../../reports/`](../../reports/)。

## 阅读边界

正文把本地 dev 指标、正式 A 榜和 B 榜平台材料分开报告；具体事实和证据等级见 [`../../project/FACTS.md`](../../project/FACTS.md)。本目录只有报告与源码，不包含模型、checkpoint、训练数据或 B 榜附件。

## 构建

在 Windows MiKTeX 下进入 `source/`，运行 `scripts\\build_windows.cmd`；在 Linux/WSL 下运行 `source/scripts/build.sh`。构建脚本默认在源目录下生成临时输出；公开仓库只保留最终 PDF 和源文件，不提交构建中间文件。

