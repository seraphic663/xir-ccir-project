# LRAT-style 中文论文稿

这是一版在 v2.5 基础上单独制作的论文式草稿，按照本地 LRAT 论文的叙事和排版组织：首页摘要与总览，正文从第 2 页开始双栏，使用蓝色编号章节、公式、表格、附录和参考文献。

它的定位是“完全采用 LRAT 风格来讲清楚本项目”，不是声称完整复现 LRAT 的 trajectory 生成、30B Agent、LLM judge 或 data flywheel 实验。现有 `../v2.5.pdf`、`../report_cn_2p.pdf` 和 Beamer 源文件保持不变。

## 文件

- `report_cn_lrat.pdf`：当前渲染并逐页检查的论文式 PDF。
- `report_cn_lrat.tex`：XeLaTeX 源文件。
- `output/.build/`：最近一次编译日志和辅助文件。
- `scripts/build_windows.cmd`：Windows MiKTeX 两遍编译脚本。

## 编译

在 Windows 下运行 `scripts\build_windows.cmd`。脚本会把 PDF 写入 `output/report_cn_lrat.pdf`，并同步到本目录根部。
