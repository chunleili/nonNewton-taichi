# 对齐报告与检查入口

2026-10-04。当前结论：两个热场景的全时长统计指标已明显接近，三个 ramp 仍未定量通过。没有约定统一验收阈值，不能将这些结果解释为逐粒子等价。

- [FULL_REPORT.md](FULL_REPORT.md)：第二轮和首轮完整报告合并版，先列当前结果，再列历史结果。
- [QUANTITATIVE_REPORT.md](QUANTITATIVE_REPORT.md)：第二轮热场景的完整定量报告。
- [REPORT.md](REPORT.md)：首轮五个场景报告，对应 Taichi `11dfc47`，热场景结论已由第二轮更新。
- [SESSION_HANDOFF.md](SESSION_HANDOFF.md)：任务、版本、变更、证据位置和待定事项。
- [REPRODUCE.md](REPRODUCE.md)：只读检查、环境安装和重跑命令。
- [CLAUDE_REVIEW_PROMPT.md](CLAUDE_REVIEW_PROMPT.md)：可直接交给 Claude 的检查要求。
- [quantitative/summary.json](quantitative/summary.json)：当前热场景统计及求解参数。
- [quantitative/overview.png](quantitative/overview.png)：C++、第二轮 Taichi、首轮 Taichi 全程曲线。
- [cpp_alignment.patch](cpp_alignment.patch)：C++ 从原基准到正式参考的完整修改。

`ramp1/`、`ramp2/`、`ramp3/`、`ice-cream/`、`hotcut/` 为首轮 CSV/图。
`quantitative/ice-cream/`、`quantitative/hotcut/` 为第二轮 CSV/图。
`quant_probe/` 为边界项、DFSPH 分步和单步检查的数据。

可复制的离线交接目录位于 `D:/Dev/nonNewton/CLAUDE_REVIEW_2026-10-04/`，从其中的 `START_HERE.md` 开始。
