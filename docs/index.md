# metapipe

> **零代码 Meta 分析流水线工具包：从研究数据到可复现报告的一条命令。**

`metapipe` 是面向 Python 生态的循证研究工具包。它将效应量计算、模型拟合、可视化、诊断、亚组分析、元回归、GRADE 评级与一键报告组织为一致且可测试的工作流。

## 为什么使用 metapipe？

研究团队常常需要在多个工具、脚本与手工表格之间传递数据。`metapipe` 让连续型和二分类结局都可以从结构化 CSV 开始，生成统计结果、图像、Markdown 报告和 Excel 工作簿。

| 工作流阶段 | metapipe 提供的能力 |
|---|---|
| 研究级数据 | 连续型 MD、Cohen's *d*、Hedges' *g*，以及二分类 OR、RR、RD。 |
| 合并分析 | 固定效应、Mantel–Haenszel 与多种随机效应 tau² 估计。 |
| 结果解释 | 异质性、Egger/Begg、逐一剔除、异常值、亚组、元回归和 GRADE。 |
| 研究交付 | 森林图、漏斗图、L'Abbé 图、PRISMA 图、Markdown 与 Excel。 |

从[快速开始](quickstart.md)进入，或直接查看[示例数据与工作流](examples.md)。
