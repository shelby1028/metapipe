# metapipe

> **零代码 Meta 分析流水线工具包：在 Python 中从研究数据到可复现报告的一条命令。**

[![PyPI version](https://img.shields.io/pypi/v/metapipe.svg)](https://pypi.org/project/metapipe/)
[![Python versions](https://img.shields.io/pypi/pyversions/metapipe.svg)](https://pypi.org/project/metapipe/)
[![License](https://img.shields.io/pypi/l/metapipe.svg)](LICENSE)
[![CI status](https://github.com/shelby1028/metapipe/actions/workflows/ci.yml/badge.svg)](https://github.com/shelby1028/metapipe/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/metapipe)](https://pepy.tech/project/metapipe)

## 中文

`metapipe` 是一个 **Python 原生、面向循证研究的 Meta 分析工具包**。它将效应量计算、固定与随机效应模型、异质性诊断、可视化、亚组分析、元回归与结果报告整合为可复现的工作流，让研究者可以从 CSV 数据快速得到论文级输出。

### 快速开始

安装公开版本：

```bash
pip install metapipe
```

下面三行代码计算两项研究的 Hedges' *g*，并使用 REML 随机效应模型合并结果：

```python
from metapipe.effects import hedges_g
from metapipe.models import random_effects
result = random_effects([hedges_g(26, 4, 42, 24, 4.2, 40), hedges_g(28, 4.1, 46, 25.9, 4.3, 44)], tau_method="reml")
```

下面的一行命令会从连续型结局 CSV 生成完整 Markdown 报告、Excel 结果工作簿和诊断图：

```bash
metapipe report examples/sample_data.csv --output report.md
```

### 已实现功能

| 模块 | 功能 |
|---|---|
| 📏 **效应量** | 连续型结局的 MD、Cohen's *d*、Hedges' *g*；二分类结局的 OR、RR、RD 与标准误；OR 和 SMD 转换。 |
| ⚖️ **模型** | 逆方差固定效应、Mantel–Haenszel OR、DerSimonian–Laird、REML 与 Paule–Mandel 随机效应模型。 |
| 📊 **可视化** | 300 DPI 森林图、漏斗图与 L'Abbé 图；支持 PNG、SVG、PDF 输出和中英文标签。 |
| 🔎 **诊断** | Egger 线性回归检验、Begg 秩相关检验、逐一剔除敏感性分析、残差与影响诊断。 |
| 🧩 **亚组分析** | 分类亚组的固定/随机效应合并、组间 Q 检验和分颜色亚组森林图。 |
| 📈 **元回归** | 支持连续和分类协变量的混合效应元回归，提供 WLS 或 ML 估计、系数表、R²、残余异质性和拟合指标。 |
| 🔀 **PRISMA** | 兼容 PRISMA 2020 风格的文献筛选流程图，支持自定义排除原因和中英文标签。 |
| ✅ **GRADE** | 五个降级维度、三个升级维度、可审计评级表与 Markdown 导出。 |
| 📝 **一键报告** | 从 CSV 自动完成效应量、模型、森林图、漏斗图、Egger 检验、敏感性分析、Markdown 报告和 Excel 导出。 |
| 🖥️ **命令行** | `metapipe --version`、`metapipe forest`、`metapipe funnel` 与 `metapipe report`。 |

### 完整示例：从 CSV 到一键报告

仓库中的 `examples/sample_data.csv` 包含 12 项有氧运动干预对老年人 MMSE 影响的连续型研究。以下流程使用默认的 Hedges' *g* 与 REML 随机效应模型：

```bash
pip install -e ".[dev]"
metapipe report examples/sample_data.csv --output outputs/mmse_report.md
```

该命令将生成以下可复现产物：

| 文件 | 内容 |
|---|---|
| `outputs/mmse_report.md` | 自动生成的方法、合并结果、异质性、Egger 检验、敏感性分析和结论模板。 |
| `outputs/mmse_report.xlsx` | Summary、Study effects、Leave-one-out 及可选 Subgroups 工作表。 |
| `outputs/mmse_report_assets/forest_plot.png` | 含单项研究 95% CI、权重、合并菱形和 I² 的森林图。 |
| `outputs/mmse_report_assets/funnel_plot.png` | 含 95% 伪置信区间和 Egger *p* 值的漏斗图。 |
| `outputs/mmse_report_assets/leave_one_out.png` | 逐一剔除研究后的敏感性分析森林图。 |

Python API 也可以按需配置效应量、模型与亚组列：

```python
from metapipe.report import AnalysisConfig, generate_report

config = AnalysisConfig(model_type="random", subgroup_column="study_type")
report = generate_report("examples/sample_data.csv", "outputs/mmse_report.md", config=config)
print(report.excel_path)
```

### 与 R metafor 的对比

`metafor` 是成熟的 R Meta 分析软件包；`metapipe` 并不试图替代其完整的统计生态，而是面向 Python 工作流提供从研究数据到报告的集成体验。[1]

| 能力 | metapipe | R metafor |
|---|---|---|
| 主要生态 | **Python 原生** | R 原生 |
| 效应量与模型 | 内置常见连续与二分类效应量，以及固定/随机效应模型 | 提供广泛的 Meta 分析建模能力 |
| 零代码流水线 | **CSV 驱动的一键工作流** | 通常需要编写 R 脚本 |
| 一键报告 | **内置 Markdown、Excel 与图像产物** | 通常借助 R Markdown / Quarto 组合 |
| CLI 支持 | **`metapipe forest`、`funnel`、`report`** | 以 R 函数调用为主 |
| 适用场景 | Python 数据管道、研究团队标准化交付、快速可复现报告 | 高级 R 建模、定制统计分析 |

### 示例数据

`examples/sample_data.csv` 是用于演示连续型结局工作流的 12 项有氧运动研究数据。`examples/sample_data_binary.csv` 则用于演示二分类结局的 OR、RR、RD 计算。两份文件均为**教学与测试用途的示例数据**，不应作为真实临床证据使用。

### 贡献

欢迎通过 [贡献指南](CONTRIBUTING.md) 提交问题、文档改进、测试或功能建议。开发环境可通过 `pip install -e ".[dev]"` 创建，并使用 `pytest`、`ruff check .` 和 `black --check .` 进行验证。

### 引用

If you use metapipe in your research, please cite our work. **[Citation placeholder]**

### 参考资料

[1] [Viechtbauer, W. *metafor*: Meta-Analysis Package for R](https://wviechtb.github.io/metafor/)

---

## English

`metapipe` is a **native Python toolkit for evidence synthesis and meta-analysis**. It combines effect-size calculation, fixed- and random-effects pooling, heterogeneity diagnostics, visualisation, subgroup analysis, meta-regression, and reproducible reporting into a single workflow from CSV data to publication-ready outputs.

### Quick start

Install the public package:

```bash
pip install metapipe
```

The following three lines calculate Hedges' *g* for two studies and pool them with a REML random-effects model:

```python
from metapipe.effects import hedges_g
from metapipe.models import random_effects
result = random_effects([hedges_g(26, 4, 42, 24, 4.2, 40), hedges_g(28, 4.1, 46, 25.9, 4.3, 44)], tau_method="reml")
```

Generate a complete Markdown report, Excel workbook, and diagnostic figures from a continuous-outcome CSV in one command:

```bash
metapipe report examples/sample_data.csv --output report.md
```

### Implemented features

| Module | Capability |
|---|---|
| 📏 **Effect sizes** | MD, Cohen's *d*, and Hedges' *g* for continuous outcomes; OR, RR, RD, and standard errors for binary outcomes; OR/SMD conversion. |
| ⚖️ **Models** | Inverse-variance fixed effect, Mantel–Haenszel OR, and DerSimonian–Laird, REML, and Paule–Mandel random-effects models. |
| 📊 **Visualisation** | 300 DPI forest, funnel, and L'Abbé plots with PNG, SVG, and PDF export plus bilingual labels. |
| 🔎 **Diagnostics** | Egger regression, Begg rank correlation, leave-one-out sensitivity analysis, and residual/influence diagnostics. |
| 🧩 **Subgroups** | Fixed- or random-effects categorical subgroup pooling, Q-between testing, and coloured subgroup forest plots. |
| 📈 **Meta-regression** | Mixed-effects meta-regression for continuous and categorical moderators with WLS or ML estimation, coefficient tables, R², residual heterogeneity, and fit statistics. |
| 🔀 **PRISMA** | PRISMA 2020-style study-selection diagrams with custom exclusion reasons and bilingual labels. |
| ✅ **GRADE** | Five downgrade domains, three upgrade domains, an auditable rating table, and Markdown export. |
| 📝 **One-click reports** | Automated effects, models, forest plots, funnel plots, Egger testing, sensitivity analysis, Markdown reports, and Excel export from CSV. |
| 🖥️ **CLI** | `metapipe --version`, `metapipe forest`, `metapipe funnel`, and `metapipe report`. |

### Full example: CSV to one-click report

The repository ships `examples/sample_data.csv`, a 12-study continuous-outcome example of aerobic exercise interventions and MMSE outcomes in older adults. The workflow below uses the default Hedges' *g* and REML random-effects model:

```bash
pip install -e ".[dev]"
metapipe report examples/sample_data.csv --output outputs/mmse_report.md
```

The command creates the following reproducible outputs:

| File | Content |
|---|---|
| `outputs/mmse_report.md` | Automatically generated methods, pooled results, heterogeneity, Egger test, sensitivity analysis, and conclusion template. |
| `outputs/mmse_report.xlsx` | Summary, Study effects, Leave-one-out, and optional Subgroups worksheets. |
| `outputs/mmse_report_assets/forest_plot.png` | Forest plot with individual 95% CIs, weights, pooled diamond, and I². |
| `outputs/mmse_report_assets/funnel_plot.png` | Funnel plot with 95% pseudo-confidence limits and Egger *p* value. |
| `outputs/mmse_report_assets/leave_one_out.png` | Leave-one-out sensitivity-analysis forest plot. |

The Python API can be configured for a chosen effect measure, model, and subgroup column:

```python
from metapipe.report import AnalysisConfig, generate_report

config = AnalysisConfig(model_type="random", subgroup_column="study_type")
report = generate_report("examples/sample_data.csv", "outputs/mmse_report.md", config=config)
print(report.excel_path)
```

### Comparison with R metafor

`metafor` is an established meta-analysis package for R. `metapipe` does not seek to replace its complete statistical ecosystem; instead, it provides an integrated data-to-report experience for Python workflows.[1]

| Capability | metapipe | R metafor |
|---|---|---|
| Primary ecosystem | **Native Python** | Native R |
| Effect sizes and models | Common continuous and binary measures plus fixed/random effects | Broad meta-analytic modelling capabilities |
| Zero-code pipeline | **CSV-driven one-command workflow** | Usually requires an R script |
| One-click reporting | **Built-in Markdown, Excel, and figure artefacts** | Commonly composed with R Markdown / Quarto |
| CLI support | **`metapipe forest`, `funnel`, and `report`** | Primarily R function calls |
| Best fit | Python data pipelines, standardised team delivery, quick reproducible reports | Advanced R modelling and bespoke statistical analysis |

### Example data

`examples/sample_data.csv` provides 12 aerobic-exercise studies for the continuous-outcome workflow. `examples/sample_data_binary.csv` demonstrates binary-outcome OR, RR, and RD calculations. Both files are **illustrative teaching and test data** and must not be used as clinical evidence.

### Contributing

Contributions are welcome through the [contribution guide](CONTRIBUTING.md), including issues, documentation improvements, tests, and feature proposals. Create a development environment with `pip install -e ".[dev]"`, then run `pytest`, `ruff check .`, and `black --check .`.

### Citation

If you use metapipe in your research, please cite our work. **[Citation placeholder]**

### References

[1] [Viechtbauer, W. *metafor*: Meta-Analysis Package for R](https://wviechtb.github.io/metafor/)
