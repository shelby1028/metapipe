# 示例数据与工作流

`examples/` 目录提供两份用于教学、测试和可复现演示的数据。它们不代表真实临床证据。

| 文件 | 结局类型 | 行数 | 可用分析 |
|---|---:|---:|---|
| `sample_data.csv` | 连续型：MMSE | 12 | MD、Cohen's *d*、Hedges' *g*、报告、亚组、元回归。 |
| `sample_data_binary.csv` | 二分类：疾病发生 | 10 | OR、RR、RD、森林图、漏斗图和报告。 |

## 连续型结局

```bash
metapipe forest examples/sample_data.csv --output outputs/mmse_forest.png
metapipe funnel examples/sample_data.csv --output outputs/mmse_funnel.png
metapipe report examples/sample_data.csv --output outputs/mmse_report.md
```

报告将产生 `mmse_report.md`、`mmse_report.xlsx` 和 `mmse_report_assets/` 图像目录。

## 二分类结局

Python API 可明确计算三类二分类效应量：

```python
from metapipe.effects import odds_ratio, risk_difference, risk_ratio

or_result = odds_ratio(8, 72, 14, 64, log_scale=True)
rr_result = risk_ratio(8, 72, 14, 64, log_scale=True)
rd_result = risk_difference(8, 72, 14, 64)
```

报告命令支持三种分析尺度：

```bash
metapipe report examples/sample_data_binary.csv --effect-measure odds_ratio --output outputs/or_report.md
metapipe report examples/sample_data_binary.csv --effect-measure risk_ratio --output outputs/rr_report.md
metapipe report examples/sample_data_binary.csv --effect-measure risk_difference --output outputs/rd_report.md
```

## 亚组和元回归

```python
import pandas as pd
from metapipe.effects import hedges_g
from metapipe.meta_regression import meta_regression
from metapipe.subgroup import subgroup_analysis

frame = pd.read_csv("examples/sample_data.csv")
effects = [hedges_g(r.mean_treatment, r.sd_treatment, r.n_treatment, r.mean_control, r.sd_control, r.n_control) for r in frame.itertuples()]
subgroups = subgroup_analysis(effects, frame["study_type"])
regression = meta_regression(effects, frame[["duration", "study_type"]])
```
