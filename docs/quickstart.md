# 快速开始

## 安装

`metapipe` 需要 Python 3.10 或更高版本。公开发布后可直接安装：

```bash
pip install metapipe
```

若在仓库中开发，请安装开发和文档依赖：

```bash
pip install -e ".[dev,docs]"
```

## Python 中的最小分析

```python
from metapipe.effects import hedges_g
from metapipe.models import random_effects

studies = [
    hedges_g(26, 4, 42, 24, 4.2, 40),
    hedges_g(28, 4.1, 46, 25.9, 4.3, 44),
]
result = random_effects(studies, tau_method="reml")
print(result.pooled_effect, result.ci_lower, result.ci_upper)
```

## 从 CSV 创建报告

连续型示例数据可直接生成 Markdown、Excel 和图像：

```bash
metapipe report examples/sample_data.csv --output outputs/continuous_report.md
```

二分类示例数据支持 log(OR)、log(RR) 或风险差：

```bash
metapipe report examples/sample_data_binary.csv \
  --effect-measure odds_ratio \
  --output outputs/binary_or_report.md
```

可将 `odds_ratio` 改为 `risk_ratio` 或 `risk_difference`。比值效应量会在对数尺度上合并，以适用于逆方差模型。
