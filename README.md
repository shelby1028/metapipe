# metapipe

**metapipe** is a Python toolkit for composing an end-to-end meta-analysis workflow from a single command-line interface. Version 0.1.0 establishes the statistical core for common effect sizes and fixed- and random-effects pooling while the broader no-code pipeline is being developed.

## Installation

Install the released package from PyPI when available:

```bash
pip install metapipe
```

For local development, clone this repository and install it in editable mode:

```bash
pip install -e ".[dev]"
```

## Three-line example

```python
from metapipe.effects import hedges_g
from metapipe.models import random_effects
result = random_effects([hedges_g(10, 2, 30, 8, 2, 30).effect], [0.07])
```

## Features

The following user-facing workflow features are **under development**:

- Importing study data from spreadsheet templates.
- One-command execution of screening, effect calculation, pooling, diagnostics, and reporting.
- Forest plots, funnel plots, sensitivity analyses, and exportable reports.

## Development

The project requires Python 3.10 or newer. Install the editable development environment and run the quality checks with:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

This project is distributed under the [MIT License](LICENSE).
