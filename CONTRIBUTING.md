# Contributing to metapipe

Thank you for considering a contribution to **metapipe**. The project aims to make rigorous meta-analysis workflows accessible and transparent in the Python ecosystem.

## Development setup

Clone the repository, create a Python 3.10+ virtual environment, activate it, and install the package with development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality checks

Run the test suite with coverage and then run static checks before opening a pull request:

```bash
pytest
ruff check .
black --check .
```

## Contribution guidelines

Please keep pull requests focused and include tests for all behavior changes. Public functions must include Google-style docstrings, numerical routines should state their assumptions in code or documentation, and changes that affect users should update `CHANGELOG.md`.

## Reporting issues

Please use the issue tracker to report reproducible bugs, documentation gaps, or feature proposals. A useful report includes the package version, Python version, a minimal reproducible example, expected behavior, and observed behavior.
