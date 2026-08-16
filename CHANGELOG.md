# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-24

### Added

#### Core effect sizes and models

- Continuous-outcome mean difference, Cohen's *d*, Hedges' *g*, exact small-sample correction, and sampling uncertainty calculations.
- Binary-outcome odds ratio, risk ratio, risk difference, zero-cell continuity correction, and OR/SMD conversion utilities.
- Inverse-variance fixed-effect, Mantel–Haenszel odds-ratio, DerSimonian–Laird, REML, and Paule–Mandel meta-analysis models.
- Structured model results with pooled effects, confidence intervals, p values, Q, I², tau², and relative study weights.

#### Visualisation and diagnostics

- Publication-ready 300 DPI forest plots, funnel plots, and L'Abbé plots with PNG, SVG, and PDF export.
- Egger regression, Begg rank correlation, leave-one-out sensitivity analysis, residual diagnostics, and Cook-style influence flags.
- Fixed- and random-effects subgroup analyses, Q-between testing, and coloured subgroup forest plots.
- Mixed-effects meta-regression for continuous and categorical moderators with WLS and maximum-likelihood estimation.

#### Evidence synthesis workflow

- PRISMA 2020-style study-selection flowcharts with bilingual labels and configurable exclusion reasons.
- GRADE certainty assessment across five downgrade and three upgrade dimensions, with audit tables and Markdown export.
- One-click CSV reporting with Markdown, Excel, forest plot, funnel plot, Egger test, leave-one-out analysis, and optional subgroup analysis.
- Continuous and binary report workflows supporting Hedges' *g*, mean difference, log odds ratio, log risk ratio, and risk difference.
- Command-line commands for version checks, forest plots, funnel plots, and complete reports.

#### Documentation and release tooling

- Bilingual README, MkDocs Material documentation site, automated API references, and GitHub Pages deployment workflow.
- Continuous and binary example datasets for reproducible demonstrations.
- GitHub Actions CI matrix for Python 3.10, 3.11, and 3.12.
- PyPI release workflow triggered by version tags.

### Changed

- Project metadata now identifies Liu Xin as author and includes public homepage, repository, documentation, and issue-tracker URLs.
- Development configuration now provides optional documentation dependencies for local MkDocs builds.

### Fixed

- Report input validation now selects continuous or binary required columns according to the configured effect measure.
- Binary report calculations now use log-scaled ratio effects where appropriate for inverse-variance pooling.

[Unreleased]: https://github.com/shelby1028/metapipe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shelby1028/metapipe/releases/tag/v0.1.0
