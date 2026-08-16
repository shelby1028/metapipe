"""One-click end-to-end meta-analysis report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from metapipe.diagnostics import egger_test, leave_one_out
from metapipe.effects import EffectSize, hedges_g, mean_difference
from metapipe.models import MetaAnalysisResult, TauMethod, fixed_effect, random_effects
from metapipe.plots import forest_plot, funnel_plot
from metapipe.subgroup import SubgroupAnalysisResult, subgroup_analysis

EffectMeasure = Literal["hedges_g", "mean_difference"]
ModelType = Literal["fixed", "random"]

_REQUIRED_COLUMNS = {
    "study",
    "n_treatment",
    "mean_treatment",
    "sd_treatment",
    "n_control",
    "mean_control",
    "sd_control",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for a CSV-driven meta-analysis report.

    Attributes:
        effect_measure: ``"hedges_g"`` or ``"mean_difference"``.
        model_type: ``"fixed"`` or ``"random"`` pooling model.
        tau_method: Tau-squared estimator used by random-effects fits.
        subgroup_column: Optional categorical CSV column for subgroup analysis.
        include_funnel: Whether to generate a funnel plot and Egger result.
        include_sensitivity: Whether to run leave-one-out sensitivity analysis.
    """

    effect_measure: EffectMeasure = "hedges_g"
    model_type: ModelType = "random"
    tau_method: TauMethod = "reml"
    subgroup_column: str | None = None
    include_funnel: bool = True
    include_sensitivity: bool = True


@dataclass(frozen=True)
class ReportResult:
    """Files and fitted results produced by :func:`generate_report`.

    Attributes:
        markdown_path: Path to the Markdown report.
        excel_path: Path to the Excel workbook containing results tables.
        asset_paths: Mapping of generated asset names to file paths.
        model_result: Overall fixed-effect or random-effects fit.
        subgroup_result: Optional categorical subgroup-analysis result.
    """

    markdown_path: Path
    excel_path: Path
    asset_paths: dict[str, Path]
    model_result: MetaAnalysisResult
    subgroup_result: SubgroupAnalysisResult | None


def _validate_config(config: AnalysisConfig) -> None:
    """Validate report configuration before any analysis is run."""
    if config.effect_measure not in {"hedges_g", "mean_difference"}:
        raise ValueError("effect_measure must be 'hedges_g' or 'mean_difference'.")
    if config.model_type not in {"fixed", "random"}:
        raise ValueError("model_type must be either 'fixed' or 'random'.")


def _load_study_effects(
    data_path: str | Path,
    effect_measure: EffectMeasure,
) -> tuple[pd.DataFrame, list[EffectSize]]:
    """Load continuous data from CSV and calculate one effect estimate per study."""
    path = Path(data_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Input data must be a CSV file.")
    dataframe = pd.read_csv(path)
    missing_columns = sorted(_REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing_columns)
        )
    calculator = hedges_g if effect_measure == "hedges_g" else mean_difference
    effect_sizes = [
        calculator(
            row.mean_treatment,
            row.sd_treatment,
            int(row.n_treatment),
            row.mean_control,
            row.sd_control,
            int(row.n_control),
        )
        for row in dataframe.itertuples(index=False)
    ]
    return dataframe, effect_sizes


def _fit_model(
    effect_sizes: list[EffectSize],
    config: AnalysisConfig,
) -> MetaAnalysisResult:
    """Fit the requested overall model using precomputed effect sizes."""
    if config.model_type == "fixed":
        return fixed_effect(effect_sizes)
    return random_effects(effect_sizes, tau_method=config.tau_method)


def _markdown_table(dataframe: pd.DataFrame) -> str:
    """Render a DataFrame as a simple dependency-free GitHub-flavored table."""
    columns = [str(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in dataframe.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _summary_table(model_result: MetaAnalysisResult) -> pd.DataFrame:
    """Build a one-row model summary table for report and Excel output."""
    return pd.DataFrame(
        [
            {
                "pooled_effect": model_result.pooled_effect,
                "ci_lower": model_result.ci_lower,
                "ci_upper": model_result.ci_upper,
                "p_value": model_result.p_value,
                "i_squared": model_result.i_squared,
                "q_statistic": model_result.q_statistic,
                "q_p_value": model_result.q_p_value,
                "tau_squared": model_result.tau_squared,
            }
        ]
    )


def _subgroup_table(result: SubgroupAnalysisResult) -> pd.DataFrame:
    """Build a tidy subgroup-result table for Markdown and Excel output."""
    rows = []
    for group, model in result.group_results.items():
        rows.append(
            {
                "group": group,
                "study_count": result.group_sizes[group],
                "pooled_effect": model.pooled_effect,
                "ci_lower": model.ci_lower,
                "ci_upper": model.ci_upper,
                "p_value": model.p_value,
                "i_squared": model.i_squared,
                "tau_squared": model.tau_squared,
            }
        )
    return pd.DataFrame(rows)


def _write_excel(
    excel_path: Path,
    summary: pd.DataFrame,
    studies: pd.DataFrame,
    sensitivity: pd.DataFrame | None,
    subgroup: pd.DataFrame | None,
) -> None:
    """Write report result tables to a multi-sheet Excel workbook."""
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        studies.to_excel(writer, sheet_name="Study effects", index=False)
        if sensitivity is not None:
            sensitivity.to_excel(writer, sheet_name="Leave-one-out", index=False)
        if subgroup is not None:
            subgroup.to_excel(writer, sheet_name="Subgroups", index=False)


def generate_report(
    data_path: str | Path,
    output_path: str | Path,
    *,
    config: AnalysisConfig | None = None,
) -> ReportResult:
    """Generate a complete Markdown and Excel meta-analysis report from CSV data.

    The pipeline calculates continuous-outcome effects, fits the selected model,
    creates forest and funnel plots, runs Egger's test and leave-one-out analysis,
    and optionally performs categorical subgroup analysis.

    Args:
        data_path: CSV path containing continuous two-arm study data.
        output_path: Destination Markdown report path.
        config: Optional analysis configuration. Defaults to a random-effects
            Hedges' g analysis with funnel and sensitivity diagnostics.

    Returns:
        Paths to the Markdown report, Excel workbook, and generated figures, plus
        the overall and optional subgroup model results.

    Raises:
        ValueError: If configuration, CSV schema, or subgroup settings are invalid.
    """
    analysis_config = config or AnalysisConfig()
    _validate_config(analysis_config)
    markdown_path = Path(output_path)
    if markdown_path.suffix.lower() != ".md":
        raise ValueError("Report output path must use a .md suffix.")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe, effect_sizes = _load_study_effects(
        data_path,
        analysis_config.effect_measure,
    )
    model_result = _fit_model(effect_sizes, analysis_config)
    asset_directory = markdown_path.parent / f"{markdown_path.stem}_assets"
    asset_directory.mkdir(parents=True, exist_ok=True)

    effect_values = [effect.effect for effect in effect_sizes]
    standard_errors = [effect.standard_error for effect in effect_sizes]
    study_labels = dataframe["study"].astype(str).tolist()
    effect_label = (
        "Hedges' g"
        if analysis_config.effect_measure == "hedges_g"
        else "Mean difference"
    )
    forest_path = asset_directory / "forest_plot.png"
    forest_figure, _ = forest_plot(
        effect_values,
        standard_errors,
        study_labels,
        model_result,
        title="Meta-analysis forest plot",
        effect_label=effect_label,
        output_path=forest_path,
    )
    forest_figure.clear()
    asset_paths: dict[str, Path] = {"forest_plot": forest_path}

    egger_p_value: float | None = None
    if analysis_config.include_funnel:
        egger_p_value = egger_test(effect_sizes).p_value
        funnel_path = asset_directory / "funnel_plot.png"
        funnel_figure, _ = funnel_plot(
            effect_values,
            standard_errors,
            pooled_effect=model_result.pooled_effect,
            egger_p_value=egger_p_value,
            title="Meta-analysis funnel plot",
            effect_label=effect_label,
            output_path=funnel_path,
        )
        funnel_figure.clear()
        asset_paths["funnel_plot"] = funnel_path

    sensitivity_table: pd.DataFrame | None = None
    if analysis_config.include_sensitivity:
        sensitivity_path = asset_directory / "leave_one_out.png"
        sensitivity_result = leave_one_out(
            effect_sizes,
            labels=study_labels,
            tau_method=analysis_config.tau_method,
            output_path=sensitivity_path,
        )
        sensitivity_table = sensitivity_result.table
        sensitivity_result.figure.clear()
        asset_paths["leave_one_out"] = sensitivity_path

    subgroup_result: SubgroupAnalysisResult | None = None
    subgroup_table: pd.DataFrame | None = None
    if analysis_config.subgroup_column is not None:
        if analysis_config.subgroup_column not in dataframe.columns:
            raise ValueError(
                f"Subgroup column not found: {analysis_config.subgroup_column}"
            )
        subgroup_result = subgroup_analysis(
            effect_sizes,
            dataframe[analysis_config.subgroup_column].tolist(),
            model_type=analysis_config.model_type,
            tau_method=analysis_config.tau_method,
        )
        subgroup_table = _subgroup_table(subgroup_result)

    studies = dataframe.copy()
    studies["effect"] = effect_values
    studies["standard_error"] = standard_errors
    studies["variance"] = [effect.variance for effect in effect_sizes]
    summary = _summary_table(model_result)
    excel_path = markdown_path.with_suffix(".xlsx")
    _write_excel(excel_path, summary, studies, sensitivity_table, subgroup_table)

    if analysis_config.model_type == "fixed":
        method_name = "inverse-variance fixed-effect model"
    else:
        tau_method_name = analysis_config.tau_method.upper()
        method_name = f"random-effects model with {tau_method_name} tau² estimation"
    significance = (
        "statistically significant"
        if model_result.p_value < 0.05
        else "not statistically significant"
    )
    relative_assets = {
        name: path.relative_to(markdown_path.parent).as_posix()
        for name, path in asset_paths.items()
    }
    lines = [
        "# Meta-analysis report",
        "",
        "## Methods",
        "",
        (
            f"Study-level {effect_label} estimates were calculated from group means, "
            f"standard deviations, and sample sizes. Results were pooled using an "
            f"{method_name}."
        ),
        "",
        "## Results",
        "",
        _markdown_table(summary.round(4)),
        "",
        "## Forest plot",
        "",
        f"![Forest plot]({relative_assets['forest_plot']})",
    ]
    if analysis_config.include_funnel:
        lines.extend(
            [
                "",
                "## Funnel plot and small-study effects",
                "",
                f"Egger's regression test p value was {egger_p_value:.4f}.",
                "",
                f"![Funnel plot]({relative_assets['funnel_plot']})",
            ]
        )
    if sensitivity_table is not None:
        lines.extend(
            [
                "",
                "## Sensitivity analysis",
                "",
                _markdown_table(sensitivity_table.round(4)),
                "",
                f"![Leave-one-out forest plot]({relative_assets['leave_one_out']})",
            ]
        )
    if subgroup_result is not None and subgroup_table is not None:
        lines.extend(
            [
                "",
                "## Subgroup analysis",
                "",
                _markdown_table(subgroup_table.round(4)),
                "",
                (
                    f"The between-group heterogeneity test was Q = "
                    f"{subgroup_result.q_between:.4f}, p = "
                    f"{subgroup_result.q_between_p_value:.4f}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"The pooled {effect_label} was {model_result.pooled_effect:.4f} "
                f"(95% CI {model_result.ci_lower:.4f} to {model_result.ci_upper:.4f}; "
                f"p = {model_result.p_value:.4f}) and was {significance}."
            ),
            "",
            f"The accompanying Excel workbook is `{excel_path.name}`.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return ReportResult(
        markdown_path=markdown_path,
        excel_path=excel_path,
        asset_paths=asset_paths,
        model_result=model_result,
        subgroup_result=subgroup_result,
    )
