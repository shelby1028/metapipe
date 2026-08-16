"""Subgroup meta-analysis, between-group tests, and subgroup forest plots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from scipy.stats import chi2

from metapipe.effects import EffectSize
from metapipe.models import MetaAnalysisResult, TauMethod, fixed_effect, random_effects

ModelType = Literal["fixed", "random"]


@dataclass(frozen=True)
class SubgroupAnalysisResult:
    """Results from a categorical subgroup meta-analysis.

    Attributes:
        group_results: Mapping from subgroup label to its fitted model result.
        overall_result: Overall fixed-effect or random-effects model result.
        q_between: Cochran's Q statistic for between-group heterogeneity.
        q_between_p_value: Upper-tail chi-squared p value for ``q_between``.
        model_type: Requested subgroup pooling model.
        group_sizes: Number of studies in each subgroup.
    """

    group_results: dict[str, MetaAnalysisResult]
    overall_result: MetaAnalysisResult
    q_between: float
    q_between_p_value: float
    model_type: ModelType
    group_sizes: dict[str, int]


def _as_arrays(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert numeric or EffectSize input to validated one-dimensional arrays."""
    if len(effects) < 4:
        raise ValueError("At least four studies are required for subgroup analysis.")
    first = effects[0]
    if isinstance(first, EffectSize):
        if variances is not None:
            raise ValueError("Do not provide variances with EffectSize objects.")
        if not all(isinstance(effect, EffectSize) for effect in effects):
            raise ValueError(
                "Effects must all be EffectSize objects or all be numbers."
            )
        effect_array = np.asarray([effect.effect for effect in effects], dtype=float)
        variance_array = np.asarray(
            [effect.variance for effect in effects], dtype=float
        )
    else:
        if variances is None:
            raise ValueError("Variances are required with numeric effects.")
        effect_array = np.asarray(effects, dtype=float)
        variance_array = np.asarray(variances, dtype=float)
    if effect_array.ndim != 1 or variance_array.ndim != 1:
        raise ValueError("Effects and variances must be one-dimensional.")
    if effect_array.size != variance_array.size:
        raise ValueError("Effects and variances must have equal length.")
    if not np.all(np.isfinite(effect_array)):
        raise ValueError("Effects must be finite.")
    if not np.all(np.isfinite(variance_array)) or np.any(variance_array <= 0):
        raise ValueError("Variances must be finite and greater than zero.")
    return effect_array, variance_array


def _ordered_groups(groups: Sequence[object]) -> tuple[list[str], list[str]]:
    """Normalize subgroup labels while preserving their first-seen order."""
    labels = [str(group) for group in groups]
    unique = list(dict.fromkeys(labels))
    if len(unique) < 2:
        raise ValueError("At least two distinct subgroups are required.")
    return labels, unique


def subgroup_analysis(
    effects: Sequence[float] | Sequence[EffectSize],
    groups: Sequence[object],
    variances: Sequence[float] | None = None,
    *,
    model_type: ModelType = "random",
    tau_method: TauMethod = "reml",
) -> SubgroupAnalysisResult:
    """Pool study effects within categorical subgroups and test group differences.

    Group-specific and overall models use the requested fixed-effect or
    random-effects method. The group-difference statistic uses the conventional
    fixed-effect Q partition: ``Q_between = Q_total - sum(Q_within)``.

    Args:
        effects: Study-level estimates or EffectSize objects.
        groups: Categorical label for each study.
        variances: Sampling variances for numeric effects.
        model_type: ``"fixed"`` or ``"random"`` pooling for reported estimates.
        tau_method: Tau-squared estimator when ``model_type="random"``.

    Returns:
        Group-level fits, the overall fit, group counts, and Q-between test.

    Raises:
        ValueError: If inputs, group sizes, or model type are invalid.
    """
    effect_array, variance_array = _as_arrays(effects, variances)
    if len(groups) != effect_array.size:
        raise ValueError("Groups must have the same length as effects.")
    group_labels, unique_groups = _ordered_groups(groups)
    if model_type not in {"fixed", "random"}:
        raise ValueError("model_type must be either 'fixed' or 'random'.")

    group_results: dict[str, MetaAnalysisResult] = {}
    group_sizes: dict[str, int] = {}
    within_q = 0.0
    for group in unique_groups:
        mask = np.asarray([label == group for label in group_labels])
        if int(np.sum(mask)) < 2:
            raise ValueError("Each subgroup must contain at least two studies.")
        fixed_result = fixed_effect(effect_array[mask], variance_array[mask])
        within_q += fixed_result.q_statistic
        if model_type == "fixed":
            group_results[group] = fixed_result
        else:
            group_results[group] = random_effects(
                effect_array[mask], variance_array[mask], tau_method=tau_method
            )
        group_sizes[group] = int(np.sum(mask))

    total_fixed = fixed_effect(effect_array, variance_array)
    q_between = max(0.0, total_fixed.q_statistic - within_q)
    q_between_p_value = float(chi2.sf(q_between, len(unique_groups) - 1))
    if model_type == "fixed":
        overall_result = total_fixed
    else:
        overall_result = random_effects(
            effect_array, variance_array, tau_method=tau_method
        )
    return SubgroupAnalysisResult(
        group_results=group_results,
        overall_result=overall_result,
        q_between=q_between,
        q_between_p_value=q_between_p_value,
        model_type=model_type,
        group_sizes=group_sizes,
    )


def subgroup_forest_plot(
    effects: Sequence[float],
    standard_errors: Sequence[float],
    groups: Sequence[object],
    study_labels: Sequence[str],
    subgroup_result: SubgroupAnalysisResult,
    *,
    title: str | None = None,
    effect_label: str = "Effect size",
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw a forest plot with colored studies and pooled diamonds by subgroup.

    Args:
        effects: Study-level estimates on a common analysis scale.
        standard_errors: Study-level standard errors on the same scale.
        groups: Categorical label for every study.
        study_labels: Label for every study.
        subgroup_result: Result returned by :func:`subgroup_analysis`.
        title: Optional chart title; Chinese and English text are accepted.
        effect_label: Label for the horizontal effect-size axis.
        output_path: Optional `.png`, `.svg`, or `.pdf` output path.
        dpi: Resolution for raster output. The default is 300 DPI.
        ax: Existing axes to draw on. A new figure and axes are created if absent.

    Returns:
        The Matplotlib figure and axes used for the subgroup forest plot.

    Raises:
        ValueError: If inputs do not align with the subgroup result or save format.
    """
    effect_array = np.asarray(effects, dtype=float)
    standard_error_array = np.asarray(standard_errors, dtype=float)
    if effect_array.ndim != 1 or standard_error_array.ndim != 1:
        raise ValueError("Effects and standard errors must be one-dimensional.")
    if effect_array.size == 0 or effect_array.size != standard_error_array.size:
        raise ValueError("Effects and standard errors must have equal non-zero length.")
    if not np.all(np.isfinite(effect_array)):
        raise ValueError("Effects must be finite.")
    if not np.all(np.isfinite(standard_error_array)) or np.any(
        standard_error_array <= 0
    ):
        raise ValueError("Standard errors must be finite and greater than zero.")
    if len(groups) != effect_array.size or len(study_labels) != effect_array.size:
        raise ValueError(
            "Groups and study labels must have the same length as effects."
        )
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    group_labels, unique_groups = _ordered_groups(groups)
    if set(unique_groups) != set(subgroup_result.group_results):
        raise ValueError("Subgroup result groups must match supplied groups.")
    if ax is None:
        figure, ax = plt.subplots(figsize=(10, max(5, 0.58 * effect_array.size + 2.5)))
    else:
        figure = ax.figure

    color_map = {
        group: plt.get_cmap("tab10")(index % 10)
        for index, group in enumerate(unique_groups)
    }
    y_position = float(effect_array.size + len(unique_groups) + 1)
    y_ticks: list[float] = []
    y_tick_labels: list[str] = []
    all_lower: list[float] = []
    all_upper: list[float] = []

    for group in unique_groups:
        group_indices = [
            index for index, label in enumerate(group_labels) if label == group
        ]
        for index in group_indices:
            ci_lower = effect_array[index] - 1.95996398454 * standard_error_array[index]
            ci_upper = effect_array[index] + 1.95996398454 * standard_error_array[index]
            ax.errorbar(
                effect_array[index],
                y_position,
                xerr=1.95996398454 * standard_error_array[index],
                fmt="o",
                color=color_map[group],
                ecolor=color_map[group],
                capsize=3,
                markersize=5,
            )
            y_ticks.append(y_position)
            y_tick_labels.append(str(study_labels[index]))
            all_lower.append(float(ci_lower))
            all_upper.append(float(ci_upper))
            y_position -= 1

        result = subgroup_result.group_results[group]
        diamond = Polygon(
            [
                (result.ci_lower, y_position),
                (result.pooled_effect, y_position + 0.23),
                (result.ci_upper, y_position),
                (result.pooled_effect, y_position - 0.23),
            ],
            closed=True,
            facecolor=color_map[group],
            edgecolor=color_map[group],
            alpha=0.9,
        )
        ax.add_patch(diamond)
        y_ticks.append(y_position)
        y_tick_labels.append(f"{group} pooled (n={subgroup_result.group_sizes[group]})")
        all_lower.append(result.ci_lower)
        all_upper.append(result.ci_upper)
        y_position -= 1.25

    overall_y = y_position
    overall = subgroup_result.overall_result
    overall_diamond = Polygon(
        [
            (overall.ci_lower, overall_y),
            (overall.pooled_effect, overall_y + 0.25),
            (overall.ci_upper, overall_y),
            (overall.pooled_effect, overall_y - 0.25),
        ],
        closed=True,
        facecolor="#dc2626",
        edgecolor="#7f1d1d",
        alpha=0.9,
    )
    ax.add_patch(overall_diamond)
    y_ticks.append(overall_y)
    y_tick_labels.append("Overall")
    all_lower.append(overall.ci_lower)
    all_upper.append(overall.ci_upper)

    x_span = max(max(all_upper) - min(all_lower), 1e-6)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlim(min(all_lower) - 0.08 * x_span, max(all_upper) + 0.08 * x_span)
    ax.set_ylim(overall_y - 0.8, effect_array.size + len(unique_groups) + 1.8)
    ax.set_yticks(y_ticks, labels=y_tick_labels)
    ax.set_xlabel(effect_label)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.01,
        0.02,
        (
            f"Q_between = {subgroup_result.q_between:.2f}; "
            f"p = {subgroup_result.q_between_p_value:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise ValueError("Output path must use a .png, .svg, or .pdf suffix.")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return figure, ax
