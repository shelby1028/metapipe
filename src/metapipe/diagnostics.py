"""Small-study, sensitivity, and influence diagnostics for meta-analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from scipy.stats import kendalltau, t

from metapipe.effects import EffectSize
from metapipe.models import MetaAnalysisResult, TauMethod, random_effects


@dataclass(frozen=True)
class EggerResult:
    """Result of Egger's linear-regression asymmetry test.

    Attributes:
        intercept: Estimated intercept from standard normal deviate regression.
        ci_lower: Lower endpoint of the two-sided 95% intercept interval.
        ci_upper: Upper endpoint of the two-sided 95% intercept interval.
        p_value: Two-sided t-test p value for the intercept.
        slope: Estimated coefficient for study precision.
    """

    intercept: float
    ci_lower: float
    ci_upper: float
    p_value: float
    slope: float


@dataclass(frozen=True)
class BeggResult:
    """Result of Begg's rank-correlation asymmetry test.

    Attributes:
        kendall_tau: Kendall rank correlation statistic.
        p_value: Two-sided p value for the rank correlation.
    """

    kendall_tau: float
    p_value: float


@dataclass
class LeaveOneOutResult:
    """Output from leave-one-out random-effects sensitivity analysis.

    Attributes:
        table: One row per omitted study and its refitted model summary.
        full_model: Random-effects fit using all studies.
        figure: Forest-style figure of leave-one-out pooled effects.
        axes: Axes used to draw the sensitivity forest plot.
    """

    table: pd.DataFrame
    full_model: MetaAnalysisResult
    figure: Figure
    axes: Axes


@dataclass(frozen=True)
class OutlierResult:
    """Residual- and influence-based study-level outlier diagnostics.

    Attributes:
        table: Residuals, leave-one-out shifts, Cook-style distances, and flags.
        standardized_residual_threshold: Absolute threshold used for residuals.
        cooks_distance_threshold: Threshold used for Cook-style distances.
    """

    table: pd.DataFrame
    standardized_residual_threshold: float
    cooks_distance_threshold: float


def _as_arrays(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert numeric or EffectSize inputs to validated one-dimensional arrays."""
    if len(effects) < 3:
        raise ValueError("At least three studies are required for diagnostics.")
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


def _labels(labels: Sequence[str] | None, count: int) -> list[str]:
    """Return supplied labels or stable sequential study labels."""
    if labels is None:
        return [f"Study {index}" for index in range(1, count + 1)]
    if len(labels) != count:
        raise ValueError("Labels must have the same length as effects.")
    return [str(label) for label in labels]


def egger_test(
    effects: Sequence[float] | Sequence[EffectSize],
    standard_errors: Sequence[float] | None = None,
) -> EggerResult:
    """Run Egger's linear-regression test for funnel-plot asymmetry.

    The regression models each study's standard normal deviate (effect divided by
    standard error) as a function of precision (one divided by standard error).
    The intercept and its two-sided 95% t interval are reported.

    Args:
        effects: Study-level estimates or EffectSize objects.
        standard_errors: Standard errors for numeric effects. Omit when passing
            EffectSize objects.

    Returns:
        The regression intercept, 95% confidence interval, p value, and slope.

    Raises:
        ValueError: If fewer than three studies, non-positive standard errors, or
            identical precisions are supplied.
    """
    if len(effects) < 3:
        raise ValueError("At least three studies are required for Egger's test.")
    first = effects[0]
    if isinstance(first, EffectSize):
        if standard_errors is not None:
            raise ValueError("Do not provide standard errors with EffectSize objects.")
        effect_array = np.asarray([effect.effect for effect in effects], dtype=float)
        standard_error_array = np.asarray(
            [effect.standard_error for effect in effects], dtype=float
        )
    else:
        if standard_errors is None:
            raise ValueError("Standard errors are required with numeric effects.")
        effect_array = np.asarray(effects, dtype=float)
        standard_error_array = np.asarray(standard_errors, dtype=float)
    if effect_array.size != standard_error_array.size:
        raise ValueError("Effects and standard errors must have equal length.")
    if not np.all(np.isfinite(effect_array)):
        raise ValueError("Effects must be finite.")
    if not np.all(np.isfinite(standard_error_array)) or np.any(
        standard_error_array <= 0
    ):
        raise ValueError("Standard errors must be finite and greater than zero.")

    precision = 1.0 / standard_error_array
    if np.ptp(precision) == 0:
        raise ValueError("Egger's test requires variation in study precision.")
    standard_normal_deviate = effect_array / standard_error_array
    fitted = sm.OLS(standard_normal_deviate, sm.add_constant(precision)).fit()
    degrees_of_freedom = int(fitted.df_resid)
    critical_value = float(t.ppf(0.975, degrees_of_freedom))
    intercept = float(fitted.params[0])
    standard_error = float(fitted.bse[0])
    return EggerResult(
        intercept=intercept,
        ci_lower=intercept - critical_value * standard_error,
        ci_upper=intercept + critical_value * standard_error,
        p_value=float(fitted.pvalues[0]),
        slope=float(fitted.params[1]),
    )


def begg_test(
    effects: Sequence[float] | Sequence[EffectSize],
    standard_errors: Sequence[float] | None = None,
) -> BeggResult:
    """Run Begg's Kendall rank-correlation test for small-study effects.

    The implementation estimates the rank association between study effects and
    their standard errors; this is the direction checked visually in a funnel
    plot.

    Args:
        effects: Study-level estimates or EffectSize objects.
        standard_errors: Standard errors for numeric effects. Omit when passing
            EffectSize objects.

    Returns:
        Kendall's tau statistic and its two-sided p value.

    Raises:
        ValueError: If inputs are invalid or contain fewer than three studies.
    """
    if isinstance(effects[0], EffectSize):
        if standard_errors is not None:
            raise ValueError("Do not provide standard errors with EffectSize objects.")
        effect_array = np.asarray([effect.effect for effect in effects], dtype=float)
        standard_error_array = np.asarray(
            [effect.standard_error for effect in effects], dtype=float
        )
        if effect_array.size < 3:
            raise ValueError("At least three studies are required for Begg's test.")
    else:
        if standard_errors is None:
            raise ValueError("Standard errors are required with numeric effects.")
        effect_array, variance_array = _as_arrays(
            effects,
            np.asarray(standard_errors, dtype=float) ** 2,
        )
        standard_error_array = np.sqrt(variance_array)
    if effect_array.size != standard_error_array.size:
        raise ValueError("Effects and standard errors must have equal length.")
    if not np.all(np.isfinite(standard_error_array)) or np.any(
        standard_error_array <= 0
    ):
        raise ValueError("Standard errors must be finite and greater than zero.")
    tau, p_value = kendalltau(effect_array, standard_error_array)
    if not np.isfinite(tau) or not np.isfinite(p_value):
        raise ValueError("Begg's test requires non-constant ranks.")
    return BeggResult(kendall_tau=float(tau), p_value=float(p_value))


def _leave_one_out_figure(
    table: pd.DataFrame,
    full_model: MetaAnalysisResult,
    output_path: str | Path | None,
    dpi: int,
) -> tuple[Figure, Axes]:
    """Draw and optionally save a forest-style leave-one-out plot."""
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")
    figure, axes = plt.subplots(figsize=(9, max(4.5, 0.55 * len(table) + 1.7)))
    y_positions = np.arange(len(table), 0, -1, dtype=float)
    estimates = table["pooled_effect"].to_numpy(dtype=float)
    lower = table["ci_lower"].to_numpy(dtype=float)
    upper = table["ci_upper"].to_numpy(dtype=float)
    axes.errorbar(
        estimates,
        y_positions,
        xerr=[estimates - lower, upper - estimates],
        fmt="o",
        color="#2563eb",
        ecolor="#334155",
        capsize=3,
    )
    pooled_y = 0.0
    diamond = Polygon(
        [
            (full_model.ci_lower, pooled_y),
            (full_model.pooled_effect, pooled_y + 0.22),
            (full_model.ci_upper, pooled_y),
            (full_model.pooled_effect, pooled_y - 0.22),
        ],
        closed=True,
        facecolor="#dc2626",
        edgecolor="#7f1d1d",
    )
    axes.add_patch(diamond)
    axes.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    axes.set_yticks(
        [*y_positions, pooled_y],
        labels=[*table["omitted_study"].tolist(), "All studies"],
    )
    axes.set_xlabel("Pooled effect after omitting one study")
    axes.set_title("Leave-one-out sensitivity analysis", fontweight="bold")
    axes.grid(axis="x", alpha=0.2)
    axes.set_ylim(-0.7, len(table) + 0.7)
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise ValueError("Output path must use a .png, .svg, or .pdf suffix.")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return figure, axes


def leave_one_out(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None = None,
    *,
    labels: Sequence[str] | None = None,
    tau_method: TauMethod = "reml",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> LeaveOneOutResult:
    """Refit a random-effects model after omitting each study in turn.

    Args:
        effects: Study-level estimates or EffectSize objects.
        variances: Sampling variances for numeric effects.
        labels: Optional labels identifying omitted studies.
        tau_method: Tau-squared estimator passed to the random-effects model.
        output_path: Optional `.png`, `.svg`, or `.pdf` sensitivity forest path.
        dpi: Resolution for raster output. The default is 300 DPI.

    Returns:
        A table of refitted summaries, full-data model result, and forest figure.

    Raises:
        ValueError: If model inputs, labels, or output settings are invalid.
    """
    effect_array, variance_array = _as_arrays(effects, variances)
    study_labels = _labels(labels, effect_array.size)
    full_model = random_effects(effect_array, variance_array, tau_method=tau_method)
    rows: list[dict[str, float | str]] = []
    for index, label in enumerate(study_labels):
        retained = np.ones(effect_array.size, dtype=bool)
        retained[index] = False
        result = random_effects(
            effect_array[retained], variance_array[retained], tau_method=tau_method
        )
        rows.append(
            {
                "omitted_study": label,
                "pooled_effect": result.pooled_effect,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "p_value": result.p_value,
                "i_squared": result.i_squared,
                "tau_squared": result.tau_squared,
                "effect_shift": result.pooled_effect - full_model.pooled_effect,
            }
        )
    table = pd.DataFrame(rows)
    figure, axes = _leave_one_out_figure(table, full_model, output_path, dpi)
    return LeaveOneOutResult(
        table=table,
        full_model=full_model,
        figure=figure,
        axes=axes,
    )


def detect_outliers(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None = None,
    *,
    labels: Sequence[str] | None = None,
    tau_method: TauMethod = "reml",
    standardized_residual_threshold: float = 1.96,
) -> OutlierResult:
    """Flag studies with extreme random-effects residuals or influence.

    A study is flagged when its absolute standardized residual exceeds the chosen
    threshold or its Cook-style distance exceeds ``4 / k``. The Cook-style
    distance is the squared full-model-standard-error-scaled shift in the pooled
    estimate after the study is omitted.

    Args:
        effects: Study-level estimates or EffectSize objects.
        variances: Sampling variances for numeric effects.
        labels: Optional labels identifying studies.
        tau_method: Tau-squared estimator passed to random-effects fitting.
        standardized_residual_threshold: Absolute residual threshold for flags.

    Returns:
        A study-level diagnostics table and both decision thresholds.

    Raises:
        ValueError: If inputs or the residual threshold are invalid.
    """
    if standardized_residual_threshold <= 0:
        raise ValueError("Standardized residual threshold must be greater than zero.")
    effect_array, variance_array = _as_arrays(effects, variances)
    study_labels = _labels(labels, effect_array.size)
    sensitivity = leave_one_out(
        effect_array,
        variance_array,
        labels=study_labels,
        tau_method=tau_method,
    )
    full_model = sensitivity.full_model
    residuals = effect_array - full_model.pooled_effect
    standard_errors = np.sqrt(variance_array + full_model.tau_squared)
    standardized_residuals = residuals / standard_errors
    full_model_standard_error = (full_model.ci_upper - full_model.ci_lower) / (
        2 * 1.95996398454
    )
    effect_shifts = sensitivity.table["effect_shift"].to_numpy(dtype=float)
    cooks_distances = (effect_shifts / full_model_standard_error) ** 2
    cooks_distance_threshold = 4 / effect_array.size
    table = pd.DataFrame(
        {
            "study": study_labels,
            "effect": effect_array,
            "standard_error": np.sqrt(variance_array),
            "residual": residuals,
            "standardized_residual": standardized_residuals,
            "leave_one_out_effect": sensitivity.table["pooled_effect"],
            "effect_shift": effect_shifts,
            "cooks_distance": cooks_distances,
        }
    )
    table["is_outlier"] = (
        np.abs(table["standardized_residual"]) > standardized_residual_threshold
    ) | (table["cooks_distance"] > cooks_distance_threshold)
    plt.close(sensitivity.figure)
    return OutlierResult(
        table=table,
        standardized_residual_threshold=standardized_residual_threshold,
        cooks_distance_threshold=cooks_distance_threshold,
    )
