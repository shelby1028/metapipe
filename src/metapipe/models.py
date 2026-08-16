"""Fixed-effect and random-effects meta-analysis models.

All generic pooling functions operate on effect estimates and their sampling
variances. This allows the same routines to pool continuous outcomes (for
example, MD, Cohen's d, or Hedges' g) and binary outcomes (for example, log OR,
log RR, or RD). Ratio measures should normally be supplied on the log scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from metapipe.effects import EffectSize, odds_ratio

TauMethod = Literal["dl", "reml", "pm"]


@dataclass(frozen=True)
class MetaAnalysisResult:
    """Summary of a fixed-effect or random-effects meta-analysis.

    Attributes:
        pooled_effect: Combined effect estimate on the input scale.
        ci_lower: Lower endpoint of a two-sided 95% confidence interval.
        ci_upper: Upper endpoint of a two-sided 95% confidence interval.
        p_value: Two-sided normal-approximation p value for the pooled effect.
        i_squared: I-squared statistic, expressed as a percentage.
        q_statistic: Cochran's Q heterogeneity statistic.
        q_p_value: Upper-tail chi-squared p value for Cochran's Q.
        tau_squared: Between-study variance; zero for fixed-effect models.
        weights: Relative study weights that sum to one.
    """

    pooled_effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    i_squared: float
    q_statistic: float
    q_p_value: float
    tau_squared: float
    weights: tuple[float, ...]


def _as_arrays(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert numerical estimates or EffectSize objects to validated arrays."""
    if len(effects) < 2:
        raise ValueError("At least two studies are required for meta-analysis.")

    first = effects[0]
    if isinstance(first, EffectSize):
        if variances is not None:
            raise ValueError(
                "Do not provide variances when passing EffectSize objects."
            )
        if not all(isinstance(item, EffectSize) for item in effects):
            raise ValueError(
                "Effects must all be EffectSize objects or all be numbers."
            )
        effect_array = np.asarray([item.effect for item in effects], dtype=float)
        variance_array = np.asarray([item.variance for item in effects], dtype=float)
    else:
        if variances is None:
            raise ValueError("Variances are required when effects are numeric.")
        effect_array = np.asarray(effects, dtype=float)
        variance_array = np.asarray(variances, dtype=float)

    if effect_array.ndim != 1 or variance_array.ndim != 1:
        raise ValueError("Effects and variances must be one-dimensional sequences.")
    if effect_array.size != variance_array.size:
        raise ValueError("Effects and variances must have equal length.")
    if not np.all(np.isfinite(effect_array)):
        raise ValueError("Effects must be finite.")
    if not np.all(np.isfinite(variance_array)) or np.any(variance_array <= 0):
        raise ValueError("Variances must be finite and greater than zero.")
    return effect_array, variance_array


def _heterogeneity(
    effects: np.ndarray, variances: np.ndarray
) -> tuple[float, float, float]:
    """Calculate Cochran's Q, its p value, and I-squared from fixed weights."""
    weights = 1.0 / variances
    pooled = float(np.sum(weights * effects) / np.sum(weights))
    q_statistic = float(np.sum(weights * (effects - pooled) ** 2))
    degrees_of_freedom = effects.size - 1
    q_p_value = float(chi2.sf(q_statistic, degrees_of_freedom))
    if q_statistic == 0:
        i_squared = 0.0
    else:
        i_squared = max(0.0, (q_statistic - degrees_of_freedom) / q_statistic) * 100
    return q_statistic, q_p_value, i_squared


def _summary(
    effects: np.ndarray,
    variances: np.ndarray,
    tau_squared: float,
) -> MetaAnalysisResult:
    """Construct a pooled-analysis summary using the supplied tau-squared."""
    weights = 1.0 / (variances + tau_squared)
    weight_sum = float(np.sum(weights))
    pooled_effect = float(np.sum(weights * effects) / weight_sum)
    standard_error = sqrt(1.0 / weight_sum)
    z_score = pooled_effect / standard_error
    critical_value = norm.ppf(0.975)
    q_statistic, q_p_value, i_squared = _heterogeneity(effects, variances)
    relative_weights = tuple(float(weight / weight_sum) for weight in weights)
    return MetaAnalysisResult(
        pooled_effect=pooled_effect,
        ci_lower=float(pooled_effect - critical_value * standard_error),
        ci_upper=float(pooled_effect + critical_value * standard_error),
        p_value=float(2 * norm.sf(abs(z_score))),
        i_squared=i_squared,
        q_statistic=q_statistic,
        q_p_value=q_p_value,
        tau_squared=float(tau_squared),
        weights=relative_weights,
    )


def fixed_effect(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None = None,
) -> MetaAnalysisResult:
    """Pool effect estimates using the inverse-variance fixed-effect method.

    The input can be parallel sequences of numerical effects and variances, or a
    sequence of :class:`~metapipe.effects.EffectSize` objects. The function is
    measure-agnostic, so it supports continuous and binary effects. Supply OR or
    RR estimates on the log scale for conventional inverse-variance pooling.

    Args:
        effects: Study-level estimates or EffectSize objects.
        variances: Study-level sampling variances for numerical effects.

    Returns:
        A fixed-effect summary with a two-sided 95% normal confidence interval.

    Raises:
        ValueError: If fewer than two studies or invalid variances are supplied.
    """
    effect_array, variance_array = _as_arrays(effects, variances)
    return _summary(effect_array, variance_array, tau_squared=0.0)


def _validate_tables(
    tables: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate a collection of [a, b, c, d] tables and apply correction."""
    table_array = np.asarray(tables, dtype=float)
    if table_array.ndim != 2 or table_array.shape[1] != 4 or table_array.shape[0] < 2:
        raise ValueError("Tables must contain at least two rows of [a, b, c, d].")
    if not np.all(np.isfinite(table_array)) or np.any(table_array < 0):
        raise ValueError("Table cells must be finite and non-negative.")
    if np.any(table_array.sum(axis=1) == 0):
        raise ValueError("Each 2x2 table must contain at least one participant.")
    has_zero = np.any(table_array == 0, axis=1)
    table_array[has_zero] += 0.5
    return tuple(table_array[:, index] for index in range(4))  # type: ignore[return-value]


def mantel_haenszel_odds_ratio(
    tables: Sequence[Sequence[float]],
) -> MetaAnalysisResult:
    """Pool 2x2 tables using the fixed-effect Mantel-Haenszel odds ratio.

    Each table is ordered ``[a, b, c, d]``, where ``a`` and ``b`` are treatment
    events and non-events and ``c`` and ``d`` are control events and non-events.
    Studies containing a zero cell receive the conventional 0.5 correction in
    every cell before calculation.

    Args:
        tables: At least two 2x2 tables in ``[a, b, c, d]`` order.

    Returns:
        A Mantel-Haenszel odds ratio with its 95% confidence interval and
        heterogeneity statistics. The effect and interval are on the OR scale.

    Raises:
        ValueError: If the table collection has an invalid shape or values.
    """
    a, b, c, d = _validate_tables(tables)
    totals = a + b + c + d
    r_values = a * d / totals
    s_values = b * c / totals
    r_total = float(np.sum(r_values))
    s_total = float(np.sum(s_values))
    if r_total <= 0 or s_total <= 0:
        raise ValueError("Mantel-Haenszel odds ratio is undefined for these tables.")

    pooled_log_or = log(r_total / s_total)
    p_values = (a + d) / totals
    q_values = (b + c) / totals
    log_variance = 0.5 * (
        np.sum(p_values * r_values) / r_total**2
        + (np.sum(p_values * s_values) + np.sum(q_values * r_values))
        / (r_total * s_total)
        + np.sum(q_values * s_values) / s_total**2
    )
    standard_error = sqrt(float(log_variance))
    critical_value = norm.ppf(0.975)
    z_score = pooled_log_or / standard_error

    study_tables = list(zip(a, b, c, d, strict=True))
    study_effects = np.asarray(
        [odds_ratio(*table, log_scale=True).effect for table in study_tables],
        dtype=float,
    )
    study_variances = np.asarray(
        [odds_ratio(*table, log_scale=True).variance for table in study_tables],
        dtype=float,
    )
    q_statistic, q_p_value, i_squared = _heterogeneity(study_effects, study_variances)
    contributions = r_values + s_values
    contribution_total = float(np.sum(contributions))
    relative_weights = tuple(
        float(value / contribution_total) for value in contributions
    )

    return MetaAnalysisResult(
        pooled_effect=float(exp(pooled_log_or)),
        ci_lower=float(exp(pooled_log_or - critical_value * standard_error)),
        ci_upper=float(exp(pooled_log_or + critical_value * standard_error)),
        p_value=float(2 * norm.sf(abs(z_score))),
        i_squared=i_squared,
        q_statistic=q_statistic,
        q_p_value=q_p_value,
        tau_squared=0.0,
        weights=relative_weights,
    )


def _tau_squared_dl(effects: np.ndarray, variances: np.ndarray) -> float:
    """Estimate tau-squared with the DerSimonian-Laird moment estimator."""
    weights = 1.0 / variances
    q_statistic, _, _ = _heterogeneity(effects, variances)
    degrees_of_freedom = effects.size - 1
    denominator = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    return max(0.0, (q_statistic - degrees_of_freedom) / denominator)


def _tau_squared_pm(effects: np.ndarray, variances: np.ndarray) -> float:
    """Estimate tau-squared by solving the Paule-Mandel Q equation."""
    degrees_of_freedom = effects.size - 1

    def equation(tau_squared: float) -> float:
        weights = 1.0 / (variances + tau_squared)
        pooled_effect = float(np.sum(weights * effects) / np.sum(weights))
        q_statistic = float(np.sum(weights * (effects - pooled_effect) ** 2))
        return q_statistic - degrees_of_freedom

    if equation(0.0) <= 0:
        return 0.0
    upper = max(float(np.var(effects)), float(np.max(variances)), 1.0)
    while equation(upper) > 0:
        upper *= 2
    return float(brentq(equation, 0.0, upper))


def _tau_squared_reml(effects: np.ndarray, variances: np.ndarray) -> float:
    """Estimate tau-squared by solving the REML score equation."""

    def score(tau_squared: float) -> float:
        weights = 1.0 / (variances + tau_squared)
        weight_sum = float(np.sum(weights))
        pooled_effect = float(np.sum(weights * effects) / weight_sum)
        residual_component = float(np.sum(weights**2 * (effects - pooled_effect) ** 2))
        information_component = float(weight_sum - np.sum(weights**2) / weight_sum)
        return residual_component - information_component

    if score(0.0) <= 0:
        return 0.0
    upper = max(float(np.var(effects)), float(np.max(variances)), 1.0)
    while score(upper) > 0:
        upper *= 2
    return float(brentq(score, 0.0, upper))


def random_effects(
    effects: Sequence[float] | Sequence[EffectSize],
    variances: Sequence[float] | None = None,
    *,
    tau_method: TauMethod = "dl",
) -> MetaAnalysisResult:
    """Pool effect estimates with a random-effects inverse-variance model.

    The input can be parallel sequences of numerical effects and variances, or a
    sequence of :class:`~metapipe.effects.EffectSize` objects. This supports both
    continuous and binary effect estimates, including log-scaled ratio measures.

    Args:
        effects: Study-level estimates or EffectSize objects.
        variances: Study-level sampling variances for numerical effects.
        tau_method: Between-study variance estimator: ``"dl"`` for
            DerSimonian-Laird, ``"reml"`` for restricted maximum likelihood, or
            ``"pm"`` for Paule-Mandel.

    Returns:
        A random-effects summary with heterogeneity statistics and relative
        inverse-variance weights.

    Raises:
        ValueError: If the input is invalid or ``tau_method`` is unknown.
    """
    effect_array, variance_array = _as_arrays(effects, variances)
    estimators = {
        "dl": _tau_squared_dl,
        "reml": _tau_squared_reml,
        "pm": _tau_squared_pm,
    }
    if tau_method not in estimators:
        raise ValueError("tau_method must be one of: 'dl', 'reml', or 'pm'.")
    tau_squared = estimators[tau_method](effect_array, variance_array)
    return _summary(effect_array, variance_array, tau_squared=tau_squared)
