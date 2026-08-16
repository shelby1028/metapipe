"""Effect-size calculations for two-arm meta-analysis studies.

For ratio measures, callers can choose the natural ratio scale or the log scale.
Meta-analysis weighting is normally performed on the log scale for odds ratios and
risk ratios; use ``log_scale=True`` for that representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, pi, sqrt

from scipy.special import gammaln


@dataclass(frozen=True)
class EffectSize:
    """A study-level effect estimate and its sampling uncertainty.

    Attributes:
        measure: Short label describing the effect measure.
        effect: Point estimate on the requested scale.
        standard_error: Standard error on the same scale as ``effect``.
        variance: Sampling variance on the same scale as ``effect``.
    """

    measure: str
    effect: float
    standard_error: float
    variance: float


def _result(measure: str, effect: float, variance: float) -> EffectSize:
    if variance < 0 or not isfinite(variance):
        raise ValueError("Calculated variance must be finite and non-negative.")
    return EffectSize(
        measure=measure,
        effect=float(effect),
        standard_error=sqrt(float(variance)),
        variance=float(variance),
    )


def _validate_group(mean: float, standard_deviation: float, sample_size: int) -> None:
    if not isfinite(mean):
        raise ValueError("Mean must be finite.")
    if not isfinite(standard_deviation) or standard_deviation < 0:
        raise ValueError("Standard deviation must be finite and non-negative.")
    if sample_size <= 1:
        raise ValueError("Sample size must be greater than one.")


def _validate_2x2(
    events_treatment: float,
    non_events_treatment: float,
    events_control: float,
    non_events_control: float,
) -> None:
    cells = (
        events_treatment,
        non_events_treatment,
        events_control,
        non_events_control,
    )
    if any(not isfinite(cell) or cell < 0 for cell in cells):
        raise ValueError("All 2x2 table cell counts must be finite and non-negative.")
    if events_treatment + non_events_treatment <= 0:
        raise ValueError("Treatment group must contain at least one participant.")
    if events_control + non_events_control <= 0:
        raise ValueError("Control group must contain at least one participant.")


def _correct_2x2(
    events_treatment: float,
    non_events_treatment: float,
    events_control: float,
    non_events_control: float,
    continuity_correction: float,
) -> tuple[float, float, float, float]:
    """Apply a continuity correction to every cell only when a zero is present."""
    _validate_2x2(
        events_treatment,
        non_events_treatment,
        events_control,
        non_events_control,
    )
    if continuity_correction < 0 or not isfinite(continuity_correction):
        raise ValueError("Continuity correction must be finite and non-negative.")
    cells = (
        float(events_treatment),
        float(non_events_treatment),
        float(events_control),
        float(non_events_control),
    )
    if 0 in cells:
        if continuity_correction == 0:
            raise ValueError(
                "A positive continuity correction is required when a cell is zero."
            )
        return tuple(cell + continuity_correction for cell in cells)  # type: ignore[return-value]
    return cells


def mean_difference(
    mean_treatment: float,
    sd_treatment: float,
    n_treatment: int,
    mean_control: float,
    sd_control: float,
    n_control: int,
) -> EffectSize:
    """Calculate a raw mean difference and its standard error.

    Args:
        mean_treatment: Treatment-arm sample mean.
        sd_treatment: Treatment-arm sample standard deviation.
        n_treatment: Treatment-arm sample size.
        mean_control: Control-arm sample mean.
        sd_control: Control-arm sample standard deviation.
        n_control: Control-arm sample size.

    Returns:
        The treatment-minus-control mean difference, its standard error, and
        sampling variance.

    Raises:
        ValueError: If a standard deviation or sample size is invalid.
    """
    _validate_group(mean_treatment, sd_treatment, n_treatment)
    _validate_group(mean_control, sd_control, n_control)
    effect = mean_treatment - mean_control
    variance = (sd_treatment**2 / n_treatment) + (sd_control**2 / n_control)
    return _result("MD", effect, variance)


def cohens_d(
    mean_treatment: float,
    sd_treatment: float,
    n_treatment: int,
    mean_control: float,
    sd_control: float,
    n_control: int,
) -> EffectSize:
    """Calculate Cohen's d using the pooled within-group standard deviation.

    Args:
        mean_treatment: Treatment-arm sample mean.
        sd_treatment: Treatment-arm sample standard deviation.
        n_treatment: Treatment-arm sample size.
        mean_control: Control-arm sample mean.
        sd_control: Control-arm sample standard deviation.
        n_control: Control-arm sample size.

    Returns:
        Cohen's d with its large-sample sampling variance and standard error.

    Raises:
        ValueError: If inputs are invalid or pooled variation is zero.
    """
    _validate_group(mean_treatment, sd_treatment, n_treatment)
    _validate_group(mean_control, sd_control, n_control)
    degrees_of_freedom = n_treatment + n_control - 2
    pooled_variance = (
        (n_treatment - 1) * sd_treatment**2 + (n_control - 1) * sd_control**2
    ) / degrees_of_freedom
    if pooled_variance <= 0:
        raise ValueError("Pooled standard deviation must be greater than zero.")
    effect = (mean_treatment - mean_control) / sqrt(pooled_variance)
    variance = (n_treatment + n_control) / (n_treatment * n_control) + effect**2 / (
        2 * (n_treatment + n_control)
    )
    return _result("Cohen's d", effect, variance)


def hedges_correction(degrees_of_freedom: int) -> float:
    """Calculate Hedges' exact small-sample correction factor J.

    Args:
        degrees_of_freedom: Degrees of freedom for the pooled standard deviation.

    Returns:
        The gamma-function correction factor applied to Cohen's d.

    Raises:
        ValueError: If degrees of freedom is not positive.
    """
    if degrees_of_freedom <= 0:
        raise ValueError("Degrees of freedom must be positive.")
    half_df = degrees_of_freedom / 2
    log_correction = (
        gammaln(half_df) - 0.5 * log(half_df) - gammaln((degrees_of_freedom - 1) / 2)
    )
    return exp(log_correction)


def hedges_g(
    mean_treatment: float,
    sd_treatment: float,
    n_treatment: int,
    mean_control: float,
    sd_control: float,
    n_control: int,
) -> EffectSize:
    """Calculate Hedges' g with the exact small-sample correction.

    Args:
        mean_treatment: Treatment-arm sample mean.
        sd_treatment: Treatment-arm sample standard deviation.
        n_treatment: Treatment-arm sample size.
        mean_control: Control-arm sample mean.
        sd_control: Control-arm sample standard deviation.
        n_control: Control-arm sample size.

    Returns:
        Hedges' g with its corrected sampling variance and standard error.

    Raises:
        ValueError: If inputs are invalid or pooled variation is zero.
    """
    d_result = cohens_d(
        mean_treatment,
        sd_treatment,
        n_treatment,
        mean_control,
        sd_control,
        n_control,
    )
    degrees_of_freedom = n_treatment + n_control - 2
    correction = hedges_correction(degrees_of_freedom)
    effect = correction * d_result.effect
    variance = correction**2 * d_result.variance
    return _result("Hedges' g", effect, variance)


def odds_ratio(
    events_treatment: float,
    non_events_treatment: float,
    events_control: float,
    non_events_control: float,
    *,
    continuity_correction: float = 0.5,
    log_scale: bool = False,
) -> EffectSize:
    """Calculate an odds ratio or log odds ratio from a 2x2 table.

    A continuity correction is applied to all four cells only when at least one
    cell is zero. The log scale is appropriate for inverse-variance pooling.

    Args:
        events_treatment: Events in the treatment arm.
        non_events_treatment: Non-events in the treatment arm.
        events_control: Events in the control arm.
        non_events_control: Non-events in the control arm.
        continuity_correction: Correction applied when a table contains a zero.
        log_scale: Return the log odds ratio and its standard error when true.

    Returns:
        An odds-ratio estimate and standard error on the requested scale.
    """
    a, b, c, d = _correct_2x2(
        events_treatment,
        non_events_treatment,
        events_control,
        non_events_control,
        continuity_correction,
    )
    ratio = (a * d) / (b * c)
    log_variance = (1 / a) + (1 / b) + (1 / c) + (1 / d)
    if log_scale:
        return _result("log(OR)", log(ratio), log_variance)
    variance = ratio**2 * log_variance
    return _result("OR", ratio, variance)


def risk_ratio(
    events_treatment: float,
    non_events_treatment: float,
    events_control: float,
    non_events_control: float,
    *,
    continuity_correction: float = 0.5,
    log_scale: bool = False,
) -> EffectSize:
    """Calculate a risk ratio or log risk ratio from a 2x2 table.

    A continuity correction is applied to all four cells only when at least one
    cell is zero. The log scale is appropriate for inverse-variance pooling.

    Args:
        events_treatment: Events in the treatment arm.
        non_events_treatment: Non-events in the treatment arm.
        events_control: Events in the control arm.
        non_events_control: Non-events in the control arm.
        continuity_correction: Correction applied when a table contains a zero.
        log_scale: Return the log risk ratio and its standard error when true.

    Returns:
        A risk-ratio estimate and standard error on the requested scale.
    """
    a, b, c, d = _correct_2x2(
        events_treatment,
        non_events_treatment,
        events_control,
        non_events_control,
        continuity_correction,
    )
    treatment_total = a + b
    control_total = c + d
    ratio = (a / treatment_total) / (c / control_total)
    log_variance = (1 / a) - (1 / treatment_total) + (1 / c) - (1 / control_total)
    if log_scale:
        return _result("log(RR)", log(ratio), log_variance)
    variance = ratio**2 * log_variance
    return _result("RR", ratio, variance)


def risk_difference(
    events_treatment: float,
    non_events_treatment: float,
    events_control: float,
    non_events_control: float,
) -> EffectSize:
    """Calculate the treatment-minus-control risk difference from a 2x2 table.

    Args:
        events_treatment: Events in the treatment arm.
        non_events_treatment: Non-events in the treatment arm.
        events_control: Events in the control arm.
        non_events_control: Non-events in the control arm.

    Returns:
        The risk difference with its binomial sampling variance and standard
        error.
    """
    _validate_2x2(
        events_treatment,
        non_events_treatment,
        events_control,
        non_events_control,
    )
    treatment_total = events_treatment + non_events_treatment
    control_total = events_control + non_events_control
    treatment_risk = events_treatment / treatment_total
    control_risk = events_control / control_total
    variance = (
        treatment_risk * (1 - treatment_risk) / treatment_total
        + control_risk * (1 - control_risk) / control_total
    )
    return _result("RD", treatment_risk - control_risk, variance)


def odds_ratio_to_smd(odds_ratio_value: float) -> float:
    """Convert an odds ratio to a standardized mean difference.

    Args:
        odds_ratio_value: Positive odds ratio on its natural scale.

    Returns:
        The approximate standardized mean difference using the logistic
        distribution conversion.

    Raises:
        ValueError: If the odds ratio is not finite and positive.
    """
    if not isfinite(odds_ratio_value) or odds_ratio_value <= 0:
        raise ValueError("Odds ratio must be finite and greater than zero.")
    return log(odds_ratio_value) * sqrt(3) / pi


def smd_to_odds_ratio(standardized_mean_difference: float) -> float:
    """Convert a standardized mean difference to an odds ratio.

    Args:
        standardized_mean_difference: Standardized mean difference value.

    Returns:
        The approximate odds ratio using the logistic distribution conversion.

    Raises:
        ValueError: If the standardized mean difference is not finite.
    """
    if not isfinite(standardized_mean_difference):
        raise ValueError("Standardized mean difference must be finite.")
    return exp(standardized_mean_difference * pi / sqrt(3))
