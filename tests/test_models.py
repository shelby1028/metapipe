"""Tests for fixed-effect and random-effects meta-analysis models."""

from math import exp, log, sqrt

import pytest

from metapipe.effects import mean_difference, odds_ratio
from metapipe.models import fixed_effect, mantel_haenszel_odds_ratio, random_effects


def test_fixed_effect_inverse_variance_matches_hand_calculation() -> None:
    result = fixed_effect([0.2, 0.4, 0.6], [0.04, 0.01, 0.04])

    assert result.pooled_effect == pytest.approx(0.4)
    assert result.ci_lower == pytest.approx(0.4 - 1.95996398454 / sqrt(150))
    assert result.ci_upper == pytest.approx(0.4 + 1.95996398454 / sqrt(150))
    assert result.q_statistic == pytest.approx(2.0)
    assert result.i_squared == pytest.approx(0.0)
    assert result.tau_squared == pytest.approx(0.0)
    assert result.weights == pytest.approx((1 / 6, 2 / 3, 1 / 6))
    assert sum(result.weights) == pytest.approx(1.0)


def test_fixed_effect_accepts_continuous_effect_size_objects() -> None:
    first = mean_difference(10, 2, 50, 8, 3, 50)
    second = mean_difference(12, 2, 50, 10, 3, 50)

    result = fixed_effect([first, second])

    assert result.pooled_effect == pytest.approx(2.0)
    assert result.tau_squared == pytest.approx(0.0)
    assert len(result.weights) == 2


def test_fixed_effect_accepts_log_binary_effects() -> None:
    first = odds_ratio(20, 80, 10, 90, log_scale=True)
    second = odds_ratio(30, 70, 15, 85, log_scale=True)

    result = fixed_effect([first, second])

    assert result.pooled_effect > 0
    assert exp(result.pooled_effect) > 1
    assert result.ci_lower < result.pooled_effect < result.ci_upper


def test_mantel_haenszel_odds_ratio_matches_stratum_formula() -> None:
    # Sum(a*d/n) = 4.75 + 9; sum(b*c/n) = 2.25 + 4.
    result = mantel_haenszel_odds_ratio([[10, 90, 5, 95], [20, 80, 10, 90]])

    assert result.pooled_effect == pytest.approx(13.75 / 6.25)
    assert result.tau_squared == pytest.approx(0.0)
    assert result.ci_lower < result.pooled_effect < result.ci_upper
    assert result.p_value < 0.05
    assert sum(result.weights) == pytest.approx(1.0)


def test_mantel_haenszel_handles_zero_cells() -> None:
    result = mantel_haenszel_odds_ratio([[0, 10, 5, 5], [5, 5, 1, 9]])

    assert result.pooled_effect > 0
    assert result.ci_lower > 0
    assert result.ci_upper > result.ci_lower


@pytest.mark.parametrize("tau_method", ["dl", "reml", "pm"])
def test_random_effects_estimators_match_equal_variance_solution(
    tau_method: str,
) -> None:
    # Q = 18, df = 2, and all three estimators yield tau² = (18 - 2) / 50.
    result = random_effects([0.2, 0.8, 1.4], [0.04, 0.04, 0.04], tau_method=tau_method)  # type: ignore[arg-type]

    assert result.pooled_effect == pytest.approx(0.8)
    assert result.tau_squared == pytest.approx(0.32)
    assert result.q_statistic == pytest.approx(18.0)
    assert result.i_squared == pytest.approx(100 * 16 / 18)
    assert result.q_p_value < 0.001
    assert result.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


@pytest.mark.parametrize("tau_method", ["dl", "reml", "pm"])
def test_random_effects_returns_zero_tau_when_heterogeneity_is_low(
    tau_method: str,
) -> None:
    result = random_effects([0.3, 0.4, 0.5], [0.04, 0.04, 0.04], tau_method=tau_method)  # type: ignore[arg-type]

    assert result.tau_squared == pytest.approx(0.0)
    assert result.pooled_effect == pytest.approx(0.4)


def test_invalid_model_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="At least two"):
        fixed_effect([0.1], [0.01])
    with pytest.raises(ValueError, match="equal length"):
        fixed_effect([0.1, 0.2], [0.01])
    with pytest.raises(ValueError, match="greater than zero"):
        fixed_effect([0.1, 0.2], [0.01, 0.0])
    with pytest.raises(ValueError, match="tau_method"):
        random_effects([0.1, 0.2], [0.01, 0.01], tau_method="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Tables"):
        mantel_haenszel_odds_ratio([[1, 2, 3, 4]])


def test_random_effects_supports_numeric_binary_log_odds_ratios() -> None:
    first = odds_ratio(20, 80, 10, 90, log_scale=True)
    second = odds_ratio(30, 70, 15, 85, log_scale=True)

    result = random_effects(
        [first.effect, second.effect],
        [first.variance, second.variance],
        tau_method="dl",
    )

    assert result.pooled_effect == pytest.approx(log(exp(result.pooled_effect)))
    assert exp(result.pooled_effect) > 1
