"""Tests for standard continuous and binary effect-size calculations."""

from collections.abc import Callable
from math import pi, sqrt

import pytest

from metapipe.effects import (
    cohens_d,
    hedges_correction,
    hedges_g,
    mean_difference,
    odds_ratio,
    odds_ratio_to_smd,
    risk_difference,
    risk_ratio,
    smd_to_odds_ratio,
)


def test_mean_difference_matches_standard_formula() -> None:
    result = mean_difference(10, 2, 50, 8, 3, 50)

    assert result.measure == "MD"
    assert result.effect == pytest.approx(2.0)
    assert result.variance == pytest.approx(0.26)
    assert result.standard_error == pytest.approx(sqrt(0.26))


def test_cohens_d_matches_pooled_standard_deviation_formula() -> None:
    result = cohens_d(10, 2, 20, 8, 2, 20)

    assert result.measure == "Cohen's d"
    assert result.effect == pytest.approx(1.0)
    assert result.variance == pytest.approx(0.1125)
    assert result.standard_error == pytest.approx(sqrt(0.1125))


def test_hedges_g_applies_exact_small_sample_correction() -> None:
    # Hedges' gamma-function correction with 18 degrees of freedom.
    correction = hedges_correction(18)
    result = hedges_g(11, 4, 10, 9, 4, 10)

    assert correction == pytest.approx(0.957646427, rel=1e-8)
    assert result.effect == pytest.approx(0.5 * correction)
    assert result.variance == pytest.approx(correction**2 * 0.20625)


def test_odds_ratio_and_log_standard_error_match_2x2_formula() -> None:
    natural = odds_ratio(20, 80, 10, 90)
    logged = odds_ratio(20, 80, 10, 90, log_scale=True)
    log_variance = 1 / 20 + 1 / 80 + 1 / 10 + 1 / 90

    assert natural.effect == pytest.approx(2.25)
    assert natural.standard_error == pytest.approx(2.25 * sqrt(log_variance))
    assert logged.effect == pytest.approx(0.8109302162)
    assert logged.standard_error == pytest.approx(sqrt(log_variance))


def test_risk_ratio_and_log_standard_error_match_2x2_formula() -> None:
    natural = risk_ratio(20, 80, 10, 90)
    logged = risk_ratio(20, 80, 10, 90, log_scale=True)

    assert natural.effect == pytest.approx(2.0)
    assert logged.effect == pytest.approx(0.6931471806)
    assert logged.variance == pytest.approx(0.13)
    assert natural.standard_error == pytest.approx(2 * sqrt(0.13))


def test_risk_difference_matches_binomial_variance_formula() -> None:
    result = risk_difference(20, 80, 10, 90)

    assert result.effect == pytest.approx(0.1)
    assert result.standard_error == pytest.approx(0.05)
    assert result.variance == pytest.approx(0.0025)


def test_zero_cell_uses_half_continuity_correction() -> None:
    result = odds_ratio(0, 10, 5, 5)

    assert result.effect == pytest.approx((0.5 * 5.5) / (10.5 * 5.5))


def test_odds_ratio_and_smd_conversions_are_inverses() -> None:
    converted_smd = odds_ratio_to_smd(2.0)

    assert converted_smd == pytest.approx(sqrt(3) * 0.6931471805599453 / pi)
    assert smd_to_odds_ratio(converted_smd) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("calculator", "arguments"),
    [
        (mean_difference, (1, 1, 1, 0, 1, 10)),
        (cohens_d, (1, 0, 10, 0, 0, 10)),
        (odds_ratio_to_smd, (0,)),
        (smd_to_odds_ratio, (float("nan"),)),
    ],
)
def test_invalid_inputs_raise_value_error(
    calculator: Callable[..., object], arguments: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError):
        calculator(*arguments)
