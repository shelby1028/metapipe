"""Tests for mixed-effects meta-regression."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metapipe.effects import hedges_g
from metapipe.meta_regression import meta_regression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "examples" / "sample_data.csv"


def _sample_effects() -> tuple[pd.DataFrame, list[object]]:
    dataframe = pd.read_csv(SAMPLE_DATA)
    effects = [
        hedges_g(
            row.mean_treatment,
            row.sd_treatment,
            int(row.n_treatment),
            row.mean_control,
            row.sd_control,
            int(row.n_control),
        )
        for row in dataframe.itertuples(index=False)
    ]
    return dataframe, effects


def test_wls_meta_regression_recovers_linear_continuous_moderator() -> None:
    moderator = np.arange(6, dtype=float)
    effects = 0.2 + 0.1 * moderator
    variances = np.repeat(0.01, 6)

    result = meta_regression(effects, {"duration": moderator}, variances, method="wls")
    coefficients = result.coefficients.set_index("term")

    assert coefficients.loc["Intercept", "coefficient"] == pytest.approx(0.2)
    assert coefficients.loc["duration", "coefficient"] == pytest.approx(0.1)
    assert result.tau_squared == pytest.approx(0.0)
    assert result.r_squared >= 0
    assert len(result.fitted_values) == 6
    assert np.max(np.abs(result.residuals)) < 1e-10


def test_meta_regression_encodes_categorical_and_continuous_moderators() -> None:
    dataframe, effects = _sample_effects()
    result = meta_regression(
        effects,
        dataframe[["duration", "study_type"]],
        method="wls",
    )

    assert "duration" in result.moderator_columns
    assert any(column.startswith("study_type_") for column in result.moderator_columns)
    assert set(result.coefficients.columns) == {
        "term",
        "coefficient",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_value",
    }
    assert result.coefficients["p_value"].between(0, 1).all()
    assert 0 <= result.r_squared <= 1
    assert result.residual_q >= 0


def test_ml_meta_regression_returns_finite_fit_statistics() -> None:
    effects = [0.12, 0.31, 0.27, 0.62, 0.48, 0.75]
    variances = [0.03, 0.02, 0.04, 0.03, 0.02, 0.04]
    moderators = pd.DataFrame(
        {"duration": [8, 10, 12, 16, 20, 24], "design": ["A", "A", "B", "B", "B", "A"]}
    )

    result = meta_regression(effects, moderators, variances, method="ml")

    assert result.estimation_method == "ml"
    assert result.tau_squared >= 0
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert 0 <= result.residual_q_p_value <= 1


def test_effect_size_input_matches_numeric_input() -> None:
    _, effects = _sample_effects()
    moderators = {"duration": [8, 10, 12, 16, 20, 24, 8, 24, 16, 20, 12, 24]}
    object_result = meta_regression(effects, moderators)
    numeric_result = meta_regression(
        [effect.effect for effect in effects],  # type: ignore[attr-defined]
        moderators,
        [effect.variance for effect in effects],  # type: ignore[attr-defined]
    )

    assert object_result.coefficients["coefficient"].tolist() == pytest.approx(
        numeric_result.coefficients["coefficient"].tolist()
    )


def test_meta_regression_validates_design_and_method() -> None:
    with pytest.raises(ValueError, match="At least one moderator"):
        meta_regression([0.1, 0.2, 0.3], {}, [0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="same number of rows"):
        meta_regression([0.1, 0.2, 0.3], {"x": [1, 2]}, [0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="rank deficient"):
        meta_regression(
            [0.1, 0.2, 0.3, 0.4],
            {"x": [1, 1, 1, 1]},
            [0.01] * 4,
        )
    with pytest.raises(ValueError, match="method"):
        meta_regression(
            [0.1, 0.2, 0.3, 0.4],
            {"x": [1, 2, 3, 4]},
            [0.01] * 4,
            method="invalid",  # type: ignore[arg-type]
        )
