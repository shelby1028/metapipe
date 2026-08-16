"""Tests for small-study, sensitivity, and influence diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from metapipe.diagnostics import (
    begg_test,
    detect_outliers,
    egger_test,
    leave_one_out,
)
from metapipe.effects import hedges_g

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


def test_egger_test_returns_regression_intercept_interval_and_p_value() -> None:
    _, effects = _sample_effects()
    result = egger_test(effects)

    assert np.isfinite(result.intercept)
    assert result.ci_lower < result.intercept < result.ci_upper
    assert 0 <= result.p_value <= 1
    assert np.isfinite(result.slope)


def test_egger_numeric_input_matches_effect_size_input() -> None:
    _, effects = _sample_effects()
    numeric_result = egger_test(
        [effect.effect for effect in effects],  # type: ignore[attr-defined]
        [effect.standard_error for effect in effects],  # type: ignore[attr-defined]
    )
    object_result = egger_test(effects)

    assert numeric_result.intercept == pytest.approx(object_result.intercept)
    assert numeric_result.p_value == pytest.approx(object_result.p_value)


def test_begg_test_matches_scipy_kendall_tau() -> None:
    _, effects = _sample_effects()
    effect_values = np.asarray(
        [effect.effect for effect in effects], dtype=float  # type: ignore[attr-defined]
    )
    standard_errors = np.asarray(
        [effect.standard_error for effect in effects], dtype=float  # type: ignore[attr-defined]
    )
    expected_tau, expected_p_value = kendalltau(effect_values, standard_errors)
    result = begg_test(effect_values, standard_errors)

    assert result.kendall_tau == pytest.approx(expected_tau)
    assert result.p_value == pytest.approx(expected_p_value)


def test_leave_one_out_returns_study_table_and_forest_plot(tmp_path: Path) -> None:
    dataframe, effects = _sample_effects()
    output = tmp_path / "leave_one_out.png"

    result = leave_one_out(
        effects,
        labels=dataframe["study"].tolist(),
        tau_method="reml",
        output_path=output,
    )

    assert len(result.table) == len(effects)
    assert set(result.table.columns) >= {
        "omitted_study",
        "pooled_effect",
        "ci_lower",
        "ci_upper",
        "effect_shift",
    }
    assert result.table["pooled_effect"].notna().all()
    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(result.figure)


def test_outlier_detection_flags_extreme_effect_and_has_influence_columns() -> None:
    effects = [0.15, 0.25, 0.30, 0.20, 2.40]
    variances = [0.04, 0.03, 0.04, 0.03, 0.04]

    result = detect_outliers(
        effects,
        variances,
        labels=["A", "B", "C", "D", "Extreme"],
        tau_method="dl",
    )

    assert set(result.table.columns) >= {
        "standardized_residual",
        "cooks_distance",
        "is_outlier",
    }
    assert result.table.loc[result.table["study"] == "Extreme", "is_outlier"].item()
    assert result.cooks_distance_threshold == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (egger_test, ([0.1, 0.2], [0.1, 0.2])),
        (begg_test, ([0.1, 0.2], [0.1, 0.2])),
    ],
)
def test_small_study_tests_require_at_least_three_studies(
    function: object, arguments: tuple[object, object]
) -> None:
    with pytest.raises(ValueError, match="At least three"):
        function(*arguments)  # type: ignore[operator]


def test_diagnostics_validate_labels_and_thresholds() -> None:
    with pytest.raises(ValueError, match="Labels"):
        leave_one_out([0.1, 0.2, 0.3], [0.01, 0.01, 0.01], labels=["A"])
    with pytest.raises(ValueError, match="greater than zero"):
        detect_outliers(
            [0.1, 0.2, 0.3],
            [0.01, 0.01, 0.01],
            standardized_residual_threshold=0,
        )
