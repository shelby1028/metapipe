"""Tests for categorical subgroup meta-analysis and subgroup forest plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from metapipe.effects import hedges_g
from metapipe.models import fixed_effect
from metapipe.subgroup import subgroup_analysis, subgroup_forest_plot

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


def test_random_subgroup_analysis_returns_each_group_and_between_group_test() -> None:
    dataframe, effects = _sample_effects()
    result = subgroup_analysis(
        effects,
        dataframe["study_type"].tolist(),
        model_type="random",
        tau_method="reml",
    )

    assert set(result.group_results) == {"RCT", "Quasi-experimental"}
    assert result.group_sizes == {"RCT": 9, "Quasi-experimental": 3}
    assert result.model_type == "random"
    assert result.q_between >= 0
    assert 0 <= result.q_between_p_value <= 1
    assert result.overall_result.ci_lower < result.overall_result.pooled_effect
    assert result.overall_result.pooled_effect < result.overall_result.ci_upper


def test_between_group_q_matches_fixed_effect_q_partition() -> None:
    effects = [0.1, 0.2, 0.9, 1.0]
    variances = [0.04, 0.04, 0.04, 0.04]
    groups = ["short", "short", "long", "long"]

    result = subgroup_analysis(effects, groups, variances, model_type="fixed")
    expected = fixed_effect(effects, variances).q_statistic
    expected -= fixed_effect(effects[:2], variances[:2]).q_statistic
    expected -= fixed_effect(effects[2:], variances[2:]).q_statistic

    assert result.q_between == pytest.approx(expected)
    assert result.overall_result.pooled_effect == pytest.approx(0.55)
    assert result.group_results["short"].pooled_effect == pytest.approx(0.15)
    assert result.group_results["long"].pooled_effect == pytest.approx(0.95)


def test_subgroup_forest_plot_saves_colored_group_output(tmp_path: Path) -> None:
    dataframe, effects = _sample_effects()
    result = subgroup_analysis(
        effects,
        dataframe["study_type"].tolist(),
        model_type="random",
    )
    output = tmp_path / "subgroup_forest.png"
    figure, axis = subgroup_forest_plot(
        [effect.effect for effect in effects],  # type: ignore[attr-defined]
        [effect.standard_error for effect in effects],  # type: ignore[attr-defined]
        dataframe["study_type"].tolist(),
        dataframe["study"].tolist(),
        result,
        title="亚组森林图 / Subgroup forest plot",
        output_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert "Q_between" in " ".join(text.get_text() for text in axis.texts)
    assert len(axis.patches) == 3
    plt.close(figure)


def test_subgroup_analysis_rejects_insufficient_group_size() -> None:
    with pytest.raises(ValueError, match="at least two studies"):
        subgroup_analysis(
            [0.1, 0.2, 0.3, 0.4],
            ["A", "A", "A", "B"],
            [0.01, 0.01, 0.01, 0.01],
        )


def test_subgroup_analysis_validates_group_length_and_model_type() -> None:
    with pytest.raises(ValueError, match="same length"):
        subgroup_analysis([0.1, 0.2, 0.3, 0.4], ["A", "A"], [0.01] * 4)
    with pytest.raises(ValueError, match="model_type"):
        subgroup_analysis(
            [0.1, 0.2, 0.3, 0.4],
            ["A", "A", "B", "B"],
            [0.01] * 4,
            model_type="unknown",  # type: ignore[arg-type]
        )
