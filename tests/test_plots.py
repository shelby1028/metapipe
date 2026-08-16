"""Tests for forest, funnel, and L'Abbé plots and the plotting CLI."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from typer.testing import CliRunner

from metapipe.cli import app
from metapipe.effects import hedges_g
from metapipe.models import random_effects
from metapipe.plots import forest_plot, funnel_plot, labbe_plot

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


def test_forest_plot_saves_png_svg_and_pdf(tmp_path: Path) -> None:
    dataframe, effects = _sample_effects()
    result = random_effects(effects, tau_method="reml")

    for extension in ("png", "svg", "pdf"):
        output = tmp_path / f"forest.{extension}"
        figure, axis = forest_plot(
            [effect.effect for effect in effects],  # type: ignore[attr-defined]
            [effect.standard_error for effect in effects],  # type: ignore[attr-defined]
            dataframe["study"].tolist(),
            result,
            title="森林图 / Forest plot",
            output_path=output,
        )
        assert output.exists()
        assert output.stat().st_size > 0
        assert "I²" in " ".join(text.get_text() for text in axis.texts)
        plt.close(figure)


def test_funnel_plot_saves_png_and_displays_egger_annotation(tmp_path: Path) -> None:
    _, effects = _sample_effects()
    output = tmp_path / "funnel.png"

    figure, axis = funnel_plot(
        [effect.effect for effect in effects],  # type: ignore[attr-defined]
        [effect.standard_error for effect in effects],  # type: ignore[attr-defined]
        egger_p_value=0.041,
        title="漏斗图 / Funnel plot",
        output_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert any("Egger test p = 0.041" in text.get_text() for text in axis.texts)
    assert axis.yaxis_inverted()
    plt.close(figure)


def test_labbe_plot_uses_binary_risks_and_labels(tmp_path: Path) -> None:
    output = tmp_path / "labbe.png"
    figure, axis = labbe_plot(
        [10, 18, 12],
        [90, 82, 88],
        [5, 10, 9],
        [95, 90, 91],
        study_labels=["A", "B", "C"],
        title="L'Abbé plot",
        output_path=output,
    )

    assert output.exists()
    assert axis.get_xlabel() == "Control risk"
    assert len(axis.collections) == 1
    plt.close(figure)


def test_plot_functions_reject_invalid_standard_errors() -> None:
    result = random_effects([0.2, 0.4], [0.01, 0.01])

    with pytest.raises(ValueError, match="greater than zero"):
        forest_plot([0.2, 0.4], [0.1, 0.0], ["A", "B"], result)
    with pytest.raises(ValueError, match="greater than zero"):
        funnel_plot([0.2, 0.4], [0.1, 0.0])


def test_labbe_plot_rejects_inconsistent_binary_data() -> None:
    with pytest.raises(ValueError, match="equal non-zero"):
        labbe_plot([1], [9], [2, 3], [8, 7])


def test_forest_and_funnel_commands_generate_files(tmp_path: Path) -> None:
    runner = CliRunner()
    forest_output = tmp_path / "command_forest.png"
    funnel_output = tmp_path / "command_funnel.png"

    forest_result = runner.invoke(
        app,
        ["forest", str(SAMPLE_DATA), "--output", str(forest_output)],
    )
    funnel_result = runner.invoke(
        app,
        ["funnel", str(SAMPLE_DATA), "--output", str(funnel_output)],
    )

    assert forest_result.exit_code == 0, forest_result.output
    assert funnel_result.exit_code == 0, funnel_result.output
    assert forest_output.exists()
    assert funnel_output.exists()
