"""End-to-end tests for Markdown and Excel meta-analysis report generation."""

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from metapipe.cli import app
from metapipe.report import AnalysisConfig, generate_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "examples" / "sample_data.csv"


def test_generate_report_creates_markdown_excel_and_diagnostic_assets(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis_report.md"

    result = generate_report(SAMPLE_DATA, output)

    assert result.markdown_path == output
    assert output.exists()
    assert result.excel_path.exists()
    assert set(result.asset_paths) == {"forest_plot", "funnel_plot", "leave_one_out"}
    assert all(
        path.exists() and path.stat().st_size > 0
        for path in result.asset_paths.values()
    )
    markdown = output.read_text(encoding="utf-8")
    assert "# Meta-analysis report" in markdown
    assert "## Methods" in markdown
    assert "## Results" in markdown
    assert "## Conclusion" in markdown
    assert "random-effects model with REML tau² estimation" in markdown
    workbook = pd.ExcelFile(result.excel_path)
    assert set(workbook.sheet_names) == {"Summary", "Study effects", "Leave-one-out"}
    assert len(pd.read_excel(result.excel_path, sheet_name="Study effects")) == 12


def test_report_supports_fixed_mean_difference_and_subgroup_analysis(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fixed_report.md"
    config = AnalysisConfig(
        effect_measure="mean_difference",
        model_type="fixed",
        subgroup_column="study_type",
        include_funnel=False,
        include_sensitivity=False,
    )

    result = generate_report(SAMPLE_DATA, output, config=config)

    markdown = output.read_text(encoding="utf-8")
    assert result.subgroup_result is not None
    assert set(result.asset_paths) == {"forest_plot"}
    assert "inverse-variance fixed-effect model" in markdown
    assert "## Subgroup analysis" in markdown
    workbook = pd.ExcelFile(result.excel_path)
    assert "Subgroups" in workbook.sheet_names


def test_report_command_generates_requested_markdown_file(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "cli_report.md"

    command_result = runner.invoke(
        app,
        ["report", str(SAMPLE_DATA), "--output", str(output)],
    )

    assert command_result.exit_code == 0, command_result.output
    assert output.exists()
    assert output.with_suffix(".xlsx").exists()
    assert "Report written to" in command_result.output


def test_report_validates_input_and_output_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".md suffix"):
        generate_report(SAMPLE_DATA, tmp_path / "report.txt")
    with pytest.raises(ValueError, match="Subgroup column"):
        generate_report(
            SAMPLE_DATA,
            tmp_path / "report.md",
            config=AnalysisConfig(subgroup_column="missing"),
        )
    with pytest.raises(ValueError, match="Input data must be a CSV"):
        generate_report(tmp_path / "data.txt", tmp_path / "report.md")


BINARY_SAMPLE_DATA = PROJECT_ROOT / "examples" / "sample_data_binary.csv"


@pytest.mark.parametrize(
    ("effect_measure", "effect_label"),
    [
        ("odds_ratio", "log(OR)"),
        ("risk_ratio", "log(RR)"),
        ("risk_difference", "Risk difference"),
    ],
)
def test_report_supports_binary_or_rr_and_rd(
    tmp_path: Path,
    effect_measure: str,
    effect_label: str,
) -> None:
    output = tmp_path / f"{effect_measure}_report.md"
    result = generate_report(
        BINARY_SAMPLE_DATA,
        output,
        config=AnalysisConfig(
            effect_measure=effect_measure,  # type: ignore[arg-type]
            include_sensitivity=False,
        ),
    )

    assert output.exists()
    assert result.excel_path.exists()
    assert all(path.exists() for path in result.asset_paths.values())
    assert effect_label in output.read_text(encoding="utf-8")
    study_effects = pd.read_excel(result.excel_path, sheet_name="Study effects")
    assert len(study_effects) == 10
    assert study_effects["effect"].notna().all()


def test_report_command_supports_binary_effect_measure(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "binary_cli_report.md"

    command_result = runner.invoke(
        app,
        [
            "report",
            str(BINARY_SAMPLE_DATA),
            "--effect-measure",
            "odds_ratio",
            "--output",
            str(output),
        ],
    )

    assert command_result.exit_code == 0, command_result.output
    assert output.exists()
    assert "log(OR)" in output.read_text(encoding="utf-8")
