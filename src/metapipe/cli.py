"""Command-line interface for metapipe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from metapipe import __version__
from metapipe.effects import EffectSize, hedges_g
from metapipe.models import random_effects
from metapipe.plots import forest_plot, funnel_plot
from metapipe.report import AnalysisConfig, generate_report

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)

_REQUIRED_CONTINUOUS_COLUMNS = {
    "study",
    "mean_treatment",
    "sd_treatment",
    "n_treatment",
    "mean_control",
    "sd_control",
    "n_control",
}
_DATA_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    dir_okay=False,
    readable=True,
    help="CSV with study-level continuous outcome data.",
)
_FOREST_OUTPUT_OPTION = typer.Option(
    Path("forest_plot.png"),
    "--output",
    "-o",
    help="PNG, SVG, or PDF path for the forest plot.",
)
_FUNNEL_OUTPUT_OPTION = typer.Option(
    Path("funnel_plot.png"),
    "--output",
    "-o",
    help="PNG, SVG, or PDF path for the funnel plot.",
)
_REPORT_OUTPUT_OPTION = typer.Option(
    Path("report.md"),
    "--output",
    "-o",
    help="Markdown destination for the generated report.",
)


def _load_continuous_effects(data_path: Path) -> tuple[pd.DataFrame, list[EffectSize]]:
    """Load a continuous-outcome CSV file and calculate Hedges' g estimates."""
    dataframe = pd.read_csv(data_path)
    missing_columns = sorted(_REQUIRED_CONTINUOUS_COLUMNS - set(dataframe.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise typer.BadParameter(f"CSV is missing required columns: {missing}")
    effect_sizes = [
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
    return dataframe, effect_sizes


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed metapipe version and exit.",
        is_eager=True,
    ),
) -> None:
    """Run the metapipe command-line interface."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def forest(
    data: Path = _DATA_ARGUMENT,
    output: Path = _FOREST_OUTPUT_OPTION,
) -> None:
    """Generate a random-effects forest plot from a continuous-outcome CSV."""
    dataframe, effect_sizes = _load_continuous_effects(data)
    result = random_effects(effect_sizes, tau_method="reml")
    figure, _ = forest_plot(
        [effect.effect for effect in effect_sizes],
        [effect.standard_error for effect in effect_sizes],
        dataframe["study"].astype(str).tolist(),
        result,
        title="Meta-analysis forest plot",
        effect_label="Hedges' g",
        output_path=output,
    )
    figure.clear()
    typer.echo(f"Forest plot written to {output}")


@app.command()
def funnel(
    data: Path = _DATA_ARGUMENT,
    output: Path = _FUNNEL_OUTPUT_OPTION,
) -> None:
    """Generate a random-effects funnel plot from a continuous-outcome CSV."""
    _, effect_sizes = _load_continuous_effects(data)
    result = random_effects(effect_sizes, tau_method="reml")
    figure, _ = funnel_plot(
        [effect.effect for effect in effect_sizes],
        [effect.standard_error for effect in effect_sizes],
        pooled_effect=result.pooled_effect,
        title="Meta-analysis funnel plot",
        effect_label="Hedges' g",
        output_path=output,
    )
    figure.clear()
    typer.echo(f"Funnel plot written to {output}")


@app.command()
def report(
    data: Path = _DATA_ARGUMENT,
    output: Path = _REPORT_OUTPUT_OPTION,
) -> None:
    """Generate a Markdown report, figures, and Excel results workbook from CSV."""
    result = generate_report(data, output, config=AnalysisConfig())
    typer.echo(f"Report written to {result.markdown_path}")
    typer.echo(f"Excel workbook written to {result.excel_path}")
