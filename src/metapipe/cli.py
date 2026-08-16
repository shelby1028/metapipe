"""Command-line interface for metapipe."""

import typer

from metapipe import __version__

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)


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
