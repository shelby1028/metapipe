"""Publication-ready plots for meta-analysis results.

The functions in this module use Matplotlib's non-interactive ``Agg`` backend,
so they can be used from notebooks, scripts, continuous integration, and the
command line without opening GUI windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from metapipe.models import MetaAnalysisResult

ImageFormat = Literal["png", "svg", "pdf"]


def _validate_effect_inputs(
    effects: Sequence[float],
    standard_errors: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate effect estimates and standard errors."""
    effect_array = np.asarray(effects, dtype=float)
    standard_error_array = np.asarray(standard_errors, dtype=float)
    if effect_array.ndim != 1 or standard_error_array.ndim != 1:
        raise ValueError("Effects and standard errors must be one-dimensional.")
    if effect_array.size == 0 or effect_array.size != standard_error_array.size:
        raise ValueError("Effects and standard errors must have equal non-zero length.")
    if not np.all(np.isfinite(effect_array)):
        raise ValueError("Effects must be finite.")
    if not np.all(np.isfinite(standard_error_array)) or np.any(
        standard_error_array <= 0
    ):
        raise ValueError("Standard errors must be finite and greater than zero.")
    return effect_array, standard_error_array


def _save_figure(
    figure: Figure,
    output_path: str | Path | None,
    dpi: int,
) -> None:
    """Save a figure after validating a supported output suffix."""
    if output_path is None:
        return
    path = Path(output_path)
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in {"png", "svg", "pdf"}:
        raise ValueError("Output path must use a .png, .svg, or .pdf suffix.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def forest_plot(
    effects: Sequence[float],
    standard_errors: Sequence[float],
    study_labels: Sequence[str],
    model_result: MetaAnalysisResult,
    *,
    title: str | None = None,
    effect_label: str = "Effect size",
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw a forest plot with study estimates, pooled diamond, and weights.

    Args:
        effects: Study-level estimates on a common analysis scale.
        standard_errors: Study-level standard errors on the same scale.
        study_labels: Labels for the studies, in the same order as ``effects``.
        model_result: Fixed-effect or random-effects result used for the pooled
            diamond and relative weights.
        title: Optional chart title; Chinese and English text are accepted.
        effect_label: Label for the horizontal effect-size axis.
        output_path: Optional `.png`, `.svg`, or `.pdf` output path.
        dpi: Resolution for raster output. The default is 300 DPI.
        ax: Existing axes to draw on. A new figure and axes are created if absent.

    Returns:
        A tuple containing the Matplotlib figure and axes.

    Raises:
        ValueError: If input lengths, standard errors, weights, or output format
            are invalid.
    """
    effect_array, standard_error_array = _validate_effect_inputs(
        effects, standard_errors
    )
    if len(study_labels) != effect_array.size:
        raise ValueError("Study labels must have the same length as effects.")
    if len(model_result.weights) != effect_array.size:
        raise ValueError("Model weights must have the same length as effects.")
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(10, max(4.5, 0.62 * effect_array.size + 2)))
    else:
        figure = ax.figure

    y_positions = np.arange(effect_array.size, 0, -1, dtype=float)
    lower = effect_array - 1.95996398454 * standard_error_array
    upper = effect_array + 1.95996398454 * standard_error_array
    marker_sizes = 35 + 220 * np.asarray(model_result.weights, dtype=float)

    ax.errorbar(
        effect_array,
        y_positions,
        xerr=1.95996398454 * standard_error_array,
        fmt="none",
        ecolor="#334155",
        capsize=3,
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        effect_array,
        y_positions,
        s=marker_sizes,
        color="#2563eb",
        edgecolor="#1e3a8a",
        linewidth=0.6,
        zorder=2,
    )

    pooled_y = 0.0
    diamond_height = 0.26
    diamond = Polygon(
        [
            (model_result.ci_lower, pooled_y),
            (model_result.pooled_effect, pooled_y + diamond_height),
            (model_result.ci_upper, pooled_y),
            (model_result.pooled_effect, pooled_y - diamond_height),
        ],
        closed=True,
        facecolor="#dc2626",
        edgecolor="#7f1d1d",
        alpha=0.9,
        zorder=3,
    )
    ax.add_patch(diamond)
    ax.axvline(0.0, color="#64748b", linestyle="--", linewidth=1, zorder=0)

    labels = [str(label) for label in study_labels] + ["Pooled"]
    ax.set_yticks([*y_positions, pooled_y], labels=labels)
    ax.set_ylim(-0.8, effect_array.size + 0.8)
    ax.set_xlabel(effect_label)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    annotation_x = max(float(np.max(upper)), model_result.ci_upper)
    x_range = max(annotation_x - min(float(np.min(lower)), model_result.ci_lower), 1e-6)
    annotation_x += 0.04 * x_range
    for y_position, effect, ci_low, ci_high, weight in zip(
        y_positions,
        effect_array,
        lower,
        upper,
        model_result.weights,
        strict=True,
    ):
        ax.text(
            annotation_x,
            y_position,
            f"{effect:.2f} [{ci_low:.2f}, {ci_high:.2f}]   {weight * 100:.1f}%",
            va="center",
            fontsize=8.5,
        )
    ax.text(
        annotation_x,
        pooled_y,
        (
            f"{model_result.pooled_effect:.2f} "
            f"[{model_result.ci_lower:.2f}, {model_result.ci_upper:.2f}]"
        ),
        va="center",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.02,
        f"I² = {model_result.i_squared:.1f}%   τ² = {model_result.tau_squared:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xlim(
        min(float(np.min(lower)), model_result.ci_lower) - 0.03 * x_range,
        annotation_x + 0.48 * x_range,
    )
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)
    return figure, ax


def funnel_plot(
    effects: Sequence[float],
    standard_errors: Sequence[float],
    *,
    pooled_effect: float | None = None,
    egger_p_value: float | None = None,
    title: str | None = None,
    effect_label: str = "Effect size",
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw a funnel plot with 95% pseudo-confidence limits.

    Args:
        effects: Study-level estimates on a common analysis scale.
        standard_errors: Study-level standard errors on the same scale.
        pooled_effect: Center line for pseudo-confidence limits. If omitted, the
            arithmetic mean of the supplied effects is used.
        egger_p_value: Optional p value from Egger's regression test.
        title: Optional chart title; Chinese and English text are accepted.
        effect_label: Label for the horizontal effect-size axis.
        output_path: Optional `.png`, `.svg`, or `.pdf` output path.
        dpi: Resolution for raster output. The default is 300 DPI.
        ax: Existing axes to draw on. A new figure and axes are created if absent.

    Returns:
        A tuple containing the Matplotlib figure and axes.

    Raises:
        ValueError: If inputs, p value, DPI, or output format are invalid.
    """
    effect_array, standard_error_array = _validate_effect_inputs(
        effects, standard_errors
    )
    if pooled_effect is None:
        pooled_effect = float(np.mean(effect_array))
    if not np.isfinite(pooled_effect):
        raise ValueError("Pooled effect must be finite.")
    if egger_p_value is not None and not 0 <= egger_p_value <= 1:
        raise ValueError("Egger p value must be between zero and one.")
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.5, 6))
    else:
        figure = ax.figure

    max_standard_error = float(np.max(standard_error_array)) * 1.08
    se_grid = np.linspace(0, max_standard_error, 200)
    lower_limit = pooled_effect - 1.95996398454 * se_grid
    upper_limit = pooled_effect + 1.95996398454 * se_grid

    ax.fill_betweenx(
        se_grid,
        lower_limit,
        upper_limit,
        color="#dbeafe",
        alpha=0.55,
        label="95% pseudo-confidence limits",
    )
    ax.plot(lower_limit, se_grid, color="#2563eb", linestyle="--", linewidth=1)
    ax.plot(upper_limit, se_grid, color="#2563eb", linestyle="--", linewidth=1)
    ax.axvline(pooled_effect, color="#dc2626", linewidth=1.3, label="Pooled effect")
    ax.scatter(
        effect_array,
        standard_error_array,
        s=52,
        color="#0f766e",
        edgecolor="#134e4a",
        linewidth=0.6,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xlabel(effect_label)
    ax.set_ylabel("Standard error")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title, fontweight="bold")
    if egger_p_value is not None:
        ax.text(
            0.02,
            0.04,
            f"Egger test p = {egger_p_value:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)
    return figure, ax


def labbe_plot(
    events_treatment: Sequence[float],
    non_events_treatment: Sequence[float],
    events_control: Sequence[float],
    non_events_control: Sequence[float],
    *,
    study_labels: Sequence[str] | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw an L'Abbé plot for binary outcomes.

    Args:
        events_treatment: Event counts in treatment arms.
        non_events_treatment: Non-event counts in treatment arms.
        events_control: Event counts in control arms.
        non_events_control: Non-event counts in control arms.
        study_labels: Optional study labels displayed next to points.
        title: Optional chart title; Chinese and English text are accepted.
        output_path: Optional `.png`, `.svg`, or `.pdf` output path.
        dpi: Resolution for raster output. The default is 300 DPI.
        ax: Existing axes to draw on. A new figure and axes are created if absent.

    Returns:
        A tuple containing the Matplotlib figure and axes.

    Raises:
        ValueError: If table cells or output settings are invalid.
    """
    arrays = [
        np.asarray(values, dtype=float)
        for values in (
            events_treatment,
            non_events_treatment,
            events_control,
            non_events_control,
        )
    ]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("All binary-data sequences must have equal non-zero length.")
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Binary-data sequences must be one-dimensional.")
    if any(not np.all(np.isfinite(array)) or np.any(array < 0) for array in arrays):
        raise ValueError("Binary table cells must be finite and non-negative.")
    treatment_total = arrays[0] + arrays[1]
    control_total = arrays[2] + arrays[3]
    if np.any(treatment_total <= 0) or np.any(control_total <= 0):
        raise ValueError("Each treatment and control arm must contain participants.")
    if study_labels is not None and len(study_labels) != arrays[0].size:
        raise ValueError("Study labels must have the same length as binary data.")
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    control_risk = arrays[2] / control_total
    treatment_risk = arrays[0] / treatment_total
    sample_sizes = treatment_total + control_total
    marker_sizes = 45 + 240 * sample_sizes / np.max(sample_sizes)

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 6))
    else:
        figure = ax.figure

    ax.scatter(
        control_risk,
        treatment_risk,
        s=marker_sizes,
        color="#7c3aed",
        edgecolor="#4c1d95",
        linewidth=0.7,
        alpha=0.75,
    )
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.2)
    if study_labels is not None:
        for x_value, y_value, label in zip(
            control_risk, treatment_risk, study_labels, strict=True
        ):
            ax.annotate(
                str(label),
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xlabel="Control risk",
        ylabel="Treatment risk",
    )
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)
    return figure, ax
