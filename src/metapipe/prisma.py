"""PRISMA 2020-style study selection flowchart generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

Language = Literal["en", "zh"]
ReasonInput = Mapping[str, int] | Sequence[str] | None


@dataclass(frozen=True)
class PrismaCounts:
    """Counts used in a PRISMA 2020-style flow diagram.

    Attributes:
        identified_records: Records identified before duplicate removal.
        records_after_duplicates_removed: Records retained after de-duplication.
        title_abstract_excluded: Records excluded during title/abstract screening.
        full_text_assessed: Full-text reports assessed for eligibility.
        full_text_excluded: Full-text reports excluded after assessment.
        included_studies: Studies included in the synthesis.
    """

    identified_records: int
    records_after_duplicates_removed: int
    title_abstract_excluded: int
    full_text_assessed: int
    full_text_excluded: int
    included_studies: int


@dataclass
class PrismaFlowResult:
    """Generated PRISMA flowchart and its validated study-selection counts."""

    counts: PrismaCounts
    figure: Figure
    axes: Axes


_LABELS = {
    "en": {
        "title": "PRISMA 2020 flow diagram",
        "identification": "Identification",
        "screening": "Screening",
        "eligibility": "Eligibility",
        "included": "Included",
        "identified": "Records identified\n(n = {count})",
        "duplicates": "Records after duplicates removed\n(n = {count})",
        "screened": "Records screened\n(n = {count})",
        "screened_excluded": (
            "Records excluded at title/abstract screening\n(n = {count})"
        ),
        "full_text": "Full-text reports assessed\n(n = {count})",
        "full_text_excluded": "Full-text reports excluded\n(n = {count})",
        "included_studies": "Studies included in synthesis\n(n = {count})",
        "reasons": "Reasons:",
    },
    "zh": {
        "title": "PRISMA 2020 文献筛选流程图",
        "identification": "识别",
        "screening": "筛选",
        "eligibility": "资格评估",
        "included": "纳入",
        "identified": "检索获得文献\n(n = {count})",
        "duplicates": "去重后文献\n(n = {count})",
        "screened": "标题/摘要筛选文献\n(n = {count})",
        "screened_excluded": "标题/摘要筛选排除\n(n = {count})",
        "full_text": "全文评估文献\n(n = {count})",
        "full_text_excluded": "全文评估排除\n(n = {count})",
        "included_studies": "最终纳入研究\n(n = {count})",
        "reasons": "排除原因：",
    },
}


def _validate_non_negative_count(value: int, name: str) -> None:
    """Validate a non-negative integer count."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_counts(counts: PrismaCounts) -> None:
    """Validate monotonic relationships between PRISMA selection stages."""
    for name, value in counts.__dict__.items():
        _validate_non_negative_count(value, name)
    if counts.records_after_duplicates_removed > counts.identified_records:
        raise ValueError(
            "Records after duplicates removed cannot exceed identified records."
        )
    if counts.title_abstract_excluded > counts.records_after_duplicates_removed:
        raise ValueError("Title/abstract exclusions cannot exceed screened records.")
    screened_remaining = (
        counts.records_after_duplicates_removed - counts.title_abstract_excluded
    )
    if counts.full_text_assessed > screened_remaining:
        raise ValueError(
            "Full-text assessments cannot exceed title/abstract screening survivors."
        )
    if counts.full_text_excluded > counts.full_text_assessed:
        raise ValueError("Full-text exclusions cannot exceed full-text assessments.")
    eligible_remaining = counts.full_text_assessed - counts.full_text_excluded
    if counts.included_studies > eligible_remaining:
        raise ValueError("Included studies cannot exceed eligible full-text reports.")


def _format_reasons(reasons: ReasonInput, heading: str) -> str:
    """Format custom exclusion reasons as a compact multiline annotation."""
    if reasons is None:
        return ""
    if isinstance(reasons, Mapping):
        lines = [f"• {reason} (n = {count})" for reason, count in reasons.items()]
    else:
        lines = [f"• {reason}" for reason in reasons]
    if not lines:
        return ""
    return "\n".join([heading, *lines])


def _add_box(
    axes: Axes,
    x_center: float,
    y_center: float,
    text: str,
    *,
    width: float = 0.43,
    height: float = 0.095,
    color: str = "#eff6ff",
) -> None:
    """Add a rounded process box centered at a normalized coordinate."""
    patch = FancyBboxPatch(
        (x_center - width / 2, y_center - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.01",
        linewidth=1.1,
        edgecolor="#1e3a8a",
        facecolor=color,
    )
    axes.add_patch(patch)
    axes.text(x_center, y_center, text, ha="center", va="center", fontsize=9)


def _add_arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Draw an arrow connecting two normalized coordinates."""
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#334155",
        )
    )


def prisma_flowchart(
    *,
    identified_records: int,
    records_after_duplicates_removed: int,
    title_abstract_excluded: int,
    full_text_excluded: int,
    included_studies: int,
    full_text_assessed: int | None = None,
    title_abstract_exclusion_reasons: ReasonInput = None,
    full_text_exclusion_reasons: ReasonInput = None,
    language: Language = "en",
    title: str | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
    ax: Axes | None = None,
) -> PrismaFlowResult:
    """Generate a PRISMA 2020-style study selection flowchart.

    When ``full_text_assessed`` is omitted, it is derived as records remaining
    after title/abstract exclusions. Custom exclusion reasons may be supplied as
    a mapping of reason to count or as a sequence of already formatted reasons.

    Args:
        identified_records: Records identified across all sources.
        records_after_duplicates_removed: Records remaining after de-duplication.
        title_abstract_excluded: Records excluded during title/abstract screening.
        full_text_excluded: Reports excluded after full-text assessment.
        included_studies: Studies included in the final synthesis.
        full_text_assessed: Full-text reports assessed; derived when omitted.
        title_abstract_exclusion_reasons: Optional title/abstract exclusion detail.
        full_text_exclusion_reasons: Optional full-text exclusion detail.
        language: Default label language: English (``"en"``) or Chinese (``"zh"``).
        title: Optional custom title overriding the language default.
        output_path: Optional `.png`, `.svg`, or `.pdf` output path.
        dpi: Resolution for raster output. The default is 300 DPI.
        ax: Existing axes to draw on. A new figure and axes are created if absent.

    Returns:
        Validated counts and the generated Matplotlib figure and axes.

    Raises:
        ValueError: If counts are inconsistent, language or DPI is invalid, or the
            requested output suffix is not supported.
    """
    if language not in _LABELS:
        raise ValueError("language must be either 'en' or 'zh'.")
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")
    if full_text_assessed is None:
        full_text_assessed = records_after_duplicates_removed - title_abstract_excluded
    counts = PrismaCounts(
        identified_records=identified_records,
        records_after_duplicates_removed=records_after_duplicates_removed,
        title_abstract_excluded=title_abstract_excluded,
        full_text_assessed=full_text_assessed,
        full_text_excluded=full_text_excluded,
        included_studies=included_studies,
    )
    _validate_counts(counts)
    labels = _LABELS[language]

    if ax is None:
        figure, axes = plt.subplots(figsize=(11, 10))
    else:
        figure = ax.figure
        axes = ax
    axes.set(xlim=(0, 1), ylim=(0, 1))
    axes.axis("off")
    axes.set_title(title or labels["title"], fontsize=15, fontweight="bold", pad=16)

    y_positions = [0.89, 0.75, 0.61, 0.45, 0.29, 0.13]
    stage_labels = [
        labels["identification"],
        "",
        labels["screening"],
        labels["eligibility"],
        "",
        labels["included"],
    ]
    for y_position, stage_label in zip(y_positions, stage_labels, strict=True):
        if stage_label:
            axes.text(
                0.025,
                y_position,
                stage_label,
                rotation=90,
                va="center",
                ha="center",
                fontsize=9,
                color="#475569",
                fontweight="bold",
            )

    central_boxes = [
        (
            0.5,
            y_positions[0],
            labels["identified"].format(count=counts.identified_records),
        ),
        (
            0.5,
            y_positions[1],
            labels["duplicates"].format(count=counts.records_after_duplicates_removed),
        ),
        (
            0.5,
            y_positions[2],
            labels["screened"].format(count=counts.records_after_duplicates_removed),
        ),
        (
            0.5,
            y_positions[3],
            labels["full_text"].format(count=counts.full_text_assessed),
        ),
        (
            0.5,
            y_positions[4],
            labels["included_studies"].format(count=counts.included_studies),
        ),
    ]
    for x_center, y_center, text in central_boxes:
        _add_box(axes, x_center, y_center, text)
    for start_y, end_y in zip(y_positions[:4], y_positions[1:5], strict=True):
        _add_arrow(axes, (0.5, start_y - 0.055), (0.5, end_y + 0.055))

    screening_exclusion_text = labels["screened_excluded"].format(
        count=counts.title_abstract_excluded
    )
    screening_reasons = _format_reasons(
        title_abstract_exclusion_reasons,
        labels["reasons"],
    )
    if screening_reasons:
        screening_exclusion_text += f"\n{screening_reasons}"
    _add_box(
        axes,
        0.82,
        y_positions[2],
        screening_exclusion_text,
        width=0.31,
        height=0.13 if screening_reasons else 0.095,
        color="#fef2f2",
    )
    _add_arrow(axes, (0.715, y_positions[2]), (0.66, y_positions[2]))

    full_text_exclusion_text = labels["full_text_excluded"].format(
        count=counts.full_text_excluded
    )
    full_text_reasons = _format_reasons(full_text_exclusion_reasons, labels["reasons"])
    if full_text_reasons:
        full_text_exclusion_text += f"\n{full_text_reasons}"
    _add_box(
        axes,
        0.82,
        y_positions[3],
        full_text_exclusion_text,
        width=0.31,
        height=0.13 if full_text_reasons else 0.095,
        color="#fef2f2",
    )
    _add_arrow(axes, (0.715, y_positions[3]), (0.66, y_positions[3]))

    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise ValueError("Output path must use a .png, .svg, or .pdf suffix.")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return PrismaFlowResult(counts=counts, figure=figure, axes=axes)
