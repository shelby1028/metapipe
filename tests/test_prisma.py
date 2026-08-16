"""Tests for PRISMA 2020-style flowchart generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from metapipe.prisma import prisma_flowchart


def _flowchart_arguments() -> dict[str, object]:
    return {
        "identified_records": 180,
        "records_after_duplicates_removed": 150,
        "title_abstract_excluded": 105,
        "full_text_assessed": 45,
        "full_text_excluded": 30,
        "included_studies": 15,
    }


def test_prisma_flowchart_saves_png_svg_and_pdf_with_reasons(tmp_path: Path) -> None:
    for extension in ("png", "svg", "pdf"):
        output = tmp_path / f"prisma.{extension}"
        result = prisma_flowchart(
            **_flowchart_arguments(),
            title_abstract_exclusion_reasons={
                "Not older adults": 55,
                "Not aerobic": 50,
            },
            full_text_exclusion_reasons={"Wrong outcome": 18, "Insufficient data": 12},
            output_path=output,
        )

        assert output.exists()
        assert output.stat().st_size > 0
        assert result.counts.included_studies == 15
        assert "PRISMA" in result.axes.get_title()
        plt.close(result.figure)


def test_prisma_flowchart_derives_full_text_assessment_and_supports_chinese() -> None:
    result = prisma_flowchart(
        identified_records=100,
        records_after_duplicates_removed=80,
        title_abstract_excluded=50,
        full_text_excluded=20,
        included_studies=10,
        language="zh",
        title_abstract_exclusion_reasons=["非随机研究", "不符合人群"],
    )

    assert result.counts.full_text_assessed == 30
    assert any("文献" in text.get_text() for text in result.axes.texts)
    plt.close(result.figure)


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "identified_records": 10,
            "records_after_duplicates_removed": 12,
            "title_abstract_excluded": 1,
            "full_text_excluded": 1,
            "included_studies": 1,
        },
        {
            "identified_records": 10,
            "records_after_duplicates_removed": 8,
            "title_abstract_excluded": 9,
            "full_text_excluded": 0,
            "included_studies": 0,
        },
        {
            "identified_records": 10,
            "records_after_duplicates_removed": 8,
            "title_abstract_excluded": 2,
            "full_text_assessed": 6,
            "full_text_excluded": 7,
            "included_studies": 0,
        },
    ],
)
def test_prisma_flowchart_rejects_inconsistent_counts(
    arguments: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        prisma_flowchart(**arguments)


def test_prisma_flowchart_validates_language_dpi_and_output_format(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="language"):
        prisma_flowchart(**_flowchart_arguments(), language="fr")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="DPI"):
        prisma_flowchart(**_flowchart_arguments(), dpi=0)
    with pytest.raises(ValueError, match="suffix"):
        prisma_flowchart(**_flowchart_arguments(), output_path=tmp_path / "flow.txt")
