"""Tests for GRADE certainty-of-evidence assessment."""

import pytest

from metapipe.grade import assess_grade


def test_grade_starts_high_when_no_downgrades_or_upgrades() -> None:
    result = assess_grade()

    assert result.quality == "high"
    assert result.starting_score == 4
    assert result.final_score == 4
    assert result.total_adjustment == 0
    assert len(result.assessment_table) == 8


def test_grade_applies_serious_and_very_serious_downgrades() -> None:
    moderate = assess_grade(risk_of_bias="serious")
    low = assess_grade(risk_of_bias="very serious")
    very_low = assess_grade(
        risk_of_bias="very serious",
        inconsistency="very serious",
    )

    assert moderate.quality == "moderate"
    assert low.quality == "low"
    assert very_low.quality == "very low"
    assert very_low.final_score == 1


def test_grade_upgrades_can_restore_certainty_within_high_cap() -> None:
    result = assess_grade(
        inconsistency="serious",
        large_effect="yes",
        dose_response="yes",
    )

    assert result.total_adjustment == 1
    assert result.quality == "high"
    assert result.final_score == 4
    assert (
        result.assessment_table.loc[
            result.assessment_table["domain"] == "large_effect", "points"
        ].item()
        == 1
    )


def test_grade_markdown_contains_summary_and_all_domains() -> None:
    result = assess_grade(imprecision="serious", publication_bias="serious")
    markdown = result.to_markdown()

    assert "## GRADE certainty of evidence" in markdown
    assert "**Overall certainty: low**" in markdown
    assert "| risk_of_bias | downgrade | no | +0" in markdown
    assert "| publication_bias | downgrade | serious | -1" in markdown
    assert "| confounding | upgrade | no | +0" in markdown


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("risk_of_bias", "unclear"),
        ("inconsistency", "yes"),
        ("large_effect", "serious"),
        ("confounding", "sometimes"),
    ],
)
def test_grade_rejects_invalid_assessment_values(keyword: str, value: str) -> None:
    with pytest.raises(ValueError):
        assess_grade(**{keyword: value})  # type: ignore[arg-type]
