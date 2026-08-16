"""GRADE certainty-of-evidence assessment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

DowngradeLevel = Literal["no", "serious", "very serious"]
UpgradeLevel = Literal["no", "yes"]
GradeLevel = Literal["high", "moderate", "low", "very low"]

_DOWNGRADE_DIMENSIONS = (
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
)
_UPGRADE_DIMENSIONS = ("large_effect", "dose_response", "confounding")
_DOWNGRADE_POINTS = {"no": 0, "serious": -1, "very serious": -2}
_UPGRADE_POINTS = {"no": 0, "yes": 1}
_EXPLANATIONS = {
    "risk_of_bias": "Risk of bias across included studies.",
    "inconsistency": "Inconsistency or unexplained heterogeneity of results.",
    "indirectness": "Indirectness of population, intervention, comparator, or outcome.",
    "imprecision": "Imprecision of the pooled estimate and confidence interval.",
    "publication_bias": "Likelihood of publication or small-study bias.",
    "large_effect": "Large or very large observed effect eligible for upgrading.",
    "dose_response": "Evidence of a dose-response gradient.",
    "confounding": (
        "Residual confounding would reduce, rather than explain, the effect."
    ),
}


@dataclass(frozen=True)
class GradeResult:
    """GRADE certainty assessment and an auditable rating table.

    Attributes:
        quality: Final certainty level: high, moderate, low, or very low.
        starting_score: Baseline score before downgrading or upgrading.
        final_score: Bounded numerical score used to derive ``quality``.
        total_adjustment: Sum of downgrade and upgrade point changes.
        assessment_table: One row per GRADE domain with assessment and points.
    """

    quality: GradeLevel
    starting_score: int
    final_score: int
    total_adjustment: int
    assessment_table: pd.DataFrame

    def to_markdown(self) -> str:
        """Render the certainty assessment as a portable Markdown table.

        Returns:
            A Markdown heading, summary paragraph, and domain-level rating table.
        """
        lines = [
            "## GRADE certainty of evidence",
            "",
            (
                f"**Overall certainty: {self.quality}** "
                f"(starting score: {self.starting_score}; "
                f"adjustment: {self.total_adjustment:+d}; "
                f"final score: {self.final_score})."
            ),
            "",
            "| Domain | Type | Assessment | Points | Explanation |",
            "|---|---|---|---:|---|",
        ]
        for row in self.assessment_table.itertuples(index=False):
            lines.append(
                f"| {row.domain} | {row.direction} | {row.assessment} | "
                f"{row.points:+d} | {row.explanation} |"
            )
        return "\n".join(lines)


def _validate_downgrade(value: str, dimension: str) -> DowngradeLevel:
    """Validate a single three-level downgrade assessment."""
    if value not in _DOWNGRADE_POINTS:
        raise ValueError(
            f"{dimension} must be one of: 'no', 'serious', or 'very serious'."
        )
    return value  # type: ignore[return-value]


def _validate_upgrade(value: str, dimension: str) -> UpgradeLevel:
    """Validate a single binary upgrade assessment."""
    if value not in _UPGRADE_POINTS:
        raise ValueError(f"{dimension} must be either 'no' or 'yes'.")
    return value  # type: ignore[return-value]


def _quality_from_score(score: int) -> GradeLevel:
    """Map a bounded numerical certainty score to its GRADE label."""
    if score >= 4:
        return "high"
    if score == 3:
        return "moderate"
    if score == 2:
        return "low"
    return "very low"


def assess_grade(
    *,
    risk_of_bias: DowngradeLevel = "no",
    inconsistency: DowngradeLevel = "no",
    indirectness: DowngradeLevel = "no",
    imprecision: DowngradeLevel = "no",
    publication_bias: DowngradeLevel = "no",
    large_effect: UpgradeLevel = "no",
    dose_response: UpgradeLevel = "no",
    confounding: UpgradeLevel = "no",
) -> GradeResult:
    """Assess certainty of evidence using standard GRADE downgrade domains.

    The assessment begins at high certainty. Each ``"serious"`` and
    ``"very serious"`` downgrade subtracts one and two levels, respectively.
    Each affirmative upgrade adds one level. The final score is bounded to the
    four GRADE categories from very low through high.

    Args:
        risk_of_bias: Risk-of-bias downgrade assessment.
        inconsistency: Inconsistency downgrade assessment.
        indirectness: Indirectness downgrade assessment.
        imprecision: Imprecision downgrade assessment.
        publication_bias: Publication-bias downgrade assessment.
        large_effect: Upgrade for a large observed effect.
        dose_response: Upgrade for dose-response evidence.
        confounding: Upgrade when residual confounding would reduce the effect.

    Returns:
        The final evidence certainty and a domain-level explanation table.

    Raises:
        ValueError: If an assessment is outside the permitted vocabulary.
    """
    downgrade_values = {
        "risk_of_bias": risk_of_bias,
        "inconsistency": inconsistency,
        "indirectness": indirectness,
        "imprecision": imprecision,
        "publication_bias": publication_bias,
    }
    upgrade_values = {
        "large_effect": large_effect,
        "dose_response": dose_response,
        "confounding": confounding,
    }
    rows: list[dict[str, object]] = []
    total_adjustment = 0
    for dimension in _DOWNGRADE_DIMENSIONS:
        assessment = _validate_downgrade(downgrade_values[dimension], dimension)
        points = _DOWNGRADE_POINTS[assessment]
        total_adjustment += points
        rows.append(
            {
                "domain": dimension,
                "direction": "downgrade",
                "assessment": assessment,
                "points": points,
                "explanation": _EXPLANATIONS[dimension],
            }
        )
    for dimension in _UPGRADE_DIMENSIONS:
        assessment = _validate_upgrade(upgrade_values[dimension], dimension)
        points = _UPGRADE_POINTS[assessment]
        total_adjustment += points
        rows.append(
            {
                "domain": dimension,
                "direction": "upgrade",
                "assessment": assessment,
                "points": points,
                "explanation": _EXPLANATIONS[dimension],
            }
        )
    starting_score = 4
    final_score = min(4, max(1, starting_score + total_adjustment))
    return GradeResult(
        quality=_quality_from_score(final_score),
        starting_score=starting_score,
        final_score=final_score,
        total_adjustment=total_adjustment,
        assessment_table=pd.DataFrame(rows),
    )
