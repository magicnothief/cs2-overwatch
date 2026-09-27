"""The judge may doubt on one line; it may not accuse on one.

A fine-tuned judge learns its targets, and every judge before v6 was trained on
targets where one measurement past its 99% line made `cheating` on its own. That
line puts 1 clean player in 100 past it by construction, so sooner or later such a
judge accuses a legitimate player for being in the tail (docs/specs/
2026-09-27-judge-v6-evidence.md, section 3). This holds a `cheating` verdict back
to `unclear` when the evidence the judge read does not meet targets.corroborated,
and says so. Nothing else about the verdict changes.

Only the measurements the judge was shown count, and against the lines it was
shown: the guard reads the same evidence text, not the pipeline's full case.
"""

from __future__ import annotations

from overwatch.layers.l4_judge.rendering import FEATURE_LABELS, PlayerCase, features_for
from overwatch.layers.l4_judge.targets import (
    RULE_REASON,
    UNCLEAR_PROBABILITY,
    WEIGHTS,
    corroborated,
    counted_points,
)
from overwatch.layers.l4_judge.verdict import CheatType, Verdict, VerdictLabel


def seen_by(case: PlayerCase, generation: int) -> PlayerCase:
    """The case as that generation of judge reads it: none of the measurements it
    was never shown."""
    shown = features_for(generation)
    return case.model_copy(
        update={
            "features": {
                k: v
                for k, v in case.features.items()
                if k not in FEATURE_LABELS or k in shown
            }
        }
    )


def guard(
    verdict: Verdict, case: PlayerCase, generation: int
) -> tuple[Verdict, str | None]:
    """The verdict to show, and why it was held back (None when it was not)."""
    if verdict.verdict is not VerdictLabel.CHEATING:
        return verdict, None
    points = counted_points(seen_by(case, generation))
    if corroborated(points):
        return verdict, None
    lines = [p for p in points if not p[0].startswith(RULE_REASON)]
    strong = sum(1 for _, weight, _ in lines if weight >= WEIGHTS["strong"])
    if strong == 1 and len(lines) == 1:
        basis = "one measurement past its 99% line alone"
    elif strong == 0 and lines:
        basis = "measurements past their 95% lines, none past 99%"
    else:
        basis = "evidence that does not include two measurements past their lines"
    note = (
        f"Held at unclear. The judge said cheating ({verdict.probability}%"
        f"{'' if verdict.cheat_type is CheatType.NONE else f', {verdict.cheat_type}'})"
        f" on {basis}. An accusation needs two measurements past their lines, one of"
        " them past 99%, or a hard-limit finding: one line alone is where 1 clean"
        " player in 100 sits by construction."
    )
    held = verdict.model_copy(
        update={
            "verdict": VerdictLabel.UNCLEAR,
            "probability": min(verdict.probability, UNCLEAR_PROBABILITY[1]),
            "cheat_type": CheatType.NONE,
        }
    )
    return held, note


__all__ = ["guard", "seen_by"]
