"""What the judge model is allowed to say.

The judge states a probability because a reviewer wants one, but a language model
writing "87%" is generating text, not measuring anything. So the number is held to
the standard of any score: evaluate_judge.py checks it against the ban labels, and
in every training target — generated or human — the probability is decided first
and the verdict word follows from fixed bands (targets.verdict_for). Layer 3's
calibrated score stays the measurement; this is the reviewer's reading of it.

The JSON schema below is handed to llama.cpp as a grammar, so malformed output is
impossible rather than merely unlikely. That matters on a 4B model.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class VerdictLabel(StrEnum):
    CHEATING = "cheating"
    CLEAN = "clean"
    UNCLEAR = "unclear"


class CheatType(StrEnum):
    AIMBOT = "aimbot"
    WALLHACK = "wallhack"
    TRIGGERBOT = "triggerbot"
    MIXED = "mixed"
    NONE = "none"


class Verdict(BaseModel):
    """The judge's reading of one player's evidence."""

    verdict: VerdictLabel
    probability: int = Field(
        default=50,
        ge=0,
        le=100,
        description="the judge's own confidence that this player cheated, 0-100",
    )
    cheat_type: CheatType = CheatType.NONE
    reasons: list[str] = Field(
        default_factory=list,
        description="short sentences, each citing a measurement from the evidence",
        max_length=5,
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="what would change this verdict",
        max_length=3,
    )

    def as_report_line(self) -> str:
        reasons = "; ".join(self.reasons) if self.reasons else "no reasons given"
        return f"{self.verdict} {self.probability}% ({self.cheat_type}): {reasons}"


#: JSON schema for constrained decoding. Kept narrow so a small model cannot
#: wander: fixed keys, closed vocabularies, short lists.
VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [v.value for v in VerdictLabel]},
        # A generated number is not a measurement, so it is checked against the
        # labels like any other score (see docs/decisions/0002) rather than trusted.
        "probability": {"type": "integer", "minimum": 0, "maximum": 100},
        "cheat_type": {"type": "string", "enum": [c.value for c in CheatType]},
        # Kept short on purpose: a 4B model on CPU generates ~10 tokens a second,
        # and an answer that does not finish inside the budget is unparseable.
        "reasons": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "minItems": 1,
            "maxItems": 3,
        },
        "caveats": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "maxItems": 2,
        },
    },
    "required": ["verdict", "probability", "cheat_type", "reasons"],
    "additionalProperties": False,
}
