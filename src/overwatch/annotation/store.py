"""Human annotations of judge cases: what a person decided, and on what evidence.

Stored as append-only JSONL (data/annotations/gold.jsonl): one line per save, and
the latest line for a case wins. Nothing is ever rewritten, so an interrupted
session cannot damage earlier work, and a changed mind leaves a history.

Every annotation records a hash of the evidence text it was made on. That text is
the contract with the model (rendering.py). If it changes — a new measurement, a
reworded line — an old annotation may no longer fit what the model is shown, so
the training-set builder skips it instead of training on a mismatch.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from overwatch.layers.l4_judge.targets import verdict_for
from overwatch.layers.l4_judge.verdict import (
    VERDICT_SCHEMA,
    CheatType,
    Verdict,
    VerdictLabel,
)

#: The limits the model's own output is held to, so a human target always fits
#: the grammar the model decodes with.
_REASONS = VERDICT_SCHEMA["properties"]["reasons"]
_CAVEATS = VERDICT_SCHEMA["properties"]["caveats"]
MAX_REASONS: int = _REASONS["maxItems"]
MAX_CAVEATS: int = _CAVEATS["maxItems"]
MAX_SENTENCE: int = _REASONS["items"]["maxLength"]


def case_key(match_id: str, player_id: str) -> str:
    return f"{match_id}/{player_id}"


def evidence_hash(text: str) -> str:
    """A short fingerprint of the exact text the model would read."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Annotation(BaseModel):
    """One person's verdict on one case."""

    match_id: str
    player_id: str
    evidence_sha: str = Field(description="evidence_hash of the text annotated")
    verdict: Verdict
    note: str = Field(
        default="", description="for the project, never for training: what was off"
    )
    seconds: float | None = Field(default=None, description="time spent on the case")
    annotator: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def key(self) -> str:
        return case_key(self.match_id, self.player_id)


def human_verdict(
    probability: int,
    cheat_type: CheatType | str,
    reasons: list[str],
    caveats: list[str],
) -> Verdict:
    """Build a verdict from the form, held to the same rules as generated targets.

    The verdict word follows from the probability, exactly as for generated targets
    (targets.verdict_for), so "unclear at 5%" cannot be written by hand either. A
    cheat type is named only when accusing: an "unclear" that names a cheat would
    teach the model to accuse while claiming not to.

    Raises:
        ValueError: with a message the form can show as it is.
    """
    verdict = verdict_for(probability)
    reasons = [r.strip() for r in reasons if r.strip()]
    caveats = [c.strip() for c in caveats if c.strip()]
    cheat_type = CheatType(cheat_type)

    if not reasons:
        msg = "give at least one reason"
        raise ValueError(msg)
    if len(reasons) > MAX_REASONS:
        msg = f"at most {MAX_REASONS} reasons"
        raise ValueError(msg)
    if len(caveats) > MAX_CAVEATS:
        msg = f"at most {MAX_CAVEATS} caveats"
        raise ValueError(msg)
    for sentence in reasons + caveats:
        if len(sentence) > MAX_SENTENCE:
            msg = (
                f"keep each sentence under {MAX_SENTENCE} characters: {sentence[:40]}…"
            )
            raise ValueError(msg)

    if verdict is not VerdictLabel.CHEATING:
        cheat_type = CheatType.NONE
    elif cheat_type is CheatType.NONE:
        msg = "name the cheat, or lower the probability below the cheating band"
        raise ValueError(msg)

    return Verdict(
        verdict=verdict,
        probability=probability,
        cheat_type=cheat_type,
        reasons=reasons,
        caveats=caveats,
    )


class AnnotationStore:
    """Append-only JSONL of annotations; the latest one per case wins."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Annotation]:
        if not self.path.exists():
            return {}
        latest: dict[str, Annotation] = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                annotation = Annotation.model_validate_json(line)
                latest[annotation.key] = annotation
        return latest

    def append(self, annotation: Annotation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(annotation.model_dump_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # an hour of judgement is worth one fsync


__all__ = [
    "MAX_CAVEATS",
    "MAX_REASONS",
    "MAX_SENTENCE",
    "Annotation",
    "AnnotationStore",
    "case_key",
    "evidence_hash",
    "human_verdict",
]
