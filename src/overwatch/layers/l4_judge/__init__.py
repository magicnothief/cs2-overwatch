"""Layer 4: turn evidence into a written verdict a reviewer can check."""

from overwatch.layers.l4_judge.judge import Judge, JudgeResult
from overwatch.layers.l4_judge.rendering import (
    SYSTEM_PROMPT,
    MomentSummary,
    PlayerCase,
    render_case,
)
from overwatch.layers.l4_judge.verdict import (
    VERDICT_SCHEMA,
    CheatType,
    Verdict,
    VerdictLabel,
)

__all__ = [
    "SYSTEM_PROMPT",
    "VERDICT_SCHEMA",
    "CheatType",
    "Judge",
    "JudgeResult",
    "MomentSummary",
    "PlayerCase",
    "Verdict",
    "VerdictLabel",
    "render_case",
]
