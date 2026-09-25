"""The detector end to end: a demo in, a report out.

Run it with `uv run python -m overwatch.pipeline match.dem`.
"""

from overwatch.pipeline.analyze import FLAG_PERCENTILE, analyze_demo, analyze_match
from overwatch.pipeline.report import (
    KillReport,
    MatchReport,
    PlayerReport,
    render_text,
)
from overwatch.pipeline.scorer import Scorer

__all__ = [
    "FLAG_PERCENTILE",
    "KillReport",
    "MatchReport",
    "PlayerReport",
    "Scorer",
    "analyze_demo",
    "analyze_match",
    "render_text",
]
