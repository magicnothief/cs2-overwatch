"""What a layer says when it finds something.

Every layer speaks in Evidence: a named observation with the value it measured,
what normal looks like, and a sentence a human can read. Layer 4 turns a Moment's
evidence into a verdict, the web UI renders it, and a reviewer checks it — so the
same object has to serve all three.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    """How much weight one observation deserves on its own."""

    INFO = "info"  # context, not suspicious by itself
    NOTABLE = "notable"  # unusual, meaningful in aggregate
    STRONG = "strong"  # hard to explain by skill
    IMPOSSIBLE = "impossible"  # outside what the game or a human hand can do


class Evidence(BaseModel):
    """One measured observation about one player at one place in a demo."""

    layer: str = Field(description="which layer produced this, e.g. 'l1_blatant'")
    name: str = Field(description="machine name of the check, e.g. 'kill_tick_snap'")
    value: float = Field(description="what was measured")
    unit: str = Field(default="", description="unit of value, e.g. 'deg/s'")
    threshold: float | None = Field(
        default=None, description="the line this value crossed, when there is one"
    )
    baseline: float | None = Field(
        default=None, description="what clean players typically show for this check"
    )
    severity: Severity = Severity.NOTABLE
    tick: int | None = Field(default=None, description="where to look in the demo")
    note: str = Field(description="one sentence a reviewer can read")


class Moment(BaseModel):
    """A stretch of one player's demo worth looking at, with its evidence."""

    match_id: str
    player_id: str
    tick_start: int
    tick_end: int
    trigger: str = Field(description="why this moment exists: a rule name, a kill, …")
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def worst_severity(self) -> Severity:
        order = list(Severity)
        return max(
            (e.severity for e in self.evidence),
            key=order.index,
            default=Severity.INFO,
        )

    def demo_command(self) -> str:
        """What to type in CS2 to watch this moment."""
        return f"demo_gototick {max(self.tick_start - 64, 0)}"
