"""What the pipeline hands a reviewer: one report per demo.

The same object feeds the terminal summary (render_text) and, as JSON, the web
interface. Every claim in it points back to a tick, with the console command that
jumps there, because a report is only as good as a human's ability to check it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from overwatch.layers.l4_judge.rendering import TrajectoryPoint
from overwatch.layers.l4_judge.verdict import Verdict
from overwatch.schemas.evidence import Evidence

#: Seconds of demo to rewind before a kill, so the approach is on screen.
REWIND_TICKS = 128


def goto(tick: int) -> str:
    """The CS2 console command that jumps the demo to just before `tick`."""
    return f"demo_gototick {max(tick - REWIND_TICKS, 0)}"


class RadarPoint(BaseModel):
    """Both players at one instant of a kill's approach, in world units."""

    ms: int = Field(description="relative to the kill; negative is before")
    ax: float
    ay: float
    az: float | None = Field(default=None, description="height: picks the radar storey")
    yaw: float | None = Field(default=None, description="where the attacker looked")
    vx: float | None = None
    vy: float | None = None
    vz: float | None = None
    visible: bool = Field(description="could the attacker see the victim")


class KillReport(BaseModel):
    """One kill, as a reviewer would look at it."""

    tick: int
    round: int | None = Field(default=None, description="1-based round number")
    victim: str | None = None
    weapon: str | None = None
    headshot: bool | None = None
    distance: float | None = Field(default=None, description="metres")
    walls_penetrated: int | None = None
    score: float | None = Field(default=None, description="Layer 3, this kill alone")
    aim_through_cover: float | None = Field(
        default=None, description="share of the last 0.5 s aimed at a hidden enemy"
    )
    reaction_ms: float | None = None
    seen_first: bool | None = Field(
        default=None, description="was the victim visible at any point before dying"
    )
    context: str | None = Field(default=None, description="what they could have known")
    trajectory: list[TrajectoryPoint] = Field(
        default_factory=list, description="crosshair to head, every ~125 ms"
    )
    attacker_side: str | None = Field(default=None, description="CT or T at the kill")
    victim_side: str | None = None
    path: list[RadarPoint] = Field(
        default_factory=list, description="both players' positions, for the radar"
    )
    demo_command: str


class RoundSpan(BaseModel):
    """Where one round sits in the demo, for placing kills on a timeline."""

    number: int
    start_tick: int
    end_tick: int


class PlayerReport(BaseModel):
    """Everything the layers found about one player."""

    player_id: str
    name: str | None = None
    side: str | None = Field(default=None, description="starting side: CT or T")
    judge_alias: str = Field(description="how the judge saw them, e.g. Player_3")
    kills: int
    score: float | None = Field(default=None, description="Layer 3, mean over kills")
    clean_percentile: float | None = Field(
        default=None, description="share of clean players who scored lower"
    )
    enough_kills: bool = True
    flagged: bool = False
    flag_reasons: list[str] = Field(default_factory=list)
    rule_findings: list[Evidence] = Field(default_factory=list)
    top_kills: list[KillReport] = Field(
        default_factory=list, description="the kills to watch first"
    )
    kill_log: list[KillReport] = Field(
        default_factory=list, description="every kill, in order"
    )
    verdict: Verdict | None = None
    judge_error: str | None = None
    judge_evidence: str | None = Field(
        default=None, description="the exact text the judge read"
    )


class MatchReport(BaseModel):
    """One demo, analysed."""

    demo: str
    sha256: str | None = None
    map_name: str | None = None
    visibility: str = Field(description="how line of sight was decided")
    kills: int
    rounds: list[RoundSpan] = Field(default_factory=list)
    side_switches: list[int] = Field(
        default_factory=list, description="rounds after which the teams swapped sides"
    )
    flag_percentile: float | None = Field(
        default=None, description="the clean-player share a score must beat to flag"
    )
    players: list[PlayerReport]
    notes: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    scorer_trained_at: str | None = None
    judge_model: str | None = None
    judge_engine: str | None = Field(
        default=None, description="where the judge ran, e.g. 'GPU: ... via vulkan'"
    )


def _percentile_text(p: PlayerReport) -> str:
    if p.clean_percentile is None:
        return "-"
    if p.clean_percentile >= 0.99:
        return "top 1%"
    return f"top {max(1, round(100 * (1 - p.clean_percentile)))}%"


def render_text(report: MatchReport) -> str:
    """A terminal summary: the table first, then detail for flagged players only."""
    lines = [
        (
            f"{report.demo} · {report.map_name or 'unknown map'} · "
            f"{len(report.players)} players, {report.kills} kills · "
            f"{sum(report.timings.values()):.0f}s"
        ),
        f"line of sight: {report.visibility}",
        "",
        (
            f"   {'player':<22} {'kills':>5} {'score':>6}  {'vs clean':<9} "
            f"{'layer 1':<16} judge"
        ),
    ]
    for p in report.players:
        rules = ", ".join(sorted({e.name for e in p.rule_findings})) or "-"
        judge = (
            f"{p.verdict.verdict} {p.verdict.probability}%"
            if p.verdict
            else ("error" if p.judge_error else "-")
        )
        score = f"{p.score:.2f}" if p.score is not None else "-"
        if not p.enough_kills and p.score is not None:
            score += "*"
        mark = "▲" if p.flagged else " "
        lines.append(
            f" {mark} {(p.name or p.player_id)[:22]:<22} {p.kills:>5} {score:>6}  "
            f"{_percentile_text(p):<9} {rules[:16]:<16} {judge}"
        )
    if any(not p.enough_kills for p in report.players):
        lines.append("   * too few kills for the score to mean much")

    for p in (p for p in report.players if p.flagged):
        lines += ["", f"▲ {p.name or p.player_id}  ({p.player_id})"]
        lines.append(f"  flagged because: {'; '.join(p.flag_reasons)}")
        for e in p.rule_findings[:3]:
            tick = f" — {goto(e.tick)}" if e.tick is not None else ""
            lines.append(f"  [{e.severity}] {e.note}{tick}")
        if p.verdict:
            v = p.verdict
            # a cheat is named only alongside an accusation, as in training
            kind = f", {v.cheat_type}" if v.verdict == "cheating" else ""
            lines.append(f"  judge: {v.verdict} {v.probability}%{kind}")
            lines += [f"    - {r}" for r in v.reasons]
            lines += [f"    ? {c}" for c in v.caveats]
        elif p.judge_error:
            lines.append(f"  judge failed: {p.judge_error}")
        if p.top_kills:
            lines.append("  kills to watch:")
            for k in p.top_kills:
                what = " ".join(
                    x for x in (k.weapon, "headshot" if k.headshot else None) if x
                )
                cover = (
                    f", {k.aim_through_cover:.0%} aimed through cover"
                    if k.aim_through_cover
                    else ""
                )
                lines.append(
                    f"    tick {k.tick} {what}{cover} → {k.demo_command}"
                    + (f"\n      {k.context}" if k.context else "")
                )

    if report.notes:
        lines += ["", *[f"note: {n}" for n in report.notes]]
    return "\n".join(lines)


__all__ = [
    "KillReport",
    "MatchReport",
    "PlayerReport",
    "RadarPoint",
    "RoundSpan",
    "goto",
    "render_text",
]
