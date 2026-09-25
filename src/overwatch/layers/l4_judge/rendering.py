"""Turn a player's evidence into the text the judge model reads.

This module is the contract between the detector and the language model, and it
is used **unchanged at training and at inference**. If the fine-tuned model is
trained on text produced here and then shown text produced some other way, it
degrades quietly and the reports look plausible while being wrong.

Three rules shape the format:

1. **Numbers come with their baseline.** "reaction 94 ms" means nothing to a
   language model; "reaction 94 ms, typical clean player 210 ms" does. Every
   figure is paired with what clean players show, so the model is comparing
   rather than recalling.
2. **No label leaks in.** Nothing in the text says or implies whether this player
   was banned, including the ordering of cases.
3. **It stays short.** A mid-range CPU generates roughly 10 tokens a second, so a
   case is kept near 300 tokens and the verdict is a small JSON object. A report
   on ten players should take a minute, not twenty.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from overwatch.layers.l2_perception.awareness import describe
from overwatch.schemas.evidence import Evidence


class TrajectoryPoint(BaseModel):
    """Where the crosshair was, relative to the victim, at one instant."""

    ms: int = Field(description="milliseconds relative to the kill, negative is before")
    angle: float = Field(description="degrees between crosshair and the victim's head")
    visible: bool = Field(description="could this attacker see the victim")
    speed: float | None = Field(default=None, description="turn rate, deg/s")


class MomentSummary(BaseModel):
    """One kill, described the way a reviewer would look at it."""

    window_uid: str | None = None
    tick: int
    weapon: str | None = None
    headshot: bool | None = None
    distance: float | None = Field(
        default=None, description="metres between the players"
    )
    walls_penetrated: int | None = Field(
        default=None, description="walls the killing bullet passed through"
    )
    through_smoke: bool | None = None
    reaction_ms: float | None = Field(
        default=None, description="victim became visible -> kill"
    )
    angle_at_first_visible: float | None = Field(
        default=None, description="degrees off target when the victim appeared"
    )
    wall_aim_share: float | None = Field(
        default=None, description="share of the last 0.5s aimed at an unseen victim"
    )
    speed_at_kill: float | None = Field(
        default=None, description="deg/s on the kill tick"
    )
    corrections: int | None = Field(
        default=None, description="aim reversals in the flick"
    )
    visible_before_kill: bool | None = Field(
        default=None, description="was the victim ever visible before they died"
    )
    trajectory: list[TrajectoryPoint] = Field(
        default_factory=list,
        description="the approach to this kill, sampled every ~125 ms",
    )
    # what the shooter could legitimately have known (see l2_perception.awareness)
    sight_checked: bool = Field(
        default=False,
        description="line of sight was ray cast, so last_seen_ms is meaningful; "
        "without a map mesh a null there means 'unknown', not 'never seen'",
    )
    last_seen_ms: float | None = Field(
        default=None, description="last clear view of the victim before engaging"
    )
    victim_fired_ms_ago: float | None = None
    victim_was_running: bool | None = None
    allies_alive: int | None = None
    enemies_alive: int | None = None


class PlayerCase(BaseModel):
    """Everything the judge is allowed to consider about one player."""

    match_id: str
    player_id: str
    map_name: str | None = None
    rank: str | None = None
    kills: int
    score: float = Field(description="calibrated suspicion from the behaviour model")
    features: dict[str, float] = Field(default_factory=dict)
    baselines: dict[str, float] = Field(default_factory=dict)
    clean_lines: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="per measurement, where 95% and 99% of clean players stop; the "
        "training targets are decided by these, so the text shows them",
    )
    rule_evidence: list[Evidence] = Field(default_factory=list)
    moments: list[MomentSummary] = Field(default_factory=list)


#: How each feature is introduced, and which way is suspicious.
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "straight_share": ("flicks that never change direction", "higher"),
    "corrections_mean": ("aim corrections per kill", "lower"),
    "wall_aim_share": ("time aimed at an enemy they could not see", "higher"),
    "visible_share": ("time the victim was actually visible", "lower"),
    "reaction_ms_min": ("fastest reaction after an enemy appeared (ms)", "lower"),
    "angle_at_first_visible_median": (
        "degrees off target the moment the enemy appeared",
        "lower",
    ),
    "snap_max": ("fastest turn on a kill tick (deg/s)", "higher"),
    "zero_motion_share": ("ticks with the crosshair perfectly still", "higher"),
    "never_visible_share": ("kills where the victim was never visible first", "higher"),
    # context, not evidence: sniping inflates several measurements on its own
    "sniper_share": ("share of kills taken with a sniper rifle", "neither"),
}


#: A window starts 2 s before the kill, so a "reaction" at least that long really
#: means the enemy was already on screen when the window opened.
WINDOW_MS = 2000.0


def format_number(value: float) -> str:
    """How every measurement appears in the evidence. Reasons that cite one must use
    this too, or they quote a figure the model cannot find in its input."""
    if value == 0:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")


def unseen_numbers(sentence: str, evidence: str) -> list[str]:
    """Figures in a sentence that the evidence does not show.

    Whole numbers are compared, not substrings: "1.43" is not in an evidence that
    says "1.4". A share shown as 0.74 may be cited as 74%. Small counts ("2 of their
    3 kills") are skipped — too common to tell apart from an invention.
    """
    shown = set(_NUMBER.findall(evidence.replace(",", "")))
    shares = {n for n in shown if not n.endswith("%") and float(n) <= 1}
    shown |= {f"{round(float(n) * 100)}%" for n in shares}
    unseen = []
    for token in _NUMBER.findall(sentence.replace(",", "")):
        number = token.rstrip("%")
        if number.isdigit() and int(number) < 10:
            continue
        if token not in shown and number not in shown:
            unseen.append(token)
    return unseen


def render_case(case: PlayerCase) -> str:
    """Render one player's evidence as the model's input text."""
    lines = [
        f"PLAYER {case.player_id} — {case.kills} kills"
        + (f" on {case.map_name}" if case.map_name else "")
        + (f", lobby rank {case.rank}" if case.rank else ""),
        f"Behaviour model score: {case.score:.2f}",
        "",
        "MEASUREMENTS (player vs typical clean player)",
    ]

    for key, (description, suspicious_direction) in FEATURE_LABELS.items():
        if key not in case.features:
            continue
        value = case.features[key]
        baseline = case.baselines.get(key)
        line = f"- {description}: {format_number(value)}"
        edges = case.clean_lines.get(key)
        if baseline is not None and edges is not None:
            side = "below" if suspicious_direction == "higher" else "above"
            line += (
                f" (clean {format_number(baseline)}; 95% of clean players are {side} "
                f"{format_number(edges[0])}, 99% {side} {format_number(edges[1])})"
            )
        elif baseline is not None:
            line += f" (clean {format_number(baseline)}, {suspicious_direction} is suspicious)"
        lines.append(line)

    if case.rule_evidence:
        lines += ["", "HARD-LIMIT CHECKS TRIGGERED"]
        lines += [
            f"- [{e.severity}] {e.note} (tick {e.tick})" for e in case.rule_evidence
        ]

    if case.moments:
        lines += ["", "MOST SUSPICIOUS KILLS"]
        for n, moment in enumerate(case.moments, start=1):
            parts = [f"tick {moment.tick}"]
            if moment.weapon:
                parts.append(moment.weapon + (" headshot" if moment.headshot else ""))
            if moment.distance is not None:
                parts.append(f"{moment.distance:.0f} m away")
            if moment.walls_penetrated:
                parts.append(
                    "shot through a wall"
                    if moment.walls_penetrated == 1
                    else f"shot through {moment.walls_penetrated} walls"
                )
            if moment.through_smoke:
                parts.append("shot through smoke")
            if moment.reaction_ms is not None and moment.reaction_ms < WINDOW_MS - 10:
                parts.append(
                    f"killed {moment.reaction_ms:.0f} ms after the enemy appeared"
                )
            elif moment.reaction_ms is not None:
                parts.append("enemy was visible the whole 2 s before the kill")
            elif moment.visible_before_kill is False:
                parts.append("enemy never became visible before dying")
            if moment.angle_at_first_visible is not None:
                parts.append(
                    f"crosshair {moment.angle_at_first_visible:.1f}° off when they appeared"
                )
            if moment.wall_aim_share is not None:
                parts.append(
                    f"{moment.wall_aim_share:.0%} of the approach aimed through cover"
                )
            if moment.corrections is not None:
                plural = "" if moment.corrections == 1 else "s"
                parts.append(f"{moment.corrections} aim correction{plural}")
            lines.append(f"{n}. " + ", ".join(parts))
            context = describe(moment.model_dump())
            if context:
                # the innocent explanations, before the numbers that look damning
                lines.append(f"   context: {context}")
            lines += _render_trajectory(moment)

    return "\n".join(lines)


def _render_trajectory(moment: MomentSummary) -> list[str]:
    """The approach to a kill, as a small table the model can read tick by tick.

    A summary says the crosshair ended up on the target. This shows *how*: whether
    it drifted onto a target it could not see, or found one after it appeared. That
    difference is the whole judgement, and it is invisible in aggregates.
    """
    if not moment.trajectory:
        return []
    rows = ["     time   off-target  enemy"]
    for point in moment.trajectory:
        marker = "  <- shot" if point.ms == 0 else ""
        seen = "visible" if point.visible else "hidden"
        rows.append(f"   {point.ms:+6d}ms {point.angle:7.1f}°  {seen}{marker}")
    return rows


SYSTEM_PROMPT = """You review Counter-Strike 2 players for cheating, like a human juror.

You are given measurements for one player, compared with what clean players show, \
and their most suspicious kills. Decide whether the evidence supports cheating.

Rules:
- Judge only from the evidence given. Never invent numbers, kills or context.
- Strong players are not cheaters. Fast reactions and high accuracy are normal at \
high skill; aiming at enemies they cannot see, and aim that never corrects itself, \
are not.
- A single unusual kill is weak evidence. A pattern across many kills is strong.
- If the evidence is thin or mixed, say so and answer "unclear".

Reply with JSON only:
{"verdict": "cheating" | "clean" | "unclear", "probability": 0-100, \
"cheat_type": "aimbot" | "wallhack" | "triggerbot" | "mixed" | "none", \
"reasons": ["short sentence citing a measurement", ...], \
"caveats": ["what would change your mind", ...]}

"probability" is how certain you are that this player cheated, as a percentage. \
Use the full range: 5 for a clearly clean player, 50 when the evidence is genuinely \
mixed, 95 when several independent measurements all point the same way. Do not \
simply repeat the behaviour model score — it is one piece of evidence among several, \
and it can be wrong."""
