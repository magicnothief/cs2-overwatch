"""The verdict a case *should* get: training targets for the judge.

There is no corpus of "CS2 evidence -> expert verdict", so targets are written
from the evidence. Each one cites only numbers that appear in the case, which is
what stops the model inventing figures later.

Four things are deliberately taught:

1. **Say what this match shows, and nothing else.** A target depends only on the
   evidence text; the ban label is used to score the judge, never to write its
   answers. The first training set let the label in ("unclear" for a banned
   player whose match showed nothing, "clean" for a clean one), and the
   fine-tune learned to guess the label from the behaviour-score line instead of
   reading the measurements: its probability tracked that score at r=0.75, and it
   ranked players worse than the score alone (AUC 0.819 against 0.884).
2. **Say "unclear" when the evidence is thin.** Some evidence past the clean
   range, but not enough to decide, gets "unclear" with a probability that rises
   with it; none at all gets "clean".
3. **Do not condemn AWP mains for sniping.** Cheaters in this dataset snipe far
   more than clean players (ADR 0008), so for a heavy sniper the visibility
   evidence does not count towards a verdict, and a caveat names the weapon.
4. **Reasons quote measurements.** Every reason is generated from a specific
   number and its clean baseline, so "cites the evidence" is the only style the
   model ever sees.

Human annotations (overwatch.annotation) replace these targets case by case; the
generated ones are the fallback and the yardstick annotations are compared with.
"""

from __future__ import annotations

import random
import zlib

import polars as pl

from overwatch.layers.l4_judge.rendering import (
    FEATURE_LABELS,
    MomentSummary,
    PlayerCase,
    format_number,
)
from overwatch.layers.l4_judge.verdict import CheatType, Verdict, VerdictLabel

#: A measurement is judged by how rare it is among clean players, the way Layer 1
#: sets its lines: past what 95% of them show is notable, past 99% is strong. This
#: used to be "times the clean median", which called a 94 ms fastest reaction
#: strong (clean median 234 ms) although a third of clean players have one, and
#: gave 62% of clean players an "unclear" target.
NOTABLE_QUANTILE = 0.95
STRONG_QUANTILE = 0.99
#: What each level adds towards a verdict. One strong measurement, or three notable
#: ones agreeing, reach DECISIVE — true of 32% of banned players and 3% of clean
#: ones on the measurements alone.
WEIGHTS = {"notable": 0.4, "strong": 1.0}
DECISIVE = 1.0
#: Layer 1 hits sit outside the clean range by construction (thresholds.py).
RULE_WEIGHTS = {"impossible": 2.0, "strong": 1.0}

#: A player with this much sniping gets the weapon caveat rather than a verdict.
SNIPER_HEAVY = 0.5
#: A kill counts as "aimed through cover" from this share of the approach upwards.
WALL_AIM_KILL = 0.5

#: Where "notable" and "strong" begin for each feature: (notable, strong).
Lines = dict[str, tuple[float, float]]

#: (feature, higher is suspicious, cheat it suggests, the sentence citing it)
CHECKS: list[tuple[str, bool, str, str]] = [
    (
        "wall_aim_share",
        True,
        "wallhack",
        (
            "they held the crosshair on an enemy they could not see for {value} of "
            "the approach, against {base} for clean players"
        ),
    ),
    (
        "never_visible_share",
        True,
        "wallhack",
        (
            "{value} of their kills were on enemies who never became visible "
            "beforehand, against {base} for clean players"
        ),
    ),
    (
        "straight_share",
        True,
        "aimbot",
        (
            "{value} of their flicks never corrected direction, against {base} "
            "for clean players"
        ),
    ),
    (
        "corrections_mean",
        False,
        "aimbot",
        (
            "they averaged {value} aim corrections per kill, against {base} for "
            "clean players"
        ),
    ),
    (
        "reaction_ms_min",
        False,
        "aimbot",
        (
            "they killed {value} ms after an enemy appeared, against {base} ms "
            "for clean players"
        ),
    ),
    (
        "snap_max",
        True,
        "aimbot",
        (
            "they turned {value} deg/s on the tick a kill landed, against "
            "{base} for clean players"
        ),
    ),
]


def case_rng(match_id: str, player_id: str, seed: int = 0) -> random.Random:
    """The random source for one case's target wording and probability.

    Seeded by the case itself, so a case gets the same target whichever order the
    cases are processed in — the annotation tool shows the very target the
    training set contains.
    """
    return random.Random(zlib.crc32(f"{match_id}/{player_id}".encode()) ^ seed)


def _phrase(rng: random.Random, options: list[str]) -> str:
    """Vary the wording so the model learns the reasoning, not the sentence."""
    return rng.choice(options)


def cited(value: float, *, percent: bool = False) -> str:
    """A number as the evidence shows it, or as a percentage of exactly that.

    Derived from the rendered figure rather than the raw value, so a reason can
    never quote "1.43" where the model was shown "1.4".
    """
    shown = format_number(value)
    return f"{round(float(shown) * 100)}%" if percent else shown


def clean_lines(players: pl.DataFrame) -> Lines:
    """Where 95% and 99% of clean players stop, for every measurement the judge sees
    that has a suspicious direction. The case carries these and the text shows them.
    """
    clean = players.filter(pl.col("label") == "clean")
    lines: Lines = {}
    for key, (_, direction) in FEATURE_LABELS.items():
        if direction not in ("higher", "lower") or key not in clean.columns:
            continue
        notable, strong = (
            (NOTABLE_QUANTILE, STRONG_QUANTILE)
            if direction == "higher"
            else (1 - NOTABLE_QUANTILE, 1 - STRONG_QUANTILE)
        )
        lines[key] = (
            float(clean[key].quantile(notable)),
            float(clean[key].quantile(strong)),
        )
    return lines


def shown(value: float) -> float:
    """A number as the evidence text shows it, so every decision can be read there."""
    return float(format_number(value))


def level(
    value: float, lines: tuple[float, float], *, higher_is_worse: bool
) -> str | None:
    """ "strong", "notable" or None: how far outside the clean range a value sits."""
    notable, strong = lines

    def past(line: float) -> bool:
        return value > line if higher_is_worse else value < line

    if past(strong):
        return "strong"
    if past(notable):
        return "notable"
    return None


def evidence_points(case: PlayerCase) -> list[tuple[str, float, str]]:
    """Which measurements are worth citing, heaviest first.

    Decided by the case's own clean lines, compared as the text displays them, so
    everything a target concludes can be worked out from the text the model reads.

    Returns (reason sentence, weight towards a verdict, which cheat it suggests).
    """
    points: list[tuple[str, float, str]] = []
    for key, higher_is_worse, cheat, template in CHECKS:
        if key not in case.features or key not in case.clean_lines:
            continue
        value = case.features[key]
        edges = tuple(shown(v) for v in case.clean_lines[key])
        found = level(shown(value), edges, higher_is_worse=higher_is_worse)
        if found is None:
            continue
        base = case.baselines.get(key, case.clean_lines[key][0])
        percent = key.endswith("_share")
        sentence = template.format(
            value=cited(value, percent=percent), base=cited(base, percent=percent)
        )
        points.append((sentence, WEIGHTS[found], cheat))

    points += awareness_points(case)

    for item in case.rule_evidence:
        weight = RULE_WEIGHTS.get(item.severity, WEIGHTS["notable"])
        points.append((f"a hard-limit check fired: {item.note}", weight, "aimbot"))

    return sorted(points, key=lambda item: -item[1])


def aimed_through_cover(moment: MomentSummary) -> bool:
    return moment.sight_checked and (moment.wall_aim_share or 0) >= WALL_AIM_KILL


def seen_or_heard(moment: MomentSummary) -> bool:
    """Whether the context line gives an innocent way to know where they were."""
    return (
        moment.last_seen_ms is not None
        or moment.victim_fired_ms_ago is not None
        or bool(moment.victim_was_running)
    )


def awareness_points(case: PlayerCase) -> list[tuple[str, float, str]]:
    """Aim through cover at someone the shooter had neither seen nor heard.

    On ray-cast maps this is 5.9x more common for cheaters per kill (14.3% vs 2.4%),
    against 4.0x for aiming through cover at all, because it strips out the honest
    way to do it: holding an angle on an enemy you just watched walk behind a wall,
    or heard shooting. Per player, two of the three most suspicious kills like this
    is true of 24% of banned players and 3.5% of clean ones — notable, not strong.
    One such kill is true of 19% of clean players, so it is not cited at all.
    """
    unexplained = [
        m for m in case.moments if aimed_through_cover(m) and not seen_or_heard(m)
    ]
    if len(unexplained) < 2:
        return []
    return [
        (
            (
                "they tracked an enemy through cover without having seen or heard them "
                f"first on {len(unexplained)} of their {len(case.moments)} most "
                "suspicious kills"
            ),
            WEIGHTS["notable"],
            "wallhack",
        )
    ]


def awareness_caveat(case: PlayerCase) -> str | None:
    """When every kill aimed through cover has an innocent explanation, say so."""
    covered = [m for m in case.moments if aimed_through_cover(m)]
    if covered and all(seen_or_heard(m) for m in covered):
        return (
            "their aim through cover came after they had seen or heard the enemy, "
            "which a clean player can do"
        )
    return None


#: Probability bands the verdict word must agree with. A target saying "unclear"
#: at 5% teaches contradiction, which is worse than teaching nothing.
CLEAN_BELOW = 25
CHEATING_ABOVE = 65


def verdict_for(probability: int) -> VerdictLabel:
    if probability < CLEAN_BELOW:
        return VerdictLabel.CLEAN
    if probability > CHEATING_ABOVE:
        return VerdictLabel.CHEATING
    return VerdictLabel.UNCLEAR


def standard_caveats(case: PlayerCase) -> list[str]:
    """Facts about a case that should temper any verdict on it."""
    caveats: list[str] = []
    if case.kills < 10:
        caveats.append(f"only {case.kills} kills to judge from")
    if explained := awareness_caveat(case):
        caveats.append(explained)
    if case.features.get("sniper_share", 0.0) >= SNIPER_HEAVY:
        caveats.append(
            "this player snipes heavily, which inflates the visibility measurements "
            "on its own"
        )
    return caveats


def split_for(match_id: str, val_share: float = 0.2) -> str:
    """ "train" or "val", fixed per match.

    Every player of a match lands in the same half, so nothing about a match leaks
    from training into validation, and the half is known from the match alone — the
    annotation tool can put validation cases first without building the whole set.
    """
    return "val" if zlib.crc32(match_id.encode()) % 1000 < val_share * 1000 else "train"


#: What an evidence-only target states. Deterministic in the evidence, so the
#: model learns the mapping instead of guessing at noise.
CLEAN_PROBABILITY = 10
FEW_KILLS_PROBABILITY = 15  # nothing unusual, but little to judge from
UNCLEAR_PROBABILITY = (30, 60)  # rises with the weight of the evidence


def target_probability(weight: float, kills: int) -> int:
    """How sure a reviewer should be, from how much evidence the text holds."""
    if weight >= DECISIVE:
        return min(95, 62 + round(10 * weight))
    if weight > 0:
        low, high = UNCLEAR_PROBABILITY
        return low + round((high - low) * weight / DECISIVE)
    return FEW_KILLS_PROBABILITY if kills < 10 else CLEAN_PROBABILITY


def build_target(case: PlayerCase, rng: random.Random) -> Verdict:
    """The verdict this case should produce, from its evidence alone.

    There is deliberately no ban label here (see the module docstring). The
    probability is decided first and the verdict word follows from it; `rng` only
    varies the wording of reasons that do not cite a measurement.
    """
    points = evidence_points(case)
    sniper_heavy = case.features.get("sniper_share", 0.0) >= SNIPER_HEAVY
    # sniping inflates the perception measurements, so for a heavy sniper only the
    # aim-shape evidence counts towards a verdict (ADR 0008)
    counted = [p for p in points if not sniper_heavy or p[2] != "wallhack"]
    weight = sum(p[1] for p in counted)

    caveats = standard_caveats(case)
    probability = target_probability(weight, case.kills)
    verdict = verdict_for(probability)

    if verdict is VerdictLabel.CHEATING:
        cheats = {p[2] for p in counted}
        cheat_type = (
            CheatType.MIXED if len(cheats) > 1 else CheatType(next(iter(cheats)))
        )
        reasons = [text for text, _, _ in counted[:3]]
    elif verdict is VerdictLabel.UNCLEAR:
        cheat_type = CheatType.NONE
        reasons = [
            _phrase(
                rng,
                [
                    "the evidence points both ways",
                    "some measurements are unusual, but not beyond what skill explains",
                ],
            )
        ]
        if points:
            reasons.append(f"the strongest signal: {points[0][0]}")
    else:
        cheat_type = CheatType.NONE
        reasons = [
            _phrase(
                rng,
                [
                    "every measurement sits inside the range clean players produce",
                    "nothing here is outside ordinary play",
                ],
            )
        ]

    return Verdict(
        verdict=verdict,
        probability=probability,
        cheat_type=cheat_type,
        reasons=reasons[:3],
        caveats=caveats[:2],
    )
