"""Layer 1: things that are impossible, not merely unusual.

These rules exist to be nearly always right, not to catch everyone. Each one
compares a measurement against a line taken from the clean population (see
thresholds.py), so a hit means "no clean player in the reference set did this",
not "this looks odd to me".

They also cover a case the models cannot: a fast enough spin aliases at 64 ticks
and comes back as small, erratic turns, so spinning is detected by its pattern
rather than by speed alone.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from overwatch.aim import MAX_MEASURABLE_SPEED, TICK_RATE, add_aim_features
from overwatch.layers.l1_blatant.thresholds import Thresholds
from overwatch.schemas.evidence import Evidence, Moment, Severity

Rule = Callable[[pl.DataFrame, pl.DataFrame, Thresholds, str], list[Moment]]


def _runs_above(frame: pl.DataFrame, condition: pl.Expr, min_len: int) -> pl.DataFrame:
    """Find runs of consecutive ticks where `condition` holds, per player.

    Returns one row per run: player_id, tick_start, tick_end, length.
    """
    flagged = frame.with_columns(hit=condition.fill_null(value=False))
    grouped = (
        flagged.sort(["player_id", "tick"])
        .with_columns(
            run_id=(pl.col("hit") != pl.col("hit").shift(1).over("player_id"))
            .cum_sum()
            .over("player_id")
        )
        .filter(pl.col("hit"))
        .group_by("player_id", "run_id")
        .agg(
            tick_start=pl.col("tick").min(),
            tick_end=pl.col("tick").max(),
            length=pl.len(),
            peak=pl.col("_metric").max(),
        )
    )
    return grouped.filter(pl.col("length") >= min_len).sort(["player_id", "tick_start"])


def spinbot(
    ticks: pl.DataFrame, deaths: pl.DataFrame, thresholds: Thresholds, match_id: str
) -> list[Moment]:
    """A view that keeps turning far faster than a hand can, for long enough to be deliberate.

    Speed is measured as a rolling median over the run length, so a single tick of
    noise (or one wrapped angle after a gap) cannot trigger the rule: the turn has
    to actually be sustained.
    """
    frame = ticks.sort(["player_id", "tick"]).with_columns(
        _metric=pl.col("yaw_speed")
        .fill_null(0)
        .rolling_median(window_size=thresholds.spin_min_ticks)
        .over("player_id")
    )
    runs = _runs_above(
        frame,
        pl.col("_metric") > thresholds.spin_speed_dps,
        thresholds.spin_min_ticks,
    )
    return [
        Moment(
            match_id=match_id,
            player_id=row["player_id"],
            tick_start=row["tick_start"],
            tick_end=row["tick_end"],
            trigger="spinbot",
            evidence=[
                Evidence(
                    layer="l1_blatant",
                    name="sustained_spin",
                    value=round(row["peak"], 1),
                    unit="deg/s",
                    threshold=thresholds.spin_speed_dps,
                    baseline=thresholds.clean_speed_p9999,
                    severity=Severity.IMPOSSIBLE,
                    tick=row["tick_start"],
                    note=(
                        f"view spun at a sustained {row['peak']:.0f} deg/s for "
                        f"{row['length'] / TICK_RATE:.2f}s; the line is "
                        f"{thresholds.spin_speed_dps:.0f} deg/s"
                    ),
                )
            ],
        )
        for row in runs.iter_rows(named=True)
    ]


def aliased_spin(
    ticks: pl.DataFrame, deaths: pl.DataFrame, thresholds: Thresholds, match_id: str
) -> list[Moment]:
    """A spin too fast to measure: large alternating turns, tick after tick.

    Above 180 deg per tick the recorded angle wraps, so a very fast spin reads as
    a sequence of big turns that flip direction almost every tick. No hand does
    that; a hand slows down and reverses at most a few times a second.
    """
    frame = ticks.sort(["player_id", "tick"]).with_columns(
        _flip=(
            (pl.col("d_yaw").abs() > thresholds.alias_min_step_deg)
            & (
                pl.col("d_yaw").sign()
                != pl.col("d_yaw").sign().shift(1).over("player_id")
            )
        ),
        _metric=pl.col("d_yaw").abs(),
    )
    runs = _runs_above(frame, pl.col("_flip"), thresholds.alias_min_ticks)
    return [
        Moment(
            match_id=match_id,
            player_id=row["player_id"],
            tick_start=row["tick_start"],
            tick_end=row["tick_end"],
            trigger="aliased_spin",
            evidence=[
                Evidence(
                    layer="l1_blatant",
                    name="aliased_spin",
                    value=float(row["length"]),
                    unit="ticks",
                    threshold=float(thresholds.alias_min_ticks),
                    severity=Severity.IMPOSSIBLE,
                    tick=row["tick_start"],
                    note=(
                        f"view reversed direction by more than "
                        f"{thresholds.alias_min_step_deg:.0f} deg on {row['length']} "
                        f"consecutive ticks: a spin faster than "
                        f"{MAX_MEASURABLE_SPEED:.0f} deg/s reads this way"
                    ),
                )
            ],
        )
        for row in runs.iter_rows(named=True)
    ]


def kill_tick_snap(
    ticks: pl.DataFrame, deaths: pl.DataFrame, thresholds: Thresholds, match_id: str
) -> list[Moment]:
    """The crosshair arrives on the target in the same tick the shot lands."""
    if deaths.is_empty():
        return []
    at_kill = deaths.join(
        ticks.select("player_id", "tick", "yaw_speed"),
        left_on=["attacker_id", "tick"],
        right_on=["player_id", "tick"],
        how="inner",
    ).filter(pl.col("yaw_speed") > thresholds.snap_speed_dps)

    return [
        Moment(
            match_id=match_id,
            player_id=row["attacker_id"],
            tick_start=row["tick"] - TICK_RATE,
            tick_end=row["tick"] + TICK_RATE // 2,
            trigger="kill_tick_snap",
            evidence=[
                Evidence(
                    layer="l1_blatant",
                    name="kill_tick_snap",
                    value=round(row["yaw_speed"], 1),
                    unit="deg/s",
                    threshold=thresholds.snap_speed_dps,
                    baseline=thresholds.clean_snap_p9999,
                    severity=Severity.STRONG,
                    tick=row["tick"],
                    note=(
                        f"crosshair moved {row['yaw_speed']:.0f} deg/s on the tick the kill "
                        f"landed, against {thresholds.clean_snap_p9999:.0f} deg/s for the "
                        "most extreme clean kills"
                    ),
                )
            ],
        )
        for row in at_kill.iter_rows(named=True)
    ]


def impossible_pitch(
    ticks: pl.DataFrame, deaths: pl.DataFrame, thresholds: Thresholds, match_id: str
) -> list[Moment]:
    """Looking further up or down than the game allows a player to."""
    frame = ticks.with_columns(_metric=pl.col("pitch").abs())
    runs = _runs_above(frame, pl.col("pitch").abs() > thresholds.max_pitch_deg, 1)
    return [
        Moment(
            match_id=match_id,
            player_id=row["player_id"],
            tick_start=row["tick_start"],
            tick_end=row["tick_end"],
            trigger="impossible_pitch",
            evidence=[
                Evidence(
                    layer="l1_blatant",
                    name="impossible_pitch",
                    value=round(row["peak"], 2),
                    unit="deg",
                    threshold=thresholds.max_pitch_deg,
                    severity=Severity.IMPOSSIBLE,
                    tick=row["tick_start"],
                    note=(
                        f"pitch reached {row['peak']:.1f} deg; the game clamps a player's "
                        f"view at {thresholds.max_pitch_deg:.0f} deg"
                    ),
                )
            ],
        )
        for row in runs.iter_rows(named=True)
    ]


RULES: dict[str, Rule] = {
    "spinbot": spinbot,
    "aliased_spin": aliased_spin,
    "kill_tick_snap": kill_tick_snap,
    "impossible_pitch": impossible_pitch,
}


def run_rules(
    ticks: pl.DataFrame,
    deaths: pl.DataFrame,
    *,
    thresholds: Thresholds | None = None,
    match_id: str = "",
    rules: dict[str, Rule] | None = None,
    with_aim_features: bool = True,
) -> list[Moment]:
    """Run every Layer 1 rule over one match.

    Args:
        ticks: canonical ticks; aim features are added unless already present.
        deaths: normalized player_death table.
        thresholds: lines taken from the clean population; defaults to the
            calibrated set shipped with the package.
        match_id: recorded on every Moment.
        rules: subset of RULES to run.
        with_aim_features: set False if ticks already went through add_aim_features().
    """
    thresholds = thresholds or Thresholds.load()
    if with_aim_features:
        ticks = add_aim_features(ticks)

    moments: list[Moment] = []
    for rule in (rules or RULES).values():
        moments.extend(rule(ticks, deaths, thresholds, match_id))
    return sorted(moments, key=lambda m: (m.player_id, m.tick_start))


def evidence_rows(moments: list[Moment]) -> list[dict]:
    """Moments flattened to one row per piece of evidence, for tables and cases."""
    return [
        {
            "match_id": moment.match_id,
            "player_id": moment.player_id,
            "trigger": moment.trigger,
            "tick_start": moment.tick_start,
            "tick_end": moment.tick_end,
            **item.model_dump(mode="json"),
        }
        for moment in moments
        for item in moment.evidence
    ]
