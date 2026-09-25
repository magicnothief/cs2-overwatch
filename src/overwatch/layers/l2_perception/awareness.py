"""What the shooter could legitimately have known about the victim.

The wall-aim measurement asks whether someone aimed at an enemy they could not
see. A human reviewer immediately asks the next question: *did he already know
where that enemy was?* There are three innocent ways to know:

    he saw them a moment ago   and they have not moved far since
    he heard them shoot        gunfire carries across the map
    he heard them move         footsteps carry about 1100 units

Without these, a player holding an angle on someone he watched walk behind a wall
is indistinguishable from a wallhacker — and that is the most common way an
honest player looks guilty. Everything here is context for the judge rather than
evidence against the player.
"""

from __future__ import annotations

import polars as pl

from overwatch.aim import TICK_RATE
from overwatch.layers.l2_perception.geometry.occlusion import has_map, line_of_sight

#: Footsteps are audible to about this range in CS2 (Hammer units).
FOOTSTEP_RANGE = 1100.0
#: Below this speed a player is walking or crouching, and makes no footstep noise.
AUDIBLE_SPEED = 130.0
#: Gunfire carries much further than footsteps.
GUNFIRE_RANGE = 3000.0
#: How far back to look for the last sighting, and how coarsely.
LOOKBACK_SECONDS = 10
LOOKBACK_STRIDE = 8
#: The engagement itself is excluded: whether they could see the victim while
#: shooting at them is already measured. The question here is what they knew
#: *before* they started tracking — so sampling stops two seconds out.
ENGAGEMENT_SECONDS = 2


def last_seen(
    windows: pl.DataFrame,
    match_ticks: pl.DataFrame,
    map_name: str | None,
    *,
    lookback_seconds: int = LOOKBACK_SECONDS,
) -> pl.DataFrame:
    """How long before each kill the attacker last had a clear view of the victim.

    Sampled every eighth tick from ten seconds out to two seconds before the kill:
    fine enough to tell "watched them walk behind that wall" from "never saw them",
    cheap enough to run over the whole dataset. The final two seconds are excluded
    because visibility during the engagement is already measured tick by tick —
    including them made this read 0 ms for almost every kill.

    Returns one row per window_id with `last_seen_ms` (null when the victim was not
    visible at any point in that span).
    """
    if windows.is_empty() or not map_name or not has_map(map_name):
        return windows.select("window_id").with_columns(
            last_seen_ms=pl.lit(None, pl.Float64)
        )

    offsets = pl.DataFrame(
        {
            "tick_offset": pl.Series(
                range(
                    -lookback_seconds * TICK_RATE,
                    -ENGAGEMENT_SECONDS * TICK_RATE + 1,
                    LOOKBACK_STRIDE,
                ),
                dtype=pl.Int32,
            )
        }
    )
    samples = (
        windows.select(
            "window_id", "attacker_id", "victim_id", kill_tick=pl.col("tick")
        )
        .join(offsets, how="cross")
        .with_columns(tick=(pl.col("kill_tick") + pl.col("tick_offset")).cast(pl.Int32))
    )

    positions = match_ticks.select("player_id", "tick", "x", "y", "z")
    samples = (
        samples.join(
            positions, left_on=["attacker_id", "tick"], right_on=["player_id", "tick"]
        )
        .join(
            positions.rename({"x": "vx", "y": "vy", "z": "vz"}),
            left_on=["victim_id", "tick"],
            right_on=["player_id", "tick"],
        )
        .sort(["window_id", "tick_offset"])
    )
    if samples.is_empty():
        return windows.select("window_id").with_columns(
            last_seen_ms=pl.lit(None, pl.Float64)
        )

    visible = line_of_sight(
        samples.select("x", "y", "z").to_numpy(),
        samples.select("vx", "vy", "vz").to_numpy(),
        map_name,
    )
    return (
        samples.with_columns(visible=pl.Series(visible))
        .filter(pl.col("visible"))
        .group_by("window_id")
        .agg(last_seen_ms=-pl.col("tick_offset").max() / TICK_RATE * 1000)
        .join(windows.select("window_id"), on="window_id", how="right")
    )


def audible_cues(
    windows: pl.DataFrame,
    match_ticks: pl.DataFrame,
    weapon_fire: pl.DataFrame,
    *,
    window_seconds: float = 2.0,
) -> pl.DataFrame:
    """Whether the victim gave themselves away by sound before dying.

    Returns one row per window_id:
        victim_fired_ms_ago   the victim's last shot before this kill
        victim_was_running    they were moving fast enough to be heard, in range
    """
    span = int(window_seconds * TICK_RATE)
    base = windows.select(
        "window_id", "attacker_id", "victim_id", kill_tick=pl.col("tick")
    )

    fired = pl.DataFrame({"window_id": [], "victim_fired_ms_ago": []}).cast(
        {"window_id": pl.UInt32, "victim_fired_ms_ago": pl.Float64}
    )
    shooter_column = next(
        (
            c
            for c in ("player_id", "user_steamid", "attacker_id")
            if c in weapon_fire.columns
        ),
        None,
    )
    if shooter_column and not weapon_fire.is_empty():
        shots = weapon_fire.select(
            victim_id=pl.col(shooter_column).cast(pl.String), shot_tick=pl.col("tick")
        )
        fired = (
            base.join(shots, on="victim_id")
            .filter(
                (pl.col("shot_tick") <= pl.col("kill_tick"))
                & (pl.col("shot_tick") >= pl.col("kill_tick") - span)
            )
            .group_by("window_id")
            .agg(
                victim_fired_ms_ago=(
                    pl.col("kill_tick").first() - pl.col("shot_tick").max()
                )
                / TICK_RATE
                * 1000
            )
        )

    # movement noise: the victim running, within earshot of the attacker
    speeds = match_ticks.sort(["player_id", "tick"]).with_columns(
        speed=(
            (
                pl.col("x").diff().over("player_id") ** 2
                + pl.col("y").diff().over("player_id") ** 2
            ).sqrt()
            * TICK_RATE
        )
    )
    moving = (
        base.join(
            speeds.select("player_id", "tick", "speed", "x", "y", "z"),
            left_on=["victim_id"],
            right_on=["player_id"],
        )
        .filter(
            (pl.col("tick") <= pl.col("kill_tick"))
            & (pl.col("tick") >= pl.col("kill_tick") - span)
        )
        .join(
            match_ticks.select(
                "player_id", "tick", ax=pl.col("x"), ay=pl.col("y"), az=pl.col("z")
            ),
            left_on=["attacker_id", "tick"],
            right_on=["player_id", "tick"],
        )
        .with_columns(
            distance=(
                (pl.col("x") - pl.col("ax")) ** 2
                + (pl.col("y") - pl.col("ay")) ** 2
                + (pl.col("z") - pl.col("az")) ** 2
            ).sqrt()
        )
        .group_by("window_id")
        .agg(
            victim_was_running=(
                (pl.col("speed") > AUDIBLE_SPEED)
                & (pl.col("distance") < FOOTSTEP_RANGE)
            ).any()
        )
    )

    return (
        base.select("window_id")
        .join(fired, on="window_id", how="left")
        .join(moving, on="window_id", how="left")
        .with_columns(
            victim_was_running=pl.col("victim_was_running").fill_null(value=False)
        )
    )


def alive_counts(windows: pl.DataFrame, match_ticks: pl.DataFrame) -> pl.DataFrame:
    """How many players were still alive on each side when the kill happened."""
    alive = (
        match_ticks.filter(pl.col("is_alive"))
        .group_by("tick", "team")
        .agg(alive=pl.len())
    )
    attacker_team = match_ticks.select("player_id", "tick", "team").unique(
        subset=["player_id", "tick"]
    )
    base = windows.select("window_id", "attacker_id", kill_tick=pl.col("tick"))
    return (
        base.join(
            attacker_team,
            left_on=["attacker_id", "kill_tick"],
            right_on=["player_id", "tick"],
        )
        .join(
            alive, left_on=["kill_tick", "team"], right_on=["tick", "team"], how="left"
        )
        .rename({"alive": "allies_alive"})
        .join(
            alive.rename({"alive": "enemies_alive"}),
            left_on="kill_tick",
            right_on="tick",
            how="left",
        )
        .filter(pl.col("team") != pl.col("team_right"))
        .select("window_id", "allies_alive", "enemies_alive")
        .unique(subset=["window_id"])
    )


def add_context(
    windows: pl.DataFrame,
    match_ticks: pl.DataFrame,
    events: dict[str, pl.DataFrame],
    map_name: str | None,
) -> pl.DataFrame:
    """Attach everything a reviewer would want to know about each kill."""
    if windows.is_empty():
        return windows
    return (
        windows.join(
            last_seen(windows, match_ticks, map_name), on="window_id", how="left"
        )
        .join(
            audible_cues(
                windows, match_ticks, events.get("weapon_fire", pl.DataFrame())
            ),
            on="window_id",
            how="left",
        )
        .join(alive_counts(windows, match_ticks), on="window_id", how="left")
    )


def describe(row: dict) -> str:
    """One line a human can read: what the shooter could have known.

    Sightings are only mentioned when `sight_checked` is set. On a map without a
    mesh `last_seen_ms` is null because nothing was measured, and reading that as
    "never saw them" would hand the judge a false accusation.
    """
    parts = []
    last = row.get("last_seen_ms")
    if row.get("sight_checked") and last is None:
        parts.append(
            f"had not seen this enemy at any point in the {LOOKBACK_SECONDS}s "
            "before the engagement"
        )
    elif row.get("sight_checked"):
        parts.append(f"last saw this enemy {last / 1000:.1f}s before the kill")

    fired = row.get("victim_fired_ms_ago")
    if fired is not None:
        parts.append(f"the enemy had fired {fired / 1000:.1f}s before")
    if row.get("victim_was_running"):
        parts.append("the enemy was running within earshot")
    if row.get("allies_alive") is not None:
        parts.append(f"{row['allies_alive']}v{row.get('enemies_alive')} at the time")
    return "; ".join(parts)


__all__ = [
    "AUDIBLE_SPEED",
    "FOOTSTEP_RANGE",
    "GUNFIRE_RANGE",
    "add_context",
    "alive_counts",
    "audible_cues",
    "describe",
    "last_seen",
]
