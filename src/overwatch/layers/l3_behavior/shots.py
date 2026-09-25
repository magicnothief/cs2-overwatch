"""When a player fires, against where their crosshair is: triggerbot timing.

A triggerbot fires the moment the crosshair is on an enemy. A person reacts to
the crosshair arriving, a few ticks later, or fires as it sweeps across and lands
a little before. So, per kill:

    opening shot   the first shot of the last burst before the kill (a burst
                   ends when BURST_GAP ticks pass without a shot); only it
                   counts, since mid-spray a shot every ~6 ticks would land on a
                   re-arrival by chance
    arrival        the last tick, at or before the opening shot, on which the
                   crosshair came onto the enemy's head (within HEAD_RADIUS world
                   units of its centre, as an angle for their distance)
    arrival_shot   the opening shot came on that very tick

On CS2CD (spec: docs/specs/2026-09-26-triggerbot-and-snap-count.md), no clean
player of 583 fired on the arrival tick on over half their kills; 11% of banned
players did. Kills where the crosshair was already on the spot (pre-aimed) have
no arrival and are not counted either way.
"""

from __future__ import annotations

import math

import polars as pl

from overwatch.aim import TICK_RATE

#: Around the head's centre, in world units: the head and neck.
HEAD_RADIUS = 8.0
#: Closer than this the angle stops meaning much; the radius is measured here.
MIN_DISTANCE = 50.0
#: Ticks without a shot that end a burst (300 ms).
BURST_GAP = 19
#: Weapons whose "shot" is not an aimed one.
NOT_AIMED = "knife|bayonet|grenade|molotov|flashbang|decoy|c4|healthshot"

SHOT_COLUMNS: tuple[str, ...] = ("arrival_shot", "arrival_delay_ms")


def mark_shots(
    window_ticks: pl.DataFrame, weapon_fire: pl.DataFrame | None
) -> pl.DataFrame:
    """Add `shot`: did this window's attacker fire an aimed weapon on this tick?"""
    shooter = None
    if weapon_fire is not None and not weapon_fire.is_empty():
        shooter = next(
            (
                c
                for c in ("player_id", "user_steamid", "attacker_id")
                if c in weapon_fire.columns
            ),
            None,
        )
    if shooter is None:
        return window_ticks.with_columns(shot=pl.lit(False))
    aimed = weapon_fire
    if "weapon" in weapon_fire.columns:
        aimed = weapon_fire.filter(
            ~pl.col("weapon").cast(pl.String).str.contains(NOT_AIMED).fill_null(False)
        )
    fired = aimed.select(
        player_id=pl.col(shooter).cast(pl.String),
        tick=pl.col("tick").cast(window_ticks.schema["tick"]),
        shot=pl.lit(True),
    ).unique(subset=["player_id", "tick"])
    return window_ticks.join(fired, on=["player_id", "tick"], how="left").with_columns(
        shot=pl.col("shot").fill_null(False)
    )


def arrival_features(window_ticks: pl.DataFrame) -> pl.DataFrame:
    """Per window: arrival_shot (bool, null without an arrival) and arrival_delay_ms."""
    ticks = window_ticks.sort("window_uid", "tick_offset").with_columns(
        on_head=pl.col("target_angle")
        < pl.lit(math.degrees(1.0))
        * HEAD_RADIUS
        / pl.col("target_distance").clip(lower_bound=MIN_DISTANCE)
    )
    ticks = ticks.with_columns(
        arrives=pl.col("on_head").fill_null(False)
        & ~pl.col("on_head").shift(1).over("window_uid").fill_null(True)
    )
    shots = (
        ticks.filter(pl.col("shot") & (pl.col("tick_offset") <= 0))
        .select("window_uid", "tick_offset")
        .with_columns(gap=pl.col("tick_offset").diff().over("window_uid"))
        .with_columns(
            burst=(pl.col("gap").is_null() | (pl.col("gap") > BURST_GAP))
            .cum_sum()
            .over("window_uid")
        )
    )
    opening = (
        shots.filter(pl.col("burst") == pl.col("burst").max().over("window_uid"))
        .group_by("window_uid")
        .agg(opening=pl.col("tick_offset").min())
    )
    arrival = (
        ticks.filter(pl.col("arrives"))
        .select("window_uid", arrived=pl.col("tick_offset"))
        .join(opening, on="window_uid")
        .filter(pl.col("arrived") <= pl.col("opening"))
        .group_by("window_uid")
        .agg(arrived=pl.col("arrived").max(), opening=pl.col("opening").first())
    )
    return arrival.select(
        "window_uid",
        arrival_shot=pl.col("opening") == pl.col("arrived"),
        arrival_delay_ms=(pl.col("opening") - pl.col("arrived")) / TICK_RATE * 1000,
    )


__all__ = [
    "BURST_GAP",
    "HEAD_RADIUS",
    "SHOT_COLUMNS",
    "arrival_features",
    "mark_shots",
]
