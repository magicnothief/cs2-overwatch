"""Angle maths shared by every layer.

Keeping this in one module means the wrap-around fix exists exactly once. Both
Layer 1 (blatant rules) and Layer 3 (behaviour features) import from here.
"""

from __future__ import annotations

import polars as pl

#: CS2 demos are recorded at 64 ticks per second.
TICK_RATE: int = 64

#: Largest turn that can be measured in one tick before it aliases (see README).
MAX_MEASURABLE_SPEED: float = 180.0 * TICK_RATE


def wrap_degrees(expr: pl.Expr) -> pl.Expr:
    """Map an angle difference onto [-180, 180): the shortest way round the circle.

    A yaw going from 179 to -179 is a +2 turn, not -358.
    """
    return ((expr + 180) % 360) - 180


def add_aim_features(
    ticks: pl.DataFrame, *, require_alive: bool = True
) -> pl.DataFrame:
    """Add per-tick aim derivatives to a canonical tick table.

    Adds:
        d_tick: ticks since this player's previous row
        d_yaw, d_pitch: signed angle change, wrap-corrected (degrees)
        yaw_speed, pitch_speed: absolute angular speed (degrees / second)

    Speeds are null where they would be meaningless: the player's first row, a
    gap in ticks, and (when require_alive) any pair of rows where the player was
    dead or in freeze time. Null means "unknown", so keep it rather than filling
    with 0, which would drag every average down.
    """
    out = ticks.sort(["player_id", "tick"]).with_columns(
        d_tick=pl.col("tick").diff().over("player_id"),
        d_yaw=wrap_degrees(pl.col("yaw").diff().over("player_id")),
        d_pitch=pl.col("pitch").diff().over("player_id"),
        _prev_alive=pl.col("is_alive").shift(1).over("player_id"),
        _prev_freeze=pl.col("is_freeze").shift(1).over("player_id"),
    )

    valid = pl.col("d_tick") > 0
    if require_alive:
        valid = (
            valid
            & pl.col("is_alive")
            & pl.col("_prev_alive")
            & ~pl.col("is_freeze")
            & ~pl.col("_prev_freeze")
        )

    return out.with_columns(
        yaw_speed=pl.when(valid)
        .then(pl.col("d_yaw").abs() / pl.col("d_tick") * TICK_RATE)
        .otherwise(None),
        pitch_speed=pl.when(valid)
        .then(pl.col("d_pitch").abs() / pl.col("d_tick") * TICK_RATE)
        .otherwise(None),
    ).drop("_prev_alive", "_prev_freeze")


def kill_windows(
    ticks: pl.DataFrame,
    deaths: pl.DataFrame,
    *,
    pre: int = 128,
    post: int = 32,
    drop_incomplete: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cut a window of ticks around every kill, from the attacker's point of view.

    Args:
        ticks: canonical ticks, already through add_aim_features().
        deaths: normalized player_death table (tick, attacker_id, victim_id, ...).
        pre: ticks before the kill (128 = 2 s).
        post: ticks after the kill (32 = 0.5 s).
        drop_incomplete: drop windows that run past either end of the demo, so
            every window has the same length.

    Returns:
        (windows, window_ticks)
        windows: one row per kill — window_id, attacker_id, victim_id, kill_tick,
            weapon, headshot, plus whatever extra columns deaths carried.
        window_ticks: one row per (window_id, tick) — every tick column plus
            tick_offset (negative before the kill) and t_ms.
    """
    kills = (
        deaths.filter(pl.col("attacker_id").is_not_null())
        .filter(pl.col("attacker_id") != pl.col("victim_id"))  # no suicides
        .sort("tick")
        .with_row_index("window_id")
    )
    if kills.is_empty():
        return kills, ticks.clear().with_columns(
            window_id=pl.lit(None, pl.UInt32),
            tick_offset=pl.lit(None, pl.Int32),
            t_ms=pl.lit(None, pl.Float64),
        )

    # Ask for exactly the ticks each window needs, then join on (player, tick).
    # Joining on player alone and filtering afterwards is the obvious way to write
    # this and roughly ten times slower: it builds every (tick, window) pair for a
    # player — millions of rows — to keep 161 of them.
    offsets = pl.DataFrame(
        {"tick_offset": pl.Series(range(-pre, post + 1), dtype=pl.Int32)}
    )
    wanted = (
        kills.select(
            "window_id",
            player_id=pl.col("attacker_id"),
            kill_tick=pl.col("tick"),
        )
        .join(offsets, how="cross")
        .with_columns(tick=(pl.col("kill_tick") + pl.col("tick_offset")).cast(pl.Int32))
    )

    window_ticks = (
        wanted.join(ticks, on=["player_id", "tick"], how="inner")
        .with_columns(t_ms=pl.col("tick_offset") / TICK_RATE * 1000)
        .sort(["window_id", "tick"])
    )

    if drop_incomplete:
        expected = pre + post + 1
        complete = (
            window_ticks.group_by("window_id")
            .len()
            .filter(pl.col("len") == expected)
            .select("window_id")
        )
        window_ticks = window_ticks.join(complete, on="window_id", how="inner")
        kills = kills.join(complete, on="window_id", how="inner")

    return kills, window_ticks
