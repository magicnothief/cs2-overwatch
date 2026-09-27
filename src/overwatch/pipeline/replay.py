"""A round, replayable on the page: everyone's movement and view, and what each knew.

Spec: docs/specs/2026-09-27-round-replay.md.

Who sees whom comes from the map mesh, as the evidence's line of sight does: a
player sees an enemy when nothing on the mesh blocks the line from eye to head
and the enemy is on their screen. Not the game's own `spotted` flag: a tick
before a gun kill it marks the victim as seen by the attacker only 64% of the
time on CS2CD (docs/evals/2026-09-27-replay-marks-gate.md).
"""

from __future__ import annotations

import polars as pl

from overwatch.layers.l2_perception.geometry.angles import (
    angles_to_target,
    in_field_of_view,
)
from overwatch.layers.l2_perception.geometry.occlusion import line_of_sight

#: One replay sample every this many ticks: 16 per second at 64 tick.
SAMPLE_TICKS = 4


def samples(ticks: pl.DataFrame) -> pl.DataFrame:
    """Every SAMPLE_TICKS-th tick of the live round, alive players only."""
    return ticks.filter(
        (pl.col("tick") % SAMPLE_TICKS == 0) & pl.col("is_alive") & ~pl.col("is_freeze")
    )


def enemy_pairs(live: pl.DataFrame) -> pl.DataFrame:
    """Every (viewer, enemy) pair per sample, with the angle from the viewer's view."""
    viewers = live.select(
        "player_id", "tick", "round", "team", "x", "y", "z", "yaw", "pitch"
    )
    enemies = live.select(
        "tick",
        enemy_id="player_id",
        enemy_team="team",
        target_x="x",
        target_y="y",
        target_z="z",
    )
    pairs = viewers.join(enemies, on="tick").filter(
        pl.col("team") != pl.col("enemy_team")
    )
    return angles_to_target(pairs)


def sightings(pairs: pl.DataFrame, map_name: str) -> pl.DataFrame:
    """Add `sees`: nothing on the map blocks the view, and the enemy is on screen.

    Line of sight is symmetric at equal eye and head heights, so each unordered
    pair is ray cast once.
    """
    first = pl.min_horizontal("player_id", "enemy_id")
    second = pl.max_horizontal("player_id", "enemy_id")
    keyed = pairs.with_columns(a=first, b=second)
    once = keyed.filter(pl.col("player_id") == pl.col("a")).select(
        "tick", "a", "b", "x", "y", "z", "target_x", "target_y", "target_z"
    )
    clear = line_of_sight(
        once.select("x", "y", "z").to_numpy(),
        once.select("target_x", "target_y", "target_z").to_numpy(),
        map_name,
    )
    los = once.select("tick", "a", "b").with_columns(los=pl.Series(clear))
    return (
        keyed.join(los, on=["tick", "a", "b"], how="left")
        .with_columns(sees=pl.col("los").fill_null(False) & in_field_of_view())
        .drop("a", "b", "los")
    )


__all__ = [
    "SAMPLE_TICKS",
    "enemy_pairs",
    "samples",
    "sightings",
]
