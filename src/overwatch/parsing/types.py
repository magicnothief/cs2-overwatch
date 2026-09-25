"""Shared types and the canonical table contract for parsed matches.

Every source (.dem files, CS2CD matches) is converted into these same tables, so
everything downstream is written once. See docs/decisions/0004-canonical-schema.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

#: Columns every normalized tick table has, in this order.
TICK_COLUMNS: tuple[str, ...] = (
    "player_id",  # str: SteamID64 as text, or "Player_3" for CS2CD
    "player_name",  # str: may be null for anonymized sources
    "tick",  # i32
    "round",  # i32: rounds played before this tick
    "team",  # i32: 2 = T, 3 = CT
    "x",
    "y",
    "z",  # f32: world position
    "pitch",  # f32: view angle, negative is up
    "yaw",  # f32: view angle, -180..180
    "health",  # i32
    "is_alive",  # bool
    "is_freeze",  # bool: freeze period (buy time) is running
    "spotted",  # bool: visible to at least one enemy, by the game's own check
    "spotted_by",  # list[str]: which enemies can see this player right now
    "flash_duration",  # f32: seconds of blindness left from a flashbang
)

#: Events every source provides, normalized to at least (tick, player columns).
EVENTS_NEEDED: tuple[str, ...] = (
    "player_death",
    "player_hurt",
    "weapon_fire",
    "round_freeze_end",
    "round_start",
)


@dataclass(frozen=True)
class ParsedMatch:
    """One parsed match: per-tick player state plus the events we care about.

    Attributes:
        ticks: one row per (player, tick), with the columns in TICK_COLUMNS.
        events: event name -> DataFrame. Missing events map to empty frames.
        meta: source, map, tick_rate, and whatever the source knows
            (file hash for demos, cheater labels and rank for CS2CD).
    """

    ticks: pl.DataFrame
    events: dict[str, pl.DataFrame]
    meta: dict = field(default_factory=dict)

    @property
    def deaths(self) -> pl.DataFrame:
        """Kills, with columns: tick, attacker_id, victim_id, weapon, headshot."""
        return self.events["player_death"]

    def player_ids(self) -> list[str]:
        return self.ticks["player_id"].unique().sort().to_list()
