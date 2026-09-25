"""Parse a CS2 .dem file into the canonical tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
from demoparser2 import DemoParser

from overwatch.parsing.types import EVENTS_NEEDED, TICK_COLUMNS, ParsedMatch

#: Props requested from demoparser2. Names on the left of RENAME below.
PROPS: list[str] = [
    "X",
    "Y",
    "Z",
    "pitch",
    "yaw",
    "health",
    "team_num",
    "is_alive",
    "is_freeze_period",
    "total_rounds_played",
    "spotted",
    "approximate_spotted_by",
    "flash_duration",
]

#: demoparser2 name -> canonical name.
RENAME: dict[str, str] = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "name": "player_name",
    "team_num": "team",
    "is_freeze_period": "is_freeze",
    "total_rounds_played": "round",
    "approximate_spotted_by": "spotted_by",
}


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file so the same demo is never parsed (or labeled) twice."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _normalize_ticks(raw: pl.DataFrame) -> pl.DataFrame:
    return (
        raw.rename({k: v for k, v in RENAME.items() if k in raw.columns})
        .with_columns(
            player_id=pl.col("steamid").cast(pl.String),
            tick=pl.col("tick").cast(pl.Int32),
            round=pl.col("round").cast(pl.Int32),
            team=pl.col("team").cast(pl.Int32),
            health=pl.col("health").cast(pl.Int32),
            spotted_by=pl.col("spotted_by").cast(pl.List(pl.String)),
        )
        .select(TICK_COLUMNS)
        .sort(["player_id", "tick"])
    )


def _normalize_deaths(raw: pl.DataFrame) -> pl.DataFrame:
    """player_death -> tick, attacker_id, victim_id, weapon, headshot, plus extras."""
    keep = [
        c
        for c in (
            "distance",
            "penetrated",
            "noscope",
            "thrusmoke",
            "attackerblind",
            "hitgroup",
        )
        if c in raw.columns
    ]
    return (
        raw.with_columns(
            attacker_id=pl.col("attacker_steamid").cast(pl.String),
            victim_id=pl.col("user_steamid").cast(pl.String),
        )
        .select(["tick", "attacker_id", "victim_id", "weapon", "headshot", *keep])
        .with_columns(tick=pl.col("tick").cast(pl.Int32))
        .sort("tick")
    )


def _normalize_generic_event(raw: pl.DataFrame) -> pl.DataFrame:
    """Events keyed on one player: rename *_steamid columns, keep the rest as-is."""
    out = raw
    if "user_steamid" in out.columns:
        out = out.with_columns(player_id=pl.col("user_steamid").cast(pl.String))
    if "attacker_steamid" in out.columns:
        out = out.with_columns(attacker_id=pl.col("attacker_steamid").cast(pl.String))
    return out.with_columns(tick=pl.col("tick").cast(pl.Int32)).sort("tick")


def parse_demo(path: str | Path, *, hash_file: bool = True) -> ParsedMatch:
    """Parse a .dem file into canonical tick and event tables.

    Args:
        path: path to the .dem file.
        hash_file: compute a sha256 of the demo (set False for speed in tests).

    Returns:
        A ParsedMatch whose ticks hold TICK_COLUMNS, sorted by (player_id, tick).
    """
    path = Path(path)
    parser = DemoParser(str(path))

    ticks = _normalize_ticks(pl.from_pandas(parser.parse_ticks(PROPS)))

    available = set(parser.list_game_events())
    events: dict[str, pl.DataFrame] = {}
    for name in EVENTS_NEEDED:
        if name not in available:
            events[name] = pl.DataFrame({"tick": []}, schema={"tick": pl.Int32})
            continue
        raw = pl.from_pandas(parser.parse_event(name))
        if raw.is_empty():
            events[name] = pl.DataFrame({"tick": []}, schema={"tick": pl.Int32})
        elif name == "player_death":
            events[name] = _normalize_deaths(raw)
        else:
            events[name] = _normalize_generic_event(raw)

    header = parser.parse_header()
    meta = {
        "source": "dem",
        "path": str(path),
        "map": header.get("map_name"),
        "server_name": header.get("server_name"),
        "patch_version": header.get("patch_version"),
        "tick_rate": 64,
        "sha256": file_sha256(path) if hash_file else None,
    }
    return ParsedMatch(ticks=ticks, events=events, meta=meta)
