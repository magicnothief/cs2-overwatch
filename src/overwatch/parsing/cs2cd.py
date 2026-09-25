"""Load a CS2CD match (N.parquet + N.json) into the canonical tables.

CS2CD was parsed with demoparser2 and then anonymized, so the tick columns match
a .dem almost exactly. The differences: players are "Player_1".."Player_10"
instead of SteamIDs, and the events live in the JSON rather than in the parser.
See docs/DATASETS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from overwatch.parsing.types import EVENTS_NEEDED, TICK_COLUMNS, ParsedMatch

#: CS2CD name -> canonical name.
RENAME: dict[str, str] = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "steamid": "player_id",
    "name": "player_name",
    "team_num": "team",
    "is_freeze_period": "is_freeze",
    "total_rounds_played": "round",
    "approximate_spotted_by": "spotted_by",
}

#: CS2CD parquets carry ~200 props; reading only these keeps a match in memory.
SOURCE_COLUMNS: tuple[str, ...] = (
    "steamid",
    "tick",
    "total_rounds_played",
    "team_num",
    "X",
    "Y",
    "Z",
    "pitch",
    "yaw",
    "health",
    "is_alive",
    "is_freeze_period",
    "spotted",
    "approximate_spotted_by",
    "flash_duration",
)


def _normalize_ticks(raw: pl.DataFrame) -> pl.DataFrame:
    out = raw.rename(
        {k: v for k, v in RENAME.items() if k in raw.columns and v not in raw.columns}
    )
    for col, dtype in (
        ("player_name", pl.String),
        ("team", pl.Int32),
        ("health", pl.Int32),
        ("round", pl.Int32),
        ("is_freeze", pl.Boolean),
        ("is_alive", pl.Boolean),
        ("spotted", pl.Boolean),
        ("spotted_by", pl.List(pl.String)),
        ("flash_duration", pl.Float32),
    ):
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype).alias(col))
    return (
        out.with_columns(
            player_id=pl.col("player_id").cast(pl.String),
            tick=pl.col("tick").cast(pl.Int32),
            round=pl.col("round").cast(pl.Int32),
            team=pl.col("team").cast(pl.Int32),
            health=pl.col("health").cast(pl.Int32),
        )
        .select(TICK_COLUMNS)
        .sort(["player_id", "tick"])
    )


def _event_frame(payload: list[dict]) -> pl.DataFrame:
    if not payload:
        return pl.DataFrame({"tick": []}, schema={"tick": pl.Int32})
    return pl.DataFrame(payload, infer_schema_length=None).with_columns(
        tick=pl.col("tick").cast(pl.Int32)
    )


def _normalize_deaths(raw: pl.DataFrame) -> pl.DataFrame:
    victim = "user_steamid" if "user_steamid" in raw.columns else "victim_steamid"
    keep = [
        c
        for c in (
            "distance",
            "penetrated",
            "num_penetrations",
            "noscope",
            "no_scope",
            "thrusmoke",
        )
        if c in raw.columns
    ]
    out = raw.with_columns(
        attacker_id=pl.col("attacker_steamid").cast(pl.String),
        victim_id=pl.col(victim).cast(pl.String),
    )
    for col, dtype in (("weapon", pl.String), ("headshot", pl.Boolean)):
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype).alias(col))
    return out.select(
        ["tick", "attacker_id", "victim_id", "weapon", "headshot", *keep]
    ).sort("tick")


def load_cs2cd(
    parquet_path: str | Path,
    json_path: str | Path | None = None,
    *,
    ticks: pl.DataFrame | None = None,
) -> ParsedMatch:
    """Load one CS2CD match into a ParsedMatch.

    Args:
        parquet_path: path to N.parquet (per-tick data).
        json_path: path to N.json (events + labels). Defaults to the sibling
            file with the same stem.
        ticks: already-normalized ticks, as handed back by parsing.cache. When
            given, the source parquet is not read at all.

    Returns:
        A ParsedMatch. meta["cheaters"] holds the labeled cheaters' player ids.
    """
    parquet_path = Path(parquet_path)
    json_path = Path(json_path) if json_path else parquet_path.with_suffix(".json")

    if ticks is None:
        lazy = pl.scan_parquet(parquet_path)
        present = [c for c in SOURCE_COLUMNS if c in lazy.collect_schema().names()]
        ticks = _normalize_ticks(lazy.select(present).collect())

    with json_path.open() as fh:
        payload = json.load(fh)

    events: dict[str, pl.DataFrame] = {}
    for name in EVENTS_NEEDED:
        frame = _event_frame(payload.get(name, []))
        if name == "player_death" and not frame.is_empty():
            frame = _normalize_deaths(frame)
        events[name] = frame

    info = (payload.get("CSstats_info") or [{}])[0]
    meta = {
        "source": "cs2cd",
        "path": str(parquet_path),
        "map": info.get("map"),
        "avg_rank": info.get("avg_rank"),
        "match_making_type": info.get("match_making_type"),
        "tick_rate": 64,
        "cheaters": [c["steamid"] for c in payload.get("cheaters", [])],
    }
    return ParsedMatch(ticks=ticks, events=events, meta=meta)
