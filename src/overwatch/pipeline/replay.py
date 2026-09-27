"""A round, replayable on the page: everyone's movement and view, and what each knew.

Spec: docs/specs/2026-09-27-round-replay.md.

Who sees whom comes from the map mesh, as the evidence's line of sight does: a
player sees an enemy when nothing on the mesh blocks the line from eye to head
and the enemy is on their screen. Not the game's own `spotted` flag: a tick
before a gun kill it marks the victim as seen by the attacker only 64% of the
time on CS2CD (docs/evals/2026-09-27-replay-marks-gate.md).

A round here is play time: from the end of its freeze to the start of the next
round's freeze, so what happens after the round is decided (exit kills, saves)
is in it. The report's RoundSpan is wider (it starts at the round counter's
change); the replay does not need the buy time.

The file is one JSON object, gzipped, laid out for the page rather than for
Python: per round, one array per player per quantity, aligned to that round's
sample ticks, with null where the player is dead or gone.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from overwatch.layers.l2_perception.geometry.angles import (
    angles_to_target,
    in_field_of_view,
)
from overwatch.layers.l2_perception.geometry.occlusion import (
    DEFAULT_MAP_DIR,
    has_map,
    line_of_sight,
)
from overwatch.layers.l3_behavior.shots import NOT_AIMED
from overwatch.parsing.types import ParsedMatch

#: The file's layout; the page refuses a version it does not know.
VERSION = 1
#: One replay sample every this many ticks: 16 per second at 64 tick.
SAMPLE_TICKS = 4
#: A smoke with no expiry event in the demo lasts this long.
SMOKE_SECONDS = 18
#: How big the page draws a smoke and a fire, in world units.
SMOKE_RADIUS = 144
FIRE_RADIUS = 120

_TEAMS = {2: "T", 3: "CT"}
_BOMB = {
    "bomb_pickup": "pickup",
    "bomb_dropped": "drop",
    "bomb_planted": "plant",
    "bomb_defused": "defuse",
    "bomb_exploded": "explode",
}


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


def sightings(
    pairs: pl.DataFrame, map_name: str, *, map_dir: str | Path = DEFAULT_MAP_DIR
) -> pl.DataFrame:
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
        map_dir=map_dir,
    )
    los = once.select("tick", "a", "b").with_columns(
        los=pl.Series(clear, dtype=pl.Boolean)
    )
    return (
        keyed.join(los, on=["tick", "a", "b"], how="left")
        .with_columns(sees=pl.col("los").fill_null(False) & in_field_of_view())
        .drop("a", "b", "los")
    )


def round_bounds(ticks: pl.DataFrame) -> pl.DataFrame:
    """Each round's play time: number (1-based), start tick, end tick (exclusive).

    Starts after the round's last freeze tick; ends where the next round's freeze
    starts, or with the demo.
    """
    per = (
        ticks.drop_nulls("round")
        .group_by("round")
        .agg(
            first=pl.col("tick").min(),
            last=pl.col("tick").max(),
            freeze_first=pl.col("tick").filter(pl.col("is_freeze")).min(),
            freeze_last=pl.col("tick").filter(pl.col("is_freeze")).max(),
        )
        .sort("round")
    )
    return per.select(
        number=(pl.col("round") + 1).cast(pl.Int32),
        start=pl.coalesce(pl.col("freeze_last") + 1, pl.col("first")).cast(pl.Int32),
        end=pl.coalesce(
            pl.col("freeze_first").shift(-1),
            pl.col("first").shift(-1),
            pl.col("last") + 1,
        ).cast(pl.Int32),
    ).filter(pl.col("end") > pl.col("start"))


def _in_round(frame: pl.DataFrame, bounds: pl.DataFrame) -> pl.DataFrame:
    """Tag each row of an event frame with the round whose play time holds its tick."""
    if frame.is_empty() or "tick" not in frame.columns:
        return frame.with_columns(number=pl.lit(None, dtype=pl.Int32)).clear()
    return (
        frame.with_columns(pl.col("tick").cast(pl.Int32))
        .sort("tick")
        .join_asof(bounds.sort("start"), left_on="tick", right_on="start")
        .filter(pl.col("tick") < pl.col("end"))
    )


def _where(ticks: pl.DataFrame, rows: pl.DataFrame, who: str) -> pl.DataFrame:
    """Add x, y: where player `who` stood at each row's tick (their last known spot)."""
    places = (
        ticks.select("tick", pl.col("player_id").alias(who), "x", "y")
        .drop_nulls(["x", "y"])
        .sort("tick")
    )
    # both sorted by tick above; polars cannot verify that across `by` groups
    return rows.sort("tick").join_asof(
        places, on="tick", by=who, check_sortedness=False
    )


def _before_round(frame: pl.DataFrame, bounds: pl.DataFrame) -> pl.DataFrame:
    """Like _in_round, but an event in the freeze belongs to the round after it.

    The bomb is picked up in the buy time; without this, every round would open
    with its carrier unknown.
    """
    if frame.is_empty() or "tick" not in frame.columns:
        return frame.with_columns(number=pl.lit(None, dtype=pl.Int32)).clear()
    return (
        frame.with_columns(pl.col("tick").cast(pl.Int32))
        .sort("tick")
        .join_asof(
            bounds.sort("end"),
            left_on="tick",
            right_on="end",
            strategy="forward",
            allow_exact_matches=False,
        )
        .drop_nulls("number")
    )


def _pairs(
    start: pl.DataFrame, end: pl.DataFrame, bounds: pl.DataFrame, cap: int | None
):
    """Smokes or fires: [start, end, x, y], paired by entity id, ending by the round."""
    if start.is_empty() or "entityid" not in start.columns:
        return {}
    begun = _in_round(start, bounds).select("number", "entityid", "tick", "x", "y")
    ends = (
        end.select("entityid", pl.col("tick").cast(pl.Int32).alias("until"))
        if not end.is_empty() and "entityid" in end.columns
        else pl.DataFrame(
            schema={"entityid": begun["entityid"].dtype, "until": pl.Int32}
        )
    )
    paired = (
        begun.join(ends, on="entityid", how="left")
        # entity ids are reused: an expiry before this start belongs to another one
        .with_columns(until=pl.when(pl.col("until") >= pl.col("tick")).then("until"))
        .group_by("number", "entityid", "tick", "x", "y")
        .agg(pl.col("until").min())
        .join(bounds.select("number", "end"), on="number")
    )
    limit = (
        pl.col("end") if cap is None else pl.min_horizontal("end", pl.col("tick") + cap)
    )
    paired = paired.with_columns(
        until=pl.min_horizontal(pl.coalesce("until", limit), "end")
    )
    out: dict[int, list] = {}
    for row in paired.sort("tick").iter_rows(named=True):
        out.setdefault(row["number"], []).append(
            [row["tick"], row["until"], round(row["x"]), round(row["y"])]
        )
    return out


def build_replay(
    match: ParsedMatch,
    *,
    map_name: str | None = None,
    map_dir: str | Path = DEFAULT_MAP_DIR,
) -> dict:
    """Everything the page needs to replay every round of `match`."""
    ticks = match.ticks
    map_name = map_name or match.meta.get("map")
    bounds = round_bounds(ticks)
    # players: whoever was ever on a side; spectators and casters are not
    names = (
        ticks.filter(pl.col("team").is_in(list(_TEAMS)))
        .group_by("player_id")
        .agg(pl.col("player_name").drop_nulls().mode().first().alias("name"))
        .sort("player_id")
    )
    ids = names["player_id"].to_list()
    index = {pid: i for i, pid in enumerate(ids)}
    order = pl.DataFrame({"player_id": ids, "i": list(range(len(ids)))})

    # one row per (round, sample tick, player present that round)
    sampled = (
        ticks.filter(pl.col("tick") % SAMPLE_TICKS == 0)
        .with_columns(pl.col("tick").cast(pl.Int32))
        .sort("tick")
        .join_asof(bounds.sort("start"), left_on="tick", right_on="start")
        .filter(pl.col("tick") < pl.col("end"))
        .join(order, on="player_id")
    )
    visibility = bool(map_name) and has_map(map_name, map_dir)
    if visibility:
        seen = sightings(enemy_pairs(samples(ticks)), map_name, map_dir=map_dir)
        bits = (
            seen.filter(pl.col("sees"))
            .join(order.rename({"player_id": "enemy_id", "i": "e"}), on="enemy_id")
            .with_columns(
                bit=pl.col("e").replace_strict(
                    {i: 1 << i for i in range(len(ids))}, return_dtype=pl.Int64
                )
            )
            .group_by("tick", "player_id")
            .agg(sees=pl.col("bit").sum())
        )
        sampled = sampled.join(bits, on=["tick", "player_id"], how="left")
    else:
        sampled = sampled.with_columns(sees=pl.lit(None, dtype=pl.Int64))

    alive = pl.col("is_alive").fill_null(False)
    values = sampled.with_columns(
        x=pl.when(alive).then(pl.col("x").round().cast(pl.Int32)),
        y=pl.when(alive).then(pl.col("y").round().cast(pl.Int32)),
        z=pl.when(alive).then(pl.col("z").round().cast(pl.Int32)),
        yaw=pl.when(alive).then(pl.col("yaw").round().cast(pl.Int32)),
        hp=pl.when(alive).then(pl.col("health")).otherwise(0),
        blind=pl.when(alive).then(
            (pl.col("flash_duration").fill_null(0) > 0).cast(pl.Int8)
        ),
        sees=pl.when(alive).then(pl.col("sees").fill_null(0)),
    )

    rounds = []
    events = match.events
    fire = events.get("weapon_fire", pl.DataFrame())
    shooter = "player_id" if "player_id" in fire.columns else None
    shots = (
        _in_round(
            fire.filter(
                ~pl.col("weapon")
                .cast(pl.String)
                .str.contains(NOT_AIMED)
                .fill_null(False)
            )
            if "weapon" in fire.columns
            else fire,
            bounds,
        )
        if shooter
        else pl.DataFrame()
    )
    deaths = _in_round(events.get("player_death", pl.DataFrame()), bounds)
    if not deaths.is_empty():
        deaths = _where(
            ticks, deaths.with_columns(pl.col("victim_id").cast(pl.String)), "victim_id"
        )
    smokes = _pairs(
        events.get("smokegrenade_detonate", pl.DataFrame()),
        events.get("smokegrenade_expired", pl.DataFrame()),
        bounds,
        SMOKE_SECONDS * 64,
    )
    fires = _pairs(
        events.get("inferno_startburn", pl.DataFrame()),
        events.get("inferno_expire", pl.DataFrame()),
        bounds,
        None,
    )
    bomb_rows = [
        _before_round(frame, bounds).with_columns(what=pl.lit(_BOMB[name]))
        for name, frame in events.items()
        if name in _BOMB and not frame.is_empty() and "player_id" in frame.columns
    ]
    bomb = (
        _where(
            ticks,
            pl.concat(
                [b.select("number", "tick", "what", "player_id") for b in bomb_rows]
            ),
            "player_id",
        )
        if bomb_rows
        else pl.DataFrame()
    )

    # a regular grid per round, every player present in it: arrays line up by
    # index, and a missing tick is a null, not a shift
    present = (
        values.filter(pl.col("team").is_in(list(_TEAMS)))
        .group_by("number", "i")
        .agg(pl.col("team").mode().first())
    )
    grid = (
        bounds.with_columns(
            t0=((pl.col("start") + SAMPLE_TICKS - 1) // SAMPLE_TICKS * SAMPLE_TICKS)
        )
        .with_columns(tick=pl.int_ranges("t0", "end", SAMPLE_TICKS, dtype=pl.Int32))
        .explode("tick", empty_as_null=False)
        .drop_nulls("tick")
        .select("number", "t0", "tick")
        .join(present, on="number")
    )
    fields = ("x", "y", "z", "yaw", "hp", "blind", "sees")
    per_player = (
        grid.join(
            values.select("number", "tick", "i", *fields),
            on=["number", "tick", "i"],
            how="left",
        )
        .group_by("number", "i")
        .agg(
            pl.col("team").first(),
            pl.col("t0").first(),
            *[pl.col(c).sort_by("tick") for c in fields],
        )
    )
    first_tick = per_player.group_by("number").agg(pl.col("t0").first())
    for span in (
        bounds.join(first_tick, on="number").sort("number").iter_rows(named=True)
    ):
        n = span["number"]
        players = []
        for row in (
            per_player.filter(pl.col("number") == n).sort("i").iter_rows(named=True)
        ):
            entry = {
                "i": row["i"],
                "side": _TEAMS.get(row["team"]),
                **{c: row[c] for c in ("x", "y", "z", "yaw", "hp", "blind")},
            }
            if visibility:
                entry["sees"] = row["sees"]
            players.append(entry)
        rounds.append(
            {
                "number": n,
                "start": span["start"],
                "end": span["end"],
                "t0": span["t0"],
                "players": players,
                "shots": [
                    [t, index[p]]
                    for t, p in shots.filter(pl.col("number") == n)
                    .select("tick", pl.col(shooter).cast(pl.String))
                    .iter_rows()
                    if p in index
                ]
                if shooter and not shots.is_empty()
                else [],
                "deaths": [
                    [
                        d["tick"],
                        index.get(d["victim_id"], -1),
                        index.get(str(d["attacker_id"]), -1),
                        d.get("weapon"),
                        bool(d.get("headshot")),
                        None if d["x"] is None else round(d["x"]),
                        None if d["y"] is None else round(d["y"]),
                    ]
                    for d in deaths.filter(pl.col("number") == n).iter_rows(named=True)
                ]
                if not deaths.is_empty()
                else [],
                "smokes": smokes.get(n, []),
                "fires": fires.get(n, []),
                "bomb": [
                    [
                        b["tick"],
                        b["what"],
                        index.get(b["player_id"], -1),
                        None if b["x"] is None else round(b["x"]),
                        None if b["y"] is None else round(b["y"]),
                    ]
                    for b in bomb.filter(pl.col("number") == n)
                    .sort("tick")
                    .iter_rows(named=True)
                ]
                if not bomb.is_empty()
                else [],
            }
        )
    return {
        "version": VERSION,
        "map": map_name,
        "tick_rate": int(match.meta.get("tick_rate") or 64),
        "sample_ticks": SAMPLE_TICKS,
        "smoke_radius": SMOKE_RADIUS,
        "fire_radius": FIRE_RADIUS,
        "visibility": visibility,
        "players": [{"id": pid, "name": name} for pid, name in names.iter_rows()],
        "rounds": rounds,
    }


def write_replay(replay: dict, path: Path) -> Path:
    """Write the replay gzipped, atomically: the page never reads half a file."""
    part = path.with_name(path.name + ".part")
    part.write_bytes(gzip.compress(json.dumps(replay, separators=(",", ":")).encode()))
    part.replace(path)
    return path


__all__ = [
    "FIRE_RADIUS",
    "SAMPLE_TICKS",
    "SMOKE_RADIUS",
    "SMOKE_SECONDS",
    "VERSION",
    "build_replay",
    "enemy_pairs",
    "round_bounds",
    "samples",
    "sightings",
    "write_replay",
]
