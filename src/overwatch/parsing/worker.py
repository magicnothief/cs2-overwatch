"""Parse one .dem in a process of its own, then exit.

    python -m overwatch.parsing.worker <demo> <out_dir>

A .dem is a file a stranger wrote, and demoparser2 is a Rust extension. A panic
in it is containable — `except BaseException` catches that. A heap overflow or a
use-after-free is not: control has already left Python. So layer 0 runs here, in
a short-lived child that holds the demo path and the output directory and
nothing else: no HTTP socket, no judge model, no reports directory, no settings.
`parse_demo()` in demo.py spawns it and reads the tables back.

The child writes `ticks.parquet`, `events/<name>.parquet` and `meta.json` into
`out_dir`, then exits 0. Anything else — a non-zero exit, a signal, a timeout —
is a failed job in the parent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import polars as pl

from overwatch.parsing.types import (
    EVENTS_NEEDED,
    TICK_COLUMNS,
    DemoParseError,
    ParsedMatch,
)

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

#: The header is written by whoever made the demo. A map name is a few letters;
#: anything longer is not a map name, and it ends up on the report page.
MAX_HEADER_CHARS = 128

#: Default address-space cap for the child, in MB, overridable with
#: OVERWATCH_PARSE_MEMORY_MB (0 disables it). Deliberately far above what a
#: parse needs: RLIMIT_AS counts reserved address space, not resident memory,
#: and polars' allocator reserves arenas on import. Measured on a 26 MB demo
#: with 136924 tick rows: 2048 MB aborts, 4096 MB parses. So the floor is a
#: fixed ~2 GB that has nothing to do with demo size, and this cap exists to
#: stop a parser that allocates without bound — not to be tight.
DEFAULT_MEMORY_MB = 8192

#: What the tables are called inside the output directory.
TICKS_FILE = "ticks.parquet"
EVENTS_DIR = "events"
META_FILE = "meta.json"


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


def _header_text(header: dict, key: str) -> str | None:
    value = header.get(key)
    return None if value is None else str(value)[:MAX_HEADER_CHARS]


def parse_in_process(path: str | Path) -> ParsedMatch:
    """Parse a .dem into the canonical tables, here, in this process.

    This is the call that reads the hostile file. Only the worker child should
    make it; everything else goes through `parse_demo()`, which spawns the
    child. `meta["sha256"]` is left unset — the parent hashes the file, because
    hashing is pure Python and does not need a process to be thrown away.
    """
    # Imported here, not at the top: the API server imports this module for its
    # read_match(), and there is no reason for the Rust extension to be mapped
    # into the process that holds the listening socket.
    from demoparser2 import DemoParser

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
        "map": _header_text(header, "map_name"),
        "server_name": _header_text(header, "server_name"),
        "patch_version": _header_text(header, "patch_version"),
        "tick_rate": 64,
        "sha256": None,
    }
    return ParsedMatch(ticks=ticks, events=events, meta=meta)


def write_match(match: ParsedMatch, out_dir: Path) -> None:
    """Write the tables where the parent will look for them."""
    events = out_dir / EVENTS_DIR
    events.mkdir(parents=True, exist_ok=True)
    match.ticks.write_parquet(out_dir / TICKS_FILE)
    for name, frame in match.events.items():
        frame.write_parquet(events / f"{name}.parquet")
    (out_dir / META_FILE).write_text(json.dumps(match.meta))


def read_match(out_dir: Path) -> ParsedMatch:
    """Read back what write_match wrote. Runs in the parent.

    Events come back in EVENTS_NEEDED order, not the order the directory lists
    them in, so a ParsedMatch from a worker is the one a direct parse gives.
    """
    files = {file.stem: file for file in (out_dir / EVENTS_DIR).glob("*.parquet")}
    missing = [name for name in EVENTS_NEEDED if name not in files]
    if missing:
        raise DemoParseError(f"the parser left no tables for {', '.join(missing)}")
    try:
        ticks = pl.read_parquet(out_dir / TICKS_FILE)
        events = {name: pl.read_parquet(files[name]) for name in EVENTS_NEEDED}
        meta = json.loads((out_dir / META_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise DemoParseError(f"the parser left no readable tables: {exc}") from exc
    return ParsedMatch(ticks=ticks, events=events, meta=meta)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _cap(which: int, soft: int, hard: int) -> None:
    """Lower one rlimit, as far as this process is allowed to lower it."""
    import resource

    ceiling = resource.getrlimit(which)[1]
    if ceiling != resource.RLIM_INFINITY:
        soft, hard = min(soft, ceiling), min(hard, ceiling)
    try:
        resource.setrlimit(which, (soft, hard))
    except (OSError, ValueError):
        pass  # a limit we cannot set is a bonus we do without


def _limit_this_process() -> None:
    """Cap memory and CPU before the parser sees the file. POSIX only.

    The child limits itself instead of the parent limiting it through
    `preexec_fn`: the API server is threaded, and running code between fork and
    exec in a threaded process is how deadlocks are made. Nothing untrusted has
    been read at this point — demoparser2 is not even imported yet.

    A parser that allocates without bound then dies by the kernel's hand rather
    than by the OOM killer picking some other process on the machine.
    """
    try:
        import resource
    except ImportError:  # Windows has no rlimits, and faking them is not worth it
        return

    memory_mb = _env_int("OVERWATCH_PARSE_MEMORY_MB", DEFAULT_MEMORY_MB)
    if memory_mb > 0:
        _cap(resource.RLIMIT_AS, memory_mb << 20, memory_mb << 20)
    cpu_seconds = _env_int("OVERWATCH_PARSE_CPU_SECONDS", 0)
    if cpu_seconds > 0:
        # Soft limit raises SIGXCPU, hard limit is SIGKILL five seconds later.
        _cap(resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 5)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(
            "usage: python -m overwatch.parsing.worker <demo> <out_dir>",
            file=sys.stderr,
        )
        return 2
    _limit_this_process()
    write_match(parse_in_process(args[0]), Path(args[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_in_process", "read_match", "write_match"]
