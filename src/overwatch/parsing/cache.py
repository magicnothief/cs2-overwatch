"""Cache normalized ticks so a schema change costs minutes, not a quarter hour.

A CS2CD match is a 40-60 MB parquet with ~200 columns; the canonical table keeps
16 of them. Reading and narrowing all 794 matches is most of a rebuild, and it
produces the same bytes every time — until the schema itself changes.

So the cache key is the schema. `TICK_COLUMNS` is hashed into the cache path,
which means adding a column (as `spotted` did) invalidates every entry
automatically: no stale-cache class of bug, and no flag to remember. The source
file's modification time is in the name too, so replacing a match invalidates
just that entry.

Only ticks are cached. Events come from a small JSON, and meta carries the
cheater labels, both of which we want re-read.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import polars as pl

from overwatch.parsing.cs2cd import load_cs2cd
from overwatch.parsing.types import TICK_COLUMNS, ParsedMatch

DEFAULT_CACHE = Path("data/interim/ticks_cache")


def schema_key() -> str:
    """Short hash of the canonical columns: the cache's version number."""
    return hashlib.sha256("|".join(TICK_COLUMNS).encode()).hexdigest()[:8]


def cache_path(source: Path, cache_dir: Path | None = None) -> Path:
    """Where this source file's normalized ticks live."""
    root = Path(cache_dir or DEFAULT_CACHE) / schema_key()
    return (
        root
        / f"{source.parent.name}__{source.stem}__{int(source.stat().st_mtime)}.parquet"
    )


def load_cs2cd_cached(
    parquet_path: str | Path,
    json_path: str | Path | None = None,
    *,
    cache_dir: Path | None = None,
) -> ParsedMatch:
    """load_cs2cd, but reusing normalized ticks from disk when they are current."""
    parquet_path = Path(parquet_path)
    target = cache_path(parquet_path, cache_dir)

    if target.exists():
        return load_cs2cd(parquet_path, json_path, ticks=pl.read_parquet(target))

    match = load_cs2cd(parquet_path, json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write under a temporary name first: a half-written file must never look
    # complete to a parallel worker reading the same cache.
    scratch = target.with_suffix(f".{os.getpid()}.tmp")
    match.ticks.write_parquet(scratch)
    scratch.replace(target)
    return match


def clear(cache_dir: Path | None = None) -> int:
    """Delete every cached tick table. Returns how many files were removed."""
    files = list(Path(cache_dir or DEFAULT_CACHE).rglob("*.parquet"))
    for file in files:
        file.unlink()
    return len(files)
