"""Turn a folder of matches into one labeled window dataset.

This is the bridge between parsing (one match at a time) and Layer 3 (a model
trained over many matches). The output is two parquet files:

    windows.parquet       one row per kill  (the label lives here)
    window_ticks.parquet  one row per (window, tick)

Labels follow CS2CD's own warning (docs/DATASETS.md):

    cheater  the attacker is listed in the match's `cheaters`
    clean    the match comes from no_cheater_present
    unknown  another player in a match that contained a cheater —
             CS2CD measured this label as only ~55% reliable, so it is
             excluded from training rather than treated as a negative.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from overwatch.aim import add_aim_features, kill_windows
from overwatch.layers.l2_perception.awareness import add_context
from overwatch.layers.l2_perception.geometry import angles_to_target
from overwatch.layers.l2_perception.geometry.occlusion import has_map, line_of_sight
from overwatch.layers.l3_behavior.shots import mark_shots
from overwatch.parsing import ParsedMatch, load_cs2cd_cached

log = logging.getLogger(__name__)

CHEATER_DIR = "with_cheater_present"
CLEAN_DIR = "no_cheater_present"


def label_window(
    attacker_id: str, cheaters: list[str], *, match_had_cheater: bool
) -> str:
    """Label one window from the attacker and the match's cheater list."""
    if attacker_id in cheaters:
        return "cheater"
    return "unknown" if match_had_cheater else "clean"


def match_windows(
    match: ParsedMatch,
    match_id: str,
    *,
    pre: int = 128,
    post: int = 32,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every kill in one parsed match, windowed the way every model reads it.

    Shared by the training set (build_match_windows) and the live pipeline, so a
    demo is cut, measured and ray cast exactly like the data the models learned
    from. Any difference here would be a train/inference mismatch that no test on
    held-out CS2CD data could see.

    Returns:
        (windows, window_ticks): one row per kill with its context (last seen,
        audible cues, players alive), and one row per (kill, tick) with the victim
        geometry. `window_uid` is "<match_id>#<n>".
    """
    map_name = match.meta.get("map")
    windows, window_ticks = kill_windows(
        add_aim_features(match.ticks), match.deaths, pre=pre, post=post
    )
    if windows.is_empty():
        return windows, window_ticks

    windows = add_context(windows, match.ticks, match.events, map_name)
    windows = windows.with_columns(
        window_uid=pl.format("{}#{}", pl.lit(match_id), pl.col("window_id")),
        match_id=pl.lit(match_id),
        map=pl.lit(map_name),
        avg_rank=pl.lit(match.meta.get("avg_rank")),
    ).rename({"tick": "kill_tick"})

    window_ticks = window_ticks.join(
        windows.select("window_id", "window_uid", "match_id", "victim_id"),
        on="window_id",
        how="inner",
    )
    window_ticks = add_victim_geometry(window_ticks, match.ticks, map_name=map_name)
    return windows, mark_shots(window_ticks, match.events.get("weapon_fire"))


def build_match_windows(
    parquet_path: str | Path,
    *,
    pre: int = 128,
    post: int = 32,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse one CS2CD match into labeled windows.

    Returns:
        (windows, window_ticks) with a `window_uid` unique across matches
        (`"<folder>/<match_id>#<n>"`) and a `label` column on both.
    """
    parquet_path = Path(parquet_path)
    folder = parquet_path.parent.name
    match_had_cheater = folder == CHEATER_DIR

    match = load_cs2cd_cached(parquet_path)
    cheaters = match.meta.get("cheaters", [])

    windows, window_ticks = match_windows(
        match, f"{folder}/{parquet_path.stem}", pre=pre, post=post
    )
    if windows.is_empty():
        return windows, window_ticks

    windows = windows.with_columns(
        match_had_cheater=pl.lit(match_had_cheater),
        label=pl.col("attacker_id")
        .map_elements(
            lambda pid: label_window(
                pid, cheaters, match_had_cheater=match_had_cheater
            ),
            return_dtype=pl.String,
        )
        .alias("label"),
    )
    window_ticks = window_ticks.join(
        windows.select("window_id", "label"), on="window_id", how="inner"
    )
    return windows, window_ticks


def add_victim_geometry(
    window_ticks: pl.DataFrame,
    match_ticks: pl.DataFrame,
    *,
    map_name: str | None = None,
) -> pl.DataFrame:
    """Attach where the victim was, relative to the attacker's crosshair.

    This is what separates "the crosshair moved fast" from "the crosshair closed
    on the target": the angle to the victim, tick by tick, before and after the
    kill. The victim keeps a position after dying, so the post-kill ticks describe
    where the crosshair drifted relative to the body.

    Visibility is computed by ray casting against the map mesh when one is
    available, and falls back to the game's spotting flag otherwise. The two are
    not interchangeable — the flag is biased by weapon and distance, which leaks
    the label (see docs/decisions/0007) — so `visibility_source` records which one
    produced the answer.
    """
    victim_state = match_ticks.select(
        victim_id=pl.col("player_id"),
        tick=pl.col("tick"),
        target_x=pl.col("x"),
        target_y=pl.col("y"),
        target_z=pl.col("z"),
        target_spotted=pl.col("spotted"),
        target_spotted_by=pl.col("spotted_by"),
    )
    joined = angles_to_target(
        window_ticks.join(victim_state, on=["victim_id", "tick"], how="left")
    ).with_columns(
        # the game's own check, per observer: could *this* attacker see them?
        flag_visible=pl.col("target_spotted_by")
        .list.contains(pl.col("player_id"))
        .fill_null(value=False),
    )

    if map_name and has_map(map_name):
        positions = joined.select("x", "y", "z", "target_x", "target_y", "target_z")
        visible = line_of_sight(
            positions.select("x", "y", "z").to_numpy(),
            positions.select("target_x", "target_y", "target_z").to_numpy(),
            map_name,
        )
        joined = joined.with_columns(
            target_visible=pl.Series(visible),
            visibility_source=pl.lit("raycast"),
        )
    else:
        joined = joined.with_columns(
            target_visible=pl.col("flag_visible"),
            visibility_source=pl.lit("spotting_flag"),
        )

    return joined.drop("target_x", "target_y", "target_z", "target_spotted_by")


def _build_one(args: tuple[Path, int, int]) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """Worker entry point: one match, or None if it could not be built."""
    path, pre, post = args
    try:
        windows, window_ticks = build_match_windows(path, pre=pre, post=post)
    except Exception:
        log.exception("failed to build windows for %s", path)
        return None
    if windows.is_empty():
        log.warning("no complete windows in %s", path)
        return None
    return windows, window_ticks


def default_workers() -> int:
    """Leave room for polars' own threads inside each worker."""
    return max(1, min(6, (os.cpu_count() or 4) // 2))


def build_window_dataset(
    paths: Iterable[str | Path],
    *,
    pre: int = 128,
    post: int = 32,
    out_dir: str | Path | None = None,
    workers: int | None = None,
    progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build one labeled window dataset from many match parquets.

    Matches are independent, so they are built in parallel. Each worker runs
    polars with its own thread pool, which is why the default worker count is
    half the cores rather than all of them.

    Args:
        paths: CS2CD `N.parquet` files (each needs its `N.json` beside it).
        pre: ticks kept before each kill.
        post: ticks kept after each kill.
        out_dir: if given, write windows.parquet and window_ticks.parquet there.
        workers: parallel processes; 1 runs in this process, useful for debugging.
        progress: print a line every 50 matches (flushed, so it shows up in logs).

    Returns:
        (windows, window_ticks) concatenated across every match that parsed.
    """
    jobs = [(Path(p), pre, post) for p in paths]
    workers = workers or default_workers()
    all_windows: list[pl.DataFrame] = []
    all_ticks: list[pl.DataFrame] = []

    def collect(result: tuple[pl.DataFrame, pl.DataFrame] | None, done: int) -> None:
        if result is not None:
            all_windows.append(result[0])
            all_ticks.append(result[1])
        if progress and done % 50 == 0:
            print(f"  {done}/{len(jobs)} matches", flush=True)

    if workers == 1:
        for done, job in enumerate(jobs, start=1):
            collect(_build_one(job), done)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, result in enumerate(
                pool.map(_build_one, jobs, chunksize=4), start=1
            ):
                collect(result, done)

    if not all_windows:
        msg = "no matches produced windows"
        raise ValueError(msg)

    windows = pl.concat(all_windows, how="diagonal_relaxed").drop("window_id")
    window_ticks = pl.concat(all_ticks, how="diagonal_relaxed").drop("window_id")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        windows.write_parquet(out_dir / "windows.parquet")
        window_ticks.write_parquet(out_dir / "window_ticks.parquet")

    return windows, window_ticks


def cs2cd_matches(root: str | Path) -> list[Path]:
    """Every downloaded CS2CD match parquet under `root`, cheater folder first."""
    root = Path(root)
    return sorted((root / CHEATER_DIR).glob("*.parquet")) + sorted(
        (root / CLEAN_DIR).glob("*.parquet")
    )
