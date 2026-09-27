"""The same arrival table as training/arrival/arrival_lab.py, built on the pro sample.

Gate 3 of docs/specs/2026-09-26-triggerbot-and-snap-count.md is a feature-level
check: how many pros a clean line puts past it. It needs no judge and no GPU, so
this stops short of the judge and dumps what a redesign has to sweep over. Every
candidate arrival definition then becomes an offline group-by on one parquet
instead of a 35-demo rerun.

    PYTHONPATH=src .venv/bin/python training/arrival/pro_arrival_lab.py \
        --per-map 5 --seed 0 --out data/processed/pro_arrival_lab

Demos stream: downloaded, parsed, deleted (--keep keeps them). A rerun skips
demos already dumped. Visibility is not computed — target_angle and
target_distance are pure geometry, so no map mesh is needed and layer 2's ray
casting is skipped.

Writes, per demo, `<out>/<demo>.parquet` with arrival_lab.py's columns plus
`kills` (the player's kill count in that match) and `snap_kills`, and
`<out>/listing.json` naming the sample.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arrival_lab import arrival_lab

from overwatch import paths
from overwatch.dataset import match_windows
from overwatch.layers.l3_behavior.player_features import (
    SNAP_THRESHOLD_DPS,
)
from overwatch.parsing.demo import parse_demo

sys.path.insert(0, str(paths.HOME / "training"))

from check_pro_demos import DATASET, sample


def dump(demo: Path, match_id: str, out: Path) -> int:
    """One demo -> one parquet of every arrival, keyed by player."""
    match = parse_demo(demo)
    windows, window_ticks = match_windows(match, match_id)
    if windows.is_empty():
        return 0
    lab = arrival_lab(window_ticks.lazy())
    kills = windows.select(
        "window_uid", player_id=pl.col("attacker_id"), weapon=pl.col("weapon")
    )
    # build_window_features:166 — the kill tick's yaw_speed, as it stands
    speeds = window_ticks.filter(pl.col("tick_offset") == 0).select(
        "window_uid", speed_at_kill=pl.col("yaw_speed")
    )
    rows = (
        kills.join(speeds, on="window_uid", how="left")
        .join(lab, on="window_uid", how="left")
        .with_columns(match_id=pl.lit(match_id))
    )
    rows.write_parquet(out)
    return windows.height


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-map", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=paths.DATA / "processed" / "pro_arrival_lab"
    )
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    demos = sample(args.per_map, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "listing.json").write_text(
        json.dumps(
            {"per_map": args.per_map, "seed": args.seed, "demos": demos}, indent=1
        )
    )
    raw = paths.DATA / "raw" / "hltv"
    for n, remote in enumerate(demos, 1):
        stem = Path(remote).stem
        target = args.out / f"{stem}.parquet"
        if target.exists():
            print(f"[{n}/{len(demos)}] {stem}: done already", flush=True)
            continue
        started = time.perf_counter()
        local = Path(
            hf_hub_download(DATASET, remote, repo_type="dataset", local_dir=raw)
        )
        fetched = time.perf_counter() - started
        try:
            kills = dump(local, stem, target)
        except Exception as exc:  # noqa: BLE001 - a broken demo must not end the run
            print(
                f"[{n}/{len(demos)}] {stem}: failed ({type(exc).__name__}: {exc})",
                flush=True,
            )
            continue
        finally:
            if not args.keep:
                local.unlink(missing_ok=True)
        print(
            f"[{n}/{len(demos)}] {stem}: {kills} kills "
            f"(download {fetched:.0f}s, parse {time.perf_counter() - started - fetched:.0f}s)",
            flush=True,
        )
    print(f"snap threshold {SNAP_THRESHOLD_DPS} deg/s; wrote {args.out}")


if __name__ == "__main__":
    main()
