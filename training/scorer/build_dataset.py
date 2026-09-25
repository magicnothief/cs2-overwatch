"""Rebuild the labeled window dataset from downloaded CS2CD matches.

Run:  uv run python training/scorer/build_dataset.py
      uv run python training/scorer/build_dataset.py --limit 50   # quick pass

Writes data/processed/{windows,window_ticks,window_features}.parquet.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import polars as pl

from overwatch.dataset import build_window_dataset, cs2cd_matches
from overwatch.layers.l3_behavior.features import build_window_features

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "cs2cd")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--limit", type=int, default=None, help="use only N matches")
    parser.add_argument("--pre", type=int, default=128, help="ticks before each kill")
    parser.add_argument("--post", type=int, default=32, help="ticks after each kill")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel processes (default: cores/2)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    paths = cs2cd_matches(args.root)
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} matches from {args.root}", flush=True)

    started = time.time()
    windows, window_ticks = build_window_dataset(
        paths, pre=args.pre, post=args.post, out_dir=args.out, workers=args.workers
    )
    features = build_window_features(window_ticks)
    features.write_parquet(args.out / "window_features.parquet")

    print(f"built in {time.time() - started:.0f}s")
    print(f"{windows.height} windows, {window_ticks.height} window ticks")
    print(windows.group_by("label").len().sort("label"))
    print(
        windows.select(
            matches=pl.col("match_id").n_unique(),
            players=pl.struct("match_id", "attacker_id").n_unique(),
        )
    )


if __name__ == "__main__":
    main()
