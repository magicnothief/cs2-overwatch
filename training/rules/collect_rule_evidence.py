"""Run every Layer 1 rule over every match, and keep the evidence.

Run:  uv run python training/rules/collect_rule_evidence.py

The judge was never shown Layer 1's findings: `PlayerCase.rule_evidence` existed
but nothing filled it, so the most damning evidence the detector produces — a
sustained 11,000 deg/s spin, a pitch past the game's clamp — never reached the
model meant to explain it. Rules need a whole match's ticks rather than kill
windows, so they are run once here and stored, and the case builder looks them up.

Output: data/processed/rule_evidence.parquet, one row per piece of evidence.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from overwatch.dataset import cs2cd_matches, default_workers
from overwatch.layers.l1_blatant.rules import evidence_rows, run_rules
from overwatch.parsing import load_cs2cd_cached

ROOT = Path(__file__).resolve().parents[2]


def evidence_for(path: Path) -> list[dict]:
    """Every rule hit in one match, flattened to rows."""
    match = load_cs2cd_cached(path)
    match_id = f"{path.parent.name}/{path.stem}"
    return evidence_rows(run_rules(match.ticks, match.deaths, match_id=match_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "cs2cd")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "rule_evidence.parquet",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    paths = cs2cd_matches(args.root)
    started = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers or default_workers()) as pool:
        for done, found in enumerate(
            pool.map(evidence_for, paths, chunksize=4), start=1
        ):
            rows.extend(found)
            if done % 100 == 0:
                print(f"  {done}/{len(paths)} matches", flush=True)

    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    frame.write_parquet(args.out)
    print(
        f"{len(rows)} pieces of evidence from {len(paths)} matches "
        f"in {time.perf_counter() - started:.0f}s -> {args.out}"
    )
    if rows:
        print(frame.group_by("name", "severity").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
