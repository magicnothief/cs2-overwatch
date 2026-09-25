"""Measure what Layer 1 catches, and what it wrongly accuses.

Run:  uv run python training/rules/evaluate_rules.py --matches 150

Layer 1 is judged differently from the models. Recall is not the point: these
rules exist to be right when they fire. So the numbers that matter are the hit
rate among labeled cheaters and, above all, the hit rate among clean players —
remembering that CS2CD's clean set is itself ~3% contaminated, so some of those
"false positives" are real cheaters the labels missed.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import polars as pl

from overwatch.dataset import CHEATER_DIR, CLEAN_DIR
from overwatch.layers.l1_blatant.rules import RULES, run_rules
from overwatch.layers.l1_blatant.thresholds import Thresholds
from overwatch.parsing import load_cs2cd
from overwatch.schemas.evidence import Moment

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def evaluate_match(
    parquet: Path, thresholds: Thresholds
) -> tuple[pl.DataFrame, list[Moment]]:
    """Run every rule over one match and label each player."""
    match = load_cs2cd(parquet)
    match_id = f"{parquet.parent.name}/{parquet.stem}"
    cheaters = set(match.meta.get("cheaters", []))
    had_cheater = parquet.parent.name == CHEATER_DIR

    moments = run_rules(
        match.ticks, match.deaths, thresholds=thresholds, match_id=match_id
    )
    hits = Counter((m.player_id, m.trigger) for m in moments)

    rows = []
    for player in match.ticks["player_id"].unique().sort().to_list():
        label = (
            "cheater" if player in cheaters else ("unknown" if had_cheater else "clean")
        )
        row = {"match_id": match_id, "player_id": player, "label": label}
        for rule in RULES:
            row[rule] = hits.get((player, rule), 0)
        rows.append(row)
    return pl.DataFrame(rows), moments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "cs2cd")
    parser.add_argument("--matches", type=int, default=150, help="per folder")
    args = parser.parse_args()

    thresholds = Thresholds.load()
    print(f"thresholds: spin {thresholds.spin_speed_dps:.0f} deg/s sustained, ")
    print(f"            snap {thresholds.snap_speed_dps:.0f} deg/s, ")
    print(f"            aliased spin {thresholds.alias_min_ticks} ticks\n")

    paths = (
        sorted((args.root / CHEATER_DIR).glob("*.parquet"))[: args.matches]
        + sorted((args.root / CLEAN_DIR).glob("*.parquet"))[: args.matches]
    )

    started = time.time()
    frames, all_moments = [], []
    for n, path in enumerate(paths, start=1):
        frame, moments = evaluate_match(path, thresholds)
        frames.append(frame)
        all_moments.extend(moments)
        if n % 50 == 0:
            print(f"  {n}/{len(paths)} matches ({time.time() - started:.0f}s)")

    players = pl.concat(frames).with_columns(
        any_rule=pl.sum_horizontal([pl.col(rule) for rule in RULES]) > 0
    )

    print(
        f"\n{players.height} players in {len(paths)} matches, {time.time() - started:.0f}s\n"
    )
    summary = (
        players.group_by("label")
        .agg(
            players=pl.len(),
            **{rule: (pl.col(rule) > 0).mean() for rule in RULES},
            any_rule=pl.col("any_rule").mean(),
        )
        .sort("label")
    )
    with pl.Config(float_precision=4, tbl_width_chars=140):
        print("share of players each rule fires on:")
        print(summary)

    flagged = players.filter(pl.col("any_rule"))
    labelled = flagged.filter(pl.col("label") != "unknown")
    if labelled.height:
        precision = (labelled["label"] == "cheater").mean()
        print(
            f"\nof {labelled.height} flagged players with a usable label, "
            f"{precision:.1%} are labeled cheaters"
        )
    print(
        "clean players flagged: "
        f"{flagged.filter(pl.col('label') == 'clean').height} "
        f"of {players.filter(pl.col('label') == 'clean').height}"
    )

    print("\nexample evidence:")
    for moment in sorted(all_moments, key=lambda m: -m.evidence[0].value)[:5]:
        e = moment.evidence[0]
        print(f"  [{e.severity}] {moment.match_id} {moment.player_id}: {e.note}")
        print(f"      {moment.demo_command()}")

    players.write_parquet(PROCESSED / "rule_hits.parquet")
    print(f"\nwrote {PROCESSED / 'rule_hits.parquet'}")


if __name__ == "__main__":
    main()
