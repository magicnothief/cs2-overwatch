"""Step 0 of the round replay: do "look here" marks separate cheaters on CS2CD?

Spec: docs/specs/2026-09-27-round-replay.md, section 3. The rule's
thresholds were fixed in the spec before the first run; both runs and why there
are two are in docs/evals/2026-09-27-replay-marks-gate.md. The gate failed and
the marks were cut, so the rule lives here, not in the replay. Marks ship only if both hold:

    1. at most 10% of clean players get one or more marks in a match
    2. banned players get one at least twice as often as clean players do

Run:  uv run python training/replay/marks_gate.py
Writes data/processed/replay_marks_gate.parquet (one row per player per match).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from overwatch.layers.l2_perception.geometry.occlusion import has_map
from overwatch.layers.l3_behavior.features import ON_TARGET_DEG
from overwatch.parsing import load_cs2cd_cached
from overwatch.pipeline.replay import SAMPLE_TICKS, enemy_pairs, samples, sightings

RAW = Path("data/raw/cs2cd")
OUT = Path("data/processed/replay_marks_gate.parquet")
CLEAN_CEILING = 0.10
MIN_LIFT = 2.0


#: Crosshair within this many degrees of the enemy's head.
MARK_ANGLE_DEG = ON_TARGET_DEG
#: For at least this long: 0.5 s at 64 tick.
MARK_MIN_TICKS = 32
#: While the enemy's bearing from the viewer turns by at least this much.
MARK_TURN_DEG = 10.0


def find_marks(ticks: pl.DataFrame, map_name: str) -> pl.DataFrame:
    """Every mark in a match: player_id, enemy_id, round, start_tick, end_tick."""
    pairs = sightings(enemy_pairs(samples(ticks)), map_name)
    # an enemy is hidden from a team when no one on it sees them
    seen = pairs.group_by("tick", "team", "enemy_id").agg(
        team_sees=pl.col("sees").any()
    )
    on = pairs.join(seen, on=["tick", "team", "enemy_id"]).filter(
        ~pl.col("team_sees") & (pl.col("target_angle") < MARK_ANGLE_DEG)
    )
    pair = ("player_id", "enemy_id")
    on = on.sort(*pair, "tick").with_columns(
        run=(pl.col("tick").diff().over(pair) != SAMPLE_TICKS)
        .fill_null(True)
        .cum_sum()
        .over(pair),
        bearing=pl.arctan2(
            pl.col("target_y") - pl.col("y"), pl.col("target_x") - pl.col("x")
        ).degrees(),
    )
    runs = on.group_by(*pair, "run").agg(
        round=pl.col("round").first(),
        start_tick=pl.col("tick").min(),
        end_tick=pl.col("tick").max(),
        # the bearing's turn over the stretch, unwrapped around its first value
        turn=(((pl.col("bearing") - pl.col("bearing").first() + 180) % 360) - 180).max()
        - (((pl.col("bearing") - pl.col("bearing").first() + 180) % 360) - 180).min(),
    )
    return (
        runs.filter(
            (pl.col("end_tick") - pl.col("start_tick") + SAMPLE_TICKS >= MARK_MIN_TICKS)
            & (pl.col("turn") >= MARK_TURN_DEG)
        )
        .select(*pair, "round", "start_tick", "end_tick")
        .sort("start_tick", *pair)
    )


def one_match(path: Path) -> pl.DataFrame | None:
    match = load_cs2cd_cached(path)
    map_name = match.meta.get("map") or ""
    if not has_map(map_name):
        return None
    cheaters = set(match.meta.get("cheaters") or [])
    players = match.ticks.select("player_id").unique()
    marks = find_marks(match.ticks, map_name).group_by("player_id").agg(marks=pl.len())
    return (
        players.join(marks, on="player_id", how="left")
        .with_columns(
            match_id=pl.lit(f"{path.parent.name}/{path.stem}"),
            map=pl.lit(map_name),
            folder=pl.lit(path.parent.name),
            marks=pl.col("marks").fill_null(0),
            label=pl.when(pl.col("player_id").is_in(list(cheaters)))
            .then(pl.lit("cheater"))
            .otherwise(pl.lit("clean")),
        )
        .with_columns(pl.col("marks").cast(pl.Int64))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    paths = sorted(RAW.glob("*/*.parquet"))
    with ProcessPoolExecutor(args.workers) as pool:
        table = pl.concat(
            t for t in pool.map(one_match, paths, chunksize=8) if t is not None
        )
    table.write_parquet(OUT)

    pl.Config.set_tbl_width_chars(160)
    summary = (
        table.with_columns(marked=pl.col("marks") > 0)
        .group_by("label")
        .agg(
            players=pl.len(),
            with_a_mark=pl.col("marked").mean(),
            marks_median=pl.col("marks").median(),
            marks_p90=pl.col("marks").quantile(0.9),
        )
        .sort("label")
    )
    print(
        f"{table['match_id'].n_unique()} of {len(paths)} matches (with a map mesh), {len(table)} player-matches"
    )
    print(summary)
    clean_only = table.filter(pl.col("folder") == "no_cheater_present")
    print(
        "clean players in matches with no cheater at all: "
        f"{(clean_only['marks'] > 0).mean():.4f} with a mark"
    )
    rates = dict(zip(summary["label"], summary["with_a_mark"], strict=True))
    clean, cheater = rates["clean"], rates["cheater"]
    gate1 = clean <= CLEAN_CEILING
    gate2 = cheater >= MIN_LIFT * clean
    print(
        f"gate 1, clean with a mark <= {CLEAN_CEILING:.0%}: {clean:.4f} -> {'pass' if gate1 else 'FAIL'}"
    )
    print(
        f"gate 2, cheater rate >= {MIN_LIFT}x clean: {cheater:.4f} / {clean:.4f} = "
        f"{cheater / clean if clean else float('inf'):.2f}x -> {'pass' if gate2 else 'FAIL'}"
    )
    print("marks ship" if gate1 and gate2 else "marks are cut (spec section 3)")


if __name__ == "__main__":
    main()
