"""Assemble player cases: the evidence the judge model reads.

Run:  uv run python training/llm/build_cases.py --per-label 60

One case per player, built from the tables the detector already produced. The same
builder serves three purposes, which is the point:

    the baseline evaluation   (does a stock model read this correctly?)
    the Unsloth training set  (input text -> verdict JSON)
    the pipeline at runtime   (a real report on a real demo)

If those ever diverge, the fine-tuned model is being fed something it never saw in
training, so they share one code path.

Sampling is deliberate: equal numbers of cheaters and clean players, plus the
*highest-scoring clean players* as hard negatives. Those are the ones a judge must
not condemn — skilled players who look suspicious to a detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from overwatch.layers.l4_judge import PlayerCase, render_case
from overwatch.layers.l4_judge.cases import KILL_COLUMNS, build_case, clean_baselines
from overwatch.layers.l4_judge.targets import clean_lines

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def build_cases(
    players: pl.DataFrame,
    windows: pl.DataFrame,
    *,
    baselines: dict[str, float],
    lines: dict[str, tuple[float, float]],
    window_ticks: pl.DataFrame | None = None,
    rules: pl.DataFrame | None = None,
) -> list[tuple[PlayerCase, str]]:
    """Build (case, label) pairs. The label never enters the rendered text."""
    return [
        (
            build_case(
                row,
                windows,
                baselines=baselines,
                lines=lines,
                window_ticks=window_ticks,
                rules=rules,
            ),
            row["label"],
        )
        for row in players.iter_rows(named=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED)
    parser.add_argument("--per-label", type=int, default=60, help="cases per class")
    parser.add_argument("--hard-negatives", type=int, default=30)
    parser.add_argument("--out", type=Path, default=PROCESSED / "judge_cases.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scores",
        default="meshes23",
        help="which cross_validate.py run supplies the behaviour score "
        "(data/processed/cv_player_scores_<label>.parquet)",
    )
    args = parser.parse_args()

    players = pl.read_parquet(args.data / "player_features.parquet")
    # per-window features carry the measurements; windows.parquet carries the kill
    # itself (tick, weapon, headshot), and a reviewer needs both
    window_meta = pl.read_parquet(args.data / "windows.parquet")
    windows = pl.read_parquet(args.data / "window_features.parquet").join(
        window_meta.select(
            "window_uid", *[c for c in KILL_COLUMNS if c in window_meta.columns]
        ),
        on="window_uid",
        how="left",
    )

    scores_path = args.data / f"cv_player_scores_{args.scores}.parquet"
    if scores_path.exists():
        scores = pl.read_parquet(scores_path).with_columns(
            match_id=pl.col("player").str.split("/").list.slice(0, 2).list.join("/"),
            player_id=pl.col("player").str.split("/").list.last(),
        )
        players = players.join(
            scores.select("match_id", "player_id", "score"),
            on=["match_id", "player_id"],
            how="left",
        )
    else:
        players = players.with_columns(score=pl.lit(0.0))

    baselines = clean_baselines(players)
    lines = clean_lines(players)

    cheaters = players.filter(pl.col("label") == "cheater").sample(
        n=min(args.per_label, players.filter(pl.col("label") == "cheater").height),
        seed=args.seed,
    )
    clean_pool = players.filter(pl.col("label") == "clean")
    clean = clean_pool.sample(n=min(args.per_label, clean_pool.height), seed=args.seed)
    # the clean players the detector likes least: a judge must not condemn them
    hard = clean_pool.sort("score", descending=True, nulls_last=True).head(
        args.hard_negatives
    )

    # a stable order: unique() alone reorders rows between runs, and anything that
    # lines results up with cases by position then compares the wrong cases
    selected = (
        pl.concat([cheaters, clean, hard])
        .unique(subset=["match_id", "player_id"], keep="first")
        .sort("match_id", "player_id")
    )
    window_ticks = pl.read_parquet(
        args.data / "window_ticks.parquet",
        columns=[
            "window_uid",
            "tick_offset",
            "t_ms",
            "target_angle",
            "target_visible",
            "yaw_speed",
        ],
    )
    rules_path = args.data / "rule_evidence.parquet"
    rules = pl.read_parquet(rules_path) if rules_path.exists() else None
    if rules is None:
        print("no rule_evidence.parquet: run training/rules/collect_rule_evidence.py")
    cases = build_cases(
        selected,
        windows,
        baselines=baselines,
        lines=lines,
        window_ticks=window_ticks,
        rules=rules,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for case, label in cases:
            fh.write(
                json.dumps(
                    {
                        "match_id": case.match_id,
                        "player_id": case.player_id,
                        "label": label,
                        "score": case.score,
                        "evidence": render_case(case),
                        "case": case.model_dump(mode="json"),
                    }
                )
                + "\n"
            )

    counts = selected.group_by("label").len().sort("label")
    print(f"{len(cases)} cases -> {args.out}")
    print(counts)
    print("\n--- example (cheater) ---")
    for case, label in cases:
        if label == "cheater":
            print(render_case(case))
            break


if __name__ == "__main__":
    main()
