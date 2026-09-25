"""Measure what a stock model does with the evidence, before any fine-tuning.

Run:  uv run python training/llm/baseline_judge.py --cases 40

This is the number the Unsloth fine-tune has to beat, and it answers a question
worth settling early: is the evidence text itself readable? If a stock 4B model
can already separate cheaters from clean players here, the format is sound and
fine-tuning is about consistency and tone. If it cannot, no amount of fine-tuning
fixes a format that does not carry the signal.

Everything runs on CPU with no GPU layers, because the whole project has to work
on a mid-range PC. Speed is reported alongside accuracy for that reason.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import polars as pl

from overwatch.layers.l4_judge import Judge, PlayerCase
from overwatch.layers.l4_judge.judge import STOCK_MODEL

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-file", type=Path, default=PROCESSED / "judge_cases.jsonl"
    )
    parser.add_argument("--model", type=Path, default=ROOT / STOCK_MODEL)
    parser.add_argument("--cases", type=int, default=40, help="how many to judge")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--gpu-layers",
        default="auto",
        help="'auto' (default), 0 to force CPU, -1 for all layers on GPU",
    )
    parser.add_argument("--out", type=Path, default=PROCESSED / "judge_baseline.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.cases_file.read_text().splitlines()]
    # alternate labels so a slow run still gives a balanced read
    cheaters = [r for r in rows if r["label"] == "cheater"]
    clean = [r for r in rows if r["label"] == "clean"]
    selected = [r for pair in zip(cheaters, clean, strict=False) for r in pair][
        : args.cases
    ]

    print(f"{len(selected)} cases, model {args.model.name}", flush=True)
    judge = Judge(args.model, threads=args.threads, gpu_layers=args.gpu_layers)
    print(f"running on {judge.describe()}", flush=True)

    results, started = [], time.perf_counter()
    for n, row in enumerate(selected, start=1):
        # rendered now rather than read back: the renderer is the contract
        result = judge.judge(PlayerCase.model_validate(row["case"]))
        results.append(
            {
                "match_id": row["match_id"],
                "player_id": row["player_id"],
                "label": row["label"],
                "score": row["score"],
                "verdict": result.verdict.verdict.value if result.valid else "invalid",
                "probability": result.verdict.probability if result.valid else None,
                "cheat_type": result.verdict.cheat_type.value
                if result.valid
                else "none",
                "reasons": result.verdict.reasons if result.valid else [],
                "seconds": round(result.seconds, 2),
                "tokens_per_second": round(result.tokens_per_second, 1),
            }
        )
        if n % 10 == 0:
            print(
                f"  {n}/{len(selected)} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    frame = pl.DataFrame(results)
    frame.write_ndjson(args.out)

    print(f"\n{len(frame)} verdicts in {time.perf_counter() - started:.0f}s")
    print(
        f"median {frame['seconds'].median():.1f}s per case, "
        f"{frame['tokens_per_second'].median():.1f} tokens/s on CPU\n"
    )

    with pl.Config(tbl_rows=20, float_precision=3):
        print(frame.group_by("label", "verdict").len().sort(["label", "verdict"]))

    # accuracy when the model commits; "unclear" is counted separately, since
    # refusing to guess is a legitimate answer for a reviewer's assistant
    invalid = (frame["verdict"] == "invalid").mean()
    print(f"\nunparseable answers: {invalid:.1%}")
    decided = frame.filter(~pl.col("verdict").is_in(["unclear", "invalid"]))
    if decided.height:
        correct = (
            (decided["label"] == "cheater") == (decided["verdict"] == "cheating")
        ).mean()
        print(
            f"\ncommitted on {decided.height}/{frame.height} cases, "
            f"{correct:.1%} correct when committing"
        )
    print(f"unclear: {(frame['verdict'] == 'unclear').mean():.1%}")
    print("\ncheat types named:", Counter(frame["cheat_type"].to_list()).most_common())

    print("\n--- example reasoning (a cheater the model caught) ---")
    for row in results:
        if row["label"] == "cheater" and row["verdict"] == "cheating":
            print(json.dumps(row["reasons"], indent=2))
            break


if __name__ == "__main__":
    main()
