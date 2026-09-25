"""Score a judge model on held-out cases: the stock one, or your fine-tune.

Run:  uv run python training/llm/evaluate_judge.py --model models/llm/<file>.gguf

Four things are measured, in the order they matter:

    valid JSON       a verdict that will not parse is worthless
    label agreement  does "cheating" land on banned players
    probability AUC  does the number rank players, not just the word
    invented numbers figures in the reasoning that are not in the evidence

The last one is the reason this script exists. A judge that reasons convincingly
from numbers it made up is more dangerous than one that refuses to answer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from overwatch.layers.l3_behavior.evaluation import report
from overwatch.layers.l4_judge import Judge, Verdict
from overwatch.layers.l4_judge.judge import DEFAULT_MODEL
from overwatch.layers.l4_judge.rendering import unseen_numbers

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def invented_numbers(verdict: Verdict, evidence: str) -> list[str]:
    """Figures in the reasoning that do not appear in the evidence."""
    return [
        token
        for sentence in verdict.reasons + verdict.caveats
        for token in unseen_numbers(sentence, evidence)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / DEFAULT_MODEL)
    parser.add_argument(
        "--cases", type=Path, default=PROCESSED / "judge_training" / "val.jsonl"
    )
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.cases.read_text().splitlines()][
        : args.limit
    ]
    judge = Judge(args.model, gpu_layers=args.gpu_layers)
    print(
        f"{len(rows)} cases | {args.model.name} | {judge.describe()}",
        flush=True,
    )

    results, started = [], time.perf_counter()
    for n, row in enumerate(rows, start=1):
        evidence = row["messages"][1]["content"]
        target = json.loads(row["messages"][2]["content"])
        result = judge.judge_text(evidence)
        verdict = result.verdict
        results.append(
            {
                "match_id": row.get("match_id"),
                "player_id": row.get("player_id"),
                "label": row["label"],
                "target_verdict": target["verdict"],
                "verdict": verdict.verdict.value if result.valid else "invalid",
                "probability": verdict.probability if result.valid else None,
                "target_probability": target["probability"],
                "invented": invented_numbers(verdict, evidence) if result.valid else [],
                # kept so a surprising score can be read, not just counted
                "reasons": verdict.reasons if result.valid else [],
                "caveats": verdict.caveats if result.valid else [],
                "cheat_type": verdict.cheat_type.value if result.valid else None,
                "seconds": result.seconds,
                "tokens_per_second": result.tokens_per_second,
            }
        )
        if n % 20 == 0:
            print(
                f"  {n}/{len(rows)} ({time.perf_counter() - started:.0f}s)", flush=True
            )

    frame = pl.DataFrame(results)
    valid = frame.filter(pl.col("verdict") != "invalid")

    print(f"\nvalid JSON:           {valid.height / frame.height:.1%}")
    print(
        f"matches the target:   {(valid['verdict'] == valid['target_verdict']).mean():.1%}"
    )
    committed = valid.filter(pl.col("verdict") != "unclear")
    if committed.height:
        correct = (
            (committed["label"] == "cheater") == (committed["verdict"] == "cheating")
        ).mean()
        print(
            f"commits on {committed.height / valid.height:.0%} of cases, "
            f"{correct:.1%} right when it commits"
        )
    scored = valid.filter(pl.col("probability").is_not_null())
    if scored["label"].n_unique() > 1:
        auc = report(
            "probability",
            (scored["label"] == "cheater").to_numpy().astype(int),
            scored["probability"].to_numpy(),
            ci=False,
        )
        print(f"probability ROC-AUC:  {auc['roc_auc']:.3f}")
    invented = valid["invented"].list.len()
    print(
        f"invented numbers:     {invented.sum()} in {(invented > 0).sum()} of "
        f"{valid.height} answers"
    )
    print(
        f"speed:                {frame['seconds'].median():.1f}s per case, "
        f"{frame['tokens_per_second'].median():.1f} tok/s"
    )

    if args.out:
        frame.write_ndjson(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
