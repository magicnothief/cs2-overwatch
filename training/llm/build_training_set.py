"""Build the judge's training set: evidence in, a reasoned verdict out.

Run:  uv run python training/llm/build_training_set.py

Each case gets a target: a human annotation when there is one for the evidence as
it renders today (data/annotations/gold.jsonl, from `python -m overwatch.annotation`),
otherwise the generated target from overwatch.layers.l4_judge.targets. How often
the two agree is reported, because that is the measure of the generated ones.

Output is chat JSONL, ready for Unsloth Studio, split by match so no player appears
in both halves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from overwatch.annotation import AnnotationStore, case_key, evidence_hash
from overwatch.layers.l4_judge import SYSTEM_PROMPT, PlayerCase, render_case
from overwatch.layers.l4_judge.targets import (
    build_target,
    case_rng,
    split_for,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-file", type=Path, default=PROCESSED / "judge_cases.jsonl"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data" / "annotations" / "gold.jsonl",
        help="human targets, which replace generated ones case by case",
    )
    parser.add_argument("--out", type=Path, default=PROCESSED / "judge_training")
    parser.add_argument("--val-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.cases_file.read_text().splitlines()]
    annotations = AnnotationStore(args.annotations).load()

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, list[str]] = {"train": [], "val": []}
    sources: list[dict] = []

    for row in rows:
        case = PlayerCase.model_validate(row["case"])
        evidence = render_case(case)
        generated = build_target(
            case, case_rng(case.match_id, case.player_id, args.seed)
        )
        human = annotations.get(case_key(case.match_id, case.player_id))
        fresh = human is not None and human.evidence_sha == evidence_hash(evidence)
        target = human.verdict if fresh else generated
        split = split_for(case.match_id, args.val_share)

        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": evidence},
                {"role": "assistant", "content": target.model_dump_json()},
            ],
            "label": row["label"],
            "match_id": row["match_id"],
            "player_id": row["player_id"],
            "target_source": "human" if fresh else "generated",
        }
        counts[split].append(json.dumps(record))
        sources.append(
            {
                "split": split,
                "label": row["label"],
                "source": record["target_source"],
                "stale": human is not None and not fresh,
                "verdict": target.verdict.value,
                "generated": generated.verdict.value,
                "gap": abs(target.probability - generated.probability),
            }
        )

    # what Unsloth Studio is given: the conversation and nothing else, so no
    # bookkeeping field (the ban label above all) can reach the model's input
    upload = args.out / "unsloth"
    upload.mkdir(exist_ok=True)
    for split, records in counts.items():
        (args.out / f"{split}.jsonl").write_text("\n".join(records) + "\n")
        conversations = [{"messages": json.loads(r)["messages"]} for r in records]
        (upload / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(c) for c in conversations) + "\n"
        )
        # the same conversations as Parquet, for importers that prefer it
        pl.DataFrame(conversations).write_parquet(upload / f"{split}.parquet")
        print(f"{split}: {len(records)} examples -> {args.out / f'{split}.jsonl'}")
    print(f"upload to Unsloth Studio: {upload}/{{train,val}}.jsonl (or .parquet)")

    table = pl.DataFrame(sources)
    print("\ntargets by label (train):")
    print(
        table.filter(pl.col("split") == "train")
        .group_by("label", "verdict")
        .len()
        .sort(["label", "verdict"])
    )
    report_annotations(table)


def report_annotations(table: pl.DataFrame) -> None:
    """How far the generated targets are from what a person decided."""
    human = table.filter(pl.col("source") == "human")
    stale = table["stale"].sum()
    if stale:
        print(
            f"\n{stale} annotations were made on evidence that has since changed; "
            "they were not used (re-annotate them in the tool)"
        )
    if human.is_empty():
        print("\nno human targets yet: python -m overwatch.annotation")
        return
    agree = (human["verdict"] == human["generated"]).mean()
    print(
        f"\nhuman targets: {human.height} "
        f"({human.filter(pl.col('split') == 'val').height} in validation)\n"
        f"generated targets agree with them on {agree:.0%} of verdicts, "
        f"probabilities {human['gap'].mean():.0f} points apart on average"
    )
    print(
        human.group_by("label")
        .agg(n=pl.len(), agree=(pl.col("verdict") == pl.col("generated")).mean())
        .sort("label")
    )


if __name__ == "__main__":
    main()
