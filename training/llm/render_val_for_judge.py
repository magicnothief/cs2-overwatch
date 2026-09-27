"""Re-render a held-out split's evidence for one judge generation, and say what moved.

Run:  uv run python training/llm/render_val_for_judge.py \
          --split data/processed/judge_training/val.jsonl \
          --cases data/processed/judge_cases_v5.jsonl \
          --judge 4 --out data/processed/judge_val_master_v4.jsonl

Why this exists: a split on disk carries the evidence text as it rendered the day it
was built, and the model's published numbers were measured against *that* text. When
the renderer changes, the only way to know whether the numbers still hold is to keep
the split — same rows, same targets, same order — and swap in the text today's code
would put in front of the same judge. Everything else stays byte-identical, so a
score difference can only come from the text.

It also reports the comparison itself: how many of the rows render differently now,
and the first few diffs, because "no row changed" is a finding, not a formality.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from overwatch.layers.l4_judge.rendering import PlayerCase, render_case

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=Path,
        default=PROCESSED / "judge_training" / "val.jsonl",
        help="the split to re-render, kept row for row",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROCESSED / "judge_cases_v5.jsonl",
        help="cases to render from; use the newest build, since render_case decides "
        "per judge which of their measurements are shown",
    )
    parser.add_argument("--judge", type=int, required=True, help="judge generation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--show-diffs", type=int, default=3)
    args = parser.parse_args()

    cases = {}
    for line in args.cases.read_text().splitlines():
        row = json.loads(line)
        cases[(row["match_id"], row["player_id"])] = row["case"]

    rows = [json.loads(line) for line in args.split.read_text().splitlines()]
    changed, missing, out = [], [], []
    for row in rows:
        key = (row["match_id"], row["player_id"])
        if key not in cases:
            missing.append(key)
            continue
        before = row["messages"][1]["content"]
        after = render_case(PlayerCase.model_validate(cases[key]), judge=args.judge)
        if before != after:
            changed.append((key, before, after))
        fresh = dict(row)
        fresh["messages"] = [
            row["messages"][0],
            {"role": "user", "content": after},
            row["messages"][2],
        ]
        out.append(json.dumps(fresh))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")

    print(
        f"{len(rows)} rows in {args.split}\n"
        f"{len(out)} re-rendered for judge {args.judge} -> {args.out}\n"
        f"{len(changed)} differ from the text stored in the split"
    )
    if missing:
        print(f"{len(missing)} rows had no case in {args.cases}: {missing[:5]}")
    for key, before, after in changed[: args.show_diffs]:
        print(f"\n--- {key[0]} {key[1]}")
        print(
            "\n".join(
                difflib.unified_diff(
                    before.splitlines(), after.splitlines(), "stored", "rendered", n=1
                )
            )
        )


if __name__ == "__main__":
    main()
