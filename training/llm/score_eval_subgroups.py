"""Score one judge eval file, and break the score down where a regression could hide.

Run:  uv run python training/llm/score_eval_subgroups.py \
          data/processed/judge_eval_v4_val206_mastertext.jsonl \
          --cases data/processed/judge_cases_v5.jsonl

Aggregate target match is one number over 206 cases, and one number cannot say
*which* cases moved. The held-out split carries no map and no rank (CS2CD cases
leave both null), so the subgroups that exist are the ones the evidence itself
draws: the label, what the target verdict was, how many kills the case had, and
where the case sits against a measurement's clean lines.

The last one is the point. When a measurement is dropped from the rendered text,
the cases that lose information are the ones where that measurement was extreme.
Pass --feature to split on any feature the cases carry (default `snap_max`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def band(value: float | None, lines: list[float] | None, higher: bool) -> str:
    """Where one value sits against its clean 95%/99% lines."""
    if value is None or not lines:
        return "no measurement"
    notable, strong = lines[0], lines[1]
    if higher:
        if value >= strong:
            return "past 99% line"
        return "past 95% line" if value >= notable else "inside clean range"
    if value <= strong:
        return "past 99% line"
    return "past 95% line" if value <= notable else "inside clean range"


def summarise(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {name:<24} 0 cases")
        return
    match = sum(r["verdict"] == r["target_verdict"] for r in rows) / len(rows)
    invented = sum(len(r["invented"]) for r in rows)
    unclear = sum(r["verdict"] == "unclear" for r in rows)
    invalid = sum(r["verdict"] == "invalid" for r in rows)
    print(
        f"  {name:<24} n={len(rows):<4} target match {match:6.1%}  "
        f"unclear {unclear:<3} invalid {invalid:<3} invented {invented}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval", type=Path, help="an evaluate_judge.py --out file")
    parser.add_argument(
        "--cases", type=Path, default=PROCESSED / "judge_cases_v5.jsonl"
    )
    parser.add_argument("--feature", default="snap_max")
    parser.add_argument("--higher", action="store_true", default=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.eval.read_text().splitlines()]
    cases = {}
    for line in args.cases.read_text().splitlines():
        row = json.loads(line)
        cases[(row["match_id"], row["player_id"])] = row["case"]

    print(f"{args.eval}: {len(rows)} cases")
    summarise("all", rows)
    invented = [r for r in rows if r["invented"]]
    if invented:
        print(f"  invented numbers in {len(invented)} answers:")
        for r in invented[:10]:
            print(f"    {r['match_id']} {r['player_id']}: {r['invented']}")

    print("\nby label")
    for label in sorted({r["label"] for r in rows}):
        summarise(label, [r for r in rows if r["label"] == label])

    print("\nby target verdict")
    for target in sorted({r["target_verdict"] for r in rows}):
        summarise(target, [r for r in rows if r["target_verdict"] == target])

    print(f"\nby {args.feature} against its clean lines")
    bands: dict[str, list[dict]] = {}
    for r in rows:
        case = cases.get((r["match_id"], r["player_id"]), {})
        value = (case.get("features") or {}).get(args.feature)
        lines = (case.get("clean_lines") or {}).get(args.feature)
        bands.setdefault(band(value, lines, args.higher), []).append(r)
    for name in (
        "inside clean range",
        "past 95% line",
        "past 99% line",
        "no measurement",
    ):
        if name in bands:
            summarise(name, bands[name])

    print("\nby kills in the case")
    for name, lo, hi in (
        ("<= 10 kills", 0, 10),
        ("11-20 kills", 11, 20),
        ("> 20 kills", 21, 10**6),
    ):
        summarise(
            name,
            [
                r
                for r in rows
                if lo
                <= (cases.get((r["match_id"], r["player_id"]), {}).get("kills") or 0)
                <= hi
            ],
        )

    print("\nverdict mix")
    print(f"  {dict(Counter(r['verdict'] for r in rows))}")


if __name__ == "__main__":
    main()
