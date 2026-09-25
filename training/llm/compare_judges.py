"""Compare judge models on the same validation cases, against today's targets.

Run:  uv run python training/llm/compare_judges.py stock=<eval.jsonl> v1=<...> v2=<...>

Each input is the per-case output of evaluate_judge.py on judge_training/val.jsonl.
Answers are matched to cases by match and player, never by position: rebuilding
the cases once reordered them, and position matching would have compared answers
with the wrong cases. They are scored against the targets as they are now.

Beyond the usual numbers, two columns exist because of what judge v1 did:

    follows evidence   correlation of the judge's probability with the target's,
                       which comes from the evidence alone
    leans on score     correlation of its probability with the behaviour-score
                       line, after removing what the evidence already explains.
                       v1 copied that score where evidence was thin; a judge that
                       reads the measurements should sit near zero.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "data" / "processed" / "judge_training" / "val.jsonl"
SCORE = re.compile(r"Behaviour model score: ([0-9.]+)")


def cases(path: Path) -> pl.DataFrame:
    """The validation cases: ban label, today's target, and the score line."""
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        target = json.loads(record["messages"][2]["content"])
        rows.append(
            {
                "match_id": record["match_id"],
                "player_id": record["player_id"],
                "label": record["label"],
                "target_verdict": target["verdict"],
                "target_probability": target["probability"],
                "score": float(SCORE.search(record["messages"][1]["content"]).group(1)),
            }
        )
    return pl.DataFrame(rows)


def residual_correlation(y: np.ndarray, x: np.ndarray, control: np.ndarray) -> float:
    """Correlation of y with x once a straight-line fit on `control` is removed."""

    def residual(v: np.ndarray) -> np.ndarray:
        slope, intercept = np.polyfit(control, v, 1)
        return v - (slope * control + intercept)

    return float(np.corrcoef(residual(y), residual(x))[0, 1])


def summarise(name: str, answers: pl.DataFrame, truth: pl.DataFrame) -> dict:
    frame = truth.join(
        answers.select(
            "match_id", "player_id", "verdict", "probability", "invented", "seconds"
        ),
        on=["match_id", "player_id"],
        how="inner",
    )
    valid = frame.filter(pl.col("verdict") != "invalid")
    clean = valid.filter(pl.col("label") == "clean")
    banned = valid.filter(pl.col("label") == "cheater")
    committed = valid.filter(pl.col("verdict") != "unclear")
    y = (valid["label"] == "cheater").to_numpy()
    p = valid["probability"].to_numpy().astype(float)
    return {
        "judge": name,
        "valid JSON": valid.height / frame.height,
        "matches target": (valid["verdict"] == valid["target_verdict"]).mean(),
        "accuses clean": (clean["verdict"] == "cheating").mean(),
        "convicts banned": (banned["verdict"] == "cheating").mean(),
        "unclear": (valid["verdict"] == "unclear").mean(),
        "right when committing": (
            (committed["label"] == "cheater") == (committed["verdict"] == "cheating")
        ).mean(),
        "AUC": roc_auc_score(y, p),
        "follows evidence": float(np.corrcoef(p, valid["target_probability"])[0, 1]),
        "leans on score": residual_correlation(
            p, valid["score"].to_numpy(), valid["target_probability"].to_numpy()
        ),
        "invented": int(valid["invented"].list.len().sum()),
        "s/case": frame["seconds"].median(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="name=path/to/judge_eval.jsonl")
    parser.add_argument("--val", type=Path, default=VAL)
    args = parser.parse_args()

    truth = cases(args.val)
    rows = []
    for run in args.runs:
        name, path = run.split("=", 1)
        answers = pl.read_ndjson(path)
        if "player_id" not in answers.columns:
            msg = f"{name}: answers carry no case identity; re-run evaluate_judge.py"
            raise SystemExit(msg)
        matched = truth.join(answers, on=["match_id", "player_id"], how="semi").height
        if matched != truth.height:
            msg = f"{name}: answers cover {matched} of {truth.height} cases"
            raise SystemExit(msg)
        rows.append(summarise(name, answers, truth))

    y = (truth["label"] == "cheater").to_numpy()
    print(
        f"{truth.height} cases; the targets themselves rank players at AUC "
        f"{roc_auc_score(y, truth['target_probability']):.3f}, the behaviour score "
        f"alone at {roc_auc_score(y, truth['score']):.3f}\n"
    )
    table = pl.DataFrame(rows)
    with pl.Config(float_precision=3, tbl_width_chars=200, tbl_cols=-1, tbl_rows=-1):
        print(table.transpose(include_header=True, column_names="judge"))


if __name__ == "__main__":
    main()
