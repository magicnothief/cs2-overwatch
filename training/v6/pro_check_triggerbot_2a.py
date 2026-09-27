"""Judge v6 spec, section 2a: triggerbot timing, second check, on a fresh sample.

The criterion was changed after the first read (docs/evals/
2026-09-27-triggerbot-pros-spike.md) and written into the spec before this
sample was fetched:

    at most as many pros past the 95% line as clean CS2CD players, of those
    shown and of all; and at most 1% of shown pros past the 99% line

The sample is `pro_arrival_lab.py --per-map 25 --seed 3`, minus every demo of the
three samples read before.

    .venv/bin/python training/v6/pro_check_triggerbot_2a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arrival"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate3_arrival import counts, lab_files, load
from pro_check_measurements import MIN_ARRIVAL_KILLS, PROCESSED, arrival

from overwatch.layers.l3_behavior.shots import ARRIVAL_SWEEP_DPS

FRESH = PROCESSED / "pro_arrival_lab_seed3"
READ_BEFORE = [
    PROCESSED / "pro_arrival_lab",
    PROCESSED / "pro_arrival_lab_seed1",
    PROCESSED / "pro_arrival_lab_big",
]
STRONG_CEILING = 0.01


def main() -> None:
    files = lab_files(str(FRESH), [str(p) for p in READ_BEFORE])
    print(f"fresh sample: {len(files)} demos after dropping every demo read before")
    cs2cd = load(str(PROCESSED / "arrival_lab.parquet"))
    pros = load(files)
    a = arrival(cs2cd, pros)
    # the clean rate the new criterion compares against, on the same definition
    c = counts(cs2cd, ARRIVAL_SWEEP_DPS).filter(pl.col("ak") >= MIN_ARRIVAL_KILLS)
    clean = c.filter(pl.col("label") == "clean")
    clean_past = float((clean["ash"] / clean["ak"] > a["line95"]).mean())
    of_shown_99 = a["past99"] / max(a["shown"], 1)
    passes = (
        a["of_shown"] <= clean_past
        and a["of_all"] <= clean_past
        and of_shown_99 <= STRONG_CEILING
    )
    print(
        f"lines {a['line95']:.4f} / {a['line99']:.4f}; clean CS2CD past the 95% line "
        f"{clean_past:.2%}\n"
        f"pros {a['pros']}, shown {a['shown']}: past 95% {a['past95']} "
        f"({a['of_shown']:.2%} of shown, {a['of_all']:.2%} of all), "
        f"past 99% {a['past99']} ({of_shown_99:.2%} of shown)\n"
        f"-> {'pass' if passes else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
