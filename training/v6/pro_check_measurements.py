"""Judge v6 spec, section 2: do pros trip the redesigned measurements?

docs/specs/2026-09-27-judge-v6-evidence.md, fixed before this ran:

    arrival_shot_share   at most 1% of pros past its 95% line, of those it is
                         shown for and of all (crosshair sweep >= 50 deg/s,
                         at least 5 eligible kills)
    snap_kills           at most 1% of pros past its notable line (non-sniper
                         kills over 175 deg/s on the kill tick; the line is the
                         clean 95% quantile, floored at 2, so 3 kills are notable)

Lines come from clean CS2CD players with 5 or more kills, as the judge's do. The
pros are the 175-match `pro_arrival_lab_big` sample minus every demo of the two
samples the definitions were chosen on (pro_arrival_lab, pro_arrival_lab_seed1).

    .venv/bin/python training/v6/pro_check_measurements.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arrival"))
from gate3_arrival import counts, lab_files, load

from overwatch.layers.l3_behavior.player_features import (
    SNAP_THRESHOLD_DPS,
    SNIPERS,
)
from overwatch.layers.l3_behavior.shots import ARRIVAL_SWEEP_DPS
from overwatch.layers.l4_judge.targets import LINE_FLOORS

PROCESSED = Path("data/processed")
PROS = PROCESSED / "pro_arrival_lab_big"
CHOSEN_ON = [PROCESSED / "pro_arrival_lab", PROCESSED / "pro_arrival_lab_seed1"]
MIN_KILLS = 5
MIN_ARRIVAL_KILLS = 5
CEILING = 0.01


def arrival(cs2cd: pl.DataFrame, pros: pl.DataFrame) -> dict:
    c, p = counts(cs2cd, ARRIVAL_SWEEP_DPS), counts(pros, ARRIVAL_SWEEP_DPS)
    share = (pl.col("ash") / pl.col("ak")).alias("share")
    c = c.with_columns(share).filter(pl.col("ak") >= MIN_ARRIVAL_KILLS)
    shown = p.with_columns(share).filter(pl.col("ak") >= MIN_ARRIVAL_KILLS)
    clean = c.filter(pl.col("label") == "clean")["share"]
    line95 = float(clean.quantile(0.95, "linear"))
    line99 = float(clean.quantile(0.99, "linear"))
    past = int((shown["share"] > line95).sum())
    return {
        "line95": line95,
        "line99": line99,
        "pros": p.height,
        "shown": shown.height,
        "past95": past,
        "past99": int((shown["share"] > line99).sum()),
        "of_shown": past / max(shown.height, 1),
        "of_all": past / max(p.height, 1),
    }


def snap(pros: pl.DataFrame) -> dict:
    fast = (pl.col("speed_at_kill") > SNAP_THRESHOLD_DPS) & ~pl.col("weapon").is_in(
        SNIPERS
    ).fill_null(False)
    kills = (
        pl.read_parquet(PROCESSED / "window_features.parquet")
        .select("window_uid", "match_id", "player_id", "label", "speed_at_kill")
        .join(
            pl.read_parquet(
                PROCESSED / "windows.parquet", columns=["window_uid", "weapon"]
            ),
            on="window_uid",
            how="left",
        )
    )
    per = (
        kills.group_by("match_id", "player_id", "label")
        .agg(n=pl.len(), snap_kills=fast.sum())
        .filter(pl.col("n") >= MIN_KILLS)
    )
    clean = per.filter(pl.col("label") == "clean")["snap_kills"]
    floor_notable, floor_strong = LINE_FLOORS["snap_kills"]
    notable = max(float(clean.quantile(0.95, "linear")), floor_notable)
    strong = max(float(clean.quantile(0.99, "linear")), floor_strong)
    pro = (
        pros.unique("window_uid")
        .group_by("match_id", "player_id")
        .agg(n=pl.len(), snap_kills=fast.sum())
        .filter(pl.col("n") >= MIN_KILLS)
    )
    cheat = per.filter(pl.col("label") == "cheater")["snap_kills"]
    return {
        "notable": notable,
        "strong": strong,
        "clean_past": float((clean > notable).mean()),
        "cheater_past": float((cheat > notable).mean()),
        "pros": pro.height,
        "past_notable": int((pro["snap_kills"] > notable).sum()),
        "past_strong": int((pro["snap_kills"] > strong).sum()),
        "of_all": float((pro["snap_kills"] > notable).mean()),
    }


def main() -> None:
    files = lab_files(str(PROS), [str(p) for p in CHOSEN_ON])
    print(f"pro sample: {len(files)} demos after dropping the selection samples'")
    pros = load(files)
    a = arrival(load(str(PROCESSED / "arrival_lab.parquet")), pros)
    verdict_a = a["of_shown"] <= CEILING and a["of_all"] <= CEILING
    print(
        f"arrival_shot_share (sweep >= {ARRIVAL_SWEEP_DPS:g} deg/s, >= {MIN_ARRIVAL_KILLS} "
        f"eligible kills): lines {a['line95']:.4f} / {a['line99']:.4f}\n"
        f"  pros {a['pros']}, shown {a['shown']}, past 95% {a['past95']} "
        f"({a['of_shown']:.2%} of shown, {a['of_all']:.2%} of all), past 99% {a['past99']}"
        f"\n  -> {'pass' if verdict_a else 'FAIL'}"
    )
    s = snap(pros)
    verdict_s = s["of_all"] <= CEILING
    print(
        f"snap_kills (non-sniper, > {SNAP_THRESHOLD_DPS:g} deg/s): notable past "
        f"{s['notable']:g}, strong past {s['strong']:g}; CS2CD clean past notable "
        f"{s['clean_past']:.2%}, cheaters {s['cheater_past']:.2%}\n"
        f"  pros {s['pros']}, past notable {s['past_notable']} ({s['of_all']:.2%}), "
        f"past strong {s['past_strong']}\n  -> {'pass' if verdict_s else 'FAIL'}"
    )
    upper = 1 - (0.05) ** (1 / max(a["shown"], 1))
    print(
        f"(0 of {a['shown']} would bound the arrival rate at {upper:.2%}, 95% one-sided)"
    )
    np.seterr(all="ignore")


if __name__ == "__main__":
    main()
