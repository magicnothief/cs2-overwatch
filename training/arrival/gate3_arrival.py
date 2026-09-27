"""Candidate arrival definitions against gate 3, with the choice rule written down.

Gate 3, docs/specs/2026-09-26-triggerbot-and-snap-count.md section 4: "if more
than 1% of pros are past its 95% line, the measurement is redesigned before
shipping". The line comes from CS2CD's clean players, the rate is read on the pro
sample, and this prints both denominators the eval argued over — of every pro,
and of the pros the measurement is shown to. A candidate has to pass both, or it
is passing by hiding itself from pros rather than by being right.

The selection rule is applied to CS2CD only, before the pro columns are read, so
picking a winner is not threshold-tuning against the gate set:

    S1  no 2-kill verdicts: at every eligible arrival count, crossing the 95%
        line takes at least 3 arrival-tick kills. This is the defect the
        redesign exists to remove, so it is structural and not negotiable.
    S2  signal kept: CS2CD cheaters past the line, at or above v5's 17.78%.
    S3  separation improved: lift (cheater past / clean past) at or above v5's
        4.41.
    S4  worth a prompt line: shown to at least 15% of CS2CD clean players with
        5 or more kills.

Inputs, both built by arrival_lab.py and pro_arrival_lab.py beside this file:

    .venv/bin/python training/arrival/gate3_arrival.py \
        --cs2cd data/processed/arrival_lab.parquet \
        --pro data/processed/pro_arrival_lab
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

TICK_RATE = 64.0
MIN_KILLS = 5
#: v5's numbers on this same data, the bar S2 and S3 are written against.
V5_CHEATER_PAST = 0.1778
V5_LIFT = 4.41
MIN_COVERAGE = 0.15

#: Candidate arrival definitions: how fast the crosshair itself must have been
#: moving, in deg/s, on the tick it came onto the head. `None` is v5 — any
#: arrival counts, including the enemy walking into a held crosshair.
SWEEP_DPS = (None, 10, 15, 20, 25, 30, 40, 50)
SWEEP_MIN = (5, 6, 7, 8, 10, 12)


def lab_files(paths: str | list[str], exclude: list[str] | None = None) -> list[str]:
    """The parquets of one or more lab outputs, minus the demos of other samples.

    A `--per-map 25` sample is not disjoint from a `--per-map 5` sample of another
    seed — both draw from the same 1,988-demo listing — so a confirmation read has
    to subtract the selection sample's demos by name rather than trust a fresh seed.
    """
    dropped = {
        Path(remote).stem
        for path in exclude or []
        for remote in json.loads(Path(path, "listing.json").read_text())["demos"]
    }
    return [
        f
        for path in ([paths] if isinstance(paths, str) else paths)
        for f in (
            sorted(glob.glob(f"{path}/*.parquet")) if Path(path).is_dir() else [path]
        )
        if Path(f).stem not in dropped
    ]


def load(paths: str | list[str], exclude: list[str] | None = None) -> pl.DataFrame:
    """One or more lab parquets or lab output directories, concatenated."""
    files = lab_files(paths, exclude)
    lab = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return lab.with_columns(dps=pl.col("cross_step") * TICK_RATE)


def counts(lab: pl.DataFrame, dps: float | None) -> pl.DataFrame:
    """Per player: kills, arrivals under this definition, and arrival-tick shots."""
    group = ["match_id", "player_id"] + (["label"] if "label" in lab.columns else [])
    kills = (
        lab.select(*group, "window_uid")
        .unique()
        .group_by(group)
        .agg(n_kills=pl.len())
        .filter(pl.col("n_kills") >= MIN_KILLS)
    )
    kept = lab.filter(pl.col("arrived").is_not_null())
    if dps is not None:
        kept = kept.filter(pl.col("dps") >= dps)
    per_window = kept.group_by("window_uid", *group).agg(
        shot=pl.col("arrived").max() == pl.col("opening").max()
    )
    per_player = per_window.group_by(group).agg(ak=pl.len(), ash=pl.col("shot").sum())
    return kills.join(per_player, on=group, how="left").fill_null(0)


def crossing_count(line: float, min_arrival: int, cap: int = 60) -> int:
    """The fewest arrival-tick kills that put any eligible player past the line."""
    return min(
        (
            shots
            for n in range(min_arrival, cap + 1)
            for shots in range(n + 1)
            if shots / n > line
        ),
        default=cap,
    )


def evaluate(cs2cd: pl.DataFrame, pro: pl.DataFrame) -> list[dict]:
    rows = []
    for dps in SWEEP_DPS:
        c, p = counts(cs2cd, dps), counts(pro, dps)
        label = np.array(c["label"].to_list())
        clean, cheat = label == "clean", label == "cheater"
        cn, cx = c["ak"].to_numpy().astype(float), c["ash"].to_numpy().astype(float)
        pn, px = p["ak"].to_numpy().astype(float), p["ash"].to_numpy().astype(float)
        n_pro = p.height
        for k in SWEEP_MIN:
            cv = np.where(cn >= k, np.divide(cx, np.where(cn > 0, cn, 1)), np.nan)
            pv = np.where(pn >= k, np.divide(px, np.where(pn > 0, pn, 1)), np.nan)
            eligible_clean, eligible_cheat = (
                clean & ~np.isnan(cv),
                cheat & ~np.isnan(cv),
            )
            if eligible_clean.sum() < 30:
                continue
            line95 = float(np.quantile(cv[eligible_clean], 0.95))
            line99 = float(np.quantile(cv[eligible_clean], 0.99))
            clean_past = float((cv[eligible_clean] > line95).mean())
            cheat_past = float((cv[eligible_cheat] > line95).mean())
            coverage = float(eligible_clean.sum()) / clean.sum()
            crossing = crossing_count(line95, k)
            shown = ~np.isnan(pv)
            rows.append(
                {
                    "dps": "v5_any" if dps is None else f">={dps}",
                    "min_arrival": k,
                    "line95": round(line95, 4),
                    "line99": round(line99, 4),
                    "crossing_count": crossing,
                    "coverage": round(coverage, 4),
                    "clean_past": round(clean_past, 4),
                    "cheat_past": round(cheat_past, 4),
                    "lift": round(cheat_past / clean_past, 2) if clean_past else None,
                    "s1": bool(crossing >= 3),
                    "s2": bool(cheat_past >= V5_CHEATER_PAST),
                    "s3": bool(
                        (cheat_past / clean_past if clean_past else 0) >= V5_LIFT
                    ),
                    "s4": bool(coverage >= MIN_COVERAGE),
                    "pro_n": n_pro,
                    "pro_shown": int(shown.sum()),
                    "pro_past95": int((pv[shown] > line95).sum()),
                    "pro_past99": int((pv[shown] > line99).sum()),
                    "pro_of_all": round(float((pv[shown] > line95).sum()) / n_pro, 4),
                    "pro_of_shown": round(
                        float((pv[shown] > line95).sum()) / max(int(shown.sum()), 1), 4
                    ),
                    "pro_pooled": round(
                        float(px[shown].sum() / max(pn[shown].sum(), 1)), 4
                    ),
                    "clean_pooled": round(
                        float(cx[eligible_clean].sum() / cn[eligible_clean].sum()), 4
                    ),
                    "cheat_pooled": round(
                        float(cx[eligible_cheat].sum() / cn[eligible_cheat].sum()), 4
                    ),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cs2cd", default="data/processed/arrival_lab.parquet")
    ap.add_argument("--pro", nargs="+", default=["data/processed/pro_arrival_lab"])
    ap.add_argument(
        "--pro-exclude",
        nargs="*",
        default=[],
        help="pro samples whose demos to drop, for a strictly disjoint read",
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    pro_files = lab_files(args.pro, args.pro_exclude)
    print(f"pro sample: {len(pro_files)} demos")
    rows = evaluate(load(args.cs2cd), load(pro_files))
    head = [
        "dps",
        "min_arrival",
        "line95",
        "crossing_count",
        "coverage",
        "clean_past",
        "cheat_past",
        "lift",
        "s1",
        "s2",
        "s3",
        "s4",
        "pro_shown",
        "pro_past95",
        "pro_of_all",
        "pro_of_shown",
        "pro_past99",
        "pro_pooled",
        "clean_pooled",
        "cheat_pooled",
    ]
    print("\t".join(head))
    for r in rows:
        print("\t".join(str(r[h]) for h in head))

    passing = [r for r in rows if r["s1"] and r["s2"] and r["s3"] and r["s4"]]
    print(f"\n{len(passing)} of {len(rows)} candidates pass S1-S4 on CS2CD alone:")
    for r in sorted(passing, key=lambda r: -r["lift"]):
        gate3 = (
            "PASS" if r["pro_of_all"] <= 0.01 and r["pro_of_shown"] <= 0.01 else "FAIL"
        )
        print(
            f"  crosshair {r['dps']} deg/s, MIN {r['min_arrival']}: line95 {r['line95']}, "
            f"lift {r['lift']}, shown to {r['coverage']:.1%} of clean, "
            f"pros past {r['pro_past95']}/{r['pro_n']} ({r['pro_of_all']:.4%} of all, "
            f"{r['pro_of_shown']:.4%} of {r['pro_shown']} shown) — gate 3 {gate3}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
