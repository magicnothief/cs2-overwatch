"""Candidate `snap_kills` definitions against gate 3, with the choice rule written down.

`docs/evals/2026-09-26-arrival-redesign.md` section 7 measured two defects in the
shipped `snap_kills` and redesigned neither:

  * its lines are 1.0 (notable) and 2.0 (strong) over clean CS2CD players, so two
    kills over 200 deg/s make a player notable and three make the evidence
    decisive — the defect the arrival redesign exists to remove;
  * it is sniper-confounded. For an AWP main, 3 such kills is a 6.93%-of-clean
    event, not the 0.86% the aggregate 99% line claims, and `snap_kills` reaches
    `DECISIVE` without target rule 3's sniping discount (ADR 0008).

This sweeps the two levers that can fix both — how fast the kill-tick turn must
be, and whether sniper kills count at all — and re-derives the lines from clean
CS2CD players as the spec requires. No judge, no GPU.

    PYTHONPATH=src .venv/bin/python training/arrival/gate3_snap.py \
        --pro data/processed/pro_arrival_lab data/processed/pro_arrival_lab_seed1

The selection rule is applied to CS2CD alone, before the pro columns are read, so
picking a winner is not threshold-tuning against the gate set:

    N1  no 2-kill verdicts: crossing the 99% line takes at least 4 qualifying
        kills, and crossing the 95% line at least 3. Structural, not negotiable.
    N2  signal kept: CS2CD cheaters past the 95% line at or above 10.00%. This is
        deliberately below the shipped 13.13% — the false-positive asymmetry is
        worth some recall — and it is pre-registered at that value here.
    N3  separation improved: aggregate lift at the 95% line at or above the
        shipped 8.14x.
    N4  sniper robustness (ADR 0008): among AWP mains (sniper_share >= 0.5), lift
        at the 95% line at or above 3.00x, against the shipped 1.73x.

Gate 3 is then read on the pro sample: at most 1% of pros past the 95% line, on
both denominators (of every pro, and of the pros the measurement is shown to).
A count is shown to every player with 5 or more kills, so for the count variants
the two denominators are the same number.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

MIN_KILLS = 5
#: `player_features.SNIPERS`, kept literal so this tool does not depend on import
#: order; asserted against the module in `load_cs2cd`.
SNIPERS = ("awp", "ssg08", "scar20", "g3sg1")
#: `targets.SNIPER_HEAVY` — the band target rule 3 already treats as a sniper.
SNIPER_HEAVY = 0.5
#: The shipped measurement's own numbers on this data, the bar N2-N4 are written
#: against: 200 deg/s, all weapons, count, line95 1.0, line99 2.0.
SHIPPED_CHEAT_PAST = 0.1313
SHIPPED_LIFT = 8.14
SHIPPED_AWP_LIFT = 1.73
N2_FLOOR = 0.10
N4_FLOOR = 3.00

SWEEP_DPS = (100.0, 120.0, 150.0, 175.0, 200.0, 250.0, 300.0, 400.0, 500.0)
#: Whether a kill taken with a sniper rifle can count towards the statistic.
SWEEP_SCOPE = ("all", "nonsniper")
#: `count` is the shipped shape. `share` divides by the player's kills, which are
#: never fewer than MIN_KILLS, and is gated to players with at least this many.
SWEEP_STAT = (("count", 0), ("share", 10), ("share", 14))
#: Where the line comes from. `pooled` is the shipped rule — one quantile over all
#: clean CS2CD players. `per_band` takes the quantile inside the player's own
#: sniper band, so an AWP main is compared against clean AWP mains (ADR 0008).
SWEEP_LINE_MODE = ("pooled", "per_band")
#: `floor` takes the more conservative of the quantile line and a kill floor: a
#: count must reach 3 to be notable and 4 to be strong. The quantile is then a
#: ceiling on false positives rather than a target. Count statistics only.
FLOOR95, FLOOR99 = 2.0, 3.0
SWEEP_FLOOR = (False, True)


def per_player(kills: pl.DataFrame, dps: float, scope: str) -> pl.DataFrame:
    """Per player: kills, sniper share, and qualifying kills under this definition.

    `kills` is one row per kill with match_id, player_id, label, weapon and
    speed_at_kill.
    """
    counted = pl.col("speed_at_kill") > dps
    if scope == "nonsniper":
        counted = counted & ~pl.col("weapon").is_in(SNIPERS)
    return (
        kills.group_by("match_id", "player_id", "label")
        .agg(
            n_kills=pl.len(),
            sniper_share=pl.col("weapon").is_in(SNIPERS).mean(),
            snap=counted.sum(),
        )
        .filter(pl.col("n_kills") >= MIN_KILLS)
    )


def statistic(players: pl.DataFrame, stat: str, min_kills: int) -> np.ndarray:
    """The candidate measurement per player, NaN where it is not shown."""
    snap = players["snap"].to_numpy().astype(float)
    n = players["n_kills"].to_numpy().astype(float)
    if stat == "count":
        return snap
    return np.where(n >= min_kills, snap / n, np.nan)


def crossing_count(line: float, stat: str, min_kills: int, cap: int = 60) -> int:
    """The fewest qualifying kills that put any shown player past the line."""
    if stat == "count":
        return int(np.floor(line)) + 1
    return min(
        (
            snap
            for n in range(max(min_kills, MIN_KILLS), cap + 1)
            for snap in range(n + 1)
            if snap / n > line
        ),
        default=cap,
    )


def bands(sniper_share: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rifler": sniper_share < 0.2,
        "mixed": (sniper_share >= 0.2) & (sniper_share < SNIPER_HEAVY),
        "awp_main": sniper_share >= SNIPER_HEAVY,
    }


def lines_for(
    cv: np.ndarray,
    shown_clean: np.ndarray,
    band_masks: dict[str, np.ndarray],
    line_mode: str,
    floor: bool,
    stat: str,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Per-player notable/strong lines, and the line each band was given.

    Returns an array of shape (n_players, 2) holding each player's 95% and 99%
    line, so a per-band line can be compared player by player.
    """
    per_band: dict[str, tuple[float, float]] = {}
    out = np.full((cv.shape[0], 2), np.nan)
    if line_mode == "pooled":
        pairs = {"all": np.ones(cv.shape[0], dtype=bool)}
    else:
        pairs = band_masks
    for name, mask in pairs.items():
        reference = cv[mask & shown_clean]
        if reference.size < 30:
            continue
        line95 = float(np.quantile(reference, 0.95))
        line99 = float(np.quantile(reference, 0.99))
        if floor and stat == "count":
            line95, line99 = max(line95, FLOOR95), max(line99, FLOOR99)
        per_band[name] = (line95, line99)
        out[mask] = (line95, line99)
    return out, per_band


def evaluate(cs2cd: pl.DataFrame, pro: pl.DataFrame) -> list[dict]:
    rows = []
    for dps in SWEEP_DPS:
        for scope in SWEEP_SCOPE:
            c, p = per_player(cs2cd, dps, scope), per_player(pro, dps, scope)
            label = np.array(c["label"].to_list())
            clean, cheat = label == "clean", label == "cheater"
            c_bands = bands(c["sniper_share"].to_numpy().astype(float))
            p_bands = bands(p["sniper_share"].to_numpy().astype(float))
            for stat, min_kills in SWEEP_STAT:
                cv = statistic(c, stat, min_kills)
                pv = statistic(p, stat, min_kills)
                shown_clean, shown_cheat = clean & ~np.isnan(cv), cheat & ~np.isnan(cv)
                shown_pro = ~np.isnan(pv)
                if shown_clean.sum() < 30:
                    continue
                for line_mode in SWEEP_LINE_MODE:
                    for floor in SWEEP_FLOOR:
                        if floor and stat != "count":
                            continue  # a kill floor is only defined for a count
                        c_lines, per_band = lines_for(
                            cv, shown_clean, c_bands, line_mode, floor, stat
                        )
                        if line_mode == "per_band" and len(per_band) < len(c_bands):
                            continue  # a band without 30 clean players has no line
                        p_lines = np.full((pv.shape[0], 2), np.nan)
                        if line_mode == "pooled":
                            p_lines[:] = per_band["all"]
                        else:
                            for name, mask in p_bands.items():
                                p_lines[mask] = per_band[name]
                        rows.append(
                            measure(
                                dps,
                                scope,
                                stat,
                                min_kills,
                                line_mode,
                                floor,
                                cv,
                                pv,
                                c_lines,
                                p_lines,
                                per_band,
                                clean,
                                shown_clean,
                                shown_cheat,
                                shown_pro,
                                c_bands,
                                p.height,
                            )
                        )
    return rows


def measure(
    dps,
    scope,
    stat,
    min_kills,
    line_mode,
    floor,
    cv,
    pv,
    c_lines,
    p_lines,
    per_band,
    clean,
    shown_clean,
    shown_cheat,
    shown_pro,
    c_bands,
    n_pro,
) -> dict:
    """One candidate's CS2CD rates, subgroup rates and pro rates."""

    def past(mask: np.ndarray, which: int, values=None, lines=None) -> float:
        values = cv if values is None else values
        lines = c_lines if lines is None else lines
        return (
            float((values[mask] > lines[mask, which]).mean())
            if mask.sum()
            else float("nan")
        )

    clean_past95, cheat_past95 = past(shown_clean, 0), past(shown_cheat, 0)
    clean_past99, cheat_past99 = past(shown_clean, 1), past(shown_cheat, 1)
    lift95 = cheat_past95 / clean_past95 if clean_past95 else None
    lift99 = cheat_past99 / clean_past99 if clean_past99 else None
    sub = {}
    for name, mask in c_bands.items():
        sc, sx = mask & shown_clean, mask & shown_cheat
        if sc.sum() < 30 or sx.sum() < 30:
            sub[name] = None
            continue
        band_clean, band_cheat = past(sc, 0), past(sx, 0)
        sub[name] = {
            "n_clean": int(sc.sum()),
            "n_cheat": int(sx.sum()),
            "line95": round(per_band.get(name, per_band.get("all"))[0], 4),
            "clean_past": round(band_clean, 4),
            "cheat_past": round(band_cheat, 4),
            "lift": round(band_cheat / band_clean, 2) if band_clean else None,
        }
    awp_lift = (sub.get("awp_main") or {}).get("lift")
    crossing95 = min(
        crossing_count(line[0], stat, min_kills) for line in per_band.values()
    )
    crossing99 = min(
        crossing_count(line[1], stat, min_kills) for line in per_band.values()
    )
    pro_past95 = int((pv[shown_pro] > p_lines[shown_pro, 0]).sum())
    pro_past99 = int((pv[shown_pro] > p_lines[shown_pro, 1]).sum())
    return {
        "dps": dps,
        "scope": scope,
        "stat": stat if stat == "count" else f"share/min{min_kills}",
        "line_mode": line_mode + ("+floor" if floor else ""),
        "line95": "/".join(f"{v[0]:g}" for v in per_band.values()),
        "line99": "/".join(f"{v[1]:g}" for v in per_band.values()),
        "crossing95": crossing95,
        "crossing99": crossing99,
        "coverage": round(float(shown_clean.sum()) / clean.sum(), 4),
        "clean_past95": round(clean_past95, 4),
        "cheat_past95": round(cheat_past95, 4),
        "lift95": round(lift95, 2) if lift95 else None,
        "clean_past99": round(clean_past99, 4),
        "cheat_past99": round(cheat_past99, 4),
        "lift99": round(lift99, 2) if lift99 else None,
        "awp_lift95": awp_lift,
        "n1": bool(crossing99 >= 4 and crossing95 >= 3),
        "n2": bool(cheat_past95 >= N2_FLOOR),
        "n3": bool((lift95 or 0) >= SHIPPED_LIFT),
        "n4": bool((awp_lift or 0) >= N4_FLOOR),
        "subgroups": sub,
        "pro_n": n_pro,
        "pro_shown": int(shown_pro.sum()),
        "pro_past95": pro_past95,
        "pro_past99": pro_past99,
        "pro_of_all": round(pro_past95 / max(n_pro, 1), 4),
        "pro_of_shown": round(pro_past95 / max(int(shown_pro.sum()), 1), 4),
    }


def load_cs2cd(window_features: str, windows: str) -> pl.DataFrame:
    """One row per CS2CD kill: label, speed_at_kill and the weapon it was taken with."""
    from overwatch.layers.l3_behavior.player_features import SNIPERS as shipped

    assert tuple(shipped) == SNIPERS, f"SNIPERS drifted: {shipped}"
    feats = pl.read_parquet(
        window_features,
        columns=["window_uid", "match_id", "player_id", "label", "speed_at_kill"],
    )
    weapons = pl.read_parquet(windows, columns=["window_uid", "weapon"])
    return feats.join(weapons, on="window_uid", how="left")


def load_pro(paths: list[str], exclude: list[str] | None = None) -> pl.DataFrame:
    """One row per pro kill, over one or more `pro_arrival_lab.py` output directories.

    The lab writes one row per (kill, arrival), so kills are de-duplicated here.
    `label` is a constant: every player in the sample is a pro, treated as clean.
    `exclude` drops the demos of other samples by name: a `--per-map 25` sample is
    not disjoint from a `--per-map 5` sample of another seed, because both draw
    from the same 1,988-demo listing.
    """
    dropped = {
        Path(remote).stem
        for path in exclude or []
        for remote in json.loads(Path(path, "listing.json").read_text())["demos"]
    }
    files = [
        f
        for p in paths
        for f in sorted(glob.glob(f"{p}/*.parquet"))
        if Path(f).stem not in dropped
    ]
    print(f"pro sample: {len(files)} demos")
    lab = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return (
        lab.select("match_id", "player_id", "window_uid", "weapon", "speed_at_kill")
        .unique(subset=["window_uid"])
        .with_columns(label=pl.lit("pro"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--window-features", default="data/processed/window_features.parquet"
    )
    ap.add_argument("--windows", default="data/processed/windows.parquet")
    ap.add_argument("--pro", nargs="+", default=["data/processed/pro_arrival_lab"])
    ap.add_argument(
        "--pro-exclude",
        nargs="*",
        default=[],
        help="pro samples whose demos to drop, for a strictly disjoint read",
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    cs2cd = load_cs2cd(args.window_features, args.windows)
    pro = load_pro(args.pro, args.pro_exclude)
    print(
        f"CS2CD {cs2cd.height} kills; pro {pro.height} kills over {len(args.pro)} samples"
    )
    rows = evaluate(cs2cd, pro)

    head = [
        "dps",
        "scope",
        "stat",
        "line_mode",
        "line95",
        "line99",
        "crossing95",
        "crossing99",
        "coverage",
        "clean_past95",
        "cheat_past95",
        "lift95",
        "clean_past99",
        "cheat_past99",
        "lift99",
        "awp_lift95",
        "n1",
        "n2",
        "n3",
        "n4",
        "pro_n",
        "pro_shown",
        "pro_past95",
        "pro_past99",
        "pro_of_all",
        "pro_of_shown",
    ]
    print("\t".join(head))
    for r in rows:
        print("\t".join(str(r[h]) for h in head))

    passing = [r for r in rows if r["n1"] and r["n2"] and r["n3"] and r["n4"]]
    print(f"\n{len(passing)} of {len(rows)} candidates pass N1-N4 on CS2CD alone:")
    for r in sorted(passing, key=lambda r: -(r["lift95"] or 0)):
        gate3 = (
            "PASS" if r["pro_of_all"] <= 0.01 and r["pro_of_shown"] <= 0.01 else "FAIL"
        )
        print(
            f"  {r['dps']:.0f} deg/s, {r['scope']}, {r['stat']}, {r['line_mode']}: "
            f"line95 {r['line95']} line99 {r['line99']} "
            f"(crossing {r['crossing95']}/{r['crossing99']} kills), "
            f"clean {r['clean_past95']:.2%} cheater {r['cheat_past95']:.2%} "
            f"lift {r['lift95']}x, AWP-main lift {r['awp_lift95']}x, "
            f"pros past {r['pro_past95']}/{r['pro_n']} ({r['pro_of_all']:.4%} of all, "
            f"{r['pro_of_shown']:.4%} of {r['pro_shown']} shown) — gate 3 {gate3}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
