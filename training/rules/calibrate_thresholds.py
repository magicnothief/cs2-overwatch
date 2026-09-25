"""Derive Layer 1's thresholds from the clean population.

Run:  uv run python training/rules/calibrate_thresholds.py --matches 120

A rule is only worth having if clean players never trip it, so every line comes
from what clean players actually produce.

**The clean set is not clean.** CS2CD's own card measures ~97% of players in
`no_cheater_present` as clean, and the first calibration run found a player
holding 10,053 deg/s for a quarter of a second — a spinbot sitting in the clean
folder. Calibrating on the maximum therefore calibrates on a cheater, which is
why every line here comes from a high quantile over *players* and the script
reports how many clean players sit above it. Those are contamination, and they
are the rule working, not failing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from overwatch.aim import MAX_MEASURABLE_SPEED, add_aim_features
from overwatch.dataset import CLEAN_DIR
from overwatch.layers.l1_blatant.thresholds import Thresholds
from overwatch.parsing import load_cs2cd

ROOT = Path(__file__).resolve().parents[2]

#: (quantile over clean players, headroom multiplier) per measurement.
#:
#: They differ because the tails differ. Sustained spin is bimodal: clean players
#: stop around 1,300 deg/s and the next values jump past 7,000, so anything above
#: p99 is contamination and the line is set from p99 with wide headroom. A fast
#: legitimate flick really can land at 1,800 deg/s on the kill tick, so that line
#: is set from the far tail instead, where the values are still human.
CALIBRATION: dict[str, tuple[float, float]] = {
    "sustained_speed": (0.99, 2.0),
    "max_kill_tick_speed": (0.999, 1.5),
    "longest_flip_run": (0.999, 1.0),
}
#: Window for "sustained": a quarter of a second.
SUSTAIN_TICKS = 16


def player_extremes(parquet: Path) -> pl.DataFrame:
    """One row per player: their most extreme value for each rule's measurement."""
    match = load_cs2cd(parquet)
    ticks = add_aim_features(match.ticks)

    sustained = (
        ticks.sort(["player_id", "tick"])
        .with_columns(
            sustained=pl.col("yaw_speed")
            .fill_null(0)
            .rolling_median(window_size=SUSTAIN_TICKS)
            .over("player_id"),
            flip=(
                (pl.col("d_yaw").abs() > 90)
                & (
                    pl.col("d_yaw").sign()
                    != pl.col("d_yaw").sign().shift(1).over("player_id")
                )
            ).fill_null(value=False),
        )
        .with_columns(
            flip_run=(pl.col("flip") != pl.col("flip").shift(1).over("player_id"))
            .cum_sum()
            .over("player_id")
        )
    )

    flips = (
        sustained.filter(pl.col("flip"))
        .group_by("player_id", "flip_run")
        .len()
        .group_by("player_id")
        .agg(longest_flip_run=pl.col("len").max())
    )

    at_kill = (
        match.deaths.join(
            ticks.select("player_id", "tick", "yaw_speed"),
            left_on=["attacker_id", "tick"],
            right_on=["player_id", "tick"],
            how="inner",
        )
        .group_by("attacker_id")
        .agg(max_kill_tick_speed=pl.col("yaw_speed").max())
    )

    return (
        sustained.group_by("player_id")
        .agg(
            sustained_speed=pl.col("sustained").max(),
            single_tick_speed=pl.col("yaw_speed").max(),
            max_abs_pitch=pl.col("pitch").abs().max(),
        )
        .join(flips, on="player_id", how="left")
        .join(at_kill, left_on="player_id", right_on="attacker_id", how="left")
        .with_columns(
            match_id=pl.lit(parquet.stem),
            longest_flip_run=pl.col("longest_flip_run").fill_null(0),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "cs2cd")
    parser.add_argument("--matches", type=int, default=120)
    args = parser.parse_args()

    paths = sorted((args.root / CLEAN_DIR).glob("*.parquet"))[: args.matches]
    print(f"calibrating on {len(paths)} clean matches")

    started = time.time()
    rows = []
    for n, path in enumerate(paths, start=1):
        rows.append(player_extremes(path))
        if n % 30 == 0:
            print(f"  {n}/{len(paths)} matches ({time.time() - started:.0f}s)")
    players = pl.concat(rows)

    def line(column: str) -> float:
        quantile, headroom = CALIBRATION[column]
        return float(players[column].quantile(quantile) or 0) * headroom

    spin = min(round(line("sustained_speed"), -1), MAX_MEASURABLE_SPEED * 0.9)
    snap = round(line("max_kill_tick_speed"), -1)
    flip_run = max(int(line("longest_flip_run")) + 2, 6)

    thresholds = Thresholds(
        spin_speed_dps=spin,
        spin_min_ticks=SUSTAIN_TICKS,
        alias_min_step_deg=90.0,
        alias_min_ticks=flip_run,
        snap_speed_dps=snap,
        max_pitch_deg=89.0,
        clean_speed_p9999=round(
            float(players["sustained_speed"].quantile(0.99) or 0), 1
        ),
        clean_snap_p9999=round(
            float(players["max_kill_tick_speed"].quantile(0.999) or 0), 1
        ),
        clean_pitch_max=round(float(players["max_abs_pitch"].max() or 0), 2),
        calibrated_on_matches=len(paths),
    )
    path = thresholds.save()

    with pl.Config(float_precision=0, tbl_width_chars=120):
        print("\nclean players, per-measurement distribution:")
        print(
            players.select(
                "sustained_speed",
                "single_tick_speed",
                "max_kill_tick_speed",
                "longest_flip_run",
            ).describe(percentiles=[0.5, 0.99, 0.999])
        )

    above = {
        "sustained spin": int((players["sustained_speed"] > spin).sum()),
        "kill-tick snap": int((players["max_kill_tick_speed"] > snap).sum()),
        "aliased spin": int((players["longest_flip_run"] >= flip_run).sum()),
    }
    print(f"\n{players.height} clean players; above each line (i.e. contamination):")
    for name, count in above.items():
        print(f"  {name:15s} {count:4d}  ({count / players.height:.2%})")
    print(f"\nwrote {path}\n{Path(path).read_text()}")


if __name__ == "__main__":
    main()
