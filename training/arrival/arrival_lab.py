"""Every crosshair arrival in CS2CD, with what moved to make it, cached once.

`shots.arrival_features` keeps only the last arrival at or before the opening
shot. That is the right thing for the feature and the wrong thing for a redesign:
a variant that ignores some arrivals changes *which* one is last. So this dumps
every arrival at or before the opening shot with the attributes a variant might
filter on, and every candidate becomes an offline group-by.

    PYTHONPATH=src .venv/bin/python training/arrival/arrival_lab.py \
        data/processed/arrival_lab.parquet

Columns, one row per (window_uid, arrival):
    arrived, opening   tick offsets; arrival_shot is arrived == opening
    head_deg           the head's angular radius at that distance (the on-head bar)
    cross_step         the crosshair's own angular step over the arrival tick,
                       yaw scaled by cos(pitch)
    closure            target_angle[t-1] - target_angle[t] over the arrival tick:
                       how much the gap shrank, whoever shrank it
    target_distance, weapon-free; join match_id/player_id/label from window_features
"""

from __future__ import annotations

import math
import sys

import polars as pl

from overwatch.layers.l3_behavior.shots import BURST_GAP, HEAD_RADIUS, MIN_DISTANCE
from overwatch.paths import DATA

TICKS = DATA / "processed" / "window_ticks.parquet"
WINDOWS = DATA / "processed" / "window_features.parquet"


def arrival_lab(ticks: pl.LazyFrame) -> pl.DataFrame:
    t = (
        ticks.select(
            "window_uid",
            "tick_offset",
            "target_angle",
            "target_distance",
            "shot",
            "d_yaw",
            "d_pitch",
            "pitch",
        )
        .sort("window_uid", "tick_offset")
        .with_columns(
            head_deg=pl.lit(math.degrees(1.0))
            * HEAD_RADIUS
            / pl.col("target_distance").clip(lower_bound=MIN_DISTANCE)
        )
        .with_columns(on_head=pl.col("target_angle") < pl.col("head_deg"))
        .with_columns(
            arrives=pl.col("on_head").fill_null(False)
            & ~pl.col("on_head").shift(1).over("window_uid").fill_null(True),
            cross_step=(
                (pl.col("d_yaw") * pl.col("pitch").radians().cos()) ** 2
                + pl.col("d_pitch") ** 2
            )
            ** 0.5,
            closure=pl.col("target_angle").shift(1).over("window_uid")
            - pl.col("target_angle"),
        )
        .collect()
    )
    shots = (
        t.filter(pl.col("shot") & (pl.col("tick_offset") <= 0))
        .select("window_uid", "tick_offset")
        .with_columns(gap=pl.col("tick_offset").diff().over("window_uid"))
        .with_columns(
            burst=(pl.col("gap").is_null() | (pl.col("gap") > BURST_GAP))
            .cum_sum()
            .over("window_uid")
        )
    )
    opening = (
        shots.filter(pl.col("burst") == pl.col("burst").max().over("window_uid"))
        .group_by("window_uid")
        .agg(opening=pl.col("tick_offset").min())
    )
    return (
        t.filter(pl.col("arrives"))
        .select(
            "window_uid",
            arrived=pl.col("tick_offset"),
            head_deg=pl.col("head_deg"),
            cross_step=pl.col("cross_step"),
            closure=pl.col("closure"),
            target_distance=pl.col("target_distance"),
        )
        .join(opening, on="window_uid", how="inner")
        .filter(pl.col("arrived") <= pl.col("opening"))
    )


def main() -> None:
    out = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(DATA / "processed" / "arrival_lab.parquet")
    )
    lab = arrival_lab(pl.scan_parquet(TICKS))
    keys = pl.read_parquet(
        WINDOWS,
        columns=["window_uid", "match_id", "player_id", "label", "arrival_shot"],
    )
    lab = keys.join(lab, on="window_uid", how="left")
    lab.write_parquet(out)
    print(f"{lab.height} rows -> {out}")
    # the stored feature must fall out of this table unchanged
    check = (
        lab.group_by("window_uid")
        .agg(
            arrival_shot=pl.col("arrival_shot").first(),
            rebuilt=pl.when(pl.col("arrived").is_null().all())
            .then(None)
            .otherwise(pl.col("arrived").max() == pl.col("opening").max()),
        )
        .select(
            values=(pl.col("arrival_shot") == pl.col("rebuilt")).all(),
            nulls=(
                pl.col("arrival_shot").is_null() == pl.col("rebuilt").is_null()
            ).all(),
        )
    )
    print("rebuilds shots.arrival_features exactly:", check.row(0))


if __name__ == "__main__":
    main()
