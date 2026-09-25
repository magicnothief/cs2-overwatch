"""Per-window aim features: one row of numbers per kill.

Each window covers 128 ticks (2 s) before a kill and 32 ticks (0.5 s) after.
We summarize it in three segments:

    idle    offset [-128, -64]   a second of ordinary play before the engagement
    engage  offset [ -32,   0]   the half second the flick happens in
    after   offset (   0,  32]   what the crosshair does once the target is gone

The features are the things a human reviewer actually looks at: how fast the
flick was, how long before the shot it peaked, whether the aim was still moving
when the shot went off, how straight the path to the target was, and how many
corrections were made on the way.
"""

from __future__ import annotations

import polars as pl

from overwatch.aim import TICK_RATE

#: Segment bounds in ticks relative to the kill (inclusive).
IDLE = (-128, -64)
ENGAGE = (-32, 0)
AFTER = (1, 32)

#: Below this, a yaw change is jitter rather than a deliberate correction.
CORRECTION_MIN_DEG = 0.05

#: Features that need victim geometry; absent when a window table has none.
PERCEPTION_COLUMNS: tuple[str, ...] = (
    "visible_share_engage",
    "aim_through_wall_share",
    "reaction_ms",
    "angle_at_first_visible",
    "angle_at_kill",
    "visible_before_kill",
)

#: Columns produced by build_window_features(), excluding the id/label columns.
FEATURE_COLUMNS: tuple[str, ...] = (
    "peak_speed_engage",
    "t_peak_ms",
    "speed_at_kill",
    "settle_ratio",
    "mean_speed_after",
    "mean_speed_idle",
    "turn_total_engage",
    "turn_net_engage",
    "straightness",
    "corrections",
    "speed_change_std",
    "peak_pitch_speed_engage",
    *PERCEPTION_COLUMNS,
)

#: Aiming this close counts as "on the target" for the wall-tracking check.
ON_TARGET_DEG = 5.0


def _segment(window_ticks: pl.DataFrame, bounds: tuple[int, int]) -> pl.DataFrame:
    lo, hi = bounds
    return window_ticks.filter(pl.col("tick_offset").is_between(lo, hi))


def direction_changes() -> pl.Expr:
    """Expression: how many times the turn reverses direction, ignoring jitter.

    Sub-threshold ticks are dropped first, so a pause or a hand tremor between two
    movements in the same direction does not count as a correction.
    """
    moves = pl.col("d_yaw").filter(pl.col("d_yaw").abs() >= CORRECTION_MIN_DEG)
    return (moves.sign().diff() != 0).sum().fill_null(0).cast(pl.Int32)


def _perception_features(
    window_ticks: pl.DataFrame, engage: pl.DataFrame
) -> pl.DataFrame:
    """Features that need to know where the victim was and whether they were seen.

    reaction_ms is the gap between the victim first becoming visible to this
    attacker and the kill. angle_at_first_visible is how close the crosshair
    already was at that instant: a human has to find the target after it appears,
    so a small angle there means the aim was waiting on someone it could not see.

    Only visibility *before* the kill counts. A victim who first becomes visible
    after dying (the body, a teammate's view) would otherwise produce a negative
    reaction time. When they were never visible beforehand, reaction_ms stays
    null and visible_before_kill is false — which is itself the signal.
    """
    first_visible = (
        window_ticks.filter(pl.col("target_visible") & (pl.col("tick_offset") <= 0))
        .group_by("window_uid")
        .agg(first_visible_offset=pl.col("tick_offset").min())
    )
    angle_then = window_ticks.join(first_visible, on="window_uid", how="inner").filter(
        pl.col("tick_offset") == pl.col("first_visible_offset")
    )

    return (
        engage.group_by("window_uid")
        .agg(
            visible_share_engage=pl.col("target_visible").mean(),
            aim_through_wall_share=(
                ~pl.col("target_visible") & (pl.col("target_angle") < ON_TARGET_DEG)
            ).mean(),
        )
        .join(
            angle_then.select(
                "window_uid",
                angle_at_first_visible=pl.col("target_angle"),
                reaction_ms=-pl.col("first_visible_offset") / TICK_RATE * 1000,
            ),
            on="window_uid",
            how="left",
        )
        .join(
            window_ticks.filter(pl.col("tick_offset") == 0).select(
                "window_uid", angle_at_kill=pl.col("target_angle")
            ),
            on="window_uid",
            how="left",
        )
        .with_columns(
            visible_before_kill=pl.col("window_uid").is_in(first_visible["window_uid"])
        )
    )


def build_window_features(window_ticks: pl.DataFrame) -> pl.DataFrame:
    """Reduce per-tick windows to one feature row per window.

    Args:
        window_ticks: output of kill_windows(), with tick_offset, yaw_speed,
            d_yaw, pitch_speed and (optionally) label/match_id columns.

    Returns:
        One row per window_uid: identifiers, label/match_id when present, and
        FEATURE_COLUMNS.
    """
    engage = _segment(window_ticks, ENGAGE)

    peak = (
        engage.sort(
            ["window_uid", "yaw_speed"], descending=[False, True], nulls_last=True
        )
        .group_by("window_uid", maintain_order=True)
        .first()
        .select(
            "window_uid",
            peak_speed_engage=pl.col("yaw_speed"),
            t_peak_ms=pl.col("t_ms"),
        )
    )

    engage_stats = engage.group_by("window_uid").agg(
        turn_total_engage=pl.col("d_yaw").abs().sum(),
        turn_net_engage=pl.col("d_yaw").sum().abs(),
        speed_change_std=pl.col("yaw_speed").diff().std(),
        peak_pitch_speed_engage=pl.col("pitch_speed").max(),
        corrections=direction_changes(),
    )

    at_kill = window_ticks.filter(pl.col("tick_offset") == 0).select(
        "window_uid", speed_at_kill=pl.col("yaw_speed")
    )
    after = (
        _segment(window_ticks, AFTER)
        .group_by("window_uid")
        .agg(mean_speed_after=pl.col("yaw_speed").mean())
    )
    idle = (
        _segment(window_ticks, IDLE)
        .group_by("window_uid")
        .agg(mean_speed_idle=pl.col("yaw_speed").mean())
    )

    id_columns = [
        c
        for c in ("window_uid", "match_id", "label", "player_id")
        if c in window_ticks.columns
    ]
    base = window_ticks.select(id_columns).unique(subset=["window_uid"])

    frames = [peak, engage_stats, at_kill, after, idle]
    if "target_visible" in window_ticks.columns:
        frames.append(_perception_features(window_ticks, engage))

    out = base
    for frame in frames:
        out = out.join(frame, on="window_uid", how="left")

    out = out.with_columns(
        # 1.0 = went straight at the target; lower = wandered or overshot
        straightness=pl.when(pl.col("turn_total_engage") > 0)
        .then(pl.col("turn_net_engage") / pl.col("turn_total_engage"))
        .otherwise(None),
        # how much of the peak speed was left when the shot went off
        settle_ratio=pl.when(pl.col("peak_speed_engage") > 0)
        .then(pl.col("speed_at_kill") / pl.col("peak_speed_engage"))
        .otherwise(None),
    )
    # perception columns are absent when windows carry no victim geometry
    present = [c for c in FEATURE_COLUMNS if c in out.columns]
    return out.select([*id_columns, *present]).sort("window_uid")
