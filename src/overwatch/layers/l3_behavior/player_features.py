"""Aggregate per-kill windows into one row per player.

Labels are per player, not per kill: a banned account cheated in *some* of its
kills, not all of them. So the model's unit is a player, and the aggregation has
to keep the tails — the single impossible snap matters more than the average
kill, which looks ordinary even for a cheater.

That is why this module mixes three kinds of aggregate:

    shares      how often a behaviour occurs at all (straight flicks, snaps)
    extremes    the worst kill the player produced (max, p90)
    typical     medians, which describe the player's normal play
"""

from __future__ import annotations

import polars as pl

#: A kill-tick turn above this is far outside anything clean play produces
#: (clean p99 is ~139 deg/s over the CS2CD sample).
SNAP_THRESHOLD_DPS = 200.0

#: Straightness this close to 1 means the flick never reversed direction.
STRAIGHT_EPS = 0.999

#: Players with fewer kills than this are too noisy to score.
MIN_KILLS = 5

#: Sniper rifles. Their share is context for a reviewer, not evidence: cheaters in
#: CS2CD snipe far more than clean players, so sniping inflates the perception
#: measurements on its own (ADR 0008).
SNIPERS = ("awp", "ssg08", "scar20", "g3sg1")

#: Aggregates that need victim geometry in the window table.
PERCEPTION_COLUMNS: tuple[str, ...] = (
    "visible_share",
    "wall_aim_share",
    "wall_aim_max",
    "reaction_ms_median",
    "reaction_ms_min",
    "angle_at_first_visible_median",
    "never_visible_share",
)

#: Shown to the judge, never given to a model. A detector that can see weapon mix
#: learns "snipes a lot" as a shortcut for "cheats" (ADR 0008); a reviewer needs
#: it for exactly the opposite reason, to discount what sniping inflates.
CONTEXT_COLUMNS: tuple[str, ...] = ("sniper_share",)

PLAYER_FEATURE_COLUMNS: tuple[str, ...] = (
    "n_kills",
    "straight_share",
    "corrections_median",
    "corrections_mean",
    "snap_max",
    "snap_p90",
    "snap_share",
    "peak_speed_p90",
    "peak_speed_median",
    "settle_ratio_median",
    "t_peak_ms_median",
    "pitch_peak_median",
    "idle_speed_median",
    "zero_motion_share",
    *PERCEPTION_COLUMNS,
)


def zero_motion_share(window_ticks: pl.DataFrame) -> pl.DataFrame:
    """Share of ticks where the crosshair did not move at all, per player.

    A human hand is never perfectly still; a hand off the mouse is.
    """
    return (
        window_ticks.filter(pl.col("yaw_speed").is_not_null())
        .group_by("match_id", "player_id")
        .agg(zero_motion_share=(pl.col("yaw_speed") == 0).mean())
    )


def build_player_features(
    window_features: pl.DataFrame,
    window_ticks: pl.DataFrame | None = None,
    *,
    min_kills: int = MIN_KILLS,
) -> pl.DataFrame:
    """Aggregate window features into one row per (match, player).

    Args:
        window_features: output of build_window_features().
        window_ticks: optional per-tick windows, used for zero_motion_share.
        min_kills: drop players with fewer kills than this.

    Returns:
        One row per player: match_id, player_id, label, PLAYER_FEATURE_COLUMNS,
        and CONTEXT_COLUMNS when the windows carry a weapon.
    """
    perception = {
        # how often the victim was actually visible during the engagement
        "visible_share": pl.col("visible_share_engage").mean(),
        # holding the crosshair on someone this player could not see
        "wall_aim_share": pl.col("aim_through_wall_share").mean(),
        "wall_aim_max": pl.col("aim_through_wall_share").max(),
        # how long after the victim appeared the kill came; the minimum is the
        # fastest reaction the player ever produced
        "reaction_ms_median": pl.col("reaction_ms").median(),
        "reaction_ms_min": pl.col("reaction_ms").min(),
        # how close the crosshair already was the instant the victim appeared
        "angle_at_first_visible_median": pl.col("angle_at_first_visible").median(),
        # kills where the victim was never visible before dying
        "never_visible_share": (~pl.col("visible_before_kill")).mean(),
    }
    context = (
        {"sniper_share": pl.col("weapon").is_in(SNIPERS).mean()}
        if "weapon" in window_features.columns
        else {}
    )
    has_perception = "visible_share_engage" in window_features.columns

    players = (
        window_features.group_by("match_id", "player_id", "label")
        .agg(
            n_kills=pl.len(),
            straight_share=(pl.col("straightness") >= STRAIGHT_EPS).mean(),
            corrections_median=pl.col("corrections").median(),
            corrections_mean=pl.col("corrections").mean(),
            snap_max=pl.col("speed_at_kill").max(),
            snap_p90=pl.col("speed_at_kill").quantile(0.9),
            snap_share=(pl.col("speed_at_kill") > SNAP_THRESHOLD_DPS).mean(),
            peak_speed_p90=pl.col("peak_speed_engage").quantile(0.9),
            peak_speed_median=pl.col("peak_speed_engage").median(),
            settle_ratio_median=pl.col("settle_ratio").median(),
            t_peak_ms_median=pl.col("t_peak_ms").median(),
            pitch_peak_median=pl.col("peak_pitch_speed_engage").median(),
            idle_speed_median=pl.col("mean_speed_idle").median(),
            **(perception if has_perception else {}),
            **context,
        )
        .filter(pl.col("n_kills") >= min_kills)
    )

    if window_ticks is not None:
        players = players.join(
            zero_motion_share(window_ticks), on=["match_id", "player_id"], how="left"
        )
    else:
        players = players.with_columns(zero_motion_share=pl.lit(None, pl.Float64))

    present = [
        c for c in (*PLAYER_FEATURE_COLUMNS, *CONTEXT_COLUMNS) if c in players.columns
    ]
    return players.select("match_id", "player_id", "label", *present).sort(
        ["match_id", "player_id"]
    )
