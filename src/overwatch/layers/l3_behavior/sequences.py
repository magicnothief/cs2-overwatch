"""Turn kill windows into fixed-shape arrays for a sequence model.

The feature table throws away the *shape* of a flick: it keeps a peak and a count
but not the curve. A sequence model reads the curve, so it needs each window as a
(channels, ticks) array with every window the same length and in the same order.

Channels:
    d_yaw          signed horizontal turn per tick, wrap-corrected
    d_pitch        signed vertical turn per tick
    valid          1 where the tick was measurable, 0 where it was not (dead,
                   freeze, a gap). The model needs to know a zero means "still"
                   in one case and "unknown" in the other.
    target_yaw     signed horizontal angle from the crosshair to the victim
    target_pitch   signed vertical angle to the victim
    log_distance   how far away the victim was
    target_visible 1 when the game says this attacker could see the victim, so
                   the model can tell tracking a visible enemy apart from
                   tracking one through a wall

The target channels are what a reviewer actually watches: not how fast the
crosshair moved, but how the gap to the enemy closed. They are added when the
window table carries them (see dataset.add_victim_geometry) and skipped otherwise,
so older caches still load.

Angle channels are compressed with a signed log so that an ordinary 0.5° tick and
an 11,000 °/s snap can share a scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

BASE_CHANNELS: tuple[str, ...] = ("d_yaw", "d_pitch", "valid")
TARGET_CHANNELS: tuple[str, ...] = (
    "target_yaw",
    "target_pitch",
    "log_distance",
    "target_visible",
)
CHANNELS: tuple[str, ...] = BASE_CHANNELS + TARGET_CHANNELS

#: Columns produced by dataset.add_victim_geometry, in channel order.
GEOMETRY_COLUMNS: tuple[str, ...] = (
    "target_yaw_error",
    "target_pitch_error",
    "target_distance",
    "target_visible",
)


@dataclass(frozen=True)
class SequenceData:
    """Model-ready windows plus everything needed to split and report honestly."""

    x: np.ndarray  # (n_windows, n_channels, n_ticks) float32
    y: np.ndarray  # (n_windows,) int8, 1 = cheater
    groups: np.ndarray  # (n_windows,) match_id, for grouped splits
    players: np.ndarray  # (n_windows,) "match_id/player_id", for per-player scores
    uids: np.ndarray  # (n_windows,) window_uid

    def __len__(self) -> int:
        return len(self.y)


def signed_log(values: np.ndarray) -> np.ndarray:
    """Compress a heavy tail while keeping direction: sign(x) * log1p(|x|)."""
    return np.sign(values) * np.log1p(np.abs(values))


def build_sequences(
    window_ticks: pl.DataFrame,
    *,
    labels: tuple[str, ...] = ("cheater", "clean"),
) -> SequenceData:
    """Pivot per-tick windows into (n_windows, channels, ticks) arrays.

    Windows are kept only if they are complete (every tick offset present) and
    carry one of `labels`.
    """
    frame = (
        window_ticks.filter(pl.col("label").is_in(labels))
        .select(
            "window_uid",
            "match_id",
            "player_id",
            "label",
            "tick_offset",
            "d_yaw",
            "d_pitch",
            "yaw_speed",
            *(c for c in GEOMETRY_COLUMNS if c in window_ticks.columns),
        )
        .sort(["window_uid", "tick_offset"])
    )

    offsets = frame["tick_offset"].unique().sort()
    n_ticks = len(offsets)

    # Drop any window that is short, so the reshape below is exact.
    complete = (
        frame.group_by("window_uid")
        .len()
        .filter(pl.col("len") == n_ticks)
        .select("window_uid")
    )
    frame = frame.join(complete, on="window_uid", how="inner").sort(
        ["window_uid", "tick_offset"]
    )

    meta = (
        frame.group_by("window_uid", maintain_order=True)
        .first()
        .select("window_uid", "match_id", "player_id", "label")
    )
    n_windows = meta.height

    d_yaw = frame["d_yaw"].to_numpy().reshape(n_windows, n_ticks)
    d_pitch = frame["d_pitch"].to_numpy().reshape(n_windows, n_ticks)
    # yaw_speed carries the validity rule (alive, not in freeze time, no tick gap);
    # d_yaw alone is non-null even for a dead player's frozen camera.
    valid = (
        ~np.isnan(frame["yaw_speed"].to_numpy().reshape(n_windows, n_ticks))
    ).astype(np.float32)

    channels = [
        (signed_log(np.nan_to_num(d_yaw)) * valid).astype(np.float32),
        (signed_log(np.nan_to_num(d_pitch)) * valid).astype(np.float32),
        valid,
    ]

    if all(c in frame.columns for c in GEOMETRY_COLUMNS):
        yaw_error, pitch_error, distance, visible = (
            frame[c].to_numpy().reshape(n_windows, n_ticks).astype(np.float64)
            for c in GEOMETRY_COLUMNS
        )
        # A missing victim position (rare) is "unknown", not "dead centre", so it
        # is pushed far off-target rather than to zero.
        known = ~np.isnan(yaw_error)
        channels += [
            (signed_log(np.nan_to_num(yaw_error, nan=180.0))).astype(np.float32),
            (signed_log(np.nan_to_num(pitch_error, nan=90.0))).astype(np.float32),
            np.log1p(np.nan_to_num(distance, nan=0.0)).astype(np.float32) * known,
            np.nan_to_num(visible).astype(np.float32),
        ]

    x = np.stack(channels, axis=1)

    return SequenceData(
        x=x,
        y=(meta["label"] == "cheater").to_numpy().astype(np.int8),
        groups=meta["match_id"].to_numpy(),
        players=(meta["match_id"] + "/" + meta["player_id"]).to_numpy(),
        uids=meta["window_uid"].to_numpy(),
    )
