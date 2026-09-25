"""Tests for the per-window feature builder."""

import polars as pl
import pytest

from overwatch.layers.l3_behavior.features import (
    FEATURE_COLUMNS,
    PERCEPTION_COLUMNS,
    build_window_features,
)


def _window(d_yaws: list[float], *, pre: int = 4, post: int = 2) -> pl.DataFrame:
    """One synthetic window whose per-tick turn is exactly `d_yaws`."""
    offsets = list(range(-pre, post + 1))
    assert len(d_yaws) == len(offsets)
    return pl.DataFrame(
        {
            "window_uid": ["w"] * len(offsets),
            "label": ["clean"] * len(offsets),
            "tick_offset": offsets,
            "t_ms": [o / 64 * 1000 for o in offsets],
            "d_yaw": d_yaws,
            "yaw_speed": [abs(v) * 64 for v in d_yaws],
            "pitch_speed": [0.0] * len(offsets),
        }
    ).with_columns(tick_offset=pl.col("tick_offset").cast(pl.Int32))


def test_straight_flick_scores_one() -> None:
    """All movement in one direction: straightness 1, no corrections."""
    features = build_window_features(_window([0.0, 5.0, 10.0, 20.0, 5.0, 0.0, 0.0]))
    row = features.row(0, named=True)
    assert row["straightness"] == pytest.approx(1.0)
    assert row["corrections"] == 0
    assert row["turn_total_engage"] == pytest.approx(40.0)


def test_overshoot_and_correction_is_detected() -> None:
    """Turn right, overshoot, come back: one direction change, straightness < 1."""
    features = build_window_features(_window([0.0, 10.0, 20.0, -6.0, 2.0, 0.0, 0.0]))
    row = features.row(0, named=True)
    assert row["corrections"] == 2  # right -> left -> right
    assert row["straightness"] < 1.0
    assert row["turn_total_engage"] == pytest.approx(38.0)
    assert row["turn_net_engage"] == pytest.approx(26.0)


def test_jitter_below_threshold_is_not_a_correction() -> None:
    features = build_window_features(_window([0.0, 10.0, -0.01, 10.0, 0.0, 0.0, 0.0]))
    assert features.row(0, named=True)["corrections"] == 0


def test_peak_and_settle() -> None:
    """Peak is found in the engage segment, with its time and the speed at the shot."""
    features = build_window_features(_window([0.0, 1.0, 10.0, 2.0, 1.0, 0.0, 0.0]))
    row = features.row(0, named=True)
    assert row["peak_speed_engage"] == pytest.approx(10.0 * 64)
    assert row["t_peak_ms"] == pytest.approx(
        -2 / 64 * 1000
    )  # two ticks before the kill
    assert row["speed_at_kill"] == pytest.approx(1.0 * 64)
    assert row["settle_ratio"] == pytest.approx(0.1)


def test_aim_feature_columns_are_present_without_geometry() -> None:
    """Windows with no victim geometry still produce every aim feature."""
    features = build_window_features(_window([0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0]))
    assert set(FEATURE_COLUMNS) - set(PERCEPTION_COLUMNS) <= set(features.columns)
    assert not set(PERCEPTION_COLUMNS) & set(features.columns)
    assert features.height == 1
