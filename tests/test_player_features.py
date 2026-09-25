"""Tests for per-player aggregation."""

import polars as pl
import pytest

from overwatch.layers.l3_behavior.player_features import (
    CONTEXT_COLUMNS,
    PERCEPTION_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    build_player_features,
    zero_motion_share,
)


def _windows(rows: list[dict]) -> pl.DataFrame:
    """Minimal window-feature frame: one row per kill."""
    defaults = {
        "match_id": "m1",
        "player_id": "p1",
        "label": "clean",
        "straightness": 0.5,
        "corrections": 1,
        "speed_at_kill": 10.0,
        "peak_speed_engage": 100.0,
        "settle_ratio": 0.1,
        "t_peak_ms": -200.0,
        "peak_pitch_speed_engage": 20.0,
        "mean_speed_idle": 15.0,
    }
    return pl.DataFrame([defaults | row for row in rows])


def test_shares_and_extremes() -> None:
    windows = _windows(
        [
            {"straightness": 1.0, "speed_at_kill": 900.0},
            {"straightness": 1.0, "speed_at_kill": 5.0},
            {"straightness": 0.4, "speed_at_kill": 5.0},
            {"straightness": 0.4, "speed_at_kill": 5.0},
            {"straightness": 0.4, "speed_at_kill": 5.0},
        ]
    )
    row = build_player_features(windows).row(0, named=True)
    assert row["n_kills"] == 5
    assert row["straight_share"] == pytest.approx(0.4)
    assert row["snap_max"] == pytest.approx(900.0)
    assert row["snap_share"] == pytest.approx(0.2)  # one kill over 200 deg/s
    assert row["snap_kills"] == 1
    # perception and timing columns need victim geometry, which this has none of
    timing = {"arrival_kills", "arrival_shot_share"}
    assert set(PLAYER_FEATURE_COLUMNS) - set(PERCEPTION_COLUMNS) - timing <= set(row)


def test_sniper_share_is_context_never_a_model_input() -> None:
    # a detector that sees weapon mix learns "snipes a lot" as "cheats" (ADR 0008)
    assert not set(CONTEXT_COLUMNS) & set(PLAYER_FEATURE_COLUMNS)
    windows = _windows([{"weapon": "awp"}] * 3 + [{"weapon": "ak47"}] * 2)
    row = build_player_features(windows).row(0, named=True)
    assert row["sniper_share"] == pytest.approx(0.6)


def test_players_below_min_kills_are_dropped() -> None:
    windows = _windows([{}, {}])
    assert build_player_features(windows, min_kills=5).is_empty()
    assert build_player_features(windows, min_kills=2).height == 1


def test_players_are_kept_separate_per_match() -> None:
    windows = _windows(
        [{"match_id": "m1"}] * 5 + [{"match_id": "m2"}] * 5 + [{"player_id": "p2"}] * 5
    )
    out = build_player_features(windows)
    assert out.height == 3


def test_perception_columns_appear_when_geometry_is_present() -> None:
    windows = _windows([{}] * 5).with_columns(
        visible_share_engage=pl.lit(0.2),
        aim_through_wall_share=pl.lit(0.7),
        reaction_ms=pl.lit(150.0),
        angle_at_first_visible=pl.lit(1.5),
        visible_before_kill=pl.lit(value=True),
    )
    row = build_player_features(windows).row(0, named=True)
    assert row["visible_share"] == pytest.approx(0.2)
    assert row["wall_aim_share"] == pytest.approx(0.7)
    assert row["reaction_ms_min"] == pytest.approx(150.0)


def test_zero_motion_share_counts_only_exact_stillness() -> None:
    ticks = pl.DataFrame(
        {
            "match_id": ["m1"] * 4,
            "player_id": ["p1"] * 4,
            "yaw_speed": [0.0, 0.0, 1.0, None],
        }
    )
    assert zero_motion_share(ticks)["zero_motion_share"][0] == pytest.approx(2 / 3)
