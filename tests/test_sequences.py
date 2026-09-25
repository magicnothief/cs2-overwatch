"""Tests for the sequence arrays and the match-grouped split."""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("sklearn")  # training-only; the app installs without it

from overwatch.layers.l3_behavior.evaluation import split_by_match
from overwatch.layers.l3_behavior.sequences import build_sequences, signed_log


def _ticks(n_windows: int = 3, n_ticks: int = 5) -> pl.DataFrame:
    rows = []
    for w in range(n_windows):
        for offset in range(-n_ticks + 1, 1):
            rows.append(
                {
                    "window_uid": f"m{w % 2}/0#{w}",
                    "match_id": f"m{w % 2}",
                    "player_id": f"p{w}",
                    "label": "cheater" if w == 0 else "clean",
                    "tick_offset": offset,
                    "d_yaw": float(offset),
                    "d_pitch": 0.5,
                    "yaw_speed": None if offset == -1 else 1.0,
                }
            )
    return pl.DataFrame(rows)


def test_shapes_and_labels() -> None:
    seq = build_sequences(_ticks())
    assert seq.x.shape == (3, 3, 5)
    assert seq.y.tolist() == [1, 0, 0]
    assert len(seq) == 3
    # rows come back sorted by window_uid: m0/0#0, m0/0#2, m1/0#1
    assert seq.uids.tolist() == ["m0/0#0", "m0/0#2", "m1/0#1"]
    assert seq.groups.tolist() == ["m0", "m0", "m1"]


def test_invalid_ticks_are_masked_and_zeroed() -> None:
    """A tick with no measurable speed must be flagged, not read as 'still'."""
    seq = build_sequences(_ticks())
    valid = seq.x[:, 2]
    assert valid[:, -2].sum() == 0  # the offset == -1 column is invalid everywhere
    assert np.all(seq.x[:, 0][valid == 0] == 0)  # and its angle is zeroed


def test_incomplete_windows_are_dropped() -> None:
    ticks = _ticks()
    short = ticks.filter(
        ~((pl.col("window_uid") == "m0/0#0") & (pl.col("tick_offset") == 0))
    )
    seq = build_sequences(short)
    assert "m0/0#0" not in seq.uids.tolist()
    assert len(seq) == 2


def test_signed_log_keeps_direction_and_compresses() -> None:
    assert signed_log(np.array([-100.0])) < 0
    assert abs(signed_log(np.array([11000.0]))[0]) < 10


def test_split_is_by_match_and_disjoint() -> None:
    groups = np.array([f"m{i // 10}" for i in range(200)])
    splits = split_by_match(groups)
    matches = {name: set(groups[idx]) for name, idx in splits.items()}
    assert not matches["train"] & matches["test"]
    assert not matches["val"] & matches["test"]
    assert sum(len(idx) for idx in splits.values()) == len(groups)


def test_split_is_deterministic() -> None:
    groups = np.array([f"m{i // 5}" for i in range(100)])
    assert (
        split_by_match(groups)["test"].tolist()
        == split_by_match(groups)["test"].tolist()
    )
    with pytest.raises(AssertionError):
        assert (
            split_by_match(groups, seed=1)["test"].tolist()
            == split_by_match(groups)["test"].tolist()
        )


def _ticks_with_geometry() -> pl.DataFrame:
    """Windows that carry victim geometry, as dataset.add_victim_geometry emits."""
    base = _ticks()
    return base.with_columns(
        target_yaw_error=pl.lit(5.0),
        target_pitch_error=pl.lit(-2.0),
        target_distance=pl.lit(500.0),
        target_visible=pl.lit(value=True),
    )


def test_geometry_channels_are_added_when_present() -> None:
    seq = build_sequences(_ticks_with_geometry())
    assert seq.x.shape[1] == 7
    assert np.allclose(seq.x[:, 5], np.log1p(500.0), atol=1e-3)
    assert np.all(seq.x[:, 6] == 1.0)  # the victim was visible throughout


def test_missing_victim_position_reads_as_far_away_not_on_target() -> None:
    """A null angle must not look like a crosshair sitting on the enemy."""
    ticks = _ticks_with_geometry().with_columns(
        target_yaw_error=pl.lit(None, pl.Float64),
        target_distance=pl.lit(None, pl.Float64),
    )
    seq = build_sequences(ticks)
    assert np.all(np.abs(seq.x[:, 3]) > 1.0)  # pushed off-target, not zero
    assert np.all(seq.x[:, 5] == 0)  # distance unknown
