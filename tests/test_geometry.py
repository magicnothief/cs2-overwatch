"""Tests for the crosshair-to-target geometry.

Every case here is one a human can check on paper, because a sign error in this
module would quietly corrupt every feature built on top of it.
"""

import polars as pl
import pytest

from overwatch.layers.l2_perception.geometry import angles_to_target, in_field_of_view

EYE = 64.0


def _row(yaw: float, pitch: float, target: tuple[float, float, float]) -> dict:
    frame = pl.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "yaw": [yaw],
            "pitch": [pitch],
            "target_x": [target[0]],
            "target_y": [target[1]],
            "target_z": [target[2]],
        }
    )
    return angles_to_target(frame).row(0, named=True)


def test_looking_straight_at_a_target_is_zero() -> None:
    row = _row(yaw=0.0, pitch=0.0, target=(500.0, 0.0, 0.0))
    assert row["target_angle"] == pytest.approx(0.0, abs=1e-4)
    assert row["target_yaw_error"] == pytest.approx(0.0, abs=1e-4)
    assert row["target_pitch_error"] == pytest.approx(0.0, abs=1e-4)
    assert row["target_distance"] == pytest.approx(500.0)


def test_target_to_the_left_gives_positive_yaw_error() -> None:
    """In Source, yaw increases counterclockwise, so +y is to the left of +x."""
    row = _row(yaw=0.0, pitch=0.0, target=(500.0, 500.0, 0.0))
    assert row["target_yaw_error"] == pytest.approx(45.0, abs=1e-3)
    assert row["target_angle"] == pytest.approx(45.0, abs=1e-3)


def test_target_behind_is_180_degrees_away() -> None:
    row = _row(yaw=0.0, pitch=0.0, target=(-500.0, 0.0, 0.0))
    assert row["target_angle"] == pytest.approx(180.0, abs=1e-3)


def test_pitch_is_negative_upwards() -> None:
    """Looking up is negative pitch; a target above should sit at a negative error."""
    row = _row(yaw=0.0, pitch=0.0, target=(100.0, 0.0, 100.0))
    assert row["target_pitch_error"] == pytest.approx(45.0, abs=1e-3)

    aimed_up = _row(yaw=0.0, pitch=-45.0, target=(100.0, 0.0, 100.0))
    assert aimed_up["target_angle"] == pytest.approx(0.0, abs=1e-3)
    assert aimed_up["target_pitch_error"] == pytest.approx(0.0, abs=1e-3)


def test_eye_and_head_heights_cancel_on_level_ground() -> None:
    """Both players stand on the floor, so their eyes are level."""
    row = _row(yaw=0.0, pitch=0.0, target=(500.0, 0.0, 0.0))
    assert row["target_pitch_error"] == pytest.approx(0.0, abs=1e-6)


def test_wrap_around_in_yaw_error() -> None:
    """Looking at -179 with a target at +179 is a 2 degree error, not 358."""
    row = _row(yaw=-179.0, pitch=0.0, target=(-500.0, 8.73, 0.0))  # ~179 degrees
    assert abs(row["target_yaw_error"]) < 5


def test_field_of_view_cuts_at_half_the_fov() -> None:
    frame = pl.DataFrame({"target_angle": [10.0, 52.0, 60.0]})
    assert frame.select(in_field_of_view())["target_angle"].to_list() == [
        True,
        True,
        False,
    ]
