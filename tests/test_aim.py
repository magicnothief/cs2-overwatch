"""Tests for the shared angle maths. If these break, every layer is wrong."""

import polars as pl
import pytest

from overwatch.aim import TICK_RATE, add_aim_features, kill_windows, wrap_degrees


@pytest.mark.parametrize(
    ("raw_diff", "expected"),
    [
        (2.0, 2.0),
        (-2.0, -2.0),
        (-358.0, 2.0),
        (358.0, -2.0),
        (90.0, 90.0),
        (-90.0, -90.0),
    ],
)
def test_wrap_degrees(raw_diff: float, expected: float) -> None:
    got = pl.DataFrame({"d": [raw_diff]}).select(wrap_degrees(pl.col("d")))["d"][0]
    assert got == pytest.approx(expected)


def _frame(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal canonical tick table for the columns aim.py touches."""
    return pl.DataFrame(rows).with_columns(
        tick=pl.col("tick").cast(pl.Int32),
        yaw=pl.col("yaw").cast(pl.Float32),
        pitch=pl.lit(0.0, pl.Float32),
        is_alive=pl.lit(True),
        is_freeze=pl.lit(False),
    )


def test_speed_uses_shortest_turn() -> None:
    """179 -> -179 is a 2 degree turn, so 2 * 64 degrees per second."""
    ticks = _frame(
        [
            {"player_id": "a", "tick": 1, "yaw": 179.0},
            {"player_id": "a", "tick": 2, "yaw": -179.0},
        ]
    )
    out = add_aim_features(ticks).sort("tick")
    assert out["yaw_speed"][0] is None  # first row of a player has no previous
    assert out["yaw_speed"][1] == pytest.approx(2.0 * TICK_RATE)


def test_diff_never_crosses_players() -> None:
    """Player b's first row must not be compared with player a's last row."""
    ticks = _frame(
        [
            {"player_id": "a", "tick": 1, "yaw": 0.0},
            {"player_id": "a", "tick": 2, "yaw": 10.0},
            {"player_id": "b", "tick": 1, "yaw": 170.0},
            {"player_id": "b", "tick": 2, "yaw": 180.0},
        ]
    )
    out = add_aim_features(ticks)
    first_rows = out.filter(pl.col("tick") == 1)
    assert first_rows["yaw_speed"].null_count() == 2
    assert out.filter((pl.col("player_id") == "b") & (pl.col("tick") == 2))["d_yaw"][
        0
    ] == (pytest.approx(10.0))


def test_dead_and_freeze_ticks_have_no_speed() -> None:
    ticks = _frame(
        [
            {"player_id": "a", "tick": 1, "yaw": 0.0},
            {"player_id": "a", "tick": 2, "yaw": 90.0},
            {"player_id": "a", "tick": 3, "yaw": 180.0},
        ]
    ).with_columns(is_alive=pl.Series([True, True, False]))
    out = add_aim_features(ticks).sort("tick")
    assert out["yaw_speed"][1] == pytest.approx(90.0 * TICK_RATE)
    assert out["yaw_speed"][2] is None  # died between tick 2 and 3


def test_kill_window_has_fixed_length() -> None:
    pre, post = 8, 2
    ticks = _frame(
        [{"player_id": "a", "tick": t, "yaw": float(t % 360)} for t in range(1, 40)]
    )
    deaths = pl.DataFrame(
        {
            "tick": [20],
            "attacker_id": ["a"],
            "victim_id": ["b"],
            "weapon": ["ak47"],
            "headshot": [True],
        }
    ).with_columns(tick=pl.col("tick").cast(pl.Int32))

    windows, window_ticks = kill_windows(
        add_aim_features(ticks), deaths, pre=pre, post=post
    )

    assert windows.height == 1
    assert window_ticks.height == pre + post + 1
    assert window_ticks["tick_offset"].min() == -pre
    assert window_ticks["tick_offset"].max() == post
    assert window_ticks.filter(pl.col("tick_offset") == 0)["tick"][0] == 20


def test_incomplete_windows_are_dropped() -> None:
    """A kill too close to the start of the demo cannot fill its window."""
    ticks = _frame([{"player_id": "a", "tick": t, "yaw": 0.0} for t in range(1, 40)])
    deaths = pl.DataFrame(
        {
            "tick": [3],
            "attacker_id": ["a"],
            "victim_id": ["b"],
            "weapon": ["ak47"],
            "headshot": [False],
        }
    ).with_columns(tick=pl.col("tick").cast(pl.Int32))

    windows, window_ticks = kill_windows(add_aim_features(ticks), deaths, pre=8, post=2)
    assert windows.is_empty()
    assert window_ticks.is_empty()


def test_suicides_are_not_windows() -> None:
    ticks = _frame([{"player_id": "a", "tick": t, "yaw": 0.0} for t in range(1, 40)])
    deaths = pl.DataFrame(
        {
            "tick": [20],
            "attacker_id": ["a"],
            "victim_id": ["a"],
            "weapon": ["world"],
            "headshot": [False],
        }
    ).with_columns(tick=pl.col("tick").cast(pl.Int32))
    windows, _ = kill_windows(add_aim_features(ticks), deaths, pre=8, post=2)
    assert windows.is_empty()
