"""Tests for triggerbot timing: which shot counts, and when the crosshair arrived.

Windows built by hand, where the answer is obvious: a target 800 units away, so
"on the head" (8 units) is about 0.57 degrees. The crosshair sweeps at SWEEP deg
a tick unless a test says otherwise; an arrival needs it to be moving at
ARRIVAL_SWEEP_DPS or more (the arrival redesign), since otherwise it is the enemy who arrived.
"""

import polars as pl

from overwatch.aim import TICK_RATE
from overwatch.layers.l3_behavior.player_features import MIN_ARRIVAL_KILLS
from overwatch.layers.l3_behavior.shots import (
    ARRIVAL_SWEEP_DPS,
    arrival_features,
    mark_shots,
)

OFF, ON = 6.0, 0.2  # degrees from the head: well off, squarely on
#: A tick's worth of crosshair motion at twice the sweep the redesign requires.
SWEEP = 2 * ARRIVAL_SWEEP_DPS / TICK_RATE
#: A crosshair standing still: the enemy is the one who moved.
HELD = 0.0


def _window(
    uid: str,
    angles: dict[int, float],
    shots: set[int],
    first: int = -60,
    sweep: dict[int, float] | float = SWEEP,
) -> pl.DataFrame:
    offsets = list(range(first, 1))
    steps = (
        [sweep] * len(offsets)
        if isinstance(sweep, float)
        else [sweep.get(o, SWEEP) for o in offsets]
    )
    return pl.DataFrame(
        {
            "window_uid": uid,
            "tick_offset": offsets,
            "target_angle": [angles.get(o, OFF) for o in offsets],
            "target_distance": 800.0,
            "shot": [o in shots for o in offsets],
            "d_yaw": steps,
            "d_pitch": 0.0,
            "pitch": 0.0,
        }
    )


def _on_from(start: int, end: int = 0) -> dict[int, float]:
    return {o: ON for o in range(start, end + 1)}


def _result(*windows: pl.DataFrame) -> dict[str, tuple]:
    out = arrival_features(pl.concat(windows))
    return {
        r["window_uid"]: (r["arrival_shot"], r["arrival_delay_ms"])
        for r in out.to_dicts()
    }


def test_a_shot_on_the_arrival_tick_counts_and_one_later_does_not() -> None:
    got = _result(
        _window("instant", _on_from(-10), {-10}),
        _window("human", _on_from(-10), {-6}),
    )
    assert got["instant"] == (True, 0.0)
    assert got["human"] == (False, 62.5)


def test_the_enemy_walking_into_a_held_crosshair_is_not_an_arrival() -> None:
    """The angle closes, but the crosshair never moved, so nothing arrived.

    This is the pro holding a pre-aimed angle and firing because they were already
    aimed at it. Counting it was why 22 of 341 pros sat past the v5 line.
    """
    got = _result(
        _window("walked-in", _on_from(-10), {-10}, sweep={-10: HELD}),
        _window(
            "crept-in",
            _on_from(-10),
            {-10},
            sweep={-10: 0.9 * ARRIVAL_SWEEP_DPS / TICK_RATE},
        ),
    )
    assert got == {}


def test_a_later_arrival_by_the_crosshair_is_the_one_that_counts() -> None:
    """Off the head, back on twice: only a swept arrival can be the last one."""
    angles = _on_from(-30, -21) | _on_from(-10)
    got = _result(_window("swept-then-held", angles, {-10}, sweep={-10: HELD}))
    # the -10 re-arrival is the enemy's, so the -30 sweep stands and the shot is late
    assert got["swept-then-held"] == (False, 20 / 64 * 1000)


def test_a_pre_aimed_kill_has_no_arrival() -> None:
    """The crosshair sat on the spot from the start: nothing arrived, nothing counts."""
    got = _result(_window("held", _on_from(-60), {-3}))
    assert "held" not in got


def test_only_the_opening_shot_of_a_spray_counts() -> None:
    """Mid-spray a shot lands on a re-arrival by chance; only the first shot counts."""
    angles = _on_from(-45, -31) | _on_from(-28)  # on, off for two ticks, back on
    got = _result(_window("spray", angles, {-40, -34, -28, -22, -16}))
    assert got["spray"] == (False, 5 / 64 * 1000)  # arrived at -45, opened at -40


def test_a_new_burst_after_a_pause_is_its_own_opening_shot() -> None:
    angles = _on_from(-50, -45) | _on_from(-12)
    got = _result(_window("two", angles, {-48, -42, -12}))  # 30 ticks apart: new burst
    assert got["two"] == (True, 0.0)


def test_knives_and_grenades_are_not_shots() -> None:
    ticks = pl.DataFrame(
        {"player_id": ["a", "a", "b"], "tick": [10, 11, 10]}
    ).with_columns(pl.col("tick").cast(pl.Int32))
    fire = pl.DataFrame(
        {
            "user_steamid": ["a", "a", "b"],
            "tick": [10, 11, 10],
            "weapon": ["weapon_ak47", "weapon_knife", "weapon_hegrenade"],
        }
    )
    marked = mark_shots(ticks, fire).sort("player_id", "tick")
    assert marked["shot"].to_list() == [True, False, False]
    assert mark_shots(ticks, None)["shot"].to_list() == [False, False, False]


def test_the_share_needs_enough_kills_with_an_arrival() -> None:
    from overwatch.layers.l3_behavior.player_features import build_player_features

    def kills(player: str, arrivals: list[bool | None]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "window_uid": [f"{player}{i}" for i in range(len(arrivals))],
                "match_id": "m",
                "player_id": player,
                "label": "clean",
                "straightness": 0.5,
                "corrections": 1.0,
                "speed_at_kill": 10.0,
                "peak_speed_engage": 100.0,
                "settle_ratio": 0.1,
                "t_peak_ms": -200.0,
                "peak_pitch_speed_engage": 5.0,
                "mean_speed_idle": 3.0,
                "arrival_shot": pl.Series(arrivals, dtype=pl.Boolean),
            }
        )

    players = build_player_features(
        pl.concat(
            [
                kills("few", [True, True, None, None, None]),
                kills("enough", [True, False] * MIN_ARRIVAL_KILLS),
            ]
        )
    )
    by_player = {r["player_id"]: r for r in players.to_dicts()}
    assert by_player["few"]["arrival_kills"] == 2
    assert by_player["few"]["arrival_shot_share"] is None
    assert by_player["enough"]["arrival_shot_share"] == 0.5
