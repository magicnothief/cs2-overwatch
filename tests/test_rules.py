"""Tests for Layer 1. These rules accuse people, so they get the strictest tests."""

import polars as pl
import pytest

from overwatch.aim import TICK_RATE, add_aim_features
from overwatch.layers.l1_blatant.rules import run_rules
from overwatch.layers.l1_blatant.thresholds import Thresholds
from overwatch.schemas.evidence import Moment, Severity

THRESHOLDS = Thresholds(
    spin_speed_dps=2620.0,
    spin_min_ticks=16,
    alias_min_step_deg=90.0,
    alias_min_ticks=6,
    snap_speed_dps=2210.0,
    max_pitch_deg=89.0,
)


def _ticks(yaws: list[float], *, pitches: list[float] | None = None) -> pl.DataFrame:
    pitches = pitches if pitches is not None else [0.0] * len(yaws)
    return add_aim_features(
        pl.DataFrame(
            {
                "player_id": ["p1"] * len(yaws),
                "tick": list(range(1, len(yaws) + 1)),
                "yaw": yaws,
                "pitch": pitches,
                "is_alive": [True] * len(yaws),
                "is_freeze": [False] * len(yaws),
            }
        ).with_columns(
            tick=pl.col("tick").cast(pl.Int32),
            yaw=pl.col("yaw").cast(pl.Float32),
            pitch=pl.col("pitch").cast(pl.Float32),
        )
    )


def _no_deaths() -> pl.DataFrame:
    return pl.DataFrame(
        {"tick": [], "attacker_id": [], "victim_id": []},
        schema={"tick": pl.Int32, "attacker_id": pl.String, "victim_id": pl.String},
    )


def _run(ticks: pl.DataFrame, deaths: pl.DataFrame | None = None) -> list[Moment]:
    return run_rules(
        ticks,
        deaths if deaths is not None else _no_deaths(),
        thresholds=THRESHOLDS,
        match_id="m1",
        with_aim_features=False,
    )


def test_sustained_spin_is_flagged() -> None:
    """60 degrees every tick for a second is 3840 deg/s, held far too long."""
    yaws = [(i * 60.0) % 360 - 180 for i in range(TICK_RATE)]
    moments = _run(_ticks(yaws))
    spins = [m for m in moments if m.trigger == "spinbot"]
    assert spins, "a sustained spin must be flagged"
    assert spins[0].evidence[0].severity == Severity.IMPOSSIBLE
    assert spins[0].evidence[0].value > THRESHOLDS.spin_speed_dps


def test_a_single_fast_flick_is_not_a_spin() -> None:
    """One 90 degree flick in two ticks is fast but human; the rule must ignore it."""
    yaws = [0.0] * 20 + [45.0, 90.0] + [90.0] * 20
    assert not [m for m in _run(_ticks(yaws)) if m.trigger == "spinbot"]


def test_aliased_spin_is_flagged() -> None:
    """A spin near 180 degrees per tick wraps into large alternating turns.

    Turning 175 then 185 degrees reads as +175 then -175, because anything past
    180 comes back round the other side. That alternation, not raw speed, is what
    gives away a spin too fast for 64 ticks to measure.
    """
    yaw, yaws = 0.0, []
    for i in range(24):
        yaw += 175.0 if i % 2 else 185.0
        yaws.append((yaw + 180) % 360 - 180)
    moments = [m for m in _run(_ticks(yaws)) if m.trigger == "aliased_spin"]
    assert moments
    assert moments[0].evidence[0].value >= THRESHOLDS.alias_min_ticks


def test_small_oscillation_near_the_wrap_point_is_not_a_spin() -> None:
    """Looking back and forth across 180 degrees is a 2 degree jitter, not a spin."""
    yaws = [179.0 if i % 2 else -179.0 for i in range(20)]
    assert not [m for m in _run(_ticks(yaws)) if m.trigger == "aliased_spin"]


def test_ordinary_aim_triggers_nothing() -> None:
    yaws = [float(i % 7) for i in range(200)]
    assert _run(_ticks(yaws)) == []


def test_kill_tick_snap_uses_the_kill_tick() -> None:
    yaws = [0.0] * 10 + [50.0] + [50.0] * 10  # 50 deg in one tick = 3200 deg/s
    deaths = pl.DataFrame(
        {"tick": [11], "attacker_id": ["p1"], "victim_id": ["p2"]}
    ).with_columns(tick=pl.col("tick").cast(pl.Int32))
    moments = [m for m in _run(_ticks(yaws), deaths) if m.trigger == "kill_tick_snap"]
    assert len(moments) == 1
    assert moments[0].evidence[0].tick == 11
    assert moments[0].evidence[0].value == pytest.approx(50 * TICK_RATE, rel=0.01)


def test_snap_on_a_different_tick_is_ignored() -> None:
    """The same flick one tick away from the kill is not this rule's business."""
    yaws = [0.0] * 10 + [50.0] + [50.0] * 10
    deaths = pl.DataFrame(
        {"tick": [15], "attacker_id": ["p1"], "victim_id": ["p2"]}
    ).with_columns(tick=pl.col("tick").cast(pl.Int32))
    assert not [m for m in _run(_ticks(yaws), deaths) if m.trigger == "kill_tick_snap"]


def test_impossible_pitch_is_flagged() -> None:
    moments = [
        m
        for m in _run(_ticks([0.0] * 5, pitches=[0.0, 0.0, 91.0, 0.0, 0.0]))
        if m.trigger == "impossible_pitch"
    ]
    assert len(moments) == 1
    assert moments[0].evidence[0].value == pytest.approx(91.0)


def test_evidence_round_trips_as_json() -> None:
    """Layer 4 and the web UI both read these as JSON."""
    yaws = [(i * 60.0) % 360 - 180 for i in range(TICK_RATE)]
    moment = _run(_ticks(yaws))[0]
    restored = Moment.model_validate_json(moment.model_dump_json())
    assert restored == moment
    assert restored.demo_command().startswith("demo_gototick ")
