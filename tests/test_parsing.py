"""Tests for the parsers. Every source must produce the same tables."""

from pathlib import Path

import polars as pl
import pytest

from overwatch.aim import add_aim_features, kill_windows
from overwatch.parsing import TICK_COLUMNS, load_cs2cd, parse_demo

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parents[1] / "data" / "raw"
#: any CS2 demo of your own: the test that parses one is skipped without it
DEMO = next(iter(sorted(DATA.glob("*.dem"))), DATA / "match.dem")
CS2CD = DATA / "cs2cd" / "with_cheater_present" / "0.parquet"


def test_fixture_has_canonical_columns() -> None:
    ticks = pl.read_parquet(FIXTURES / "mini_ticks.parquet")
    assert tuple(ticks.columns) == TICK_COLUMNS
    assert ticks.height > 0


def test_fixture_runs_through_the_pipeline() -> None:
    """ticks -> features -> windows, the path every layer depends on."""
    ticks = add_aim_features(pl.read_parquet(FIXTURES / "mini_ticks.parquet"))
    deaths = pl.read_parquet(FIXTURES / "mini_deaths.parquet")
    windows, window_ticks = kill_windows(ticks, deaths, pre=32, post=8)
    assert windows.height >= 1
    assert window_ticks.height == windows.height * (32 + 8 + 1)


@pytest.mark.skipif(not DEMO.exists(), reason="demo not downloaded")
def test_parse_demo() -> None:
    match = parse_demo(DEMO, hash_file=False)
    assert tuple(match.ticks.columns) == TICK_COLUMNS
    assert match.ticks.height > 0
    assert match.meta["source"] == "dem"
    assert match.meta["map"]
    assert set(match.deaths.columns) >= {
        "tick",
        "attacker_id",
        "victim_id",
        "weapon",
        "headshot",
    }
    assert match.ticks["player_id"].dtype == pl.String


@pytest.mark.skipif(not CS2CD.exists(), reason="CS2CD match not downloaded")
def test_load_cs2cd_matches_demo_schema() -> None:
    match = load_cs2cd(CS2CD)
    assert tuple(match.ticks.columns) == TICK_COLUMNS
    assert match.ticks.height > 0
    assert match.meta["source"] == "cs2cd"
    assert match.meta["cheaters"]
    assert set(match.deaths.columns) >= {"tick", "attacker_id", "victim_id"}
