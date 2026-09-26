"""Tests for the parsers. Every source must produce the same tables."""

import os
import sys
import time
from pathlib import Path

import polars as pl
import pytest

from overwatch.aim import add_aim_features, kill_windows
from overwatch.demos import DEMO_MAGIC
from overwatch.parsing import (
    TICK_COLUMNS,
    DemoParseError,
    load_cs2cd,
    parse_demo,
    worker,
)
from overwatch.parsing import demo as demo_module

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


@pytest.mark.skipif(not DEMO.exists(), reason="demo not downloaded")
def test_the_worker_returns_what_an_in_process_parse_returns() -> None:
    """Moving layer 0 into a child changed nothing about the tables it produces."""
    here = worker.parse_in_process(DEMO)
    child = parse_demo(DEMO, hash_file=False)
    assert child.ticks.equals(here.ticks)
    assert list(child.events) == list(here.events)
    for name, frame in here.events.items():
        assert child.events[name].equals(frame), name
    assert child.meta == here.meta


def test_a_truncated_demo_fails_the_parse_without_naming_home(tmp_path: Path) -> None:
    """The real worker on a real hostile file: it dies, the parent says so.

    This is the boundary itself, not a stand-in — demoparser2 panics on eight
    bytes of magic and nothing behind it, in the child, and this process lives.
    """
    cut = tmp_path / "cut.dem"
    cut.write_bytes(DEMO_MAGIC + b"nothing behind the magic")
    with pytest.raises(DemoParseError) as raised:
        parse_demo(cut)
    assert str(Path.home()) not in str(raised.value)


def test_a_missing_demo_is_refused_before_a_worker_starts(tmp_path: Path) -> None:
    with pytest.raises(DemoParseError, match="no demo to parse"):
        parse_demo(tmp_path / "gone.dem")


def _fake_worker(script: str):
    """A stand-in worker, so a crash can be asked for instead of waited for."""
    return lambda demo, out_dir: [sys.executable, "-c", script, str(demo), str(out_dir)]


def test_a_worker_that_exits_nonzero_fails_the_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = tmp_path / "x.dem"
    demo.write_bytes(DEMO_MAGIC)
    monkeypatch.setattr(
        demo_module,
        "_worker_command",
        _fake_worker(
            "import sys; print('parser gave up', file=sys.stderr); sys.exit(3)"
        ),
    )
    with pytest.raises(DemoParseError, match="exited 3") as raised:
        parse_demo(demo)
    assert "parser gave up" in str(raised.value)


def test_a_worker_that_writes_no_tables_fails_the_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0 is not enough: the frames have to be there to be read."""
    demo = tmp_path / "x.dem"
    demo.write_bytes(DEMO_MAGIC)
    monkeypatch.setattr(demo_module, "_worker_command", _fake_worker("pass"))
    with pytest.raises(DemoParseError, match="no tables for player_death"):
        parse_demo(demo)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_worker_killed_mid_parse_fails_the_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGKILL is what a memory-safety bug looks like from out here."""
    demo = tmp_path / "x.dem"
    demo.write_bytes(DEMO_MAGIC)
    monkeypatch.setattr(
        demo_module,
        "_worker_command",
        _fake_worker("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
    )
    with pytest.raises(DemoParseError, match="SIGKILL"):
        parse_demo(demo)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_worker_that_hangs_is_killed_and_leaves_no_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = tmp_path / "x.dem"
    demo.write_bytes(DEMO_MAGIC)
    pidfile = tmp_path / "pid"
    monkeypatch.setattr(
        demo_module,
        "_worker_command",
        _fake_worker(
            "import os, sys, time;"
            f" open({str(pidfile)!r}, 'w').write(str(os.getpid()));"
            " time.sleep(60)"
        ),
    )
    with pytest.raises(DemoParseError, match="did not finish within 2s"):
        parse_demo(demo, timeout=2.0)

    child = int(pidfile.read_text())
    for _ in range(50):  # kill(2) is asynchronous; the reap is not instant
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise AssertionError(f"worker {child} outlived its timeout")


@pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX only")
@pytest.mark.skipif(not DEMO.exists(), reason="demo not downloaded")
def test_the_worker_holds_the_memory_cap_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """64 MB is below what the allocator reserves, so the child cannot even
    load the parser: proof the cap reaches the process that reads the demo."""
    monkeypatch.setenv("OVERWATCH_PARSE_MEMORY_MB", "64")
    with pytest.raises(DemoParseError):
        parse_demo(DEMO, hash_file=False)


def test_a_header_string_from_the_demo_is_capped() -> None:
    """map_name and server_name come from whoever made the file."""
    long = worker._header_text({"map_name": "d" * 5000}, "map_name")
    assert len(long) == worker.MAX_HEADER_CHARS
    assert worker._header_text({}, "map_name") is None


@pytest.mark.skipif(not CS2CD.exists(), reason="CS2CD match not downloaded")
def test_load_cs2cd_matches_demo_schema() -> None:
    match = load_cs2cd(CS2CD)
    assert tuple(match.ticks.columns) == TICK_COLUMNS
    assert match.ticks.height > 0
    assert match.meta["source"] == "cs2cd"
    assert match.meta["cheaters"]
    assert set(match.deaths.columns) >= {"tick", "attacker_id", "victim_id"}
