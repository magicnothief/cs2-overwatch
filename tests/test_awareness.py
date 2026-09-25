"""Tests for the context a reviewer needs: what the shooter could have known.

This is the evidence *for* the player. If it is wrong, an honest player who watched
an enemy walk behind a wall reads exactly like a wallhacker.
"""

import polars as pl

from overwatch.layers.l2_perception.awareness import alive_counts, describe, last_seen
from overwatch.layers.l4_judge.cases import rule_evidence_for


def test_describe_never_seen_is_stated_plainly() -> None:
    text = describe(
        {
            "sight_checked": True,
            "last_seen_ms": None,
            "allies_alive": 2,
            "enemies_alive": 4,
        }
    )
    assert "had not seen this enemy" in text
    assert "2v4" in text


def test_no_map_mesh_is_not_read_as_never_seen() -> None:
    # without a mesh nothing was ray cast, so a null is "unknown" — saying "had not
    # seen this enemy" would be an accusation built on a missing file
    text = describe({"sight_checked": False, "last_seen_ms": None})
    assert "seen" not in text and "saw" not in text


def test_describe_includes_every_innocent_explanation() -> None:
    text = describe(
        {
            "sight_checked": True,
            "last_seen_ms": 3400.0,
            "victim_fired_ms_ago": 500.0,
            "victim_was_running": True,
        }
    )
    assert "3.4s" in text
    assert "fired 0.5s" in text
    assert "running within earshot" in text


def _ticks(rows: list[tuple[str, int, str, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["player_id", "tick", "team", "is_alive"], orient="row"
    )


def test_alive_counts_are_from_the_attackers_side() -> None:
    ticks = _ticks(
        [
            ("a1", 100, "CT", True),
            ("a2", 100, "CT", True),
            ("a3", 100, "CT", False),
            ("e1", 100, "T", True),
        ]
    )
    windows = pl.DataFrame({"window_id": [0], "attacker_id": ["a1"], "tick": [100]})
    counts = alive_counts(windows, ticks).row(0, named=True)
    assert counts["allies_alive"] == 2
    assert counts["enemies_alive"] == 1


def test_last_seen_is_null_without_a_map() -> None:
    windows = pl.DataFrame(
        {"window_id": [0], "attacker_id": ["a"], "victim_id": ["v"], "tick": [1000]}
    )
    ticks = pl.DataFrame(
        {"player_id": ["a"], "tick": [1000], "x": [0.0], "y": [0.0], "z": [0.0]}
    )
    seen = last_seen(windows, ticks, map_name=None)
    assert seen["last_seen_ms"].to_list() == [None]


def _rule(name: str, severity: str, value: float, tick: int) -> dict:
    return {
        "match_id": "m",
        "player_id": "p",
        "layer": "l1_blatant",
        "name": name,
        "value": value,
        "unit": "deg/s",
        "severity": severity,
        "tick": tick,
        "note": f"{name} at {value:.0f}",
    }


def test_repeated_rule_hits_collapse_to_one_line_with_a_count() -> None:
    rules = pl.DataFrame(
        [
            _rule("sustained_spin", "impossible", 5140.0, 10),
            _rule("sustained_spin", "impossible", 5141.0, 20),
            _rule("sustained_spin", "impossible", 5139.0, 30),
            _rule("kill_tick_snap", "strong", 3000.0, 40),
        ]
    )
    evidence = rule_evidence_for(rules, "m", "p")
    assert [e.name for e in evidence] == ["sustained_spin", "kill_tick_snap"]
    assert evidence[0].value == 5141.0  # the worst hit is the one shown
    assert "3 such episodes" in evidence[0].note
    assert "episodes" not in evidence[1].note  # a single hit needs no count


def test_other_players_rule_hits_are_not_borrowed() -> None:
    rules = pl.DataFrame([_rule("sustained_spin", "impossible", 5140.0, 10)])
    assert rule_evidence_for(rules, "m", "someone_else") == []
    assert rule_evidence_for(None, "m", "p") == []
