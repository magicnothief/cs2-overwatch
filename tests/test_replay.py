"""Tests for the round replay's data (pipeline/replay.py).

A made-up match: T player A and CT player B, two rounds of 32 freeze ticks and
168 live ones. A stands at x = -200 looking along +x, B at x = +200. The map is
either a wall at x = 0 or open ground far away, so who sees whom is known.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from overwatch.parsing import TICK_COLUMNS, ParsedMatch
from overwatch.pipeline import replay

WALL = np.array(
    [
        [[0, -500, -500], [0, 500, -500], [0, 500, 500]],
        [[0, -500, -500], [0, 500, 500], [0, -500, 500]],
    ],
    dtype=np.float32,
)
FAR = WALL + np.float32(10_000)  # the same wall, nowhere near anyone


def _ticks(*, b_yaw: float = 180.0, gap: range = range(0)) -> pl.DataFrame:
    rows = []
    for tick in range(400):
        rnd, local = divmod(tick, 200)
        for pid, team, x, yaw in (("A", 2, -200.0, 0.0), ("B", 3, 200.0, b_yaw)):
            if pid == "A" and tick in gap:
                continue
            dead = pid == "B" and rnd == 0 and local >= 100
            rows.append(
                {
                    "player_id": pid,
                    "player_name": f"name {pid}",
                    "tick": tick,
                    "round": rnd,
                    "team": team,
                    "x": x,
                    "y": 0.0,
                    "z": 0.0,
                    "pitch": 0.0,
                    "yaw": yaw,
                    "health": 0 if dead else 100,
                    "is_alive": not dead,
                    "is_freeze": local < 32,
                    "spotted": False,
                    "spotted_by": [],
                    "flash_duration": 1.5 if pid == "A" and 36 <= tick < 44 else 0.0,
                }
            )
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("tick", "round", "team", "health").cast(pl.Int32),
            pl.col("x", "y", "z", "pitch", "yaw", "flash_duration").cast(pl.Float32),
        )
        .select(TICK_COLUMNS)
    )


def _events() -> dict[str, pl.DataFrame]:
    def ev(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows).with_columns(pl.col("tick").cast(pl.Int32))

    return {
        "player_death": ev(
            [
                {
                    "tick": 100,
                    "attacker_id": "A",
                    "victim_id": "B",
                    "weapon": "ak47",
                    "headshot": True,
                }
            ]
        ),
        "weapon_fire": ev(
            [
                {"tick": 40, "player_id": "A", "weapon": "weapon_ak47"},
                {"tick": 44, "player_id": "A", "weapon": "weapon_knife"},
            ]
        ),
        "smokegrenade_detonate": ev(
            [
                {"tick": 50, "entityid": 7, "x": 10.0, "y": 20.0, "player_id": "A"},
                {"tick": 300, "entityid": 9, "x": 0.0, "y": 0.0, "player_id": "B"},
            ]
        ),
        # 9's expiry at tick 10 belongs to an older entity with the same id
        "smokegrenade_expired": ev(
            [{"tick": 10, "entityid": 9, "x": 0.0, "y": 0.0, "player_id": "B"}]
        ),
        "inferno_startburn": ev(
            [{"tick": 60, "entityid": 7, "x": 1.0, "y": 2.0, "player_id": "A"}]
        ),
        "inferno_expire": ev(
            [{"tick": 120, "entityid": 7, "x": 1.0, "y": 2.0, "player_id": "A"}]
        ),
        # picked up in round 2's buy time
        "bomb_pickup": ev([{"tick": 210, "player_id": "A"}]),
        "bomb_planted": ev([{"tick": 330, "player_id": "A", "site": 1}]),
    }


def _map(tmp_path: Path, triangles: np.ndarray) -> Path:
    (tmp_path / "de_test.tri").write_bytes(triangles.tobytes())
    return tmp_path


def _build(tmp_path: Path, triangles: np.ndarray = WALL, **ticks) -> dict:
    match = ParsedMatch(
        ticks=_ticks(**ticks), events=_events(), meta={"map": "de_test"}
    )
    return replay.build_replay(match, map_dir=_map(tmp_path, triangles))


def test_rounds_are_play_time_from_freeze_end_to_the_next_freeze() -> None:
    bounds = replay.round_bounds(_ticks())
    assert bounds.rows() == [(1, 32, 200), (2, 232, 400)]


def test_every_player_has_one_value_per_sample(tmp_path: Path) -> None:
    first = _build(tmp_path, gap=range(60, 80))["rounds"][0]
    assert first["t0"] == 32
    expected = (200 - 32) // replay.SAMPLE_TICKS
    a, b = first["players"]
    for field in ("x", "y", "z", "yaw", "hp", "blind", "sees"):
        assert len(a[field]) == len(b[field]) == expected, field
    at = (64 - 32) // replay.SAMPLE_TICKS  # inside A's gap: missing, not shifted
    assert a["x"][at] is None and a["x"][at + 5] == -200
    assert b["hp"][(100 - 32) // replay.SAMPLE_TICKS] == 0  # dead after tick 100
    assert b["x"][(104 - 32) // replay.SAMPLE_TICKS] is None
    assert a["blind"][(40 - 32) // replay.SAMPLE_TICKS] == 1
    assert (a["side"], b["side"]) == ("T", "CT")


def test_a_wall_hides_both_players(tmp_path: Path) -> None:
    first = _build(tmp_path, WALL)["rounds"][0]
    assert not any(first["players"][0]["sees"])


def test_seen_means_a_clear_line_and_on_screen(tmp_path: Path) -> None:
    built = _build(tmp_path, FAR, b_yaw=0.0)  # B looks away from A
    assert built["visibility"] is True
    a, b = built["rounds"][0]["players"]
    bit_b = 1 << [p["id"] for p in built["players"]].index("B")
    assert a["sees"][0] == bit_b  # A faces B across open ground
    assert b["sees"][0] == 0  # B has A behind them


def test_no_mesh_means_no_visibility(tmp_path: Path) -> None:
    match = ParsedMatch(ticks=_ticks(), events=_events(), meta={"map": "de_nowhere"})
    built = replay.build_replay(match, map_dir=tmp_path)
    assert built["visibility"] is False
    assert "sees" not in built["rounds"][0]["players"][0]


def test_events_land_in_their_round(tmp_path: Path) -> None:
    built = _build(tmp_path)
    ids = [p["id"] for p in built["players"]]
    first, second = built["rounds"]
    assert first["shots"] == [[40, ids.index("A")]]  # the knife is not a shot
    assert first["deaths"] == [
        [100, ids.index("B"), ids.index("A"), "ak47", True, 200, 0]
    ]
    # no expiry: 18 s, cut at the round's end
    assert first["smokes"] == [[50, 200, 10, 20]]
    assert first["fires"] == [[60, 120, 1, 2]]
    # an expiry from before the detonation is another smoke's
    assert second["smokes"] == [[300, 400, 0, 0]]
    assert second["bomb"][0] == [210, "pickup", ids.index("A"), -200, 0]
    assert second["bomb"][1][:2] == [330, "plant"]


def test_the_file_is_gzipped_json_written_whole(tmp_path: Path) -> None:
    built = _build(tmp_path)
    path = replay.write_replay(built, tmp_path / "r.replay.json.gz")
    assert json.loads(gzip.decompress(path.read_bytes())) == built
    assert not list(tmp_path.glob("*.part"))


DEMO = next(
    iter(sorted((Path(__file__).parents[1] / "data" / "raw").glob("*.dem"))), None
)


@pytest.mark.skipif(DEMO is None, reason="demo not downloaded")
def test_a_real_demo_stays_within_budget(tmp_path: Path) -> None:
    from overwatch.parsing import parse_demo

    built = replay.build_replay(parse_demo(DEMO, hash_file=False))
    size = replay.write_replay(built, tmp_path / "r.json.gz").stat().st_size
    rounds = len(built["rounds"])
    assert size < 2_000_000 * rounds / 24  # the spec's 2 MB for 24 rounds, pro rata
    for rnd in built["rounds"]:
        lengths = {len(p[f]) for p in rnd["players"] for f in ("x", "hp")}
        assert len(lengths) == 1
