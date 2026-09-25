"""Tests for the end-to-end pipeline: demo tables in, a report out.

The models here are stand-ins (random weights, a made-up reference, a fake judge):
these tests check the plumbing — every player reported, flags explained, the judge
shown exactly what it was trained on — not the detector's accuracy, which the
cross-validation measures.
"""

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from overwatch.layers.l3_behavior.sequences import CHANNELS
from overwatch.layers.l4_judge.judge import JudgeResult
from overwatch.layers.l4_judge.verdict import CheatType, Verdict, VerdictLabel
from overwatch.parsing import ParsedMatch
from overwatch.pipeline import Scorer, analyze_match, render_text

FIXTURES = Path(__file__).parent / "fixtures"


def _bundle(path: Path, *, clean_line: float | None = None) -> Path:
    """A scorer with random weights; `clean_line` sets every clean quantile.

    fixtures/scorer_tiny.onnx is a FlickCNN of width 8 with torch.manual_seed(0)
    weights, exported by l3_behavior/export.py, so these tests need no torch.
    """
    levels = [round(q / 100, 2) for q in range(1, 100)] + [0.995, 0.999]
    quantiles = {str(q): (q if clean_line is None else clean_line) for q in levels}
    onnx = path.with_suffix(".onnx")
    shutil.copy(FIXTURES / "scorer_tiny.onnx", onnx)
    meta = {
        "config": {
            "in_channels": len(CHANNELS),
            "channels": list(CHANNELS),
            "width": 8,
            "pooling": "avgmax",
            "pre": 128,
            "post": 32,
        },
        "reference": {
            "score": {
                "aggregate": "mean",
                "min_windows": 1,
                "clean_quantiles": quantiles,
            },
            "judge": {"baselines": {"wall_aim_share": 0.14}, "lines": {}},
        },
        "trained_at": "test",
    }
    onnx.with_suffix(".json").write_text(json.dumps(meta))
    return onnx


@pytest.fixture
def match() -> ParsedMatch:
    ticks = pl.read_parquet(FIXTURES / "mini_ticks.parquet")
    deaths = pl.read_parquet(FIXTURES / "mini_deaths.parquet")
    return ParsedMatch(
        ticks=ticks, events={"player_death": deaths}, meta={"map": "de_nowhere"}
    )


class FakeJudge:
    """Records what it was shown and always says the same thing."""

    model_path = Path("fake.gguf")

    def __init__(self) -> None:
        self.seen = []

    def judge(self, case):
        self.seen.append(case)
        verdict = Verdict(
            verdict=VerdictLabel.UNCLEAR,
            probability=50,
            cheat_type=CheatType.NONE,
            reasons=["the evidence points both ways"],
        )
        return JudgeResult(verdict, "{}", 0.1, 10)


# --- the scorer ---------------------------------------------------------------


def test_a_score_is_placed_among_clean_players(tmp_path: Path) -> None:
    scorer = Scorer(_bundle(tmp_path / "s.pt"))
    assert scorer.clean_percentile(0.0) == 0.0
    assert scorer.clean_percentile(0.505) == pytest.approx(0.5)
    assert scorer.clean_percentile(0.9995) == pytest.approx(0.999)
    assert scorer.clean_line(0.9) == pytest.approx(0.9)


def test_player_score_is_the_mean_of_their_kills(tmp_path: Path) -> None:
    scorer = Scorer(_bundle(tmp_path / "s.pt"))
    windows = pl.DataFrame(
        {
            "window_uid": ["a", "b", "c"],
            "player_id": ["p", "p", "q"],
            "score": [0.2, 0.6, 0.9],
        }
    )
    players = {
        r["player_id"]: r for r in scorer.score_players(windows).iter_rows(named=True)
    }
    assert players["p"]["score"] == pytest.approx(0.4)
    assert players["p"]["n_windows"] == 2


def test_a_missing_bundle_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_final.py"):
        Scorer(tmp_path / "missing.pt")


# --- the pipeline -------------------------------------------------------------


def test_every_player_is_reported(match: ParsedMatch, tmp_path: Path) -> None:
    report = analyze_match(match, "fixture", scorer=Scorer(_bundle(tmp_path / "s.pt")))
    assert {p.player_id for p in report.players} == set(match.ticks["player_id"])
    assert report.kills == match.deaths.height
    assert {"layer 1", "layer 2", "layer 3", "layer 4"} <= set(report.timings)
    text = render_text(report)
    assert "not a 5v5 match" in text and "no collision mesh" in text


def test_a_flag_always_says_why(match: ParsedMatch, tmp_path: Path) -> None:
    # every clean player scored 0: anyone with a score is above the line
    scorer = Scorer(_bundle(tmp_path / "s.pt", clean_line=0.0))
    report = analyze_match(match, "fixture", scorer=scorer)
    scored = [p for p in report.players if p.score is not None]
    assert scored and all(p.flagged for p in scored)
    assert all("higher than" in p.flag_reasons[0] for p in scored)
    assert all(p.top_kills[0].demo_command.startswith("demo_gototick") for p in scored)


def test_nobody_is_flagged_below_the_line(match: ParsedMatch, tmp_path: Path) -> None:
    scorer = Scorer(_bundle(tmp_path / "s.pt", clean_line=1.0))
    report = analyze_match(match, "fixture", scorer=scorer, judge=FakeJudge())
    assert not any(p.flagged for p in report.players)
    assert all(p.verdict is None for p in report.players)  # judge only reads flags


def test_the_judge_sees_an_alias_never_a_steam_id(
    match: ParsedMatch, tmp_path: Path
) -> None:
    """Trained on "Player_3", the judge must be shown "Player_3" — and a report must
    not leak who was judged into the text the model reads."""
    judge = FakeJudge()
    scorer = Scorer(_bundle(tmp_path / "s.pt", clean_line=0.0))
    report = analyze_match(match, "fixture", scorer=scorer, judge=judge)
    judged = [p for p in report.players if p.verdict is not None]
    assert judged and len(judge.seen) == len(judged)
    for player in judged:
        assert player.judge_evidence.startswith(f"PLAYER {player.judge_alias} ")
        assert player.player_id not in player.judge_evidence
        assert (player.name or "~") not in player.judge_evidence


def test_the_timeline_has_every_kill_in_its_round(
    match: ParsedMatch, tmp_path: Path
) -> None:
    messages = []
    report = analyze_match(
        match,
        "fixture",
        scorer=Scorer(_bundle(tmp_path / "s.pt")),
        progress=lambda stage, message: messages.append(stage),
    )
    assert report.rounds and report.rounds[0].number >= 1
    logged = [k for p in report.players for k in p.kill_log]
    assert len(logged) == sum(p.kills for p in report.players)
    for kill in logged:
        span = next(r for r in report.rounds if r.number == kill.round)
        assert span.start_tick <= kill.tick <= span.end_tick
        assert kill.trajectory  # the detail panel draws this
    assert all(p.side in {"CT", "T"} for p in report.players)
    assert messages == ["layer 1", "layer 2", "layer 3"]  # no judge, no layer 4 line
