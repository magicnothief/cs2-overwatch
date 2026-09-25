"""Tests for the evidence text and the verdict schema.

The renderer is the contract between the detector and the model, and it is used
unchanged at training and inference, so its output is pinned by tests.
"""

import json

import pytest

from overwatch.layers.l4_judge import (
    MomentSummary,
    PlayerCase,
    Verdict,
    VerdictLabel,
    render_case,
)
from overwatch.schemas.evidence import Evidence, Severity


def _case(**overrides) -> PlayerCase:
    defaults = {
        "match_id": "m1",
        "player_id": "Player_3",
        "map_name": "de_mirage",
        "rank": "Gold Nova Master",
        "kills": 24,
        "score": 0.87,
        "features": {
            "straight_share": 0.62,
            "wall_aim_share": 0.64,
            "reaction_ms_min": 94.0,
        },
        "baselines": {
            "straight_share": 0.17,
            "wall_aim_share": 0.38,
            "reaction_ms_min": 210.0,
        },
    }
    return PlayerCase(**(defaults | overrides))


def test_every_number_is_paired_with_a_baseline() -> None:
    text = render_case(_case())
    for line in text.splitlines():
        if line.startswith("- ") and "clean" not in line:
            pytest.fail(f"measurement without a baseline: {line}")
    assert "0.62" in text and "0.17" in text


def test_no_label_leaks_into_the_text() -> None:
    """The model must never be told, or hinted, what the answer is."""
    text = render_case(_case()).lower()
    for forbidden in (
        "banned",
        "vac",
        "label",
        "cheater=",
        "ground truth",
        "is a cheater",
    ):
        assert forbidden not in text


def test_rule_evidence_and_moments_are_included() -> None:
    case = _case(
        rule_evidence=[
            Evidence(
                layer="l1_blatant",
                name="sustained_spin",
                value=11520.0,
                unit="deg/s",
                severity=Severity.IMPOSSIBLE,
                tick=42359,
                note="view spun at a sustained 11520 deg/s for 3.16s",
            )
        ],
        moments=[
            MomentSummary(
                tick=1724,
                weapon="ak47",
                headshot=True,
                reaction_ms=94.0,
                wall_aim_share=0.8,
            )
        ],
    )
    text = render_case(case)
    assert "11520" in text and "42359" in text
    assert "tick 1724" in text and "ak47 headshot" in text
    assert "94 ms" in text


def test_kill_context_is_rendered_from_its_numbers() -> None:
    moment = MomentSummary(
        tick=500,
        weapon="awp",
        distance=23.4,
        walls_penetrated=1,
        sight_checked=True,
        last_seen_ms=None,
        victim_fired_ms_ago=400.0,
    )
    text = render_case(_case(moments=[moment]))
    assert "23 m away" in text
    assert "shot through a wall" in text
    assert "context: had not seen this enemy" in text
    assert "fired 0.4s before" in text


def test_case_without_optional_parts_still_renders() -> None:
    case = PlayerCase(match_id="m", player_id="p", kills=7, score=0.1)
    text = render_case(case)
    assert "7 kills" in text
    assert "HARD-LIMIT" not in text  # nothing triggered, so no empty section


def test_rendering_is_deterministic() -> None:
    """Train and inference must produce byte-identical text for the same case."""
    assert render_case(_case()) == render_case(_case())


def test_verdict_round_trips_and_rejects_nonsense() -> None:
    verdict = Verdict.model_validate_json(
        json.dumps(
            {
                "verdict": "cheating",
                "cheat_type": "wallhack",
                "reasons": ["aimed through cover on 64% of the approach"],
                "caveats": ["only 24 kills"],
            }
        )
    )
    assert verdict.verdict is VerdictLabel.CHEATING
    assert "wallhack" in verdict.as_report_line()

    with pytest.raises(ValueError, match="verdict"):
        Verdict.model_validate(
            {"verdict": "definitely a cheater", "cheat_type": "none"}
        )


def test_a_full_window_of_visibility_is_not_a_reaction_time() -> None:
    """2000 ms is the window length, not a reaction: say so in words."""
    case = _case(moments=[MomentSummary(tick=1, reaction_ms=2000.0)])
    text = render_case(case)
    assert "2000 ms after" not in text
    assert "visible the whole" in text


def test_never_visible_is_stated_plainly() -> None:
    case = _case(moments=[MomentSummary(tick=1, visible_before_kill=False)])
    assert "never became visible" in render_case(case)
