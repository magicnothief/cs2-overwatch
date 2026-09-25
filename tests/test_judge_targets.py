"""Tests for the judge's training targets.

These targets are written by code, so a mistake here teaches the model to be wrong
in a consistent, confident way — the worst failure mode available.
"""

import inspect
import random

from overwatch.layers.l4_judge import MomentSummary, PlayerCase, render_case
from overwatch.layers.l4_judge.rendering import unseen_numbers
from overwatch.layers.l4_judge.targets import build_target, case_rng, verdict_for
from overwatch.layers.l4_judge.verdict import VerdictLabel
from overwatch.schemas.evidence import Evidence, Severity

BASELINES = {
    "wall_aim_share": 0.08,
    "never_visible_share": 0.17,
    "straight_share": 0.15,
    "corrections_mean": 1.4,
    "reaction_ms_min": 210.0,
    "snap_max": 63.0,
}


#: Where notable and strong begin, as measured on CS2CD's clean players.
LINES = {
    "wall_aim_share": (0.46, 0.67),
    "never_visible_share": (0.37, 0.86),
    "straight_share": (0.48, 0.67),
    "corrections_mean": (0.58, 0.25),
    "reaction_ms_min": (15.6, 0.0),
    "snap_max": (283.0, 720.0),
}


def _case(**features) -> PlayerCase:
    return PlayerCase(
        match_id="m",
        player_id="p",
        kills=20,
        score=0.5,
        features=features,
        baselines=BASELINES,
        clean_lines=LINES,
    )


def test_verdict_agrees_with_probability() -> None:
    assert verdict_for(5) is VerdictLabel.CLEAN
    assert verdict_for(50) is VerdictLabel.UNCLEAR
    assert verdict_for(90) is VerdictLabel.CHEATING


def test_every_target_is_internally_consistent() -> None:
    rng = random.Random(0)
    for wall, straight in ((0.05, 0.10), (0.5, 0.7), (0.9, 0.95)):
        target = build_target(_case(wall_aim_share=wall, straight_share=straight), rng)
        assert verdict_for(target.probability) is target.verdict
        assert target.reasons


def test_the_ban_label_cannot_reach_a_target() -> None:
    """The first fine-tune learned to guess the label from the behaviour score
    because targets depended on it. A target is a function of the evidence only."""
    assert "label" not in inspect.signature(build_target).parameters


def test_a_match_that_shows_nothing_is_clean() -> None:
    target = build_target(
        _case(wall_aim_share=0.07, straight_share=0.14), random.Random(0)
    )
    assert target.verdict is VerdictLabel.CLEAN


def test_thin_evidence_is_unclear_and_more_evidence_is_surer() -> None:
    one = build_target(_case(straight_share=0.5), random.Random(0))
    two = build_target(
        _case(straight_share=0.5, corrections_mean=0.5), random.Random(0)
    )
    assert one.verdict is two.verdict is VerdictLabel.UNCLEAR
    assert one.probability < two.probability


def test_a_heavy_sniper_is_not_condemned_on_visibility_alone() -> None:
    """Sniping inflates the visibility measurements (ADR 0008)."""
    sniper = _case(wall_aim_share=0.8, never_visible_share=0.9, sniper_share=0.9)
    target = build_target(sniper, random.Random(0))
    assert target.verdict is not VerdictLabel.CHEATING
    assert any("snipe" in caveat for caveat in target.caveats)

    # the same evidence from a rifle player is damning
    rifler = _case(wall_aim_share=0.8, never_visible_share=0.9, sniper_share=0.05)
    assert build_target(rifler, random.Random(0)).verdict is VerdictLabel.CHEATING


def test_reasons_only_cite_numbers_that_appear_in_the_evidence() -> None:
    """The model must never learn to invent figures."""
    case = _case(wall_aim_share=0.64, straight_share=0.62, sniper_share=0.1)
    target = build_target(case, random.Random(0))
    evidence = render_case(case)
    for reason in target.reasons:
        for token in reason.split():
            if token.rstrip("%,.").replace(".", "").isdigit() and len(token) > 2:
                assert token.rstrip("%,.") in evidence.replace(",", ""), reason


def _kill(tick: int, **context) -> MomentSummary:
    """A kill aimed entirely through cover, on a map where sight was ray cast."""
    return MomentSummary(tick=tick, wall_aim_share=1.0, sight_checked=True, **context)


def test_repeated_aim_at_unseen_unheard_enemies_tips_notable_evidence() -> None:
    # two notable measurements (0.8) fall short on their own; the pattern adds 0.4
    case = _case(straight_share=0.5, corrections_mean=0.5).model_copy(
        update={"moments": [_kill(100), _kill(200), _kill(300, last_seen_ms=900.0)]}
    )
    target = build_target(case, random.Random(0))
    assert target.verdict is VerdictLabel.CHEATING
    assert any("2 of their 3" in reason for reason in target.reasons)


def test_one_unexplained_kill_is_not_enough() -> None:
    """The system prompt says a single unusual kill is weak evidence."""
    case = _case(straight_share=0.14).model_copy(update={"moments": [_kill(100)]})
    target = build_target(case, random.Random(0))
    assert target.verdict is not VerdictLabel.CHEATING
    assert not any("tracked an enemy" in reason for reason in target.reasons)


def test_aim_through_cover_after_seeing_or_hearing_gets_a_caveat() -> None:
    case = _case(straight_share=0.14).model_copy(
        update={
            "moments": [
                _kill(100, last_seen_ms=1200.0),
                _kill(200, victim_fired_ms_ago=300.0),
                _kill(300, victim_was_running=True),
            ]
        }
    )
    target = build_target(case, random.Random(0))
    assert target.verdict is VerdictLabel.CLEAN
    assert any("seen or heard" in caveat for caveat in target.caveats)


def test_no_mesh_means_no_claim_about_what_they_saw() -> None:
    """Without ray casting, a missing sighting is unknown, not incriminating."""
    blind = MomentSummary(tick=100, wall_aim_share=1.0, sight_checked=False)
    case = _case(straight_share=0.14).model_copy(update={"moments": [blind, blind]})
    target = build_target(case, random.Random(0))
    assert target.verdict is not VerdictLabel.CHEATING
    assert not any("seen or heard" in reason for reason in target.reasons)


def test_a_zero_ms_reaction_does_not_outrank_a_hard_limit() -> None:
    case = _case(reaction_ms_min=0.0).model_copy(
        update={
            "rule_evidence": [
                Evidence(
                    layer="l1_blatant",
                    name="sustained_spin",
                    value=5141.0,
                    severity=Severity.IMPOSSIBLE,
                    note="view spun at a sustained 5141 deg/s",
                )
            ]
        }
    )
    target = build_target(case, random.Random(0))
    assert "5141" in target.reasons[0]


def test_a_fast_single_reaction_is_not_strong_evidence() -> None:
    """A third of clean players have one kill under 100 ms: a pre-aimed angle."""
    case = _case(reaction_ms_min=94.0, straight_share=0.14)
    assert build_target(case, random.Random(0)).verdict is (VerdictLabel.CLEAN)


def test_a_case_gets_the_same_target_in_any_order() -> None:
    """The annotation tool reveals the target the training set holds."""
    case = _case(wall_aim_share=0.5, straight_share=0.14)
    first = build_target(case, case_rng("m", "p"))
    for _ in range(3):
        build_target(_case(), case_rng("other", "x"))
    assert build_target(case, case_rng("m", "p")) == first


def test_every_cited_number_is_one_the_model_was_shown() -> None:
    """Reasons quoted "1.43" while the evidence said "1.4"; the model must never
    learn to produce a figure its input does not contain."""
    from overwatch.layers.l4_judge.targets import evidence_points

    baselines = {
        "wall_aim_share": 0.1349,
        "never_visible_share": 0.0417,
        "straight_share": 0.1467,
        "corrections_mean": 1.4306,
        "reaction_ms_min": 234.375,
        "snap_max": 63.41,
    }
    case = PlayerCase(
        match_id="m",
        player_id="p",
        kills=20,
        score=0.5,
        features={
            "wall_aim_share": 0.7351,
            "never_visible_share": 0.8765,
            "straight_share": 0.6849,
            "corrections_mean": 0.2449,
            "reaction_ms_min": 0.0,
            "snap_max": 1362.46,
        },
        baselines=baselines,
        clean_lines=LINES,
    )
    evidence = render_case(case)
    points = evidence_points(case)
    assert len(points) == 6
    for sentence, _, _ in points:
        assert unseen_numbers(sentence, evidence) == [], sentence


def test_an_invented_figure_is_caught() -> None:
    evidence = (
        "- aim corrections per kill: 1.4 (clean 0.74)\n"
        "1. tick 5, 100% of the approach aimed through cover\n  context: 2v4"
    )
    assert unseen_numbers("averaged 1.43 corrections", evidence) == ["1.43"]
    assert unseen_numbers("reacted in 94 ms", evidence) == ["94"]
    assert unseen_numbers("1.4 corrections against 74% of clean", evidence) == []
    assert unseen_numbers("on 2 of their 3 kills", evidence) == []
    assert unseen_numbers("100% of the approach through cover", evidence) == []


def test_the_lines_a_target_is_decided_by_are_in_the_text() -> None:
    """v2 leaned on the behaviour score because the text showed clean medians but
    not the 95th/99th-percentile lines its targets were decided by."""
    case = _case(straight_share=0.6, corrections_mean=0.2)
    text = render_case(case)
    assert "95% of clean players are below 0.48, 99% below 0.67" in text
    assert "95% of clean players are above 0.58, 99% above 0.25" in text


def test_a_value_shown_equal_to_a_line_is_not_past_it() -> None:
    """0.4801 reads as 0.48 in the text; the target must agree with the text."""
    at_line = build_target(_case(straight_share=0.4801), random.Random(0))
    assert at_line.verdict is VerdictLabel.CLEAN
