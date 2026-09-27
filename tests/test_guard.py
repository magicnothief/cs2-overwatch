"""The judge may doubt on one line; it may not accuse on one (l4_judge/guard.py)."""

from overwatch.layers.l4_judge import PlayerCase
from overwatch.layers.l4_judge.guard import guard
from overwatch.layers.l4_judge.verdict import CheatType, Verdict, VerdictLabel

LINES = {
    "straight_share": (0.48, 0.67),
    "corrections_mean": (0.58, 0.25),
    "snap_kills": (2.0, 3.0),
}
BASELINES = {"straight_share": 0.15, "corrections_mean": 1.4, "snap_kills": 0.0}


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


def _said(verdict: VerdictLabel, probability: int = 82) -> Verdict:
    return Verdict(
        verdict=verdict,
        probability=probability,
        cheat_type=CheatType.AIMBOT
        if verdict is VerdictLabel.CHEATING
        else CheatType.NONE,
        reasons=["flicks that never change direction: 0.70"],
    )


def test_an_accusation_on_one_line_is_held_at_unclear() -> None:
    shown, held = guard(_said(VerdictLabel.CHEATING), _case(straight_share=0.7), 4)
    assert shown.verdict is VerdictLabel.UNCLEAR
    assert shown.probability <= 60 and shown.cheat_type is CheatType.NONE
    assert shown.reasons == ["flicks that never change direction: 0.70"]
    assert held and "one measurement past its 99% line alone" in held
    assert "82%, aimbot" in held


def test_a_corroborated_accusation_stands() -> None:
    said = _said(VerdictLabel.CHEATING)
    shown, held = guard(said, _case(straight_share=0.7, corrections_mean=0.5), 4)
    assert shown == said and held is None


def test_a_doubt_or_a_clearance_is_never_touched() -> None:
    for verdict in (VerdictLabel.UNCLEAR, VerdictLabel.CLEAN):
        said = _said(verdict, 40)
        assert guard(said, _case(straight_share=0.7), 4) == (said, None)


def test_a_line_the_judge_was_not_shown_does_not_corroborate() -> None:
    # v4 is not shown snap_kills (a judge-6 measurement), so it cannot back v4 up
    case = _case(straight_share=0.7, snap_kills=5.0)
    shown, held = guard(_said(VerdictLabel.CHEATING), case, 4)
    assert shown.verdict is VerdictLabel.UNCLEAR and held
    # the judge that is shown it is backed up by it
    assert guard(_said(VerdictLabel.CHEATING), case, 6)[1] is None
