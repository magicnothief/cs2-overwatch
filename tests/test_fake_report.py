"""The synthetic report the design work looks at must stay a valid report.

If a field moves in MatchReport, this fails here rather than as an empty page.
"""

from __future__ import annotations

from overwatch.pipeline.report import MatchReport
from overwatch.web import fake_report


def test_flagged_report_matches_the_schema():
    report = MatchReport.model_validate(fake_report.build())
    flagged = [p for p in report.players if p.flagged]
    assert len(flagged) == 1
    assert flagged[0].verdict is not None
    assert flagged[0].top_kills
    # the page's radar and trace need both, or the moment renders empty
    assert flagged[0].top_kills[0].path and flagged[0].top_kills[0].trajectory


def test_clean_report_flags_nobody():
    report = MatchReport.model_validate(fake_report.build(clean=True))
    assert not any(p.flagged for p in report.players)
    assert all(p.verdict is None for p in report.players)


def test_nobody_kills_themselves():
    report = MatchReport.model_validate(fake_report.build())
    for player in report.players:
        assert all(k.victim != player.name for k in player.kill_log)
