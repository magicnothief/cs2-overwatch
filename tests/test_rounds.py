"""What a round is (pipeline/rounds.py), for the report's timeline and the replay.

The game's round counter goes up on the tick the round-deciding kill lands, and
once more after the final round. Two rounds of 200 ticks here, each starting with
32 ticks of buy time, then 50 ticks after the match with no buy time.
"""

import polars as pl

from overwatch.pipeline.analyze import round_of, round_spans
from overwatch.pipeline.rounds import round_bounds


def _ticks() -> pl.DataFrame:
    rows = []
    for tick in range(450):
        counter, local = min(divmod(tick, 200)[0], 2), tick % 200
        rows.append(
            {"tick": tick, "round": counter, "is_freeze": counter < 2 and local < 32}
        )
    return pl.DataFrame(rows).with_columns(pl.col("tick", "round").cast(pl.Int32))


def test_a_round_is_play_time_and_the_aftermath_is_the_last_round() -> None:
    # round 1 holds tick 200, where its counter moved on; round 2 runs to the end
    # of the demo, and there is no round 3 made of the aftermath
    assert round_bounds(_ticks()).rows() == [(1, 32, 201), (2, 232, 450)]


def test_the_round_deciding_kill_is_in_the_round_it_decided() -> None:
    rounds = round_spans(_ticks())
    assert [r.number for r in rounds] == [1, 2]
    # tick 200: the counter has moved on, the kill that moved it has not
    assert round_of(rounds, 199) == 1
    assert round_of(rounds, 200) == 1
    # the match-ending kill, after the counter's last step
    assert round_of(rounds, 449) == 2


def test_a_match_that_freezes_as_it_ends_keeps_its_last_kill() -> None:
    # after the match, frozen from the counter's step to the end of the demo
    ticks = _ticks().with_columns(
        is_freeze=pl.col("is_freeze") | (pl.col("round") == 2)
    )
    assert round_bounds(ticks).rows() == [(1, 32, 201), (2, 232, 401)]
    assert round_of(round_spans(ticks), 400) == 2


def test_buy_time_and_warmup_count_for_a_neighbouring_round() -> None:
    rounds = round_spans(_ticks())
    assert round_of(rounds, 210) == 1  # round 2's buy time: the round before it
    assert round_of(rounds, 5) == 1  # before any play: the first round
    assert round_of([], 5) is None
