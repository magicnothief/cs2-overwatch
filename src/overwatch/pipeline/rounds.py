"""What a round is, once, for the report's timeline and the replay alike.

A round is its play time: from the end of its freeze to the start of the next
round's freeze. The game's own counter (`round`, rounds already played) is not
enough on its own: it goes up on the very tick the round-deciding kill lands, so
counting by it puts every round's last kill in the next round, and the time after
the final round becomes a round of its own. Here a round always holds the tick
its counter moves on (at the end of a match the freeze can start on that same
tick), and the time after the final round belongs to it: dropped when it is all
freeze, added to the final round when it has none.
"""

from __future__ import annotations

import polars as pl


def round_bounds(ticks: pl.DataFrame) -> pl.DataFrame:
    """Each round's play time: number (1-based), start tick, end tick (exclusive)."""
    per = (
        ticks.drop_nulls("round")
        .group_by("round")
        .agg(
            first=pl.col("tick").min(),
            last=pl.col("tick").max(),
            freeze_first=pl.col("tick").filter(pl.col("is_freeze")).min(),
            freeze_last=pl.col("tick").filter(pl.col("is_freeze")).max(),
        )
        .sort("round")
    )
    # after the final round the counter moves on, but no buy time follows
    after = per.height > 1 and per["freeze_first"][-1] is None
    bounds = per.select(
        number=(pl.col("round") + 1).cast(pl.Int32),
        start=pl.coalesce(pl.col("freeze_last") + 1, pl.col("first")).cast(pl.Int32),
        # at least one tick past the next round's first: the deciding kill's tick
        end=pl.max_horizontal(
            pl.coalesce(
                pl.col("freeze_first").shift(-1),
                pl.col("first").shift(-1),
                pl.col("last") + 1,
            ),
            pl.col("first").shift(-1) + 1,
        ).cast(pl.Int32),
    )
    if after:
        bounds = bounds.head(-1).with_columns(
            end=pl.when(pl.int_range(pl.len()) == pl.len() - 1)
            .then(pl.lit(int(per["last"][-1]) + 1, dtype=pl.Int32))
            .otherwise(pl.col("end"))
        )
    return bounds.filter(pl.col("end") > pl.col("start"))


__all__ = ["round_bounds"]
