"""Tests for telling a stacked map (Nuke, Vertigo) from a flat one, for the radars."""

import numpy as np

from overwatch.maps.radar import split_height, storeys


def _stand(rng, n, x0, x1, y0, y1, z):
    """n players standing about in a box, on a floor at height z."""
    return np.column_stack(
        [rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), z + rng.normal(0, 4, n)]
    )


def test_a_flat_map_has_one_storey() -> None:
    rng = np.random.default_rng(0)
    where = np.vstack(
        [
            _stand(rng, 40_000, 0, 2000, 0, 2000, 0),
            _stand(rng, 4_000, 0, 200, 0, 200, 250),  # one raised platform
        ]
    )
    split, share = split_height(*where.T)
    assert split is None
    assert share < 0.10


def test_a_stacked_map_splits_between_its_storeys() -> None:
    rng = np.random.default_rng(0)
    where = np.vstack(
        [
            _stand(rng, 40_000, 0, 2000, 0, 2000, -400),  # the upper storey
            _stand(rng, 20_000, 0, 1000, 0, 1000, -760),  # a site beneath it
        ]
    )
    split, share = split_height(*where.T)
    assert share > 0.2
    assert -700 < split < -450


def test_storeys_name_the_lower_radar() -> None:
    assert storeys(None) == [("", -np.inf, np.inf)]
    assert storeys(-500.0) == [("", -500.0, np.inf), ("_lower", -np.inf, -500.0)]
