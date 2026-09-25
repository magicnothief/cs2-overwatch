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


NUKE_OVERVIEW = """// HLTV overview description file for de_nuke.bsp
"de_nuke"
{
	"material"	"overviews/de_nuke"	// texture file
	"pos_x"		"-3453"	// upper left world coordinate
	"pos_y"		"2887"
	"scale"		"7"
	"verticalsections"
	{
		"default" // use the primary radar image
		{
			"AltitudeMax" "10000"
			"AltitudeMin" "-495"
		}
		"lower" // i.e. de_nuke_lower_radar.dds
		{
			"AltitudeMax" "-495"
			"AltitudeMin" "-10000"
		}
	}
}
"""


def test_valves_overview_places_the_radar_and_splits_the_floors() -> None:
    from overwatch.maps.valve_radar import parse_overview

    nuke = parse_overview(NUKE_OVERVIEW)
    assert (nuke.x_min, nuke.y_max, nuke.scale) == (-3453, 2887, 7)
    assert nuke.split_z == -495
    flat = parse_overview('"de_dust2" { "pos_x" "-2476" "pos_y" "3239" "scale" "4.4" }')
    assert flat.split_z is None
    assert flat.scale == 4.4


def test_a_valve_radar_is_recoloured_into_the_pages_greys() -> None:
    """Colour off, the playable area inside a wall line, the void around it."""
    from PIL import Image

    from overwatch.maps.radar import EDGE, VOID, restyle

    image = np.zeros((64, 64, 4), dtype=np.uint8)
    image[16:48, 16:48] = (60, 140, 90, 255)  # a green room
    image[16:48, 32:48] = (200, 120, 40, 255)  # an orange one beside it
    out = np.asarray(restyle(Image.fromarray(image, "RGBA"))).astype(int)
    assert (out[2, 2] == VOID).all()  # outside the map
    assert (out[16, 30] == EDGE).all()  # the outer wall line
    room = out[30, 24]
    assert (
        room[0] == room[1] == room[2] or abs(room[0] - room[2]) < 20
    )  # grey, not green
    assert (
        np.abs(out[30, 31:34] - room).sum(axis=1) > 20
    ).any()  # a seam between rooms
