"""Tests for the walkable area worked out from a collision mesh.

A small map built from boxes, where the right answer is obvious: a floor split by
a tall wall, a low step and a high ledge, and a ceiling too low to stand under.
"""

import numpy as np
import pytest
import trimesh

pytest.importorskip("embreex")

from overwatch.layers.l2_perception.geometry.walkable import walkable


def _box(x0, x1, y0, y1, z0, z1) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    box.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return box


@pytest.fixture(scope="module")
def area():
    parts = [
        _box(-512, 512, -512, 512, -16, 0),  # the ground
        _box(-8, 8, -512, 512, 0, 256),  # a wall splitting it at x=0
        _box(-400, -300, 100, 200, 0, 12),  # a step up: walkable
        _box(-400, -300, -200, -100, 0, 100),  # a ledge too high to climb
        _box(-250, -150, -400, -300, 40, 50),  # a ceiling 40 units up
    ]
    mesh = trimesh.util.concatenate(parts)
    seeds = np.array([[-200.0, 0.0, 0.0]])  # a spawn on the left half
    return walkable(mesh, seeds)


def _at(area, x: float, y: float) -> bool:
    row = int((area.y_max - y) // area.cell)
    col = int((x - area.x_min) // area.cell)
    return bool(area.reach[row, col])


def test_the_spawn_side_is_walkable(area) -> None:
    assert _at(area, -200, 0)
    assert _at(area, -450, 450)


def test_a_wall_keeps_the_other_side_out(area) -> None:
    assert not _at(area, 200, 0)
    assert not _at(area, 450, -450)


def test_a_low_step_is_walked_up_but_a_high_ledge_is_not(area) -> None:
    assert _at(area, -350, 150)
    assert area.height[
        int((area.y_max - 150) // area.cell), int((-350 - area.x_min) // area.cell)
    ] == pytest.approx(12)
    assert (
        not _at(area, -350, -150)
        or area.height[
            int((area.y_max + 150) // area.cell), int((-350 - area.x_min) // area.cell)
        ]
        < 50
    )


def test_no_standing_under_a_low_ceiling(area) -> None:
    # the floor under the slab has 40 units of room; the slab's top is unreachable
    assert not _at(area, -200, -350)


def test_spawn_origins_are_read_in_both_formats() -> None:
    from overwatch.maps.source2 import spawn_points

    text = """====1====
classname                      "info_player_terrorist"
origin                         [ -822.5, -795.25, 150.75 ]
====2====
classname                      "info_player_counterterrorist"
origin                         "110.000000 1212.000000 98.430054"
====3====
classname                      "prop_physics_multiplayer"
origin                         [ 1.0, 2.0, 3.0 ]
"""
    points = spawn_points(text)
    assert points.tolist() == [[-822.5, -795.25, 150.75], [110.0, 1212.0, 98.430054]]


def test_one_storey_can_be_sliced_out(area) -> None:
    step = int((area.y_max - 150) // area.cell), int((-350 - area.x_min) // area.cell)
    ground = int((area.y_max - 0) // area.cell), int((-200 - area.x_min) // area.cell)
    upper, lower = area.between(6), area.between(high=6)
    assert upper.reach[step] and not upper.reach[ground]
    assert lower.reach[ground] and not lower.reach[step]
    assert upper.reach.shape == lower.reach.shape == area.reach.shape
    assert len(upper.floors) + len(lower.floors) == len(area.floors)
