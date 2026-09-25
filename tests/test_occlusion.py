"""Tests for line-of-sight ray casting.

A wall between two players must block the view; open space must not. Both .tri
encodings have to load, because reading hex text as binary produces a mesh the
size of a galaxy and then nothing ever blocks anything.
"""

import numpy as np
import pytest

from overwatch.layers.l2_perception.geometry import occlusion


def _wall_tri_bytes() -> bytes:
    """Two triangles forming a wall at x = 0, spanning y and z."""
    wall = np.array(
        [
            [[0, -500, -500], [0, 500, -500], [0, 500, 500]],
            [[0, -500, -500], [0, 500, 500], [0, -500, 500]],
        ],
        dtype=np.float32,
    )
    return wall.tobytes()


@pytest.fixture
def wall_map(tmp_path):
    path = tmp_path / "de_test.tri"
    path.write_bytes(_wall_tri_bytes())
    return tmp_path


def test_wall_blocks_the_view(wall_map) -> None:
    eyes = np.array([[-200.0, 0.0, 0.0]])
    targets = np.array([[200.0, 0.0, 0.0]])  # straight through the wall
    assert not occlusion.line_of_sight(eyes, targets, "de_test", map_dir=wall_map)[0]


def test_open_space_is_visible(wall_map) -> None:
    eyes = np.array([[-200.0, 0.0, 0.0]])
    targets = np.array([[-100.0, 0.0, 0.0]])  # both on the same side
    assert occlusion.line_of_sight(eyes, targets, "de_test", map_dir=wall_map)[0]


def test_hex_text_and_binary_load_identically(tmp_path) -> None:
    binary = tmp_path / "bin.tri"
    binary.write_bytes(_wall_tri_bytes())
    hex_text = tmp_path / "hex.tri"
    hex_text.write_text(" ".join(f"{b:02x}" for b in _wall_tri_bytes()))

    assert np.array_equal(
        occlusion.read_triangles(binary), occlusion.read_triangles(hex_text)
    )
    assert occlusion.read_triangles(hex_text).shape == (2, 3, 3)


def test_empty_input_returns_empty(wall_map) -> None:
    assert occlusion.line_of_sight(
        np.zeros((0, 3)), np.zeros((0, 3)), "de_test", map_dir=wall_map
    ).shape == (0,)
