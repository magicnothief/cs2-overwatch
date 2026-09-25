"""True line of sight: is a wall between these two players?

The game's own spotting flag turned out to be unreliable in a way that matters:
at the moment of a kill — when line of sight is guaranteed — it reports "visible"
for only 79% of clean rifle kills and 50% of clean sniper kills. Since cheaters in
CS2CD take 60% of their kills with snipers and clean players 15%, that bias leaks
the label, and a model trained on it partly learns "this player uses an AWP".

So visibility is computed here instead, by casting a ray from the shooter's eye to
the target's head against the map's collision mesh. The answer does not depend on
the weapon, the distance or the game's spotting rules.

Speed matters: a full dataset is ~13M rays. awpy's pure-Python BVH does 75 rays a
second (47 hours); Embree through trimesh does ~27,000 (8 minutes), which is why
this module batches rays through trimesh rather than calling awpy per tick.

Map meshes are `.tri` files: nine float32 per triangle. They come in two flavours
in the wild — raw binary, and the same bytes written as ASCII hex — so the loader
sniffs which one it has. Mistaking hex text for binary yields a mesh spanning 1e23
units, and then every ray misses and everything looks visible, which is how this
was found.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from overwatch import paths
from overwatch.layers.l2_perception.geometry.angles import EYE_HEIGHT, HEAD_HEIGHT

DEFAULT_MAP_DIR = paths.MAPS

#: Nothing is occluded closer than this; avoids self-hits on the shooter's own box.
NEAR_CLIP = 8.0


def read_triangles(path: str | Path) -> np.ndarray:
    """Read a `.tri` file into an (n, 3, 3) array of triangle corners."""
    raw = Path(path).read_bytes()
    sample = raw[:64]
    is_hex_text = all(byte in b"0123456789abcdefABCDEF \r\n" for byte in sample)
    if is_hex_text:
        raw = bytes.fromhex("".join(raw.decode("ascii", errors="ignore").split()))
    floats = np.frombuffer(raw, dtype=np.float32)
    usable = (floats.size // 9) * 9
    return floats[:usable].reshape(-1, 3, 3).astype(np.float32)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a `.tri` collision mesh."""
    triangles = read_triangles(path)
    return trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.shape[0] * 3).reshape(-1, 3),
        process=False,
    )


@lru_cache(maxsize=4)
def _intersector(path: str) -> trimesh.ray.ray_pyembree.RayMeshIntersector:
    """Build (and keep) the ray accelerator for one map."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    return RayMeshIntersector(load_mesh(path))


def map_path(map_name: str, map_dir: str | Path = DEFAULT_MAP_DIR) -> Path:
    return Path(map_dir) / f"{map_name}.tri"


def has_map(map_name: str, map_dir: str | Path = DEFAULT_MAP_DIR) -> bool:
    return map_path(map_name, map_dir).exists()


def line_of_sight(
    eyes: np.ndarray,
    targets: np.ndarray,
    map_name: str,
    *,
    map_dir: str | Path = DEFAULT_MAP_DIR,
    eye_height: float = EYE_HEIGHT,
    head_height: float = HEAD_HEIGHT,
) -> np.ndarray:
    """Which of these shooter -> target pairs have an unobstructed view?

    Args:
        eyes: (n, 3) shooter origins (player positions, not eye positions).
        targets: (n, 3) target origins.
        map_name: e.g. "de_mirage"; its mesh must exist in map_dir.
        eye_height / head_height: offsets added to the two positions.

    Returns:
        (n,) bool array, True where nothing blocks the segment.
    """
    if len(eyes) == 0:
        return np.zeros(0, dtype=bool)

    origins = eyes.astype(np.float64).copy()
    origins[:, 2] += eye_height
    ends = targets.astype(np.float64).copy()
    ends[:, 2] += head_height

    delta = ends - origins
    distance = np.linalg.norm(delta, axis=1)
    visible = np.zeros(len(origins), dtype=bool)

    usable = distance > NEAR_CLIP
    if not usable.any():
        return ~np.isnan(distance)  # degenerate: same spot, treat as visible

    directions = np.zeros_like(delta)
    directions[usable] = delta[usable] / distance[usable, None]

    intersector = _intersector(str(map_path(map_name, map_dir)))
    locations, index_ray, _ = intersector.intersects_location(
        origins[usable], directions[usable], multiple_hits=False
    )

    # a hit only blocks the view if it happens *before* the target
    blocked = np.zeros(int(usable.sum()), dtype=bool)
    if len(index_ray):
        hit_distance = np.linalg.norm(locations - origins[usable][index_ray], axis=1)
        target_distance = distance[usable][index_ray]
        blocked[index_ray] = hit_distance < target_distance - NEAR_CLIP

    visible[usable] = ~blocked
    visible[~usable] = True
    return visible
