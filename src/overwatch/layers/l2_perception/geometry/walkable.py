"""Where a player could walk, worked out from a map's collision mesh alone.

The radars for CS2CD maps are drawn from where players actually walked. A map
nobody has recorded has no such positions, so its walkable area is computed the
way a player moves through it:

    floors     rays cast straight down through every grid cell find each surface
               stacked there; one facing up with room to stand above it is floor
    moves      between neighbouring cells, a step up of at most STEP units or any
               drop down, if a ray at waist height between them hits no wall
    reach      everything connected to a spawn point through those moves

The mesh used here must include player clips: they are invisible, so they do not
block sight (occlusion leaves them out), but they are exactly what stops a
player walking off the map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order

#: Grid spacing, in Hammer units (a player is 32 units wide).
CELL = 16.0
#: The highest step a player walks up without jumping.
STEP = 18.0
#: Room needed above a floor to stand there (crouching needs 54).
HEADROOM = 60.0
#: Height of the ray that checks for walls between two cells.
WAIST = 36.0
#: A surface this flat, facing up, is floor.
FLOOR_NZ = 0.7
#: How far from any spawn a playable area can extend (a large map is ~6000 across).
REACH = 6000.0


@dataclass(frozen=True)
class Walkable:
    """Reachable floor on a regular grid; row 0 is the top (highest y)."""

    x_min: float
    y_max: float
    cell: float
    reach: np.ndarray  # (rows, cols) bool: some reachable floor in this cell
    height: np.ndarray  # (rows, cols) float: the highest reachable floor, else nan
    floors: np.ndarray  # (k, 3): row, col, z of every reachable floor

    def between(self, low: float = -np.inf, high: float = np.inf) -> Walkable:
        """Only the floors with low <= z < high: one storey of a stacked map."""
        keep = (self.floors[:, 2] >= low) & (self.floors[:, 2] < high)
        return _gridded(
            self.x_min, self.y_max, self.cell, self.reach.shape, self.floors[keep]
        )


def _gridded(
    x_min: float, y_max: float, cell: float, shape: tuple[int, int], floors: np.ndarray
) -> Walkable:
    reach = np.zeros(shape, dtype=bool)
    height = np.full(shape, np.nan)
    for r, c, h in floors:
        r, c = int(r), int(c)
        reach[r, c] = True
        height[r, c] = h if np.isnan(height[r, c]) else max(height[r, c], h)
    return Walkable(x_min, y_max, cell, reach, height, floors)


def _intersector(mesh: trimesh.Trimesh):
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    return RayMeshIntersector(mesh)


def floors(
    mesh: trimesh.Trimesh,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    cell: float = CELL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Every standable floor in the box: (row, col, z) per floor, and the grid shape."""
    cols = int(np.ceil((x1 - x0) / cell))
    rows = int(np.ceil((y1 - y0) / cell))
    cx = x0 + (np.arange(cols) + 0.5) * cell
    cy = y1 - (np.arange(rows) + 0.5) * cell
    gx, gy = np.meshgrid(cx, cy)
    top = mesh.bounds[1][2] + 100.0
    origins = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, top)])
    down = np.tile([0.0, 0.0, -1.0], (len(origins), 1))

    hits, rays, faces = _intersector(mesh).intersects_location(
        origins, down, multiple_hits=True
    )
    if not len(rays):
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0), (rows, cols)
    z = hits[:, 2]
    facing_up = mesh.face_normals[faces][:, 2] > FLOOR_NZ

    # per ray, surfaces from the top down; a floor needs clearance to the one above
    order = np.lexsort((-z, rays))
    rays, z, facing_up = rays[order], z[order], facing_up[order]
    first = np.r_[True, rays[1:] != rays[:-1]]
    above = np.r_[np.inf, z[:-1]]
    above[first] = np.inf
    standable = facing_up & (above - z >= HEADROOM)
    rays, z = rays[standable], z[standable]
    return rays // cols, rays % cols, z, (rows, cols)


def flood(
    mesh: trimesh.Trimesh,
    row: np.ndarray,
    col: np.ndarray,
    z: np.ndarray,
    seeds: np.ndarray,
    *,
    x0: float,
    y1: float,
    cell: float = CELL,
) -> np.ndarray:
    """Which floors can be walked to from the seed points (x, y, z)."""
    n = len(z)
    if not n or not len(seeds):
        return np.zeros(n, dtype=bool)
    by_cell: dict[tuple[int, int], list[int]] = {}
    for i, (r, c) in enumerate(zip(row.tolist(), col.tolist(), strict=True)):
        by_cell.setdefault((r, c), []).append(i)

    # candidate moves: to the floor in each neighbouring cell nearest in height
    src, dst = [], []
    for i in range(n):
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            options = by_cell.get((row[i] + dr, col[i] + dc))
            if not options:
                continue
            j = min(options, key=lambda k: abs(z[k] - z[i]))
            if z[j] - z[i] <= STEP:  # any drop is fine; a climb must be a step
                src.append(i)
                dst.append(j)
    src_a, dst_a = np.array(src), np.array(dst)

    def centre(idx: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                x0 + (col[idx] + 0.5) * cell,
                y1 - (row[idx] + 0.5) * cell,
                z[idx] + WAIST,
            ]
        )

    # a move is open unless a wall (or player clip) is in the way at waist height
    start, end = centre(src_a), centre(dst_a)
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    open_ = np.ones(len(src_a), dtype=bool)
    intersector = _intersector(mesh)
    for chunk in range(0, len(src_a), 200_000):
        part = slice(chunk, chunk + 200_000)
        hit, ray, _ = intersector.intersects_location(
            start[part], delta[part] / length[part, None], multiple_hits=False
        )
        if len(ray):
            gap = np.linalg.norm(hit - start[part][ray], axis=1)
            blocked = np.zeros(len(start[part]), dtype=bool)
            blocked[ray] = gap < length[part][ray]
            open_[part] &= ~blocked

    # the seeds' floors, joined to one extra start node so a single search covers all
    starts = []
    for sx, sy, sz in seeds:
        options = by_cell.get((int((y1 - sy) // cell), int((sx - x0) // cell)))
        if options:
            nearest = min(options, key=lambda k: abs(z[k] - sz))
            if abs(z[nearest] - sz) < 72:
                starts.append(nearest)
    if not starts:
        return np.zeros(n, dtype=bool)

    # directed: a drop goes one way, so what is reachable is searched, not grouped
    heads = np.r_[src_a[open_], np.full(len(starts), n)]
    tails = np.r_[dst_a[open_], np.array(starts)]
    graph = sparse.csr_matrix(
        (np.ones(len(heads)), (heads, tails)), shape=(n + 1, n + 1)
    )
    order = breadth_first_order(graph, n, directed=True, return_predecessors=False)
    reached = np.zeros(n + 1, dtype=bool)
    reached[order] = True
    return reached[:n]


def walkable(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    *,
    cell: float = CELL,
    margin: float = 256.0,
) -> Walkable:
    """The floor reachable on foot from the seeds, over the mesh's extent.

    The search is limited to REACH units around the seeds: some meshes carry
    kilometres of scenery around the playable area, and casting rays over all of
    it costs minutes for nothing.
    """
    (x0, y0, _), (x1, y1, _) = mesh.bounds
    (sx0, sy0, _), (sx1, sy1, _) = seeds.min(axis=0), seeds.max(axis=0)
    x0, y0 = max(x0, sx0 - REACH) - margin, max(y0, sy0 - REACH) - margin
    x1, y1 = min(x1, sx1 + REACH) + margin, min(y1, sy1 + REACH) + margin
    row, col, z, shape = floors(mesh, x0, x1, y0, y1, cell)
    reached = flood(mesh, row, col, z, seeds, x0=x0, y1=y1, cell=cell)
    kept = np.column_stack([row[reached], col[reached], z[reached]]).astype(float)
    return _gridded(x0, y1, cell, shape, kept)


__all__ = ["CELL", "HEADROOM", "STEP", "Walkable", "flood", "floors", "walkable"]
