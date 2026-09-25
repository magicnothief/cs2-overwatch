"""Draw a map's radar: its walkable floor from above, shaded by height.

Two sources feed it. For maps CS2CD recorded, training/rules/render_radars.py
draws from where thousands of players actually walked. For every other map, and
on any machine without CS2CD, the floor comes from the collision mesh instead:
what a player can reach on foot from the spawn points (geometry/walkable.py),
which covers 96-98% of where players went on the maps that have both.

A stacked map (Nuke, Vertigo: one storey over another) is drawn twice, split at
the height between the storeys, as <map>.png (upper) and <map>_lower.png (lower)
over the same square; <map>.json says how to place world coordinates on them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from PIL import Image
from scipy import ndimage

from overwatch.layers.l2_perception.geometry.walkable import Walkable

#: Output image size.
SIZE = 1024
#: Room around the walkable area, as a share of its size.
MARGIN = 0.05

VOID = np.array([214, 221, 228])
LOW = np.array([226, 232, 238])  # lowest floors
HIGH = np.array([255, 255, 255])  # highest floors
EDGE = np.array([96, 106, 117])  # the wall line
#: Walkable specks smaller than this share of the map, and holes smaller than
#: this many grid cells, are noise (a grenade throw spot, a player jumping).
MIN_PIECE = 0.004
MAX_HOLE = 40
#: Stacked maps: places this size where players stand a STOREY apart, over more
#: than STACKED_SHARE of the map, get the map drawn as two radars.
STACK_CELL = 64.0
STOREY = 150.0
STACKED_SHARE = 0.10


class Frame(NamedTuple):
    """A square patch of the world, as a grid: what draw() turns into a radar."""

    walkable: np.ndarray  # (n, n) bool
    shade: np.ndarray  # (n, n) 0..1, lighter is higher
    x_min: float
    y_max: float
    side: float  # world units across


def tidy(walkable: np.ndarray) -> np.ndarray:
    """Close hairline gaps, drop specks, fill pinholes: noise, not layout."""
    walkable = ndimage.binary_closing(walkable, iterations=2)
    pieces, n = ndimage.label(walkable)
    if n:
        sizes = ndimage.sum(walkable, pieces, range(1, n + 1))
        keep = np.flatnonzero(sizes >= MIN_PIECE * sizes.sum()) + 1
        walkable = np.isin(pieces, keep)
    holes, n = ndimage.label(ndimage.binary_fill_holes(walkable) & ~walkable)
    if n:
        sizes = ndimage.sum(np.ones_like(holes), holes, range(1, n + 1))
        walkable |= np.isin(holes, np.flatnonzero(sizes < MAX_HOLE) + 1)
    return walkable


def shading(height: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    """Heights spread into cells that had none of their own, then scaled 0..1."""
    known = ~np.isnan(height)
    weight = ndimage.gaussian_filter(known.astype(float), 2)
    spread = ndimage.gaussian_filter(np.where(known, height, 0), 2) / np.maximum(
        weight, 1e-6
    )
    return np.clip((spread - z_lo) / max(z_hi - z_lo, 1), 0, 1)


def split_height(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    cell: float = STACK_CELL,
    min_samples: int = 30,
) -> tuple[float | None, float]:
    """The height between a map's two storeys (None for one), and how stacked it is.

    A place is stacked when what stands there spans more than a STOREY in
    height; a map is when more than STACKED_SHARE of its floor is (Nuke 27%,
    Vertigo 24%; stairs and catwalks leave every other map at 7% or less). The split
    goes in the gap between the two height clusters in the stacked places
    (Otsu's threshold).

    Positions are pooled over `cell`-sized places, each needing `min_samples`.
    A mesh's floors are exact, so there each grid cell is a place of its own: a
    pooled place would count a ledge beside a floor as a storey above it.
    """
    key = np.floor(x / cell).astype(np.int64) * 1_000_003 + np.floor(y / cell).astype(
        np.int64
    )
    cells = (
        pl.DataFrame({"k": key, "z": z})
        .group_by("k")
        .agg(n=pl.len(), lo=pl.col("z").quantile(0.1), hi=pl.col("z").quantile(0.9))
        .filter(pl.col("n") >= min_samples)
    )
    stacked = cells.filter(pl.col("hi") - pl.col("lo") > STOREY)
    share = stacked.height / max(cells.height, 1)
    if share < STACKED_SHARE:
        return None, share
    heights = z[np.isin(key, stacked["k"].to_numpy())]
    counts, edges = np.histogram(heights, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    below = np.cumsum(counts)
    below_sum = np.cumsum(counts * centres)
    total, total_sum = below[-1], below_sum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (total_sum * below / total - below_sum) ** 2 / (
            below * (total - below)
        )
    # across an empty gap every height splits equally well: take the gap's middle
    best = np.flatnonzero(spread >= np.nanmax(spread) * 0.999)
    return float((centres[best[0]] + centres[best[-1]]) / 2), share


def storeys(split: float | None) -> list[tuple[str, float, float]]:
    """(file suffix, lowest z, highest z) of each radar to draw, the upper first."""
    if split is None:
        return [("", -np.inf, np.inf)]
    return [("", split, np.inf), ("_lower", -np.inf, split)]


def floor_points(found: Walkable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World x, y, z of every reachable floor cell."""
    row, col, z = found.floors.T
    return (
        found.x_min + (col + 0.5) * found.cell,
        found.y_max - (row + 0.5) * found.cell,
        z,
    )


def from_mesh(found: Walkable, whole: Walkable | None = None) -> Frame:
    """Walkable where the mesh lets a player reach on foot from a spawn.

    `whole` fixes the patch of the world drawn (the whole map's, for one storey).
    """
    whole = whole or found
    rows, cols = np.nonzero(whole.reach)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    n = int(max(r1 - r0, c1 - c0) * (1 + 2 * MARGIN)) + 1
    top = (r0 + r1) // 2 - n // 2
    left = (c0 + c1) // 2 - n // 2

    def crop(grid: np.ndarray, fill: float) -> np.ndarray:
        out = np.full((n, n), fill, dtype=float)
        rs = slice(max(top, 0), min(top + n, grid.shape[0]))
        cs = slice(max(left, 0), min(left + n, grid.shape[1]))
        out[rs.start - top : rs.stop - top, cs.start - left : cs.stop - left] = grid[
            rs, cs
        ]
        return out

    walkable = crop(found.reach.astype(float), 0) > 0.5
    height = crop(found.height, np.nan)
    z_lo, z_hi = np.nanquantile(found.height, [0.02, 0.98])
    return Frame(
        tidy(walkable),
        shading(height, z_lo, z_hi),
        found.x_min + left * found.cell,
        found.y_max - top * found.cell,
        n * found.cell,
    )


def draw(frame: Frame) -> Image.Image:
    """The radar: floors shaded by height, the walkable edge drawn as the wall line."""
    zoom = SIZE / frame.walkable.shape[0]
    # upscale, blur away the grid's staircase, then cut the edge sharp again
    soft = (
        ndimage.gaussian_filter(
            ndimage.zoom(frame.walkable.astype(float), zoom, order=1), 1.5
        )
        > 0.5
    )
    shade = ndimage.zoom(frame.shade, zoom, order=1)[:SIZE, :SIZE]
    soft = soft[:SIZE, :SIZE]
    edge = soft & ~ndimage.binary_erosion(soft, iterations=2)
    pixels = np.empty((SIZE, SIZE, 3))
    pixels[:] = VOID
    floor = LOW + (HIGH - LOW) * shade[..., None]
    pixels[soft] = floor[soft]
    pixels[edge] = EDGE
    return Image.fromarray(pixels.astype(np.uint8))


def meta(map_name: str, frame: Frame, source: str, split: float | None = None) -> dict:
    """How to place world coordinates on the radar; a stacked map names both images.

    Positions at or above `split_z` belong on the upper radar (`<map>.png`), the
    rest on the lower one (`<map>_lower.png`); both cover the same square.
    """
    out = {
        "map": map_name,
        "size": SIZE,
        "x_min": float(frame.x_min),
        "y_max": float(frame.y_max),
        "units_per_pixel": float(frame.side / SIZE),
        "source": source,
    }
    if split is not None:
        out["levels"] = {
            "split_z": round(split, 1),
            "upper": map_name,
            "lower": f"{map_name}_lower",
        }
    return out


#: Valve radars, recoloured: tunnels and lower areas are a dimmer floor, the rest
#: runs up to white; Valve's seams between areas stay as faint lines.
DEEP = np.array([228, 233, 238])
SEAM = np.array([170, 179, 188])


def restyle(image: Image.Image) -> Image.Image:
    """Valve's coloured radar in the page's greys, on the same void as ours.

    Brightness becomes the floor tone (Valve draws lower and covered areas
    darker), a change of colour becomes a thin seam (the edges of rooms, ramps
    and boxes), Valve's thin dark outlines stay dark, and the playable area's
    outer edge is drawn as the wall line, as on the radars drawn from meshes.
    """
    a = np.asarray(image.convert("RGBA"), dtype=float)
    rgb, alpha = a[..., :3], a[..., 3] / 255
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    inside = alpha > 0.5
    if not inside.any():
        msg = "the radar image is empty"
        raise ValueError(msg)
    lo, hi = np.percentile(lum[inside], [2, 98])
    t = np.clip((lum - lo) / max(hi - lo, 1), 0, 1) ** 0.7
    out = DEEP + (HIGH - DEEP) * t[..., None]
    change = np.max(
        [
            np.hypot(
                ndimage.sobel(ndimage.gaussian_filter(rgb[..., c], 0.6), 0),
                ndimage.sobel(ndimage.gaussian_filter(rgb[..., c], 0.6), 1),
            )
            for c in range(3)
        ],
        axis=0,
    )
    out[(change > 90) & inside] = SEAM
    # Valve's outlines are thin dark lines; a dark *area* (a tunnel, the floor
    # below) is floor, and keeps its dimmer tone
    dark = (lum < lo + 0.18 * (hi - lo)) & inside
    out[dark & ~ndimage.binary_opening(dark, iterations=2)] = EDGE
    solid = ndimage.binary_opening(inside, iterations=1)
    out[solid & ~ndimage.binary_erosion(solid, iterations=2)] = EDGE
    mix = alpha[..., None]
    return Image.fromarray((VOID * (1 - mix) + out * mix).astype(np.uint8))


def save_valve(
    map_name: str, images: dict[str, Image.Image], overview, out: Path
) -> None:
    """Write Valve's radar images, recoloured, and <map>.json to place them."""
    out.mkdir(parents=True, exist_ok=True)
    size = images[""].width
    for suffix, image in images.items():
        restyle(image).save(out / f"{map_name}{suffix}.png", optimize=True)
    stale = out / f"{map_name}_lower.png"
    if "_lower" not in images and stale.exists():
        stale.unlink()
    meta = {
        "map": map_name,
        "size": size,
        "x_min": overview.x_min,
        "y_max": overview.y_max,
        "units_per_pixel": overview.scale * 1024 / size,
        "source": "valve",
    }
    if "_lower" in images:
        meta["levels"] = {
            "split_z": overview.split_z,
            "upper": map_name,
            "lower": f"{map_name}_lower",
        }
    part = out / f"{map_name}.json.part"
    part.write_text(json.dumps(meta, indent=2) + "\n")
    part.replace(out / f"{map_name}.json")


def mesh_frames(whole: Walkable) -> tuple[list[tuple[str, Frame]], float | None, float]:
    """A mesh-drawn map's radars, one per storey: ([(suffix, frame)], split, share)."""
    split, share = split_height(*floor_points(whole), cell=whole.cell, min_samples=1)
    frames = [
        (suffix, from_mesh(whole.between(low, high), whole))
        for suffix, low, high in storeys(split)
    ]
    return frames, split, share


def save(
    map_name: str,
    frames: list[tuple[str, Frame]],
    source: str,
    split: float | None,
    out: Path,
) -> None:
    """Write the radar images and <map>.json; the .json last, as the sign it is done."""
    out.mkdir(parents=True, exist_ok=True)
    for suffix, frame in frames:
        draw(frame).save(out / f"{map_name}{suffix}.png", optimize=True)
    stale = out / f"{map_name}_lower.png"
    if split is None and stale.exists():
        stale.unlink()
    part = out / f"{map_name}.json.part"
    part.write_text(
        json.dumps(meta(map_name, frames[0][1], source, split), indent=2) + "\n"
    )
    part.replace(out / f"{map_name}.json")


__all__ = [
    "MARGIN",
    "SIZE",
    "Frame",
    "draw",
    "floor_points",
    "from_mesh",
    "mesh_frames",
    "meta",
    "restyle",
    "save",
    "save_valve",
    "shading",
    "split_height",
    "storeys",
    "tidy",
]
