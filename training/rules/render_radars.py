"""A top-down radar for every map: Valve's own where the game has one, else drawn.

Maps with a radar in the game files (overwatch/maps/valve_radar.py) get that,
recoloured for the page: it is the radar players know. The rest are drawn, as
below, from where players walked or, failing that, from the collision mesh.

The drawn radars, for maps CS2CD recorded, come from where players actually walked.

Run:  uv run python training/rules/render_radars.py

Writes data/radars/<map>.png and <map>.json (how to place world coordinates on
the image). The web interface draws each kill's sightline on top.

The first attempt drew the collision mesh from above. It was faithful and
unreadable: roofs, props and stair treads buried the layout, and it only existed
for the nine maps with a mesh. A radar's job is to show where people can be, and
the data already says that: every alive player's position, a few dozen matches
per map, marks every corridor anyone walks. Floors are shaded by height (lighter
is higher), and the edge of the walkable area is drawn as the wall line.

A map CS2CD never recorded is drawn from its collision mesh instead: the floor a
player can reach on foot from the spawn points (geometry/walkable.py). Run with
--compare to measure that against the walked radar on maps that have both; on
dust2, mirage and inferno it covers 96-98% of where players went.

Nuke and Vertigo stack one storey over another, so a single image shows B site
and A site on top of each other. A map where players stand a storey apart over
much of its floor is drawn twice, split at the height between the storeys:
<map>.png is the upper one and <map>_lower.png the lower, over the same square.
The same test runs on the reachable floors of a mesh-drawn map.

Radars stay in data/ (never committed): they are drawn from match recordings of
Valve's maps and from Valve's map files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from overwatch.dataset import cs2cd_matches
from overwatch.layers.l2_perception.geometry.walkable import Walkable, walkable
from overwatch.maps import source2
from overwatch.maps.radar import (
    MARGIN,
    Frame,
    mesh_frames,
    save,
    save_valve,
    shading,
    split_height,
    storeys,
    tidy,
)
from overwatch.maps.valve_radar import valve_radar
from overwatch.parsing import load_cs2cd_cached

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_maps import DEFAULT_CS2, VIEWER

ROOT = Path(__file__).resolve().parents[2]

#: The grid positions are counted on (then smoothed up to the image size).
GRID = 512
#: Matches per map, and every n-th tick of each: plenty to cover every corridor.
MATCHES_PER_MAP = 30
TICK_STRIDE = 4
#: A cell counts as walkable once this many samples fell in it.
MIN_SAMPLES = 3


def positions(map_name: str, paths: list[Path]) -> pl.DataFrame:
    """Alive players' positions on this map, from up to MATCHES_PER_MAP matches."""
    frames = []
    for path in paths:
        match = load_cs2cd_cached(path)
        if match.meta.get("map") != map_name:
            continue
        frames.append(
            match.ticks.filter(pl.col("is_alive") & (pl.col("tick") % TICK_STRIDE == 0))
            .select("x", "y", "z")
            .drop_nulls()
        )
        if len(frames) >= MATCHES_PER_MAP:
            break
    return pl.concat(frames) if frames else pl.DataFrame()


class Box(NamedTuple):
    """The square of the world a radar covers: every storey of a map shares it."""

    x_min: float
    y_max: float
    side: float


def box_around(where: pl.DataFrame) -> Box:
    x0, x1 = where["x"].quantile(0.001), where["x"].quantile(0.999)
    y0, y1 = where["y"].quantile(0.001), where["y"].quantile(0.999)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN)
    return Box((x0 + x1) / 2 - side / 2, (y0 + y1) / 2 + side / 2, side)


def from_positions(where: pl.DataFrame, box: Box | None = None) -> Frame:
    """Walkable where many alive players stood, over a few dozen matches.

    `box` fixes the patch of the world drawn (the whole map's, for one storey).
    """
    x_min, y_max, side = box or box_around(where)
    x, y, z = (where[c].to_numpy() for c in ("x", "y", "z"))
    col = np.clip(((x - x_min) / side * GRID).astype(int), 0, GRID - 1)
    row = np.clip(((y_max - y) / side * GRID).astype(int), 0, GRID - 1)
    count = np.zeros((GRID, GRID))
    np.add.at(count, (row, col), 1)
    height_sum = np.zeros((GRID, GRID))
    np.add.at(height_sum, (row, col), z)
    height = np.where(count > 0, height_sum / np.maximum(count, 1), np.nan)
    z_lo, z_hi = np.quantile(z, [0.02, 0.98])
    return Frame(
        tidy(count >= MIN_SAMPLES), shading(height, z_lo, z_hi), x_min, y_max, side
    )


def mesh_walkable(map_name: str, cs2: Path, work: Path) -> Walkable:
    vpk = cs2 / f"{map_name}.vpk"
    glb = work / "maps" / map_name / "world_physics_physics.glb"
    if not glb.exists():
        glb = source2.export_physics(vpk, work, exe=VIEWER)
    seeds = source2.spawns(vpk, work, exe=VIEWER)
    if not len(seeds):
        msg = f"{map_name}: no spawn points in its entities"
        raise ValueError(msg)
    return walkable(source2.movement_mesh(glb), seeds)


def overlap(a: Frame, b: Walkable) -> dict:
    """How well the mesh's walkable area matches where players actually went."""
    n = a.walkable.shape[0]
    centres = (np.arange(n) + 0.5) * a.side / n
    xs, ys = np.meshgrid(a.x_min + centres, a.y_max - centres)
    rows = ((b.y_max - ys) // b.cell).astype(int)
    cols = ((xs - b.x_min) // b.cell).astype(int)
    inside = (
        (rows >= 0)
        & (rows < b.reach.shape[0])
        & (cols >= 0)
        & (cols < b.reach.shape[1])
    )
    mesh = np.zeros_like(a.walkable)
    mesh[inside] = tidy(b.reach)[rows[inside], cols[inside]]
    walked = a.walkable
    return {
        "overlap (IoU)": (mesh & walked).sum() / max((mesh | walked).sum(), 1),
        "walked area the mesh covers": (mesh & walked).sum() / max(walked.sum(), 1),
        "mesh area anyone walked": (mesh & walked).sum() / max(mesh.sum(), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=ROOT / "data" / "raw" / "cs2cd")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "radars")
    parser.add_argument("--maps", nargs="+", default=None)
    parser.add_argument("--cs2", type=Path, default=DEFAULT_CS2)
    parser.add_argument(
        "--work", type=Path, default=ROOT / "data" / "maps" / "extracted"
    )
    parser.add_argument(
        "--source",
        choices=("auto", "valve", "positions", "mesh"),
        default="auto",
        help="auto: Valve's radar where the game has one, else positions where "
        "CS2CD has the map, else the mesh",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="for maps with both, measure how the mesh's area matches the walked one",
    )
    args = parser.parse_args()

    paths = cs2cd_matches(args.demos)
    recorded = sorted({load_cs2cd_cached(p).meta.get("map") for p in paths} - {None})
    meshed = sorted(p.stem for p in (ROOT / "data" / "maps").glob("*.tri"))
    maps = args.maps or sorted(set(recorded) | set(meshed))
    args.out.mkdir(parents=True, exist_ok=True)

    for map_name in maps:
        started = time.perf_counter()
        use_mesh = args.source == "mesh" or (
            args.source == "auto" and map_name not in recorded
        )
        try:
            if args.source in ("auto", "valve") and not args.compare:
                found = valve_radar(map_name, args.cs2, args.work, exe=VIEWER)
                if found is not None:
                    images, where = found
                    save_valve(map_name, images, where, args.out)
                    floors = " with a lower floor" if "_lower" in images else ""
                    print(f"{map_name}: Valve's radar{floors}", flush=True)
                    continue
                if args.source == "valve":
                    print(f"{map_name}: the game has no radar for it", flush=True)
                    continue
            if args.compare:
                walked = from_positions(positions(map_name, paths))
                scores = overlap(walked, mesh_walkable(map_name, args.cs2, args.work))
                print(
                    f"{map_name}: "
                    + ", ".join(f"{k} {v:.0%}" for k, v in scores.items()),
                    flush=True,
                )
                continue
            if use_mesh:
                whole, source = mesh_walkable(map_name, args.cs2, args.work), "mesh"
                frames, split, share = mesh_frames(whole)
            else:
                where = positions(map_name, paths)
                if where.height < 10_000:
                    print(f"{map_name}: too few positions for a radar, skipped")
                    continue
                source, box = "positions", box_around(where)
                split, share = split_height(*(where[c].to_numpy() for c in "xyz"))
                frames = [
                    (
                        suffix,
                        from_positions(
                            where.filter(
                                pl.col("z").is_between(low, high, closed="left")
                            ),
                            box,
                        ),
                    )
                    for suffix, low, high in storeys(split)
                ]
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"{map_name}: no radar ({exc})", flush=True)
            continue
        save(map_name, frames, source, split, args.out)
        levels = f", split at z={split:.0f}" if split is not None else ""
        print(
            f"{map_name}: drawn from {source}, {share:.0%} stacked{levels} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
