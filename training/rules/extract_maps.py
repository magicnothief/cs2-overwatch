"""Extract collision meshes from an installed CS2, and check each one against demos.

Run:  uv run python training/rules/extract_maps.py --maps de_dust2 de_mirage ...

Source 2 ships map collision inside `world_physics.vmdl_c` in each map's VPK.
Source2Viewer exports it as glTF, which is in metres and Y-up, while demos are in
Hammer units and Z-up. Rather than trusting a guessed axis convention, every
candidate transform is *scored against the demos themselves*: a kill with no
wallbang and no smoke must have had line of sight, so the right transform is the
one where those kills come out visible.

A mesh that cannot reach the threshold is not written, because a silently wrong
mesh would poison every visibility feature downstream.

Two things the physics mesh contains that must be dropped first: **clip volumes**
(player clips, grenade clips) block movement but not sight, and the **skybox** is
not a wall. Leaving them in costs nine points of accuracy on de_mirage.

That test alone only punishes *over*-occlusion, and it was not enough: a mirrored
mesh sits beside the playable area rather than around it, blocks nothing, and so
explained 100% of the kills. de_overpass and de_vertigo shipped that way, and every
visibility feature on those maps read "visible". So a second test runs alongside
it: random pairs of opposing players, most of whom have a wall between them at any
given moment. The transform must make both come out right.

To re-check meshes already written, without the game installed:

    uv run python training/rules/extract_maps.py --retest

Maps CS2CD never recorded are checked against your own demos (every .dem under
data/raw by default, or --dem). A map with no demos at all can still be written
with --trust-convention: every validated map turned out to use the same axis
convention, so it is applied as is, and meshes.json marks the map unvalidated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import trimesh

from overwatch.aim import TICK_RATE, kill_windows
from overwatch.dataset import CHEATER_DIR, CLEAN_DIR
from overwatch.layers.l2_perception.geometry.occlusion import (
    EYE_HEIGHT,
    HEAD_HEIGHT,
    NEAR_CLIP,
    read_triangles,
)
from overwatch.maps.source2 import export_physics, solid_geometry, to_hammer
from overwatch.maps.steam import find_cs2_maps
from overwatch.parsing import ParsedMatch, load_cs2cd_cached
from overwatch.parsing.demo import parse_demo

ROOT = Path(__file__).resolve().parents[2]
#: CS2's maps folder: found through Steam, else the Windows drive it lives on here.
DEFAULT_CS2 = find_cs2_maps() or Path(
    "/run/media/magicnothief/E/SteamLibrary/steamapps/common/"
    "Counter-Strike Global Offensive/game/csgo/maps"
)
VIEWER = ROOT / "tools" / "Source2Viewer-CLI"

#: A transform is only accepted if it explains at least this share of the kills
#: that are known to have had line of sight. The remainder is honest error:
#: crouching, lag compensation, and props modelled as convex hulls.
MIN_ACCURACY = 0.82

#: On a real map almost every pair of opposing players is behind cover at any
#: moment: the correct meshes block 98-99.7% of random pairs. A misplaced mesh can
#: block almost none, so anything under this is rejected however well it explains
#: the kills.
MIN_BLOCKED = 0.5


def candidate_transforms() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Axis orders and sign flips worth trying.

    glTF from Source 2 is Y-up, so the vertical axis is known; the horizontal pair
    may be swapped and either may be mirrored, which is four possibilities, and the
    remaining orders are included in case an exporter changes.
    """
    orders = [(0, 2, 1), (2, 0, 1)]
    signs = list(itertools.product((1, -1), (1, -1), (1,)))
    return [(order, sign) for order in orders for sign in signs]


def matches_on(
    map_name: str, root: Path, dems: dict[str, list[ParsedMatch]]
) -> Iterator[ParsedMatch]:
    """Every recorded match on this map: CS2CD first, then the given .dem files."""
    for folder in (CLEAN_DIR, CHEATER_DIR):
        for path in sorted((root / folder).glob("*.parquet")):
            match = load_cs2cd_cached(path)
            if match.meta.get("map") == map_name:
                yield match
    yield from dems.get(map_name, [])


def parse_dems(paths: list[Path]) -> dict[str, list[ParsedMatch]]:
    """Your own demos, by map, for maps CS2CD never recorded (de_eldorado, ...)."""
    by_map: dict[str, list[ParsedMatch]] = {}
    for path in paths:
        match = parse_demo(path, hash_file=False)
        by_map.setdefault(match.meta.get("map"), []).append(match)
    return by_map


def sample_kills(matches: Iterable[ParsedMatch], limit: int) -> pl.DataFrame:
    """Kills on one map that must have had line of sight: no wallbang, no smoke."""
    frames = []
    for match in matches:
        windows, ticks = kill_windows(match.ticks, match.deaths, pre=1, post=0)
        if windows.is_empty():
            continue
        victim = match.ticks.select(
            victim_id=pl.col("player_id"),
            tick=pl.col("tick"),
            vx=pl.col("x"),
            vy=pl.col("y"),
            vz=pl.col("z"),
        )
        frame = (
            ticks.filter(pl.col("tick_offset") == 0)
            .join(
                windows.select(
                    "window_id",
                    "victim_id",
                    *[c for c in ("penetrated", "thrusmoke") if c in windows.columns],
                ),
                on="window_id",
            )
            .join(victim, on=["victim_id", "tick"], how="inner")
        )
        if "penetrated" in frame.columns:
            frame = frame.filter(pl.col("penetrated").fill_null(0) == 0)
        if "thrusmoke" in frame.columns:
            frame = frame.filter(~pl.col("thrusmoke").fill_null(value=False))
        frames.append(frame.select("x", "y", "z", "vx", "vy", "vz"))
        if sum(f.height for f in frames) >= limit:
            break
    return pl.concat(frames).head(limit) if frames else pl.DataFrame()


def sample_pairs(
    matches: Iterable[ParsedMatch], limit: int, *, seed: int = 0
) -> pl.DataFrame:
    """Random pairs of living opponents on one map, at random moments.

    Unlike the kills, these carry no guarantee either way — but most of them are
    blocked on any real map, which is exactly what an empty mesh gets wrong.
    """
    frames = []
    for match in matches:
        alive = match.ticks.filter(
            pl.col("is_alive") & (pl.col("tick") % TICK_RATE == 0)
        ).select("tick", "team", "x", "y", "z")
        pairs = (
            alive.join(alive, on="tick", suffix="_v")
            .filter(pl.col("team") != pl.col("team_v"))
            .select("x", "y", "z", vx="x_v", vy="y_v", vz="z_v")
            .drop_nulls()
        )
        frames.append(pairs.sample(min(pairs.height, limit), seed=seed))
        if sum(f.height for f in frames) >= limit * 3:
            break
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sample(
        min(limit, sum(f.height for f in frames)), seed=seed
    )


def sight(
    mesh_vertices: np.ndarray, faces: np.ndarray, pairs: pl.DataFrame
) -> np.ndarray:
    """Which eye -> head segments this mesh leaves unobstructed."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=faces, process=False)
    intersector = RayMeshIntersector(mesh)

    origins = pairs.select("x", "y", "z").to_numpy().astype(np.float64)
    origins[:, 2] += EYE_HEIGHT
    ends = pairs.select("vx", "vy", "vz").to_numpy().astype(np.float64)
    ends[:, 2] += HEAD_HEIGHT

    delta = ends - origins
    distance = np.linalg.norm(delta, axis=1)
    keep = distance > NEAR_CLIP
    visible = np.ones(len(origins), dtype=bool)
    if not keep.any():
        return visible

    directions = delta[keep] / distance[keep, None]
    locations, index_ray, _ = intersector.intersects_location(
        origins[keep], directions, multiple_hits=False
    )
    blocked = np.zeros(int(keep.sum()), dtype=bool)
    if len(index_ray):
        hit = np.linalg.norm(locations - origins[keep][index_ray], axis=1)
        blocked[index_ray] = hit < distance[keep][index_ray] - NEAR_CLIP
    visible[keep] = ~blocked
    return visible


class Scored(NamedTuple):
    name: str
    vertices: np.ndarray
    accuracy: float  # guaranteed-sight kills that come out visible
    blocked: float  # random opposing pairs that come out blocked
    validated: bool = True  # False: no demos to check, the known convention used

    @property
    def acceptable(self) -> bool:
        if not self.validated:
            return True
        return self.accuracy >= MIN_ACCURACY and self.blocked >= MIN_BLOCKED


def choose_transform(
    candidates: list[tuple[str, np.ndarray]],
    faces: np.ndarray,
    kills: pl.DataFrame,
    pairs: pl.DataFrame,
) -> list[Scored]:
    """Score every candidate on both tests, best first.

    Ranked by the mean of the two, so a transform cannot win by being right about
    only one direction: the kills reward seeing, the random pairs reward blocking.
    """
    scored = [
        Scored(
            name,
            vertices,
            float(sight(vertices, faces, kills).mean()),
            float(1 - sight(vertices, faces, pairs).mean()),
        )
        for name, vertices in candidates
    ]
    return sorted(scored, key=lambda s: -(s.accuracy + s.blocked))


def gltf_candidates(vertices: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Every axis convention worth trying on a fresh glTF export."""
    return [
        (f"{list(order)}{list(signs)}", to_hammer(vertices, order, signs))
        for order, signs in candidate_transforms()
    ]


#: The name horizontal_symmetries gives the mesh as it already is.
IDENTITY = "x+1 y+1"


def horizontal_symmetries(vertices: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """A mesh already in Hammer units, and its seven swapped or mirrored twins.

    Swapping and mirroring the horizontal axes is a group, so starting from any
    one of them reaches all eight: a mesh written with the wrong transform can be
    corrected without re-exporting it from the game.
    """
    out = []
    for swap in (False, True):
        for sx, sy in itertools.product((1, -1), (1, -1)):
            columns = [1, 0, 2] if swap else [0, 1, 2]
            name = ("swap " if swap else "") + f"x{sx:+d} y{sy:+d}"
            out.append((name, vertices[:, columns] * np.array([sx, sy, 1])))
    return out


#: The glTF -> Hammer transform all nine validated maps turned out to share.
CONVENTION = "[2, 0, 1][1, 1, 1]"


def write_mesh(out: Path, map_name: str, triangles: np.ndarray) -> None:
    (out / f"{map_name}.tri").write_bytes(triangles.astype(np.float32).tobytes())


def check(
    map_name: str,
    candidates: list[tuple[str, np.ndarray]],
    faces: np.ndarray,
    matches: list[ParsedMatch],
    n_kills: int,
    *,
    trust_convention: bool = False,
) -> Scored | None:
    """Pick and report the transform for one map; None if nothing passes."""
    started = time.perf_counter()
    kills = sample_kills(matches, n_kills)
    pairs = sample_pairs(matches, n_kills)
    if kills.is_empty() or pairs.is_empty():
        convention = dict(candidates).get(CONVENTION)
        if trust_convention and convention is not None:
            print(
                f"{map_name}: no demos to check against; written with the axis "
                f"convention every validated map shares, marked unvalidated",
                flush=True,
            )
            return Scored(CONVENTION, convention, float("nan"), float("nan"), False)
        print(f"{map_name}: no demos to check against, skipped", flush=True)
        return None

    ranked = choose_transform(candidates, faces, kills, pairs)
    best = ranked[0]
    status = "ok" if best.acceptable else "REJECTED"
    print(
        f"{map_name}: {len(faces):,} faces, best {best.name}: "
        f"{best.accuracy:.1%} of sighted kills visible, "
        f"{best.blocked:.1%} of random pairs blocked -> {status} "
        f"({time.perf_counter() - started:.0f}s)",
        flush=True,
    )
    for other in ranked[1:3]:
        print(
            f"    runner-up {other.name}: {other.accuracy:.1%} visible, "
            f"{other.blocked:.1%} blocked",
            flush=True,
        )
    return best if best.acceptable else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cs2", type=Path, default=DEFAULT_CS2)
    parser.add_argument("--demos", type=Path, default=ROOT / "data" / "raw" / "cs2cd")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "maps")
    parser.add_argument(
        "--work", type=Path, default=ROOT / "data" / "maps" / "extracted"
    )
    parser.add_argument("--maps", nargs="+", default=None)
    parser.add_argument("--kills", type=int, default=300, help="kills used to check")
    parser.add_argument(
        "--retest",
        action="store_true",
        help="re-check the .tri files already in --out instead of exporting",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        nargs="*",
        default=sorted((ROOT / "data" / "raw").rglob("*.dem")),
        help="your own demos, to check maps CS2CD never recorded",
    )
    parser.add_argument(
        "--trust-convention",
        action="store_true",
        help="write maps with no demos at all using the shared axis convention",
    )
    args = parser.parse_args()
    dems = parse_dems(args.dem)

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "meshes.json"
    # merge: a run over two maps must not erase what is known about the others
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    if args.retest:
        maps = args.maps or sorted(p.stem for p in args.out.glob("*.tri"))
    elif args.maps:
        maps = args.maps
    else:
        parser.error("--maps is required unless --retest is given")

    for map_name in maps:
        if args.retest:
            triangles = read_triangles(args.out / f"{map_name}.tri")
            vertices = triangles.reshape(-1, 3).astype(np.float64)
            faces = np.arange(len(vertices)).reshape(-1, 3)
            candidates = horizontal_symmetries(vertices)
        else:
            vpk = args.cs2 / f"{map_name}.vpk"
            if not vpk.exists():
                print(f"{map_name}: no vpk", flush=True)
                continue
            try:
                merged = solid_geometry(export_physics(vpk, args.work, exe=VIEWER))
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                # one map that will not export must not stop the others
                print(f"{map_name}: export failed ({exc}), skipped", flush=True)
                continue
            faces = np.asarray(merged.faces)
            candidates = gltf_candidates(np.asarray(merged.vertices))

        matches = list(matches_on(map_name, args.demos, dems))
        best = check(
            map_name,
            candidates,
            faces,
            matches,
            args.kills,
            trust_convention=args.trust_convention and not args.retest,
        )
        if best is None:
            if args.retest:
                # a mesh that fails is worse than none: without it the pipeline
                # falls back to the spotting flag, which is biased but not blind
                (args.out / f"{map_name}.tri").unlink()
                report.pop(map_name, None)
                print(f"    removed {map_name}.tri", flush=True)
            continue

        unchanged = args.retest and best.name == IDENTITY
        if not unchanged:
            write_mesh(args.out, map_name, best.vertices[faces])
        entry = report.get(map_name, {}) | {
            "faces": len(faces),
            "validated": best.validated,
            "accuracy": round(best.accuracy, 4) if best.validated else None,
            "blocked": round(best.blocked, 4) if best.validated else None,
            "matches_checked": len(matches),
        }
        if args.retest and not unchanged:
            entry["corrected_by"] = best.name
        elif not args.retest:
            entry["transform"] = best.name
        report[map_name] = entry

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n{len(report)} meshes recorded in {report_path}")


if __name__ == "__main__":
    main()
