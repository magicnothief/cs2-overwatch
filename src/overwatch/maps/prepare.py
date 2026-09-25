"""Make a map ready the first time a demo on it arrives: its mesh and its radar.

Neither can be shipped with the app: both are drawn from Valve's map files. So
they are made on the user's machine, from the user's own CS2 install, once per
map, and kept:

    <home>/data/maps/<map>.tri       line of sight (Layer 2 and the features)
    <home>/data/radars/<map>.*       the radar the web page draws kills on

It takes about a minute for a large map. Without CS2 (not installed, not found,
a workshop map it does not have) the analysis still runs: visibility falls back
to the game's spotting flag and the report says so.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from overwatch import paths
from overwatch.downloads import Progress
from overwatch.maps import radar, source2
from overwatch.maps.steam import find_cs2_maps

_NAME = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class Prepared:
    """What a map has now, and why not, when it lacks something."""

    mesh: bool
    radar: bool
    note: str | None = None


def prepare_map(
    map_name: str,
    *,
    cs2: str | Path | None = None,
    maps: Path = paths.MAPS,
    radars: Path = paths.RADARS,
    work: Path = paths.WORK,
    tools: Path = paths.TOOLS,
    progress: Progress | None = None,
) -> Prepared:
    """Make <map>.tri and the radar if they are missing; never raises for a map.

    Args:
        cs2: CS2's folder (or its maps folder), when Steam's libraries do not
            list it; found automatically otherwise.
    """
    tell = progress or (lambda _message: None)
    need_mesh = not (maps / f"{map_name}.tri").exists()
    need_radar = not (radars / f"{map_name}.json").exists()
    if not (need_mesh or need_radar):
        return Prepared(True, True)
    have = Prepared(not need_mesh, not need_radar)
    if not _NAME.match(map_name):
        return Prepared(
            have.mesh, have.radar, f"cannot prepare a map named {map_name!r}"
        )

    folder = find_cs2_maps(cs2)
    if folder is None:
        return Prepared(
            have.mesh,
            have.radar,
            "CS2 was not found on this PC, so there is no map mesh: set its folder "
            "in the settings",
        )
    vpk = folder / f"{map_name}.vpk"
    if not vpk.exists():
        return Prepared(
            have.mesh, have.radar, f"{map_name} is not in the CS2 install at {folder}"
        )

    scratch = work / map_name
    try:
        exe = source2.viewer(tools, tell)
        tell(f"Reading {map_name} from your CS2 (first time only, about a minute)")
        glb = source2.export_physics(vpk, scratch, exe=exe)
        if need_mesh:
            tell(f"Building {map_name}'s line-of-sight mesh")
            source2.write_triangles(
                maps / f"{map_name}.tri", source2.sight_triangles(glb)
            )
        if need_radar:
            tell(f"Drawing {map_name}'s radar")
            from overwatch.layers.l2_perception.geometry.walkable import walkable

            seeds = source2.spawns(vpk, scratch, exe=exe)
            if not len(seeds):
                msg = "no spawn points in its entities"
                raise ValueError(msg)
            frames, split, _ = radar.mesh_frames(
                walkable(source2.movement_mesh(glb), seeds)
            )
            radar.save(map_name, frames, "mesh", split, radars)
    except Exception as exc:  # noqa: BLE001 - one map must never stop an analysis
        return Prepared(
            (maps / f"{map_name}.tri").exists(),
            (radars / f"{map_name}.json").exists(),
            f"could not prepare {map_name}: {type(exc).__name__}: {exc}",
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return Prepared(True, True)


__all__ = ["Prepared", "prepare_map"]
