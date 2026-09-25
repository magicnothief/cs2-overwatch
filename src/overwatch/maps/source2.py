"""Read a map's collision and spawn points out of CS2's own files.

Source 2 keeps a map's collision in `world_physics.vmdl_c` inside the map's VPK,
and its entities (spawn points among them) in `default_ents.vents_c`.
Source2Viewer's command-line tool (ValveResourceFormat, MIT) exports both; it is
downloaded on first use, pinned to one release and checked against its SHA-256.

The physics mesh comes out as glTF: metres, Y-up. Demos are in Hammer units,
Z-up. Which axes map to which was measured against thousands of kills per map
(training/rules/extract_maps.py, ADR 0007): all 14 validated maps share one
convention, so it is applied as is here.

Two meshes are made from it, because sight and movement are stopped by different
things. Clip brushes and the skybox are invisible walls: they stop players, not
bullets or eyes.

    sight_triangles   what blocks a line of sight (clips and sky dropped)
    movement_mesh     what stops a player (player clips kept, other clips and sky
                      dropped), for working out where people can walk
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from overwatch import paths
from overwatch.downloads import Progress, fetch, unpack

#: The Source2Viewer release every download comes from, and its files.
VIEWER_RELEASE = "20.0"
VIEWER_FILES = {
    "linux": (
        "cli-linux-x64.zip",
        "3e8af47cd6ce52e8068904f2aa1dda23c56a6b96a8310b25090f0711cda76a8a",
    ),
    "windows": (
        "cli-windows-x64.zip",
        "d32ab327b8bbb42a2528866afb03bb582bdb779d0005488da32b90292afd3ff5",
    ),
}
VIEWER_URL = (
    "https://github.com/ValveResourceFormat/ValveResourceFormat/releases/download/"
    f"{VIEWER_RELEASE}/"
)

#: glTF is in metres, Hammer units are inches.
UNITS_PER_METRE = 1.0 / 0.0254
#: glTF -> Hammer: which glTF axis becomes x, y, z, and their signs.
CONVENTION = ((2, 0, 1), (1, 1, 1))
#: Physics parts that stop players but not sight.
NON_VISUAL_PARTS = ("clip", "sky")
#: Entity classes a round starts from: team spawns, and deathmatch/arms race ones.
SPAWN = re.compile(
    r"^info_player_(terrorist|counterterrorist)$|^info_deathmatch_spawn$"
)


def viewer(tools: Path = paths.TOOLS, progress: Progress | None = None) -> Path:
    """Source2Viewer's command-line tool, downloaded the first time."""
    system = platform.system().lower()
    if system not in VIEWER_FILES:
        msg = f"no Source2Viewer download for {platform.system()}"
        raise RuntimeError(msg)
    home = tools / f"source2viewer-{VIEWER_RELEASE}"
    exe = home / (
        "Source2Viewer-CLI.exe" if system == "windows" else "Source2Viewer-CLI"
    )
    if exe.exists():
        return exe
    name, sha256 = VIEWER_FILES[system]
    tell = progress or (lambda _message: None)
    tell("Downloading Source2Viewer (~53 MB, once), to read maps from your CS2")
    archive = fetch(VIEWER_URL + name, tools / name, sha256, tell)
    unpack(archive, home)
    archive.unlink()
    if not exe.exists():
        msg = f"no {exe.name} in the Source2Viewer download"
        raise RuntimeError(msg)
    if os.name != "nt":
        exe.chmod(exe.stat().st_mode | 0o111)
    return exe


def run(exe: Path, vpk: Path, out_dir: Path, file: str, *extra: str) -> str:
    """Decompile the files matching `file` (comma-separated paths) out of a VPK."""
    # Source2Viewer writes nothing, and says nothing, when the folder is missing
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(exe), "-i", str(vpk), "-o", str(out_dir), "-d", "-f", file, *extra],
        check=True,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return (result.stdout + result.stderr).decode(errors="replace").strip()


def export_physics(vpk: Path, out_dir: Path, *, exe: Path | None = None) -> Path:
    """The map's physics mesh, exported as glTF (.glb) into out_dir."""
    said = run(
        exe or viewer(),
        vpk,
        out_dir,
        f"maps/{vpk.stem}/world_physics.vmdl_c",
        "--gltf_export_format",
        "glb",
    )
    mesh = out_dir / "maps" / vpk.stem / "world_physics_physics.glb"
    if not mesh.exists():
        msg = f"no physics mesh exported for {vpk.stem}: {said[-300:]}"
        raise FileNotFoundError(msg)
    return mesh


def spawns(vpk: Path, out_dir: Path, *, exe: Path | None = None) -> np.ndarray:
    """Every spawn point in the map's entities: (x, y, z) per spawn."""
    entities = f"maps/{vpk.stem}/entities/default_ents.vents_c"
    run(exe or viewer(), vpk, out_dir, entities)
    text = (out_dir / entities.removesuffix("_c")).read_text(errors="replace")
    return spawn_points(text)


def spawn_points(text: str) -> np.ndarray:
    """Spawn origins from a decompiled entity lump.

    Origins come two ways depending on how old the map is: a list,
    `[ -822.3, -795.6, 150.7 ]`, or a string, `"110.0 1212.0 98.4"`.
    """
    points = []
    for block in re.split(r"^====\d+====$", text, flags=re.MULTILINE):
        kind = re.search(r'^classname\s+"([^"]+)"', block, re.MULTILINE)
        where = re.search(r'^origin\s+(?:\[([^\]]+)\]|"([^"]+)")', block, re.MULTILINE)
        if kind and where and SPAWN.search(kind.group(1)):
            numbers = (where.group(1) or where.group(2)).replace(",", " ").split()
            points.append([float(v) for v in numbers])
    return np.array(points)


def to_hammer(
    vertices: np.ndarray,
    order: tuple[int, ...] = CONVENTION[0],
    signs: tuple[int, ...] = CONVENTION[1],
) -> np.ndarray:
    """glTF vertices (metres, Y-up) in Hammer units, Z-up."""
    return vertices[:, list(order)] * np.array(signs) * UNITS_PER_METRE


def _parts(glb: Path, keep) -> trimesh.Trimesh:
    scene = trimesh.load(glb)
    parts = []
    for node in scene.graph.nodes_geometry:
        if not keep(node.lower()):
            continue
        transform, name = scene.graph[node]
        part = scene.geometry[name].copy()
        part.apply_transform(transform)  # nodes carry their own placement
        parts.append(part)
    return trimesh.util.concatenate(parts)


def solid_geometry(glb: Path) -> trimesh.Trimesh:
    """Everything in the physics mesh that blocks a line of sight, still in glTF axes.

    Node names carry the brush type: `physics_npcclip_playerclip`,
    `physics_csgo_grenadeclip`, `physics_sky`. Those are invisible in game.
    """
    return _parts(glb, lambda name: not any(w in name for w in NON_VISUAL_PARTS))


def sight_triangles(glb: Path) -> np.ndarray:
    """The line-of-sight mesh in Hammer units, as (n, 3, 3) float32 triangles."""
    merged = solid_geometry(glb)
    vertices = to_hammer(np.asarray(merged.vertices))
    return vertices[np.asarray(merged.faces)].astype(np.float32)


def movement_mesh(glb: Path) -> trimesh.Trimesh:
    """Everything that stops a player, including the invisible player clips."""
    merged = _parts(
        glb,
        lambda name: "sky" not in name and ("clip" not in name or "playerclip" in name),
    )
    return trimesh.Trimesh(
        vertices=to_hammer(np.asarray(merged.vertices)),
        faces=merged.faces,
        process=False,
    )


def write_triangles(path: Path, triangles: np.ndarray) -> None:
    """A .tri file (nine float32 per triangle), written so a crash leaves none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".tri.part")
    part.write_bytes(triangles.astype(np.float32).tobytes())
    part.replace(path)


__all__ = [
    "CONVENTION",
    "NON_VISUAL_PARTS",
    "UNITS_PER_METRE",
    "export_physics",
    "movement_mesh",
    "run",
    "sight_triangles",
    "solid_geometry",
    "spawn_points",
    "spawns",
    "to_hammer",
    "viewer",
    "write_triangles",
]
