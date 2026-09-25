"""Valve's own radar for a map, read from the user's CS2 and recoloured for the page.

CS2 ships a hand-drawn radar for most official maps (the one players know from
the game), a second image for the lower floor of stacked maps, and an overview
file that places them in the world:

    panorama/images/overheadmaps/<map>_radar_psd.vtex_c        (in pak01)
    panorama/images/overheadmaps/<map>_lower_radar_psd.vtex_c  (stacked maps)
    resource/overviews/<map>.txt    pos_x, pos_y: the image's top-left corner
                                    scale: world units per pixel at 1024 px
                                    verticalsections: where the lower floor begins

Like the meshes, they are read on the user's machine and never shipped. The
colours are Valve's; the page keeps team colours for players, so the image is
recoloured into its greys (radar.restyle): light floors, dimmer tunnels and
lower areas, Valve's seams between areas and its outlines kept as lines.

Newer maps (Eldorado, Thera, ...) have no radar in the game files; they keep the
radar drawn from where players can walk (radar.mesh_frames).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from overwatch.maps import source2

OVERHEAD = "panorama/images/overheadmaps"


@dataclass(frozen=True)
class Overview:
    x_min: float  # world x of the image's left edge
    y_max: float  # world y of its top edge
    scale: float  # world units per pixel, for a 1024 px image
    split_z: float | None  # the lower image covers everything below this


def parse_overview(text: str) -> Overview:
    """The numbers that place a radar image, from Valve's KeyValues overview file."""
    text = re.sub(r"//[^\n]*", "", text)  # comments sit between keys and braces

    def number(key: str) -> float:
        found = re.search(rf'"{key}"\s+"(-?[\d.]+)"', text)
        if found is None:
            msg = f"overview has no {key}"
            raise ValueError(msg)
        return float(found.group(1))

    split = None
    lower = re.search(r'"lower"\s*\{([^}]*)\}', text)
    if lower:
        top = re.search(r'"AltitudeMax"\s+"(-?[\d.]+)"', lower.group(1))
        split = float(top.group(1)) if top else None
    return Overview(number("pos_x"), number("pos_y"), number("scale"), split)


def _extract(archive: Path, files: list[str], out: Path, exe: Path) -> None:
    source2.run(exe, archive, out, ",".join(files))


def valve_radar(
    map_name: str, maps_dir: Path, work: Path, *, exe: Path
) -> tuple[dict[str, Image.Image], Overview] | None:
    """The map's radar images by file suffix ("" and "_lower"), and where they sit.

    None when the game has no radar for this map. Looks in the map's own VPK
    first, then in the game's main archive, where the official maps keep theirs.
    """
    game = maps_dir.parent
    wanted = [
        f"{OVERHEAD}/{map_name}_radar_psd.vtex_c",
        f"{OVERHEAD}/{map_name}_lower_radar_psd.vtex_c",
        f"{OVERHEAD}/{map_name}_radar_tga.vtex_c",
        f"resource/overviews/{map_name}.txt",
    ]
    scratch = work / f"{map_name}-radar"
    try:
        for archive in (maps_dir / f"{map_name}.vpk", game / "pak01_dir.vpk"):
            if not archive.exists():
                continue
            _extract(archive, wanted, scratch, exe)
            overview = scratch / "resource" / "overviews" / f"{map_name}.txt"
            images = scratch / OVERHEAD
            upper = next(
                (
                    p
                    for p in (
                        images / f"{map_name}_radar_psd.png",
                        images / f"{map_name}_radar_tga.png",
                    )
                    if p.exists()
                ),
                None,
            )
            if upper is None or not overview.exists():
                continue
            where = parse_overview(overview.read_text(errors="replace"))
            found = {"": Image.open(upper).copy()}
            lower = images / f"{map_name}_lower_radar_psd.png"
            if lower.exists() and where.split_z is not None:
                found["_lower"] = Image.open(lower).copy()
            return found, where
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


__all__ = ["Overview", "parse_overview", "valve_radar"]
