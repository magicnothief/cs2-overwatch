"""Find the maps folder of the CS2 installed on this machine.

Steam keeps a list of its library folders (steamapps/libraryfolders.vdf); CS2 is
in one of them, under steamapps/common/Counter-Strike Global Offensive. Steam's
own folder is found from the registry on Windows and from the usual places on
Linux (native, Flatpak, Snap).

A library Steam does not list (the other OS's drive on a dual-boot machine, say)
is not found this way, so a folder can also be given: the OVERWATCH_CS2
environment variable or the "cs2" setting, pointing at the CS2 folder, its
game/csgo/maps folder, or anything in between.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

GAME = Path("steamapps", "common", "Counter-Strike Global Offensive")
MAPS = Path("game", "csgo", "maps")
#: Any real CS2 install has this map.
PROBE = "de_dust2.vpk"


def maps_folder(given: str | Path) -> Path | None:
    """The maps folder inside a folder someone pointed at, if it is one."""
    base = Path(given).expanduser()
    for candidate in (
        base,
        base / "maps",
        base / "csgo" / "maps",
        base / MAPS,
        base / GAME / MAPS,
    ):
        if (candidate / PROBE).exists():
            return candidate
    return None


def steam_roots() -> Iterator[Path]:
    """Where Steam itself may be installed."""
    if os.name == "nt":
        yield from _registry_roots()
        yield Path(
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Steam"
        )
        return
    home = Path.home()
    yield from (
        home / ".steam" / "steam",
        home / ".local" / "share" / "Steam",
        home
        / ".var"
        / "app"
        / "com.valvesoftware.Steam"
        / ".local"
        / "share"
        / "Steam",
        home / "snap" / "steam" / "common" / ".local" / "share" / "Steam",
    )


def _registry_roots() -> Iterator[Path]:  # pragma: no cover - Windows only
    import winreg

    for hive, key, value in (
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
    ):
        try:
            with winreg.OpenKey(hive, key) as handle:
                yield Path(winreg.QueryValueEx(handle, value)[0])
        except OSError:
            continue


def library_folders(root: Path) -> list[Path]:
    """Every library a Steam install lists, itself included."""
    folders = [root]
    for vdf in (
        root / "steamapps" / "libraryfolders.vdf",
        root / "config" / "libraryfolders.vdf",
    ):
        try:
            text = vdf.read_text(errors="replace")
        except OSError:
            continue
        for raw in re.findall(r'"path"\s+"([^"]+)"', text):
            folders.append(Path(raw.replace("\\\\", "\\")))
    return folders


def find_cs2_maps(setting: str | Path | None = None) -> Path | None:
    """The CS2 maps folder: from the setting if given, else from Steam's libraries."""
    for given in (setting, os.environ.get("OVERWATCH_CS2")):
        if given:
            return maps_folder(given)
    seen: set[Path] = set()
    for root in steam_roots():
        for library in library_folders(root):
            if library in seen:
                continue
            seen.add(library)
            found = maps_folder(library / GAME / MAPS)
            if found is not None:
                return found
    return None


__all__ = ["find_cs2_maps", "library_folders", "maps_folder", "steam_roots"]
