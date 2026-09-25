"""Where the app keeps its files: one folder, laid out the same way everywhere.

    <home>/models/     the detector and the judge (downloaded on first run)
    <home>/data/maps/  collision meshes, drawn from the user's own CS2 install
    <home>/data/radars/, reports/, uploads/, engines/, tools/

In a source checkout <home> is the checkout itself, so everything stays where it
always was (data/ and models/ beside src/). Installed as a tool, it is the
platform's per-user data folder (~/.local/share/cs2-overwatch on Linux,
%LOCALAPPDATA%\\cs2-overwatch on Windows). OVERWATCH_HOME overrides both.
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP = "cs2-overwatch"


def _checkout() -> Path | None:
    """The source checkout this code runs from, if it runs from one."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").exists() and (root / "src" / "overwatch").is_dir():
        return root
    return None


def home() -> Path:
    if env := os.environ.get("OVERWATCH_HOME"):
        return Path(env).expanduser().resolve()
    return _checkout() or Path(platformdirs.user_data_dir(APP, appauthor=False))


HOME = home()
#: Running from a source checkout, on its own files: models there are the
#: developer's (perhaps just retrained) and are never replaced by downloads.
IN_CHECKOUT = _checkout() == HOME
DATA = HOME / "data"
MODELS = HOME / "models"
MAPS = DATA / "maps"
RADARS = DATA / "radars"
REPORTS = DATA / "reports"
UPLOADS = DATA / "uploads"
ENGINES = DATA / "engines"
TOOLS = DATA / "tools"
#: Scratch space for map extraction; emptied after each map.
WORK = DATA / "maps" / "extracted"

__all__ = [
    "DATA",
    "ENGINES",
    "HOME",
    "IN_CHECKOUT",
    "MAPS",
    "MODELS",
    "RADARS",
    "REPORTS",
    "TOOLS",
    "UPLOADS",
    "WORK",
    "home",
]
