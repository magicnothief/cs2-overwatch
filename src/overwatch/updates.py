"""Is there a newer release, and how to install it.

Releases are GitHub releases of the code repository, tagged vX.Y.Z; the release
workflow attaches the built package (.whl) to each. The app asks GitHub for the
latest one at most once a day (<home>/data/update.json caches the answer), and
only when the "updates" setting is on; any failure (offline, rate-limited)
means "no news", never an error.

Updating is the install command again: it fetches the latest release's package.
`overwatch update` runs it where a program may replace itself (Linux); on
Windows a running program's file is locked, so it prints the command instead.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from overwatch import paths

REPO = "magicnothief/cs2-overwatch"
LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
CACHE = paths.DATA / "update.json"
#: How long a check's answer is trusted.
MAX_AGE = 24 * 3600
INSTALL = {
    "windows": (
        'powershell -ExecutionPolicy ByPass -c "irm '
        f'https://raw.githubusercontent.com/{REPO}/master/install.ps1 | iex"'
    ),
    "other": (
        f"curl -LsSf https://raw.githubusercontent.com/{REPO}/master/install.sh | sh"
    ),
}


@dataclass(frozen=True)
class Release:
    version: str  # "0.3.0"
    url: str  # the release page, with its notes
    wheel: str | None  # the package to install, when the release has one


def installed_version() -> str:
    try:
        return importlib.metadata.version("cs2-overwatch")
    except importlib.metadata.PackageNotFoundError:
        return "0"


def _parts(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version)[:3])


def newer(candidate: str, current: str) -> bool:
    return _parts(candidate) > _parts(current)


def latest_release(timeout: float = 5.0) -> Release | None:
    """The newest published release, from GitHub; None when that cannot be told."""
    request = urllib.request.Request(
        LATEST, headers={"Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
    except (OSError, ValueError):
        return None
    tag = str(data.get("tag_name") or "")
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", tag):
        return None
    wheel = next(
        (
            a.get("browser_download_url")
            for a in data.get("assets", [])
            if str(a.get("name", "")).endswith(".whl")
        ),
        None,
    )
    return Release(tag.lstrip("v"), str(data.get("html_url") or ""), wheel)


def check(cache: Path = CACHE, *, now: float | None = None) -> Release | None:
    """A release newer than this one, if there is one; asks GitHub once a day."""
    now = time.time() if now is None else now
    try:
        cached = json.loads(cache.read_text())
        fresh = now - float(cached["checked"]) < MAX_AGE
    except (OSError, ValueError, KeyError, TypeError):
        cached, fresh = None, False
    if fresh:
        release = Release(**cached["release"]) if cached.get("release") else None
    else:
        release = latest_release()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(
                    {"checked": now, "release": asdict(release) if release else None}
                )
            )
        except OSError:
            pass
    if release and newer(release.version, installed_version()):
        return release
    return None


def update_command() -> str:
    return INSTALL["windows" if sys.platform == "win32" else "other"]


def run_update() -> int:
    """Install the latest release over this one, or say how where that cannot work."""
    release = latest_release()
    if release is None:
        print(
            "No release found: GitHub could not be reached, or nothing is released "
            "yet. Try again later."
        )
        return 1
    if not newer(release.version, installed_version()):
        print(f"Overwatch review {installed_version()} is the latest release.")
        return 0
    print(f"Updating {installed_version()} -> {release.version} ({release.url})")
    uv = shutil.which("uv")
    if sys.platform == "win32" or uv is None or release.wheel is None:
        print("Close the app, then run:\n\n  " + update_command())
        return 0
    source = release.wheel
    return subprocess.run(
        [uv, "tool", "install", "--python", "3.12", "--force", source], check=False
    ).returncode


__all__ = [
    "Release",
    "check",
    "installed_version",
    "latest_release",
    "newer",
    "run_update",
    "update_command",
]
