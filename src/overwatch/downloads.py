"""Fetch a pinned file over HTTPS, and refuse it unless its SHA-256 matches.

Everything the app downloads at run time goes through here: the judge engine, the
Source2Viewer tool, the models. Each is pinned to one version with the checksum
published for it, so a changed or tampered file is rejected, not run.
"""

from __future__ import annotations

import hashlib
import tarfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

Progress = Callable[[str], None]


def fetch(url: str, dest: Path, sha256: str, progress: Progress | None = None) -> Path:
    """Download to dest, checking the SHA-256; a partial file is never kept."""
    tell = progress or (lambda _message: None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total and time.monotonic() - last > 1:
                last = time.monotonic()
                tell(f"Downloading {dest.name}: {done * 100 // total}%")
    if digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        msg = f"{dest.name} failed its checksum; not using it"
        raise RuntimeError(msg)
    part.replace(dest)
    return dest


def unpack(archive: Path, into: Path) -> None:
    """Extract a .zip or .tar.gz, refusing paths that would land outside `into`."""
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            root = into.resolve()
            for name in z.namelist():
                if not (into / name).resolve().is_relative_to(root):
                    msg = f"{archive.name} holds a path outside its folder: {name}"
                    raise RuntimeError(msg)
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into, filter="data")


__all__ = ["Progress", "fetch", "unpack"]
