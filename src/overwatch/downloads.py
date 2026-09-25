"""Fetch a pinned file over HTTPS, and refuse it unless its SHA-256 matches.

Everything the app downloads at run time goes through here: the judge engine, the
Source2Viewer tool, the models. Each is pinned to one version with the checksum
published for it, so a changed or tampered file is rejected, not run.

Connections drop, and the judge is 2.8 GB: a dropped download is retried, and
picks up where it stopped (an HTTP range request) rather than starting over. A
partial file survives even a failed run, so the next start resumes it too.
"""

from __future__ import annotations

import hashlib
import http.client
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

Progress = Callable[[str], None]

#: Tries per file, with a growing pause between them.
ATTEMPTS = 5
#: Network failures worth another try.
TRANSIENT = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
    http.client.IncompleteRead,
)


def fetch(url: str, dest: Path, sha256: str, progress: Progress | None = None) -> Path:
    """Download to dest, checking the SHA-256; a file that fails it is never kept."""
    tell = progress or (lambda _message: None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _download(url, part, dest.name, tell)
            break
        except TRANSIENT as exc:
            if attempt == ATTEMPTS:
                raise
            tell(
                f"The connection dropped ({exc}); trying again ({attempt}/{ATTEMPTS - 1})"
            )
            time.sleep(2**attempt)
    if _sha256(part) != sha256:
        part.unlink(missing_ok=True)
        msg = f"{dest.name} failed its checksum; not using it"
        raise RuntimeError(msg)
    part.replace(dest)
    return dest


def _download(url: str, part: Path, name: str, tell: Progress) -> None:
    """Fetch into part, continuing from what is already there when the server can."""
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={have}-"} if have else {}
    )
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # nothing left to fetch: the part is whole
            return
        raise
    with response:
        if have and getattr(response, "status", None) != 206:
            have = 0  # the server sent the whole file: start over
        total = have + int(response.headers.get("Content-Length") or 0)
        done, last = have, 0.0
        with part.open("ab" if have else "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total and time.monotonic() - last > 1:
                    last = time.monotonic()
                    tell(f"Downloading {name}: {done * 100 // total}%")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


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


__all__ = ["ATTEMPTS", "Progress", "fetch", "unpack"]
