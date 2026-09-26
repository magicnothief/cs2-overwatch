"""Demos already on this PC: where CS2 and the browser put them, and opening them.

CS2 saves a match downloaded from Watch → Your Matches into its `replays`
folder; everything else (FACEIT, HLTV, a friend) usually lands in Downloads.
The first page lists both, so a demo can be reviewed where it lies, without an
upload or a copy.

Only files directly inside those folders, with a demo's name, are ever opened:
a request names a folder by key and a file by name, never a path.

Demos often come compressed. Gzip, bzip2 and Zstandard are unpacked to a
temporary .dem first, and every demo, packed or not, must start with CS2's
magic bytes before it is parsed.

A packed demo comes from a stranger, and a compressed file says nothing about
how large it becomes: 131 bytes of bzip2 unpack to 128 MB. So unpacking stops
at MAX_UNPACKED_BYTES, and the magic bytes are checked on the first eight bytes
out of the decompressor rather than on the finished file, so a bomb that is not
a demo at all costs eight bytes instead of a full disk.
"""

from __future__ import annotations

import bz2
import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from overwatch.maps.steam import find_cs2_maps

#: Every CS2 demo starts with these bytes; CS:GO demos ("HL2DEMO") do not parse.
DEMO_MAGIC = b"PBDEMS2\x00"
#: File endings the list shows, packed ones included.
SUFFIXES = (".dem", ".dem.gz", ".dem.bz2", ".dem.zst")
#: Magic bytes of the compressed formats, and how to open each.
PACKED = {
    b"\x1f\x8b": "gz",
    b"BZh": "bz2",
    b"\x28\xb5\x2f\xfd": "zst",
}
#: The list stops here: a Downloads folder can hold years of files.
LIMIT = 40
#: Largest demo ever written out of a packed one. Competitive demos run 50-150
#: MB and tournament ones up to ~500 MB, so anything past this is a
#: decompression bomb rather than a match.
MAX_UNPACKED_BYTES = 2 * 1024**3


class DemoTooLarge(ValueError):
    """A packed demo that unpacks to more than MAX_UNPACKED_BYTES."""


@dataclass(frozen=True)
class LocalDemo:
    folder: str  # "cs2" or "downloads"
    name: str
    size: int
    modified: float


def demo_folders(cs2: str | Path | None = None) -> dict[str, Path]:
    """The folders demos are looked for in, by key; only those that exist."""
    found: dict[str, Path] = {}
    maps = find_cs2_maps(cs2)
    if maps is not None and (maps.parent / "replays").is_dir():
        found["cs2"] = maps.parent / "replays"
    downloads = Path(platformdirs.user_downloads_dir())
    if downloads.is_dir():
        found["downloads"] = downloads
    return found


def is_demo_name(name: str) -> bool:
    return name.lower().endswith(SUFFIXES)


def list_demos(folders: dict[str, Path]) -> list[LocalDemo]:
    """The demos in those folders, newest first."""
    demos = []
    for key, folder in folders.items():
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not is_demo_name(entry.name):
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            if entry.is_file():
                demos.append(LocalDemo(key, entry.name, stat.st_size, stat.st_mtime))
    demos.sort(key=lambda d: d.modified, reverse=True)
    return demos[:LIMIT]


def resolve(folders: dict[str, Path], key: str, name: str) -> Path | None:
    """The file `name` directly inside folder `key`, or None: never another path."""
    folder = folders.get(key)
    if folder is None or not is_demo_name(name) or Path(name).name != name:
        return None
    path = folder / name
    if not path.is_file() or path.resolve().parent != folder.resolve():
        return None
    return path


def packing(head: bytes) -> str | None:
    """Which packing (gz, bz2 or zst) a file's first bytes show; None if none."""
    return next(
        (kind for magic, kind in PACKED.items() if head.startswith(magic)), None
    )


def unpack(source: Path, dest: Path, *, limit: int | None = None) -> Path:
    """A plain .dem at dest from source, packed or not; checked to be a CS2 demo.

    Raises ValueError when the result is not a CS2 demo, and DemoTooLarge when a
    packed one unpacks to more than `limit` bytes (MAX_UNPACKED_BYTES by default).
    """
    limit = MAX_UNPACKED_BYTES if limit is None else limit
    with source.open("rb") as fh:
        head = fh.read(len(DEMO_MAGIC))
    kind = packing(head)
    if kind is None:
        if head != DEMO_MAGIC:
            msg = "That is not a CS2 demo (.dem from CS2)"
            raise ValueError(msg)
        if source != dest:
            shutil.copyfile(source, dest)
    else:
        opener = {"gz": gzip.open, "bz2": bz2.open, "zst": _zstd_open}[kind]
        part = dest.with_suffix(".part")
        try:
            with opener(source, "rb") as packed, part.open("wb") as out:
                _copy_demo(packed, out, limit)
        except BaseException:  # a cut-off, bomb or corrupt archive leaves no .part
            part.unlink(missing_ok=True)
            raise
        part.replace(dest)
    with dest.open("rb") as fh:
        head = fh.read(len(DEMO_MAGIC))
    if head != DEMO_MAGIC:  # closed first: Windows won't delete an open file
        dest.unlink(missing_ok=True)
        msg = "That is not a CS2 demo (.dem from CS2)"
        raise ValueError(msg)
    return dest


def _copy_demo(packed, out, limit: int) -> None:
    """Copy the decompressed stream to `out`, refusing anything that is not a
    CS2 demo or that runs past `limit` bytes."""
    head = packed.read(len(DEMO_MAGIC))
    if head != DEMO_MAGIC:  # not a demo: stop after eight bytes, not a terabyte
        msg = "That is not a CS2 demo (.dem from CS2)"
        raise ValueError(msg)
    out.write(head)
    written = len(head)
    while chunk := packed.read(1 << 20):
        written += len(chunk)
        if written > limit:
            msg = "That file unpacks to more than any CS2 demo"
            raise DemoTooLarge(msg)
        out.write(chunk)


def _zstd_open(path: Path, mode: str = "rb"):
    import zstandard

    return zstandard.open(path, mode)


__all__ = [
    "DEMO_MAGIC",
    "MAX_UNPACKED_BYTES",
    "SUFFIXES",
    "DemoTooLarge",
    "LocalDemo",
    "demo_folders",
    "is_demo_name",
    "list_demos",
    "packing",
    "resolve",
    "unpack",
]
