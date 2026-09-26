"""Tests for finding demos on this PC and opening packed ones.

What must never happen: a request reaching a file outside the listed folders.
"""

import bz2
import gzip
from pathlib import Path

import pytest

from overwatch import demos

DEMO = demos.DEMO_MAGIC + b"rest of a demo"


@pytest.fixture
def folders(tmp_path: Path) -> dict[str, Path]:
    replays, downloads = tmp_path / "replays", tmp_path / "Downloads"
    replays.mkdir()
    downloads.mkdir()
    (replays / "match730_1.dem").write_bytes(DEMO)
    (downloads / "faceit.dem.gz").write_bytes(gzip.compress(DEMO))
    (downloads / "notes.txt").write_text("not a demo")
    (tmp_path / "secret.dem").write_bytes(DEMO)
    return {"cs2": replays, "downloads": downloads}


def test_only_demos_are_listed(folders: dict[str, Path]) -> None:
    names = {d.name for d in demos.list_demos(folders)}
    assert names == {"match730_1.dem", "faceit.dem.gz"}


def test_nothing_outside_a_listed_folder_resolves(folders: dict[str, Path]) -> None:
    assert demos.resolve(folders, "cs2", "match730_1.dem") is not None
    for key, name in [
        ("cs2", "../secret.dem"),
        ("downloads", "notes.txt"),
        ("elsewhere", "match730_1.dem"),
        ("cs2", "missing.dem"),
        ("cs2", str(folders["cs2"].parent / "secret.dem")),
    ]:
        assert demos.resolve(folders, key, name) is None, (key, name)


def test_a_symlink_out_of_the_folder_does_not_resolve(folders: dict[str, Path]) -> None:
    link = folders["downloads"] / "sneaky.dem"
    link.symlink_to(folders["cs2"].parent / "secret.dem")
    assert demos.resolve(folders, "downloads", "sneaky.dem") is None


@pytest.mark.parametrize("pack", [gzip.compress, bz2.compress])
def test_packed_demos_are_unpacked_and_checked(tmp_path: Path, pack) -> None:
    packed = tmp_path / "in.dem.x"
    packed.write_bytes(pack(DEMO))
    out = demos.unpack(packed, tmp_path / "out.dem")
    assert out.read_bytes() == DEMO


def test_zstandard_demos_are_unpacked(tmp_path: Path) -> None:
    zstandard = pytest.importorskip("zstandard")
    packed = tmp_path / "in.dem.zst"
    packed.write_bytes(zstandard.ZstdCompressor().compress(DEMO))
    assert demos.unpack(packed, tmp_path / "out.dem").read_bytes() == DEMO


def test_anything_else_is_refused(tmp_path: Path) -> None:
    for content in (gzip.compress(b"HL2DEMO\x00 a CS:GO demo"), b"just text"):
        bad = tmp_path / "bad.dem.gz"
        bad.write_bytes(content)
        with pytest.raises(ValueError, match="not a CS2 demo"):
            demos.unpack(bad, tmp_path / "out.dem")
        assert not (tmp_path / "out.dem").exists()
