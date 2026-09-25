"""Tests for finding CS2 and making a map ready, without CS2 or a network.

What must never happen: a missing or oddly installed CS2 stopping an analysis.
Every way of not finding the game ends in a note on the report, not an error.
"""

from pathlib import Path

import numpy as np
import pytest

from overwatch.maps import Prepared, prepare_map, source2, steam


def _install(root: Path) -> Path:
    """A fake CS2 install: the folder layout and one map file."""
    maps = root / steam.GAME / steam.MAPS
    maps.mkdir(parents=True)
    (maps / steam.PROBE).write_bytes(b"")
    return maps


def test_a_cs2_folder_is_found_from_anywhere_along_its_path(tmp_path: Path) -> None:
    maps = _install(tmp_path / "SteamLibrary")
    for given in (
        tmp_path / "SteamLibrary",
        tmp_path / "SteamLibrary" / steam.GAME,
        tmp_path / "SteamLibrary" / steam.GAME / "game" / "csgo",
        maps,
    ):
        assert steam.maps_folder(given) == maps
    assert steam.maps_folder(tmp_path) is None


def test_steams_library_list_is_read_on_both_systems(tmp_path: Path) -> None:
    root = tmp_path / "Steam"
    (root / "steamapps").mkdir(parents=True)
    (root / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"/home/me/.local/share/Steam"\n'
        '\t}\n\t"1"\n\t{\n\t\t"path"\t\t"D:\\\\SteamLibrary"\n\t\t"apps"\n\t\t{\n'
        '\t\t\t"730"\t\t"1"\n\t\t}\n\t}\n}\n'
    )
    assert steam.library_folders(root) == [
        root,
        Path("/home/me/.local/share/Steam"),
        Path("D:\\SteamLibrary"),
    ]


def test_cs2_is_found_in_a_second_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "Steam"
    (root / "steamapps").mkdir(parents=True)
    maps = _install(tmp_path / "Games")
    (root / "steamapps" / "libraryfolders.vdf").write_text(
        f'"libraryfolders" {{ "0" {{ "path" "{root}" }} "1" {{ "path" "{tmp_path / "Games"}" }} }}'
    )
    monkeypatch.delenv("OVERWATCH_CS2", raising=False)
    monkeypatch.setattr(steam, "steam_roots", lambda: iter([root]))
    assert steam.find_cs2_maps() == maps


def test_a_given_folder_wins_and_a_wrong_one_is_not_second_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps = _install(tmp_path / "Elsewhere")
    monkeypatch.setattr(steam, "steam_roots", lambda: iter([]))
    assert steam.find_cs2_maps(tmp_path / "Elsewhere") == maps
    assert steam.find_cs2_maps(tmp_path / "nothing here") is None
    monkeypatch.setenv("OVERWATCH_CS2", str(tmp_path / "Elsewhere"))
    assert steam.find_cs2_maps() == maps


def test_without_cs2_a_map_gets_a_note_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OVERWATCH_CS2", raising=False)
    monkeypatch.setattr(steam, "steam_roots", lambda: iter([]))
    ready = prepare_map("de_dust2", maps=tmp_path, radars=tmp_path)
    assert not ready.mesh and not ready.radar
    assert "CS2 was not found" in ready.note


def test_a_map_the_install_lacks_gets_a_note(tmp_path: Path) -> None:
    _install(tmp_path / "Steam")
    ready = prepare_map(
        "de_workshop_thing", cs2=tmp_path / "Steam", maps=tmp_path, radars=tmp_path
    )
    assert "not in the CS2 install" in ready.note


def test_a_ready_map_needs_nothing(tmp_path: Path) -> None:
    (tmp_path / "de_dust2.tri").write_bytes(b"")
    (tmp_path / "de_dust2.json").write_text("{}")
    ready = prepare_map("de_dust2", cs2="/nowhere", maps=tmp_path, radars=tmp_path)
    assert ready == Prepared(True, True)


def test_odd_map_names_are_refused_before_touching_the_disk(tmp_path: Path) -> None:
    ready = prepare_map("../../etc", maps=tmp_path, radars=tmp_path)
    assert "cannot prepare" in ready.note


def test_gltf_axes_become_hammer_axes() -> None:
    """glTF is Y-up in metres; Hammer is Z-up in inches (ADR 0007's convention)."""
    gltf = np.array([[1.0, 2.0, 3.0]])  # x, y (up), z
    hammer = source2.to_hammer(gltf)
    assert hammer / source2.UNITS_PER_METRE == pytest.approx(
        np.array([[3.0, 1.0, 2.0]])
    )
