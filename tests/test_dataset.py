"""Tests for the window dataset builder."""

from pathlib import Path

import pytest

from overwatch.dataset import build_match_windows, cs2cd_matches, label_window

CS2CD_ROOT = Path(__file__).parents[1] / "data" / "raw" / "cs2cd"


def test_label_window() -> None:
    assert label_window("Player_3", ["Player_3"], match_had_cheater=True) == "cheater"
    # another player in a cheater match: CS2CD's "not cheater" label is unreliable
    assert label_window("Player_1", ["Player_3"], match_had_cheater=True) == "unknown"
    assert label_window("Player_1", [], match_had_cheater=False) == "clean"


@pytest.mark.skipif(not CS2CD_ROOT.exists(), reason="CS2CD not downloaded")
def test_build_match_windows_labels_and_ids() -> None:
    path = cs2cd_matches(CS2CD_ROOT)[0]
    windows, window_ticks = build_match_windows(path, pre=32, post=8)

    assert windows.height > 0
    assert set(windows["label"]).issubset({"cheater", "clean", "unknown"})
    assert windows["window_uid"].n_unique() == windows.height
    assert window_ticks.height == windows.height * (32 + 8 + 1)
    # every window tick carries its window's label, for filtering without a join
    assert window_ticks["label"].null_count() == 0
