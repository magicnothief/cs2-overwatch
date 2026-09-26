"""Tests for the update check: never a network call here, never an error there."""

import json
from pathlib import Path

import pytest

from overwatch import updates


def test_versions_compare_as_numbers_not_text() -> None:
    assert updates.newer("0.10.0", "0.9.9")
    assert updates.newer("1.0.0", "0.99.0")
    assert not updates.newer("0.3.0", "0.3.0")
    assert not updates.newer("0.2.9", "0.3.0")


def test_github_is_asked_at_most_once_a_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = []
    release = updates.Release("9.9.9", "https://example.invalid/r", None)

    def latest(timeout: float = 5.0):
        asked.append(timeout)
        return release

    monkeypatch.setattr(updates, "latest_release", latest)
    cache = tmp_path / "update.json"
    assert updates.check(cache, now=1000.0) == release
    assert updates.check(cache, now=1000.0 + 3600) == release  # from the cache
    assert len(asked) == 1
    updates.check(cache, now=1000.0 + updates.MAX_AGE + 1)
    assert len(asked) == 2


def test_no_news_when_github_cannot_be_reached_or_this_is_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(updates, "latest_release", lambda timeout=5.0: None)
    assert updates.check(tmp_path / "a.json", now=0) is None
    same = updates.Release(updates.installed_version(), "", None)
    monkeypatch.setattr(updates, "latest_release", lambda timeout=5.0: same)
    assert updates.check(tmp_path / "b.json", now=0) is None


def test_a_broken_cache_is_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "update.json"
    cache.write_text("{not json")
    monkeypatch.setattr(updates, "latest_release", lambda timeout=5.0: None)
    assert updates.check(cache, now=5.0) is None
    assert json.loads(cache.read_text()) == {"checked": 5.0, "release": None}


def _answer(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """GitHub's reply to the release check, without a network call."""

    class Response:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(
        updates.urllib.request, "urlopen", lambda *a, **k: Response()
    )


def test_a_release_page_that_is_not_on_github_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The page goes into an href; a javascript: URI there would run in the app.
    for bad in ("javascript:alert(1)", "http://github.com/x", "https://evil.test/x"):
        _answer(monkeypatch, {"tag_name": "v9.9.9", "html_url": bad, "assets": []})
        release = updates.latest_release()
        assert release is not None
        assert release.version == "9.9.9"
        assert release.url == ""


def test_a_real_release_page_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    page = f"https://github.com/{updates.REPO}/releases/tag/v9.9.9"
    _answer(
        monkeypatch,
        {
            "tag_name": "v9.9.9",
            "html_url": page,
            "assets": [{"name": "x.whl", "browser_download_url": "https://w"}],
        },
    )
    release = updates.latest_release()
    assert release is not None
    assert release.url == page
    assert release.wheel == "https://w"


def test_the_update_command_is_the_install_command() -> None:
    command = updates.update_command()
    assert updates.REPO in command
    assert "install." in command
