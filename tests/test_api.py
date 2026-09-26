"""Tests for the web backend: uploads, jobs, reports.

The pipeline is replaced by a stand-in that returns a small report, so these check
the web layer — what it accepts, where it stores things, what it refuses to
serve — not the detector.
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from overwatch.api import jobs
from overwatch.api.app import DEMO_MAGIC, create_app
from overwatch.api.jobs import Runner
from overwatch.pipeline.report import MatchReport, PlayerReport

#: What a browser is told to open, so Host and Origin are what a real one sends.
LOCAL = "http://127.0.0.1:8000"


def _fake_report(path, **kwargs) -> MatchReport:
    progress = kwargs.get("progress")
    for stage in ("layer 1", "layer 2", "layer 3"):
        progress(stage, stage)
    return MatchReport(
        demo=Path(path).name,
        map_name="de_dust2",
        visibility="ray cast against the de_dust2 mesh",
        kills=1,
        players=[PlayerReport(player_id="1", judge_alias="Player_0", kills=1)],
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(jobs, "analyze_demo", _fake_report)
    monkeypatch.setattr(jobs, "Scorer", lambda path: object())
    scorer = tmp_path / "scorer.pt"
    scorer.write_bytes(b"x")
    runner = Runner(
        tmp_path / "reports",
        scorer_path=scorer,
        judge_model=None,
        settings_file=tmp_path / "settings.json",
    )
    return TestClient(
        create_app(runner, uploads=tmp_path / "uploads", radars=tmp_path / "radars"),
        base_url=LOCAL,
    )


def _wait(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        job = client.get(f"/api/analyses/{job_id}").json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job never finished")


def test_a_demo_becomes_a_report(client: TestClient, tmp_path: Path) -> None:
    body = DEMO_MAGIC + b"rest of a demo"
    started = client.post(
        "/api/analyses?name=../../evil name.dem&judge=none", content=body
    )
    assert started.status_code == 200
    job = _wait(client, started.json()["id"])
    assert job["state"] == "done", job
    assert job["demo"] == "evil name.dem"  # displayed, stripped of any path

    stored = list((tmp_path / "uploads").iterdir())
    assert [p.suffix for p in stored] == [".dem"]  # named by content hash only
    assert "evil" not in stored[0].name

    listed = client.get("/api/reports").json()
    assert [r["id"] for r in listed] == [job["report_id"]]
    report = client.get(f"/api/reports/{job['report_id']}").json()
    assert report["map_name"] == "de_dust2"


def test_something_that_is_not_a_demo_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/analyses?name=cat.jpg", content=b"\xff\xd8\xff\xe0 jpeg"
    )
    assert response.status_code == 422
    assert "not a CS2 demo" in response.json()["detail"]


def test_a_report_id_cannot_walk_the_filesystem(client: TestClient) -> None:
    assert client.get("/api/reports/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/reports/nope").status_code == 404


def test_without_a_scorer_it_says_what_to_run(tmp_path: Path) -> None:
    runner = Runner(tmp_path, scorer_path=tmp_path / "missing.pt", judge_model=None)
    client = TestClient(
        create_app(runner, uploads=tmp_path / "uploads"), base_url=LOCAL
    )
    assert client.get("/api/status").json()["scorer"] is False
    response = client.post("/api/analyses", content=DEMO_MAGIC)
    assert response.status_code == 503
    assert "train_final.py" in response.json()["detail"]


def test_the_page_and_its_files_are_served(client: TestClient) -> None:
    assert "Overwatch review" in client.get("/").text
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/fonts/archivo-latin.woff2").status_code == 200


def test_radars_are_served_by_map_name_only(client: TestClient, tmp_path: Path) -> None:
    assert client.get("/api/radars/de_dust2").status_code == 404  # none rendered
    radars = tmp_path / "radars"
    radars.mkdir()
    (radars / "de_dust2.json").write_text('{"map": "de_dust2"}')
    (radars / "de_dust2.png").write_bytes(b"\x89PNG")
    assert client.get("/api/radars/de_dust2").json() == {"map": "de_dust2"}
    assert client.get("/radars/de_dust2.png").status_code == 200
    assert client.get("/api/radars/..%2F..%2Fsecret").status_code == 404
    assert client.get("/radars/..%2Fde_dust2.png").status_code == 404


def test_settings_start_empty_and_find_cs2_by_themselves(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from overwatch.maps import steam

    monkeypatch.delenv("OVERWATCH_CS2", raising=False)
    monkeypatch.setattr(steam, "steam_roots", lambda: iter([]))
    body = client.get("/api/settings").json()
    assert body["cs2"] is None
    assert body["cs2_found"] is None
    assert body["gpu"] == "auto"


def test_a_cs2_folder_is_checked_before_it_is_saved(
    client: TestClient, tmp_path: Path
) -> None:
    refused = client.put("/api/settings", json={"cs2": str(tmp_path), "gpu": "auto"})
    assert refused.status_code == 422
    assert not (tmp_path / "settings.json").exists()

    maps = tmp_path / "CS2" / "game" / "csgo" / "maps"
    maps.mkdir(parents=True)
    (maps / "de_dust2.vpk").write_bytes(b"")
    saved = client.put(
        "/api/settings", json={"cs2": str(tmp_path / "CS2"), "gpu": "off"}
    ).json()
    assert saved["cs2_found"] == str(maps)
    assert saved["gpu"] == "off"
    assert client.get("/api/settings").json() == saved


def test_an_unknown_gpu_choice_is_refused(client: TestClient) -> None:
    assert client.put("/api/settings", json={"gpu": "yes"}).status_code == 422


def test_local_demos_are_listed_and_reviewed_in_place(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip

    from overwatch.api import app as app_module

    replays = tmp_path / "replays"
    replays.mkdir()
    (replays / "match730_9.dem").write_bytes(DEMO_MAGIC + b"a demo")
    (replays / "old.dem.gz").write_bytes(gzip.compress(DEMO_MAGIC + b"packed"))
    monkeypatch.setattr(app_module, "demo_folders", lambda cs2=None: {"cs2": replays})

    listed = client.get("/api/demos").json()
    assert {d["name"] for d in listed["demos"]} == {"match730_9.dem", "old.dem.gz"}
    assert all(not d["reviewed"] for d in listed["demos"])

    started = client.post(
        "/api/analyses/local",
        json={"folder": "cs2", "name": "old.dem.gz", "judge": "none"},
    )
    assert started.status_code == 200
    assert _wait(client, started.json()["id"])["state"] == "done"
    refused = client.post(
        "/api/analyses/local", json={"folder": "cs2", "name": "../secret.dem"}
    )
    assert refused.status_code == 404


def test_a_packed_upload_is_unpacked(client: TestClient) -> None:
    import gzip

    body = gzip.compress(DEMO_MAGIC + b"packed demo")
    started = client.post("/api/analyses?name=faceit.dem.gz&judge=none", content=body)
    assert started.status_code == 200
    bad = client.post("/api/analyses?name=x.dem.gz", content=gzip.compress(b"nope"))
    assert bad.status_code == 422


def test_a_demo_the_parser_panics_on_fails_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated demo makes the parser panic; the job must fail, not hang."""

    class Panic(BaseException):
        pass

    def explode(path, **kwargs):
        raise Panic("range end index 16 out of range")

    monkeypatch.setattr(jobs, "analyze_demo", explode)
    monkeypatch.setattr(jobs, "Scorer", lambda path: object())
    scorer = tmp_path / "scorer.onnx"
    scorer.write_bytes(b"x")
    runner = Runner(
        tmp_path / "reports",
        scorer_path=scorer,
        judge_model=None,
        settings_file=tmp_path / "settings.json",
    )
    client = TestClient(
        create_app(runner, uploads=tmp_path / "up", radars=tmp_path), base_url=LOCAL
    )
    started = client.post("/api/analyses?judge=none", content=DEMO_MAGIC + b"cut")
    job = _wait(client, started.json()["id"])
    assert job["state"] == "failed"
    assert "Panic" in job["error"]


def test_another_site_cannot_start_an_analysis(client: TestClient) -> None:
    """A page on evil.example can POST to 127.0.0.1 without a preflight: POST
    with text/plain is a simple request. The Origin check is what stops it."""
    refused = client.post(
        "/api/analyses?name=x.dem&judge=none",
        content=DEMO_MAGIC + b"a demo",
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
    )
    assert refused.status_code == 403
    assert client.get("/api/reports").json() == []


def test_another_site_cannot_change_the_settings(client: TestClient) -> None:
    refused = client.put(
        "/api/settings", json={"gpu": "off"}, headers={"Origin": "https://evil.example"}
    )
    assert refused.status_code == 403
    assert client.get("/api/settings").json()["gpu"] == "auto"


def test_a_sandboxed_page_sending_origin_null_is_refused(client: TestClient) -> None:
    refused = client.post(
        "/api/analyses/local",
        json={"folder": "cs2", "name": "x.dem"},
        headers={"Origin": "null"},
    )
    assert refused.status_code == 403


def test_the_page_itself_and_the_terminal_still_work(client: TestClient) -> None:
    from_page = client.put(
        "/api/settings", json={"gpu": "off"}, headers={"Origin": LOCAL}
    )
    assert from_page.status_code == 200
    no_origin = client.put("/api/settings", json={"gpu": "auto"})  # curl, the CLI
    assert no_origin.status_code == 200


def test_a_rebound_dns_name_cannot_read_the_reports(client: TestClient) -> None:
    """evil.example resolved to 127.0.0.1 is same-origin with this server, so
    the same-origin policy does not help; the Host check does."""
    for path in ("/api/reports", "/api/settings", "/api/demos"):
        rebound = client.get(path, headers={"Host": "evil.example"})
        assert rebound.status_code == 421, path


def test_the_page_carries_its_security_headers(client: TestClient) -> None:
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert "script-src 'self'" in headers["Content-Security-Policy"]


def test_a_packed_upload_that_is_a_bomb_is_refused(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bz2

    from overwatch import demos

    monkeypatch.setattr(demos, "MAX_UNPACKED_BYTES", 1 << 20)
    body = bz2.compress(DEMO_MAGIC + b"\0" * (8 << 20), 9)
    assert len(body) < 64 * 1024
    refused = client.post("/api/analyses?name=bomb.dem.bz2&judge=none", content=body)
    assert refused.status_code == 413
    assert not any((tmp_path / "uploads").glob("*.dem"))
