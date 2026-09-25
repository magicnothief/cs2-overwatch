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
        create_app(runner, uploads=tmp_path / "uploads", radars=tmp_path / "radars")
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
    client = TestClient(create_app(runner, uploads=tmp_path / "uploads"))
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
