"""Tests for the annotation tool: what a human writes becomes training data.

Two properties matter most. The annotator must not see the label or the generated
target before committing, or the gold set measures agreement with our own guess.
And a human target must obey the same rules as a generated one, or the model is
trained on two contradicting styles.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from overwatch.annotation import (
    Annotation,
    AnnotationStore,
    CaseBook,
    evidence_hash,
    human_verdict,
)
from overwatch.annotation.app import create_app
from overwatch.layers.l4_judge import PlayerCase, render_case
from overwatch.layers.l4_judge.targets import build_target, case_rng, split_for
from overwatch.layers.l4_judge.verdict import CheatType, VerdictLabel

LINES = {
    "wall_aim_share": (0.46, 0.67),
    "straight_share": (0.48, 0.67),
}
BASELINES = {"wall_aim_share": 0.14, "straight_share": 0.15}


def _write_cases(path: Path, n_matches: int = 40) -> Path:
    rows = []
    for m in range(n_matches):
        for p, (label, wall) in enumerate((("cheater", 0.9), ("clean", 0.05))):
            case = PlayerCase(
                match_id=f"set/{m}",
                player_id=f"Player_{p}",
                kills=20,
                score=0.5,
                features={"wall_aim_share": wall, "straight_share": 0.1},
                baselines=BASELINES,
                clean_lines=LINES,
            )
            rows.append(
                {
                    "match_id": case.match_id,
                    "player_id": case.player_id,
                    "label": label,
                    "case": case.model_dump(mode="json"),
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture
def book(tmp_path: Path) -> CaseBook:
    return CaseBook(_write_cases(tmp_path / "cases.jsonl"))


@pytest.fixture
def client(book: CaseBook, tmp_path: Path) -> TestClient:
    store = AnnotationStore(tmp_path / "gold.jsonl")
    return TestClient(create_app(book, store, annotator="tester"))


# --- the verdict a human can write ------------------------------------------


def test_the_verdict_word_follows_the_probability() -> None:
    assert human_verdict(10, "none", ["r"], []).verdict is VerdictLabel.CLEAN
    assert human_verdict(50, "none", ["r"], []).verdict is VerdictLabel.UNCLEAR
    assert human_verdict(90, "aimbot", ["r"], []).verdict is VerdictLabel.CHEATING


def test_a_cheat_is_named_only_when_accusing() -> None:
    # "unclear, but wallhack" would teach accusing while claiming not to
    assert human_verdict(50, "wallhack", ["r"], []).cheat_type is CheatType.NONE
    with pytest.raises(ValueError, match="name the cheat"):
        human_verdict(90, "none", ["r"], [])


@pytest.mark.parametrize(
    ("reasons", "caveats", "message"),
    [
        ([], [], "at least one reason"),
        (["  "], [], "at least one reason"),
        (["a", "b", "c", "d"], [], "at most 3 reasons"),
        (["a"], ["x", "y", "z"], "at most 2 caveats"),
        (["x" * 161], [], "under 160"),
    ],
)
def test_a_human_target_fits_the_model_grammar(reasons, caveats, message) -> None:
    with pytest.raises(ValueError, match=message):
        human_verdict(50, "none", reasons, caveats)


# --- storage ----------------------------------------------------------------


def test_the_latest_annotation_wins_and_history_is_kept(tmp_path: Path) -> None:
    store = AnnotationStore(tmp_path / "sub" / "gold.jsonl")
    assert store.load() == {}
    for probability in (20, 80):
        store.append(
            Annotation(
                match_id="m",
                player_id="p",
                evidence_sha="abc",
                verdict=human_verdict(probability, "aimbot", ["r"], []),
            )
        )
    assert store.load()["m/p"].verdict.probability == 80
    assert len(store.path.read_text().splitlines()) == 2


# --- the queue --------------------------------------------------------------


def test_validation_cases_come_first(book: CaseBook) -> None:
    halves = [book.split(key) for key in book.queue]
    assert sorted(book.queue) == sorted(book._rows)
    assert "val" in halves and "train" in halves
    assert halves == sorted(halves, key=lambda h: h != "val")


def test_the_queue_interleaves_labels(book: CaseBook) -> None:
    val = [k for k in book.queue if book.split(k) == "val"]
    assert {book.label(k) for k in val[:2]} == {"cheater", "clean"}


def test_generated_target_is_the_training_set_target(book: CaseBook) -> None:
    key = book.queue[0]
    case = book.case(key)
    expected = build_target(case, case_rng(case.match_id, case.player_id))
    assert book.generated(key) == expected


def test_split_is_fixed_per_match() -> None:
    assert split_for("set/3") == split_for("set/3")
    assert {split_for(f"set/{m}") for m in range(200)} == {"train", "val"}


# --- the app ----------------------------------------------------------------


def test_nothing_is_revealed_before_saving(client: TestClient) -> None:
    key = client.get("/api/next").json()["key"]
    case = client.get(f"/api/case/{key}").json()
    assert case["reveal"] is None and case["annotation"] is None
    assert "cheater" not in json.dumps(case) and "clean player" in case["evidence"]


def test_saving_reveals_and_moves_on(client: TestClient, book: CaseBook) -> None:
    key = client.get("/api/next").json()["key"]
    case = client.get(f"/api/case/{key}").json()
    saved = client.post(
        "/api/annotations",
        json={
            "key": key,
            "evidence_sha": case["evidence_sha"],
            "probability": 90,
            "cheat_type": "wallhack",
            "reasons": ["they held the crosshair on hidden enemies"],
        },
    ).json()
    assert saved["reveal"]["label"] == book.label(key)
    assert saved["progress"]["done"] == 1
    assert client.get("/api/next").json()["key"] != key
    # coming back to it shows what was decided, and the reveal
    again = client.get(f"/api/case/{key}").json()
    assert again["annotation"]["verdict"]["probability"] == 90
    assert again["reveal"] is not None


def test_stale_evidence_is_refused(client: TestClient) -> None:
    key = client.get("/api/next").json()["key"]
    response = client.post(
        "/api/annotations",
        json={
            "key": key,
            "evidence_sha": "0" * 16,
            "probability": 10,
            "reasons": ["r"],
        },
    )
    assert response.status_code == 409


def test_an_invalid_verdict_explains_itself(client: TestClient) -> None:
    key = client.get("/api/next").json()["key"]
    sha = client.get(f"/api/case/{key}").json()["evidence_sha"]
    response = client.post(
        "/api/annotations",
        json={"key": key, "evidence_sha": sha, "probability": 90, "reasons": ["r"]},
    )
    assert response.status_code == 422
    assert "name the cheat" in response.json()["detail"]


def test_skipped_cases_are_passed_over(client: TestClient) -> None:
    first = client.get("/api/next").json()["key"]
    assert client.get(f"/api/next?skip={first}").json()["key"] != first


def test_annotations_on_old_evidence_do_not_count(
    client: TestClient, book: CaseBook, tmp_path: Path
) -> None:
    key = book.queue[0]
    case = book.case(key)
    AnnotationStore(tmp_path / "gold.jsonl").append(
        Annotation(
            match_id=case.match_id,
            player_id=case.player_id,
            evidence_sha=evidence_hash(render_case(case) + " (old wording)"),
            verdict=human_verdict(10, "none", ["r"], []),
        )
    )
    stats = client.get("/api/stats").json()
    assert stats["progress"]["done"] == 0 and stats["stale"] == 1
    assert client.get("/api/next").json()["key"] == key


def test_the_page_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "Overwatch" in page.text
