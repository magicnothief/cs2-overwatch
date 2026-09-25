"""The annotation web app: one case at a time, blind until saved.

    GET  /                        the page (static/annotate.html)
    GET  /api/next?skip=k1,k2     the next case nobody has annotated
    GET  /api/case/{key}          one case: evidence text, chart data, phrases
    POST /api/annotations         save a verdict; reveals label and generated target
    GET  /api/stats               progress and agreement with generated targets

Local only: it binds to 127.0.0.1 and has no accounts. The label and the generated
target are withheld from /api/case for an unannotated case, so the page cannot
show them early even by accident.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from overwatch.annotation.casebook import CaseBook
from overwatch.annotation.store import (
    MAX_CAVEATS,
    MAX_REASONS,
    MAX_SENTENCE,
    Annotation,
    AnnotationStore,
    evidence_hash,
    human_verdict,
)
from overwatch.layers.l4_judge.targets import CHEATING_ABOVE, CLEAN_BELOW
from overwatch.layers.l4_judge.verdict import CheatType

STATIC = Path(__file__).parent / "static"

#: How many annotations make a first useful gold set: enough to measure agreement
#: with generated targets within about ±7 points, and to test the fine-tuned judge.
GOAL = 200


class AnnotationIn(BaseModel):
    key: str
    evidence_sha: str
    probability: int = Field(ge=0, le=100)
    cheat_type: CheatType = CheatType.NONE
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    note: str = ""
    seconds: float | None = None


def create_app(
    book: CaseBook, store: AnnotationStore, *, annotator: str = ""
) -> FastAPI:
    app = FastAPI(title="Overwatch annotation", docs_url=None, redoc_url=None)

    def current() -> dict[str, Annotation]:
        """Annotations made on the evidence as it renders today."""
        return {
            key: annotation
            for key, annotation in store.load().items()
            if key in book
            and annotation.evidence_sha == evidence_hash(book.evidence(key))
        }

    def progress(done: dict[str, Annotation]) -> dict:
        return {"done": len(done), "goal": GOAL, "total": len(book)}

    def reveal(key: str, annotation: Annotation) -> dict:
        generated = book.generated(key)
        return {
            "label": book.label(key),
            "generated": generated.model_dump(mode="json"),
            "agrees": generated.verdict == annotation.verdict.verdict,
        }

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "annotate.html")

    @app.get("/api/next")
    def next_case(skip: str = "") -> dict:
        done = current()
        skipped = set(filter(None, skip.split(",")))
        key = next((k for k in book.queue if k not in done and k not in skipped), None)
        return {"key": key, "progress": progress(done)}

    @app.get("/api/case/{key:path}")
    def get_case(key: str) -> dict:
        if key not in book:
            raise HTTPException(404, f"no case {key}")
        evidence = book.evidence(key)
        existing = current().get(key)
        return {
            "key": key,
            "evidence": evidence,
            "evidence_sha": evidence_hash(evidence),
            "case": book.case(key).model_dump(mode="json"),
            "phrases": book.phrases(key),
            "limits": {
                "reasons": MAX_REASONS,
                "caveats": MAX_CAVEATS,
                "sentence": MAX_SENTENCE,
                "clean_below": CLEAN_BELOW,
                "cheating_above": CHEATING_ABOVE,
            },
            # only once the annotator has committed to their own verdict
            "annotation": existing.model_dump(mode="json") if existing else None,
            "reveal": reveal(key, existing) if existing else None,
        }

    @app.post("/api/annotations")
    def save(body: AnnotationIn) -> dict:
        if body.key not in book:
            raise HTTPException(404, f"no case {body.key}")
        if body.evidence_sha != evidence_hash(book.evidence(body.key)):
            raise HTTPException(
                409, "the evidence text changed since this page loaded; reload it"
            )
        try:
            verdict = human_verdict(
                body.probability, body.cheat_type, body.reasons, body.caveats
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        case = book.case(body.key)
        annotation = Annotation(
            match_id=case.match_id,
            player_id=case.player_id,
            evidence_sha=body.evidence_sha,
            verdict=verdict,
            note=body.note.strip(),
            seconds=body.seconds,
            annotator=annotator,
        )
        store.append(annotation)
        return {
            "annotation": annotation.model_dump(mode="json"),
            "reveal": reveal(body.key, annotation),
            "progress": progress(current()),
        }

    @app.get("/api/stats")
    def stats() -> dict:
        done = current()
        rows = [(book.label(k), a.verdict, book.generated(k)) for k, a in done.items()]
        by_label: dict[str, dict] = {}
        for label in sorted({r[0] for r in rows}):
            mine = [r for r in rows if r[0] == label]
            by_label[label] = {
                "n": len(mine),
                "agree": sum(h.verdict == g.verdict for _, h, g in mine) / len(mine),
            }
        return {
            "progress": progress(done),
            "stale": len(store.load()) - len(done),
            "agree": (
                sum(h.verdict == g.verdict for _, h, g in rows) / len(rows)
                if rows
                else None
            ),
            "probability_gap": (
                sum(abs(h.probability - g.probability) for _, h, g in rows) / len(rows)
                if rows
                else None
            ),
            "by_label": by_label,
        }

    return app


__all__ = ["GOAL", "AnnotationIn", "create_app"]
