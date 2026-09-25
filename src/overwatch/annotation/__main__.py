"""Start the annotation tool.

Run:  uv run python -m overwatch.annotation

then open http://127.0.0.1:8765. Annotations are appended to
data/annotations/gold.jsonl as you save them; stop with Ctrl+C at any point.
"""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import uvicorn

from overwatch.annotation.app import create_app
from overwatch.annotation.casebook import CaseBook
from overwatch.annotation.store import AnnotationStore

PROCESSED = Path("data/processed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=PROCESSED / "judge_cases.jsonl")
    parser.add_argument("--out", type=Path, default=Path("data/annotations/gold.jsonl"))
    parser.add_argument("--annotator", default=getpass.getuser())
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    book = CaseBook(args.cases)
    store = AnnotationStore(args.out)
    print(
        f"{len(book)} cases, {len(store.load())} annotated so far -> {args.out}\n"
        f"open http://127.0.0.1:{args.port}",
        flush=True,
    )
    uvicorn.run(
        create_app(book, store, annotator=args.annotator),
        host="127.0.0.1",  # local only: there are no accounts
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
