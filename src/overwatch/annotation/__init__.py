"""Human annotation of judge cases: the reasoning the judge is trained to copy.

Start it with `uv run python -m overwatch.annotation`.
"""

from overwatch.annotation.casebook import CaseBook
from overwatch.annotation.store import (
    Annotation,
    AnnotationStore,
    case_key,
    evidence_hash,
    human_verdict,
)

__all__ = [
    "Annotation",
    "AnnotationStore",
    "CaseBook",
    "case_key",
    "evidence_hash",
    "human_verdict",
]
