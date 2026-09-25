"""The models the app runs, and where a fresh install downloads them from.

    detector   the behaviour CNN (ONNX) and its frozen clean reference: 150 KB
    judge      the fine-tuned Qwen3.5-4B, judge v4, Q4_K_M: 2.8 GB

Both are published in a Hugging Face model repo (MODELS_REPO), pinned to one
commit (REVISION), and every file is checked against the SHA-256 recorded here,
so a download is exactly the model that was evaluated. A development checkout
already has them in models/, and nothing is downloaded.

After retraining, training/publish_models.py uploads the new files and prints
the lines to paste here.

OVERWATCH_MODELS_URL points downloads at another copy (a mirror, or a local
folder as file:///...) with the same layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from overwatch import paths
from overwatch.downloads import Progress, fetch

#: The Hugging Face model repo, and the commit every download is pinned to.
MODELS_REPO = "MagicNoThief/cs2-overwatch"
REVISION = "b0cb2672be269eb771a74d44aff120744c680694"


def base_url() -> str:
    custom = os.environ.get("OVERWATCH_MODELS_URL")
    if custom:
        return custom.rstrip("/") + "/"
    return f"https://huggingface.co/{MODELS_REPO}/resolve/{REVISION}/"


@dataclass(frozen=True)
class ModelFile:
    remote: str  # path inside the repo
    local: Path  # where the app reads it
    sha256: str
    size: int

    @property
    def present(self) -> bool:
        return self.local.exists()


DETECTOR = (
    ModelFile(
        "detector/scorer.onnx",
        paths.MODELS / "scorer" / "scorer.onnx",
        "c85e576e119925c524d3ebc7e33ba722c0042b9d1588e02ad54510b551eeb667",
        145_665,
    ),
    ModelFile(
        "detector/scorer.json",
        paths.MODELS / "scorer" / "scorer.json",
        "ec3497315adedba528c686d55f4c31befe2f87bca2f2dd97b6ee8f7790195bd1",
        5_627,
    ),
)
JUDGE = ModelFile(
    "judge/judge-v4.Q4_K_M.gguf",
    paths.MODELS / "llm" / "V4" / "Qwen3.5-4B.Q4_K_M.gguf",
    "8ef3824fed200541f456e3ebfc1aaa4d40533797255f4c31b583ccb14d697068",
    2_783_446_976,
)


def missing(*, judge: bool = True) -> list[ModelFile]:
    wanted = [*DETECTOR, JUDGE] if judge else list(DETECTOR)
    return [m for m in wanted if not m.present]


def download(files: list[ModelFile], progress: Progress | None = None) -> None:
    """Fetch the files, each checked against its SHA-256 before it is kept."""
    tell = progress or (lambda _message: None)
    for model in files:
        tell(f"Downloading {model.remote} ({model.size / 1e6:,.0f} MB)")
        fetch(base_url() + model.remote, model.local, model.sha256, tell)


__all__ = ["DETECTOR", "JUDGE", "MODELS_REPO", "ModelFile", "download", "missing"]
