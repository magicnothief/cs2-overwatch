"""The models the app runs, and where a fresh install downloads them from.

    detector   the behaviour CNN (ONNX) and its frozen clean reference: 150 KB
    judge      the fine-tuned Qwen3.5-4B, judge v3, Q4_K_M: 2.8 GB

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
REVISION = "main"


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
        "73670d621f93d93ab0c25a89149b963e7008970be7d04d2246cc5db40c71b407",
        5_642,
    ),
)
JUDGE = ModelFile(
    "judge/judge-v3.Q4_K_M.gguf",
    paths.MODELS / "llm" / "V3" / "Full" / "Qwen3.5-4B.Q4_K_M.gguf",
    "bf6d4b2e902870cba677e3a18bee8f9e8b09e5eaa45df5a612e1dbcb66fc46b0",
    2_783_447_008,
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
