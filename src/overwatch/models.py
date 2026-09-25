"""The models the app runs, and where a fresh install downloads them from.

    detector   the behaviour CNN (ONNX) and its frozen clean reference: 150 KB
    judge      the fine-tuned Qwen3.5-4B, judge v4, Q4_K_M: 2.8 GB

Both are published in a Hugging Face model repo (MODELS_REPO), pinned to one
commit (REVISION), and every file is checked against the SHA-256 recorded here,
so a download is exactly the model that was evaluated.

A file counts as present only when its hash matches: after an update pins a new
model, the old file is replaced rather than kept. Hashes are cached beside each
file (<file>.sha256, keyed on its size and modification time), so the 2.8 GB
judge is hashed once, not on every start. A development checkout already has
its models in models/, and keeps them even when they differ (a model just
retrained there must not be overwritten by a download).

After retraining, training/publish_models.py uploads the new files and prints
the lines to paste here.

OVERWATCH_MODELS_URL points downloads at another copy (a mirror, or a local
folder as file:///...) with the same layout.
"""

from __future__ import annotations

import hashlib
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
        """Here, and exactly the pinned version."""
        return self.local.exists() and local_sha256(self.local) == self.sha256


def local_sha256(path: Path) -> str:
    """A file's SHA-256, cached beside it against its size and modification time."""
    stat = path.stat()
    key = f"{stat.st_size} {stat.st_mtime_ns}"
    cache = path.with_name(path.name + ".sha256")
    try:
        cached_key, digest = cache.read_text().split("\n")[:2]
        if cached_key == key and len(digest) == 64:
            return digest
    except (OSError, ValueError):
        pass
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    try:
        cache.write_text(f"{key}\n{digest}\n")
    except OSError:
        pass  # a read-only folder only costs the next start a re-hash
    return digest


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
    """The files to download: absent ones, and outdated ones outside a checkout."""
    wanted = [*DETECTOR, JUDGE] if judge else list(DETECTOR)
    return [
        m
        for m in wanted
        if not m.local.exists() or (not paths.IN_CHECKOUT and not m.present)
    ]


def download(files: list[ModelFile], progress: Progress | None = None) -> None:
    """Fetch the files, each checked against its SHA-256 before it is kept."""
    tell = progress or (lambda _message: None)
    for model in files:
        tell(f"Downloading {model.remote} ({model.size / 1e6:,.0f} MB)")
        fetch(base_url() + model.remote, model.local, model.sha256, tell)


__all__ = [
    "DETECTOR",
    "JUDGE",
    "MODELS_REPO",
    "ModelFile",
    "download",
    "local_sha256",
    "missing",
]
