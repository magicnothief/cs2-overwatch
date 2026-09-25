"""The few choices a user makes, kept in <home>/settings.json.

cs2   CS2's folder, when Steam's library list does not show it (a dual-boot
      machine's other drive, say); found automatically when empty
gpu   "auto" runs the judge on a GPU with room for it (Vulkan), "cuda" tries
      NVIDIA's CUDA build first (a little faster, a 600 MB download), "off"
      keeps it on the CPU (the GPU is left free for games or training)
updates  ask GitHub once a day whether a newer release is out (updates.py)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from overwatch import paths

FILE = paths.HOME / "settings.json"


class Settings(BaseModel):
    cs2: str | None = None
    gpu: Literal["auto", "cuda", "off"] = "auto"
    updates: bool = True

    @property
    def gpu_layers(self) -> int | str:
        return 0 if self.gpu == "off" else "auto"

    @property
    def prefer_cuda(self) -> bool:
        return self.gpu == "cuda"


def load_settings(path: Path = FILE) -> Settings:
    try:
        return Settings.model_validate_json(path.read_text())
    except (OSError, ValueError):
        return Settings()


def save_settings(settings: Settings, path: Path = FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(settings.model_dump(), indent=2) + "\n")
    part.replace(path)


__all__ = ["FILE", "Settings", "load_settings", "save_settings"]
