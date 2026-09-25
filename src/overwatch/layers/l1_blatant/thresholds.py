"""Where Layer 1's lines come from.

Every threshold is derived from the clean population rather than chosen by hand,
and the values that produced it are kept alongside so a future reader can see how
much headroom each line has. Recalibrate with:

    uv run python training/rules/calibrate_thresholds.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("thresholds.json")


@dataclass(frozen=True)
class Thresholds:
    """Lines a clean player does not cross."""

    # sustained spin: speed, and how long it must hold
    spin_speed_dps: float = 3000.0
    spin_min_ticks: int = 16
    # an aliased (too fast to measure) spin: size of each flip and how many in a row
    alias_min_step_deg: float = 90.0
    alias_min_ticks: int = 8
    # a turn landing exactly on the kill tick
    snap_speed_dps: float = 900.0
    # the game clamps player pitch at 89
    max_pitch_deg: float = 89.0

    # what the clean population actually showed, for context in reports
    clean_speed_p9999: float = 0.0
    clean_snap_p9999: float = 0.0
    clean_pitch_max: float = 0.0
    calibrated_on_matches: int = 0

    @classmethod
    def load(cls, path: Path | None = None) -> Thresholds:
        path = path or DEFAULT_PATH
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, path: Path | None = None) -> Path:
        path = path or DEFAULT_PATH
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path
