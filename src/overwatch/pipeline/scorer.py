"""Layer 3 at inference: the production CNN and the reference it is read against.

The scorer ships as two files (training/scorer/train_final.py writes both, via
l3_behavior/export.py): scorer.onnx, the network, and scorer.json, its
architecture and a frozen reference: clean players' score distribution from
cross-validation. A single demo has no clean population of its own, so "0.62"
means nothing until it becomes "higher than 94% of clean players".

It runs on ONNX Runtime, on the CPU: the network is tiny, and PyTorch would add
gigabytes to every install for nothing.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import polars as pl

from overwatch import models
from overwatch.layers.l3_behavior.sequences import CHANNELS, build_sequences

DEFAULT_SCORER = models.DETECTOR[0].local
#: Windows scored per call: bounds memory on a long demo.
BATCH = 1024


class Scorer:
    """Scores kill windows, and players from their windows, exactly as in training."""

    def __init__(self, path: str | Path = DEFAULT_SCORER) -> None:
        import onnxruntime as ort

        path = Path(path).with_suffix(".onnx")
        meta = path.with_suffix(".json")
        if not path.exists() or not meta.exists():
            msg = (
                f"no scorer at {path} (and {meta.name} beside it): run "
                "training/scorer/train_final.py, or let first-run setup download it"
            )
            raise FileNotFoundError(msg)
        bundle = json.loads(meta.read_text())
        self.config: dict = bundle["config"]
        self.reference: dict = bundle["reference"]
        self.trained_at: str = bundle.get("trained_at", "unknown")
        if list(self.config["channels"]) != list(CHANNELS):
            msg = f"scorer was trained on channels {self.config['channels']}"
            raise ValueError(msg)
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

        quantiles = self.reference["score"]["clean_quantiles"]
        self._levels = sorted(float(q) for q in quantiles)
        self._cuts = [quantiles[str(q)] for q in self._levels]

    @property
    def min_windows(self) -> int:
        return int(self.reference["score"]["min_windows"])

    def clean_line(self, share: float) -> float:
        """The score that `share` of clean players stay below (e.g. 0.9 -> p90)."""
        return float(self.reference["score"]["clean_quantiles"][str(share)])

    def clean_percentile(self, score: float) -> float:
        """Share of clean players who scored lower, from the frozen quantiles."""
        return (
            self._levels[bisect.bisect_right(self._cuts, score) - 1]
            if score >= self._cuts[0]
            else 0.0
        )

    def score_windows(self, window_ticks: pl.DataFrame) -> pl.DataFrame:
        """One suspicion score per complete kill window: window_uid, player_id, score.

        Windows cut short (a kill in the first two seconds of the demo, a gap in
        the attacker's ticks) are dropped, as they were in training.
        """
        frame = window_ticks.with_columns(label=pl.lit("unknown"))
        seq = build_sequences(frame, labels=("unknown",))
        if len(seq) == 0:
            return pl.DataFrame(
                schema={
                    "window_uid": pl.String,
                    "player_id": pl.String,
                    "score": pl.Float64,
                }
            )
        x = seq.x.astype(np.float32)
        scores = np.concatenate(
            [
                self.session.run(None, {"windows": x[i : i + BATCH]})[0]
                for i in range(0, len(x), BATCH)
            ]
        ).astype(np.float64)
        players = [p.rsplit("/", 1)[-1] for p in seq.players]
        return pl.DataFrame(
            {"window_uid": seq.uids, "player_id": players, "score": scores}
        )

    def score_players(self, window_scores: pl.DataFrame) -> pl.DataFrame:
        """Per player: the mean window score (as cross-validation measured it)."""
        if self.reference["score"]["aggregate"] != "mean":
            msg = f"unknown aggregate {self.reference['score']['aggregate']}"
            raise ValueError(msg)
        players = window_scores.group_by("player_id").agg(
            n_windows=pl.len(), score=pl.col("score").mean()
        )
        return players.with_columns(
            clean_percentile=pl.col("score").map_elements(
                self.clean_percentile, return_dtype=pl.Float64
            ),
            enough_kills=pl.col("n_windows") >= self.min_windows,
        ).sort("score", descending=True)


__all__ = ["DEFAULT_SCORER", "Scorer"]
