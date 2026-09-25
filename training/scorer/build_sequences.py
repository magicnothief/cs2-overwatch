"""Turn the window table into the arrays the sequence models train on.

Run:  uv run python training/scorer/build_sequences.py

Reads data/processed/window_ticks.parquet (from build_dataset.py) and writes
data/processed/sequences.npz, which train_sequence_model.py, cross_validate.py,
train_mil_model.py and train_invariant.py all read.

This used to be a one-off snippet, and that cost a wrong result: after the map
meshes were fixed the window table was rebuilt but sequences.npz was not, so the
cross-validation that followed scored the old visibility. Run this after every
build_dataset.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from overwatch.layers.l3_behavior.sequences import build_sequences

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED)
    args = parser.parse_args()

    started = time.perf_counter()
    source = args.data / "window_ticks.parquet"
    seq = build_sequences(pl.read_parquet(source))
    out = args.data / "sequences.npz"
    np.savez_compressed(
        out, x=seq.x, y=seq.y, groups=seq.groups, players=seq.players, uids=seq.uids
    )
    print(
        f"{len(seq)} windows, {seq.x.shape[1]} channels x {seq.x.shape[2]} ticks, "
        f"{seq.y.mean():.1%} cheater -> {out} ({time.perf_counter() - started:.0f}s)"
    )


if __name__ == "__main__":
    main()
