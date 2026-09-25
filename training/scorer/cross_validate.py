"""Cross-validate a sequence model, so architecture choices stop being guesswork.

Run:  uv run python training/scorer/cross_validate.py --folds 5

Every architecture tried so far differed from the baseline by less than its own
confidence interval on one split: width 64 + attention 0.928 vs 0.948, MIL 0.929,
257-tick windows 0.935. Those gaps are the size of the noise, so a single split
cannot rank them.

This trains one model per fold over matches, scores every player out of fold, and
reports the metrics on all of them at once, plus the spread across folds. A change
worth keeping has to move the out-of-fold number by more than that spread.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import GroupKFold

from overwatch.layers.l3_behavior.evaluation import report

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sequence_model import predict, train

PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--pooling", choices=("avgmax", "attention"), default="avgmax")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--label", default="baseline", help="name for this configuration"
    )
    args = parser.parse_args()

    cached = np.load(args.data / "sequences.npz", allow_pickle=True)
    data = {k: cached[k] for k in ("x", "y", "groups", "players", "uids")}
    groups = data["groups"]

    window_scores = np.zeros(len(groups))
    fold_rows = []
    started = time.time()

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=args.folds).split(data["x"], data["y"], groups), start=1
    ):
        # hold out whole matches inside the training part for early stopping
        train_matches = np.unique(groups[train_idx])
        rng = np.random.default_rng(fold)
        rng.shuffle(train_matches)
        val_matches = set(train_matches[: max(1, len(train_matches) // 6)])
        is_val = np.array([g in val_matches for g in groups[train_idx]])

        splits = {
            "train": train_idx[~is_val],
            "val": train_idx[is_val],
            "test": test_idx,
        }
        model, _ = train(
            data,
            splits,
            epochs=args.epochs,
            device=args.device,
            width=args.width,
            pooling=args.pooling,
        )
        scores = predict(model, torch.from_numpy(data["x"][test_idx]), args.device)
        window_scores[test_idx] = scores

        fold_row = report(f"fold {fold}", data["y"][test_idx], scores, ci=False)
        fold_rows.append(fold_row)
        print(
            f"fold {fold}: window ROC-AUC {fold_row['roc_auc']:.3f} "
            f"({time.time() - started:.0f}s elapsed)",
            flush=True,
        )

    # per player, the aggregation that won on the single split
    players = (
        pl.DataFrame(
            {
                "player": data["players"],
                "label": data["y"].astype(int),
                "score": window_scores,
            }
        )
        .group_by("player")
        .agg(
            label=pl.col("label").max(),
            n_windows=pl.len(),
            score=pl.col("score").mean(),
        )
        .filter(pl.col("n_windows") >= 5)
    )

    rows = [
        report(f"{args.label}: per window (oof)", data["y"], window_scores),
        report(
            f"{args.label}: per player (oof)",
            players["label"].to_numpy(),
            players["score"].to_numpy(),
        ),
    ]
    with pl.Config(float_precision=3, tbl_width_chars=150, fmt_str_lengths=40):
        print(pl.DataFrame(rows))

    spread = pl.DataFrame(fold_rows)["roc_auc"]
    print(
        f"\nper-fold window ROC-AUC: {spread.min():.3f}-{spread.max():.3f} "
        f"(sd {spread.std():.3f}) over {args.folds} folds, "
        f"{players.height} players, {int(players['label'].sum())} cheaters"
    )
    players.write_parquet(args.data / f"cv_player_scores_{args.label}.parquet")


if __name__ == "__main__":
    main()
