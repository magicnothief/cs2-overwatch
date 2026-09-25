"""Train the production scorer: the model the pipeline loads, with what it compares to.

Run:  uv run python training/scorer/train_final.py

Cross-validation (cross_validate.py) measures how good the architecture is; this
trains the one model that ships, on every match except a sixth held out for early
stopping, and saves it as a single bundle, models/scorer/scorer.pt (kept for
training), exported for the app as scorer.onnx + scorer.json beside it:

    state_dict   the FlickCNN weights
    config       architecture and window shape, so loading needs no guesswork
    reference    everything a live demo is compared against, frozen at training
                 time, because a single demo has no clean population of its own:
                   score      clean players' out-of-fold score distribution, so a
                              score reads as "higher than 97% of clean players"
                   judge      the clean medians and the notable/strong lines the
                              judge's evidence and targets are built on

Inputs: sequences.npz (build_sequences.py), player_features.parquet
(train_player_model.py) and cv_player_scores_<label>.parquet (cross_validate.py).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from overwatch.layers.l3_behavior.export import export_scorer
from overwatch.layers.l3_behavior.sequences import CHANNELS
from overwatch.layers.l4_judge.cases import clean_baselines
from overwatch.layers.l4_judge.targets import clean_lines

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sequence_model import train

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "models" / "scorer" / "scorer.pt"

#: Must match dataset.match_windows' defaults and the sequences it was built from.
PRE, POST = 128, 32
#: How window scores become a player score, exactly as cross_validate.py does it.
AGGREGATE = "mean"
MIN_WINDOWS = 5


def score_reference(cv_scores: pl.DataFrame) -> dict:
    """Clean players' out-of-fold scores, as quantiles to place a new score among."""
    clean = cv_scores.filter(pl.col("label") == 0)["score"]
    cheat = cv_scores.filter(pl.col("label") == 1)["score"]
    quantiles = [round(q / 100, 2) for q in range(1, 100)] + [0.995, 0.999]
    return {
        "aggregate": AGGREGATE,
        "min_windows": MIN_WINDOWS,
        "clean_quantiles": {str(q): float(clean.quantile(q)) for q in quantiles},
        "cheater_median": float(cheat.median()),
        "n_clean": clean.len(),
        "n_cheater": cheat.len(),
    }


def reference(data: Path, cv_label: str) -> dict:
    """Everything a live demo is compared against, from the training data."""
    players = pl.read_parquet(data / "player_features.parquet")
    cv_scores = pl.read_parquet(data / f"cv_player_scores_{cv_label}.parquet")
    return {
        "score": score_reference(cv_scores),
        "judge": {
            "baselines": clean_baselines(players),
            "lines": {k: list(v) for k, v in clean_lines(players).items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED)
    parser.add_argument(
        "--cv-label",
        default=None,
        help="cross_validate.py run to take the reference from (default: meshes23, "
        "or with --refresh-reference the run the bundle already names)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--pooling", choices=("avgmax", "attention"), default="avgmax")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="keep the trained network, recompute only what it is compared against",
    )
    args = parser.parse_args()

    if args.refresh_reference:
        bundle = torch.load(args.out, map_location="cpu", weights_only=False)
        args.cv_label = args.cv_label or bundle.get("cv_label", "meshes23")
        bundle["reference"] = reference(args.data, args.cv_label)
        torch.save(bundle, args.out)
        shipped = export_scorer(bundle, args.out)
        print(f"refreshed the reference in {args.out} and {shipped}")
        return

    args.cv_label = args.cv_label or "meshes23"
    cached = np.load(args.data / "sequences.npz", allow_pickle=True)
    data = {k: cached[k] for k in ("x", "y", "groups", "players", "uids")}
    if data["x"].shape[1] != len(CHANNELS) or data["x"].shape[2] != PRE + POST + 1:
        msg = f"sequences.npz has shape {data['x'].shape}; rebuild it"
        raise SystemExit(msg)

    # every match trains, except a sixth held out to decide when to stop
    groups = data["groups"]
    matches = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(matches)
    held_out = set(matches[: len(matches) // 6])
    is_val = np.array([g in held_out for g in groups])
    splits = {"train": np.flatnonzero(~is_val), "val": np.flatnonzero(is_val)}
    print(
        f"training on {len(splits['train'])} windows, stopping on "
        f"{len(splits['val'])} ({len(held_out)} held-out matches)",
        flush=True,
    )

    model, history = train(
        data,
        splits,
        epochs=args.epochs,
        device=args.device,
        width=args.width,
        pooling=args.pooling,
    )

    bundle = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {
            "model": "FlickCNN",
            "in_channels": len(CHANNELS),
            "channels": list(CHANNELS),
            "width": args.width,
            "pooling": args.pooling,
            "pre": PRE,
            "post": POST,
        },
        "reference": reference(args.data, args.cv_label),
        "history": history,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cv_label": args.cv_label,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    shipped = export_scorer(bundle, args.out)

    q = bundle["reference"]["score"]["clean_quantiles"]
    print(
        f"saved {args.out}, and {shipped} for the app\n"
        f"clean players' score: median {q['0.5']:.3f}, 90th pct {q['0.9']:.3f}, "
        f"99th pct {q['0.99']:.3f}; banned players' median "
        f"{bundle['reference']['score']['cheater_median']:.3f}"
    )


if __name__ == "__main__":
    main()
