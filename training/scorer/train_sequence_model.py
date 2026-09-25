"""Train a 1D CNN on raw kill windows and compare it with the feature model.

Run:  uv run python training/scorer/train_sequence_model.py

The feature model sees 14 summary numbers per player. This one sees the curve:
161 ticks of turn per axis, and reads the shape itself. Both are judged on the
same held-out matches, per window and per player, so the comparison is fair.

Matches are split 70/15/15 into train/validation/test. Splitting by window or by
player would leak, since windows from one match share a server, a map and an
opponent pool.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from overwatch.layers.l3_behavior.evaluation import report, split_by_match
from overwatch.layers.l3_behavior.models import FlickCNN

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "scorer"

SEED = 0


def predict(
    model: nn.Module, x: torch.Tensor, device: str, batch: int = 1024
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = x[start : start + batch].to(device)
            out.append(torch.sigmoid(model(chunk)).cpu().numpy())
    return np.concatenate(out)


def per_player_scores(
    window_scores: np.ndarray,
    players: np.ndarray,
    labels: np.ndarray,
    *,
    top_k: int = 5,
) -> pl.DataFrame:
    """Aggregate window scores into one score per player.

    A cheater's ordinary kills look ordinary, so a mean over every window hides
    them. The mean of the k most suspicious windows keeps the evidence that
    matters while staying steadier than a single maximum.
    """
    frame = pl.DataFrame({"player": players, "score": window_scores, "label": labels})
    return (
        frame.sort("score", descending=True)
        .group_by("player", maintain_order=True)
        .agg(
            label=pl.col("label").max(),
            n_windows=pl.len(),
            score_mean=pl.col("score").mean(),
            score_max=pl.col("score").max(),
            score_topk=pl.col("score").head(top_k).mean(),
        )
    )


def train(
    data: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    *,
    epochs: int,
    device: str,
    width: int = 32,
    pooling: str = "avgmax",
    lr: float = 1e-3,
    batch: int = 256,
    patience: int = 4,
) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(SEED)
    x, y = data["x"], data["y"].astype(np.float32)

    x_train = torch.from_numpy(x[splits["train"]])
    y_train = torch.from_numpy(y[splits["train"]])
    x_val = torch.from_numpy(x[splits["val"]])
    y_val = y[splits["val"]]

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch, shuffle=True, drop_last=True
    )

    model = FlickCNN(in_channels=x.shape[1], width=width, pooling=pooling).to(device)
    # positives are the minority; weight them so the loss cannot ignore them
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(
        device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[dict] = []
    best_state, best_pr, since_best = None, -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(xb)
        scheduler.step()

        val_scores = predict(model, x_val, device)
        row = report("val", y_val, val_scores, ci=False)
        history.append(
            {
                "epoch": epoch,
                "loss": total / len(x_train),
                "val_roc_auc": row["roc_auc"],
                "val_pr_auc": row["pr_auc"],
                "seconds": time.time() - started,
            }
        )
        print(
            f"epoch {epoch:2d}  loss {history[-1]['loss']:.4f}  "
            f"val ROC-AUC {row['roc_auc']:.3f}  val PR-AUC {row['pr_auc']:.3f}  "
            f"({history[-1]['seconds']:.0f}s)"
        )

        if row["pr_auc"] > best_pr:
            best_pr, since_best = row["pr_auc"], 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            since_best += 1
            if since_best >= patience:
                print(f"no val improvement for {patience} epochs, stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--width", type=int, default=32, help="first conv width")
    parser.add_argument("--pooling", choices=("avgmax", "attention"), default="avgmax")
    parser.add_argument("--tag", default="", help="suffix for saved files")
    parser.add_argument(
        "--data", type=Path, default=PROCESSED, help="folder holding sequences.npz"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    cached = np.load(args.data / "sequences.npz", allow_pickle=True)
    data = {k: cached[k] for k in ("x", "y", "groups", "players", "uids")}
    splits = split_by_match(data["groups"])

    print(f"device: {args.device}")
    for name, idx in splits.items():
        print(
            f"  {name:5s} {len(idx):6d} windows, "
            f"{len(np.unique(data['groups'][idx])):3d} matches, "
            f"{data['y'][idx].mean():.1%} cheater"
        )

    model, history = train(
        data,
        splits,
        epochs=args.epochs,
        device=args.device,
        width=args.width,
        pooling=args.pooling,
    )

    test = splits["test"]
    test_scores = predict(model, torch.from_numpy(data["x"][test]), args.device)

    rows = [report("cnn, per window", data["y"][test], test_scores)]
    players = per_player_scores(test_scores, data["players"][test], data["y"][test])
    for column in ("score_mean", "score_max", "score_topk"):
        rows.append(
            report(
                f"cnn, per player ({column})",
                players["label"].to_numpy(),
                players[column].to_numpy(),
            )
        )

    with pl.Config(float_precision=3, tbl_width_chars=140):
        print(pl.DataFrame(rows))

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "history": history}, MODELS / "flick_cnn.pt"
    )
    players.with_columns(split=pl.lit("test")).write_parquet(
        PROCESSED / "cnn_player_scores.parquet"
    )
    pl.DataFrame(
        {"window_uid": data["uids"][test], "score": test_scores}
    ).write_parquet(PROCESSED / "cnn_window_scores.parquet")
    print(f"saved {MODELS / name}")


if __name__ == "__main__":
    main()
