"""Train the per-player model directly, instead of averaging per-window scores.

Run:  uv run python training/scorer/train_mil_model.py --data data/processed

Every score so far was produced per window and then averaged into a player. But
the label *is* a player: a banned account cheated in some kills, not all of them.
This trains on that directly — a bag of one player's windows, one label, and an
attention layer that decides which windows carry the verdict.

The attention weights are the useful side effect: they name which kills the model
objected to, which is exactly what a reviewer (and Layer 4) needs.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from overwatch.layers.l3_behavior.evaluation import report, split_by_match
from overwatch.layers.l3_behavior.models import PlayerMIL

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "scorer"
SEED = 0


class PlayerBags(Dataset):
    """One item per player: up to `bag_size` of their windows, plus a mask.

    During training the windows are resampled every epoch, which acts as
    augmentation — the model sees different subsets of the same player. At
    evaluation the selection is deterministic so the score is reproducible.
    """

    def __init__(
        self,
        x: np.ndarray,
        players: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        bag_size: int = 16,
        train: bool = True,
        min_windows: int = 5,
    ) -> None:
        self.x, self.bag_size, self.train = x, bag_size, train

        by_player: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            by_player[players[i]].append(int(i))

        self.players = [p for p, rows in by_player.items() if len(rows) >= min_windows]
        self.rows = [by_player[p] for p in self.players]
        self.labels = np.array(
            [labels[rows[0]] for rows in self.rows], dtype=np.float32
        )
        self.rng = np.random.default_rng(SEED)

    def __len__(self) -> int:
        return len(self.players)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self.rows[index]
        if len(rows) > self.bag_size:
            chosen = (
                self.rng.choice(rows, self.bag_size, replace=False)
                if self.train
                # deterministic, spread over the match rather than the first few
                else np.array(rows)[
                    np.linspace(0, len(rows) - 1, self.bag_size).astype(int)
                ]
            )
        else:
            chosen = np.array(rows)

        bag = np.zeros((self.bag_size, *self.x.shape[1:]), dtype=np.float32)
        mask = np.zeros(self.bag_size, dtype=np.float32)
        bag[: len(chosen)] = self.x[chosen]
        mask[: len(chosen)] = 1.0
        return (
            torch.from_numpy(bag),
            torch.from_numpy(mask),
            torch.tensor(self.labels[index]),
        )


def evaluate(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for bags, mask, y in loader:
            logits, _ = model(bags.to(device), mask.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cached = np.load(args.data / "sequences.npz", allow_pickle=True)
    x, y, groups, players = (cached[k] for k in ("x", "y", "groups", "players"))
    splits = split_by_match(groups)

    loaders = {}
    for name, indices in splits.items():
        dataset = PlayerBags(
            x, players, y, indices, bag_size=args.bag_size, train=(name == "train")
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=32,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
        )
        print(
            f"  {name:5s} {len(dataset):5d} players, "
            f"{dataset.labels.mean():.1%} cheater",
            flush=True,
        )

    torch.manual_seed(SEED)
    model = PlayerMIL(in_channels=x.shape[1], width=args.width).to(args.device)
    positives = loaders["train"].dataset.labels
    pos_weight = torch.tensor([(1 - positives).sum() / max(positives.sum(), 1)]).to(
        args.device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state, best_pr, since_best = None, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        for bags, mask, labels in loaders["train"]:
            bags, mask, labels = (t.to(args.device) for t in (bags, mask, labels))
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(bags, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        scheduler.step()

        scores, labels = evaluate(model, loaders["val"], args.device)
        row = report("val", labels, scores, ci=False)
        print(
            f"epoch {epoch:2d}  loss {total / len(loaders['train'].dataset):.4f}  "
            f"val ROC-AUC {row['roc_auc']:.3f}  val PR-AUC {row['pr_auc']:.3f}  "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        if row["pr_auc"] > best_pr:
            best_pr, since_best = row["pr_auc"], 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            since_best += 1
            if since_best >= args.patience:
                print("stopping: no validation improvement", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)

    scores, labels = evaluate(model, loaders["test"], args.device)
    with pl.Config(float_precision=3, tbl_width_chars=140):
        print(pl.DataFrame([report("MIL, per player", labels, scores)]))

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"in_channels": x.shape[1], "width": args.width},
        },
        MODELS / f"player_mil{args.tag}.pt",
    )
    pl.DataFrame(
        {"player": loaders["test"].dataset.players, "score": scores, "label": labels}
    ).write_parquet(args.data / f"mil_player_scores{args.tag}.parquet")
    print(f"saved {MODELS / f'player_mil{args.tag}.pt'}")


if __name__ == "__main__":
    main()
