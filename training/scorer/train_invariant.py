"""Train the scorer to ignore weapon choice, and measure whether it worked.

Run:  uv run python training/scorer/train_invariant.py --strengths 0 0.3 1.0

The problem this addresses: in CS2CD, cheaters take 60% of their kills with
snipers and clean players 15%. A model can score well by learning sniper play,
and then it flags clean AWP mains — measured, the score of a *clean* player
correlates with their sniper share at r=0.63, and the clean players flagged at the
95th percentile have a median sniper share of 0.83.

A gradient-reversal head fights that: it predicts the weapon class from the same
features while pushing the encoder to make that prediction *harder*.

Success is not a higher AUC. Success is the same AUC with a lower correlation, so
the reported metrics are: overall accuracy, the correlation among clean players,
and accuracy split by weapon group.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from overwatch.layers.l3_behavior.evaluation import report
from overwatch.layers.l3_behavior.models import InvariantFlickCNN

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

SNIPERS = ("awp", "ssg08", "scar20", "g3sg1")
RIFLES = ("ak47", "m4a1", "m4a1_silencer", "galilar", "famas", "aug", "sg556")
SEED = 0


def weapon_classes(uids: np.ndarray) -> np.ndarray:
    """0 = sniper, 1 = rifle, 2 = anything else, per window."""
    windows = pl.read_parquet(
        PROCESSED / "windows.parquet", columns=["window_uid", "weapon"]
    ).with_columns(
        weapon_class=pl.when(pl.col("weapon").is_in(SNIPERS))
        .then(0)
        .when(pl.col("weapon").is_in(RIFLES))
        .then(1)
        .otherwise(2)
    )
    lookup = dict(zip(windows["window_uid"], windows["weapon_class"], strict=True))
    return np.array([lookup.get(uid, 2) for uid in uids], dtype=np.int64)


def train_fold(
    x: np.ndarray,
    y: np.ndarray,
    weapons: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    strength: float,
    epochs: int,
    device: str,
) -> nn.Module:
    torch.manual_seed(SEED)
    model = InvariantFlickCNN(in_channels=x.shape[1]).to(device)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x[train_idx]),
            torch.from_numpy(y[train_idx].astype(np.float32)),
            torch.from_numpy(weapons[train_idx]),
        ),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )
    positives = y[train_idx]
    pos_weight = torch.tensor([(1 - positives).sum() / max(positives.sum(), 1)]).to(
        device
    )
    cheat_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weapon_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state, best_pr = None, -1.0
    for epoch in range(1, epochs + 1):
        # ramp the adversary in: fighting it from step one destabilises training
        ramp = strength * min(1.0, epoch / max(epochs * 0.3, 1))
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            cheat, weapon = model(xb, ramp)
            loss = cheat_loss(cheat, yb) + weapon_loss(weapon, wb)
            loss.backward()
            optimizer.step()

        scores = predict(model, x[val_idx], device)
        pr = report("val", y[val_idx], scores, ci=False)["pr_auc"]
        if pr > best_pr:
            best_pr = pr
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(
    model: nn.Module, x: np.ndarray, device: str, batch: int = 1024
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.from_numpy(x[start : start + batch]).to(device)
            out.append(torch.sigmoid(model(chunk)[0]).cpu().numpy())
    return np.concatenate(out)


def evaluate(
    scores: np.ndarray, data: dict, weapons: np.ndarray, strength: float
) -> dict:
    """Accuracy, and whether the score still tracks weapon choice."""
    players = (
        pl.DataFrame(
            {
                "player": data["players"],
                "label": data["y"].astype(int),
                "score": scores,
                "sniper": (weapons == 0).astype(float),
            }
        )
        .group_by("player")
        .agg(
            label=pl.col("label").max(),
            score=pl.col("score").mean(),
            sniper_share=pl.col("sniper").mean(),
            n=pl.len(),
        )
        .filter(pl.col("n") >= 5)
    )
    y = players["label"].to_numpy()
    s = players["score"].to_numpy()
    share = players["sniper_share"].to_numpy()

    clean = players.filter(pl.col("label") == 0)
    row = report(f"strength {strength}", y, s, ci=False)
    row["r_clean_vs_sniper"] = float(
        np.corrcoef(clean["score"].to_numpy(), clean["sniper_share"].to_numpy())[0, 1]
    )
    for name, mask in (("awp_mains", share > 0.5), ("rifle_players", share < 0.2)):
        subset_y, subset_s = y[mask], s[mask]
        row[f"auc_{name}"] = (
            report(name, subset_y, subset_s, ci=False)["roc_auc"]
            if len(set(subset_y)) > 1
            else float("nan")
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.5, 1.5])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    cached = np.load(args.data / "sequences.npz", allow_pickle=True)
    data = {k: cached[k] for k in ("x", "y", "groups", "players", "uids")}
    weapons = weapon_classes(data["uids"])
    print(
        f"{len(weapons):,} windows: "
        f"{(weapons == 0).mean():.0%} sniper, {(weapons == 1).mean():.0%} rifle",
        flush=True,
    )

    rows = []
    for strength in args.strengths:
        started = time.perf_counter()
        oof = np.zeros(len(data["y"]))
        for train_idx, test_idx in GroupKFold(n_splits=args.folds).split(
            data["x"], data["y"], data["groups"]
        ):
            cut = int(len(train_idx) * 0.85)
            model = train_fold(
                data["x"],
                data["y"],
                weapons,
                train_idx[:cut],
                train_idx[cut:],
                strength=strength,
                epochs=args.epochs,
                device=args.device,
            )
            oof[test_idx] = predict(model, data["x"][test_idx], args.device)
        row = evaluate(oof, data, weapons, strength)
        rows.append(row)
        print(
            f"strength {strength}: ROC-AUC {row['roc_auc']:.3f}, "
            f"r(clean, sniper) {row['r_clean_vs_sniper']:.3f}, "
            f"AWP mains {row['auc_awp_mains']:.3f}, rifles {row['auc_rifle_players']:.3f} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        np.save(args.data / f"oof_invariant_{strength}.npy", oof)

    with pl.Config(float_precision=3, tbl_width_chars=170):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
