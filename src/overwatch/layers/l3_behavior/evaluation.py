"""Metrics shared by every scorer, so numbers stay comparable across models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def split_by_match(groups: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """70/15/15 split over matches, not windows or players.

    Windows from one match share a server, a map and an opponent pool, so any
    split finer than a match leaks information into the test set.
    """
    matches = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)
    n_train, n_val = int(0.7 * len(matches)), int(0.85 * len(matches))
    assignment = {
        "train": set(matches[:n_train]),
        "val": set(matches[n_train:n_val]),
        "test": set(matches[n_val:]),
    }
    return {
        name: np.array([i for i, g in enumerate(groups) if g in members])
        for name, members in assignment.items()
    }


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Recall available when false positives are capped at max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    allowed = fpr <= max_fpr
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def bootstrap_auc_ci(
    y: np.ndarray, scores: np.ndarray, *, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """90% interval for ROC-AUC, by resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95))


def report(name: str, y: np.ndarray, scores: np.ndarray, *, ci: bool = True) -> dict:
    """One row of the comparison table: the same metrics for every model."""
    row: dict[str, object] = {"model": name, "roc_auc": roc_auc_score(y, scores)}
    if ci:
        lo, hi = bootstrap_auc_ci(y, scores)
        row["auc_ci90"] = f"{lo:.2f}-{hi:.2f}"
    row["pr_auc"] = average_precision_score(y, scores)
    row["tpr_at_1pct_fpr"] = tpr_at_fpr(y, scores, 0.01)
    row["tpr_at_10pct_fpr"] = tpr_at_fpr(y, scores, 0.10)
    if scores.min() >= 0 and scores.max() <= 1:
        row["brier"] = brier_score_loss(y, scores)
    return row
