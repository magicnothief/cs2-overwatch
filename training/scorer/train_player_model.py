"""Train and honestly evaluate a per-player cheat scorer.

Run:  uv run python training/scorer/train_player_model.py
      uv run python training/scorer/train_player_model.py --features-only

Splits are grouped by match, never by player or window: two players from the
same match share a server, a map and an opponent pool, so putting one in train
and one in test would flatter the score.

Everything reported comes from out-of-fold predictions — every player is scored
by a model that never saw their match.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

from overwatch.layers.l3_behavior.player_features import (
    PLAYER_FEATURE_COLUMNS,
    build_player_features,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"

MODEL_COLOR = "#2a78d6"  # categorical slot 1
BASELINE_COLOR = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK_MUTED = "#52514e"
BAR_NEUTRAL = "#8a8880"

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 0,
}
N_ROUNDS = 300


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Recall available when false positives are capped at max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    allowed = fpr <= max_fpr
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def out_of_fold_scores(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """Score every player with a model trained on other matches."""
    oof = np.zeros(len(labels))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
        features, labels, groups
    ):
        booster = lgb.train(
            PARAMS,
            lgb.Dataset(features[train_idx], label=labels[train_idx]),
            num_boost_round=N_ROUNDS,
        )
        oof[test_idx] = booster.predict(features[test_idx])
    return oof


def bootstrap_auc_ci(
    y: np.ndarray, scores: np.ndarray, *, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """90% interval for ROC-AUC, by resampling players with replacement.

    With ~100 players the headline number is noisy; this says how noisy.
    """
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95))


def report(name: str, y: np.ndarray, scores: np.ndarray) -> dict:
    lo, hi = bootstrap_auc_ci(y, scores)
    row = {
        "model": name,
        "roc_auc": roc_auc_score(y, scores),
        "auc_ci90": f"{lo:.2f}-{hi:.2f}",
        "pr_auc": average_precision_score(y, scores),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, scores, 0.01),
        "tpr_at_10pct_fpr": tpr_at_fpr(y, scores, 0.10),
    }
    if scores.min() >= 0 and scores.max() <= 1:
        row["brier"] = brier_score_loss(y, scores)
    return row


def make_figure(
    y: np.ndarray,
    oof: np.ndarray,
    baseline: np.ndarray,
    shap_ranking: pl.DataFrame,
    out: Path,
) -> None:
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.2]}
    )

    for scores, label, color in (
        (oof, "model (out of fold)", MODEL_COLOR),
        (baseline, "straight_share alone", BASELINE_COLOR),
    ):
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{label} — AUC {roc_auc_score(y, scores):.2f}",
        )
    ax.plot([0, 1], [0, 1], color=BAR_NEUTRAL, linewidth=1, linestyle="--")
    ax.set_xlabel("false positive rate: clean players accused", color=INK_MUTED)
    ax.set_ylabel("true positive rate: cheaters caught", color=INK_MUTED)
    ax.set_title("Per-player detection, grouped by match", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    top = shap_ranking.head(10).reverse()
    y_pos = np.arange(top.height)
    ax2.barh(y_pos, top["mean_abs_shap"].to_numpy(), color=BAR_NEUTRAL, height=0.6)
    ax2.set_yticks(y_pos, top["feature"].to_list(), fontsize=9)
    ax2.set_xlabel("mean |SHAP|: contribution to the score", color=INK_MUTED)
    ax2.set_title("What the model actually used", color=INK, loc="left")

    for axis in (ax, ax2):
        axis.grid(color="#e6e5e1", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#d5d4cf")
        axis.tick_params(colors=INK_MUTED, labelsize=9)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="write player_features.parquet and stop (no model, no figures)",
    )
    features_only = parser.parse_args().features_only
    # the weapon only feeds sniper_share, which is context for the judge and is
    # kept out of the model's inputs (CONTEXT_COLUMNS)
    window_features = pl.read_parquet(PROCESSED / "window_features.parquet").join(
        pl.read_parquet(
            PROCESSED / "windows.parquet", columns=["window_uid", "weapon"]
        ),
        on="window_uid",
        how="left",
    )
    window_ticks = pl.read_parquet(PROCESSED / "window_ticks.parquet")

    players = build_player_features(window_features, window_ticks).filter(
        pl.col("label").is_in(["cheater", "clean"])
    )
    players.write_parquet(PROCESSED / "player_features.parquet")
    if features_only:
        print(f"wrote {PROCESSED / 'player_features.parquet'} ({players.height} players)")
        return

    feature_names = [c for c in PLAYER_FEATURE_COLUMNS if c in players.columns]
    x = players.select(feature_names).to_numpy().astype(float)
    y = (players["label"] == "cheater").to_numpy().astype(int)
    groups = players["match_id"].to_numpy()

    print(
        f"{len(y)} players from {len(set(groups))} matches: "
        f"{y.sum()} cheater, {len(y) - y.sum()} clean\n"
    )

    oof = out_of_fold_scores(x, y, groups)
    baseline = players["straight_share"].fill_null(0).to_numpy()

    results = pl.DataFrame(
        [report("lightgbm (oof)", y, oof), report("straight_share", y, baseline)]
    )
    with pl.Config(float_precision=3, tbl_width_chars=120):
        print(results)

    booster = lgb.train(PARAMS, lgb.Dataset(x, label=y), num_boost_round=N_ROUNDS)
    import shap  # imported here: only the training script needs it

    shap_values = shap.TreeExplainer(booster).shap_values(x)
    ranking = pl.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort("mean_abs_shap", descending=True)
    print(ranking)

    players.with_columns(score=oof).write_parquet(PROCESSED / "player_scores.parquet")
    make_figure(y, oof, baseline, ranking, FIGURES / "player_model.png")
    print(f"\nwrote {FIGURES / 'player_model.png'}")


if __name__ == "__main__":
    main()
