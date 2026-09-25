"""Judge the feature model and the sequence model on identical held-out matches.

Run:  uv run python training/scorer/compare_models.py

train_player_model.py reports cross-validated numbers over all matches, and
train_sequence_model.py reports one held-out split, so their headline figures are
not comparable. This script puts every model on the same split, the same players
and the same metrics, and also asks whether the two disagree usefully enough to
be worth combining.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_curve

from overwatch.layers.l3_behavior.evaluation import report, split_by_match
from overwatch.layers.l3_behavior.features import FEATURE_COLUMNS
from overwatch.layers.l3_behavior.player_features import (
    PLAYER_FEATURE_COLUMNS,
    build_player_features,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"

FEATURE_MODEL_COLOR = "#2a78d6"
SEQUENCE_MODEL_COLOR = "#eb6834"
BLEND_COLOR = "#1baf7a"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
BAR_NEUTRAL = "#8a8880"

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 0,
}


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map scores onto 0..1 by rank, so two models can be averaged fairly."""
    order = values.argsort().argsort()
    return order / max(len(values) - 1, 1)


def main() -> None:
    windows = pl.read_parquet(PROCESSED / "window_features.parquet").filter(
        pl.col("label").is_in(["cheater", "clean"])
    )
    cnn_scores = pl.read_parquet(PROCESSED / "cnn_window_scores.parquet")

    # The same split the CNN used: by match, same seed.
    groups = windows["match_id"].to_numpy()
    splits = split_by_match(groups)
    split_of = np.empty(len(groups), dtype=object)
    for name, idx in splits.items():
        split_of[idx] = name
    windows = windows.with_columns(split=pl.Series(split_of))

    train = windows.filter(pl.col("split") != "test")
    test = windows.filter(pl.col("split") == "test")
    print(
        f"train {train.height:,} windows / {train['match_id'].n_unique()} matches, "
        f"test {test.height:,} windows / {test['match_id'].n_unique()} matches"
    )

    # --- feature model, per window -----------------------------------------
    x_train = train.select(FEATURE_COLUMNS).to_numpy().astype(float)
    y_train = (train["label"] == "cheater").to_numpy().astype(int)
    x_test = test.select(FEATURE_COLUMNS).to_numpy().astype(float)

    window_model = lgb.train(
        PARAMS, lgb.Dataset(x_train, label=y_train), num_boost_round=400
    )
    lgb_window_scores = window_model.predict(x_test)

    test_with_scores = (
        test.with_columns(lgb_score=pl.Series(lgb_window_scores))
        .join(cnn_scores, on="window_uid", how="inner")
        .rename({"score": "cnn_score"})
    )

    # Both per-window rows are read off the joined frame, so every score is
    # paired with its own label rather than with a row at the same position.
    y_joined = (test_with_scores["label"] == "cheater").to_numpy().astype(int)
    rows = [
        report(
            "per window: feature LightGBM",
            y_joined,
            test_with_scores["lgb_score"].to_numpy(),
        ),
        report(
            "per window: sequence CNN",
            y_joined,
            test_with_scores["cnn_score"].to_numpy(),
        ),
    ]

    # --- per player ---------------------------------------------------------
    per_player = (
        test_with_scores.group_by("match_id", "player_id")
        .agg(
            label=(pl.col("label") == "cheater").max().cast(pl.Int8),
            n_kills=pl.len(),
            lgb_window_mean=pl.col("lgb_score").mean(),
            cnn_mean=pl.col("cnn_score").mean(),
            cnn_topk=pl.col("cnn_score").sort(descending=True).head(5).mean(),
        )
        .filter(pl.col("n_kills") >= 5)
    )

    # the player-level feature model, trained on the same training matches
    player_train = build_player_features(train).filter(
        pl.col("label").is_in(["cheater", "clean"])
    )
    player_test = build_player_features(test).filter(
        pl.col("label").is_in(["cheater", "clean"])
    )
    feature_names = [c for c in PLAYER_FEATURE_COLUMNS if c in player_train.columns]
    player_model = lgb.train(
        PARAMS,
        lgb.Dataset(
            player_train.select(feature_names).to_numpy().astype(float),
            label=(player_train["label"] == "cheater").to_numpy().astype(int),
        ),
        num_boost_round=400,
    )
    player_test = player_test.with_columns(
        lgb_player=pl.Series(
            player_model.predict(
                player_test.select(feature_names).to_numpy().astype(float)
            )
        )
    )

    players = per_player.join(
        player_test.select("match_id", "player_id", "lgb_player"),
        on=["match_id", "player_id"],
        how="inner",
    )
    y_player = players["label"].to_numpy()

    blend = (
        rank_normalize(players["lgb_player"].to_numpy())
        + rank_normalize(players["cnn_mean"].to_numpy())
    ) / 2

    rows += [
        report(
            "per player: feature LightGBM", y_player, players["lgb_player"].to_numpy()
        ),
        report(
            "per player: sequence CNN (mean)", y_player, players["cnn_mean"].to_numpy()
        ),
        report(
            "per player: sequence CNN (top 5)", y_player, players["cnn_topk"].to_numpy()
        ),
        report("per player: both, rank-averaged", y_player, blend),
    ]

    with pl.Config(float_precision=3, tbl_width_chars=160, fmt_str_lengths=40):
        print(pl.DataFrame(rows))

    print(
        "\nagreement between the two player scores (Spearman): "
        f"{np.corrcoef(rank_normalize(players['lgb_player'].to_numpy()), rank_normalize(players['cnn_mean'].to_numpy()))[0, 1]:.3f}"
    )
    print(f"{players.height} players in the test matches, {y_player.sum()} cheaters")

    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for scores, label, color in (
        (players["lgb_player"].to_numpy(), "feature model", FEATURE_MODEL_COLOR),
        (players["cnn_mean"].to_numpy(), "sequence model", SEQUENCE_MODEL_COLOR),
        (blend, "both, rank-averaged", BLEND_COLOR),
    ):
        fpr, tpr, _ = roc_curve(y_player, scores)
        auc = report(label, y_player, scores, ci=False)["roc_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} — AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=BAR_NEUTRAL, linewidth=1, linestyle="--")
    ax.set_xlabel("false positive rate: clean players accused", color=INK_MUTED)
    ax.set_ylabel("true positive rate: cheaters caught", color=INK_MUTED)
    ax.set_title("Per player, on the same held-out matches", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d5d4cf")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "model_comparison.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    players.with_columns(blend=pl.Series(blend)).write_parquet(
        PROCESSED / "model_comparison.parquet"
    )
    print(f"wrote {FIGURES / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
