"""Compare cheater and clean kill windows: a figure and a separation table.

Run:  uv run python training/scorer/compare_labels.py

Reads data/processed/{window_ticks,window_features}.parquet, writes
docs/figures/cheater_vs_clean.png and prints one row per feature with its
single-feature AUC (how well that number alone separates cheater from clean).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from overwatch.layers.l3_behavior.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"

# Validated categorical slots 1 and 2 (see dataviz reference palette).
CLEAN_COLOR = "#2a78d6"
CHEATER_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
BAR_NEUTRAL = "#8a8880"  # not a series colour: these bars are not an entity


def single_feature_auc(values: np.ndarray, is_positive: np.ndarray) -> float:
    """Probability that a random cheater window scores above a random clean one.

    0.5 means the feature says nothing; 1.0 or 0.0 means it separates perfectly
    (0.0 = perfectly separating in the other direction).
    """
    mask = ~np.isnan(values)
    values, is_positive = values[mask], is_positive[mask]
    n_pos, n_neg = int(is_positive.sum()), int((~is_positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # average ranks within ties, so ties count as coin flips
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return (ranks[is_positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def separation_table(features: pl.DataFrame) -> pl.DataFrame:
    """Median per label plus the single-feature AUC, sorted by separation."""
    pairs = features.filter(pl.col("label").is_in(["cheater", "clean"]))
    is_cheater = (pairs["label"] == "cheater").to_numpy()

    rows = []
    for name in FEATURE_COLUMNS:
        column = pairs[name].cast(pl.Float64).to_numpy()
        auc = single_feature_auc(column, is_cheater)
        rows.append(
            {
                "feature": name,
                "cheater_median": pairs.filter(pl.col("label") == "cheater")[
                    name
                ].median(),
                "clean_median": pairs.filter(pl.col("label") == "clean")[name].median(),
                "auc": auc,
                "separation": abs(auc - 0.5),
            }
        )
    return pl.DataFrame(rows).sort("separation", descending=True)


def _band(window_ticks: pl.DataFrame, label: str) -> pl.DataFrame:
    return (
        window_ticks.filter(pl.col("label") == label)
        .group_by("t_ms")
        .agg(
            lo=pl.col("yaw_speed").median(),
            mid=pl.col("yaw_speed").mean(),
            hi=pl.col("yaw_speed").quantile(0.9),
        )
        .sort("t_ms")
    )


def make_figure(
    window_ticks: pl.DataFrame, table: pl.DataFrame, out_path: Path
) -> None:
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.7, 1]}
    )

    # --- left: aim speed through the engagement -----------------------------
    for label, color in (("clean", CLEAN_COLOR), ("cheater", CHEATER_COLOR)):
        band = _band(window_ticks, label)
        x = band["t_ms"].to_numpy()
        ax.fill_between(x, band["lo"], band["hi"], color=color, alpha=0.16, linewidth=0)
        ax.plot(x, band["mid"].to_numpy(), color=color, linewidth=2, label=label)
        # direct label at the right edge, in ink rather than series colour
        ax.annotate(
            label,
            xy=(x[-1], band["mid"][-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
        )

    ax.axvline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.annotate(
        "kill",
        xy=(0, ax.get_ylim()[1]),
        xytext=(4, -10),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=9,
    )
    ax.set_xlabel("time relative to kill (ms)", color=INK_MUTED)
    ax.set_ylabel("yaw speed: mean, band = median to p90 (°/s)", color=INK_MUTED)
    ax.set_title("How the crosshair moves into a kill", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left")

    # --- right: which single features separate the two groups ---------------
    top = table.head(8).reverse()
    y = np.arange(top.height)
    ax2.barh(y, top["separation"].to_numpy(), color=BAR_NEUTRAL, height=0.6)
    ax2.set_yticks(y, top["feature"].to_list(), fontsize=9)
    for i, (sep, auc) in enumerate(zip(top["separation"], top["auc"], strict=True)):
        ax2.annotate(
            f"AUC {auc:.2f} {'↑' if auc > 0.5 else '↓'}",
            xy=(sep, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_MUTED,
        )
    ax2.set_xlim(0, max(0.25, float(table["separation"].max()) * 1.35))
    ax2.set_xlabel("separation  |AUC − 0.5|   (↑ higher in cheaters)", color=INK_MUTED)
    ax2.set_title("What each feature is worth alone", color=INK, loc="left")

    for axis in (ax, ax2):
        axis.grid(axis="both", color="#e6e5e1", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#d5d4cf")
        axis.tick_params(colors=INK_MUTED, labelsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    window_ticks = pl.read_parquet(PROCESSED / "window_ticks.parquet")
    features = pl.read_parquet(PROCESSED / "window_features.parquet")

    table = separation_table(features)
    counts = features.group_by("label").len().sort("label")

    with pl.Config(tbl_rows=20, float_precision=3):
        print(counts)
        print(table)

    table.write_parquet(PROCESSED / "feature_separation.parquet")
    make_figure(window_ticks, table, FIGURES / "cheater_vs_clean.png")
    print(f"wrote {FIGURES / 'cheater_vs_clean.png'}")


if __name__ == "__main__":
    main()
