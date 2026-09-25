# training/scorer (Layer 3 models)

Training scripts and experiment configs for the Layer 3 scorer. Run locally, or on Colab for
the sequence models.

- Track every run: data version, features, split seed, and metrics. A CSV is enough to start
  with; add MLflow later if you want.
- Always report the same metric set: ROC-AUC, PR-AUC, TPR@1%FPR, Brier, reliability diagram.
- Keep the split definition (which players/matches go in test) in a versioned file. Never
  regenerate it by accident.
- Suggested order: LightGBM on CS2CD features → calibrate → sequence model on the Kaggle CS:GO
  windows (learning) → sequence model on CS2CD windows → compare.
