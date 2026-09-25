# Layer 3: behavior → calibrated probability

**Purpose:** most of your detection power lives here. It has two parts:
- `features/`: engagement windows → numbers.
- `models/`: numbers → calibrated probability, plus SHAP explanations.

Feature ideas are in `docs/ARCHITECTURE.md` §4.

**Learn first:** numerical derivatives of noisy signals (smoothing), windowing, gradient
boosting, `GroupKFold` (group by player!), class imbalance, calibration, SHAP.

**First exercise (a notebook, before any model):**
1. Take one legit pro flick and one flick from a verified CS2CD cheater.
2. Plot angular velocity, acceleration, and crosshair-to-head error for both, side by side.
3. **Write down 5 hypotheses** about what differs. Each hypothesis becomes a feature.

**Model progression:** thresholds → LightGBM → sequence model (1D-CNN/GRU/transformer).
Keep the older model as a baseline every time.

**Done when:** you have an evaluation report (ROC-AUC, PR-AUC, TPR@1%FPR, reliability diagram)
on a **player-grouped** test split, and it beats "headshot % only".

**Pitfalls:**
- The same player in both train and test (leakage).
- Treating player-level labels as moment-level truth. Read about multiple-instance learning.
- Skill as a confounder: good players look "suspicious" to naive features.
