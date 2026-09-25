# 0006: Keep the simple CNN; stop tuning on one split

- **Date:** 2026-09-22
- **Status:** accepted

## Context
Four changes were tried against the window CNN (width 32, avg+max pooling, 161-tick
windows, scores averaged per player):

| variant | single split | 5-fold out-of-fold |
|---|---|---|
| baseline | 0.948 | **0.938** |
| width 64 + attention pooling | 0.928 | 0.933 |
| 257-tick windows | 0.935 | 0.935 |
| multiple-instance (train on players) | 0.929 | not run |

On one split the variants looked like losses of 0.013-0.020. Cross-validated over
5,152 players they are all within 0.005 of the baseline, against a fold-to-fold
standard deviation of 0.006-0.009. The single split was ranking noise.

The same split also flattered the headline: TPR at a 1% false-positive budget read
41% on the split and 18% out of fold, because at that operating point the metric
rested on about six clean players.

## Decision
Keep the baseline architecture. Report cross-validated numbers only, and treat any
difference below ~0.02 ROC-AUC as unmeasured until it is cross-validated.

Multiple-instance learning stays shelved: training on ~4,000 player labels has an
order of magnitude less signal than 56,000 window labels, and averaging window
scores is already a form of instance aggregation. Its one advantage was calibration
(Brier 0.076 vs 0.104), which belongs to the calibration step rather than the
architecture.

## Consequences
- Accuracy gains now have to come from *signal* (Layer 2 added 0.07) rather than
  architecture, which has been flat across four attempts.
- Every future model claim needs `cross_validate.py`, which costs about 70 seconds.
- The 257-tick dataset is kept (`data/processed_256/`) since it is equal on accuracy
  and gives Layer 4 more context around a moment to show a reviewer.
