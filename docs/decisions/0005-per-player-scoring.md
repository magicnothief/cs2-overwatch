# 0005: Score players, not kills

- **Date:** 2026-09-22
- **Status:** accepted

## Context
Labels come from bans, which are attached to an account, not to a moment. A banned
player did not cheat in every kill: cheats get toggled, and most of a cheater's
kills look ordinary. Measured on 20 CS2CD matches, the strongest per-kill feature
reaches AUC 0.71, while the same information aggregated per player reaches 0.88.

The single clearest evidence is rare: 5.3% of cheater kills have an impossible turn
on the kill tick, against 0.2% of clean kills. A per-kill median cannot see that;
a per-player maximum can.

## Options considered
1. **Score each kill, then average per player.** Simple, but averaging buries the
   rare impossible kill among ordinary ones.
2. **Score each player from aggregated kill features**, keeping shares and extremes
   (max, p90) alongside medians. This is multiple-instance learning by hand.
3. **A sequence model over raw ticks** (AntiCheatPT's approach). Stronger in
   principle; needs far more data than we have, and explains itself less readily.

## Decision
Option 2 for now. `player_features.py` aggregates with three kinds of statistic:
shares (how often a behaviour occurs), extremes (the worst kill), and medians (normal
play). Players with fewer than 5 kills are dropped as too noisy to score.

Evaluation uses out-of-fold predictions with GroupKFold over `match_id`. Splitting by
player would leak: players in one match share a server, a map and an opponent pool.

## Consequences
- The reported unit is a player in a match, which is also what a reviewer judges.
- Layer 4 will explain a player, citing their worst kills as evidence.
- Option 3 stays open: the window tables are already the right shape for it, and the
  per-player numbers become its baseline.
- Any feature that correlates with *playing well* (kill count, headshot rate) risks
  measuring skill. `n_kills` is in the feature set and worth watching in SHAP.
