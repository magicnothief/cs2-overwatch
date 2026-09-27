# Design D spike: is "crosshair on an unseen enemy" evidence? — 2026-09-27

**Answer: not as it stands. It separates overall, but mostly by sniper use: clean
AWP players trip it as often as the line lets through cheaters, and among riflers
it catches few. Not worth a judge line on its own; a candidate for the v6
evidence redesign, with sniper-aware lines, and only after a pro check.**

Spike, brainstormed as design D ("unusual pre-aims"), started from the replay's
marks gate (`docs/evals/2026-09-27-replay-marks-gate.md`), where cheaters had
twice the clean median count. No product code.

## Fixed before looking

- **Measure:** per player per match, the number of stretches where the crosshair
  stays within 5 deg of the head of an enemy nobody on the player's team can see
  (mesh ray cast plus field of view), for 0.5 s or more, while the enemy's
  bearing turns 10 deg or more. Exactly the replay gate's rule
  (`training/replay/marks_gate.py`, `find_marks`), counted like `fast_kills` and
  `snap_kills`.
- **Evidence if:** (1) at least 8% of cheaters past the clean 99% line, the level
  `snap_kills` reached; (2) it holds among AWP players; (3) at most 5% of pros
  past the clean 95% line.

## Numbers

CS2CD, 685 matches with a map mesh, 6,849 player-matches (the gate's table,
`data/processed/replay_marks_gate.parquet`). Clean lines: 95% at 8 marks, 99% at
12.

| | players | past 95% | past 99% | median | p90 |
|---|---|---|---|---|---|
| clean | 5,564 | 3.76% | 0.99% | 2 | 6 |
| cheater | 1,285 | 21.56% | **12.22%** | 4 | 14 |

Criterion 1 passes. By sniper share (players with 5+ kills, joined from
`player_features.parquet`, 4,199 of 6,849), against the same lines:

| sniper share | clean past 95 | cheater past 95 | lift | clean past 99 | cheater past 99 |
|---|---|---|---|---|---|
| < 20% | 3.61% (of 2,218) | 11.71% (of 205) | 3.25x | 0.63% | 3.90% |
| 20-50% | 6.80% (of 618) | 16.46% (of 164) | 2.42x | 1.78% | 9.15% |
| >= 50%, AWP main | **16.77%** (of 322) | 31.85% (of 672) | **1.90x** | **8.70%** | 19.64% |

Criterion 2 fails. A clean AWP main is past the 95% line one time in six and
past the 99% line one time in eleven: holding an angle scoped *is* keeping the
crosshair on where an unseen enemy stands. CS2CD's cheaters take 60% of their
kills with snipers, so the overall 12% is mostly this band; among riflers the
measure reaches 3.9% of cheaters at the 99% line.

Criterion 3 was not run: with criterion 2 failing, a pro sample (where every team
fields an AWPer) could only make it worse.

## What it would take

- **Sniper-aware lines**, as target rule 3 already does for the visibility
  measurements: lines per sniper-share band, re-derived from clean players.
  Within the AWP band the lift would stay near 2x.
- **Counting only rifle moments** would be cleaner, but the tick tables carry no
  held weapon (CS2CD has none), so it needs a parser change and does not apply
  to the training data.
- **A pro check** (35 matches, as for v5) before any judge sees it.

Worth doing only together with the arrival redesign, as one v6 evidence change
with one pro sample and one retrain.

## Reproduce

    t = pl.read_parquet("data/processed/replay_marks_gate.parquet")
    clean = t.filter(pl.col("label") == "clean")["marks"]
    l95, l99 = clean.quantile(0.95, "higher"), clean.quantile(0.99, "higher")
    # bands: join player_features.parquet on (match_id, player_id), sniper_share
