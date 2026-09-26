# Redesigning `arrival_shot_share` and `snap_kills` — 2026-09-26

**Recommendation: do not land a redesign yet. The direction is settled and it is
a large improvement; the gate-3 number that would justify shipping it cannot be
measured on a 35-match pro sample. Enlarge the pro sample first — the run is in
flight.**

This is the follow-up to `docs/evals/2026-09-26-judge-v5-ship-gates.md`
(commit `5582696`), which failed gates 2 and 3 of
`docs/specs/2026-09-26-triggerbot-and-snap-count.md` section 4 and named the
small denominator as the cause. That diagnosis was right and incomplete. Two
further causes are measured below, and one of them means **gate 2 clause 1
cannot be met by redesigning `arrival_shot_share` at all**.

| finding | number | consequence |
|---|---|---|
| binomial lower bound, my own suggested fix | pros 25 of 341 (7.3314%) against v5's 22 (6.4516%) | **the suggestion was wrong; withdrawn** |
| raising `MIN_ARRIVAL_KILLS` | pro rate never falls on the shown denominator | passes only by hiding the measurement |
| pros' pooled arrival-shot rate | 9.64% against CS2CD clean 6.7555% | no estimator over (shots, arrivals) can pass |
| arrival counts the enemy walking into a held crosshair | clean fire on arrival at 35.6 deg/s crosshair speed, cheaters at 91.2 | **the real defect in the definition** |
| requiring the crosshair to have moved | CS2CD lift 4.41x to 10.32x | the redesign |
| one measurement past a 99% line is a cheating target | 4 of 341 pros (1.1730%) carry one | **gate 2 clause 1 is structurally unreachable** |
| `snap_kills` among AWP mains | clean 9.42% past the 1.0 line against cheaters 16.33%, lift 1.73 | sniper-confounded, ADR 0008 |
| power of the 35-match pro check | 1 of 117 shown, 95% CI [0.0216%, 4.6701%] | **gate 3 is not measurable on it** |

---

## What was measured, and on what

Nothing here involves the judge, the GPU or a fine-tune. Gate 3 is a
feature-level check — where a clean line falls and how many pros are past it —
so every number below is recomputed from tick data.

**CS2CD.** `data/processed/window_ticks.parquet`, 14,671,608 tick rows over
`data/processed/window_features.parquet`, 91,128 kill windows. Players with 5 or
more kills: 4,093 clean, 1,059 cheater (877 more are label `unknown` and are used
nowhere). Same corpus, same labels, same `MIN_KILLS = 5` as the v5 eval.

**Pros.** The same 35 HLTV matches as the v3, v4 and v5 pro checks —
`training/check_pro_demos.py` `sample(per_map=5, seed=0)` over the 1,988-demo
listing of `blanchon/cs2_dataset_demo` — re-parsed to per-arrival resolution.
341 players with enough kills, the same denominator as every previous run.

**Two new tools, both rerunnable, both self-checking.**

```
# every arrival in CS2CD with what moved to make it        110,658 rows
PYTHONPATH=src .venv/bin/python training/arrival/arrival_lab.py \
    data/processed/arrival_lab.parquet

# the same table on the pro sample, no judge, no GPU       ~15 s a demo
PYTHONPATH=src .venv/bin/python training/arrival/pro_arrival_lab.py \
    --per-map 5 --seed 0 --out data/processed/pro_arrival_lab

# candidates against gate 3, with the selection rule in the module docstring
.venv/bin/python training/arrival/gate3_arrival.py \
    --cs2cd data/processed/arrival_lab.parquet \
    --pro data/processed/pro_arrival_lab
```

These live under `training/`, not `tools/`, because `.gitignore:26` ignores
`tools/` whole — `tools/gate23_pro.py`, cited by the v5 eval, is not in the repo
and that eval cannot be rerun from a fresh clone. These can.

`training/arrival/arrival_lab.py` asserts that its table rebuilds
`shots.arrival_features` exactly — same values, same nulls, on all 91,128
windows. `pro_arrival_lab.py`'s output was joined against the 341 pros'
`arrival_kills` and `arrival_shots` as parsed out of the judge evidence text that
v5 actually read: **0 mismatching players of 341**, kill counts included. So the
new instrument sees what the v5 run saw, and a difference below is a real
difference and not a reimplementation artefact.

Environment: polars 1.44.2, numpy 2.5.3, scipy 1.18.1, python 3.12.14.

---

## 1. My suggested direction was wrong. Withdrawn.

The v5 eval suggested "the lower bound of a binomial confidence interval on the
share would put the 2-of-6 players back in the clean band while leaving 19-of-30
cheaters past it". That is true against a **fixed** 0.25 line. It is false once
the 95% line is re-derived as the clean quantile of the bound itself, which the
spec requires, because the bound is then very nearly a monotone re-map and the
same players come out past it.

| statistic, `MIN_ARRIVAL_KILLS = 5` | line95 | CS2CD clean past | CS2CD cheater past | pros past, of 341 |
|---|---|---|---|---|
| share (v5, shipped) | 0.2500 | 4.03% | 17.78% | 22 (6.4516%) |
| Wilson lower bound, 90% | 0.1089 | 4.66% | 18.37% | 24 (7.0381%) |
| Wilson lower bound, 95% | 0.0872 | 5.01% | 18.37% | 25 (7.3314%) |
| Wilson lower bound, 99% | 0.0603 | 5.01% | 18.37% | 25 (7.3314%) |
| Clopper-Pearson lower bound, 90% | 0.0769 | 5.01% | 18.52% | 25 (7.3314%) |
| Clopper-Pearson lower bound, 95% | 0.0534 | 4.35% | 17.33% | 16 (4.6921%) |

The best of them, Clopper-Pearson at 95%, still puts 4.69x the gate's ceiling of
pros past the line, and costs signal doing it (cheaters 17.33% against 17.78%).

## 2. Raising `MIN_ARRIVAL_KILLS` does not work, and the reason matters

| `MIN_ARRIVAL_KILLS` | line95 | shown to (CS2CD clean) | pros past, of 341 | pros past, of shown |
|---|---|---|---|---|
| 5 (v5) | 0.2500 | 62.4% | 22 (6.4516%) | 22 of 278 (7.9137%) |
| 6 | 0.2500 | 51.6% | 18 (5.2786%) | 18 of 235 (7.6596%) |
| 8 | 0.2222 | 33.0% | 18 (5.2786%) | 18 of 164 (10.9756%) |
| 10 | 0.2143 | 19.6% | 8 (2.3460%) | 8 of 89 (8.9888%) |
| 12 | 0.2308 | 10.9% | 3 (0.8798%) | 3 of 50 (6.0000%) |
| 15 | 0.2500 | 4.3% | 2 (0.5865%) | 2 of 19 (10.5263%) |

The of-all column falls; the of-shown column does not. `MIN = 12` meets gate 3's
1% only on the of-all reading, by withholding the measurement from 85% of pros.
That is passing the gate by hiding from it, so it is not proposed, and every
table in this document reports both denominators for the same reason.

**Negative result, recorded so it is not retried.** Moving the denominator to the
player's kill count (median 14, never below 5) instead of their arrival count
makes things worse, not better: lift falls from 4.41x to 3.09-3.77x and 67-88% of
the clean players past the line are there on exactly 2 arrival shots, because the
line drops faster than the denominator grows.

## 3. Why no estimator over (arrival shots, arrival kills) can pass

Pooled arrival-shot rate, over players with 5 or more arrivals:

| population | eligible players | arrival kills | arrival-tick shots | pooled rate |
|---|---|---|---|---|
| CS2CD clean | 2,554 | 21,982 | 1,485 | **6.7555%** |
| pros | 278 | 2,500 | 241 | **9.6400%** |
| CS2CD cheater | 675 | 6,689 | 965 | **14.4267%** |

Pros fire on the arrival tick 1.43x as often as matchmaking clean players — about
40% of the way from clean to cheater. A line at the clean 95th percentile
therefore puts **more** than 5% of pros past it however accurately the rate is
estimated. Gate 3 asks for 1%. The granularity artefact named in the v5 eval is
real and sits on top of this, not instead of it.

## 4. The defect is in the arrival definition, not the estimator

`shots.arrival_features` (`src/overwatch/layers/l3_behavior/shots.py:72-107`)
sets `on_head` from `target_angle`, the angle between the crosshair and the
victim's head. That angle closes when **either** the crosshair moves onto the
head **or** the head walks into the crosshair. The second case is a player
holding a pre-aimed angle, and they fire on that tick because they were already
aimed at it. Pros do this constantly.

Measured over the arrival ticks that are also the opening shot, CS2CD:

| | n | crosshair's own angular step at the arrival tick, deg/s (q25 / q50 / q75 / q90) | step / head radius, median |
|---|---|---|---|
| clean, fired on arrival | 1,768 | 14.9 / 35.6 / 78.8 / 150.3 | 0.672 |
| cheater, fired on arrival | 1,094 | 39.4 / 91.2 / 199.0 / 413.8 | 2.323 |
| clean, arrived but did not fire | 36,994 | 3.9 / 16.9 / 43.1 / 89.8 | 0.380 |
| cheater, arrived but did not fire | 11,347 | 2.4 / 13.7 / 38.4 / 82.7 | 0.329 |

A cheater's arrival shot is a sweep onto the head at 91 deg/s median. A clean
player's is a creep at 36, or the enemy arriving. The crosshair's step is
computed as `hypot(d_yaw * cos(pitch), d_pitch)` over the arrival tick, so it is
the crosshair's own motion and nothing else.

Requiring the crosshair to have been moving at the arrival tick, and re-deriving
the line from CS2CD clean players as the spec requires:

| arrival requires crosshair | `MIN` | line95 | shown (CS2CD clean) | clean past | cheater past | lift | pros past, of 341 | of shown |
|---|---|---|---|---|---|---|---|---|
| any motion (v5) | 5 | 0.2500 | 62.4% | 4.03% | 17.78% | 4.41x | 22 (6.4516%) | 22 of 278 (7.9137%) |
| >= 15 deg/s | 6 | 0.3333 | 25.4% | 2.40% | 29.35% | 12.21x | 3 (0.8798%) | 3 of 144 (2.0833%) |
| **>= 20 deg/s** | **6** | **0.3333** | **20.1%** | **3.16%** | **32.59%** | **10.32x** | **1 (0.2933%)** | **1 of 117 (0.8547%)** |
| >= 25 deg/s | 6 | 0.3333 | 15.2% | 4.18% | 35.44% | 8.48x | 2 (0.5865%) | 2 of 91 (2.1978%) |
| >= 30 deg/s | 5 | 0.4000 | 18.8% | 3.24% | 33.70% | 10.39x | 3 (0.8798%) | 3 of 113 (2.6549%) |

The full 44-candidate grid is in `training/arrival/gate3_arrival.py`'s output. The selection
rule is in its module docstring and is applied to CS2CD alone, before the pro
columns are read: **S1** crossing the line takes at least 3 arrival-tick kills at
every eligible count, **S2** cheaters past the line at or above v5's 17.78%,
**S3** lift at or above v5's 4.41x, **S4** shown to at least 15% of CS2CD clean
players. 8 of 44 candidates pass S1-S4; exactly one of the 8 also passes gate 3
on both denominators.

## 5. Why that one candidate is not yet a recommendation

**It passes S1 on a floating-point tie.** Its 95% line is
0.3333333333333333 and `2 / 6` is the same double, so two arrival-tick kills at
the minimum count do not cross it — by exact equality, not by margin. Bootstrap
over the 823 eligible clean players, 2,000 resamples: the line lands at or below
`2/6` in **9.1%** of them (usually at 0.2857 = 2/7). In roughly one clean
reference in eleven, two kills decides again. That is the defect the redesign
exists to remove, so a tie is not good enough.

Every neighbouring configuration that clears `2/MIN` with a real margin fails
gate 3 on the shown denominator: `>= 20 deg/s, MIN 7` (line 0.3083 against
0.2857) puts 3 of 82 shown pros past, 3.6585%.

**The sample cannot certify 1% anyway.** 1 of 117 shown pros is 0.8547% with a
95% confidence interval of **[0.0216%, 4.6701%]**. A run showing **zero** of 117
would still only bound the rate at 3.1037%. Demonstrating a rate at or below 1%
with 95% confidence takes **299 shown pros with no hits**; the 35-match sample
yields 117 shown under this definition, because a stricter arrival definition is
shown to fewer players. Reading a PASS off 1 of 117 after looking at 44
candidates would be reading noise.

**Pros stay elevated underneath.** Their pooled crosshair-driven arrival rate is
11.89% against CS2CD clean's 9.00% at `>= 20 deg/s`. The gate passes because the
line has moved up into a thin part of the distribution, not because pros have
become cleaner than clean players. That is a fragile place to draw a line.

**Therefore: enlarge the pro sample, then decide.** `pro_arrival_lab.py
--per-map 25 --seed 2` (175 matches, ~1,700 players, disjoint seed) is running;
`--per-map 5 --seed 1` is running as a quick independent read. Both resume — a
rerun skips demos already dumped. Selection stays on seed 0; the verdict will be
read off the fresh samples.

## 6. Gate 2 clause 1 cannot be met by redesigning `arrival_shot_share`

This is independent of everything above and it changes the release plan.

`targets.py:59-63`: `WEIGHTS = {"notable": 0.4, "strong": 1.0}` and
`DECISIVE = 1.0`. One measurement past a 99% line is a `cheating` target on its
own. A 99% line puts about 1% of clean players past it by construction, so on 341
pros about 3 players per run will carry one whatever the measurement is.

Counted on the v5 pro run's own evidence text, all measurements:

| pro | strong measurement | v5's verdict |
|---|---|---|
| pro A, a de_overpass match | degrees off target the moment the enemy appeared | clean, 10% |
| pro B, a de_anubis match | kills with a turn over 200 deg/s on the kill tick | unclear, 42% |
| pro C, a de_inferno match | kills with a turn over 200 deg/s on the kill tick | **cheating, 82%** |
| pro D, another de_overpass match | kills fired on the very tick the crosshair reached the head | unclear, 42% |

**4 of 341 (1.1730%) carry a strong measurement.** A fifth pro carries 3 notable
measurements, which also reaches `DECISIVE` at 3 x 0.4 = 1.2. So the targets
would accuse up to 5 of 341 (1.4663%); the fine-tune accused 1. v5 came in under
its own targets by luck, not by design, and v3's and v4's zero-accusation runs
were the same luck.

**Remove `arrival_shot_share` from the evidence entirely and 3 of the 4 remain.**
A gate written as "accuses none of 341 pros" cannot be met while one measurement
past a 99% line is decisive on its own. Either the gate or the rule has to
change, and per house convention the gate is the contract, so the rule should:
require corroboration for a `cheating` target — a second measurement past a line,
or a Layer 1 strong or impossible finding — rather than one line in isolation. I
have not implemented that; it changes every training target and needs its own
measurement of what it costs on CS2CD recall. **Flagged to CTO as a release-plan
change.**

## 7. `snap_kills` is sniper-confounded (ADR 0008)

Pros past its lines: **9 of 341 (2.6393%)** past 1.0, **2 of 341 (0.5865%)** past
2.0. Against CS2CD clean at 1.61% and 0.86%. Past the strong line pros are
*below* CS2CD clean players, so the accusation is not a rate problem — it is
section 6's rule.

The line itself is confounded, though. Splitting CS2CD by sniper share:

| subgroup | n clean / cheater | clean past 1.0 | cheater past 1.0 | lift | clean past 2.0 | cheater past 2.0 |
|---|---|---|---|---|---|---|
| sniper share < 20% | 2,969 / 207 | 0.71% | 3.38% | 4.76x | 0.27% | 1.93% |
| 20-50% | 763 / 166 | 1.44% | 12.05% | 8.37x | 0.26% | 7.83% |
| **>= 50%, AWP main** | **361 / 686** | **9.42%** | **16.33%** | **1.73x** | **6.93%** | **11.22%** |
| all | 4,093 / 1,059 | 1.61% | 13.13% | 8.14x | 0.86% | 8.88% |

For an AWP main, 3 kills over 200 deg/s is a 6.93%-of-clean event, not the
0.86% the aggregate 99% line claims — the line overstates it by 8x. Target rule 3
discounts sniping for the *visibility* measurements only; `snap_kills` goes
straight to `DECISIVE`. The accused pro (C) is not a sniper (no sniper kills; rifles, an SMG and a pistol), so this did not cause that
accusation — but it will cause the next one, and it belongs in the same redesign.

## Reproducibility

Everything above reruns from the three commands at the top plus:

```
# section 1, 2, 3 — statistics over (arrival shots, arrival kills)
.venv/bin/python training/arrival/gate3_arrival.py \
    --cs2cd data/processed/arrival_lab.parquet \
    --pro data/processed/pro_arrival_lab --json /tmp/gate3.json

# section 6 — strong and notable measurements in the v5 pro run's evidence text
uv run python training/check_pro_demos.py --summary --out data/processed/pro_check_v5
```

Seeds: pro sample `--seed 0` (selection), `--seed 1` and `--seed 2`
(confirmation, running). Bootstrap in section 5 is
`numpy.random.default_rng(0)`, 2,000 resamples.

## Eval-set burn

The seed-0 pro set had been used for the v3, v4 and v5 gate decisions. This run
makes four, and within this run **44 candidate arrival definitions were scored
against it**. It is a tuning set now and no verdict should rest on it again.
That is why the confirmation runs use fresh seeds, and why section 5 declines to
call the one passing candidate a pass.

## What was not tested

- **No judge, no fine-tune, no GPU.** Gate 1 (target match on 206 held-out cases)
  and gate 2 (pro verdicts) need a v6 trained on the redesigned evidence. Nothing
  here speaks to either. Gate 1's v5 numbers do not carry across a feature change.
- **No ablation.** The attribution in section 6 is read from the verdicts' own
  stated reasons, not from re-prompting with a measurement removed.
- **Sub-tick shots.** Unchanged from the v5 eval: CS2 records shots at sub-tick
  times and everything here runs at tick resolution.
- **Tickrate and patch.** Both CS2CD and the pro sample are one tickrate at one
  point in time. `cross_step` is a per-tick quantity, so a tickrate change moves
  the deg/s thresholds in section 4 directly.
- **Cross-match player identity.** Still impossible in CS2CD — 10 anonymous
  `player_id` values across the whole corpus. Held-out discipline is match-level
  only, as in the v5 eval.
- **`snap_kills` was not redesigned**, only diagnosed. Section 6 has to be settled
  first: a count whose 99% line is 2 is decisive on 3 kills whatever the count
  measures.
