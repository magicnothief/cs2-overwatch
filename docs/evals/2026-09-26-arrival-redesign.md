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

### 5a. The fresh sample killed it, as expected

`pro_arrival_lab.py --per-map 5 --seed 1`, 35 matches, **346 players, zero demo
overlap with seed 0** (checked against both `listing.json` files).

| candidate | seed 0 (341 pros) | seed 1 (346 pros) | pooled, 70 matches (687 pros) |
|---|---|---|---|
| v5, any arrival, MIN 5 | 22 of 278 (7.9137%) | 20 of 277 (7.2202%) | 42 of 555 (7.5676%) |
| >= 15 deg/s, MIN 6 | 3 of 144 (2.0833%) | 3 of 158 (1.8987%) | 6 of 302 (1.9868%) |
| **>= 20 deg/s, MIN 6** | **1 of 117 (0.8547%)** | **3 of 129 (2.3256%)** | **4 of 246 (1.6260%)** |
| >= 25 deg/s, MIN 6 | 2 of 91 (2.1978%) | 4 of 100 (4.0000%) | 6 of 191 (3.1414%) |
| >= 30 deg/s, MIN 5 | 3 of 113 (2.6549%) | 3 of 123 (2.4390%) | 6 of 236 (2.5424%) |

The v5 baseline reproduces on the fresh sample to within 0.7 points, so the pro
population is stable and the seed-0 reading was not a strange sample. The
**seed-0 winner was selection noise**: 0.8547% became 2.3256%, and **all 8
S1-S4 candidates fail gate 3 on seed 1**. Picking the best of 44 against one
35-match sample bought a number that did not survive one disjoint sample. That is
the whole reason the confirmation run exists.

Pooled over 70 matches, the redesign still cuts the pro rate 4.65x — 7.5676% of
shown to 1.6260% — and still misses gate 3's 1% ceiling on the shown denominator.

### 5b. What does pass, and why S4 was the wrong criterion

On the pooled 687 pros, 15 candidates put **zero** pros past their 95% line on
both denominators. Every one of them fails **S4** — my own coverage floor of 15%
of CS2CD clean players — and nothing else. S4 was pre-registered and it was
wrong: it assumed low coverage means hiding, when for this measurement it means
the behaviour is genuinely rare. Gate 3's two denominators already catch hiding,
because withholding raises the of-shown rate.

Stating the change plainly: **S4 was relaxed after the pro numbers were seen.**
That makes what follows a hypothesis, not a verdict.

| candidate | line95 | line99 | shown (CS2CD clean) | clean past | cheater past | lift | pros past, pooled |
|---|---|---|---|---|---|---|---|
| >= 20 deg/s, MIN 8 | 0.3500 | 0.6127 | 8.55% | 5.14% | 41.67% | 8.10x | 1 of 118 (0.8475%) |
| **>= 50 deg/s, MIN 5** | **0.4857** | **0.8327** | **7.45%** | **5.25%** | **47.90%** | **9.13x** | **0 of 102 (0%)** |
| >= 50 deg/s, MIN 6 | 0.5000 | 0.8429 | 3.93% | 4.35% | 52.14% | 11.99x | 0 of 55 (0%) |
| >= 40 deg/s, MIN 7 | 0.4286 | 0.7705 | 3.91% | 4.37% | 53.10% | 12.14x | 0 of 49 (0%) |

`>= 50 deg/s, MIN 5` is the one to take forward: it keeps the most coverage of
the zero-hit set, and its line clears `2/5 = 0.4` by 0.0857 — no floating-point
tie, three arrival-tick kills needed at every eligible count.

**What it costs, over the whole population rather than the eligible subset:**

| | flags this share of all 4,093 clean CS2CD players | flags this share of all 1,059 cheaters | ratio |
|---|---|---|---|
| v5, any arrival, MIN 5 | 2.5165% | 11.3314% | 4.50x |
| >= 20 deg/s, MIN 6 | 0.6352% | 8.3097% | 13.08x |
| **>= 50 deg/s, MIN 5** | **0.3909%** | **7.5543%** | **19.32x** |

It reaches a third fewer cheaters than v5 (7.5543% against 11.3314%) and flags
6.4x fewer clean players (0.3909% against 2.5165%). Given the product's output is
an accusation, that is the trade to take, and it is the trade being made
deliberately rather than by accident.

**Still not certifiable.** 0 of 102 shown pros bounds the rate at **3.5519%**
with 95% confidence, not 1%. `--per-map 25 --seed 2` (175 matches, disjoint
again) is running and will bring the shown count to roughly 500; 0 of 500 bounds
it at **0.7351%** and settles gate 3. That run resumes — a rerun skips demos already
dumped.

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

## 7. `snap_kills`: the line, not the count, is what two kills decide

Pros past its shipped lines, on the pooled 687: **13 of 687 (1.8923%)** past 1.0,
**3 of 687 (0.4367%)** past 2.0. Against CS2CD clean at 1.61% and 0.86%. Gate 3
asks for 1% past the 95% line, so `snap_kills` fails it too, 1.9x over.

The count itself is not the problem. **A 95%/99% quantile over a rare count lands
on a tiny integer by construction**, and that is what makes two kills decide:
clean CS2CD players' `snap_kills` is 0 for 96% of them, so the 95th percentile is
1 and the 99th is 2. Three kills are then "strong", and one strong measurement is
`DECISIVE` (section 6).

Raising the speed threshold makes this worse, not better, because the count gets
rarer and the line gets smaller:

| threshold | clean q95 / q99 | notable at | strong at | CS2CD clean past 95 | cheater past 95 | pros past 95, of 687 |
|---|---|---|---|---|---|---|
| 100 deg/s | 2 / 4 | 3 kills | 5 kills | 2.20% | 15.68% | 19 (2.7657%) |
| 150 deg/s | 1 / 3 | 2 kills | 4 kills | 2.76% | 16.90% | 24 (3.4934%) |
| 175 deg/s | 1 / 2 | 2 kills | 3 kills | 2.08% | 15.11% | 18 (2.6201%) |
| 200 deg/s (shipped) | 1 / 2 | 2 kills | 3 kills | 1.61% | 13.13% | 13 (1.8923%) |
| 300 deg/s | 0 / 1 | 1 kill | 2 kills | 4.35% | 18.32% | 37 (5.3857%) |
| 500 deg/s | 0 / 1 | 1 kill | 2 kills | 1.61% | 10.86% | 14 (2.0378%) |

**144 candidates were swept** — 9 speeds x (count, share over kills with a 10- or
14-kill floor) x (all weapons, non-sniper kills only) x (one pooled line, a line
per sniper band) — and **none puts the quantile line where three kills cannot
decide a verdict**. Rewriting the count as a share does not help either: it turns
1 kill of 14 into a value past the line.

### 7a. The fix: the quantile is a ceiling on false positives, not a target

`targets.LINE_FLOORS` raises a measurement's lines when the clean quantile falls
below a floor, and never lowers them. For `snap_kills` the floor is `(2.0, 3.0)`:
**3 kills to be notable, 4 to be strong**. This is strictly more conservative than
the spec's quantile rule — it can only ever cost recall, never add a false
positive — and it is the one deviation from
`docs/specs/2026-09-26-triggerbot-and-snap-count.md` section 1 ("95% line
notable, 99% strong") that this redesign needs. **It needs CTO's approval as a
spec change.**

With the floor in place, 175 deg/s is the best of the 9 speeds:

| | notable / strong | CS2CD clean past notable | cheater past notable | lift | cheater past strong | pros past notable, of 687 | past strong |
|---|---|---|---|---|---|---|---|
| shipped, 200 deg/s, quantile | 2 / 3 kills | 1.61% | 13.13% | 8.14x | 8.88% | 13 (1.8923%) | 3 (0.4367%) |
| 150 deg/s + floor | 3 / 4 kills | 1.08% | 11.43% | 10.63x | 8.97% | 3 (0.4367%) | 1 (0.1456%) |
| **175 deg/s + floor** | **3 / 4 kills** | **0.90%** | **10.39%** | **11.49x** | **7.93%** | **3 (0.4367%)** | **1 (0.1456%)** |
| 300 deg/s + floor | 3 / 4 kills | 0.51% | 5.76% | 11.23x | 3.97% | 2 (0.2911%) | 1 (0.1456%) |

175 over 150 on the false-positive asymmetry: same pro rate, same crossing
counts, 0.90% of clean players against 1.08%, and a higher lift. 175 over 300
because 300 drops cheater recall by 45% (5.76% against 10.39%) to buy one pro.

Against the pre-registered criteria in `training/arrival/gate3_snap.py`, applied
to CS2CD before the pro columns were read:

| criterion | threshold | 175 deg/s + floor | verdict |
|---|---|---|---|
| N1 no 2-kill verdicts | notable needs >= 3 kills, strong >= 4 | 3 and 4 | **pass** |
| N2 signal kept | cheaters past notable >= 10.00% | 10.39% | **pass** |
| N3 separation improved | lift >= the shipped 8.14x | 11.49x | **pass** |
| N4 sniper robustness | AWP-main lift >= 3.00x | 1.77x | **fail** |
| gate 3 | <= 1% of pros past the 95% line | 0.4367% (3 of 687) | **pass** |

### 7b. N4 fails, and no definition of the measurement fixes it

The sniper confound stands. Splitting CS2CD by sniper share, at 175 deg/s with the
floor:

| subgroup | n clean / cheater | line | clean past notable | cheater past notable | lift |
|---|---|---|---|---|---|
| sniper share < 20% | 2,969 / 207 | 3 kills | 0.27% | 2.42% | 8.96x |
| 20-50% | 763 / 166 | 3 kills | 0.26% | 8.43% | 32.17x |
| **>= 50%, AWP main** | **361 / 686** | **3 kills** | **7.48%** | **13.27%** | **1.77x** |
| all | 4,093 / 1,059 | 3 kills | 0.90% | 10.39% | 11.49x |

A clean AWP main is **28x** more likely to be flagged by this measurement than a
clean rifler (7.48% against 0.27%). The aggregate lift of 11.49x is carried
entirely by riflers, exactly as it was at 200 deg/s. **Max AWP-main lift over all
144 candidates is 1.91x**, so N4 is not reachable by redefining the measurement.

Two ways out were measured; neither is landed:

- **A line per sniper band** (an AWP main compared against clean AWP mains, line
  4 kills): clean AWP mains past notable falls 7.48% to 4.99% — the 5% a 95% line
  means by construction — but cheaters past notable falls 10.39% to 6.42%, a 38%
  recall loss to remove 2.5 points of AWP false positives. Rejected on that trade.
- **Extend target rule 3 to `snap_kills`** (ADR 0008): for a heavy sniper the
  visibility measurements already do not count towards a verdict; `snap_kills`
  should join them, since for that subgroup it measures the weapon. This is the
  recommendation, it is **not implemented**, and it belongs with section 6's
  corroboration change because both rewrite `targets.evidence_points` and every
  training target. **Both are one decision for CTO.**

The accused pro (C) is not a sniper (no sniper kills; rifles, an SMG and a pistol), so the confound did not cause that accusation. The
floor does remove it: pro C has 3 kills over 200 deg/s, which was strong under
the shipped lines and is notable under the redesigned ones — 0.4 weight, not 1.0,
so no longer decisive on its own.

## 8. What landed in the code

The measurement changes are landed; the rule changes (section 6, section 7b) are
not, and are CTO's decision.

| file | change |
|---|---|
| `src/overwatch/layers/l3_behavior/shots.py` | `ARRIVAL_SWEEP_DPS = 50.0`; an arrival needs the crosshair's own step over that tick to be 50 deg/s or more |
| `src/overwatch/layers/l3_behavior/player_features.py` | `SNAP_THRESHOLD_DPS` 200.0 to 175.0 |
| `src/overwatch/layers/l4_judge/targets.py` | `LINE_FLOORS = {"snap_kills": (2.0, 3.0)}`, applied in `clean_lines`; the two judge-text sentences follow the new labels |
| `src/overwatch/layers/l4_judge/rendering.py` | both labels reworded ("swept onto the head", "175 deg/s"); `FEATURE_SINCE_JUDGE` 5 to 6 for both; `LATEST_JUDGE` 5 to 6 |
| `tests/test_shots.py` | the enemy walking into a held crosshair is not an arrival; a crept arrival below the threshold is not one either |
| `tests/test_judge_targets.py` | a floored line is the floor when the quantile is below it, the quantile when above |

`MIN_ARRIVAL_KILLS` stays 5 — the chosen arrival definition needs no change there.

**Which judge sees what.** `FEATURE_SINCE_JUDGE` now says 6 for both, not 5,
because the redesign changed what they measure: v5's training data holds the old
definitions, so the same number means a different thing to it. The pinned judge is
v4 (`models.JUDGE_GENERATION = 4`) and is shown neither. Training, annotation and
the baseline render at `LATEST_JUDGE = 6`.

**The shipped code reproduces the lab exactly.** Rebuilding
`shots.arrival_features` over all 14,671,608 CS2CD tick rows and aggregating per
player through the feature path gives the same numbers as
`training/arrival/gate3_arrival.py`'s `>= 50 deg/s, MIN 5` row: line95 0.4857,
line99 0.8327, shown to 305 of 4,093 clean players (7.4517%), clean past 5.2459%,
cheater past 47.9042%, lift 9.13x, and over the whole population 0.3909% of clean
players flagged against 7.5543% of cheaters. `snap_kills` through
`targets.clean_lines`: clean q95 1.0 and q99 2.0, floored to notable 2.0 and
strong 3.0, so 2 kills are nothing, 3 notable, 4 strong; clean past notable
0.9040%, cheaters 10.3872%.

230 tests pass (`.venv/bin/python -m pytest tests/ -q`).

## Reproducibility

Everything above reruns from the three commands at the top plus:

```
# sections 1-5 — statistics over (arrival shots, arrival kills), pooled samples
PYTHONPATH=src .venv/bin/python training/arrival/gate3_arrival.py \
    --cs2cd data/processed/arrival_lab.parquet \
    --pro data/processed/pro_arrival_lab data/processed/pro_arrival_lab_seed1 \
    --json /tmp/gate3.json

# section 7 — 144 snap_kills candidates, the same two denominators
PYTHONPATH=src .venv/bin/python training/arrival/gate3_snap.py \
    --pro data/processed/pro_arrival_lab data/processed/pro_arrival_lab_seed1 \
    --json /tmp/snap.json

# section 9 — the confirmation read, strictly disjoint from the selection samples
PYTHONPATH=src .venv/bin/python training/arrival/pro_arrival_lab.py \
    --per-map 25 --seed 2 --out data/processed/pro_arrival_lab_big
PYTHONPATH=src .venv/bin/python training/arrival/gate3_arrival.py \
    --cs2cd data/processed/arrival_lab.parquet \
    --pro data/processed/pro_arrival_lab_big \
    --pro-exclude data/processed/pro_arrival_lab data/processed/pro_arrival_lab_seed1

# section 6 — strong and notable measurements in the v5 pro run's evidence text
uv run python training/check_pro_demos.py --summary --out data/processed/pro_check_v5
```

Seeds: pro sample `--seed 0` and `--seed 1` (selection), `--seed 2` (confirmation).
Bootstrap in section 5 is `numpy.random.default_rng(0)`, 2,000 resamples. Both
gate tools take `--pro-exclude`, because `--per-map 25 --seed 2` is **not** disjoint
from the seed-0 and seed-1 samples — all three draw from the same 1,988-demo
listing, and seed 2 shares 2 demos with seed 0 and 3 with seed 1. Section 9
subtracts those 5 by name rather than assuming a fresh seed is enough.

## Eval-set burn

The seed-0 pro set had been used for the v3, v4 and v5 gate decisions. This run
makes four, and within this run **44 candidate arrival definitions and 144
`snap_kills` candidates were scored against it** — 188 looks, pooled with seed 1
for sections 5b and 7. Both are tuning sets now and no verdict rests on them: the
verdict is read off seed 2 in section 9, which was never used for selection.

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
