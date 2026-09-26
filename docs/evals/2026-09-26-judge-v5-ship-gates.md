# Judge v5 against the three ship gates — 2026-09-26

**Recommendation: REDESIGN the measurement. Do not ship v5 in 0.4.0.**

Gate 1 passes. Gate 2 fails on both clauses. Gate 3 fails by 6.5x. Gates named
from `docs/specs/2026-09-26-triggerbot-and-snap-count.md` section 4.

| gate | measure | threshold | v5 | verdict |
|---|---|---|---|---|
| 1 | target match, 206 held-out | >= 92% | **94.17%** | **pass** |
| 1 | invented numbers | 0 | **0** | **pass** |
| 1 | accuses clean only where own target says cheating | 0 violations | **0 of 4** | **pass** |
| 2 | pros accused | 0 | **1 of 341 (0.2933%)** | **fail** |
| 2 | pros "unclear" | < 6.7% (v4) | **29 of 341 (8.5044%)** | **fail** |
| 3 | pros past `arrival_shot_share` 95% line | <= 1% | **22 of 341 (6.4516%)** | **fail** |

---

## What was compared, and on what

**Subject of the verdict: `models/llm/V5/Qwen3.5-4B.Q4_K_M.gguf`** — the
quantisation 0.4.0 would ship. Q8_0 and BF16-mmproj were not evaluated.

Runtime, identical on both sides: llama.cpp **b11177**, Vulkan, RTX 3060.
Reported by the harness as `GPU: NVIDIA GeForce RTX 3060 via vulkan (llama.cpp b11177)`.

### v4 identified, not assumed

`data/processed/judge_eval_v4_fastkills.jsonl`, 206 rows, is v4. Confirmed by
reproducing `training/llm/README.md`'s v4 column off it exactly: target match
0.922, accuses clean 0.028, convicts banned 0.286, invented 0, 2.219 s/case.

### Correction on the pro baseline

On-disk `data/processed/pro_check/` is **not** v4. It holds 35 reports, 341
players with enough kills, **5 unclear = 1.4663%**, and its evidence text still
reads "fastest turn" with zero occurrences of `arrival` or `snap_kills`. That is
README's **v3** run ("v3: 1.5%"). v4's pro reports are not on disk.

**Gate 2's 6.7% baseline is therefore carried forward from
`training/llm/README.md:120` (23 unclear of 341 = 6.745%), not recomputed here.**
It does not change the verdict: v5 is at 8.5044%, above the threshold, and the
accusation clause fails against a threshold of zero regardless of baseline.

The v3 directory was not overwritten; v5 wrote to `data/processed/pro_check_v5/`.

---

## Dataset and split

`data/processed/judge_training_v5/` — 972 train, 206 val, split by `match_id`.

- 496 train matches against 111 val matches
- **match overlap: 0**
- **(match, player) pair overlap: 0**
- val label mix: 108 clean, 98 cheater

**Held-out discipline, with a caveat that cannot be engineered away.** CS2CD
anonymises players per match as `Player_1`..`Player_10`, so the whole corpus
holds **10 distinct `player_id` values**. No stable cross-match player identity
exists in this dataset. "No player appears on both sides" is unverifiable by
construction and is guaranteed at **match level only**. v4's split has the same
property, so the comparison is not biased by it, but neither number is protected
against a player recurring across matches under a different anonymous slot.

Pro set: `training/check_pro_demos.py`, 35 HLTV matches, `--per-map 5 --seed 0`,
355 players, **341 with enough kills** — the same sample and denominator as the
v3 and v4 runs.

CS2CD reference: `data/processed/player_features.parquet`, 5,152 players
(4,093 clean, 1,059 cheater).

---

## Commands

```
# Gate 1
uv run python training/llm/compare_judges.py \
  --val data/processed/judge_training_v5/val.jsonl \
  v5=data/processed/judge_eval_v5.jsonl

# v4 baseline reproduction
uv run python training/llm/compare_judges.py \
  --val data/processed/judge_training_fastkills/val.jsonl \
  v3_fastkills=data/processed/judge_eval_v3_fastkills.jsonl \
  v4_fastkills=data/processed/judge_eval_v4_fastkills.jsonl

# Gates 2 and 3 — the pro run
uv run python training/check_pro_demos.py --per-map 5 --seed 0 \
  --model models/llm/V5/Qwen3.5-4B.Q4_K_M.gguf \
  --scorer models/scorer/v5/scorer.onnx \
  --out data/processed/pro_check_v5

# Gate readout (re-runs without the GPU)
uv run python tools/gate23_pro.py data/processed/pro_check_v5
uv run python training/check_pro_demos.py --summary --out data/processed/pro_check_v5
```

### Blocker cleared before it corrupted gates 2 and 3

Installed `models/scorer/scorer.json` still carries v4-era judge lines:
`snap_max [283.5126953125, 719.669921875]`, no `snap_kills`, no
`arrival_shot_share`. `check_pro_demos.py` built `Scorer()` with no argument, so
a pro run would have fed v5 the evidence v4 was trained on and gate 3 would have
had nothing to measure.

The installed detector was **not** overwritten — release artefact, Nora's pin.
Instead `models/scorer/v5/scorer.{onnx,json}` was exported and `--scorer` added
to `check_pro_demos.py`. Both bundles report `trained_at
2026-09-24T21:53:07+00:00` and an equal score reference, so behaviour scores stay
comparable to the v3 pro run; only the judge's evidence lines differ.

---

## Gate 1 — PASS

206 held-out cases, 1.869 s/case.

| measure | v4 | v5 |
|---|---|---|
| valid JSON | 1.000 | 1.000 |
| **matches target** | 0.922 | **0.942** |
| accuses clean | 0.028 | 0.037 |
| convicts banned | 0.286 | 0.378 |
| unclear | 0.262 | 0.223 |
| right when committing | 0.776 | 0.800 |
| AUC | 0.779 | 0.795 |
| follows evidence | 0.948 | 0.966 |
| leans on score | 0.125 | **0.314** |
| **invented** | 0 | **0** |
| s/case | 2.219 | 1.869 |

v4 is scored against `judge_training_fastkills/val.jsonl`, v5 against
`judge_training_v5/val.jsonl`. The target files differ because the evidence text
changed; each side is scored against its own targets. The columns are not two
runs on one fixed set.

**Clause 3 in full.** v5 accuses 4 of 108 clean-labelled players (3.7037%). All 4
carry `target_verdict == cheating`:

| case | v5 | target |
|---|---|---|
| `no_cheater_present/67` Player_7 | cheating 82% | cheating 80% |
| `no_cheater_present/78` Player_7 | cheating 82% | cheating 82% |
| `no_cheater_present/78` Player_8 | cheating 76% | cheating 74% |
| `no_cheater_present/78` Player_9 | cheating 76% | cheating 80% |

Zero violations. Note 3 of the 4 sit in one match, `no_cheater_present/78`; CS2CD
clean labels are unverified and that match's own evidence targets call all three
cheating.

### Gate 1 subgroups

| subgroup | n (v5 / v4) | v5 match | v4 match |
|---|---|---|---|
| label=clean | 108 | 0.9722 | 0.9537 |
| label=cheater | 98 | 0.9082 | 0.8878 |
| target=cheating | 45 / 42 | 0.8889 | 0.7381 |
| target=clean | 123 / 122 | **0.9593** | 0.9754 |
| target=unclear | 38 / 42 | 0.9474 | 0.9524 |

One subgroup regresses: `target=clean` falls 1.61 points, 0.9754 to 0.9593. It is
the false-positive-cost side, so it is named rather than waved through, but it is
not a gate and both label groups improve.

**Non-gate regression.** "Leans on score" rises 0.125 to **0.314**, 2.5x. Spec
section 1 says the behaviour score is never evidence. v5 correlates its stated
probability with the behaviour score far more than v4 did. Does not block
anything by itself; it is the weakest number in an otherwise passing gate 1.

---

## Gate 2 — FAIL, both clauses

341 pros.

| verdict | count | share |
|---|---|---|
| clean | 311 | 91.2023% |
| unclear | 29 | **8.5044%** |
| cheating | 1 | **0.2933%** |
| no answer | 0 | 0% |

- **accused: 1, threshold 0 — FAIL.**
- **unclear: 8.5044%, threshold < 6.7% — FAIL.** Up from v4's carried-forward
  6.7% and from v3's measured 1.4663%.

### The accused pro

A pro on de_inferno (pro C in docs/evals/2026-09-26-arrival-redesign.md).
The match and the player are not named here: this is a model's accusation
of a real person.

- behaviour score **0.168**, `clean_percentile` **0.16** — below the pro median
- `flagged: false`, `rule_findings: []` — layer 1 found nothing
- judge verdict **cheating, 82%, cheat_type aimbot**
- sole reason: *"3.0 of their kills came with a turn over 200 deg/s on the kill
  tick, against 0 for clean players"*

Every other measurement in that player's evidence is ordinary. `snap_kills` lines
are `[1.0, 2.0]`, so 3 qualifying kills is past the 99% line. The judge overrode
an unremarkable behaviour score and an empty layer 1 to accuse an elite
legitimate player at 82% on **one** new evidence line.

This is a separate player from the single score-flagged pro
(another de_inferno match, score 0.483, p90, judged unclear).

### Gate 2 unclear by map

| map | players | unclear |
|---|---|---|
| de_ancient | 46 | 4 (8.70%) |
| de_anubis | 50 | 4 (8.00%) |
| de_dust2 | 50 | 5 (10.00%) |
| de_inferno | 47 | 4 (8.51%) |
| de_mirage | 48 | 4 (8.33%) |
| de_nuke | 50 | 3 (6.00%) |
| de_overpass | 50 | 5 (10.00%) |

Uniform. No single map carries the failure; de_nuke alone would pass, every
other map fails.

### What drives the 30 non-clean verdicts

Evidence named in the verdict's own reasons (a verdict may name more than one):

| evidence line | verdicts naming it |
|---|---|
| **`arrival_shot_share`** | **19** |
| `snap_kills` | 7 |
| `wall_aim_share` | 1 |
| `fast_kills` | 1 |
| `straight_share` | 1 |
| none named | 1 |

The two measurements this spec introduced account for 26 of 30. **19 of 30 rest
on `arrival_shot_share` alone** — the same shape, and the same count, as the
failure the spec was written to fix: "19 of judge v4's 23 unclear verdicts rest
on fastest turn on a kill tick."

---

## Gate 3 — FAIL

`arrival_shot_share`, judge text "kills fired on the very tick the crosshair
reached the head". Lines from `models/scorer/v5/scorer.json`: 95% = **0.25**,
99% = **0.40**. Withheld below `MIN_ARRIVAL_KILLS = 5`.

Shown to **278 of 341** pros (81.5249%).

| denominator | past 95% line | rate | threshold | verdict |
|---|---|---|---|---|
| all pros (341) | 22 | **6.4516%** | <= 1% | **FAIL** |
| shown (278) | 22 | **7.9137%** | <= 1% | **FAIL** |

Both readings fail, so the denominator question does not need resolving. 6.5x
over on the stricter-for-v5 reading. Past the 99% line: 1 pro (0.2933% of all).

### Gate 3 by map

| map | shown | past 95% | past 99% |
|---|---|---|---|
| de_ancient | 42 | 3 (7.14%) | 0 |
| de_anubis | 47 | 3 (6.38%) | 0 |
| de_dust2 | 43 | **6 (13.95%)** | 0 |
| de_inferno | 34 | 3 (8.82%) | 0 |
| de_mirage | 36 | 3 (8.33%) | 0 |
| de_nuke | 38 | **0 (0.00%)** | 0 |
| de_overpass | 38 | 4 (10.53%) | 1 |

de_nuke passes at 0%; de_dust2 is worst at 13.95%. Six of seven maps fail
individually. The aggregate is not hiding a single bad map — it is hiding that
de_nuke is unlike the rest, which is itself worth a look during redesign.

### The measurement is not measuring what it claims

Every other evidence line puts at most 3.52% of pros past its 95% line, and most
put 0.29% — roughly the 5% a clean line should produce, or less, since pros are
cleaner than CS2CD clean players. The two new lines are the outliers:

| measurement | pros past 95% line (of 341) |
|---|---|
| **kills fired on the very tick the crosshair reached the head** | **22 (6.45%)** |
| kills with a turn over 200 deg/s on the kill tick | 9 (2.64%) |
| degrees off target the moment the enemy appeared | 12 (3.52%) |
| time aimed at an enemy they could not see | 2 (0.59%) |
| every remaining measurement | 1 each (0.29%) |

**Cause: the share is computed on a denominator too small to carry it.** Among
the 3,229 eligible CS2CD players, `arrival_kills` has median 8 and 25th
percentile 6; 56.43% have 8 or fewer. The 0.25 line falls inside the gap between
one and two arrival shots for any player with 5, 6 or 7 eligible kills:

| arrival_kills | 2 shots gives | past 95% (>0.25)? |
|---|---|---|
| 5 | 0.4000 | yes |
| 6 | 0.3333 | yes |
| 7 | 0.2857 | yes |
| 8 | 0.2500 | no |
| 10 | 0.2000 | no |

So two arrival-tick kills makes a player "notable" or not depending only on how
many eligible kills they happened to have. The data confirms the artefact
dominates:

- **16 of the 22 pros past the line are there on exactly 2 arrival shots**
  (5 on 3 shots, 1 on 5).
- In CS2CD, **55.34% of the 103 clean players past the line sit on exactly 2
  arrival shots**, against **11.67% of the 120 cheaters**. Cheaters spread from 2
  to 19; clean players pile up at the granularity floor.

The underlying signal is real — 17.7778% of eligible cheaters past the line
against 4.0329% of eligible clean players, 4.4x — but at the shipped threshold
the false-positive side is mostly quantisation noise in the denominator.

This is the same class of flaw as `snap_max`, which this spec set out to remove
because "one kill decides a player's maximum". `arrival_shot_share` replaced
one-kill-decides with two-kills-decide. `snap_kills`, at lines `[1.0, 2.0]`, has
it too: 2 qualifying kills is past its 95% line, and it is the sole reason the
one pro was accused.

### CS2CD reference (recomputed this run)

| | eligible | past 0.25 | past 0.40 |
|---|---|---|---|
| clean | 2,554 of 4,093 | 103 (4.0329% of eligible, 2.5165% of all) | 23 (0.9005% of eligible) |
| cheater | 675 of 1,059 | 120 (17.7778% of eligible, 11.3314% of all) | 81 (12.0000% of eligible) |

---

## What was not tested

- **Q8_0 and BF16-mmproj.** The verdict is about `Q4_K_M` only. A gate failure on
  Q4_K_M says nothing about the other exports, and they were not run.
- **v4's pro check was not rerun.** The 6.7% is carried forward from
  `training/llm/README.md:120`. Both gate 2 clauses fail without needing it.
- **No cross-match player identity check** — impossible in CS2CD, see split above.
- **Patch, tickrate and map pool.** The pro set is 35 HLTV matches across 7 maps
  at one point in time. Nothing here speaks to other tickrates or a different map
  pool, and `arrival_shot_share` is tick-resolution, so a tickrate change would
  move it.
- **Eval-set burn.** The 206 v5 held-out cases were looked at once, in this run.
  The pro set has now been used for v3, v4 and v5 gate decisions — three looks.
  It is drifting toward a tuning set and should be resampled with a new seed
  before a v6 decision rests on it.
- **No ablation rerun with `arrival_shot_share` removed from the evidence.** The
  attribution above is read from the judge's own stated reasons, not from a
  retrained or re-prompted model. It shows what the judge *says* moved it, which
  is weaker than showing what *would* change if the line were dropped.

---

## Recommendation

**Redesign the measurement. Do not ship v5 in 0.4.0.**

Gate 3 is written to produce exactly this outcome, and it fired: 6.4516% of pros
past the 95% line against a 1% ceiling.

**Single strongest piece of evidence:** 16 of the 22 pros past the line are there
on exactly **2** arrival-tick kills, and in CS2CD **55.34%** of clean players past
the line sit on 2 while only **11.67%** of cheaters do. The 0.25 threshold sits
inside the gap between 1 and 2 kills for any player with 5-7 eligible kills, so
for the median player the measurement is a two-kill coin flip, not a rate. That
is the `snap_max` flaw this spec was written to remove, reintroduced in a new
measurement.

A retrain cannot fix this. The judge is reading the line correctly and following
its targets; the line itself is wrong at these denominators.

### Suggested direction for redesign — Mira's view, not a gate

Raise `MIN_ARRIVAL_KILLS` well above 5, or replace the share with a count-based
or interval-based statistic that does not divide by a single-digit denominator
(the lower bound of a binomial confidence interval on the share would put the
2-of-6 players back in the clean band while leaving 19-of-30 cheaters past it).
Re-derive both `arrival_shot_share` and `snap_kills` lines afterwards; `snap_kills`
has the same defect and caused the pro accusation.

**Gate 1 is unaffected by the redesign** and its numbers stand — 94.17% target
match, 0 invented, 0 clause-3 violations. The training pipeline is sound. The
evidence definition is not.
