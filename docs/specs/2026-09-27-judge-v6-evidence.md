# Judge v6: evidence that pros do not trip, and no accusation on one line

Status: approved 2026-09-27; partly done, the retrain parked. Section 2 ran
(`docs/evals/2026-09-27-v6-measurements-pro-check.md`): `arrival_shot_share`
failed its pro check and stays out; `snap_kills` (non-sniper) passed but rarely
fires. With little new evidence left, the user chose to ship section 3 now as a
guard on the pinned judge's verdicts (`l4_judge/guard.py`, 0.5.1) and to park
the retrain and the `preaim_count` pro run until a stronger measurement joins
them. The training targets already follow section 3 (`targets.corroborated`). Brainstormed with
the superpowers brainstorming process, architectural path. Replaces the v5 plan
(`docs/specs/2026-09-26-triggerbot-and-snap-count.md`), whose judge failed its pro
gates (`docs/evals/2026-09-26-judge-v5-ship-gates.md`).

## Words used here

- **Clean line.** For each measurement, all clean CS2CD players (not banned) are
  sorted by it. The **95% line** is the value 95 of 100 of them stay below; the
  **99% line** the value 99 of 100 stay below. Past the 95% line is *notable*,
  past the 99% line *strong*. By construction about 5% of clean players are past
  a 95% line and 1% past a 99% line.
- **Target.** The answer the judge is trained to give for one player: a verdict,
  a probability and reasons. Built from the player's own measurements and their
  clean lines, never from the ban label (`l4_judge/targets.py`).
- **Pro check.** A measurement, or a judge, run on professional matches from
  HLTV. Pros are the most skilled honest players available; a measurement they
  trip is measuring skill.

## Why

- **v5 accused a pro, and any judge trained on today's targets eventually
  will.** One measurement past its 99% line makes a `cheating` target on its own
  (`DECISIVE = 1.0`); 4 of 341 pros carry one, and a fifth carries three notable
  lines. Changing measurements cannot fix that; the rule has to change.
- **The triggerbot measure counted held angles.** "Fired on the tick the
  crosshair reached the head" also closes when the enemy walks into a crosshair
  held still, which is a pro's pre-aimed kill. Requiring the crosshair to sweep
  onto the head lifts CS2CD separation from 4.41x to 9.13x
  (`docs/evals/2026-09-26-arrival-redesign.md`).
- **Fast turns and pre-aims are sniper-confounded.** AWP flicks count as fast
  turns (lift among AWP mains 1.77x), and holding an angle scoped counts as
  keeping the crosshair on an unseen enemy (clean AWP mains past the 95% line
  16.8% of the time, `docs/evals/2026-09-27-preaim-spike.md`).

## Decided in conversation

| question | answer |
|---|---|
| when a target says `cheating` | only with corroboration |
| snipers | fast turns count non-sniper kills only; pre-aims are wallhack evidence, set aside for heavy snipers by rule 3 |
| the pre-aim count | in v6 only if pros pass its check |

## Design

### 1. What judge v6 reads

The pinned judge (v4) is shown exactly what it reads today
(`rendering.FEATURE_SINCE_JUDGE`, `FEATURE_UNTIL_JUDGE`); everything below is
for judge 6 (`LATEST_JUDGE = 6`).

- **`arrival_shot_share`, redesigned.** An arrival counts only when the
  crosshair itself moved onto the head, at 50 deg/s or more on that tick
  (`shots.ARRIVAL_SWEEP_DPS`); at least 5 eligible kills (`MIN_ARRIVAL_KILLS`).
  Judge text: "kills fired on the very tick the crosshair swept onto the head".
- **`snap_kills`, non-sniper kills only.** Kills made with any weapon but a
  sniper rifle (pistols and SMGs count) with a turn over 175 deg/s on the kill
  tick (`SNAP_THRESHOLD_DPS`). Its lines are floored (`targets.LINE_FLOORS`): at
  least 3 kills to be notable, 4 to be strong, so two fast kills never decide
  anything. Judge text: "non-sniper kills with a turn over 175 deg/s on the kill
  tick".
- **`preaim_count`, new, if it passes section 2.** Per match, the stretches where
  the crosshair stays within 5 deg of the head of an enemy nobody on the
  player's team can see (mesh line of sight and field of view), for 0.5 s or
  more, while that enemy's bearing turns 10 deg or more: the rule measured in
  `docs/evals/2026-09-27-replay-marks-gate.md`. Suggests a wallhack. Judge text:
  "times the crosshair followed an enemy nobody on the team could see".
- `snap_max` stays shown to v4 only. Everything else is unchanged.

### 2. Pro check of the measurements, before anything is built

On the 175 HLTV matches of `data/processed/pro_arrival_lab_big` (`--per-map 25
--seed 2`), which no definition was chosen on. No judge, no GPU. Lines come from
clean CS2CD players, as the judge's do.

| measurement | passes if | data |
|---|---|---|
| `arrival_shot_share` | at most 1% of pros past its 95% line, of those it is shown for and of all | the dumps as they are |
| `snap_kills` (non-sniper) | at most 1% of pros past its notable line | the dumps as they are (weapon and turn per kill) |
| `preaim_count` | at most 5% of pros past its 95% line and 1% past its 99% line | the 175 demos again, parsed and ray cast (a few hours of CPU) |

A measurement that fails stays out of v6; the others go ahead. The thresholds
above are fixed here and not tuned after the numbers are read.

#### 2a. Triggerbot timing, second check (added 2026-09-27, after the first read)

`arrival_shot_share` failed the table above: 4 of 253 shown pros past its 95% line
(1.58%). A spike (`docs/evals/2026-09-27-triggerbot-pros-spike.md`) found them to
be riflers on 3 of 5 or 6 kills, and pros as a group past the line less often than
clean CS2CD players (1.58% against 5.25%). The 1% ceiling was written when one
line could accuse on its own; since 0.5.1 none can. The user chose to change the
criterion, **after seeing those numbers**, to:

- at most as many pros past the 95% line as clean CS2CD players (5%), of those
  shown and of all; and
- at most 1% of shown pros past the 99% line.

Because it was changed after a read, it is judged only on a fresh sample:
`pro_arrival_lab.py --per-map 25 --seed 3`, minus every demo of the three samples
read before (seed 0, seed 1, the seed-2 175). Nothing about the measurement
changes between the two reads.

### 3. When a target says `cheating`: corroboration

A `cheating` target needs one of:

1. two measurements past their lines, at least one of them past its 99% line;
2. one measurement past its 99% line and a Layer 1 strong finding;
3. a Layer 1 `impossible` finding, which is not a statistical line and stands
   alone as today.

Otherwise the target is `unclear`, and its reasons cite the strongest line. Rule
3 (heavy snipers' wallhack evidence does not count) applies before this.

Estimated on today's 1,178 cases (built with the old triggerbot and fast-turn
definitions, so indicative only):

| | cheaters with a `cheating` target | clean-labelled with a `cheating` target |
|---|---|---|
| today | 42.8% (214 of 500) | 14.2% (96 of 678) |
| with corroboration | 34.2% (171 of 500) | 10.8% (73 of 678) |

66 targets move from `cheating` to `unclear`: 26 rested on one strong line
alone, 40 on notable lines only.

### 4. Building v6

- Features: `player_features` gains `preaim_count` (CS2CD matches with a map
  mesh; null otherwise) if section 2 passes it; `snap_kills` counts non-sniper kills.
  The analysis pipeline computes the same from a demo (the replay already ray
  casts who sees whom, so the cost is small).
- Rebuild the dataset, the judge cases and the training set
  (`judge_training_v6`); refresh the clean reference (`scorer.json` lines for the
  new measurements); copy `unsloth/{train,val}.jsonl` to
  `C:\cs2-overwatch-v6-data`. You train v6 with v4's settings (QLoRA, context
  2048, train on completions, gradient checkpointing, export Q4_K_M).

### 5. v6 ships only if

1. On the 206 held-out cases it matches its targets at least as often as v4
   (92.2%), invents no numbers, and gives `cheating` to a clean-labelled player
   only where that player's target says so.
2. On the 175-match pro sample it accuses no one, and says `unclear` for at most
   6.7% of pros (v4's rate).

Then: publish v6 and its reference to Hugging Face, pin them
(`models.JUDGE`, `JUDGE_GENERATION = 6`), release 0.6.0.

## Risks

- **Fewer cheaters get a `cheating` target** (section 3). Deliberate: the output
  is an accusation of a named person. Gate 1 is measured against the new
  targets, so it does not hide this; the table above says what it costs.
- **The pro sample is used twice**: once to accept or reject the three
  measurements (section 2, pass or fail, no choosing between definitions), once
  for the judge's gate. Nothing is tuned on it in between.
- **CS2CD has no held weapon per tick**, so the pre-aim count cannot leave out
  sniper rifles; rule 3 is what keeps it from convicting AWP players.
- **The `preaim_count` pro run needs the demos again** (they were streamed and
  deleted); a download that fails is retried, and a demo that cannot be fetched
  is left out and counted.

## Plan

0. Land the measurement changes already written (arrival sweep, 175 deg/s, line
   floors, judge 6 labels), with their tests.
1. `snap_kills` counts non-sniper kills; tests.
2. Section 2 for `arrival_shot_share` and `snap_kills` from the dumps; results to
   `docs/evals/`.
3. `preaim_count`: the rule into product code (`l2_perception`), the pro run on
   the 175 demos, section 2's check; results to `docs/evals/`. If it fails,
   stop here for this measurement.
4. The corroboration rule in `targets.py`; tests.
5. Features, dataset, cases, training set, reference; copy to C:.
6. You train v6.
7. Section 5's gates; then publish, pin, release 0.6.0.
