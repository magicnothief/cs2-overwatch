# Judge v6, section 2: pros against the redesigned measurements — 2026-09-27

**Verdict: `arrival_shot_share` fails its pre-set check and stays out of v6.
`snap_kills` (non-sniper) passes, but carries almost no signal left.**

Spec: `docs/specs/2026-09-27-judge-v6-evidence.md` section 2, criteria fixed there
before this ran. Script: `training/v6/pro_check_measurements.py`. No judge, no GPU.

## The sample

`data/processed/pro_arrival_lab_big` (`--per-map 25 --seed 2`, 175 HLTV matches)
minus every demo of the two samples the definitions were chosen on
(`pro_arrival_lab`, `pro_arrival_lab_seed1`): **5 overlapped, 170 remain**, 1,673
pro player-matches. The spec said 175; the overlap was found by name when the
read ran (the samples draw from one listing, `gate3_arrival.lab_files`).

Lines are CS2CD clean players' with 5 or more kills, computed the way the judge's
are.

## `arrival_shot_share`: fails

Crosshair sweep of 50 deg/s or more onto the head, at least 5 eligible kills.

| | value |
|---|---|
| clean lines | 95% at 0.4857, 99% at 0.8327 |
| pros shown the measurement | 253 of 1,673 |
| pros past the 95% line | **4: 1.58% of shown**, 0.24% of all |
| pros past the 99% line | 0 |
| criterion | at most 1% of shown and of all |

It fails on the shown denominator. Even with no pro past the line, 253 shown pros
could only bound the rate at 1.18% (one-sided 95%), so the sample could not
have certified 1% either way. The 70-match selection samples had it at 0 of 102;
the disjoint read puts it at four.

## `snap_kills`, non-sniper kills: passes, and says little

Kills with a non-sniper weapon and a turn over 175 deg/s on the kill tick. The
clean 95% quantile is below the floor, so the lines are the floor: past 2 kills
notable, past 3 strong.

| | value |
|---|---|
| CS2CD clean past notable | 0.44% |
| CS2CD cheaters past notable | **1.23%** |
| pros past notable | 1 of 1,673 (0.06%) |
| pros past strong | 0 |
| criterion | at most 1% of pros past notable: **pass** |

Leaving out sniper kills took the confound and most of the signal with it: at 200
deg/s over all kills, 13.13% of cheaters were past the notable line
(`docs/evals/2026-09-26-arrival-redesign.md` section 7). CS2CD's cheaters take
60% of their kills with snipers; their fast non-sniper kills are rare. The
measurement is safe to show and will rarely have anything to say.

## What it leaves for v6

- The **corroboration rule** (spec section 3), independent of all measurements.
- `snap_kills` (non-sniper), which passes and rarely fires.
- `preaim_count`, if its own pro run passes (spec plan step 3, not yet run).

Triggerbot timing, the measurement v5 was built for, does not come back in this
form.
