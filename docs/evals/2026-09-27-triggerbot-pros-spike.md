# Spike: why 4 pros tripped triggerbot timing — 2026-09-27

**Answer: not sniping and not a flawed measurement. Pros trip its 95% line less
often than ordinary clean players do (1.58% against 5.25%); the check it failed
asks pros to trip it at most 1%, a ceiling set when one line could accuse on its
own. Whether that ceiling still fits is a decision, and a changed criterion would
have to be confirmed on a pro sample nobody has looked at.**

Follow-up to `docs/evals/2026-09-27-v6-measurements-pro-check.md`, where
`arrival_shot_share` (crosshair sweep of 50 deg/s or more, at least 5 eligible
kills) put 4 of 253 shown pros past its 95% line and failed the v6 spec's 1%
ceiling. Same data: the 170 disjoint matches of `pro_arrival_lab_big`, CS2CD's
`arrival_lab.parquet`. No judge, no GPU. The pros are real players; they are
numbered here, not named.

## The four

| pro | map | arrival-tick kills | of eligible | sniper kills among them | weapons on the tick |
|---|---|---|---|---|---|
| 1 | de_ancient | 4 | 5 | 0 | AK-47 x2, Galil, USP-S |
| 2 | de_overpass | 3 | 5 | 0 | MP9 x2, M4A1-S |
| 3 | de_ancient | 3 | 5 | 0 | Galil, M4A4, Five-SeveN |
| 4 | de_dust2 | 3 | 6 | 0 | M4A4 x2, AK-47 |

No sniping: a sniper-free version would change nothing. Each sits on 3 or 4
arrival-tick kills out of 5 or 6, the fewest the line allows (it takes 3 of 5).

## The populations

| | arrival kills | fired on the arrival tick, pooled | players shown | past the 95% line |
|---|---|---|---|---|
| CS2CD clean | 7,499 | 9.05% | 305 | 5.25% |
| CS2CD cheaters | 2,506 | **30.05%** | 167 | **47.90%** |
| pros | 4,440 | 12.05% | 253 | **1.58%** |

Pros fire on the arrival a little more often than clean players, which is what
skill looks like, and far less than cheaters. Per player they are past the line
three times less often than the clean population the line was drawn on.

## What the ceiling was for

The 1% ceiling is gate 3 of the v5 spec (2026-09-26), written when one
measurement past a line could make a `cheating` verdict on its own. Since 0.5.1
none can: a 95% line is notable, never decisive, and an accusation needs a
second line past its own and one of them past 99%. Pros past this measurement's
99% line: 0 of 253.

## Options

- **Keep the verdict.** The criterion was fixed before the data; the measurement
  stays out.
- **Change the criterion to "pros no more often past the 95% line than clean
  players", and confirm it on a fresh pro sample** (`pro_arrival_lab.py --per-map
  25 --seed 3`, never read, about two hours of CPU). This is a change made after
  seeing the numbers, which is why it cannot be settled on the sample that
  prompted it.
