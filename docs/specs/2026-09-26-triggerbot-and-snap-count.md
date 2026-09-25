# Triggerbot timing, and fast turns counted (judge v5)

Status: approved design, 2026-09-26. Brainstormed with the superpowers
brainstorming process; spikes and their numbers are below.

## Why

- **A triggerbot fires the moment the crosshair is on an enemy.** Nothing in the
  evidence measures that today. The spike found it separates cleanly.
- **One fast turn made pros "unclear".** 19 of judge v4's 23 "unclear" verdicts on
  341 pro players rest on "fastest turn on a kill tick" past the clean 95% line:
  one kill decides a player's maximum, and 17 of the 19 used no sniper at all
  (rifle turns of 287-776 deg/s). The same flaw "fastest reaction" had, fixed by
  counting fast kills.

## Evidence the spikes found (CS2CD, 240 matches)

Shot on the very tick the crosshair reached the enemy's head (the first shot
after the crosshair's last arrival, same tick):

|  | clean | banned |
|---|---|---|
| kills, sniper rifles | 7.4% | 21.7% |
| kills, other weapons | 9.3% | 15.0% |
| players over half their non-sniper kills | 0% (of 536) | 5.9% (of 101) |
| players over a third | 4.1% | 14.9% |

Not a sniping artefact: clean snipers do it less often than clean riflers.

Fast turns (all 5,152 players with 5+ kills):

| evidence | clean past 95% line | banned past 95% | banned past 99% |
|---|---|---|---|
| fastest single turn (today) | 5.0% | 18.7% | 7.6% |
| kills with a turn over 200 deg/s | 1.6% | 13.1% | 8.9% |

## Design

### 1. What the judge reads

- New measurement, **`arrival_shot_share`**: of a player's kills where the
  crosshair arrived on the head before the first shot, the share fired on the
  arrival tick itself. Shown only with at least 5 such kills. Past the clean
  95% line it is notable, past 99% strong; it points to a **triggerbot**.
  Judge text: "kills fired on the very tick the crosshair reached the head".
- **`snap_kills`** replaces `snap_max` in the evidence: the number of kills with a
  turn over 200 deg/s on the kill tick (`SNAP_THRESHOLD_DPS`). Judge text:
  "kills with a turn over 200 deg/s on the kill tick". Points to an **aimbot**.
  `snap_max` stays a player feature (models may use it); the judge no longer sees it.
- Target rules unchanged: 95% line notable, 99% strong, the behaviour score is
  never evidence, lines come from clean CS2CD players.

### 2. Where the data comes from

- Shots: `weapon_fire` (already parsed from demos and CS2CD). Knife, grenades
  and C4 do not count.
- `dataset.match_windows` marks, per window tick, whether the attacker fired
  (`shot`). `features.build_window_features` adds per kill:
  - the **opening shot**: the first shot of the last burst before the kill (a
    burst ends when 300 ms pass without a shot). Only it counts: mid-spray, a
    shot every ~6 ticks would land on a re-arrival by chance.
  - `arrival_tick_offset`: the last tick at or before the opening shot on which
    the crosshair came onto the head. "On the head" is within 8 world units of
    the head's centre, as an angle for the target's distance (floor 50 units).
  - `arrival_shot`: the opening shot came on the arrival tick (null when the
    crosshair never arrived before it).
  - `shots_ms`: the attacker's shot times relative to the kill, for the page.
- `player_features` adds `arrival_shot_share`, `arrival_kills` (the eligible
  count) and `snap_kills`.
- The detector is unchanged. The dataset is rebuilt (build_dataset, build_sequences
  unchanged in content), the clean reference refreshed with the new lines.

### 3. What the page shows

- The crosshair trace gets a short mark on the time axis at every shot.
- A kill fact: "Fired on the tick the crosshair reached the head" or "Fired N ms
  after the crosshair reached the head" (omitted when it never arrived first).
- `KillReport` gains `shots_ms: list[int]` and `arrival_ms: int | None`; older
  reports simply lack them.

### 4. Training and checks

- Rebuild cases and the training set with both changes; copy
  `unsloth/{train,val}.jsonl` to `C:\cs2-overwatch-v5-data`; the user trains v5
  in Unsloth Studio on Windows with v3/v4's settings.
- v5 ships only if:
  1. on the 206 held-out players it matches or beats v4's target match (92%),
     invents no numbers, and accuses a clean-labelled player only where that
     player's evidence target itself says cheating;
  2. on the 35-match pro check it accuses none, and says "unclear" less often
     than v4's 6.7%;
  3. pros do not trip the new line: if more than 1% of pros are past its 95%
     line, the measurement is redesigned before shipping.
- Then: publish v5 and the reference, pin them, release 0.4.0.

## Risks

- **The spike's definition was looser** (the first shot after the last arrival
  before the kill); plan step 0 re-measures with the opening shot.
- **Sub-tick shots.** CS2 records shots at sub-tick times; a click landing
  just before the arrival tick counts as "1 tick early". The spike already ran
  at tick resolution and still separated, so this lowers recall, not precision.
- **Pre-aimed kills never "arrive"** (the crosshair was already on the spot), so
  they are not eligible; the share only speaks for kills with a movement onto
  the head. That is the triggerbot's case.
- **Pro behaviour is unmeasured** for the new line until the pro check; check 3
  guards it.

## Plan

0. Re-measure the spike with the opening-shot definition; go on only if clean
   and banned still separate (players over half: clean near 0%, banned clearly
   above).
1. Window shot flags and features (dataset, features), with tests on the fixtures.
2. Player features and judge evidence (targets, rendering), with tests.
3. Report fields and the page (trace marks, kill fact), screenshot-checked.
4. Rebuild the dataset, cases and training set; refresh the reference; copy to C:.
5. After the user trains v5: evaluate, pro check, then ship per section 4.
