# Round replay (design C)

Status: approved 2026-09-27. Revised the same day after step 0: the "look here"
marks failed their gate and are cut, and "who sees whom" comes from the mesh ray
cast instead of the game's spotting flag (docs/evals/2026-09-27-replay-marks-gate.md;
the user chose the ray cast, smokes not counted). Brainstormed with the
superpowers brainstorming process, architectural path. Ships in 0.5.0. No judge
change.

## Why

A report points at moments; a reviewer then needs the round around them. Today
that means CS2's own demo player: CS2 installed, the demo open, and still no way
to see what the suspect *could* have known. The replay answers the wallhack
question on the page itself, without CS2.

## What was decided

| question | answer |
|---|---|
| what a replay is for | both, switchable: the whole round by default; pick a player to see what they knew |
| what it shows besides players | smokes and fires, shots, the bomb, flashed players |
| where it lives | under the timeline (the approved "Round under the timeline" layout) |
| where its data comes from | saved at analysis, beside the report |
| "look here" marks | cut: failed the step 0 gate on CS2CD |
| who sees whom | the mesh ray cast plus the field of view, as the evidence; smokes drawn, not counted |

## 1. The page

The report keeps its order (answer, timeline, moments and the moment, player
panel; `web/DESIGN.md` "The report, top to bottom"). A replay is a full-width
section **between the timeline and the evidence columns**, present only while a
round is open. The evidence below it is unchanged.

```
dust2      1 of 10 players flagged: ...
           1  2  3  4  5  6  7  8 [9] 10 ...
Player A   .  o     O  .          |           <- playhead in round 9's column
...
Round 9    Play   1x 2x 4x    0:41 since the round began   demo_gototick 51234 [Copy]   Close
+--------------------------+   CT  Player A  ---o---x.............
|                          |       Player B  --------|--o--x......
|  radar: the whole map,   |       ...               |
|  everyone; a followed    |   T   Player F  -----x..|............
|  player's cone strongest |       ...           playhead
+--------------------------+
Enemies of Player A: solid, they see them. Outline, a teammate sees them.
Faint, nobody on their team does. Walls count; smokes and sound do not.
[ Watch these first ... | player panel ... ]
```

### Opening, closing, the address

- The timeline's round numerals become buttons: "Replay round 9".
- The moment gets "Replay this round", which opens the round 5 s before that kill
  and follows the attacker.
- The address carries `r` (round) and `f` (followed player id), e.g.
  `#/report/<id>?p=..&k=..&r=9&f=..`. Back and a pasted link reopen the same
  round. The moment on screen is written as `t` with `history.replaceState` when
  playback pauses or a scrub ends, never while playing: a hash change rebuilds the
  page.
- "Close" (and Esc inside the round) removes `r` and `f`.
- The fetched replay is cached per report in memory, so rebuilding the page on a
  hash change does not fetch it again.

### The radar (left)

- The whole map, Valve's radar where there is one. The existing kill radar's
  drawing (teardrop pointer, cone of view, names, floors) moves into shared
  helpers that both use; the kill radar looks the same afterwards.
- Every alive player: a pointer and a cone in the colour of the side they are on
  that round (`--ct`, `--t`, the mark strengths), a short name in the
  semi-condensed width. The followed player's cone is the stronger, longer one
  the attacker has on the kill radar; everyone else gets the victim's.
- Dead: a faded hollow ring where they died, until the round ends.
- Flashed (`flash_duration > 0`): the pointer at half strength and no cone.
- A shot: a short graphite tracer from the shooter along their view, fading over
  150 ms.
- A smoke: a disc of the smoke's radius (144 units) in `--paper` at 80%, with a
  `--grid` edge, from detonation to expiry. A fire: a disc hatched in graphite at
  35% with a dashed edge, from start of burn to expiry.
- The bomb: a small graphite square on its carrier, on the ground where it was
  dropped, or at the plant with a ring round it. Defused and exploded end it.
- Stacked maps: the Upper / Lower switch as on the kill radar. While following,
  the floor follows the followed player. Players on the other floor are drawn
  hollow and faded, as now.

### The round lanes (right)

- One lane per player, grouped by the side they are on that round, CT first.
- Along time, from the end of the freeze to the round's end: a line while alive,
  an x at death, the timeline's kill rings (filled for a headshot), a small tick
  per shot, and blind stretches hatched.
- A playhead crosses every lane and runs through the round's column in the
  timeline above.
- Click or drag anywhere on the lanes to move there. Click a name to follow that
  player; click it again to return to the whole round. The followed lane is
  tinted, as the selected lane is in the timeline.

### Controls and keyboard

- Play / Pause, speed 1x 2x 4x, as words in a segmented group like the floor
  switch. No icons, no arrows.
- The clock: seconds since the round began, and the `demo_gototick` command for
  the moment on screen, with Copy (principle 4: every claim one paste from its
  tick).
- While focus is inside the round: Space plays and pauses, the left and right
  arrows jump to the previous and next kill of the round, Esc closes. Outside it
  the page keeps its keyboard rules (arrows only in the timeline).
- The playhead is a `role="slider"` with the time as its value text, so the round
  can be scrubbed without a mouse.

### Following a player: what they knew

A player sees an enemy when nothing on the map mesh blocks the line from their
eye to the enemy's head (`occlusion.line_of_sight`, the evidence's own line of
sight) and the enemy is on their screen (`angles.in_field_of_view`, 106 deg). Not
the game's `spotted` flag: it misses a third of real sightings (step 0). A
followed player P sees every enemy E drawn one of three ways:

| E is | drawn |
|---|---|
| seen by P | solid, as now |
| seen by a teammate of P only (so on P's radar) | outline only |
| seen by nobody on P's team | faint ghost |

Teammates of P are drawn normally. The key under the radar says it in words:
walls count, smokes and sound do not, so "faint" means no one on the team had
them in sight past a wall, not that P could not have heard them. On a map with no
mesh there is no "who sees whom": following only strengthens the cone, and the
key says why.

### No "look here" marks

The spec's marks (a crosshair within 5 deg of an unseen enemy for 0.5 s while
their bearing turns 10 deg) failed the step 0 gate: 78% of clean CS2CD players
get one in a match. The replay has none, and magenta keeps meaning "look here"
only where the rest of the page puts it. The count difference the gate found
(cheaters' median twice clean's) belongs to design D.

## 2. The data

### Parsing

`parsing/worker.py` also parses, when the demo has them: `smokegrenade_detonate`,
`smokegrenade_expired`, `inferno_startburn`, `inferno_expire`, `bomb_pickup`,
`bomb_dropped`, `bomb_planted`, `bomb_defused`, `bomb_exploded`. They go into
`ParsedMatch.events` like the others but are not in `EVENTS_NEEDED`: CS2CD has
none of them, and nothing outside the replay reads them. Budget: parsing a demo
gets at most 1 s slower (measured on the 26 MB fixture demo, as 8645eb1 did).

### The replay file

`pipeline/replay.py` builds it after parsing, from the tick table and the events,
and analysis writes it beside the report as `<report id>.replay.json.gz`.
`analyze_demo(..., replay=True)`; the pro check passes `False`.

Per round, from the end of the freeze to the round's end:

- players: id, side that round, and every 4th tick (16 per second) x, y, z
  (whole units), yaw (whole degrees), health, alive, blind, and the enemies they
  see (as defined above) as a bitmask over the match's players;
- shots and deaths at their exact ticks;
- smokes and fires as (start, end, x, y); a smoke with no expiry event ends 18 s
  after detonation or at the round's end, a fire at its round's end;
- the bomb as a list of (tick, what, who, x, y); the carrier before the first
  pickup is unknown and drawn nowhere.

Budget: at most 2 MB compressed for a full 24-round match (measured: 0.08 MB for
a 10-round Wingman match at 16 per second), built in under 2 s with the ray casts
(0.2 s per CS2CD match once the mesh is loaded, which layer 2 has already done).
The report gains `replay: bool`.

### Serving it

`GET /api/reports/{id}/replay` serves the file as is, with
`Content-Encoding: gzip`, behind the same Host and Origin check as every route.
`{id}` goes through the report id check the report route already uses; a report
without a replay answers 404.

A report analysed before 0.5.0 has no replay: its round numerals stay plain text,
and one line under the timeline says "Replays need the demo analysed again with
0.5.0 or later."

## 3. Checks

**Step 0, done: the marks gate on CS2CD.** Marks were to ship only if at most 10%
of clean players got one in a match and banned players got one at least twice as
often. Both runs failed (docs/evals/2026-09-27-replay-marks-gate.md); the marks
are cut.

**Tests.**
- Building: sampling every 4th tick, the side per round, the sees bitmask.
- Who sees whom on a made-up mesh: a wall blocks, an enemy behind the viewer is
  not seen, a clear line in front is.
- Smokes and fires paired by entity id, and the missing-expiry rules.
- The bomb's carrier from pickups and drops.
- The endpoint: gzip headers, 404 without a replay, a bad id refused.
- The worker: a demo without the new events still parses.

**The page.** `web/fake_report.py` gains a synthetic replay for one round, with
invented names, so the page can be looked at and screenshotted without a real
player. Screenshot checks: desktop, phone, and a stacked map (a real Nuke demo,
screenshots kept on this machine).

## Not in this version

Weapons held, money, grenade flight paths, sound, and anything the judge reads.

## Risks

- **The ray cast is eye to head.** A crouched player, or one showing only a
  shoulder, can be seen with the head line blocked, and the other way round. It
  is the same line the evidence uses, so the replay never contradicts a kill's
  facts.
- **Smokes and sound are not counted.** A "faint" enemy may have been heard, and
  a "solid" one may have been inside a smoke. Smokes are drawn, and the key says
  both.
- **Report size.** A whole match of samples is new data per report; the budget
  above is checked in the tests on the fixture demo.
- **Page rebuilds.** The router rebuilds the report on every hash change; the
  replay caches its data and never writes the hash while playing.

## Plan

0. Done: marks gate on CS2CD (`training/replay/marks_gate.py`); failed, marks
   cut.
1. Worker parses the replay events; tests.
2. `pipeline/replay.py`: samples, who sees whom, events; tests.
3. Analysis writes the file, the report gains `replay`, the endpoint; tests.
4. `fake_report` writes a synthetic replay.
5. The page: shared radar helpers (the kill radar unchanged, screenshot before
   and after), the round section, lanes, playback, following, address and
   keyboard. Screenshot-checked on desktop, phone and Nuke.
6. `web/DESIGN.md` section, README line, CHANGELOG, then 0.5.0.
