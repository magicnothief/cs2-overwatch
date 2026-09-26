# Overwatch review: design notes

**Brief.** A local tool for reviewing CS2 demos for cheating. The reviewer is one
person checking their own and others' matches. Its job: show who is suspicious,
why, and get them to the exact moment in the demo to check it themselves.

**Direction: round timeline.** A match is rounds, and a reviewer thinks in rounds
("the round 9 deagle"). The page is built around a timeline: one lane per player,
one column per round, every kill marked where it happened. Chosen from three
directions on 2026-09-23 (the others were a radar built from the collision meshes,
and a case-file layout around the judge's prose).

## Tokens

Every one of these is a CSS custom property in `app.css` `:root`. Nothing in
this page gets a loose hex or a loose rem: if a value is used twice, it is a
token, and if it is not in this table it does not exist.

| name | custom property | hex | job | contrast |
|---|---|---|---|---|
| paper | `--paper` | `#FBFCFD` | the page |  |
| graphite | `--graphite` | `#1A1F24` | text, kill marks, the trace | 16.16 on paper |
| muted | `--muted` | `#5E666F` | support text, never a new hue | 5.67 paper, 5.14 tint |
| grid | `--grid` | `#DDE2E7` | round columns, lane rules, meter track |  |
| tint | `--tint` | `#EEF1F4` | the selected lane, the answer block |  |
| flag magenta | `--flag` | `#B8336A` | "look here", and nothing else | 5.50 paper, 4.98 tint |
| CT blue | `--ct` | `#3E6FB0` | a CT mark on the radar | 3.73 on radar grey |
| CT blue, ink | `--ct-ink` | `#35609B` | the word "CT" in running text | 6.20 paper |
| T gold | `--t` | `#A8700F` | a T mark on the radar | 3.07 on radar grey |
| T gold, ink | `--t-ink` | `#8C5A08` | the word "T" in running text | 5.71 paper |

Two strengths per team colour because a mark and a word are not the same job:
marks clear 3:1 (WCAG 1.4.11), ink clears 4.5:1 (1.4.3). The numbers above are
measured, not assumed; recompute them if a hex moves.

**Type: Archivo, one variable family, width as hierarchy.** Expanded (`wdth` 125)
for match-level things: the map name, round numerals, section titles. Normal
width for reading. Semi-condensed (`wdth` 87) for the dense player lanes, where
names must fit. Tabular figures everywhere a number sits in a column. The one
monospace string is the `demo_gototick` command, because it is code you paste.

| step | property | size | used for |
|---|---|---|---|
| −1 | `--s-1` | 12.8px | legends, the support sentence beside a score, `.moments .why` |
| 0 | `--s0` | 16px | body, `h3`, the `demo_gototick` string |
| 1 | `--s1` | 20px | `h2`, the answer line, a measured fact |
| 2 | `--s2` | 25px | the selected player's name |
| 3 | `--s3` | 31.25px | unused; kept so the ramp is a ramp |
| 4 | `--s4` | 39.06px | `h1`, the map name |

Two weights carry everything: 400 and 700 (600 only where a control must read as
a control). Line height 1.45 for reading, 1.12 for headings, 1.3 for the answer.

**Spacing: 4px root, doubling.** `--space-1` 4px, `--space-2` 8px, `--space-3`
12px, `--space-4` 16px, `--space-5` 24px, `--space-6` 32px, `--space-7` 48px.
The page gutter is `--gutter`, `clamp(1rem, 3vw, 2.5rem)`. New CSS uses these;
the older rules that still carry loose rems are being converted as they are
touched, not in one sweep.

## Layout

```
de_dust2                        xi-vs-rebels-m1-dust2.dem, 18 rounds, 98 kills
line of sight: ray cast against the de_dust2 mesh           Analyse another demo

           1   2   3   4   5   6   7   8   9  10  11  12 ‖ 13  14  15  16  17  18
|Player A  .   o       O   .                 O          ‖  .           o        0.44  above 88%
|Player B  o           .   o   O                        ‖      o                0.32  above 75%
 ...                                                    ‖
|Player C      o               o   o                    ‖          o            0.19  above 34%
 ^ team colour (starting side)                          ^ sides switch

Round 9, 32952 ticks in                   | Player A
[ crosshair-to-head trace, hidden         | Flagged: score above 96% of clean
  stretches hatched ]                     | Judge: cheating, 92%, aimbot
Deagle headshot on ubir, 17 m             |   reasons ...
100% of the approach aimed through cover  |   what would change its mind ...
demo_gototick 32824   [Copy]              | What the judge read (expands)
```

Everything is left-aligned; numbers are right-aligned in their column. Kill
marks: a ring, filled when it was a headshot, larger the higher its score; the
kills a reviewer should watch first are magenta. The selected player's lane is
tinted; ← → step through their kills, ↑ ↓ change player.

## Principles

1. **Time is the axis.** Rounds and the side switch are the page's structure.
2. **Magenta means look here.** Flagged players and their kills to watch. Team
   colours only say which side someone started on.
3. **Width carries hierarchy**, so size and colour do not have to.
4. **Every claim is one click from its tick.** The command is always there to copy.
5. **Boldness in one place: the timeline.** Everything around it stays quiet.

## The report, top to bottom (2026-09-26)

The reviewer is deciding one thing — did this named player cheat — and has to be
able to defend it afterwards. The page is ordered as that answer and its
support, verdict first (inverted pyramid), not as the data structure.

1. **The answer, `.finding`.** One line under the map name: how many players are
   flagged, who they are as buttons that jump to their first moment, and each
   verdict as a chip. Under it, in `--s-1` muted, the thing that keeps this
   sober: *a score and a verdict are not proof*. Most runs flag nobody, so that
   case gets its own sentence — "Nobody here is flagged", with the line it was
   measured against — and a graphite rule instead of a magenta one. The quiet
   case is the product; it is not an absence.
2. **The timeline.** Unchanged as the page's frame. Each score now carries
   `.meter`: the player's percentile as a length, with a 1.5px graphite tick at
   the flag line, because "above 86%" and "above 91%" are one word apart and a
   world apart. The supporting sentence dropped to `--s-1` so the number leads —
   and so the verdict chip stays inside a 1200px viewport.
3. **The moments, `.moments`.** `top_kills` was already ranked and already in
   the report; the only way to reach the second one used to be an arrow key
   nobody was told about. Now it is a numbered list: round, what happened, why
   it is here, the kill's score. The chosen row is tinted *and* says "→ shown
   below", because a choice must never be carried by colour alone.
4. **The moment itself.** Radar, crosshair trace, then the facts split in two:
   **What was measured** at `--s1` (the measurements an accusation could rest
   on) and **The setting** at body size (true, useful, not what the claim rests
   on). Then the `demo_gototick` command, which is the whole point: every claim
   one paste from the tick that made it.
5. **The player panel.** Score, flag reasons, hard limits (`.limit`), the
   judge's verdict, what would change its mind, and the raw text the judge read
   behind a `<details>`. Progressive disclosure: the answer is up top, the
   arithmetic is one interaction away.

Class names to keep apart: `.finding` is the match-level answer block; `.limit`
is one broken hard limit in the player panel. They were briefly the same name
and the panel inherited a box it should not have had.

## Keyboard and screen reader

- A **skip control** before the timeline jumps to the evidence, because roughly
  110 kill rings and 10 names sit between the top of the page and the moment.
  It is a `<button>`, not an anchor: the page routes on `location.hash`, so
  `href="#the-moment"` would leave the report and land on the home page.
- Choosing a moment rebuilds the detail column under the reader's own finger, so
  focus is put back on the chosen row (`.moments li.here button`).
- ↑ and ↓ change player **only while focus is inside the timeline**. Everywhere
  else they are how a keyboard scrolls, and stealing them made the detail column
  unreadable.
- A flagged lane is magenta *and* carries the word: a verdict chip when the judge
  ran, a "Flagged" chip when it did not, plus an `.sr-only` "(flagged)" in the
  name.
- The timeline is `role="group"`, not `role="grid"`: a CSS grid has no row
  elements to carry `role="row"`, and a grid a screen reader cannot walk is
  worse than no role.
- The meter is `aria-hidden`; its sentence ("above 81% of clean players") is the
  text alternative and sits right beside it.

## Looking at the page without a demo

`python -m overwatch.web.fake_report` writes two synthetic reports into the
reports folder — one with a flagged player and a judge verdict, one where nobody
is flagged and the judge did not run — and prints their URLs. Every name, Steam
ID and tick in it is invented, which is also the rule for screenshots: no real
player ever appears in an artifact that leaves this machine.

## Checked against the generic defaults (frontend-design skill)

- Not cream + serif + clay, not black + acid accent, not a broadsheet. The rules
  that exist are the round grid: they carry data.
- No card kit: panels are separated by space and one rule, no shadows, no
  gradients, no single radius on everything.
- No all-caps eyebrows, no middle-dot meta strings (commas instead), no arrows on
  buttons, no monospace labels.
- **Revised:** the first sketch opened with a stats row (players, kills, flagged
  as big numbers with small labels). That is the default hero; cut. The timeline
  is the hero, and counts sit in one sentence under the map name.
- **Revised:** kill marks were first coloured on a continuous score ramp. A ramp
  invites reading small differences that mean nothing; now size carries the score
  and colour is reserved for "look here".
- Motion: one moment only. The timeline's kill marks appear round by round when a
  report opens; nothing else animates, and reduced-motion turns that off.

## Radar (added 2026-09-24)

Asked for as "make it cooler": the radar direction from the first round of
choices, brought in as a component rather than the page's frame.

- **The map is drawn from where players walked**, not from the collision mesh.
  The mesh from above was faithful and unreadable (roofs, props, stair treads);
  positions from 30 CS2CD matches per map trace every corridor, like the in-game
  radar, and exist for all 15 maps. Floors lighter where higher; the walkable
  edge is the wall line. Drawn offline by `training/rules/render_radars.py`.
- **Team colours finally carry meaning here**: each player's trail, in the side
  they were on at that kill. The sightline is graphite: solid when they could see
  each other, dashed through a wall.
- **Who looks where** (reworked 2026-09-26, after "make it clearer which player
  is looking in which direction"): each player is a teardrop pointing along
  their view, one shape with one outline so the point reads at a glance, and
  carries a cone of view that fades with distance like a torch beam, in their
  side's colour (the attacker's stronger and longer). The caption says it in
  words: how many degrees the attacker's view is off the victim, and whether the
  victim faces them, faces away, or is side-on. After the shot the victim is a
  hollow ring with neither: a dead player looks nowhere. Reports analysed before
  this change have no victim view, so the victim stays a plain dot.
- **Names on the dots.** The timeline colours a player by the side they started
  on, the radar by the side they are on that round; after the teams swap (round 9
  in Wingman, 13 in a full match) the colours alone suggested the wrong player
  had died. Each dot now carries its player's name, bold for the attacker.
- **One moment drives both views.** Moving along the crosshair trace (or the
  slider, for keyboard and touch) steps both players through the approach, with
  a cursor on the trace. This is the page's one piece of motion that answers the
  reader, which the principles allow.
- A 10 m scale bar, because the kill facts give distances in metres.
- Maps with no radar still get the trails on a plain field, and say so.
- **Valve's radars where the game has one** (2026-09-26, after "the Nuke radar is
  almost unrecognisable"). Drawn radars are faithful but a floor of many small
  rooms turns into blobs; Nuke was the worst. CS2 ships a hand-drawn radar for 16
  maps (every active-duty map, Office, Italy, Cache, the Arms Race maps), with a
  lower-floor image for Nuke, Vertigo, Train and Baggage and an overview file
  that places it in the world. It is read from the user's own CS2 like the
  meshes, and recoloured into the page's greys (`maps/radar.py` restyle):
  brightness becomes the floor tone, so tunnels and covered areas are dimmer;
  a change of Valve's colour becomes a faint seam, which keeps rooms, ramps and
  boxes apart; Valve's thin outlines and our outer wall line stay dark. Team
  colours stay reserved for players. Maps without one keep the drawn radar, and
  a map prepared before this is upgraded on its next analysis.
- **Maps nobody recorded** (de_eldorado, cache, the Arms Race maps, ...) are drawn
  from the collision mesh instead: floors with room to stand, reached on foot from
  the spawn points without passing a wall or player clip
  (`l2_perception/geometry/walkable.py`). Checked against the walked radars first:
  on dust2, mirage and inferno it covers 96-98% of where players went, and the
  area it adds is the corners and spawn rooms people rarely stand in.
- **Stacked maps get a radar per floor** (added 2026-09-25). On Nuke and Vertigo
  one image put B site under A site, so a kill on either floor was drawn over
  both. A map is split when players stand a storey (150 units) apart over more
  than 10% of it (Nuke 27%, Vertigo 24%, every other recorded map 7% or less), at
  the height between the two floors (Nuke -505, Vertigo 11646; Valve's own
  radars split at -495 and 11700). The radar opens on the floor the attacker
  stood on at the shot; a small Upper / Lower switch in its corner changes floor.
  Whatever is on the other floor (a player, a stretch of trail) is drawn hollow
  and faded rather than hidden, and the caption says who is where, so a shot
  through a floor still reads.
  Mesh-drawn maps take the same test on their reachable floors, cell by cell:
  Arms Race Baggage (the carousel hall over its tunnels) and Pool Day (two copies
  of the arena, 8192 units apart) split; the rest stay one image.
