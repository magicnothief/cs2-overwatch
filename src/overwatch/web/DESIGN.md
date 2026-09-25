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

| name | hex | job |
|---|---|---|
| paper | `#FBFCFD` | the page |
| graphite | `#1A1F24` | text, kill marks, the trace |
| grid | `#DDE2E7` | round columns, lane rules |
| CT blue | `#3E6FB0` | a team that started as CT |
| T gold | `#C98A1B` | a team that started as T |
| flag magenta | `#B8336A` | "look here", and nothing else |

Muted text is graphite at reduced strength (`#5E666F`), never a new hue.

**Type: Archivo, one variable family, width as hierarchy.** Expanded (`wdth` 125)
for match-level things: the map name, round numerals, section titles. Normal
width for reading. Semi-condensed (`wdth` 87) for the dense player lanes, where
names must fit. Scale 1.25 from 16px: 12.8 / 16 / 20 / 25 / 31.25 / 39. Tabular
figures everywhere a number sits in a column. The one monospace string is the
`demo_gototick` command, because it is code you paste.

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
  each other, dashed through a wall. The attacker's view is a faint wedge.
- **One moment drives both views.** Moving along the crosshair trace (or the
  slider, for keyboard and touch) steps both players through the approach, with
  a cursor on the trace. This is the page's one piece of motion that answers the
  reader, which the principles allow.
- A 10 m scale bar, because the kill facts give distances in metres.
- Maps with no radar still get the trails on a plain field, and say so.
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
