# Round replay, step 0: the "look here" marks gate — 2026-09-27

**Verdict: marks are cut. They fire for most clean players in every match, with
either source of visibility. The game's own `spotted` flag is also unfit for the
replay's "what they knew" view; the mesh ray cast replaces it.**

Spec: `docs/specs/2026-09-27-round-replay.md` section 3. Script:
`training/replay/marks_gate.py`. Data: CS2CD, every match, per-player labels
(`meta["cheaters"]`); no pro data, no judge, no GPU.

## The rule, fixed before either run

A mark is a stretch where a player's crosshair stays within 5 deg of the head of
an enemy nobody on their team can see, for at least 0.5 s, while the enemy's
bearing from the player turns by at least 10 deg. Samples every 4th tick of live
play, alive players only. Marks ship only if (1) at most 10% of clean players get
one in a match and (2) cheaters get one at least twice as often.

## Run 1: "can see" from the game's `spotted` flag, as the spec said

795 matches, 7,949 player-matches.

| | players | with a mark | marks, median | marks, p90 |
|---|---|---|---|---|
| clean | 6,640 | **84.4%** | 3 | 8 |
| cheater | 1,309 | 89.5% | 5 | 15 |

Gate 1 fails (84.4% against 10%), gate 2 fails (1.06x against 2x).

## Why the flag cannot carry "what they knew"

A tick before a gun kill the victim is, by definition, almost always in the
attacker's sight. The flag says so far less often:

| | gun kills | victim spotted by anyone | by the attacker |
|---|---|---|---|
| CS2CD, 40 random matches | 4,343 | 65.6% | 63.9% |
| a real matchmaking demo (Wingman) | 24 | 79.2% | 75.0% |

It is not lag. Over the whole last second before the kill, the attacker "saw" the
victim at least once on only 67.7% of CS2CD kills (64.1% on the last tick, 65.3%
over 16 ticks, 66.5% over 32). A third of real sightings are missing, so a replay
that drew enemies "unseen" from this flag would be wrong about a third of the
time exactly where a reviewer looks.

## Run 2: "can see" from the mesh ray cast, the evidence's own line of sight

Pre-registered after run 1, before it ran: the same rule and thresholds, with one
input changed. An enemy is hidden from a team when no one on it, the viewer
included, has both a clear line on the map mesh (`occlusion.line_of_sight`, eye
to head at 64 units) and the enemy inside their field of view
(`angles.in_field_of_view`, 106 deg). 685 of 795 matches have a map mesh here.

| | players | with a mark | marks, median | marks, p90 |
|---|---|---|---|---|
| clean | 5,564 | **77.8%** | 2 | 6 |
| cheater | 1,285 | 87.1% | 4 | 14 |

Clean players in matches with no cheater at all: 84.3%. Gate 1 fails (77.8%
against 10%), gate 2 fails (1.12x against 2x). Cost: 0.2 s per match once the
map's mesh is loaded.

## What this means

- **Marks are cut** from the replay, as the spec says. The rule moves out of the
  product into `training/replay/marks_gate.py`, which keeps both runs
  reproducible.
- **Crosshair "following" an unseen enemy is ordinary play.** A player who moves
  turns the bearing to everyone; players pre-aim at head height along the walls
  enemies really stand behind. Whether a mark happens at all says nothing.
- **How many is a lead, not a result.** Cheaters have twice the median count (4
  against 2) and more than twice the p90 (14 against 6). A count against a clean
  line is how `fast_kills` and `snap_kills` became evidence; it belongs to
  design D (unusual pre-aims), with its own gate.
- **The replay's "what they knew" view uses the ray cast, not the flag**: seen
  by the followed player (clear line and on their screen), seen by a teammate
  only, or by nobody on the team. Spec change, for approval.
