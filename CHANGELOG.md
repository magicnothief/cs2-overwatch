# Changelog

Each version's section becomes its release notes on GitHub. Newest first.

## Unreleased

**Getting demos in**
- The first page lists the demos already on this PC: those CS2 saved from
  Watch → Your Matches, and those in Downloads. One click reviews a demo where
  it lies, without an upload or a copy, and a reviewed one links to its review.
- Compressed demos (.dem.gz, .dem.bz2, .dem.zst, as FACEIT and others serve
  them) can be dropped or picked as they come.
- "How to get a demo" explains where a demo comes from, for matchmaking,
  FACEIT and tournaments.
- A broken or cut-off demo now fails with a message instead of leaving the
  review "running" for ever.

## 0.3.0 (2026-09-26)

**The judge**
- Judge v4 is the default. It reads a new piece of evidence: how many kills
  came within 50 ms of the enemy appearing, instead of the single fastest
  reaction (one pre-fired angle was enough to count before). Against v3 on 206
  held-out cases it matches its targets 92% of the time (was 81%), convicts 29%
  of banned players (was 15%), and invents no numbers. Checked on 341 players
  from 35 pro matches: none accused.

**The radar**
- Valve's own radar where the game has one (16 maps, with lower floors for
  Nuke, Vertigo, Train and Baggage), read from your CS2 and recoloured to fit
  the page. Nuke is recognisable again. Other maps keep the drawn radar.
- Each player is a pointer showing where they look, with a cone of view, and
  carries their name: colours alone misled once teams had swapped sides.
- The caption says how far the attacker's view is off the victim, and whether
  the victim faces them.

**The page**
- Choosing a kill no longer makes the timeline's rings slide into place, and a
  phone keeps its sideways scroll.
- A notice when a new version is out, with the command to update. The check
  asks GitHub once a day and can be turned off under "This computer".

**Installing and updating**
- The install scripts install the latest release; running them again updates.
  `overwatch update` does it in one step on Linux; `overwatch --version` says
  which version runs.
- Models are checked against their pinned checksums on every start, so an
  update that ships a new model replaces the old file.
- Dropped downloads are retried and resume where they stopped.

## 0.2.0 (2026-09-25)

First public version: one-command install on Windows and Linux, the detector on
ONNX Runtime, judge v3 on llama.cpp's prebuilt server (Vulkan, CUDA or CPU),
maps and radars prepared from your own CS2, stacked maps split by floor.
