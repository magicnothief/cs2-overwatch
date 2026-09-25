# Changelog

Each version's section becomes its release notes on GitHub. Newest first.

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
