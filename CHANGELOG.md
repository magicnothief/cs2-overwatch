# Changelog

Each version's section becomes its release notes on GitHub. Newest first.

## 0.4.0 (2026-09-27)

**The report**
- A report opens with its answer: how many players are flagged, who, and each
  verdict, with the reminder that a score is not proof. When nobody is flagged,
  it says so.
- The kills to watch first are a numbered list. Under each kill, what was
  measured is kept apart from the setting (distance, what the player could have
  known), because only a measurement can point at cheating.
- Each behaviour score has a bar with the 90% flag line marked on it.
- Every shot is marked on the crosshair trace, and a kill says how long after
  the head came under the crosshair the first shot came.
- Keyboard: a skip link jumps past the timeline to the evidence, and ↑ ↓ change
  player only while the timeline has focus.

**The judge**
- Still judge v4, and it now reads exactly the evidence it was trained on.
  Re-checked on this version: its targets matched 92.2% of the time on 206
  held-out cases, with no invented numbers, and on 35 pro matches it accused
  none of 341 players and said "unclear" for 23 (6.7%), the same as 0.3.0.
- Two new measurements (a triggerbot's timing, and a count of very fast turns)
  are computed but not shown to v4, which was never trained on them. Judge v5
  was trained on them and failed its checks on pro players; they are being
  redesigned.

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

**Safety**
- Demos are now read in a separate short-lived process, which is given the demo
  and nowhere to write but its own scratch directory. A demo is a file someone
  else made; if reading one ever crashes, it takes down that process and the
  review fails, instead of touching the program that holds your reports and the
  page. Costs about half a second per demo.
- A compressed demo cannot fill your disk: unpacking stops at 2 GB, and a file
  that does not start like a CS2 demo is refused after its first bytes.
- Other websites open in your browser can no longer reach the review server on
  127.0.0.1. The page also has a Content-Security-Policy.
- The names, map and rank a demo carries are checked before they reach the
  judge, so a crafted demo cannot write instructions into its prompt.
- The update notice links only to GitHub release pages.
- A fresh install takes the demo parser and web server libraries only up to the
  versions this release was tested with.
- The README now says exactly what leaves your PC, and what a report is not.

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
