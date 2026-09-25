# 0004: One canonical table shape for every source

- **Date:** 2026-09-22
- **Status:** accepted

## Context
Data arrives from two places: raw `.dem` files (demoparser2) and CS2CD matches
(parquet + JSON, anonymized). Both were parsed with demoparser2, so they are close,
but not identical: CS2CD has no player names, identifies players as `"Player_3"`
rather than SteamIDs, and keeps events in a JSON file instead of the parser.

Without one shape, every layer needs two code paths, and the CS2CD labels can't be
used to test code that runs on real demos.

## Options considered
1. **Convert both into one canonical table.** One extra module per source; every
   layer written once.
2. **Let each layer handle both.** No conversion, but the `if source == ...`
   branches spread everywhere and the two paths drift apart.

## Decision
Option 1. `parsing/types.py` defines `TICK_COLUMNS` and `ParsedMatch`; `demo.py` and
`cs2cd.py` each produce exactly that. `player_id` is a **string** in both, because
CS2CD's `"Player_3"` cannot become a 64-bit integer. The real SteamID is kept
in `player_id` too for demos (as text), so nothing is lost.

## Consequences
- A layer never asks where its data came from. `meta["source"]` exists for reporting.
- Adding a source (FACEIT, a different parser) means writing one loader, nothing else.
- Joining against Steam APIs needs `int(player_id)`, which only the collector does.
- The tick table is intentionally small (13 columns). Props like `flash_duration`,
  `spotted` and `shots_fired` get added when a layer actually needs them, and adding
  one means updating `TICK_COLUMNS`, both loaders, and the tests together.
