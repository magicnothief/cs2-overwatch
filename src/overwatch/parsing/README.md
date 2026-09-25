# parsing: `.dem` / CS2CD → normalized tables

**Purpose:** turn raw demos *and* CS2CD matches into the **same** parquet tables (ticks plus
events), so everything downstream only has to handle one format.

**Learn first:** the demoparser2 API (`parse_ticks`, `parse_event`, `list_game_events`),
parquet, polars or pandas.

**First exercise** (`notebooks/01_first_demo.ipynb`):
1. Parse one HLTV demo and list every event type in it.
2. Get X, Y, Z, pitch, and yaw for all players.
3. Plot one player's yaw over one round.
4. Compute angular speed. You will hit the **wrap-around bug**: going from 179° to −179° is a
   2° move, not 358°. Fix it and understand why the fix works.

**Done when:** `parse(path)` returns identical column sets for a `.dem` and a CS2CD match.

**Pitfalls:** warmup and knife rounds; halftime side swap; bots without SteamIDs; disconnects;
tick vs. game time; dead players still have view angles.
---

## The contract (M2, implemented)

Both `parse_demo()` and `load_cs2cd()` return a `ParsedMatch` with these tick columns,
in this order (`TICK_COLUMNS` in `types.py` is the source of truth):

| Column | Type | Notes |
|---|---|---|
| `player_id` | str | SteamID64 as text, or `"Player_3"` for CS2CD |
| `player_name` | str | null for anonymized sources |
| `tick` | i32 | 64 ticks per second |
| `round` | i32 | rounds played before this tick |
| `team` | i32 | 2 = T, 3 = CT |
| `x`, `y`, `z` | f32 | world position |
| `pitch`, `yaw` | f32 | view angles, degrees |
| `health` | i32 | |
| `is_alive` | bool | |
| `is_freeze` | bool | freeze period (buy time) |

Events are `ParsedMatch.events[name]`, always present (empty frame if the source has
none), for: `player_death`, `player_hurt`, `weapon_fire`, `round_freeze_end`,
`round_start`. `player_death` is normalized to `tick, attacker_id, victim_id, weapon,
headshot` plus whatever extras the source carried (`distance`, `penetrated`, …).

`meta` holds `source` (`"dem"` / `"cs2cd"`), `map`, `tick_rate`, and per-source extras:
`sha256` for demos, `cheaters` and `avg_rank` for CS2CD.

Derived aim columns (`d_yaw`, `yaw_speed`, …) are **not** here. They live in
`overwatch/aim.py`, because Layers 1 and 3 both need them. See ADR 0004.
