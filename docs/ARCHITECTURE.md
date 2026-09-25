# Architecture

## 1. Pipeline at a glance

```
 .dem files (or CS2CD parsed matches)
    │
    ▼
[0] Parse & normalize   demoparser2 → per-tick table + event tables (parquet)       ~seconds/demo
    │
    ▼
[1] Blatant rules       hard physical/game limits → near-certain flags               ~ms
    │
    ▼
[3] Behavior            engagement windows → features → scorer → calibrated prob.   ~seconds
    │                   (+ SHAP: which features pushed the score up)
    ▼
[4] Judge               small fine-tuned LLM writes the reasoning for top moments   ~30 s/moment on CPU
    │
    ▼
 Report JSON  ──►  web UI

[2] Perception is a *service*, not a stage. Layers 1 and 3 both need to know
    "could the suspect see this enemy at tick T?", so they call it.
```

Why is Layer 2 a service? The cross-map kill rule (Layer 1) needs to know the victim was never
visible, and the wallhack features (Layer 3) need the same answer. If perception were a stage
that ran between them, Layer 1 would either run too early to use it or have to be reordered.

## 2. The core idea: one shared `Moment` object

Design the schemas **before** you write any layer (see `src/overwatch/schemas/README.md`).
Every layer reads Moments and appends Evidence to them. Sketch:

```
Match     demo_id, source (dem | cs2cd), map, tick_rate, players[], paths to parquet tables
Moment    match_id, suspect_steamid, tick_start, tick_end, trigger (kill | damage | first_sight | rule)
          evidence: list[Evidence], verdict: Verdict | None
Evidence  layer, name, value, baseline (what legit players do), severity, note (human-readable)
Verdict   probability (calibrated 0–1), label, reasoning, model_versions
```

This gives you:
- Layers are independent and each can be tested alone.
- The LLM input is just "render this Moment as text".
- The web UI renders the same object.
- CS2CD data and your own demos flow through the same pipeline.

## 3. Design review: two changes I recommend

### 3.1 Layer 2: demo files are not video

Facts:
- A `.dem` has exact positions, view angles, health, weapons, etc. for all 10 players on every
  tick. It contains no pixels.
- Running YOLO on a demo means rendering it through the game first. That needs an installed
  CS2 plus CS Demo Manager, and playback runs in about **real time**. The
  `cs2-demo-to-dataset` project measured about **30 min per player POV per match** (~5.5 h for
  all 10). Parsing the same demo takes about **14 s**.
  CS Demo Manager records CS2 on Linux too, through the game's own `startmovie`
  (raw TGA frames → FFmpeg). HLAE, the faster recorder, is Windows-only.
- YOLO would then *estimate* where enemies are on screen. You can compute that exactly by
  projecting their known 3D positions through the suspect's camera.

So YOLO as a required stage breaks the "CPU only, modest requirements" goal: every analysis
would need the game installed, a GPU to render, and minutes of playback per moment.

**Recommendation:** give Layer 2 two backends behind one interface.
- `geometry/` (default): FOV projection, plus ray casting against map triangles (awpy
  `VisibilityChecker`), plus smoke volumes from grenade events, plus flash duration from
  `player_blind`. Start even simpler with the game's own `spotted` / `approximate_spotted_by` fields.
- `vision/` (optional): render **only flagged moments** (a few seconds each) with CS
  Demo Manager, run YOLO26n on them, and use the output to
  (a) produce annotated review clips for a human, Overwatch-style, and
  (b) cross-check the geometry backend where it is weakest (volumetric smokes, molotovs, props).

Side quest that teaches a lot: use the demo as ground truth to **auto-label YOLO frames** by
projecting 3D head/body positions into rendered frames. You learn camera math and get free labels.

→ Write this up as `docs/decisions/0001-layer2-backends.md`, whatever you decide.

### 3.2 Layer 4: don't ask the LLM to invent the percentage

- When an LLM writes "87%", it is producing text. It is not measuring a probability. Small
  models' numbers cluster (80, 85, 90, 95) and are not calibrated. They are also weak at
  arithmetic over hundreds of tick values.
- **Recommendation:** separate *scoring* from *explaining*.
  - **Scoring:** the Layer 3 model outputs a probability, which you **calibrate** on held-out data
    (isotonic or Platt). That is your 0–100%.
  - **Explaining:** the LLM receives a compact evidence summary (features vs. legit baselines,
    rule hits, SHAP top contributors, the score). It returns reasoning plus a verdict label in a
    fixed JSON schema.
  - **If you want the LLM to affect the number:** make it answer with a single class token
    (`CHEAT` / `LEGIT`), read that token's log-probability, and calibrate *that*. Then combine it
    with the Layer 3 score (a small logistic-regression stacker) and **measure** whether it helps.

→ `docs/decisions/0002-llm-role-in-scoring.md`

## 4. Layers in detail

### Layer 0: Parse & normalize
- Tools: **demoparser2** (Rust core, Python API) and **awpy** (built on demoparser2; adds map
  data and visibility).
- Output tables:
  - `ticks`: tick, steamid, X/Y/Z, pitch, yaw, velocity, health, is_alive, active_weapon, spotted, …
  - Event tables: `player_death`, `player_hurt`, `weapon_fire`, `smokegrenade_detonate`,
    `player_blind`, `round_start`/`round_end`, …
- Make the normalized format one that **CS2CD also loads into** (CS2CD was parsed with
  demoparser2, so the column names already match). Then a single pipeline serves both sources.
- CS2 demos are 64 tick, and sub-tick means a shot can happen *between* recorded ticks. Your
  angle derivatives are approximations. Keep that in mind when you pick thresholds.

### Layer 1: Blatant rules
| Rule | Signal | Watch out for |
|---|---|---|
| Spinbot / anti-aim | sustained yaw speed above the legit max; impossible pitch (\|pitch\| > 89) | take the max from pro demos (e.g. 99.99th pct), not a guess |
| Snap kill | > X° of angle change within 1–2 ticks right before a headshot kill | legit flicks near the threshold, so measure them |
| Through walls / cross-map | `player_death.penetrated` ≥ 2, long distance, victim never visible in the prior N ticks | needs Layer 2 |
| Speed / teleport | position delta per tick > max movement speed | round start, respawns, deathmatch |
| Rapid fire | time between shots < weapon cycle time | burst weapons, weapon switches |

Every hit becomes Evidence with value, threshold, and a note. A blatant hit can set a high prior,
but it still goes to the judge for the write-up.

### Layer 3: Behavior (where most detection power lives)
Build **engagement windows** around kills, damage, and "first sight" events (the moment an enemy
becomes visible), e.g. 2 s before to 0.5 s after. Candidate features:
- **Reaction time:** enemy becomes visible → crosshair on target, and visible → first shot.
- **Flick shape:** peak angular velocity, time to peak, overshoot, number of corrective
  sub-movements, jerk. Human flicks look roughly bell-shaped with small corrections (Fitts's law);
  aimbots are often too straight, too smooth, or too instant.
- **Tracking:** angular error to the head while the target moves. Humans lag by about their
  reaction time; locks don't.
- **Triggerbot latency:** crosshair enters the hitbox → shot, in ticks, consistently.
- **Wallhack:** crosshair following enemies who are *not* visible, prefire rate, correlation
  between aim and a hidden enemy's movement.
- **Recoil:** how far the spray deviates from the ideal compensation. "Too perfect" is suspicious.

Then **aggregate per player** and compare against a skill-matched legit baseline (z-scores).
One great flick proves nothing. A distribution of them might.

Model progression, measuring each step:
1. Thresholds.
2. **LightGBM** on window features.
3. **Sequence model** (1D-CNN / GRU / small transformer) on raw per-tick windows. AntiCheatPT
   used 256-tick kill-centered windows with 44 features.

Use SHAP for explanations. They become LLM input.

**The label problem:** labels are per player or per match (VAC ban), not per moment, and cheaters
toggle their cheats. That makes this **multiple-instance learning**: a player's "bag" of moments is
positive if *some* moments are cheating. Aggregate moment scores with max or top-k mean.

### Layer 4: Judge
- **Model:** a ~4B instruct model (e.g. Qwen3.5-4B; alternatives: a small Gemma 4, Phi-4-mini).
  QLoRA fine-tune in Unsloth Studio on your 3060, export to GGUF Q4_K_M (~3 GB), and serve on
  CPU with llama.cpp. Constrain the output with a JSON schema/grammar.
- **Training data** doesn't exist, so you build it by **distillation**: evidence summaries from
  labeled data → a larger teacher model writes the reasoning → you review a sample by hand →
  fine-tune.
  - **Leakage trap:** the teacher knows the label. The reasoning may cite *only* evidence that
    appears in the input.
  - Include many **high-skill legit** examples (pro demos) so the model learns that skill ≠ cheating.
- **Budget:** CPU generation on a 4B Q4 model is roughly in the ~10 tokens/s range (measure it).
  Judge only the top few moments per player, not every kill.

### Backend & web
- **FastAPI:** upload → job → worker process runs the pipeline → results in SQLite plus parquet.
  Report progress by polling (simple) or SSE.
- Parsing and models are CPU-bound, so **never run them inside the async event loop.** Use a
  process pool or a worker process. Add a real queue (arq/RQ + Redis) only when you need one.
- **Web pages:** upload; jobs; match report (players ranked by score); moment detail (2D radar
  replay, aim-angle chart, evidence table, verdict + reasoning).
- **Colab:** for training only, not serving.

## 5. Evaluation: how you'll know it works

- **Split by player and by match.** The same player must never appear in both train and test.
  Otherwise you are measuring memorization.
- **Metrics:** ROC-AUC, PR-AUC, and above all **TPR at FPR = 1%** (a false accusation is the
  expensive error).
- **Calibration:** reliability diagram and Brier score. "70%" should be right about 70% of the time.
- **Sanity suites:** pro demos should score near 0, and blatant spinbot matches near 1.
- **Golden set:** ~50 moments you labeled by hand. Rerun it after every change.
- **Baselines:** always compare against "headshot % only" and against AntiCheatPT's published
  numbers (AUC 93.4% on CS2CD).

## 6. Ethics & practicalities

- This is **decision support, not auto-ban.** Overwatch uses human jurors; your tool ranks
  moments and explains them.
- Demos contain SteamIDs and names. Anonymize before sharing anything (CS2CD shows how).
- Rendered footage is Valve's IP. A large CS2 video dataset (RekaAI/CS2-10k) was removed in 2026
  after a takedown notice from Valve. Keep rendered video local.
- Don't generate "cheater data" by running cheats on your own account.
- Ultralytics YOLO is **AGPL-3.0**. If you ever publish the web app with it inside, AGPL applies.
  RF-DETR (Apache-2.0) is an alternative.
- Detectors trained on pixels also power "AI aimbots" that never touch game memory. Those still
  leave behavioral traces in demos, which is exactly what Layer 3 targets.
