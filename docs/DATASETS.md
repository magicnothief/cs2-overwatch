# Datasets

Checked 2026-09-19. Always read a dataset's card and license before building on it. Keep a line
per downloaded file in `data/SOURCES.md` (where it came from, when, and which license).

## A. Cheat-labeled gameplay (Layers 1 & 3)

### CS2CD: Counter-Strike 2 Cheat Detection ★ primary
- **Where:** https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection
- **What:** 795 CS2 matches.
  - **317 with at least one VAC-banned cheater.** These were manually labeled and verified.
  - **478 with no cheater.** These are *not* verified.
- **Format:** per-tick CSV (10 rows per tick) plus an events JSON. It was parsed with
  demoparser2, so the column names match what you'll get from your own demos.
- **Other details:** anonymized; the per-match JSON includes the cheater label, map, and average rank.
  License CC-BY-4.0.
- **Label noise:** the authors measured 97.2% clean players in the no-cheater sample. In matches
  *with* a cheater, the "not cheater" label was only 55.6% precise (trust-factor matchmaking puts
  cheaters together). Treat "not cheater" in those matches as *unknown*, not negative.
- **Companions:**
  - Preprocessed kill-centered windows: https://huggingface.co/datasets/CS2CD/Context_window_256
    (and `_1024`).
  - Baseline model to beat: https://huggingface.co/CS2CD/AntiCheatPT_256, from the paper
    https://arxiv.org/abs/2508.06348 (AUC 93.4%, 89.2% accuracy).
  - How they collected and labeled the data: https://github.com/Pinkvinus/CS2-demo-scraper

### Kaggle: CSGO cheating dataset
- **Where:** https://www.kaggle.com/datasets/emstatsl/csgo-cheating-dataset
- **What:** CS:GO (not CS2). 10,000 legit players and 2,000 cheaters. The numpy array has shape
  `(players, 30 engagements, 192 ticks, 5)`. The 5 features are attacker Δyaw, attacker Δpitch,
  crosshair-to-victim yaw, crosshair-to-victim pitch, and firing.
- **Use for:** learning sequence models fast, since the data is already windowed and clean.
  You could also pretrain on it and then fine-tune on CS2.
- **Caveat:** it's a different game, tick rate, and era of cheats. Don't report final numbers on it.

### XGuardian datasets
- **Paper:** https://arxiv.org/abs/2601.18068 (2026)
- **What:** CS2 data covering ~3,000 matches, 5,486 players, and ~32k elimination windows, using
  only pitch/yaw. The labels were manually re-verified. The explanations use SHAP, which is a
  useful reference for your Layer 4 inputs.
- **Access:** the authors say the code and datasets are public. Find the links in the paper's
  Open Science section.

### jeffdoesmath/cs2-anticheat-raw (use with care)
- **Where:** https://huggingface.co/datasets/jeffdoesmath/cs2-anticheat-raw
- **What:** `gaze_vectors/cheat_*.parquet` plus `all_player_summaries.parquet`. There is **no
  dataset card**, so provenance and labels are unknown. Use it for exploration only, and don't
  trust the labels until you verify them.

### Label your own demos
CS2CD's recipe:
1. Collect match demos.
2. Look up every SteamID with the Steam Web API **GetPlayerBans** endpoint
   (https://partner.steamgames.com/doc/webapi/ISteamUser#GetPlayerBans) some weeks later.
3. Players banned after the match date are cheater candidates. Verify them by watching.

Respect site ToS and rate limits.

## B. Legit baselines (skill ≠ cheating)

### HLTV CS2 pro demos
- https://huggingface.co/datasets/blanchon/cs2_dataset_demo: raw `.dem` files plus a metadata
  parquet (kills, rounds, player stats), which you can query with DuckDB without downloading.
  CC-BY-4.0.
- https://huggingface.co/datasets/aldigobbler/cs2demos: 177 HLTV CS2 matches.
- **Why:** pros are the upper bound of legit skill. Use them for Layer 1 thresholds, as hard
  negatives for Layer 3, and as "impressive but clean" examples for the Layer 4 judge.
- **Caveat:** pros aren't your target population (matchmaking players). LAN pro demos are
  also cleaner than online play.
- **Caveat, measured 2026-09-26:** pros are *not* uniformly cleaner than CS2CD's clean players.
  Their pooled arrival-shot rate is 9.64% against CS2CD clean's 6.7555% (2,500 eligible arrival
  kills over 278 pros, against 21,982 over 2,554 clean players), because they hold pre-aimed
  angles and fire the tick an enemy steps into the crosshair. A line drawn at the CS2CD clean
  95th percentile therefore puts *more* than 5% of pros past it on that measurement, whatever
  estimator sits on top. Check the direction of the shift per measurement before treating pros
  as a false-positive proxy. See `docs/evals/2026-09-26-arrival-redesign.md` section 3.
- **Sample size, measured 2026-09-26:** the 35-match `--per-map 5 --seed 0` pro check yields 341
  players, and a measurement withheld below a minimum kill count reaches far fewer — 117 for one
  candidate. 1 of 117 has a 95% confidence interval of [0.0216%, 4.6701%]. A gate written as
  "at most 1% of pros" needs **299 shown pros with no hits** to be demonstrable at 95%
  confidence, so it needs roughly `--per-map 25` (175 matches) or more. The 1,988-demo listing
  is cached at `data/processed/pro_check/listing.json` and supports it.

### Arrival tables (derived, 2026-09-26)
- **What:** every crosshair arrival at or before a kill's opening shot, with the crosshair's own
  angular step on that tick, so an arrival definition can be changed without reparsing.
  `data/processed/arrival_lab.parquet` (CS2CD, 110,658 rows over 91,128 kill windows) and
  `data/processed/pro_arrival_lab/` (pro sample, one parquet per demo).
- **Built by:** `training/arrival/arrival_lab.py` and `training/arrival/pro_arrival_lab.py`.
  Both self-check: the CS2CD table rebuilds `shots.arrival_features` exactly, and the pro table
  reproduces all 341 pros' `arrival_kills` and `arrival_shots` as the v5 judge run saw them,
  0 mismatches.
- **Why:** gate 3 of `docs/specs/2026-09-26-triggerbot-and-snap-count.md` is a feature-level
  check. It needs no judge and no GPU, and at ~15 s a demo a fresh pro sample is minutes, not
  hours. Regenerate rather than trust a stale copy — neither file is in git.

## C. Vision (Layer 2 `vision/` backend, optional)

### fvossel/csgo-player-detection ★ best starting point
- **Dataset:** https://huggingface.co/datasets/fvossel/csgo-player-detection
  - 5,473 CS2 frames with 10,643 boxes.
  - Classes: `ct_body`, `ct_head`, `t_body`, `t_head`.
  - Images are 640×640 **native-resolution crops**, so head size isn't distorted by resizing.
  - The split is block-wise to avoid leakage.
  - Every box was confirmed by hand.
- **Model:** https://huggingface.co/fvossel/csgo-player-detection. A YOLO26 trained on it (ONNX
  included), which is your baseline to compare against.
- **Code** for capture, rendering, and labeling: https://github.com/fvossel/RealTimeObjectDetectionCSGO
- **License:** "other". The images are derived from Valve assets; the annotations are free to use.
- **Read the card:** its notes on crop geometry and block-wise splits are a good lesson in themselves.

### Tick-aligned POV renders (for the auto-labeling side quest)
- https://huggingface.co/datasets/blanchon/cs2_dataset_render (OpenCS2): 1280×720 POV video of
  HLTV demos, plus per-tick inputs, view angles, and all-player world state. The dataset is
  huge, so download a single match or player (the card shows how).
- https://github.com/vast7777/cs2-demo-to-dataset: a pipeline that renders demos to POV video
  using CS Demo Manager. That pipeline targets **Windows**, and it runs in real time, about 30 min per
  POV per match. (CS Demo Manager itself can also record CS2 on Linux.)
  - Sample output: https://huggingface.co/datasets/Vasy7777/cs2-demo-archive

### Older / smaller
- https://huggingface.co/datasets/perctrix/cs2-yolo (Roboflow export, YOLO format)
- https://huggingface.co/datasets/keremberke/csgo-object-detection (CS:GO)

## D. LLM judge (Layer 4)

No public dataset of "CS2 moment → expert reasoning" exists. You create one
(`training/llm/README.md`):
1. **Evidence:** your own `render_evidence(moment)` applied to CS2CD and pro-demo moments.
2. **Reasoning:** written by a larger teacher model. It must cite only the given evidence
   (watch for label leakage).
3. **Review:** you check at least 100 samples by hand.
4. **Mix:** clear cheats, clear legit, and **high-skill legit** (pros). Aim for plenty of the
   third kind.

## Unavailable / removed
- **RekaAI/CS2-10k:** a large egocentric CS2 video dataset, removed in 2026 after a takedown
  notice from Valve. It's a reminder to keep rendered footage local.
