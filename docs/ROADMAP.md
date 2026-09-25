# Roadmap

Each milestone lists what to **learn**, what to **build**, and when it is **done**. Don't move on
until the "done when" is true. The milestones are ordered so each one produces data or code the
next one needs.

M10 (vision/YOLO) doesn't depend on anything else. Pick it up as a side quest whenever you want
a break from tabular work.

---

## M0: Setup (≈1 day)
- **Learn:** git basics; virtual environments; `uv` (or `mise`); `ruff`; `pytest`.
- **Build:**
  - `git init`.
  - Project-local Python 3.12 venv and a `pyproject.toml` (write it yourself).
  - Add demoparser2, polars/pandas, pyarrow, jupyter.
- **Done when:** `pytest` runs (zero tests is fine) and `import demoparser2` works in a notebook.

## M1: Your first demo (≈1 week)
- **Learn:** demoparser2 API; ticks vs. seconds; pitch/yaw conventions; the angle wrap-around bug.
- **Build** (`notebooks/01_first_demo.ipynb`):
  - Download one HLTV demo (DATASETS.md → blanchon/cs2_dataset_demo).
  - List the players, the kills, and the available events.
  - Plot one player's yaw and pitch for one round.
  - Compute angular speed in °/s. Look closely at 179° → −179°.
- **Done when:** you have a plot of angular speed over the 2 s before one kill, you can explain
  every spike, and you have loaded one CS2CD match and seen that it has the same columns.

## M2: Schemas & parsing module (≈1 week)
- **Learn:** pydantic v2; parquet; writing testable functions.
- **Build:** `schemas/` (Match, Moment, Evidence, Verdict) and `parsing/` (`.dem` and CS2CD → the
  same normalized tables). Add one test on a small fixture.
- **Done when:** `parse(path)` returns the same table shapes for a raw demo and a CS2CD match,
  and a Moment round-trips through JSON.

## M3: Layer 1, blatant rules (≈1 week)
- **Learn:** distributions and percentiles; why thresholds come from data.
- **Build:**
  - Spinbot, snap-kill, and speed rules.
  - Take thresholds from ~20 pro demos.
  - Run everything over the CS2CD with-cheater set.
- **Done when:** false positives on pro demos = 0, you know how many CS2CD cheaters get flagged,
  and you've looked at 10 flags by hand.

## M4: Layer 2, geometric perception (≈2 weeks)
- **Learn:**
  - Vectors; converting pitch/yaw to a direction vector; the angle between two vectors; FOV.
  - Ray–triangle intersection (Möller–Trumbore).
  - awpy's `VisibilityChecker`.
- **Build:** `visible_enemies(match, steamid, tick)`, in steps:
  1. The `spotted` field.
  2. Angular distance from the crosshair to each enemy's head.
  3. Wall occlusion.
  4. Smokes.
  5. Flashes.
- **Done when:** for any kill you can print "enemy became visible at tick T, crosshair reached
  the head at T+k", and it matches what you see when you watch that moment in CS2.

## M5: Layer 3, features & first model (≈2–3 weeks)
- **Learn:** windowing; derivatives of noisy signals; LightGBM; GroupKFold; class imbalance; SHAP.
- **Build:**
  - Engagement windows and 10–20 features.
  - A LightGBM model with player-grouped splits.
  - An evaluation notebook.
- **Done when:** you have a report with ROC-AUC, PR-AUC, TPR@1%FPR, and a reliability diagram,
  and it beats the "headshot % only" baseline.

## M6: Calibration & sequence model (≈2–3 weeks)
- **Learn:** isotonic/Platt calibration; PyTorch basics; 1D-CNN/GRU/transformer on sequences;
  multiple-instance aggregation.
- **Build:**
  - Calibrate the M5 model.
  - Train a sequence model on raw windows (3060 or Colab).
  - Pretrain on the Kaggle CS:GO data if it helps.
  - Compare against LightGBM and AntiCheatPT honestly.
- **Done when:** you have one calibrated scorer chosen with evidence, and ADR 0002 is written.

## M7: Backend MVP (≈1 week)
- **Learn:** FastAPI; async vs. CPU-bound work; process pools; SQLite.
- **Build:**
  - A CLI first: `python -m overwatch analyze x.dem --out report.json`.
  - Then `POST /demos`, `GET /jobs/{id}`, `GET /reports/{id}`.
- **Done when:** you upload a demo with curl and get a report back while the server stays responsive.

## M8: Web UI MVP (≈2 weeks)
- **Learn:** HTMX + Jinja (least JS) *or* Svelte; a chart library; canvas drawing.
- **Build:** upload page, job list, match report, and a moment detail view with a 2D radar replay
  and an aim chart.
- **Done when:** a friend can use it without you explaining anything.

## M9: Layer 4, LLM judge (≈3–4 weeks)
- **Learn:** prompt design; structured output; GGUF/quantization; LoRA/QLoRA; distillation;
  LLM calibration.
- **Build**, in this order:
  1. `render_evidence(moment) -> str`. Read 20 outputs yourself: could *you* judge from them?
  2. A zero-shot baseline with an off-the-shelf 4B GGUF in llama.cpp.
  3. The distillation dataset (`training/llm/`).
  4. A fine-tune in Unsloth Studio.
  5. GGUF export.
  6. Evaluation against the baseline.
- **Done when:**
  - The fine-tuned model beats zero-shot on the golden set.
  - Its output is always valid JSON.
  - Its reasoning cites only the evidence it was given.

## M10: Vision side quest (anytime, ≈2–3 weeks)
- **Learn:** YOLO training, mAP, ONNX export, CPU inference; camera projection.
- **Build:**
  - Train YOLO26n on `fvossel/csgo-player-detection`, locally on the 3060.
  - Benchmark CPU inference with ONNX Runtime.
  - Stretch: auto-label frames from `blanchon/cs2_dataset_render` by projecting demo positions.
- **Done when:** you've written a model card with mAP50-95 and CPU ms/frame, compared against
  the published fvossel model.

## M11: Hardening (ongoing)
- Golden-set regression tests, profiling (where do the seconds go?), caching by demo hash,
  docs, and ADRs for everything you changed your mind about.
