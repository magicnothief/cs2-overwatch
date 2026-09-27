# Judge v4 on master's evidence text, against 0.3.0's published numbers — 2026-09-26

**Superseded, never finished.** Its one completed measure is the 85.9% target
match of v4 on text without the fastest-turn line, which `43247d5` fixed by
rendering that line for v4 again. The full parity check, on the committed 0.4.0
code, is `docs/evals/2026-09-27-judge-v4-for-0.4.0.md`: 92.2%, no invented numbers,
0 of 341 pros accused.

0.4.0 ships v4 (founder's call on MAG-7). 0.3.0's claims were measured on the
evidence text as it rendered on 2026-09-25. `7322d02` and `ab798bf` changed that
text. This is the parity check: same model, same runtime, same 206 held-out rows,
new text.

| measure | 0.3.0 | threshold | this run | verdict |
|---|---|---|---|---|
| target match, 206 held-out | 92% | >= 92% | pending | pending |
| invented numbers | 0 | 0 | pending | pending |
| pros accused, 35-match check | 0 of 341 | 0 | pending | pending |
| pros "unclear" | <= 6.7% | <= 6.7% | pending | pending |

0.3.0's published claims are `CHANGELOG.md:29-32` ("matches its targets 92% of
the time", "invents no numbers", "Checked on 341 players from 35 pro matches:
none accused") and `training/llm/README.md:120` for the 6.7% unclear figure
(23 of 341).

---

## Subject, runtime, pin

**`models/llm/V4/Qwen3.5-4B.Q4_K_M.gguf`** — the quantisation 0.4.0 pins.
sha256 `8ef3824fed200541f456e3ebfc1aaa4d40533797255f4c31b583ccb14d697068`,
recomputed this run, equal to the pin at `src/overwatch/models.py:97`.
`Q8_0` and `BF16-mmproj` were not evaluated.

Runtime, identical on every run below: llama.cpp **b11177**, Vulkan, RTX 3060,
reported by the harness as
`GPU: NVIDIA GeForce RTX 3060 via vulkan (llama.cpp b11177)`.

Repo at `bc9e01e`. `src/overwatch/models.py` `JUDGE_GENERATION = 4`.

## Split, and the leakage check

`data/processed/judge_training_fastkills/` — 972 train rows, 206 val rows. The
same split the v5 eval used, so the two evals are comparable.

- 496 train matches against 111 val matches
- **match overlap: 0**
- **(match, player) pair overlap: 0**
- val label mix: 108 clean, 98 cheater; all 206 rows are CS2CD

Bare `player_id` overlaps on 10 values, because CS2CD anonymises each demo's
players as `Player_1`..`Player_10`. Those are per-demo slots, not identities;
the identity key is (match, player), and that overlap is 0. There is no stable
cross-match player identity in this corpus to check beyond that.

**Eval-set burn.** These 206 rows have now been scored four times: v3, v4 (twice
— the v5 eval's control and this run's re-verification, byte-identical), and v5.
No threshold was tuned against them in any of those runs, and nothing was tuned
in this one: it re-scores a frozen model.

## The evidence text actually differs, and in exactly one way

`training/llm/render_val_for_judge.py` keeps the split row for row, target for
target, and swaps in the text today's code renders for judge 4:

```
.venv/bin/python training/llm/render_val_for_judge.py \
  --split data/processed/judge_training_fastkills/val.jsonl \
  --cases data/processed/judge_cases_v5.jsonl \
  --judge 4 --out data/processed/judge_val_master_v4.jsonl
```

`206 rows / 206 re-rendered / 206 differ from the text stored in the split`.
Every diff is the same single deleted line:

```
-- fastest turn on a kill tick (deg/s): 86.6 (clean 63.4; 95% of clean players are below 284, 99% below 720)
```

v4's fine-tune read **10** measurements. On master it reads **9**.

This is not only the renderer. `snap_max` is absent from `FEATURE_LABELS`
(`src/overwatch/layers/l4_judge/rendering.py:129`) **and** absent from the
features of every case in `data/processed/judge_cases_v5.jsonl`. `FEATURE_SINCE_JUDGE`
gates measurements *newer* than a judge; it cannot restore one deleted from the
label table. So neither the eval nor the app can put `snap_max` in front of v4.

- `snap_max` is still computed: `src/overwatch/layers/l3_behavior/player_features.py:66,150`.
- The pinned reference still has its baseline: `models/scorer/scorer.json`,
  sha256 `ec3497315adedba528c686d55f4c31befe2f87bca2f2dd97b6ee8f7790195bd1`,
  `reference.judge.baselines.snap_max = 63.369140625`,
  `reference.judge.lines.snap_max = [283.5126953125, 719.669921875]`.
- `snap_kills` is the replacement measurement and is correctly held at
  `FEATURE_SINCE_JUDGE = 5`, off v4's prompt. v4 therefore lost turn speed
  outright; it was not substituted.

Because the app renders those same 9 measurements for the pinned judge, the
numbers below are the release's real numbers, hole included.

## Control: v4 on its own training-era text

`data/processed/judge_eval_v4_val206_v4text.jsonl` — 206 rows, **target match
92.2% (190/206), 0 invalid, 0 invented numbers**.

Byte-identical to `data/processed/judge_eval_v4_fastkills.jsonl` from the v5
eval: 0 verdict differences, 0 probability differences, 0 reason-text
differences across all 206 rows. The runtime is deterministic at these settings,
so a difference on master's text can only come from the text.

```
.venv/bin/python training/llm/evaluate_judge.py \
  --model models/llm/V4/Qwen3.5-4B.Q4_K_M.gguf \
  --cases data/processed/judge_training_fastkills/val.jsonl --limit 206 \
  --out data/processed/judge_eval_v4_val206_v4text.jsonl
```

## Pro baseline: the correction stands

On-disk `data/processed/pro_check/` is the **v3** run (35 reports, 341 players
with enough kills, 5 unclear = 1.4663%), not v4. v4's pro reports were never
kept, so 0.3.0's 6.7% unclear survives only as the figure quoted in
`training/llm/README.md:120`. **This run writes v4's pro numbers to disk for the
first time**, at `data/processed/pro_check_v4/`. Neither `pro_check/` (v3) nor
`pro_check_v5/` was touched.

---

## Measures 1 and 2: the 206 held-out cases on master's text

```
.venv/bin/python training/llm/evaluate_judge.py \
  --model models/llm/V4/Qwen3.5-4B.Q4_K_M.gguf \
  --cases data/processed/judge_val_master_v4.jsonl --limit 206 \
  --out data/processed/judge_eval_v4_val206_mastertext.jsonl
```

**Target match 85.9223% (177/206) — FAIL against >= 92%. Invented numbers 0 — PASS.**

| | v4 on its own training-era text | v4 on master's text | delta |
|---|---|---|---|
| target match | 92.2330% (190/206) | **85.9223% (177/206)** | **-6.3107 pts** |
| valid JSON | 100.0% | 100.0% | 0 |
| invented numbers | 0 in 0 of 206 | **0 in 0 of 206** | 0 |
| accuses clean-labelled players | 2.7778% (3/108) | 2.7778% (3/108) | 0 |
| convicts banned players | 28.5714% (28/98) | 25.5102% (25/98) | -3.0612 pts |
| answers "unclear" | 26.2136% (54) | 23.7864% (49) | -2.4272 pts |
| probability ROC-AUC | 0.7789 | 0.7563 | -0.0226 |
| median seconds per case | 2.386 | 2.270 | — |

### Ablation: one deleted measurement, 6.31 points

Everything except the text is held fixed — same GGUF, same runtime, same rows in
the same order, same targets, and the control reproduces byte-for-byte. The only
difference between the two columns is the deleted `snap_max` line. So the 6.31
points are attributable to it and to nothing else. This is the strongest single
piece of evidence in this document.

16 rows change verdict. **14 of the 16 break an answer that was right; 1 fixes
one that was wrong** (the 16th was wrong before and after). Directions:

| flip | count |
|---|---|
| `unclear` -> `clean` | 8 |
| `cheating` -> `unclear` | 4 |
| `unclear` -> `cheating` | 2 |
| `clean` -> `unclear` | 1 |
| `cheating` -> `clean` | 1 |

Mean probability moves -1.854 points. The model is not made reckless, it is made
vaguer, and it loses the cheater side.

### False-positive cost asymmetry

The accusation rate on clean-labelled players does not move: 3 of 108 in both
runs, the same three 0.3.0 measured. The loss is missed cheaters and lost
commitment. That is the cheaper direction to break in — and it is still a gate
failure, because the gate is the number the release note publishes.

### Subgroups

`training/llm/score_eval_subgroups.py`, new this run. The drop is not uniform.

| subgroup | own text | master text | delta |
|---|---|---|---|
| target verdict `clean` (n=122) | 97.5% | 97.5% | 0 |
| target verdict `cheating` (n=42) | 73.8% | 61.9% | -11.9 |
| target verdict `unclear` (n=42) | 95.2% | 76.2% | -19.0 |
| label clean (n=108) | 95.4% | 92.6% | -2.8 |
| label cheater (n=98) | 88.8% | 78.6% | -10.2 |
| <= 10 kills (n=59) | 94.9% | 93.2% | -1.7 |
| 11-20 kills (n=96) | 91.7% | 89.6% | -2.1 |
| > 20 kills (n=51) | 90.2% | 70.6% | **-19.6** |

The high-kill band loses the most, which fits the cause: more kills means more
chance of one extreme turn, so `snap_max` carried the most information there.
Aggregate parity would have hidden that even if the aggregate had held.

## Measures 3 and 4: the pro check

_(running — filled when the 35 matches finish)_

## What was not tested

- Only `Q4_K_M`. `Q8_0` and `BF16-mmproj` were not run.
- Only llama.cpp b11177 on Vulkan. No CUDA, no CPU-only, no other build.
- **No map or rank subgroup on the held-out set.** Every CS2CD case carries
  `map_name: null` and `rank: null`, so the 206 rows cannot be broken down that
  way. The pro check covers 7 maps and is broken down by map below; the held-out
  206 is not.
- No tickrate breakdown: the cases do not carry tickrate.
- Nothing about detection at pro skill level. The pro check is a false-positive
  control only.
- Layers 1-3 unchanged and not re-measured here; this is a judge-text eval.
- The clean lines come from CS2CD matchmaking demos, so they are loose for pro
  play — a distribution-shift limit the pro check inherits and cannot remove.
