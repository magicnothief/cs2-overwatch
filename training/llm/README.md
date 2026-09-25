# training/llm — the judge model

## What is here

| file | what it does |
|---|---|
| `build_cases.py` | detector output → player cases (the evidence text) |
| `build_training_set.py` | cases + ban labels → chat JSONL targets for Unsloth |
| `baseline_judge.py` | what a stock model does, before any fine-tuning |
| `evaluate_judge.py` | score any GGUF on held-out cases |

The evidence text comes from `overwatch.layers.l4_judge.rendering`, which is used
**unchanged at training and at inference**. Never build training text another way:
a model trained on one format and served another degrades quietly, and the reports
stay plausible while being wrong.

## The data

```bash
uv run python training/llm/build_cases.py --per-label 500 --hard-negatives 200
uv run python training/llm/build_training_set.py
```

Produces `data/processed/judge_training/{train,val}.jsonl` (965 / 208), split by
match so no player appears in both, plus `unsloth/{train,val}.jsonl`: the same
conversations with nothing else in them, which is what to upload. The full files
also carry the ban label for evaluation, and that must never reach the model.

Targets come from your annotations where there are any, and are generated
otherwise (`overwatch.layers.l4_judge.targets`). Generated targets depend on the
**evidence text alone**: the ban label only scores the judge. Four behaviours
are taught deliberately:

- **Say what this match shows.** The first training set let the ban label in
  (banned players with no visible evidence got "unclear", clean ones "clean"),
  and the fine-tune learned to guess the label from the behaviour-score line
  instead of reading the measurements. Now no evidence means "clean", whatever
  the account's ban says; `build_target` cannot even be given a label.
- **"Unclear" is a real answer.** Evidence past what 95-99% of clean players show,
  but not enough to decide, gets "unclear", with a probability that rises with it.
- **Heavy snipers are not condemned on visibility alone** (ADR 0008).
- **Every reason quotes a measurement** exactly as the evidence shows it.

Across the cases this convicts ~34% of banned players and calls ~3% of clean
players cheaters (11% of the case set, which includes the 200 clean players the
detector suspects most).

## Fine-tuning in Unsloth Studio

Open Studio (http://127.0.0.1:8888 once the app is running) and set it up like this:

1. **Dataset**, *Local* tab: upload
   `data/processed/judge_training/unsloth/train.jsonl`, format `chatml` (or leave it
   on `auto`: every line is an OpenAI-style `messages` list of system, user and
   assistant). `train.parquet` holds the same conversations if an importer prefers
   Parquet. If Studio offers an evaluation split, give it `unsloth/val.jsonl`.
2. **Model**: `unsloth/Qwen3.5-4B`, Model Type *Text*, Training Method **LoRA**.
3. **Hyperparameters**, from the table below.
4. **Start Training**, then **Export** as GGUF `Q4_K_M` and copy the `.gguf` into
   `models/llm/`.

| setting | value | why |
|---|---|---|
| model | `unsloth/Qwen3.5-4B` | the same model as the stock baseline, so the comparison is fair (not `-Base`, the untuned one) |
| training method | LoRA (bf16) | Unsloth: QLoRA is "not recommended … on the Qwen3.5 models … due to higher than normal quantization differences"; they list ~10 GB for 4B LoRA, which fits the 3060 |
| context length | 2048 | examples are 1,442 tokens at the median, 1,697 at most; at 1024 every answer would be cut off |
| train on completions | **on** | the answer is ~4% of each example; otherwise the loss is spent on copying evidence text |
| epochs | 2 | the targets are templated; more epochs memorise the phrasing |
| batch size / gradient accumulation | 1 / 8 | effective batch 8 (~240 steps); Studio's default batch of 4 is unlikely to fit at 2048 tokens in 12 GB (not measured) |
| learning rate | 2e-4 (Studio's default), linear, a few warmup steps | the usual LoRA range |
| rank / alpha | 16 / 16, all seven target modules | Unsloth's Qwen3.5 example; the task is format and judgement, not new knowledge |
| export | GGUF `Q4_K_M` | the same quantisation as the stock baseline |

Close anything else using the GPU first (the web UI's judge runs on the CPU with
`--gpu-layers 0`). If the exported model returns broken JSON while the training
loss looked fine, the usual cause is a chat-template or end-of-text mismatch
between training and llama.cpp (Unsloth's docs flag the same); the
evaluation's "valid JSON" line shows it straight away.

A full fine-tune of the 4B does not fit in 12 GB (Unsloth: 4x the VRAM of LoRA).

Then compare against the stock model on the same cases:

```bash
uv run python training/llm/evaluate_judge.py --model models/llm/<your>.gguf --limit 207
```

## The number to beat

`compare_judges.py` scores saved evaluations against today's targets, matching
answers to cases by match and player. Per-case answers are in
`data/processed/judge_eval_*.jsonl`. RTX 3060.

**v4 against v3** (2026-09-26), 206 validation cases on the evidence text with
fast kills counted (`judge_training_fastkills/`), prebuilt llama-server on Vulkan:

| measure | v3 | v4 (default) |
|---|---|---|
| valid JSON | 100% | 100% |
| matches the target | 80.6% | 92.2% |
| calls clean players cheaters | 0.9% | 2.8% |
| calls banned players cheaters | 15.3% | 28.6% |
| answers "unclear" | 32% | 26% |
| right when it commits | 77.9% | 77.6% |
| probability ROC-AUC | 0.792 | 0.779 |
| follows the evidence | 0.90 | 0.95 |
| leans on the behaviour score | 0.27 | 0.13 |
| invented numbers | 2 | 0 |
| speed | 1.7 s/case | 2.2 s/case |

What changed in the evidence: "fastest reaction" (one kill decides it; 3.4% of
clean players have a 0 ms kill, from pre-fires at held angles) became the number
of kills within 50 ms of the enemy appearing (two or more: 3.7% of clean
players, 29% of banned ones, 0.3% of pros). v3, never trained on the new line,
garbled it once; v4 reads it. v4's three accused clean-labelled players each
show evidence past the clean 99th percentile (CS2CD's clean labels are
unverified), and it held back on four more whose targets say cheating.

On 341 pro players (`training/check_pro_demos.py`, 35 HLTV matches) v4 accused
none, like v3, but answered "unclear" for 6.7% (v3: 1.5%): 19 of those 23 rest on
one very fast turn on a kill tick, usually an AWP flick. The targets make that
"unclear" and v4 follows its targets closely; the rule is the thing to change
before a v5.

**v3 against v2** (2026-09-25), 207 validation cases, both on today's evidence
text (which shows the clean-player lines):

| measure | v2 | v3 |
|---|---|---|
| valid JSON | 100% | 100% |
| matches the target | 81.6% | 88.9% |
| calls clean players cheaters | 3.7% | 1.8% |
| calls banned players cheaters | 23.5% | 17.3% |
| answers "unclear" | 27% | 36% |
| right when it commits | 76.8% | 78.0% |
| probability ROC-AUC | 0.775 | 0.780 |
| follows the evidence | 0.87 | 0.93 |
| leans on the behaviour score | 0.37 | 0.31 |
| invented numbers | 1 | 0 |
| speed | 2.7 s/case | 2.8 s/case |

The targets themselves rank players at AUC 0.742, the behaviour score alone at
0.884; a judge that reads the evidence faithfully sits near the first. v3's
remaining lean on the score is almost all inside the clean band: where the target
is "clean, 10%", it nudges the number up for high-score players without changing
the verdict (r=0.49 there, 0.11 on unclear targets, -0.11 on cheating ones).

After the dataset was rebuilt on 23 map meshes (cs_office and cs_italy ray cast;
every clean-player line recomputed), v3 on the rebuilt 206 validation cases:
87.4% target match, 1.9% of clean players called cheaters, follows the evidence
at 0.92, leans on the score at 0.31, no invented numbers. Unchanged within noise,
so it was not retrained.

**Earlier, stock and v1** (2026-09-24, the same cases on the evidence text of the
day, before the lines were shown):

| measure | stock | v1 | v2 |
|---|---|---|---|
| matches the target | 18.5% | 69.6% | 84.1% |
| calls clean players cheaters | 67% | 12.8% | 3.7% |
| calls banned players cheaters | 93% | 67% | 30% |
| right when it commits | 55.8% | 81.1% | 79.2% |
| follows the evidence | 0.42 | 0.78 | 0.89 |
| leans on the behaviour score | 0.20 | 0.56 | 0.40 |
| invented numbers | 39 | 0 | 0 |
| speed | 7.7 s/case | 3.6 s/case | 2.7 s/case |

- **v1** (`Qwen3.5-4B.Q4_K_M.gguf`) trained on targets that depended on the ban
  label, and learned to copy the behaviour score where evidence was thin.
- **v2** (`Qwen3.5-4B_STEP_122.Q4_K_M.gguf`) trained on evidence-only targets.
- **v3** (`V3/Full/Qwen3.5-4B.Q4_K_M.gguf`) trained on the same targets with the
  95th/99th-percentile lines shown in the text, so every target can be read off it.
- **v4** (`V4/Qwen3.5-4B.Q4_K_M.gguf`, default since 2026-09-26) trained on the
  fast-kill evidence, same settings as v3, in Unsloth Studio on Windows.

Through llama-cpp-python, on an RTX 3060 with every layer offloaded: **8 s per case**. The
JSON grammar, not the GPU, is now the limit: it is enforced on the CPU token by
token, and generation runs at 23 tok/s with it against 69 tok/s without. It stays
on, because without it the stock model produced a valid verdict only 16 times in 30.
A fine-tuned model may not need it; measure before removing it.

## Where the judge runs

The judge runs in llama.cpp's own prebuilt `llama-server`, which the app downloads
the first time it needs it (`src/overwatch/layers/l4_judge/server.py`), pinned to
one release and checked against GitHub's SHA-256 for each file. Nothing is
compiled. A GPU build is used only when it sees a GPU with room for the whole
model, so a card busy with a fine-tuning run is left alone. Builds are tried in
this order (chosen 2026-09-25: Vulkan's verdicts matched CUDA's, for a twentieth
of the download):

    any GPU      Vulkan (30 MB)
    NVIDIA       CUDA 13 or 12, whichever the driver runs (600 MB); first
                 instead when "This computer" is set to CUDA
    always last  CPU

`--gpu-layers 0` on the web app or the CLI keeps the judge on the CPU build, which
cannot see any GPU. The report records where it ran (`judge_engine`).

Measured 2026-09-25 on the same 60 held-out cases, against the answers the old
llama-cpp-python CUDA build gave (2.7 s per case):

| build | same verdict | same probability | speed | download |
|---|---|---|---|---|
| CUDA 13, RTX 3060 | 59/60 | 59/60 | 1.6 s/case | ~600 MB |
| Vulkan, RTX 3060 | 59/60 | 58/60 | 2.1 s/case | ~30 MB |
| CPU, Ryzen 5 5500 | 7/8 | 6/8 | 32 s/case | ~30 MB |

The wording of the reasons differs more often (GPU kernels round differently, and
greedy decoding follows the first different token); the verdicts do not.

The prompt is rendered by `server.render_prompt`, byte-for-byte what the GGUF's
chat template gives (a test checks it against llama-cpp-python's rendering). Note
that it ends in an open `<think>` block: the model answers inside it, because the
grammar allows nothing but the JSON. Every judge so far was evaluated that way.

## Annotating: the human reasoning the judge learns from

```bash
uv run python -m overwatch.annotation     # then open http://127.0.0.1:8765
```

One case at a time, exactly as the model reads it, with a chart of each kill's
approach. You set a probability (the verdict word follows from it: under 25 clean,
over 65 cheating), name the cheat if accusing, and write up to three reasons and two
caveats. Click-to-insert phrases cover every measurement outside the clean range;
edit them freely. A reason quoting a number that is not in the evidence gets a
warning, because the model must never learn to invent figures.

- **Blind until saved.** The ban label and the generated target appear only after
  you commit, so the gold set measures agreement, not anchoring.
- **Validation cases come first**, so the first ~200 form a human-judged test set
  for the fine-tuned model. The queue interleaves banned/clean and
  convicted/unconvicted cases so the hard ones are not all at the end.
- **Saved as you go** to `data/annotations/gold.jsonl` (append-only; the latest
  save per case wins, so changing your mind is safe). Stop any time.
- **Tied to the evidence text.** Each annotation stores a hash of what you read. If
  the rendering changes later, stale annotations are reported and not trained on.

`build_training_set.py` then uses your verdict wherever there is one and reports
how often the generated targets agree with you: the honest measure of the other
~1,000.
