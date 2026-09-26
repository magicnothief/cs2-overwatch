# cs2-overwatch

An offline, CPU-friendly, Overwatch-style cheat review system for Counter-Strike 2.
It takes one or more `.dem` files, runs them through several detection layers, and
produces a ranked list of suspicious moments. Each moment gets a calibrated 0–100%
suspicion score and a written explanation.

This is a **learning project**. You write the code, and every folder's `README.md` tells you
what belongs there, what to learn first, a first exercise, and when that part is "done".

```
.dem ─► [0] parse ─► [1] blatant rules ─► [3] behavior model ─► [4] LLM judge ─► report ─► web UI
                            │                   │
                            └──── [2] perception (who could see whom?) ◄──┘
```

## Install

On **Windows**, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/magicnothief/cs2-overwatch/master/install.ps1 | iex"
```

On **Linux**:

```bash
curl -LsSf https://raw.githubusercontent.com/magicnothief/cs2-overwatch/master/install.sh | sh
```

Then start it with `overwatch`: it opens in your browser. Nothing leaves your PC.

- **No compiler, no admin rights, no Python needed.** The script gets
  [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the app into
  your user folder (about 700 MB).
- **The first start downloads the models** (2.8 GB, once), checked against
  their SHA-256. `overwatch --no-judge` skips the judge: scores and evidence, no
  written verdicts.
- **The judge runs on your graphics card when it has room** (through Vulkan, on
  NVIDIA, AMD and Intel alike), else on the processor. The engine is llama.cpp's
  own prebuilt server, downloaded for your machine the first time it is needed
  (30 MB). "This computer" on the first page keeps it off the graphics card, or,
  on NVIDIA, switches to CUDA (a little faster, a 600 MB download).
- **Maps come from your own CS2.** The first demo on a map reads its collision
  mesh from the game's files (about a minute, once per map) for line of sight,
  and its radar: Valve's own where the game has one, else drawn from the mesh. CS2 is found through Steam; if it is on a drive Steam does not
  list, set the folder under "This computer".
- **Demos:** the first page lists those already on this PC, the ones CS2 saved
  from Watch → Your Matches and those in Downloads, and reviews one where it
  lies. Or drop a file, compressed ones (.gz, .bz2, .zst) as they come.
- `overwatch setup` downloads everything up front and reports what it found.
  `overwatch analyze match.dem` reviews a demo in the terminal.
- **Updating:** the app says on its first page when a new version is out.
  `overwatch update` installs it (on Windows: run the install command again).
  What changed is in [CHANGELOG.md](CHANGELOG.md).
- Uninstall: `uv tool uninstall cs2-overwatch`, then delete the app's folder
  (`%LOCALAPPDATA%\cs2-overwatch` or `~/.local/share/cs2-overwatch`).

## Analyse a demo

```bash
overwatch analyze path/to/match.dem              # judge on flagged players
overwatch analyze match.dem --judge none         # layers 1-3 only, ~10 s
overwatch analyze match.dem --gpu-layers 0       # keep the judge on the CPU
```

Prints a table of every player — Layer 3's score and where it falls among clean
players, Layer 1's findings, the judge's verdict — then, for each flagged player,
why they were flagged, the judge's reasons, and the kills to watch with the
`demo_gototick` command for each. The full report is written to
`data/reports/<demo>.json` for the web interface. A player is flagged by a strong
Layer 1 finding, or a score above 90% of clean players.

Or in the browser: `overwatch` (in this checkout: `uv run overwatch`), then drop
a demo on the page. Every report is laid out on a round timeline, one lane per
player, with each kill's crosshair trace one click away.

## Develop

`uv sync` installs everything, training included (PyTorch and friends are in the
`train` group, which a user's install never pulls in). In a checkout the app
keeps its files in `data/` and `models/` here; installed, in the user folder
above (`src/overwatch/paths.py`). After retraining,
`training/publish_models.py` uploads the models and prints the manifest lines
for `src/overwatch/models.py`.

To release: raise `version` in `pyproject.toml`, add its section to
`CHANGELOG.md`, commit, then `git tag vX.Y.Z && git push --tags`. The release
workflow checks the tag against the version, runs the tests, builds the package
and publishes the release with that section as its notes; installs and
`overwatch update` pick it up from there.

Needs `models/scorer/scorer.onnx` + `scorer.json` and, for the judge, a GGUF in `models/llm/`. To
rebuild everything from CS2CD, in order:

```bash
uv run python training/scorer/build_dataset.py          # windows, geometry, context
uv run python training/scorer/build_sequences.py        # arrays for the CNN
uv run python training/scorer/train_player_model.py     # player features (+ LightGBM)
uv run python training/scorer/cross_validate.py --label meshfix
uv run python training/scorer/train_final.py            # -> models/scorer/scorer.{pt,onnx,json}
uv run python training/rules/collect_rule_evidence.py   # Layer 1 findings for the judge
uv run python training/rules/render_radars.py           # map radars for the web page
uv run python training/llm/build_cases.py --per-label 500 --hard-negatives 200
uv run python training/llm/build_training_set.py        # -> judge_training/unsloth/
```

## Where to start

1. Read `docs/ARCHITECTURE.md`, especially **Design review**. It suggests two changes to the
   original plan (Layer 2 and Layer 4) and explains why.
2. Follow `docs/ROADMAP.md` milestone by milestone.
3. Look up datasets in `docs/DATASETS.md` and learning material in `docs/LEARNING.md`.
4. Record every significant decision in `docs/decisions/` (template included).

## Layout

| Path | What lives there |
|---|---|
| `src/overwatch/schemas/` | Shared data contracts (Match, Moment, Evidence, Verdict) |
| `src/overwatch/collect/` | Demo collector (share codes → `csdm dl-valve`) + delayed ban labeling |
| `src/overwatch/parsing/` | `.dem` / CS2CD → normalized parquet tables |
| `src/overwatch/layers/l1_blatant/` | Hard-limit rules (spinbot, snap kills, speed, rapid fire, …) |
| `src/overwatch/layers/l2_perception/` | Visibility service: `geometry/` (default) and `vision/` (YOLO, optional) |
| `src/overwatch/layers/l3_behavior/` | Engagement features + scorer model → calibrated probability |
| `src/overwatch/layers/l4_judge/` | Small fine-tuned LLM → reasoning + verdict |
| `src/overwatch/pipeline/` | A demo through all four layers into one report (`python -m overwatch.pipeline`) |
| `src/overwatch/annotation/` | The annotation tool: human verdicts for the judge (`python -m overwatch.annotation`) |
| `src/overwatch/api/` | The web interface's backend (`python -m overwatch.api`) |
| `src/overwatch/web/` | The page: plain HTML, CSS and JS, no build step; design notes in its `DESIGN.md` |
| `training/` | YOLO (local), scorer (local/Colab), LLM (Unsloth Studio/Colab) |
| `notebooks/` | Exploration. Code moves to `src/` once it works |
| `data/`, `models/` | Not committed; see `data/README.md` |
| `tests/` | pytest, fixtures, golden set |

## Your machine (checked 2026-09-19)

Ryzen 5 5500 (6c/12t, AVX2), 46 GB RAM, RTX 3060 12 GB, Arch Linux.
- **Inference:** CPU only, which is the project goal.
- **Training:** the 3060 can handle YOLO26n/s and QLoRA on a ~4B LLM. Colab is a fallback.
- **Python:** the system Python is 3.14. ML wheels often lag behind new Python releases,
  so pin a project-local 3.12 or 3.13 (with `uv` or `mise`).

## Licence

The code is **AGPL-3.0-or-later** (`LICENSE`): use it, change it, share it; if
you share a changed version, or let people use one over a network, publish its
source under the same licence. The trained models
([MagicNoThief/cs2-overwatch](https://huggingface.co/MagicNoThief/cs2-overwatch))
are **CC BY-NC 4.0**: free for non-commercial use, with credit.

Built on: the [CS2CD dataset](https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection)
(CC-BY-4.0, Mille Mei Zhen Loo and Gert Lužkov), Qwen3.5-4B (Apache-2.0),
llama.cpp (MIT), Source2Viewer / ValveResourceFormat (MIT), and the Archivo
typeface (SIL OFL 1.1). Maps are read from your own copy of CS2 and never
redistributed.
