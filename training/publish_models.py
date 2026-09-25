"""Publish the detector and the judge to the Hugging Face repo the app downloads from.

Run:  uv run python training/publish_models.py --repo MagicNoThief/cs2-overwatch

Needs a Hugging Face token with write access (`hf auth login`). Uploads, in one
commit:

    detector/scorer.onnx, detector/scorer.json   models/scorer/
    judge/judge-v3.Q4_K_M.gguf                   the judge in use (--judge)
    README.md                                    the model card below

then prints what to paste into src/overwatch/models.py: the commit every
download is pinned to, and each file's SHA-256 and size. Until that is pasted
and committed, installs keep fetching the previous files.

--dry-run writes the upload folder and prints the manifest without uploading.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CARD = """---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3.5-4B
datasets:
  - CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection
tags:
  - counter-strike
  - cs2
  - cheat-detection
  - gguf
  - onnx
---

# CS2 Overwatch review: models

The two models behind [Overwatch review]({code}), an offline reviewer for
Counter-Strike 2 demos that runs on your own PC. The app downloads these files
itself, pinned to one commit and checked against their SHA-256; you do not need
to fetch them by hand.

| file | what it is |
|---|---|
| `detector/scorer.onnx` | a 1D CNN that scores each kill's aim trajectory (136 KB) |
| `detector/scorer.json` | its architecture and the frozen clean-player reference a score is read against |
| `judge/judge-v3.Q4_K_M.gguf` | Qwen3.5-4B fine-tuned (QLoRA) to write a verdict from the evidence, Q4_K_M |

## What they are for

To help a person review a demo: which players and which kills deserve a look,
and why, in terms that can be checked in the demo. Not to ban anyone
automatically. A high score says a player's kills look unlike clean players'
kills in the training data; it is evidence to examine, not proof.

## How they were made

- **Detector:** trained on CS2CD (795 matches, 317 with a VAC-banned player);
  per-player ROC-AUC 0.93 in match-grouped cross-validation. Line of sight comes
  from ray casts against each map's collision mesh, not the game's spotting flag,
  which is biased against snipers.
- **Judge (v3):** fine-tuned on generated verdicts whose targets depend only on
  evidence shown in the text (never on the ban label), with clean players' 95th
  and 99th percentiles printed beside every measurement. On held-out cases it
  matches its targets 87% of the time, accuses 1.9% of clean players, and cited
  no number absent from the evidence in 206 answers.

## Credits and licences

- These models are released under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free to use,
  share and adapt for non-commercial purposes, with credit. The app's code is
  AGPL-3.0-or-later.
- The judge is a fine-tune of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
  by the Qwen team, licensed under Apache-2.0; its licence is included as
  `judge/LICENSE-Qwen3.5-Apache-2.0.txt`.
- Both models were trained on the
  [CS2CD dataset](https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection)
  (CC-BY-4.0) by Mille Mei Zhen Loo and Gert Lužkov.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="OWNER/NAME on Hugging Face")
    parser.add_argument(
        "--judge",
        type=Path,
        default=ROOT / "models" / "llm" / "V3" / "Full" / "Qwen3.5-4B.Q4_K_M.gguf",
    )
    parser.add_argument(
        "--code", default="https://github.com/magicnothief/cs2-overwatch"
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = {
        "detector/scorer.onnx": ROOT / "models" / "scorer" / "scorer.onnx",
        "detector/scorer.json": ROOT / "models" / "scorer" / "scorer.json",
        "judge/judge-v3.Q4_K_M.gguf": args.judge,
    }
    stage = ROOT / "data" / "publish"
    shutil.rmtree(stage, ignore_errors=True)
    for remote, local in files.items():
        (stage / remote).parent.mkdir(parents=True, exist_ok=True)
        (stage / remote).hardlink_to(local.resolve())
    (stage / "README.md").write_text(CARD.format(code=args.code))
    # the Qwen base model's own licence travels with the fine-tune (Apache-2.0, 4a)
    shutil.copy(
        Path(__file__).with_name("LICENSE-Qwen3.5-Apache-2.0.txt"),
        stage / "judge" / "LICENSE-Qwen3.5-Apache-2.0.txt",
    )

    revision = "(dry run: no commit)"
    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.repo,
            folder_path=stage,
            commit_message="Detector and judge v3",
        )
        revision = commit.oid

    print("\nPaste into src/overwatch/models.py:\n")
    print(f'MODELS_REPO = "{args.repo}"')
    print(f'REVISION = "{revision}"')
    for remote, local in files.items():
        print(f"  {remote}: sha256 {sha256(local)}, size {local.stat().st_size:_}")


if __name__ == "__main__":
    main()
