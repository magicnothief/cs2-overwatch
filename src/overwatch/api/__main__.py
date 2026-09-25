"""Start the web interface.

Run:  uv run python -m overwatch.api   (the `overwatch` command does this too,
after downloading any missing models, and opens the browser)

then open http://127.0.0.1:8000 and drop a demo on the page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from overwatch import paths
from overwatch.api.app import create_app
from overwatch.api.jobs import Runner
from overwatch.layers.l4_judge.judge import DEFAULT_MODEL
from overwatch.pipeline.scorer import DEFAULT_SCORER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--judge-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--gpu-layers",
        default=None,
        help="0 keeps the judge on the CPU; by default it follows the app's setting",
    )
    parser.add_argument("--reports", type=Path, default=paths.REPORTS)
    parser.add_argument("--uploads", type=Path, default=paths.UPLOADS)
    args = parser.parse_args()

    gpu = args.gpu_layers
    if gpu is not None and gpu != "auto":
        gpu = int(gpu)
    runner = Runner(
        args.reports,
        scorer_path=args.scorer,
        judge_model=args.judge_model,
        gpu_layers=gpu,
    )
    print(f"open http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(
        create_app(runner, uploads=args.uploads),
        host="127.0.0.1",  # local only: there are no accounts
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
