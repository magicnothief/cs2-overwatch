"""Analyse a demo from the command line.

Run:  overwatch analyze path/to/match.dem   (or: python -m overwatch.pipeline ...)

Prints a summary and writes the full report as JSON (default:
data/reports/<demo>.json). The judge runs on flagged players when a GGUF model is
present; --judge all reads every player, --judge none skips it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from overwatch import paths
from overwatch.layers.l4_judge.judge import DEFAULT_MODEL, Judge
from overwatch.pipeline.analyze import analyze_demo
from overwatch.pipeline.report import render_text
from overwatch.pipeline.scorer import DEFAULT_SCORER, Scorer
from overwatch.settings import load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="overwatch analyze", description=__doc__)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--judge-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--judge",
        choices=("flagged", "all", "none"),
        default="flagged",
        help="which players the judge reads (it costs ~8 s each on a GPU, ~30 s on CPU)",
    )
    parser.add_argument(
        "--gpu-layers",
        default=None,
        help="0 keeps the judge on the CPU; by default it follows the app's setting",
    )
    parser.add_argument("--out", type=Path, default=None, help="where to write JSON")
    args = parser.parse_args(argv)
    chosen = load_settings()

    judge = None
    if args.judge != "none":
        if args.judge_model.exists():
            gpu = chosen.gpu_layers if args.gpu_layers is None else args.gpu_layers
            judge = Judge(
                args.judge_model,
                gpu_layers=gpu if gpu == "auto" else int(gpu),
                prefer_cuda=chosen.prefer_cuda,
                progress=lambda m: print(f"  {m}", flush=True),
            )
        else:
            print(f"no judge model at {args.judge_model}; running layers 1-3 only")

    report = analyze_demo(
        args.demo,
        scorer=Scorer(args.scorer),
        judge=judge,
        judge_policy=args.judge,
        cs2=chosen.cs2,
        progress=lambda _stage, m: print(f"  {m}", flush=True),
    )
    print(render_text(report))

    out = args.out or paths.REPORTS / f"{args.demo.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    print(f"\nfull report: {out}")


if __name__ == "__main__":
    main()
