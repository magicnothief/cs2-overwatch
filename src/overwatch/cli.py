"""The `overwatch` command.

    overwatch                     start the app and open it in the browser
    overwatch analyze match.dem   review one demo in the terminal
    overwatch setup               download everything now, and show what was found

The first start downloads the models (the detector, 150 KB, and the judge,
2.8 GB); the judge engine and each map follow when first needed. `--no-judge`
skips the judge entirely: reports then have scores and evidence, but no written
verdict.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from overwatch import models, paths
from overwatch.settings import load_settings


def _progress(message: str) -> None:
    """One line that updates in place while a download runs.

    Padded with spaces rather than cleared with an escape code, which the older
    Windows console prints as garbage.
    """
    end = "" if message.endswith("%") else "\n"
    print(f"\r  {message}".ljust(72), end=end, flush=True)


def fetch_models(*, judge: bool) -> bool:
    """Download what is missing; False if the app cannot run (no detector)."""
    wanted = models.missing(judge=judge)
    if not wanted:
        return True
    total = sum(m.size for m in wanted) / 1e9
    print(
        f"First run: downloading the models ({total:.1f} GB, once) into {paths.MODELS}"
    )
    for model in wanted:
        try:
            models.download([model], _progress)
        except Exception as exc:  # noqa: BLE001 - say what failed, carry on if we can
            print(f"\n  could not download {model.remote}: {exc}")
    print()
    if models.missing(judge=False):
        print(
            "The detector model is missing, so there is nothing to run. Try again later."
        )
        return False
    if judge and models.missing(judge=True):
        print("Running without the judge: reports get scores but no written verdicts.")
    return True


def start(args: argparse.Namespace) -> int:
    import uvicorn

    from overwatch.api.app import create_app
    from overwatch.api.jobs import Runner

    if not fetch_models(judge=not args.no_judge):
        return 1
    judge = None if args.no_judge else models.JUDGE.local
    gpu = (
        None
        if args.gpu_layers is None
        else ("auto" if args.gpu_layers == "auto" else int(args.gpu_layers))
    )
    runner = Runner(
        paths.REPORTS,
        scorer_path=models.DETECTOR[0].local,
        judge_model=judge,
        gpu_layers=gpu,
    )
    url = f"http://127.0.0.1:{args.port}"
    print(f"Overwatch review is running at {url}  (Ctrl+C stops it)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(
        create_app(runner),
        host="127.0.0.1",  # local only: there are no accounts
        port=args.port,
        log_level="warning",
    )
    return 0


def setup(args: argparse.Namespace) -> int:
    """Everything the first review would download, now, with a report of what is here."""
    from overwatch.layers.l4_judge import server
    from overwatch.maps import find_cs2_maps, source2

    print(f"Files live in {paths.HOME}")
    if not fetch_models(judge=not args.no_judge):
        return 1
    if not args.no_judge:
        chosen = load_settings()
        build = server.candidates(chosen.gpu_layers, prefer_cuda=chosen.prefer_cuda)[0]
        print(f"Judge engine: llama.cpp {server.RELEASE}, {build.name} build")
        server.install(build, paths.ENGINES, _progress)
        print()
    source2.viewer(paths.TOOLS, _progress)
    print()
    found = find_cs2_maps(load_settings().cs2)
    print(
        f"CS2: {found}"
        if found
        else "CS2: not found. Set its folder on the app's first page, under This computer."
    )
    print("Ready. Start the app with: overwatch")
    return 0


def _wants_judge(argv: list[str]) -> bool:
    """Does `overwatch analyze ...` ask for the judge (anything but --judge none)?"""
    for i, arg in enumerate(argv):
        if arg == "--judge=none" or (
            arg == "--judge" and argv[i + 1 : i + 2] == ["none"]
        ):
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["analyze"]:
        from overwatch.pipeline.__main__ import main as analyze

        asks_help = "-h" in argv or "--help" in argv
        if not asks_help and not fetch_models(judge=_wants_judge(argv)):
            sys.exit(1)
        analyze(argv[1:])
        return
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["start", *argv]  # `overwatch --no-judge` means start without it

    parser = argparse.ArgumentParser(
        prog="overwatch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    for name, what in (
        ("start", "start the app (the default)"),
        ("setup", "download everything now"),
    ):
        command = sub.add_parser(name, help=what)
        command.add_argument(
            "--no-judge", action="store_true", help="skip the 2.8 GB judge model"
        )
    sub.add_parser(
        "analyze", help="review one demo in the terminal (see: overwatch analyze -h)"
    )
    start_args = sub.choices["start"]
    start_args.add_argument("--port", type=int, default=8000)
    start_args.add_argument("--no-browser", action="store_true")
    start_args.add_argument(
        "--gpu-layers",
        default=None,
        help="0 keeps the judge on the CPU; by default it follows the app's setting",
    )
    args = parser.parse_args(argv)
    sys.exit(setup(args) if args.command == "setup" else start(args))


__all__ = ["main"]
