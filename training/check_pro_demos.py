"""How often does the app accuse clean players outside CS2CD? Ask pro matches.

Run:  uv run python training/check_pro_demos.py --per-map 5

Pros at HLTV-listed events are the cleanest large population there is, and the
best players in the world: if anything looks like cheating to a model trained
on matchmaking, it is them. So every player in a sample of pro demos
(blanchon/cs2_dataset_demo, CC-BY-4.0) goes through the whole app, the judge
reading all of them, and every flag and accusation counts as a false positive.

What it reports, against what CS2CD's clean players show by construction:

    flagged by score   above the clean 90th percentile: ~10% expected
    Layer 1            strong or impossible findings: ~0% expected
    judge              "cheating" verdicts: 1.9% of clean CS2CD players

Result, 2026-09-25 (35 matches, 5 per map, seed 0; 341 players with enough kills):

    flagged by score   1 (0.3%): 6 kills at 0.48, just over the 0.47 line;
                       the judge read it as clean
    Layer 1            0
    judge              0 accused; 5 unclear (1.5%), each on one extreme kill:
                       "killed 0 ms after an enemy appeared" (pre-firing a
                       held angle, twice), 760 deg/s on the kill tick (AWP
                       flicks, twice), 54% of flicks never corrected
    pros vs clean      median score 0.17 against CS2CD clean 0.26, 90th
                       percentile 0.29 against 0.48: pros look *less* like
                       CS2CD's cheaters than matchmaking players do (they
                       pre-aim, so their kills need small corrections)

So the app does not mistake elite aim for cheating. It says nothing about
catching a cheater at pro level, and the clean lines, drawn from matchmaking,
are loose for pro play.

Demos are streamed (downloaded, analysed, deleted; --keep keeps them), reports
land in data/processed/pro_check/, and a rerun skips demos already done. The
sample is stratified by map and seeded, so it is the same sample every time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from overwatch import paths
from overwatch.layers.l4_judge.judge import DEFAULT_MODEL, Judge
from overwatch.pipeline.analyze import analyze_demo
from overwatch.pipeline.scorer import Scorer
from overwatch.settings import load_settings

DATASET = "blanchon/cs2_dataset_demo"
OUT = paths.DATA / "processed" / "pro_check"
#: "<team>-vs-<team>-m1-<map>.dem": one whole map, not a part of one.
NAME = re.compile(r"-m\d-(?:de_)?([a-z0-9]+)\.dem$")
MAPS = ("ancient", "anubis", "dust2", "inferno", "mirage", "nuke", "overpass")


def sample(per_map: int, seed: int) -> list[str]:
    from huggingface_hub import HfApi

    listing = OUT / "listing.json"  # the dataset's file list, shared by every run
    if listing.exists():
        demos = json.loads(listing.read_text())
    else:
        tree = HfApi().list_repo_tree(DATASET, repo_type="dataset", recursive=True)
        demos = sorted(f.path for f in tree if f.path.endswith(".dem"))
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(demos))
    by_map = defaultdict(list)
    for path in demos:
        found = NAME.search(path)
        if found and found.group(1) in MAPS:
            by_map[found.group(1)].append(path)
    rng = random.Random(seed)
    return [
        p for m in MAPS for p in rng.sample(by_map[m], min(per_map, len(by_map[m])))
    ]


def run(demos: list[str], *, out: Path, keep: bool, gpu_layers: int | str) -> None:
    from huggingface_hub import hf_hub_download

    chosen = load_settings()
    scorer = Scorer()
    judge = Judge(DEFAULT_MODEL, gpu_layers=gpu_layers, prefer_cuda=chosen.prefer_cuda)
    print(f"judge: {judge.describe()}", flush=True)
    raw = paths.DATA / "raw" / "hltv"
    for n, remote in enumerate(demos, 1):
        report_path = out / "reports" / (Path(remote).stem + ".json")
        if report_path.exists():
            continue
        started = time.perf_counter()
        local = Path(
            hf_hub_download(DATASET, remote, repo_type="dataset", local_dir=raw)
        )
        fetched = time.perf_counter() - started
        try:
            report = analyze_demo(
                local, scorer=scorer, judge=judge, judge_policy="all", cs2=chosen.cs2
            )
        except Exception as exc:  # noqa: BLE001 - a broken demo must not end the run
            print(f"[{n}/{len(demos)}] {remote}: failed ({type(exc).__name__}: {exc})")
            continue
        finally:
            if not keep:
                local.unlink(missing_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json())
        accused = [
            p for p in report.players if p.verdict and p.verdict.verdict == "cheating"
        ]
        print(
            f"[{n}/{len(demos)}] {report.map_name} {Path(remote).stem}: "
            f"{sum(p.flagged for p in report.players)} flagged, {len(accused)} accused "
            f"(download {fetched:.0f}s, analysis {time.perf_counter() - started - fetched:.0f}s)",
            flush=True,
        )
    judge.close()


def summarise(out: Path) -> None:
    reports = [
        json.loads(p.read_text()) for p in sorted((out / "reports").glob("*.json"))
    ]
    players = [
        (r, p) for r in reports for p in r["players"] if p.get("enough_kills", True)
    ]
    if not players:
        print("no reports yet")
        return
    n = len(players)

    def share(k: int) -> str:
        return f"{k:3d} of {n} ({k / n:.1%})"

    flagged = sum(p["flagged"] for _, p in players)
    by_score = sum(
        (p.get("clean_percentile") or 0) >= r["flag_percentile"] for r, p in players
    )
    strong = Counter(
        e["name"]
        for _, p in players
        for e in p["rule_findings"]
        if e["severity"] in ("strong", "impossible")
    )
    ruled = sum(
        any(e["severity"] in ("strong", "impossible") for e in p["rule_findings"])
        for _, p in players
    )
    verdicts = Counter(
        (p.get("verdict") or {}).get("verdict", "none") for _, p in players
    )
    print(
        f"\n{len(reports)} pro matches, {n} players with enough kills to score\n"
        f"flagged (any reason):        {share(flagged)}\n"
        f"  score above clean p90:     {share(by_score)}   (CS2CD clean: 10% by construction)\n"
        f"  layer 1 strong/impossible: {share(ruled)}   {dict(strong.most_common(5))}\n"
        f"judge: cheating {share(verdicts['cheating'])}   (CS2CD clean: 1.9%)\n"
        f"       unclear  {share(verdicts['unclear'])}\n"
        f"       clean    {share(verdicts['clean'])}\n"
        f"       no answer {verdicts['none']}"
    )
    per_map = defaultdict(lambda: [0, 0, 0])
    for r, p in players:
        row = per_map[r["map_name"]]
        row[0] += 1
        row[1] += p["flagged"]
        row[2] += (p.get("verdict") or {}).get("verdict") == "cheating"
    print("\nper map (players, flagged, accused):")
    for name, (count, flag, acc) in sorted(per_map.items()):
        print(f"  {name:12s} {count:4d} {flag:4d} {acc:4d}")
    percentiles = sorted(p["clean_percentile"] or 0 for _, p in players)
    print(
        "\npros' place among clean CS2CD players (median, p90): "
        f"{percentiles[n // 2]:.2f}, {percentiles[int(n * 0.9)]:.2f}"
    )
    accused = [
        (r, p)
        for r, p in players
        if (p.get("verdict") or {}).get("verdict") == "cheating"
    ]
    for r, p in accused:
        print(
            f"\naccused: {r['demo']} ({r['map_name']}), score {p['score']:.2f}, "
            f"top {1 - (p['clean_percentile'] or 0):.0%}, {p['verdict']['probability']}%"
        )
        for reason in p["verdict"]["reasons"]:
            print(f"  - {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-map", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", action="store_true", help="keep the downloaded demos")
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--summary", action="store_true", help="only summarise")
    parser.add_argument(
        "--out", type=Path, default=OUT, help="where reports go (one folder per run)"
    )
    args = parser.parse_args()
    if not args.summary:
        gpu = args.gpu_layers if args.gpu_layers == "auto" else int(args.gpu_layers)
        run(
            sample(args.per_map, args.seed),
            out=args.out,
            keep=args.keep,
            gpu_layers=gpu,
        )
    summarise(args.out)


if __name__ == "__main__":
    main()
