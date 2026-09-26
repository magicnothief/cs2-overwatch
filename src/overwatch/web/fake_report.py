"""A synthetic match report, so the review page can be looked at without a demo.

Parsing a demo takes minutes and needs a demo; a design change needs the page on
screen in seconds, and the screenshots that go with it must never carry a real
player's name. This writes one made-up report into the reports folder and prints
the URL to open.

    python -m overwatch.web.fake_report            # into <home>/data/reports
    OVERWATCH_HOME=/tmp/look python -m overwatch.web.fake_report

Every name, Steam ID, score and tick below is invented. The shape is checked
against MatchReport, so a field that moves breaks this loudly.
"""

from __future__ import annotations

import json
import math
import random
import sys

from overwatch import paths
from overwatch.pipeline.report import goto

REPORT_ID = "fake-de_dust2"

#: de_dust2 world coordinates, from data/radars/de_dust2.json: a long A duel.
ATTACKER_FROM = (250.0, 2250.0)
ATTACKER_TO = (700.0, 1850.0)
VICTIM_AT = (1300.0, 500.0)

NAMES = [
    ("swiftcurrent", "CT"),
    ("panelbeater", "CT"),
    ("orbitfall", "CT"),
    ("hushpuppy", "CT"),
    ("lateral_v", "CT"),
    ("quietpart", "T"),
    ("mossbank", "T"),
    ("tinfoilhat", "T"),
    ("ninebar", "T"),
    ("crumplezone", "T"),
]
WEAPONS = ["ak47", "m4a1", "deagle", "awp", "usp_silencer", "galilar", "mp9"]
CONTEXT = [
    "the enemy had been called out on the radio",
    "the enemy had just fired, which is a sound",
    "nothing the player could have known says where the enemy was",
    "a teammate had seen the enemy four seconds earlier",
]


def _approach(rng: random.Random, *, suspicious: bool) -> list[dict]:
    """Both players through the second before the kill, for the radar."""
    path = []
    for step in range(-13, 3):
        ms = step * 125
        t = (step + 13) / 13
        ax = ATTACKER_FROM[0] + t * (ATTACKER_TO[0] - ATTACKER_FROM[0])
        ay = ATTACKER_FROM[1] + t * (ATTACKER_TO[1] - ATTACKER_FROM[1])
        vx = VICTIM_AT[0] - 40 * step
        vy = VICTIM_AT[1] + 18 * step
        # an aimbot's crosshair sits on the head the whole way in; a human's
        # swings onto it in the last few frames
        to_victim = math.degrees(math.atan2(vy - ay, vx - ax))
        drift = 2.0 if suspicious else max(0.0, 55.0 * (1 - t) ** 2)
        path.append(
            {
                "ms": ms,
                "ax": ax,
                "ay": ay,
                "az": -120.0,
                "yaw": to_victim + drift * (1 if step % 2 else -1),
                "vx": vx,
                "vy": vy,
                "vz": -120.0,
                "vyaw": to_victim + 180 + (8 if suspicious else 70),
                "visible": step > (-3 if suspicious else -8),
            }
        )
    return path


def _trajectory(rng: random.Random, *, suspicious: bool) -> list[dict]:
    """Crosshair-to-head angle, every 125 ms, the way the judge reads it."""
    out = []
    for step in range(-13, 3):
        t = min(1.0, (step + 13) / 13)
        angle = 1.2 + 1.5 * abs(math.sin(step)) if suspicious else 46.0 * (1 - t) ** 1.7
        out.append(
            {
                "ms": step * 125,
                "angle": round(angle, 2),
                "visible": step > (-3 if suspicious else -8),
                "speed": round(rng.uniform(0, 240), 1),
            }
        )
    return out


def _kill(
    rng: random.Random, rnd: int, tick: int, *, shooter: str, suspicious: bool
) -> dict:
    hs = suspicious or rng.random() < 0.45
    kill = {
        "tick": tick,
        "round": rnd,
        "victim": rng.choice([n for n, _ in NAMES if n != shooter]),
        "weapon": "deagle" if suspicious else rng.choice(WEAPONS),
        "headshot": hs,
        "distance": round(rng.uniform(4, 38), 1),
        "walls_penetrated": 1 if rng.random() < 0.08 else 0,
        "score": round(
            rng.uniform(0.55, 0.94) if suspicious else rng.uniform(0.02, 0.5), 3
        ),
        "aim_through_cover": round(
            rng.uniform(0.6, 1.0) if suspicious else rng.uniform(0, 0.25), 3
        ),
        "reaction_ms": round(
            rng.uniform(80, 170) if suspicious else rng.uniform(260, 900), 1
        ),
        "shots_ms": sorted(rng.sample(range(-400, 40), rng.randint(1, 4))),
        "arrival_delay_ms": 0.0
        if suspicious and rng.random() < 0.5
        else round(rng.uniform(30, 420), 1),
        "seen_first": not (suspicious and rng.random() < 0.4),
        "context": rng.choice(CONTEXT),
        "trajectory": _trajectory(rng, suspicious=suspicious),
        "attacker_side": "CT" if rnd <= 12 else "T",
        "victim_side": "T" if rnd <= 12 else "CT",
        "path": _approach(rng, suspicious=suspicious),
        "demo_command": goto(tick),
    }
    return kill


def build(*, clean: bool = False) -> dict:
    """`clean` is the ordinary result — nobody flagged, no judge — which is most
    runs and therefore the state the page is judged on."""
    rng = random.Random(4)
    rounds = [
        {
            "number": n,
            "start_tick": 8000 + (n - 1) * 9000,
            "end_tick": 8000 + n * 9000 - 400,
        }
        for n in range(1, 19)
    ]
    players = []
    for i, (name, side) in enumerate(NAMES):
        suspicious = i == 7 and not clean  # the one the page has to be right about
        borderline = i == 2
        kills = []
        for rnd in sorted(rng.sample(range(1, 19), rng.randint(6, 12))):
            span = rounds[rnd - 1]
            tick = rng.randint(span["start_tick"] + 600, span["end_tick"] - 600)
            kills.append(
                _kill(
                    rng,
                    rnd,
                    tick,
                    shooter=name,
                    suspicious=suspicious and rng.random() < 0.6,
                )
            )
        kills.sort(key=lambda k: k["tick"])
        top = sorted(kills, key=lambda k: -(k["score"] or 0))[:5]
        score = round(sum(k["score"] or 0 for k in kills) / len(kills), 3)
        player = {
            "player_id": f"7656119800000{i:04d}",
            "name": name,
            "side": side,
            "judge_alias": f"Player_{i + 1}",
            "kills": len(kills),
            "score": score,
            "clean_percentile": round(min(0.999, score * 1.35), 3),
            "enough_kills": len(kills) >= 7,
            "flagged": suspicious,
            "flag_reasons": (
                [
                    "mean kill score above 90% of clean players",
                    "three kills aimed through cover for over 80% of the last half second",
                ]
                if suspicious
                else []
            ),
            "rule_findings": (
                [
                    {
                        "layer": "l3",
                        "name": "aim_through_cover",
                        "value": 0.91,
                        "unit": "share",
                        "threshold": 0.7,
                        "baseline": 0.14,
                        "severity": "strong",
                        "tick": top[0]["tick"],
                        "note": "the crosshair tracked a hidden enemy",
                    }
                ]
                if suspicious
                else []
            ),
            "top_kills": top,
            "kill_log": kills,
        }
        if suspicious:
            player["verdict"] = {
                "verdict": "cheating",
                "probability": 92,
                "cheat_type": "aimbot",
                "reasons": [
                    "the crosshair held the head through a wall for most of the approach",
                    "the opening shot landed on the tick the crosshair arrived, four times",
                    "reaction times cluster under 150 ms with no spread a human shows",
                ],
                "caveats": [
                    "a demo records the server's view, so a 60 ms lag spike can look like a snap",
                    "the player's own settings and hardware are not in this demo",
                ],
            }
            player["judge_evidence"] = (
                "Player_8, 9 kills, mean score 0.71\n"
                "  round 6, deagle headshot, 17 m: crosshair on head 0.94 of last 0.5 s, "
                "enemy never visible, first shot +0 ms\n"
                "  round 11, deagle headshot, 31 m: crosshair on head 0.88, reaction 96 ms\n"
            )
        elif borderline:
            player["flag_reasons"] = []
        players.append(player)

    return {
        "demo": "scrim-2026-09-24-dust2.dem",
        "sha256": "0" * 64,
        "map_name": "de_dust2",
        "visibility": "ray cast against the de_dust2 mesh",
        "kills": sum(p["kills"] for p in players),
        "rounds": rounds,
        "side_switches": [12],
        "flag_percentile": 0.9,
        "players": players,
        "notes": [
            (
                "this report is synthetic, written by overwatch.web.fake_report "
                "for looking at the page"
            ),
        ],
        "timings": {"parse": 41.2, "score": 3.8, "judge": 0.0 if clean else 19.6},
        "scorer_trained_at": "2026-09-01",
        "judge_model": None if clean else "qwen3.5-4b-overwatch",
        "judge_engine": None if clean else "GPU: synthetic",
    }


def main() -> int:
    from overwatch.pipeline.report import MatchReport

    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    for clean in (False, True):
        report = MatchReport.model_validate(build(clean=clean))
        name = f"{REPORT_ID}-clean" if clean else REPORT_ID
        out = paths.REPORTS / f"{name}.json"
        out.write_text(json.dumps(report.model_dump(mode="json"), indent=1))
        print(f"wrote {out}")
        print(f"     http://127.0.0.1:8000/#/report/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
