"""A synthetic match report, so the review page can be looked at without a demo.

Parsing a demo takes minutes and needs a demo; a design change needs the page on
screen in seconds, and the screenshots that go with it must never carry a real
player's name. This writes one made-up report into the reports folder and prints
the URL to open.

    python -m overwatch.web.fake_report            # into <home>/data/reports
    OVERWATCH_HOME=/tmp/look python -m overwatch.web.fake_report

Every name, Steam ID, score and tick below is invented. The shape is checked
against MatchReport, so a field that moves breaks this loudly. Each report also
gets a round replay (pipeline/replay.py's layout) made up to match its kills.
"""

from __future__ import annotations

import functools
import json
import math
import random
import sys

from overwatch import paths
from overwatch.pipeline.replay import write_replay
from overwatch.pipeline.report import goto

REPORT_ID = "fake-de_dust2"

#: de_dust2 world coordinates, from data/radars/de_dust2.json: a long A duel.
ATTACKER_FROM = (250.0, 2250.0)
ATTACKER_TO = (700.0, 1850.0)
VICTIM_AT = (1300.0, 500.0)

#: Coarse routes over de_dust2, a waypoint every 3 s, rounded to 25 units: the
#: shapes of routes in anonymised CS2CD rounds. The timing on them is invented.
T_ROUTES = [
    [
        (-500, -800),
        (-375, -450),
        (-200, -525),
        (350, -225),
        (650, -200),
        (425, -75),
        (-50, 325),
        (-150, 500),
    ],
    [
        (-525, -750),
        (-825, -775),
        (-1375, -675),
        (-1600, 0),
        (-1650, 675),
        (-1900, 1275),
        (-2025, 1475),
        (-1675, 1150),
        (-1250, 1150),
        (-925, 1375),
    ],
    [
        (-750, -825),
        (-1125, -600),
        (-1525, 0),
        (-1675, 700),
        (-1925, 1300),
        (-1875, 1900),
    ],
    [
        (-425, -850),
        (25, -525),
        (-400, -400),
        (-425, 225),
        (-275, 525),
        (-275, 625),
        (-500, 450),
    ],
    [
        (-1150, -800),
        (-475, -650),
        (150, -425),
        (150, 250),
        (-150, 625),
        (-475, 1025),
        (-450, 1350),
        (-350, 1600),
        (-525, 1775),
    ],
]
CT_ROUTES = [
    [
        (350, 2375),
        (875, 2575),
        (1350, 2775),
        (150, 2375),
        (-125, 2175),
        (-850, 2225),
        (-1450, 2025),
        (-1750, 1900),
        (-1600, 1800),
    ],
    [
        (350, 2350),
        (925, 2225),
        (1375, 2550),
        (1225, 2775),
        (600, 2500),
        (425, 1875),
        (75, 1475),
        (-75, 1500),
        (-225, 1475),
    ],
    [
        (175, 2450),
        (675, 2300),
        (500, 2325),
        (375, 1800),
        (350, 1575),
        (325, 1475),
        (25, 1525),
        (-150, 1425),
    ],
    [
        (250, 2475),
        (-50, 2200),
        (-725, 2350),
        (-1325, 2675),
        (-1575, 2825),
        (-1700, 2675),
        (-1800, 2550),
    ],
    [
        (350, 2350),
        (950, 2225),
        (1425, 1825),
        (1375, 1300),
        (1425, 1525),
        (1175, 2150),
        (650, 2325),
    ],
]
#: Buy time at the start of each synthetic round, in ticks.
FREEZE_TICKS = 15 * 64

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
            tick = rng.randint(
                span["start_tick"] + FREEZE_TICKS + 300, span["end_tick"] - 600
            )
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


def _along(route: list[tuple[int, int]], ticks: float) -> tuple[float, float, float]:
    """Where a player walking `route` is after `ticks`, and which way they head."""
    leg = ticks / (3 * 64)
    i = min(int(leg), len(route) - 2)
    f = min(1.0, leg - i)
    (ax, ay), (bx, by) = route[i], route[i + 1]
    return (
        ax + f * (bx - ax),
        ay + f * (by - ay),
        math.degrees(math.atan2(by - ay, bx - ax)),
    )


def _spot(
    i: int,
    tick: int,
    *,
    route: list,
    speed: list[float],
    start: int,
    deaths: dict[int, tuple[int, int, dict]],
) -> tuple[float, float, float]:
    """Player i at `tick`: along their route, pulled into the fight that kills them."""
    x, y, heading = _along(route[i], (tick - start) * speed[i])
    if i in deaths:
        at, killer, _ = deaths[i]
        w = max(0.0, 1 - (at - tick) / 192)
        if w > 0:
            kx, ky, _ = _along(route[killer], (at - start) * speed[killer])
            x, y = x + w * (kx + 450 - x), y + w * (ky + 300 - y)
    return x, y, heading


def replay(report: dict) -> dict:
    """A made-up round replay to go with `report`, in pipeline/replay.py's layout."""
    rng = random.Random(9)
    people = report["players"]
    index = {p["name"]: i for i, p in enumerate(people)}
    rounds = []
    for span in report["rounds"]:
        n = span["number"]
        start = span["start_tick"] + FREEZE_TICKS
        end = span["start_tick"] + 9000
        t0 = start + (-start) % 4
        ticks = range(t0, end, 4)
        second_half = n > report["side_switches"][0]
        side = [("T" if s == "CT" else "CT") if second_half else s for _, s in NAMES]
        # who dies when, from the report's own kills, in order: a victim dies
        # once, and the dead kill nobody
        deaths: dict[int, tuple[int, int, dict]] = {}
        kills = sorted(
            (k["tick"], index[p["name"]], index[k["victim"]], k)
            for p in people
            for k in p["kill_log"]
            if k["round"] == n
        )
        for tick, killer, victim, k in kills:
            if victim in deaths or killer in deaths or side[victim] == side[killer]:
                continue
            deaths[victim] = (tick, killer, k)
        speed = [rng.uniform(0.8, 1.25) for _ in people]
        route = [
            (T_ROUTES if side[i] == "T" else CT_ROUTES)[(i + n) % 5]
            for i in range(len(people))
        ]

        spot = functools.partial(
            _spot, route=route, speed=speed, start=start, deaths=deaths
        )

        blind_one, blind_at = rng.randrange(len(people)), start + rng.randint(640, 3200)
        players = []
        where = {i: [spot(i, tick) for tick in ticks] for i in range(len(people))}
        for i in range(len(people)):
            died = deaths.get(i, (end + 1,))[0]
            xs, ys, zs, yaws, hps, blinds, sees = [], [], [], [], [], [], []
            for j, tick in enumerate(ticks):
                if tick >= died:
                    for field in (xs, ys, zs, yaws, blinds, sees):
                        field.append(None)
                    hps.append(0)
                    continue
                x, y, heading = where[i][j]
                enemies = [
                    e
                    for e in range(len(people))
                    if side[e] != side[i] and deaths.get(e, (end + 1,))[0] > tick
                ]
                near = min(
                    enemies,
                    key=lambda e: math.dist((x, y), where[e][j][:2]),
                    default=None,
                )
                yaw = heading + rng.uniform(-12, 12)
                if near is not None and math.dist((x, y), where[near][j][:2]) < 1100:
                    ex, ey, _ = where[near][j]
                    yaw = math.degrees(math.atan2(ey - y, ex - x)) + rng.uniform(-4, 4)
                bits = 0
                for e in enemies:
                    ex, ey, _ = where[e][j]
                    off = (
                        math.degrees(math.atan2(ey - y, ex - x)) - yaw + 180
                    ) % 360 - 180
                    if math.dist((x, y), (ex, ey)) < 1400 and abs(off) < 53:
                        bits |= 1 << e
                xs.append(round(x))
                ys.append(round(y))
                zs.append(0)
                yaws.append(round((yaw + 180) % 360 - 180))
                hps.append(100 if tick < died - 96 else 41)
                blinds.append(1 if i == blind_one and 0 <= tick - blind_at < 96 else 0)
                sees.append(bits)
            players.append(
                {
                    "i": i,
                    "side": side[i],
                    "x": xs,
                    "y": ys,
                    "z": zs,
                    "yaw": yaws,
                    "hp": hps,
                    "blind": blinds,
                    "sees": sees,
                }
            )
        shots, dead = [], []
        for v, (at, killer, k) in sorted(deaths.items(), key=lambda d: d[1][0]):
            shots += [[at + round(ms * 64 / 1000), killer] for ms in k["shots_ms"]]
            x, y, _ = spot(v, at)
            dead.append([at, v, killer, k["weapon"], k["headshot"], round(x), round(y)])
        carrier = next(i for i in range(len(people)) if side[i] == "T")
        bomb = [
            [
                span["start_tick"] + 200,
                "pickup",
                carrier,
                *map(round, route[carrier][0]),
            ]
        ]
        plant_at = start + 60 * 64
        if rng.random() < 0.5 and deaths.get(carrier, (end + 1,))[0] > plant_at:
            px, py, _ = spot(carrier, plant_at)
            bomb.append([plant_at, "plant", carrier, round(px), round(py)])
        smoke_at = [route[i][3] for i in rng.sample(range(len(people)), 2)]
        rounds.append(
            {
                "number": n,
                "start": start,
                "end": end,
                "t0": t0,
                "players": players,
                "shots": sorted(shots),
                "deaths": dead,
                "smokes": [
                    [start + 1600 + 400 * s, start + 1600 + 400 * s + 18 * 64, x, y]
                    for s, (x, y) in enumerate(smoke_at)
                ],
                "fires": [[start + 2400, start + 2400 + 7 * 64, *route[carrier][2]]],
                "bomb": bomb,
            }
        )
    return {
        "version": 1,
        "map": report["map_name"],
        "tick_rate": 64,
        "sample_ticks": 4,
        "smoke_radius": 144,
        "fire_radius": 120,
        "visibility": True,
        "players": [{"id": p["player_id"], "name": p["name"]} for p in people],
        "rounds": rounds,
    }


def main() -> int:
    from overwatch.pipeline.report import MatchReport

    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    for clean in (False, True):
        built = build(clean=clean)
        built["replay"] = True
        report = MatchReport.model_validate(built)
        name = f"{REPORT_ID}-clean" if clean else REPORT_ID
        out = paths.REPORTS / f"{name}.json"
        out.write_text(json.dumps(report.model_dump(mode="json"), indent=1))
        write_replay(replay(built), paths.REPORTS / f"{name}.replay.json.gz")
        print(f"wrote {out}")
        print(f"     http://127.0.0.1:8000/#/report/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
