"""The whole detector on one demo: parse, four layers, one report.

    parse     demoparser2 -> canonical ticks and events
    layer 1   hard limits: spins, snaps, impossible pitch
    layer 2   kill windows, line of sight, what the shooter could have known
    layer 3   the CNN's score per kill and per player, against clean players
    layer 4   the judge reads flagged players' evidence and writes a verdict

Every step calls the same code the training data went through (match_windows,
build_window_features, build_player_features, build_case, render_case), so a
demo is seen exactly as the models were taught to see one.
"""

from __future__ import annotations

import bisect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import polars as pl

from overwatch.dataset import match_windows
from overwatch.layers.l1_blatant.rules import evidence_rows, run_rules
from overwatch.layers.l2_perception.awareness import describe
from overwatch.layers.l2_perception.geometry.occlusion import has_map
from overwatch.layers.l3_behavior.features import build_window_features
from overwatch.layers.l3_behavior.player_features import build_player_features
from overwatch.layers.l4_judge.cases import (
    KILL_COLUMNS,
    build_case,
    pick_moments,
    trajectory_for,
)
from overwatch.layers.l4_judge.judge import Judge
from overwatch.layers.l4_judge.rendering import render_case
from overwatch.maps import prepare_map
from overwatch.parsing import ParsedMatch
from overwatch.parsing.demo import parse_demo
from overwatch.pipeline.report import (
    KillReport,
    MatchReport,
    PlayerReport,
    RadarPoint,
    RoundSpan,
    goto,
)
from overwatch.pipeline.scorer import Scorer
from overwatch.schemas.evidence import Evidence, Severity

#: A player is flagged when their score beats this share of clean players. In
#: cross-validation that line catches ~85% of banned players and flags ~1 clean
#: player in 10 — for the judge to look at, which it is trained to clear.
FLAG_PERCENTILE = 0.90
#: Layer 1 findings at these levels flag a player on their own.
FLAGGING_SEVERITIES = (Severity.STRONG, Severity.IMPOSSIBLE)
#: Kills listed per player in the report.
TOP_KILLS = 3
#: What every model was trained on: 5v5 competitive.
MATCH_PLAYERS = 10

JudgePolicy = Literal["flagged", "all", "none"]
#: Called as each stage starts: (stage key, a sentence for a person).
Progress = Callable[[str, str], None]


def _quiet(stage: str, message: str) -> None:
    """The default progress callback: say nothing."""


def player_names(ticks: pl.DataFrame) -> dict[str, str | None]:
    """Each player's last known name (names can change mid-match)."""
    named = (
        ticks.select("player_id", "tick", "player_name")
        .drop_nulls("player_name")
        .sort("tick")
        .group_by("player_id")
        .agg(pl.col("player_name").last())
    )
    names: dict[str, str | None] = dict.fromkeys(ticks["player_id"].unique().to_list())
    names.update(dict(named.iter_rows()))
    return names


def judge_aliases(player_ids: list[str]) -> dict[str, str]:
    """The judge was trained on anonymised players ("Player_3"), so it sees those."""
    return {pid: f"Player_{n}" for n, pid in enumerate(sorted(player_ids))}


def analyze_demo(
    path: str | Path,
    *,
    scorer: Scorer,
    judge: Judge | None = None,
    judge_policy: JudgePolicy = "flagged",
    progress: Progress | None = None,
    cs2: str | Path | None = None,
    prepare: bool = True,
) -> MatchReport:
    """Parse a .dem file and run every layer on it.

    The first demo on a map also makes that map's mesh and radar, from the CS2
    install (`cs2`, or found through Steam); `prepare=False` skips that.
    """
    path = Path(path)
    tell = progress or _quiet
    tell("parse", "Reading the demo")
    started = time.perf_counter()
    match = parse_demo(path)
    parsed = time.perf_counter() - started
    map_name = match.meta.get("map")
    ready = (
        prepare_map(map_name, cs2=cs2, progress=lambda m: tell("parse", m))
        if prepare and map_name
        else None
    )
    report = analyze_match(
        match,
        path.stem,
        scorer=scorer,
        judge=judge,
        judge_policy=judge_policy,
        progress=progress,
    )
    report.demo = path.name
    report.timings = {"parse": round(parsed, 2), **report.timings}
    if ready is not None and ready.note:
        report.notes.insert(0, ready.note)
    return report


def analyze_match(
    match: ParsedMatch,
    match_id: str,
    *,
    scorer: Scorer,
    judge: Judge | None = None,
    judge_policy: JudgePolicy = "flagged",
    progress: Progress | None = None,
) -> MatchReport:
    """Run layers 1-4 on an already parsed match.

    `progress(stage, message)` is called as each stage starts, with a sentence a
    person can read ("Judge reading Player A, 1 of 2").
    """
    tell = progress or _quiet
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = round(now - clock, 2)
        clock = now

    map_name = match.meta.get("map")
    names = player_names(match.ticks)
    aliases = judge_aliases([p for p in names if p is not None])
    notes: list[str] = []
    if len(names) != MATCH_PLAYERS:
        notes.append(
            f"{len(names)} players, not a 5v5 match: every model was trained on 5v5 "
            "competitive games, so scores here are less reliable (Wingman rounds, "
            "for one, are shorter and closer-range)"
        )

    rounds = round_spans(match.ticks)
    sides = starting_sides(match.ticks)

    # layer 1: hard limits, over the whole match
    tell("layer 1", "Checking hard limits: spins, snaps, impossible angles")
    rule_rows = evidence_rows(run_rules(match.ticks, match.deaths, match_id=match_id))
    rules = pl.DataFrame(rule_rows) if rule_rows else None
    lap("layer 1")

    # layer 2: kill windows with geometry and context
    tell("layer 2", "Working out who could see whom at every kill")
    windows, window_ticks = match_windows(match, match_id)
    lap("layer 2")

    # layer 3: per-kill measurements, per-player features, the CNN's scores
    tell("layer 3", "Scoring every kill against clean players")
    kills = pl.DataFrame()
    features = pl.DataFrame()
    scores = pl.DataFrame(schema={"player_id": pl.String})
    window_scores = pl.DataFrame(schema={"window_uid": pl.String, "score": pl.Float64})
    tracks = pl.DataFrame()
    if not windows.is_empty():
        window_ticks = window_ticks.with_columns(label=pl.lit("unknown"))
        kills = build_window_features(window_ticks).join(
            windows.select(
                "window_uid",
                "victim_id",
                *(c for c in KILL_COLUMNS if c in windows.columns),
            ),
            on="window_uid",
            how="left",
        )
        features = build_player_features(kills, window_ticks, min_kills=1)
        window_scores = scorer.score_windows(window_ticks)
        scores = scorer.score_players(window_scores)
        tracks = radar_tracks(window_ticks, match.ticks)
    else:
        notes.append("no kills with enough surrounding ticks to measure")
    lap("layer 3")

    reports: dict[str, PlayerReport] = {}
    flag_line = scorer.clean_line(FLAG_PERCENTILE)
    for pid in sorted(p for p in names if p is not None):
        row = _row(features, pid) or {}
        scored = _row(scores, pid) or {}
        findings = _findings(rules, pid)
        report = PlayerReport(
            player_id=pid,
            name=names[pid],
            side=sides.get(pid) or None,
            judge_alias=aliases[pid],
            kills=int(row.get("n_kills") or 0),
            score=scored.get("score"),
            clean_percentile=scored.get("clean_percentile"),
            enough_kills=bool(scored.get("enough_kills", False)),
            rule_findings=findings,
        )
        strong = sorted({e.name for e in findings if e.severity in FLAGGING_SEVERITIES})
        if strong:
            report.flag_reasons.append(f"layer 1: {', '.join(strong)}")
        if (
            report.enough_kills
            and report.score is not None
            and report.score >= flag_line
        ):
            report.flag_reasons.append(
                f"behaviour score {report.score:.2f} is higher than "
                f"{report.clean_percentile:.0%} of clean players"
            )
        report.flagged = bool(report.flag_reasons)
        if row:
            ranked = _kill_reports(
                kills, window_ticks, window_scores, tracks, match_id, pid, names, rounds
            )
            report.top_kills = ranked[:TOP_KILLS]
            report.kill_log = sorted(ranked, key=lambda k: k.tick)
        reports[pid] = report

    # layer 4: the judge reads the evidence of the players worth its time
    if judge is not None and judge_policy != "none":
        to_judge = [
            pid
            for pid, report in reports.items()
            if _row(features, pid) is not None
            and (judge_policy == "all" or report.flagged)
        ]
        for n, pid in enumerate(to_judge, start=1):
            report = reports[pid]
            row = _row(features, pid)
            tell(
                "layer 4",
                f"Judge reading {report.name or pid}, {n} of {len(to_judge)}",
            )
            case = build_case(
                {**row, "score": report.score, "map": map_name},
                kills,
                baselines=scorer.reference["judge"]["baselines"],
                lines=scorer.reference["judge"]["lines"],
                window_ticks=window_ticks,
                rules=rules,
            ).model_copy(update={"player_id": report.judge_alias})
            result = judge.judge(case)
            report.judge_evidence = render_case(case)
            report.verdict = result.verdict
            report.judge_error = result.error
    lap("layer 4")

    if map_name and has_map(map_name):
        visibility = f"ray cast against the {map_name} mesh"
    else:
        visibility = "the game's spotting flag"
        notes.append(
            f"no collision mesh for {map_name}: visibility falls back to the spotting "
            "flag, which is biased against snipers (ADR 0007), and 'last seen' is "
            "unavailable"
        )

    ordered = sorted(
        reports.values(),
        key=lambda r: (not r.flagged, -(r.score if r.score is not None else -1)),
    )
    return MatchReport(
        demo=match_id,
        sha256=match.meta.get("sha256"),
        map_name=map_name,
        visibility=visibility,
        kills=match.deaths.height,
        rounds=rounds,
        side_switches=side_switches(match.ticks),
        flag_percentile=FLAG_PERCENTILE,
        players=ordered,
        notes=notes,
        timings=timings,
        scorer_trained_at=scorer.trained_at,
        judge_model=judge.model_path.name if judge is not None else None,
        judge_engine=getattr(judge, "describe", lambda: None)() if judge else None,
    )


def _row(frame: pl.DataFrame, player_id: str) -> dict | None:
    if frame.is_empty() or "player_id" not in frame.columns:
        return None
    hit = frame.filter(pl.col("player_id") == player_id)
    return hit.row(0, named=True) if hit.height else None


def _findings(rules: pl.DataFrame | None, player_id: str) -> list[Evidence]:
    """Every Layer 1 finding for a player, most severe first."""
    if rules is None:
        return []
    order = {s: n for n, s in enumerate(reversed(list(Severity)))}
    fields = set(Evidence.model_fields)
    found = [
        Evidence.model_validate({k: v for k, v in r.items() if k in fields})
        for r in rules.filter(pl.col("player_id") == player_id).iter_rows(named=True)
    ]
    return sorted(found, key=lambda e: (order[e.severity], -e.value))


def round_spans(ticks: pl.DataFrame) -> list[RoundSpan]:
    """Each round's first and last tick. `round` counts rounds already played."""
    spans = (
        ticks.group_by("round")
        .agg(start=pl.col("tick").min(), end=pl.col("tick").max())
        .drop_nulls("round")
        .sort("round")
    )
    return [
        RoundSpan(number=int(r) + 1, start_tick=int(a), end_tick=int(b))
        for r, a, b in spans.iter_rows()
    ]


def side_switches(ticks: pl.DataFrame) -> list[int]:
    """Rounds after which the teams swapped sides (halftime, overtime halves).

    Read from the data rather than assumed: competitive switches after 12, Wingman
    after 8, overtime every 3, and a demo can start mid-match.
    """
    per_round = (
        ticks.drop_nulls(["round", "team"])
        .group_by("player_id", "round")
        .agg(pl.col("team").mode().first())
        .sort("player_id", "round")
        .with_columns(prev=pl.col("team").shift(1).over("player_id"))
        .filter(pl.col("prev").is_not_null() & (pl.col("team") != pl.col("prev")))
        .group_by("round")
        .agg(pl.len())
    )
    players = ticks["player_id"].n_unique()
    # a switch moves everyone; one player changing team is a reconnect, not a half
    return sorted(int(r) for r, n in per_round.iter_rows() if n >= max(2, players // 2))


def starting_sides(ticks: pl.DataFrame) -> dict[str, str]:
    """The side each player started on: team 3 is CT, team 2 is T."""
    first = ticks.drop_nulls("team").sort("tick").group_by("player_id").first()
    return {
        pid: {3: "CT", 2: "T"}.get(team, "")
        for pid, team in first.select("player_id", "team").iter_rows()
    }


#: Radar and trace share one sampling: every 8 ticks (125 ms) of the approach.
PATH_STRIDE = 8


def radar_tracks(window_ticks: pl.DataFrame, ticks: pl.DataFrame) -> pl.DataFrame:
    """Both players' positions through every kill's approach, sampled for the radar."""
    victims = ticks.select(
        victim_id=pl.col("player_id"),
        tick=pl.col("tick"),
        vx=pl.col("x"),
        vy=pl.col("y"),
        vz=pl.col("z"),
        vteam=pl.col("team"),
    )
    return (
        window_ticks.filter(
            pl.col("t_ms").is_between(-1500, 250)
            & (
                (pl.col("tick_offset") % PATH_STRIDE == 0)
                | (pl.col("tick_offset") == 0)
            )
        )
        .select(
            "window_uid",
            "tick",
            "tick_offset",
            "t_ms",
            "x",
            "y",
            "z",
            "yaw",
            "team",
            "target_visible",
            "victim_id",
        )
        .join(victims, on=["victim_id", "tick"], how="left")
        .sort("window_uid", "tick_offset")
    )


def _side(team: int | None) -> str | None:
    return {3: "CT", 2: "T"}.get(team) if team is not None else None


def _path(tracks: pl.DataFrame, window_uid: str) -> tuple[list[RadarPoint], dict]:
    rows = tracks.filter(pl.col("window_uid") == window_uid).to_dicts()
    shot = next((r for r in rows if r["tick_offset"] == 0), rows[-1] if rows else {})
    sides = {
        "attacker_side": _side(shot.get("team")),
        "victim_side": _side(shot.get("vteam")),
    }
    path = [
        RadarPoint(
            ms=int(r["t_ms"]),
            ax=r["x"],
            ay=r["y"],
            az=r["z"],
            yaw=r["yaw"],
            vx=r["vx"],
            vy=r["vy"],
            vz=r["vz"],
            visible=bool(r["target_visible"]),
        )
        for r in rows
        if r["x"] is not None and r["y"] is not None
    ]
    return path, sides


def _kill_reports(
    kills: pl.DataFrame,
    window_ticks: pl.DataFrame,
    window_scores: pl.DataFrame,
    tracks: pl.DataFrame,
    match_id: str,
    player_id: str,
    names: dict[str, str | None],
    rounds: list[RoundSpan],
) -> list[KillReport]:
    """Every kill of one player, most suspicious first (as the judge ranks them)."""
    by_uid = dict(window_scores.select("window_uid", "score").iter_rows())
    victims = dict(kills.select("window_uid", "victim_id").iter_rows())
    starts = [r.start_tick for r in rounds]

    def round_of(tick: int) -> int | None:
        at = bisect.bisect_right(starts, tick) - 1
        return rounds[at].number if at >= 0 else None

    reports = []
    for m in pick_moments(kills, match_id, player_id, limit=kills.height):
        path, sides = _path(tracks, m.window_uid)
        victim = victims.get(m.window_uid)
        reports.append(
            KillReport(
                tick=m.tick,
                round=round_of(m.tick),
                victim=names.get(victim) or victim,
                weapon=m.weapon,
                headshot=m.headshot,
                distance=m.distance,
                walls_penetrated=m.walls_penetrated,
                score=by_uid.get(m.window_uid),
                aim_through_cover=m.wall_aim_share,
                reaction_ms=m.reaction_ms,
                seen_first=m.visible_before_kill,
                context=describe(m.model_dump()) or None,
                trajectory=trajectory_for(window_ticks, m.window_uid),
                path=path,
                **sides,
                demo_command=goto(m.tick),
            )
        )
    return reports


__all__ = ["FLAG_PERCENTILE", "analyze_demo", "analyze_match"]
