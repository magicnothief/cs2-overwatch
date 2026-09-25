"""Assemble a player's case: the evidence the judge model reads.

One code path for every use of the judge — the training set, the baseline and
evaluation runs, and the live pipeline on a real demo. The case is data; the text
the model reads comes from rendering.render_case, which is the contract.
"""

from __future__ import annotations

import polars as pl

from overwatch.layers.l2_perception.geometry.occlusion import has_map
from overwatch.layers.l4_judge.rendering import (
    FEATURE_LABELS,
    MomentSummary,
    PlayerCase,
    TrajectoryPoint,
)
from overwatch.schemas.evidence import Evidence, Severity

#: Per-kill columns from windows.parquet (dataset.match_windows) that describe a
#: kill beyond its measurements: what happened, and what the shooter could know.
KILL_COLUMNS: tuple[str, ...] = (
    "kill_tick",
    "weapon",
    "headshot",
    "distance",
    "penetrated",
    "thrusmoke",
    "map",
    "last_seen_ms",
    "victim_fired_ms_ago",
    "victim_was_running",
    "allies_alive",
    "enemies_alive",
)


def clean_baselines(players: pl.DataFrame) -> dict[str, float]:
    """What a typical clean player shows, for every feature the judge sees."""
    clean = players.filter(pl.col("label") == "clean")
    return {
        key: float(clean[key].median())
        for key in FEATURE_LABELS
        if key in clean.columns and clean[key].median() is not None
    }


#: How often to sample the approach: every 8 ticks is 125 ms, ~13 rows per kill.
TRAJECTORY_STRIDE = 8
#: The approach shown to the model: 1.5 s before the shot, and just enough after
#: it to see the follow-through.
TRAJECTORY_FROM_MS = -1500
TRAJECTORY_TO_MS = 250


def trajectory_for(
    window_ticks: pl.DataFrame, window_uid: str
) -> list[TrajectoryPoint]:
    """Sample one kill's approach, coarsely enough to stay readable."""
    rows = (
        window_ticks.filter(pl.col("window_uid") == window_uid)
        .filter(pl.col("t_ms").is_between(TRAJECTORY_FROM_MS, TRAJECTORY_TO_MS))
        .filter(
            (pl.col("tick_offset") % TRAJECTORY_STRIDE == 0)
            | (pl.col("tick_offset") == 0)
        )
        .sort("tick_offset")
    )
    return [
        TrajectoryPoint(
            ms=int(row["t_ms"]),
            angle=float(row["target_angle"])
            if row["target_angle"] is not None
            else 180.0,
            visible=bool(row["target_visible"]),
            speed=float(row["yaw_speed"]) if row["yaw_speed"] is not None else None,
        )
        for row in rows.iter_rows(named=True)
    ]


def pick_moments(
    windows: pl.DataFrame, match_id: str, player_id: str, limit: int
) -> list[MomentSummary]:
    """The kills most worth a reviewer's time for this player.

    Ranked by time spent aiming at an enemy they could not see, then by how little
    the crosshair had to move once the enemy appeared — the two things that are
    hard to explain by skill.
    """
    rows = windows.filter(
        (pl.col("match_id") == match_id) & (pl.col("player_id") == player_id)
    )
    if rows.is_empty():
        return []

    ranked = rows.sort(
        ["aim_through_wall_share", "angle_at_first_visible"],
        descending=[True, False],
        nulls_last=True,
    ).head(limit)

    return [
        MomentSummary(
            window_uid=row["window_uid"],
            tick=int(row["kill_tick"]),
            weapon=row.get("weapon"),
            headshot=row.get("headshot"),
            reaction_ms=row.get("reaction_ms"),
            angle_at_first_visible=row.get("angle_at_first_visible"),
            wall_aim_share=row.get("aim_through_wall_share"),
            speed_at_kill=row.get("speed_at_kill"),
            corrections=int(row["corrections"])
            if row.get("corrections") is not None
            else None,
            visible_before_kill=row.get("visible_before_kill"),
            distance=row.get("distance"),
            walls_penetrated=row.get("penetrated"),
            through_smoke=row.get("thrusmoke"),
            sight_checked=bool(row.get("map")) and has_map(row["map"]),
            last_seen_ms=row.get("last_seen_ms"),
            victim_fired_ms_ago=row.get("victim_fired_ms_ago"),
            victim_was_running=row.get("victim_was_running"),
            allies_alive=row.get("allies_alive"),
            enemies_alive=row.get("enemies_alive"),
        )
        for row in ranked.iter_rows(named=True)
    ]


#: The hard-limit hits shown per player, most severe first. Three is enough to
#: establish a pattern; more is repetition the CPU pays for in tokens.
RULE_EVIDENCE_PER_CASE = 3
SEVERITY_ORDER = {s.value: n for n, s in enumerate(reversed(list(Severity)))}


def rule_evidence_for(
    rules: pl.DataFrame | None, match_id: str, player_id: str
) -> list[Evidence]:
    """Layer 1's findings for one player: the worst hit of each rule, most severe first.

    A spinbot trips the same rule dozens of times with near-identical numbers.
    Listing three copies tells the model nothing the count does not, so each rule
    appears once, with how often it fired.
    """
    if rules is None or rules.is_empty():
        return []
    hits = rules.filter(
        (pl.col("match_id") == match_id) & (pl.col("player_id") == player_id)
    )
    if hits.is_empty():
        return []
    ranked = (
        hits.with_columns(
            rank=pl.col("severity").replace_strict(SEVERITY_ORDER, default=99),
            episodes=pl.len().over("name"),
        )
        .sort(["rank", "value"], descending=[False, True])
        .unique(subset=["name"], keep="first", maintain_order=True)
    )
    fields = set(Evidence.model_fields)
    evidence = []
    for row in ranked.head(RULE_EVIDENCE_PER_CASE).iter_rows(named=True):
        item = Evidence.model_validate({k: v for k, v in row.items() if k in fields})
        if row["episodes"] > 1:
            item.note += f" — {row['episodes']} such episodes this match"
        evidence.append(item)
    return evidence


#: How many kills a case describes, and how many of those get a full trajectory.
#: Each trajectory costs ~150 tokens, and a CPU generates about ten a second.
MOMENTS_PER_CASE = 3
TRAJECTORIES_PER_CASE = 2


def build_case(
    player: dict,
    windows: pl.DataFrame,
    *,
    baselines: dict[str, float],
    lines: dict[str, tuple[float, float]] | None = None,
    window_ticks: pl.DataFrame | None = None,
    rules: pl.DataFrame | None = None,
    moments_per_case: int = MOMENTS_PER_CASE,
    trajectories_per_case: int = TRAJECTORIES_PER_CASE,
) -> PlayerCase:
    """One player's case, from their row in the player table.

    Args:
        player: a row of build_player_features output, plus `score` (Layer 3's
            per-player suspicion), and optionally `map` and `avg_rank`.
        windows: per-kill features joined with KILL_COLUMNS.
        baselines: what a typical clean player shows (clean_baselines).
        lines: where 95% and 99% of clean players stop (targets.clean_lines).
        window_ticks: per-tick windows, for the trajectories.
        rules: Layer 1 findings as rows (rules.evidence_rows).
    """
    features = [key for key in FEATURE_LABELS if player.get(key) is not None]
    case = PlayerCase(
        match_id=player["match_id"],
        player_id=player["player_id"],
        map_name=player.get("map"),
        rank=player.get("avg_rank"),
        kills=int(player["n_kills"]),
        score=float(player.get("score") or 0.0),
        features={key: float(player[key]) for key in features},
        baselines=baselines,
        clean_lines={k: tuple(v) for k, v in (lines or {}).items()},
        moments=pick_moments(
            windows, player["match_id"], player["player_id"], moments_per_case
        ),
        rule_evidence=rule_evidence_for(rules, player["match_id"], player["player_id"]),
    )
    if window_ticks is not None:
        for moment in case.moments[:trajectories_per_case]:
            moment.trajectory = trajectory_for(window_ticks, moment.window_uid)
    return case


__all__ = [
    "KILL_COLUMNS",
    "MOMENTS_PER_CASE",
    "TRAJECTORIES_PER_CASE",
    "build_case",
    "clean_baselines",
    "pick_moments",
    "rule_evidence_for",
    "trajectory_for",
]
