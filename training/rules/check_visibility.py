"""Compare the game's spotting flag with real line of sight, on de_mirage.

Run:  uv run python training/rules/check_visibility.py --matches 40

Two questions, both answerable because a kill requires line of sight:

1. **Is ray casting right?** At the kill tick it should say "visible" for nearly
   every kill, for cheaters and clean players alike. If it does not, the geometry
   or the eye heights are wrong.
2. **Does it remove the weapon bias?** The spotting flag says clean sniper kills
   are visible half the time and clean rifle kills 79% of the time. True geometry
   should not care which gun someone is holding.

Only de_mirage has a published collision mesh, so this runs on mirage matches.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from overwatch.aim import add_aim_features, kill_windows
from overwatch.dataset import CHEATER_DIR, CLEAN_DIR
from overwatch.layers.l2_perception.geometry import angles_to_target
from overwatch.layers.l2_perception.geometry.occlusion import has_map, line_of_sight
from overwatch.parsing import load_cs2cd_cached

ROOT = Path(__file__).resolve().parents[2]

SNIPERS = ("awp", "ssg08", "scar20", "g3sg1")
RIFLES = ("ak47", "m4a1", "m4a1_silencer", "galilar", "famas", "aug", "sg556")


def weapon_kind() -> pl.Expr:
    return (
        pl.when(pl.col("weapon").is_in(SNIPERS))
        .then(pl.lit("sniper"))
        .when(pl.col("weapon").is_in(RIFLES))
        .then(pl.lit("rifle"))
        .otherwise(pl.lit("other"))
    )


def windows_for(path: Path, map_name: str) -> pl.DataFrame | None:
    """Kill windows for one match, carrying both players' positions."""
    match = load_cs2cd_cached(path)
    if match.meta.get("map") != map_name:
        return None

    cheaters = set(match.meta.get("cheaters", []))
    windows, ticks = kill_windows(add_aim_features(match.ticks), match.deaths)
    if windows.is_empty():
        return None

    victim_state = match.ticks.select(
        victim_id=pl.col("player_id"),
        tick=pl.col("tick"),
        target_x=pl.col("x"),
        target_y=pl.col("y"),
        target_z=pl.col("z"),
        target_spotted_by=pl.col("spotted_by"),
    )
    joined = (
        ticks.join(
            windows.select(
                "window_id",
                "victim_id",
                "weapon",
                *[c for c in ("penetrated", "thrusmoke") if c in windows.columns],
            ),
            on="window_id",
        )
        .join(victim_state, on=["victim_id", "tick"], how="inner")
        .with_columns(
            label=pl.when(pl.col("player_id").is_in(list(cheaters)))
            .then(pl.lit("cheater"))
            .otherwise(pl.lit("clean" if path.parent.name == CLEAN_DIR else "unknown")),
            match_id=pl.lit(path.stem),
            # the game's own answer, for comparison with the geometric one
            flag_visible=pl.col("spotted_by_right")
            .list.contains(pl.col("player_id"))
            .fill_null(value=False)
            if "spotted_by_right" in ticks.columns
            else pl.col("target_spotted_by")
            .list.contains(pl.col("player_id"))
            .fill_null(value=False),
        )
    )
    return angles_to_target(joined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "cs2cd")
    parser.add_argument("--map", default="de_mirage")
    parser.add_argument("--matches", type=int, default=40, help="per folder")
    args = parser.parse_args()

    if not has_map(args.map):
        msg = f"no collision mesh for {args.map} in data/maps"
        raise SystemExit(msg)

    paths = (
        sorted((args.root / CHEATER_DIR).glob("*.parquet"))[:200]
        + sorted((args.root / CLEAN_DIR).glob("*.parquet"))[:200]
    )

    collected, started = [], time.perf_counter()
    for path in paths:
        frame = windows_for(path, args.map)
        if frame is not None:
            collected.append(frame)
            if len({f["match_id"][0] for f in collected}) >= args.matches * 2:
                break
    if not collected:
        raise SystemExit("no matches on this map")

    ticks = pl.concat(collected, how="diagonal_relaxed")
    print(
        f"{ticks['match_id'].n_unique()} {args.map} matches, "
        f"{ticks.height:,} window ticks ({time.perf_counter() - started:.0f}s)",
        flush=True,
    )

    # only the engagement matters here, and it keeps the ray count sane
    engage = ticks.filter(pl.col("tick_offset").is_between(-32, 0))
    eyes = engage.select("x", "y", "z").to_numpy()
    targets = engage.select("target_x", "target_y", "target_z").to_numpy()

    started = time.perf_counter()
    visible = line_of_sight(eyes, targets, args.map)
    elapsed = time.perf_counter() - started
    print(
        f"{len(visible):,} rays in {elapsed:.0f}s ({len(visible) / elapsed:,.0f}/s)\n"
    )

    engage = engage.with_columns(
        ray_visible=pl.Series(visible), kind=weapon_kind()
    ).filter(pl.col("label") != "unknown")

    at_kill = engage.filter(pl.col("tick_offset") == 0)
    if "penetrated" in at_kill.columns:
        # a wallbang or a shot through smoke legitimately has no line of sight,
        # so those kills cannot test the geometry
        clean_shots = at_kill.filter(
            (pl.col("penetrated").fill_null(0) == 0)
            & (~pl.col("thrusmoke").fill_null(value=False))
        )
        print(
            "share visible at the kill tick, EXCLUDING wallbangs and smoke kills\n"
            "(these must have had line of sight, so this is the geometry's accuracy):"
        )
        print(
            clean_shots.group_by("label", "kind")
            .agg(
                ray_cast=pl.col("ray_visible").mean().round(3),
                spotting_flag=pl.col("flag_visible").mean().round(3),
                n=pl.len(),
            )
            .sort(["kind", "label"])
        )
        print()

    print("share 'visible' AT THE KILL TICK (all kills):")
    print(
        at_kill.group_by("label", "kind")
        .agg(
            spotting_flag=pl.col("flag_visible").mean().round(3),
            ray_cast=pl.col("ray_visible").mean().round(3),
            n=pl.len(),
        )
        .sort(["kind", "label"])
    )

    print("\naiming at an enemy they could not see, during the engagement:")
    print(
        engage.group_by("label", "kind")
        .agg(
            wall_aim_ray=(~pl.col("ray_visible") & (pl.col("target_angle") < 5))
            .mean()
            .round(3),
            visible_ray=pl.col("ray_visible").mean().round(3),
            n=pl.len(),
        )
        .sort(["kind", "label"])
    )


if __name__ == "__main__":
    main()
