"""Where a target sits relative to where a player is looking.

This is the question the feature model could never ask: not "how fast did the
crosshair move" but "how far was it from the enemy, tick by tick". A human closes
that gap with a flick, a correction and a settle; an aimbot closes it in one step
and holds it at zero.

Conventions (Source engine, which CS2 inherits):
    yaw     0 points along +x and increases toward +y
    pitch   *negative is up* — this catches everyone once
    units   Hammer units; a player is 72 units tall, eyes at about 64
"""

from __future__ import annotations

import polars as pl

#: Eye height above a standing player's origin, in Hammer units.
EYE_HEIGHT: float = 64.0
#: Eye height while crouched.
CROUCH_EYE_HEIGHT: float = 46.0
#: Aim point on a target: the head sits at roughly eye height.
HEAD_HEIGHT: float = EYE_HEIGHT


def view_direction(yaw: pl.Expr, pitch: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Unit vector a player is looking along."""
    yaw_rad, pitch_rad = yaw.radians(), pitch.radians()
    return (
        yaw_rad.cos() * pitch_rad.cos(),
        yaw_rad.sin() * pitch_rad.cos(),
        -pitch_rad.sin(),
    )


def angles_to_target(
    frame: pl.DataFrame,
    *,
    viewer: tuple[str, str, str] = ("x", "y", "z"),
    target: tuple[str, str, str] = ("target_x", "target_y", "target_z"),
    yaw: str = "yaw",
    pitch: str = "pitch",
    eye_height: float = EYE_HEIGHT,
    head_height: float = HEAD_HEIGHT,
    prefix: str = "target",
) -> pl.DataFrame:
    """Add where the target is relative to the crosshair, per row.

    Adds (with `prefix`):
        {prefix}_distance      distance between eye and target head, in units
        {prefix}_yaw_error     signed horizontal angle to the target, degrees
                               (positive = the target is to the left of the crosshair)
        {prefix}_pitch_error   signed vertical angle, degrees (positive = above)
        {prefix}_angle         total angle between crosshair and target, degrees

    `{prefix}_angle` is the one to watch: 0 means the crosshair is exactly on the
    target's head.
    """
    vx, vy, vz = viewer
    tx, ty, tz = target

    dx = pl.col(tx) - pl.col(vx)
    dy = pl.col(ty) - pl.col(vy)
    dz = (pl.col(tz) + head_height) - (pl.col(vz) + eye_height)
    distance = (dx**2 + dy**2 + dz**2).sqrt()

    yaw_to_target = pl.arctan2(dy, dx).degrees()
    pitch_to_target = -(dz / distance).arcsin().degrees()

    look_x, look_y, look_z = view_direction(pl.col(yaw), pl.col(pitch))
    cos_angle = (look_x * dx + look_y * dy + look_z * dz) / distance

    return frame.with_columns(
        **{
            f"{prefix}_distance": distance,
            f"{prefix}_yaw_error": ((yaw_to_target - pl.col(yaw) + 180) % 360) - 180,
            f"{prefix}_pitch_error": pl.col(pitch) - pitch_to_target,
            f"{prefix}_angle": cos_angle.clip(-1.0, 1.0).arccos().degrees(),
        }
    )


def in_field_of_view(
    angle_column: str = "target_angle", fov_degrees: float = 106.0
) -> pl.Expr:
    """Whether the target falls inside the player's screen.

    CS2's default is 90 degrees measured on a 4:3 screen, which works out to about
    106 degrees horizontally on 16:9. Half of that is the limit from the crosshair.
    """
    return pl.col(angle_column) <= fov_degrees / 2
