# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Simulation layer for PyHexen."""

from pyhexen.sim.attacks import (
    BOTTOMSLOPE,
    MAXRADIUS,
    TOPSLOPE,
    aim_line_attack,
    check_missile_spawn,
    face_mobj,
    line_attack,
    radius_attack,
    seeker_missile,
    spawn_blood_splatter,
    spawn_blood_splatter2,
    spawn_missile,
    spawn_missile_angle,
    spawn_missile_angle_speed,
    spawn_missile_xyz,
    spawn_puff,
    spawn_ripper_blood,
)
from pyhexen.sim.sight import check_sight

__all__ = [
    "BOTTOMSLOPE",
    "MAXRADIUS",
    "TOPSLOPE",
    "aim_line_attack",
    "check_missile_spawn",
    "check_sight",
    "face_mobj",
    "line_attack",
    "radius_attack",
    "seeker_missile",
    "spawn_blood_splatter",
    "spawn_blood_splatter2",
    "spawn_missile",
    "spawn_missile_angle",
    "spawn_missile_angle_speed",
    "spawn_missile_xyz",
    "spawn_puff",
    "spawn_ripper_blood",
]
