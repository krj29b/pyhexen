# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Line activation (cross, use, monster-cross, impact, push, projectile-cross) and use-ray trace.

Port of P_ActivateLine (p_spec.c), P_UseLines, and PTR_UseTraverse (p_map.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import ANGLETOFINESHIFT, fine_cosine, fine_sine
from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType, SfxEnum
from pyhexen.sim.maputl import (
    PT_ADDLINES,
    Intercept,
    line_opening,
    path_traverse,
    point_on_line_side,
)
from pyhexen.sim.sound import start_sound
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.sim.specials.switches import p_change_switch_texture
from pyhexen.world.level import (
    GET_SPAC,
    ML_REPEAT_SPECIAL,
    ML_SECRET,
    SPAC_CROSS,
    SPAC_IMPACT,
    SPAC_MCROSS,
    SPAC_PCROSS,
    SPAC_PUSH,
    SPAC_USE,
    Line,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.mobj import Mobj
    from pyhexen.sim.player import Player

USERANGE: int = 64 * FRACUNIT

__all__ = [
    "SPAC_CROSS",
    "SPAC_IMPACT",
    "SPAC_MCROSS",
    "SPAC_PCROSS",
    "SPAC_PUSH",
    "SPAC_USE",
    "USERANGE",
    "p_activate_line",
    "p_use_lines",
]


def _failed_use_sfx(player_class: int) -> int:
    """Return the failed-use SfxEnum for `player_class`."""
    if player_class == 0:
        return int(SfxEnum.SFX_PLAYER_FIGHTER_FAILED_USE)
    if player_class == 1:
        return int(SfxEnum.SFX_PLAYER_CLERIC_FAILED_USE)
    if player_class == 2:
        return int(SfxEnum.SFX_PLAYER_MAGE_FAILED_USE)
    if player_class == 3:
        return int(SfxEnum.SFX_PIG_ACTIVE1)
    return int(SfxEnum.SFX_NONE)


def p_activate_line(
    sim: GameSim,
    line: Line,
    mo: Mobj | None,
    side: int,
    activation_type: int,
) -> bool:
    """Attempt to activate `line` with `activation_type` (`SPAC_*`).

    Port of P_ActivateLine (p_spec.c).
    """
    if line.special == 0:
        return False

    line_activation = GET_SPAC(line.flags)
    if line_activation != activation_type:
        return False

    if mo is not None:
        is_player = mo.player is not None or mo.type in (
            MobjType.MT_PLAYER_FIGHTER,
            MobjType.MT_PLAYER_CLERIC,
            MobjType.MT_PLAYER_MAGE,
            MobjType.MT_PIGPLAYER,
        )
        is_missile_or_impact = bool(mo.flags & MobjFlag.MF_MISSILE) or bool(
            mo.flags2 & (MobjFlag2.MF2_PCROSS | MobjFlag2.MF2_IMPACT | MobjFlag2.MF2_PUSHWALL)
        )
        if not is_player and not is_missile_or_impact:
            if line_activation != SPAC_MCROSS:
                return False
            if line.flags & ML_SECRET:
                return False

    repeat = bool(line.flags & ML_REPEAT_SPECIAL)
    button_success = execute_line_special(sim, line.special, line.args, line, mo, side=side)

    if not repeat and button_success:
        line.special = 0

    if line_activation in (SPAC_USE, SPAC_IMPACT, SPAC_PUSH) and button_success:
        p_change_switch_texture(sim, line, repeat)

    return button_success


def p_use_lines(sim: GameSim, player: Player) -> bool:
    """Trace `USERANGE` ahead of `player` and activate the first `SPAC_USE` special line.

    Port of P_UseLines and PTR_UseTraverse (p_map.c).
    """
    mo = player.mo
    level = sim.level
    if mo is None or level is None:
        return False

    fine = (mo.angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    x1 = mo.x
    y1 = mo.y
    x2 = x1 + (USERANGE >> FRACBITS) * fine_cosine(fine)
    y2 = y1 + (USERANGE >> FRACBITS) * fine_sine(fine)

    activated = False

    def _ptr_use_traverse(in_: Intercept) -> bool:
        nonlocal activated
        li = in_.line
        assert li is not None

        if not li.special:
            opening = line_opening(li)
            if opening.openrange <= 0:
                start_sound(sim, mo, _failed_use_sfx(player.player_class))
                return False  # Can't use through a closed wall

            pheight = mo.z + (mo.height // 2)
            if opening.opentop < pheight or opening.openbottom > pheight:
                start_sound(sim, mo, _failed_use_sfx(player.player_class))
            return True  # Keep checking behind open portal

        if point_on_line_side(mo.x, mo.y, li) == 1:
            return False  # Don't use back sides

        activated = p_activate_line(sim, li, mo, 0, SPAC_USE)
        return False  # Can't use more than one special line in a row

    path_traverse(level, x1, y1, x2, y2, PT_ADDLINES, _ptr_use_traverse)
    return activated
