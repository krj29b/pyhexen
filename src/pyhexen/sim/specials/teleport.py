# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Teleport and thing line specials (T7.7, T13.1).

Port of p_telept.c and p_things.c.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import (
    ANGLETOFINESHIFT,
    FINEMASK,
    fine_cosine,
    fine_sine,
)
from pyhexen.core.fixed import (
    FRACUNIT,
    fixed_mul,
    to_uint32,
)
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType, SfxEnum
from pyhexen.sim.map import teleport_move
from pyhexen.sim.mobj import spawn_mobj
from pyhexen.sim.player import PW_FLIGHT, PW_SPEED
from pyhexen.sim.sound import start_sound
from pyhexen.sim.transitions import teleport_end_game, teleport_new_map

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.mobj import Mobj

TELEFOGHEIGHT: int = 24 * FRACUNIT


def p_teleport(
    sim: GameSim,
    thing: Mobj,
    x: int,
    y: int,
    angle: int,
    *,
    use_fog: bool = True,
) -> bool:
    """Teleport `thing` to `(x, y)` with facing `angle`.

    Port of P_Teleport (p_telept.c:50).
    """
    oldx = thing.x
    oldy = thing.y
    oldz = thing.z
    above_floor = thing.z - thing.floorz

    if not teleport_move(sim, thing, x, y):
        return False

    if thing.player is not None:
        player = thing.player
        if player.powers[PW_FLIGHT] and above_floor > 0:
            thing.z = thing.floorz + above_floor
            if thing.z + thing.height > thing.ceilingz:
                thing.z = thing.ceilingz - thing.height
            player.viewz = thing.z + player.viewheight
        else:
            thing.z = thing.floorz
            player.viewz = thing.z + player.viewheight
            if use_fog:
                player.lookdir = 0
    elif bool(thing.flags & MobjFlag.MF_MISSILE):
        thing.z = thing.floorz + above_floor
        if thing.z + thing.height > thing.ceilingz:
            thing.z = thing.ceilingz - thing.height
    else:
        thing.z = thing.floorz

    if use_fog:
        fog_delta = 0 if bool(thing.flags & MobjFlag.MF_MISSILE) else TELEFOGHEIGHT
        fog1 = spawn_mobj(sim, oldx, oldy, oldz + fog_delta, MobjType.MT_TFOG)
        start_sound(sim, fog1, int(SfxEnum.SFX_TELEPORT))

        an = ((angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        dest_fog_x = x + 20 * fine_cosine(an)
        dest_fog_y = y + 20 * fine_sine(an)
        fog2 = spawn_mobj(sim, dest_fog_x, dest_fog_y, thing.z + fog_delta, MobjType.MT_TFOG)
        start_sound(sim, fog2, int(SfxEnum.SFX_TELEPORT))

        if thing.player is not None and not thing.player.powers[PW_SPEED]:
            thing.reactiontime = 18

        thing.angle = to_uint32(angle)

    if bool(thing.flags2 & MobjFlag2.MF2_FLOORCLIP):
        thing.floorclip = 10 * FRACUNIT

    if bool(thing.flags & MobjFlag.MF_MISSILE):
        an = ((angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        speed = getattr(thing.info, "speed", 0) if hasattr(thing, "info") else 0
        thing.momx = fixed_mul(speed, fine_cosine(an))
        thing.momy = fixed_mul(speed, fine_sine(an))
    elif use_fog:
        thing.momx = 0
        thing.momy = 0
        thing.momz = 0

    return True


def ev_teleport(
    sim: GameSim,
    tid: int,
    thing: Mobj | None,
    *,
    fog: bool = True,
) -> bool:
    """Teleport `thing` to a mapspot with `tid`.

    Port of EV_Teleport (p_telept.c:152).
    """
    if thing is None or bool(thing.flags2 & MobjFlag2.MF2_NOTELEPORT):
        return False

    targets = sim.find_mobjs_by_tid(tid)
    if not targets:
        return False

    count = len(targets)
    if count == 1:
        dest = targets[0]
    else:
        idx = sim.rng.p_random() % count
        dest = targets[idx]

    return p_teleport(sim, thing, dest.x, dest.y, dest.angle, use_fog=fog)


def special_teleport(
    sim: GameSim,
    special: int,
    args: Sequence[int],
    line: Any,
    mo: Any,
    side: int = 0,
) -> bool:
    """Line special 70 (Teleport) and 71 (Teleport_NoFog)."""
    _ = line
    if side != 0 or mo is None:
        return False
    if getattr(mo, "player", None) is not None and getattr(mo.player, "player_state", 0) == 1:
        return False

    tid = args[0] if len(args) > 0 else 0
    use_fog = special != 71
    return ev_teleport(sim, tid, mo, fog=use_fog)


def special_teleport_new_map(
    sim: GameSim,
    special: int,
    args: Sequence[int],
    line: Any,
    mo: Any,
    side: int = 0,
) -> bool:
    """Line special 74 (Teleport_NewMap)."""
    _ = (special, line)
    if side != 0:
        return False
    if mo is not None and getattr(mo, "player", None) is not None:
        if getattr(mo.player, "player_state", 0) == 1:
            return False

    map_num = args[0] if len(args) > 0 else 0
    pos = args[1] if len(args) > 1 else 0
    return teleport_new_map(sim, map_num, pos)


def special_teleport_end_game(
    sim: GameSim,
    special: int,
    args: Sequence[int],
    line: Any,
    mo: Any,
    side: int = 0,
) -> bool:
    """Line special 75 (Teleport_EndGame)."""
    _ = (special, args, line)
    if side != 0:
        return False
    if mo is not None and getattr(mo, "player", None) is not None:
        if getattr(mo.player, "player_state", 0) == 1:
            return False

    return teleport_end_game(sim)


SPECIAL_HANDLERS: dict[int, Any] = {
    70: special_teleport,
    71: special_teleport,
    74: special_teleport_new_map,
    75: special_teleport_end_game,
}
