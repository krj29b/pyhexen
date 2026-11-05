# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Teleport and thing line specials (T7.7).

Port of p_telept.c and p_things.c.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.events import MapExit
from pyhexen.sim.mobj import remove_mobj

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


def special_teleport(sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any) -> bool:
    # args: [tid, ...]
    tid = args[0]
    if mo is None:
        return False
    targets = sim.find_mobjs_by_tid(tid)
    if not targets:
        return False
    dest = targets[0]
    mo.x = dest.x
    mo.y = dest.y
    mo.z = dest.z
    mo.angle = dest.angle
    mo.momx = 0
    mo.momy = 0
    mo.momz = 0
    if mo.player is not None:
        mo.player.viewz = mo.z + mo.player.viewheight
    return True


def special_teleport_new_map(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    # args: [map, position, facing]
    map_num, pos = args[0], args[1]
    sim.events.emit(MapExit(next_map=map_num, position=pos))
    return True


def special_thing_activate(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tid = args[0]
    targets = sim.find_mobjs_by_tid(tid)
    for t in targets:
        # Clear dormant flag if set
        from pyhexen.sim.info_data import MobjFlag2

        t.flags2 &= ~MobjFlag2.MF2_DORMANT
        if t.tics == -1:
            t.tics = 1
    return True


def special_thing_deactivate(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tid = args[0]
    targets = sim.find_mobjs_by_tid(tid)
    for t in targets:
        from pyhexen.sim.info_data import MobjFlag2

        t.flags2 |= MobjFlag2.MF2_DORMANT
        t.tics = -1
    return True


def special_thing_remove(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tid = args[0]
    targets = sim.find_mobjs_by_tid(tid)
    for t in targets:
        remove_mobj(sim, t)
    return True


SPECIAL_HANDLERS: dict[int, Any] = {
    70: special_teleport,
    71: special_teleport,
    74: special_teleport_new_map,
    130: special_thing_activate,
    131: special_thing_deactivate,
    132: special_thing_remove,
}
