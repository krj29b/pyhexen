# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Ceiling movers and crushers (T7.4).

Port of p_ceilng.c.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.specials.dispatcher import find_sectors_from_tag
from pyhexen.sim.specials.floors import MoveResult, t_move_plane
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class CeilingMove(Thinker):
    """Port of ceiling_t and T_MoveCeiling (p_ceilng.c)."""

    def __init__(
        self,
        sector: Sector,
        speed: int,
        topheight: int,
        bottomheight: int,
        direction: int,
        crush: int = 0,
        repeat: bool = False,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.speed = speed
        self.topheight = topheight
        self.bottomheight = bottomheight
        self.direction = direction
        self.crush = crush
        self.repeat = repeat

    def tick(self, sim: GameSim) -> None:
        dest = self.bottomheight if self.direction == -1 else self.topheight
        res = t_move_plane(
            sim,
            self.sector,
            self.speed,
            dest,
            crush=self.crush,
            floor_or_ceiling=1,
            direction=self.direction,
        )
        if res == MoveResult.RES_PASTDEST:
            if self.repeat:
                self.direction = -self.direction
            else:
                self.removed = True
        elif res == MoveResult.RES_CRUSHED:
            if self.repeat:
                self.direction = 1  # Reverse on crush


def special_ceiling_lower_by_value(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte, height_byte = args[0], args[1], args[2]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        c = CeilingMove(
            sec,
            (speed_byte * FRACUNIT) >> 3,
            topheight=sec.ceiling_height,
            bottomheight=sec.ceiling_height - height_byte * FRACUNIT,
            direction=-1,
        )
        sim.thinkers.add(c)
    return True


def special_ceiling_raise_by_value(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte, height_byte = args[0], args[1], args[2]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        c = CeilingMove(
            sec,
            (speed_byte * FRACUNIT) >> 3,
            topheight=sec.ceiling_height + height_byte * FRACUNIT,
            bottomheight=sec.ceiling_height,
            direction=1,
        )
        sim.thinkers.add(c)
    return True


def special_ceiling_crush_and_raise(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte, crush = args[0], args[1], args[2]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        c = CeilingMove(
            sec,
            (speed_byte * FRACUNIT) >> 3,
            topheight=sec.ceiling_height,
            bottomheight=sec.floor_height + (8 * FRACUNIT),
            direction=-1,
            crush=crush or 1,
            repeat=True,
        )
        sim.thinkers.add(c)
    return True


SPECIAL_HANDLERS: dict[int, Any] = {
    40: special_ceiling_lower_by_value,
    41: special_ceiling_raise_by_value,
    42: special_ceiling_crush_and_raise,
}
