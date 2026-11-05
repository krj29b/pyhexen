# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Pillar movers (T7.5).

Port of EV_BuildPillar, EV_OpenPillar, and T_BuildPillar (p_floor.c).
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


class Pillar(Thinker):
    """Port of pillar_t and T_BuildPillar (p_floor.c)."""

    def __init__(
        self,
        sector: Sector,
        speed: int,
        floordest: int,
        ceilingdest: int,
        crush: int = 0,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.speed = speed
        self.floordest = floordest
        self.ceilingdest = ceilingdest
        self.crush = crush
        self.floor_done = False
        self.ceiling_done = False

    def tick(self, sim: GameSim) -> None:
        if not self.floor_done:
            direction = 1 if self.floordest >= self.sector.floor_height else -1
            res = t_move_plane(
                sim,
                self.sector,
                self.speed,
                self.floordest,
                crush=self.crush,
                floor_or_ceiling=0,
                direction=direction,
            )
            if res == MoveResult.RES_PASTDEST:
                self.floor_done = True

        if not self.ceiling_done:
            direction = 1 if self.ceilingdest >= self.sector.ceiling_height else -1
            res = t_move_plane(
                sim,
                self.sector,
                self.speed,
                self.ceilingdest,
                crush=self.crush,
                floor_or_ceiling=1,
                direction=direction,
            )
            if res == MoveResult.RES_PASTDEST:
                self.ceiling_done = True

        if self.floor_done and self.ceiling_done:
            self.removed = True


def special_pillar_build(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte, _height_byte = args[0], args[1], args[2]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    speed = (speed_byte * FRACUNIT) >> 3
    for sec in sectors:
        mid = (sec.floor_height + sec.ceiling_height) // 2
        pillar = Pillar(sec, speed, floordest=mid, ceilingdest=mid)
        sim.thinkers.add(pillar)
    return True


SPECIAL_HANDLERS: dict[int, Any] = {
    29: special_pillar_build,
}
