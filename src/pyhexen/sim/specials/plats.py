# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Platform movers (T7.5).

Port of p_plats.c.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.sound import start_platform_sequence, stop_sequence
from pyhexen.sim.specials.dispatcher import (
    find_lowest_floor_surrounding,
    find_sectors_from_tag,
)
from pyhexen.sim.specials.floors import MoveResult, t_move_plane
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

PLATWAIT = 105  # 3 seconds


class Plat(Thinker):
    """Port of plat_t and T_PlatRaise (p_plats.c)."""

    def __init__(
        self,
        sector: Sector,
        speed: int,
        low: int,
        high: int,
        direction: int = -1,
        wait: int = PLATWAIT,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.speed = speed
        self.low = low
        self.high = high
        self.direction = direction
        self.wait = wait
        self.count = 0

    def tick(self, sim: GameSim) -> None:
        if self.direction != 0:
            dest = self.low if self.direction == -1 else self.high
            res = t_move_plane(
                sim,
                self.sector,
                self.speed,
                dest,
                crush=0,
                floor_or_ceiling=0,
                direction=self.direction,
            )
            if res == MoveResult.RES_PASTDEST:
                stop_sequence(sim, self.sector)
                if self.direction == -1:
                    self.direction = 0
                    self.count = self.wait
                else:
                    self.removed = True
                    if self.sector is not None:
                        self.sector.special_data = None
                        sim.tag_finished(self.sector.tag)
        else:
            self.count -= 1
            if self.count <= 0:
                self.direction = 1
                start_platform_sequence(sim, self.sector)

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "Plat",
            self.sector.index if self.sector is not None else -1,
            self.speed,
            self.low,
            self.high,
            self.direction,
            self.wait,
            self.count,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "Plat",
            "sector_index": self.sector.index if self.sector is not None else -1,
            "speed": self.speed,
            "low": self.low,
            "high": self.high,
            "direction": self.direction,
            "wait": self.wait,
            "count": self.count,
        }


def special_plat_down_wait_up_stay(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte = args[0], args[1]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        low = find_lowest_floor_surrounding(sec)
        plat = Plat(sec, (speed_byte * FRACUNIT) >> 3, low=low, high=sec.floor_height)
        sec.special_data = plat
        sim.thinkers.add(plat)
        start_platform_sequence(sim, sec)
    return True


SPECIAL_HANDLERS: dict[int, Any] = {
    62: special_plat_down_wait_up_stay,
}
