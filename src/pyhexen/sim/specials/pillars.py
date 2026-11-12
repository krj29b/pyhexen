# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Pillars: a floor and a ceiling closing on each other, or opening (T7.3, T15.8).

Port of T_BuildPillar, EV_BuildPillar and EV_OpenPillar (p_floor.c).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div, fixed_div, fixed_mul
from pyhexen.sim.sound import start_platform_sequence, stop_sequence
from pyhexen.sim.specials.dispatcher import (
    find_highest_ceiling_surrounding,
    find_lowest_floor_surrounding,
    find_sectors_from_tag,
)
from pyhexen.sim.specials.floors import MoveResult, t_move_plane
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class Pillar(Thinker):
    """Port of pillar_t and T_BuildPillar (p_floor.c).

    `direction` is 1 to close (floor up, ceiling down) and -1 to open.
    """

    def __init__(
        self,
        sector: Sector,
        floor_speed: int = 0,
        ceiling_speed: int = 0,
        floordest: int = 0,
        ceilingdest: int = 0,
        direction: int = 1,
        crush: int = 0,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.floor_speed = floor_speed
        self.ceiling_speed = ceiling_speed
        self.floordest = floordest
        self.ceilingdest = ceilingdest
        self.direction = direction
        self.crush = crush

    def tick(self, sim: GameSim) -> None:
        # First, raise the floor
        res1 = t_move_plane(
            sim, self.sector, self.floor_speed, self.floordest, self.crush, 0, self.direction
        )
        # Then, lower the ceiling
        res2 = t_move_plane(
            sim, self.sector, self.ceiling_speed, self.ceilingdest, self.crush, 1, -self.direction
        )
        if res1 == MoveResult.RES_PASTDEST and res2 == MoveResult.RES_PASTDEST:
            self.sector.special_data = None
            stop_sequence(sim, self.sector)
            sim.tag_finished(self.sector.tag)
            self.removed = True

    _FIELDS = ("floor_speed", "ceiling_speed", "floordest", "ceilingdest", "direction", "crush")

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "Pillar",
            self.sector.index if self.sector is not None else -1,
            *(getattr(self, f) for f in self._FIELDS),
        )

    def to_state(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "Pillar",
            "sector_index": self.sector.index if self.sector is not None else -1,
        }
        data.update({f: getattr(self, f) for f in self._FIELDS})
        return data

    def load_fields(self, entry: dict[str, Any]) -> None:
        """Restore the fields written by `to_state`.

        Saves from before T15.8 have one `speed` and no direction; they continue as a
        closing pillar with that speed on both planes.
        """
        if "speed" in entry and "floor_speed" not in entry:
            self.floor_speed = self.ceiling_speed = int(entry["speed"])
        for f in self._FIELDS:
            if f in entry:
                setattr(self, f, int(entry[f]))


def ev_build_pillar(sim: GameSim, args: Sequence[int], crush: bool) -> bool:
    """Port of EV_BuildPillar (p_floor.c).

    `args`: tag, speed, height the floor rises (0: meet in the middle), crush damage.
    The slower plane's speed is scaled so both arrive together.
    """
    assert sim.level is not None
    rtn = False
    for sec in find_sectors_from_tag(sim.level, args[0]):
        if sec.special_data is not None:
            continue  # already moving
        if sec.floor_height == sec.ceiling_height:
            continue  # pillar is already closed
        rtn = True
        if not args[2]:
            new_height = sec.floor_height + c_div(sec.ceiling_height - sec.floor_height, 2)
        else:
            new_height = sec.floor_height + (args[2] << FRACBITS)

        pillar = Pillar(sec)
        sec.special_data = pillar
        sim.thinkers.add(pillar)
        if not args[2]:
            pillar.ceiling_speed = pillar.floor_speed = args[1] * (FRACUNIT // 8)
        elif new_height - sec.floor_height > sec.ceiling_height - new_height:
            pillar.floor_speed = args[1] * (FRACUNIT // 8)
            pillar.ceiling_speed = fixed_mul(
                sec.ceiling_height - new_height,
                fixed_div(pillar.floor_speed, new_height - sec.floor_height),
            )
        else:
            pillar.ceiling_speed = args[1] * (FRACUNIT // 8)
            pillar.floor_speed = fixed_mul(
                new_height - sec.floor_height,
                fixed_div(pillar.ceiling_speed, sec.ceiling_height - new_height),
            )
        pillar.floordest = new_height
        pillar.ceilingdest = new_height
        pillar.direction = 1
        pillar.crush = int(crush) * args[3]
        start_platform_sequence(sim, sec)
    return rtn


def ev_open_pillar(sim: GameSim, args: Sequence[int]) -> bool:
    """Port of EV_OpenPillar (p_floor.c).

    `args`: tag, speed, floor drop (0: to the lowest neighbour floor), ceiling rise
    (0: to the highest neighbour ceiling). Only closed pillars open.
    """
    assert sim.level is not None
    rtn = False
    for sec in find_sectors_from_tag(sim.level, args[0]):
        if sec.special_data is not None:
            continue  # already moving
        if sec.floor_height != sec.ceiling_height:
            continue  # pillar isn't closed
        rtn = True
        pillar = Pillar(sec)
        sec.special_data = pillar
        sim.thinkers.add(pillar)
        if not args[2]:
            pillar.floordest = find_lowest_floor_surrounding(sec)
        else:
            pillar.floordest = sec.floor_height - (args[2] << FRACBITS)
        if not args[3]:
            pillar.ceilingdest = find_highest_ceiling_surrounding(sec)
        else:
            pillar.ceilingdest = sec.ceiling_height + (args[3] << FRACBITS)
        if sec.floor_height - pillar.floordest >= pillar.ceilingdest - sec.ceiling_height:
            pillar.floor_speed = args[1] * (FRACUNIT // 8)
            pillar.ceiling_speed = fixed_mul(
                sec.ceiling_height - pillar.ceilingdest,
                fixed_div(pillar.floor_speed, pillar.floordest - sec.floor_height),
            )
        else:
            pillar.ceiling_speed = args[1] * (FRACUNIT // 8)
            pillar.floor_speed = fixed_mul(
                pillar.floordest - sec.floor_height,
                fixed_div(pillar.ceiling_speed, sec.ceiling_height - pillar.ceilingdest),
            )
        pillar.direction = -1  # open the pillar
        start_platform_sequence(sim, sec)
    return rtn


def special_pillar_build(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Pillar_Build (29): no crushing."""
    return ev_build_pillar(sim, args, False)


def special_pillar_open(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Pillar_Open (30)."""
    return ev_open_pillar(sim, args)


def special_pillar_build_and_crush(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Pillar_BuildAndCrush (94)."""
    return ev_build_pillar(sim, args, True)


SPECIAL_HANDLERS: dict[int, Any] = {
    29: special_pillar_build,
    30: special_pillar_open,
    94: special_pillar_build_and_crush,
}
