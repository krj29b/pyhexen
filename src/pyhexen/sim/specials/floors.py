# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Floor movers, stairs, donut, and floor waggle (T7.3).

Port of p_floor.c.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.mobj import FLOAT_BOB_OFFSETS
from pyhexen.sim.specials.dispatcher import (
    find_sectors_from_tag,
    notify_sector_changed,
)
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class MoveResult(IntEnum):
    RES_OK = 0
    RES_CRUSHED = 1
    RES_PASTDEST = 2


def t_move_plane(
    sim: GameSim,
    sector: Sector,
    speed: int,
    dest: int,
    crush: int,
    floor_or_ceiling: int,
    direction: int,
) -> MoveResult:
    """Port of T_MovePlane (p_floor.c).

    Moves a floor (0) or ceiling (1) toward `dest` at `speed`.
    """
    flag = False
    if floor_or_ceiling == 0:  # Floor
        if direction == 1:  # UP
            new_height = sector.floor_height + speed
            if new_height >= dest:
                new_height = dest
                flag = True
        else:  # DOWN
            new_height = sector.floor_height - speed
            if new_height <= dest:
                new_height = dest
                flag = True

        old_height = sector.floor_height
        sector.floor_height = new_height
        notify_sector_changed(sim, sector)

        # Adjust things in the sector
        mo = sector.thing_list
        while mo is not None:
            next_mo = mo.snext
            if mo.z == old_height or mo.z < new_height:
                mo.z = new_height
                if mo.player is not None:
                    mo.player.viewz = mo.z + mo.player.viewheight
            mo = next_mo

        if flag:
            return MoveResult.RES_PASTDEST
        return MoveResult.RES_OK

    else:  # Ceiling
        if direction == 1:  # UP
            new_height = sector.ceiling_height + speed
            if new_height >= dest:
                new_height = dest
                flag = True
        else:  # DOWN
            new_height = sector.ceiling_height - speed
            if new_height <= dest:
                new_height = dest
                flag = True

        sector.ceiling_height = new_height
        notify_sector_changed(sim, sector)

        # Check crushing
        if direction == -1:
            mo = sector.thing_list
            while mo is not None:
                next_mo = mo.snext
                if mo.z + mo.height > new_height:
                    if crush:
                        mo.health = max(0, mo.health - 10)
                        return MoveResult.RES_CRUSHED
                mo = next_mo

        if flag:
            return MoveResult.RES_PASTDEST
        return MoveResult.RES_OK


class FloorMove(Thinker):
    """Port of floormove_t and T_MoveFloor (p_floor.c)."""

    def __init__(
        self,
        sector: Sector,
        speed: int,
        dest: int,
        direction: int,
        crush: int = 0,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.speed = speed
        self.floordestheight = dest
        self.direction = direction
        self.crush = crush

    def tick(self, sim: GameSim) -> None:
        res = t_move_plane(
            sim,
            self.sector,
            self.speed,
            self.floordestheight,
            self.crush,
            floor_or_ceiling=0,
            direction=self.direction,
        )
        if res == MoveResult.RES_PASTDEST:
            self.removed = True
            if self.sector is not None:
                self.sector.special_data = None
                sim.tag_finished(self.sector.tag)


class FloorWaggle(Thinker):
    """Port of T_FloorWaggle (p_floor.c)."""

    def __init__(
        self,
        sector: Sector,
        original_height: int,
        amplitude: int,
        frequency: int,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.original_height = original_height
        self.amplitude = amplitude
        self.frequency = frequency
        self.accumulator = 0

    def tick(self, sim: GameSim) -> None:
        self.accumulator = (self.accumulator + self.frequency) & 63
        offset = (FLOAT_BOB_OFFSETS[self.accumulator] * self.amplitude) >> 6
        self.sector.floor_height = self.original_height + offset
        notify_sector_changed(sim, self.sector)


def ev_do_floor(
    sim: GameSim,
    tag: int,
    speed: int,
    height: int,
    direction: int,
    crush: int = 0,
) -> bool:
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        dest = sec.floor_height + direction * height if height > 0 else height
        mover = FloorMove(sec, speed, dest, direction, crush)
        sec.special_data = mover
        sim.thinkers.add(mover)
    return True


def special_floor_lower_by_value(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    # args: [tag, speed, height]
    tag, speed_byte, height_byte = args[0], args[1], args[2]
    return ev_do_floor(sim, tag, (speed_byte * FRACUNIT) >> 3, height_byte * FRACUNIT, direction=-1)


def special_floor_raise_by_value(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    # args: [tag, speed, height]
    tag, speed_byte, height_byte = args[0], args[1], args[2]
    return ev_do_floor(sim, tag, (speed_byte * FRACUNIT) >> 3, height_byte * FRACUNIT, direction=1)


def special_floor_waggle(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    # args: [tag, amp, freq, offset, time]
    tag, amp, freq = args[0], args[1], args[2]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        waggle = FloorWaggle(sec, sec.floor_height, amp * FRACUNIT, max(1, freq))
        sim.thinkers.add(waggle)
    return True


SPECIAL_HANDLERS: dict[int, Any] = {
    20: special_floor_lower_by_value,
    23: special_floor_raise_by_value,
    138: special_floor_waggle,
}
