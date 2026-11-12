# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Floor movers, stairs, donut, and floor waggle (T7.3).

Port of p_floor.c.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div, fixed_mul, to_int32
from pyhexen.sim.mobj import FLOAT_BOB_OFFSETS
from pyhexen.sim.sound import start_platform_sequence, stop_sequence
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

    Moves a floor (0) or ceiling (1) toward `dest` at `speed`, rechecking nearby things
    with P_ChangeSector. `crush` is the damage dealt every 4 tics to things that no
    longer fit (0: none). When things don't fit the plane goes back; a normal step then
    reports RES_CRUSHED (a rising ceiling never does). Reaching `dest` reports
    RES_PASTDEST, only when the step would go past it.
    """
    from pyhexen.sim.map import change_sector

    is_floor = floor_or_ceiling == 0
    lastpos = sector.floor_height if is_floor else sector.ceiling_height

    def set_height(height: int) -> None:
        if is_floor:
            sector.floor_height = height
        else:
            sector.ceiling_height = height

    if direction == -1:
        past = lastpos - speed < dest
        target = dest if past else lastpos - speed
    else:
        past = lastpos + speed > dest
        target = dest if past else lastpos + speed

    result = MoveResult.RES_PASTDEST if past else MoveResult.RES_OK
    set_height(target)
    if change_sector(sim, sector, crush):
        # A rising ceiling keeps going even if something doesn't fit (#if 0 in vanilla).
        if past or is_floor or direction == -1:
            set_height(lastpos)
            change_sector(sim, sector, crush)
            if not past:
                result = MoveResult.RES_CRUSHED
    if (sector.floor_height if is_floor else sector.ceiling_height) != lastpos:
        notify_sector_changed(sim, sector)
    return result


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
            stop_sequence(sim, self.sector)
            self.removed = True
            if self.sector is not None:
                self.sector.special_data = None
                sim.tag_finished(self.sector.tag)

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "FloorMove",
            self.sector.index if self.sector is not None else -1,
            self.speed,
            self.floordestheight,
            self.direction,
            self.crush,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "FloorMove",
            "sector_index": self.sector.index if self.sector is not None else -1,
            "speed": self.speed,
            "dest": self.floordestheight,
            "direction": self.direction,
            "crush": self.crush,
        }


WGLSTATE_EXPAND = 1
WGLSTATE_STABLE = 2
WGLSTATE_REDUCE = 3


class FloorWaggle(Thinker):
    """Port of floorWaggle_t and T_FloorWaggle (p_floor.c).

    The floor bobs around `original_height` by FloatBobOffsets scaled by `scale`,
    which grows to `target_scale`, holds for `ticker` tics (-1: forever), then shrinks
    back to zero, when the floor is restored and the thinker removed.
    """

    def __init__(
        self,
        sector: Sector,
        original_height: int,
        accumulator: int = 0,
        acc_delta: int = 0,
        target_scale: int = 0,
        scale_delta: int = 0,
        ticker: int = -1,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.original_height = original_height
        self.accumulator = accumulator
        self.acc_delta = acc_delta
        self.scale = 0
        self.target_scale = target_scale
        self.scale_delta = scale_delta
        self.ticker = ticker
        self.state = WGLSTATE_EXPAND

    def tick(self, sim: GameSim) -> None:
        from pyhexen.sim.map import change_sector

        if self.state == WGLSTATE_EXPAND:
            self.scale += self.scale_delta
            if self.scale >= self.target_scale:
                self.scale = self.target_scale
                self.state = WGLSTATE_STABLE
        elif self.state == WGLSTATE_REDUCE:
            self.scale -= self.scale_delta
            if self.scale <= 0:  # Remove
                self.sector.floor_height = self.original_height
                change_sector(sim, self.sector, 1)
                notify_sector_changed(sim, self.sector)
                self.sector.special_data = None
                sim.tag_finished(self.sector.tag)
                self.removed = True
                return
        elif self.state == WGLSTATE_STABLE:
            if self.ticker != -1:
                self.ticker -= 1
                if self.ticker == 0:
                    self.state = WGLSTATE_REDUCE

        self.accumulator = to_int32(self.accumulator + self.acc_delta)
        self.sector.floor_height = self.original_height + fixed_mul(
            FLOAT_BOB_OFFSETS[(self.accumulator >> FRACBITS) & 63], self.scale
        )
        change_sector(sim, self.sector, 1)  # P_ChangeSector(sector, true)
        notify_sector_changed(sim, self.sector)

    _FIELDS = (
        "original_height",
        "accumulator",
        "acc_delta",
        "scale",
        "target_scale",
        "scale_delta",
        "ticker",
        "state",
    )

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "FloorWaggle",
            self.sector.index if self.sector is not None else -1,
            *(getattr(self, f) for f in self._FIELDS),
        )

    def to_state(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "FloorWaggle",
            "sector_index": self.sector.index if self.sector is not None else -1,
        }
        data.update({f: getattr(self, f) for f in self._FIELDS})
        return data

    def load_fields(self, entry: dict[str, Any]) -> None:
        """Restore the fields written by `to_state`."""
        for f in self._FIELDS:
            if f in entry:
                setattr(self, f, int(entry[f]))


def ev_start_floor_waggle(
    sim: GameSim, tag: int, height: int, speed: int, offset: int, timer: int
) -> bool:
    """Port of EV_StartFloorWaggle (p_floor.c)."""
    assert sim.level is not None
    started = False
    for sector in find_sectors_from_tag(sim.level, tag):
        if sector.special_data is not None:
            continue  # Already busy with another thinker
        started = True
        target_scale = height << 10
        waggle = FloorWaggle(
            sector,
            sector.floor_height,
            accumulator=offset * FRACUNIT,
            acc_delta=speed << 10,
            target_scale=target_scale,
            scale_delta=c_div(target_scale, 35 + c_div((3 * 35) * height, 255)),
            ticker=timer * 35 if timer else -1,
        )
        sector.special_data = waggle
        sim.thinkers.add(waggle)
    return started


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
    # Vanilla EV_DoFloor starts one sequence after the loop, on the last sector.
    start_platform_sequence(sim, mover.sector)
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
    """Floor_Waggle (138): args tag, height, speed, offset, timer."""
    return ev_start_floor_waggle(sim, args[0], args[1], args[2], args[3], args[4])


SPECIAL_HANDLERS: dict[int, Any] = {
    20: special_floor_lower_by_value,
    23: special_floor_raise_by_value,
    138: special_floor_waggle,
}
