# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Floor movers, stairs and floor waggle (T7.3, T15.8).

Port of p_floor.c, except the pillars (see pillars.py).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div, fixed_div, fixed_mul, to_int32
from pyhexen.sim.mobj import FLOAT_BOB_OFFSETS
from pyhexen.sim.sound import start_platform_sequence, stop_sequence
from pyhexen.sim.specials.dispatcher import (
    find_highest_floor_surrounding,
    find_lowest_ceiling_surrounding,
    find_lowest_floor_surrounding,
    find_next_highest_floor,
    find_sectors_from_tag,
    notify_sector_changed,
)
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import ML_TWOSIDED, Sector

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
    RES_PASTDEST, only when the step would go past it. A `direction` of 0 does nothing
    (vanilla's switch has no such case), which is what keeps a Floor_MoveToValueTimes8
    thinker with nowhere to go alive.
    """
    if direction not in (-1, 1):
        return MoveResult.RES_OK

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


class FloorType(IntEnum):
    """Port of floor_e (p_spec.h)."""

    FLEV_LOWERFLOOR = 0  # lower floor to highest surrounding floor
    FLEV_LOWERFLOORTOLOWEST = 1  # lower floor to lowest surrounding floor
    FLEV_LOWERFLOORBYVALUE = 2
    FLEV_RAISEFLOOR = 3  # raise floor to lowest surrounding CEILING
    FLEV_RAISEFLOORTONEAREST = 4  # raise floor to next highest surrounding floor
    FLEV_RAISEFLOORBYVALUE = 5
    FLEV_RAISEFLOORCRUSH = 6
    FLEV_RAISEBUILDSTEP = 7  # One step of a staircase
    FLEV_RAISEBYVALUETIMES8 = 8
    FLEV_LOWERBYVALUETIMES8 = 9
    FLEV_LOWERTIMES8INSTANT = 10
    FLEV_RAISETIMES8INSTANT = 11
    FLEV_MOVETOVALUETIMES8 = 12


class StairsType(IntEnum):
    """Port of stairs_e (p_spec.h). STAIRS_PHASED is never started by vanilla."""

    STAIRS_NORMAL = 0
    STAIRS_SYNC = 1
    STAIRS_PHASED = 2


class FloorMove(Thinker):
    """Port of floormove_t and T_MoveFloor (p_floor.c).

    `textureChange` is left out: only STAIRS_PHASED sets it, and no line special
    starts phased stairs.
    """

    def __init__(
        self,
        sector: Sector,
        speed: int,
        dest: int,
        direction: int,
        crush: int = 0,
        floor_type: int = FloorType.FLEV_RAISEFLOORBYVALUE,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.type = int(floor_type)
        self.speed = speed
        self.floordestheight = dest
        self.direction = direction
        self.crush = crush
        self.reset_height = 0
        self.reset_delay = 0
        self.reset_delay_count = 0
        self.delay_count = 0
        self.delay_total = 0
        self.stairs_delay_height = 0
        self.stairs_delay_height_delta = 0

    def tick(self, sim: GameSim) -> None:
        if self.reset_delay_count:
            self.reset_delay_count -= 1
            if not self.reset_delay_count:
                self.floordestheight = self.reset_height
                self.direction = -self.direction
                self.reset_delay = 0
                self.delay_count = 0
                self.delay_total = 0
        if self.delay_count:
            self.delay_count -= 1
            return

        res = t_move_plane(
            sim,
            self.sector,
            self.speed,
            self.floordestheight,
            self.crush,
            floor_or_ceiling=0,
            direction=self.direction,
        )

        if self.type == FloorType.FLEV_RAISEBUILDSTEP:
            floor_height = self.sector.floor_height
            if (self.direction == 1 and floor_height >= self.stairs_delay_height) or (
                self.direction == -1 and floor_height <= self.stairs_delay_height
            ):
                self.delay_count = self.delay_total
                self.stairs_delay_height += self.stairs_delay_height_delta
        if res == MoveResult.RES_PASTDEST:
            stop_sequence(sim, self.sector)
            if self.delay_total:
                self.delay_total = 0
            if self.reset_delay:
                # Stairs that reset wait here until reset_delay_count runs out.
                return
            self.sector.special_data = None
            sim.tag_finished(self.sector.tag)
            self.removed = True

    _FIELDS = (
        "type",
        "speed",
        "direction",
        "crush",
        "reset_height",
        "reset_delay",
        "reset_delay_count",
        "delay_count",
        "delay_total",
        "stairs_delay_height",
        "stairs_delay_height_delta",
    )

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "FloorMove",
            self.sector.index if self.sector is not None else -1,
            self.floordestheight,
            *(getattr(self, f) for f in self._FIELDS),
        )

    def to_state(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "FloorMove",
            "sector_index": self.sector.index if self.sector is not None else -1,
            "dest": self.floordestheight,
        }
        # "type" is the thinker class name in saves, so the floor_e value is "floor_type".
        data.update({("floor_type" if f == "type" else f): getattr(self, f) for f in self._FIELDS})
        return data

    def load_fields(self, entry: dict[str, Any]) -> None:
        """Restore the fields written by `to_state` (older saves lack most of them)."""
        for f in self._FIELDS:
            key = "floor_type" if f == "type" else f
            if key in entry:
                setattr(self, f, int(entry[key]))


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


def ev_do_floor(sim: GameSim, args: Sequence[int], floortype: int) -> bool:
    """Port of EV_DoFloor (p_floor.c): start a floor mover in every idle sector tagged `args[0]`.

    As in vanilla, Floor_MoveToValueTimes8 on a floor already at the value still leaves
    a thinker with direction 0 in the sector (it never finishes) and reports failure.
    """
    assert sim.level is not None
    rtn = False
    floor: FloorMove | None = None
    for sec in find_sectors_from_tag(sim.level, args[0]):
        if sec.special_data is not None:
            continue  # ALREADY MOVING?  IF SO, KEEP GOING...

        rtn = True
        speed = args[1] * (FRACUNIT // 8)
        if floortype in (FloorType.FLEV_LOWERTIMES8INSTANT, FloorType.FLEV_RAISETIMES8INSTANT):
            speed = 2000 << FRACBITS
        floor = FloorMove(sec, speed, 0, 0, 0, floor_type=floortype)
        sim.thinkers.add(floor)
        sec.special_data = floor

        if floortype == FloorType.FLEV_LOWERFLOOR:
            floor.direction = -1
            floor.floordestheight = find_highest_floor_surrounding(sec)
        elif floortype == FloorType.FLEV_LOWERFLOORTOLOWEST:
            floor.direction = -1
            floor.floordestheight = find_lowest_floor_surrounding(sec)
        elif floortype == FloorType.FLEV_LOWERFLOORBYVALUE:
            floor.direction = -1
            floor.floordestheight = sec.floor_height - args[2] * FRACUNIT
        elif floortype in (
            FloorType.FLEV_LOWERTIMES8INSTANT,
            FloorType.FLEV_LOWERBYVALUETIMES8,
        ):
            floor.direction = -1
            floor.floordestheight = sec.floor_height - args[2] * FRACUNIT * 8
        elif floortype == FloorType.FLEV_RAISEFLOORCRUSH:
            floor.crush = args[2]  # arg[2] = crushing value
            floor.direction = 1
            floor.floordestheight = sec.ceiling_height - 8 * FRACUNIT
        elif floortype == FloorType.FLEV_RAISEFLOOR:
            floor.direction = 1
            floor.floordestheight = min(find_lowest_ceiling_surrounding(sec), sec.ceiling_height)
        elif floortype == FloorType.FLEV_RAISEFLOORTONEAREST:
            floor.direction = 1
            floor.floordestheight = find_next_highest_floor(sec, sec.floor_height)
        elif floortype == FloorType.FLEV_RAISEFLOORBYVALUE:
            floor.direction = 1
            floor.floordestheight = sec.floor_height + args[2] * FRACUNIT
        elif floortype in (
            FloorType.FLEV_RAISETIMES8INSTANT,
            FloorType.FLEV_RAISEBYVALUETIMES8,
        ):
            floor.direction = 1
            floor.floordestheight = sec.floor_height + args[2] * FRACUNIT * 8
        elif floortype == FloorType.FLEV_MOVETOVALUETIMES8:
            floor.floordestheight = args[2] * FRACUNIT * 8
            if args[3]:
                floor.floordestheight = -floor.floordestheight
            if floor.floordestheight > sec.floor_height:
                floor.direction = 1
            elif floor.floordestheight < sec.floor_height:
                floor.direction = -1
            else:  # already at lowest position
                rtn = False
        else:
            rtn = False
    if rtn and floor is not None:
        # One sequence after the loop, on the last sector that got a thinker.
        start_platform_sequence(sim, floor.sector)
    return rtn


def ev_do_floor_and_ceiling(sim: GameSim, args: Sequence[int], raise_: bool) -> bool:
    """Port of EV_DoFloorAndCeiling (p_floor.c).

    The floor thinker's claim on the sector is dropped so the ceiling thinker can start
    too; the sector then holds the ceiling thinker until either of them finishes.
    """
    from pyhexen.sim.specials.ceilings import CeilingType, ev_do_ceiling

    assert sim.level is not None
    floor = ev_do_floor(
        sim,
        args,
        FloorType.FLEV_RAISEFLOORBYVALUE if raise_ else FloorType.FLEV_LOWERFLOORBYVALUE,
    )
    for sec in find_sectors_from_tag(sim.level, args[0]):
        sec.special_data = None
    ceiling = ev_do_ceiling(
        sim,
        args,
        CeilingType.CLEV_RAISEBYVALUE if raise_ else CeilingType.CLEV_LOWERBYVALUE,
    )
    return floor or ceiling


STAIR_SECTOR_TYPE = 26
STAIR_QUEUE_SIZE = 32


def ev_build_stairs(sim: GameSim, args: Sequence[int], direction: int, stairs_type: int) -> bool:
    """Port of EV_BuildStairs, QueueStairSector and ProcessStairSector (p_floor.c).

    Starts at the idle sectors tagged `args[0]` and spreads breadth-first to neighbours
    whose special alternates 26 / 27 and whose floor flat matches. Each step ends
    `args[2]` units past the previous one. `direction` is 1 (up) or -1 (down).
    """
    level = sim.level
    assert level is not None

    step_delta = direction * (args[2] * FRACUNIT)
    speed = args[1] * (FRACUNIT // 8)
    reset_delay = args[4]
    delay = args[3]
    # Vanilla keeps these in globals set from every tagged sector, busy or not, so the
    # last tagged sector decides them for the whole staircase.
    texture = ""
    start_height = 0

    queue: deque[tuple[Sector, int, int]] = deque()

    def queue_stair_sector(sec: Sector, type_: int, height: int) -> None:
        if len(queue) + 1 >= STAIR_QUEUE_SIZE:
            raise RuntimeError("BuildStairs:  Too many branches located.")
        queue.append((sec, type_, height))

    valid_count = level.blockmap.next_valid_count()
    for sec in find_sectors_from_tag(level, args[0]):
        texture = sec.floor_pic
        start_height = sec.floor_height
        if sec.special_data is not None:
            continue  # ALREADY MOVING?  IF SO, KEEP GOING...
        queue_stair_sector(sec, 0, sec.floor_height)
        sec.special = 0

    while queue:
        sec, type_, height = queue.popleft()
        # ProcessStairSector
        height += step_delta
        floor = FloorMove(sec, 0, height, direction, 0, floor_type=FloorType.FLEV_RAISEBUILDSTEP)
        sim.thinkers.add(floor)
        sec.special_data = floor
        if stairs_type == StairsType.STAIRS_NORMAL:
            floor.speed = speed
            if delay:
                floor.delay_total = delay
                floor.stairs_delay_height = sec.floor_height + step_delta
                floor.stairs_delay_height_delta = step_delta
            floor.reset_delay = reset_delay
            floor.reset_delay_count = reset_delay
            floor.reset_height = sec.floor_height
        elif stairs_type == StairsType.STAIRS_SYNC:
            floor.speed = fixed_mul(speed, fixed_div(height - start_height, step_delta))
            floor.reset_delay = delay  # arg4
            floor.reset_delay_count = delay
            floor.reset_height = sec.floor_height
        start_platform_sequence(sim, sec)

        # Find nearby sectors with sector special equal to type
        for line in sec.lines:
            if not (line.flags & ML_TWOSIDED):
                continue
            for tsec in (line.front_sector, line.back_sector):
                if (
                    tsec is not None
                    and tsec.special == type_ + STAIR_SECTOR_TYPE
                    and tsec.special_data is None
                    and tsec.floor_pic == texture
                    and tsec.valid_count != valid_count
                ):
                    queue_stair_sector(tsec, type_ ^ 1, height)
                    tsec.valid_count = valid_count
    return True


def ev_floor_crush_stop(sim: GameSim) -> bool:
    """Port of EV_FloorCrushStop (p_floor.c): remove every crushing floor, whatever its tag."""
    rtn = False
    for thinker in sim.thinkers:
        if not isinstance(thinker, FloorMove) or thinker.type != FloorType.FLEV_RAISEFLOORCRUSH:
            continue
        # Completely remove the crushing floor
        stop_sequence(sim, thinker.sector)
        thinker.sector.special_data = None
        sim.tag_finished(thinker.sector.tag)
        thinker.removed = True
        rtn = True
    return rtn


def _floor_special(floortype: FloorType) -> Any:
    def handler(sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any) -> bool:
        return ev_do_floor(sim, args, floortype)

    handler.__doc__ = f"EV_DoFloor(line, args, {floortype.name}) (p_spec.c)."
    return handler


special_floor_lower_by_value = _floor_special(FloorType.FLEV_LOWERFLOORBYVALUE)
special_floor_raise_by_value = _floor_special(FloorType.FLEV_RAISEFLOORBYVALUE)


def _stairs_special(direction: int, stairs_type: StairsType) -> Any:
    def handler(sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any) -> bool:
        return ev_build_stairs(sim, args, direction, stairs_type)

    handler.__doc__ = f"EV_BuildStairs(line, args, {direction}, {stairs_type.name}) (p_spec.c)."
    return handler


def special_floor_crush_stop(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Floor_CrushStop (46)."""
    return ev_floor_crush_stop(sim)


def special_floor_and_ceiling_lower_by_value(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """FloorAndCeiling_LowerByValue (95)."""
    return ev_do_floor_and_ceiling(sim, args, False)


def special_floor_and_ceiling_raise_by_value(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """FloorAndCeiling_RaiseByValue (96)."""
    return ev_do_floor_and_ceiling(sim, args, True)


def special_floor_waggle(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Floor_Waggle (138): args tag, height, speed, offset, timer."""
    return ev_start_floor_waggle(sim, args[0], args[1], args[2], args[3], args[4])


SPECIAL_HANDLERS: dict[int, Any] = {
    20: special_floor_lower_by_value,
    21: _floor_special(FloorType.FLEV_LOWERFLOORTOLOWEST),
    22: _floor_special(FloorType.FLEV_LOWERFLOOR),
    23: special_floor_raise_by_value,
    24: _floor_special(FloorType.FLEV_RAISEFLOOR),
    25: _floor_special(FloorType.FLEV_RAISEFLOORTONEAREST),
    26: _stairs_special(-1, StairsType.STAIRS_NORMAL),
    27: _stairs_special(1, StairsType.STAIRS_NORMAL),
    28: _floor_special(FloorType.FLEV_RAISEFLOORCRUSH),
    31: _stairs_special(-1, StairsType.STAIRS_SYNC),
    32: _stairs_special(1, StairsType.STAIRS_SYNC),
    35: _floor_special(FloorType.FLEV_RAISEBYVALUETIMES8),
    36: _floor_special(FloorType.FLEV_LOWERBYVALUETIMES8),
    46: special_floor_crush_stop,
    66: _floor_special(FloorType.FLEV_LOWERTIMES8INSTANT),
    67: _floor_special(FloorType.FLEV_RAISETIMES8INSTANT),
    68: _floor_special(FloorType.FLEV_MOVETOVALUETIMES8),
    95: special_floor_and_ceiling_lower_by_value,
    96: special_floor_and_ceiling_raise_by_value,
    138: special_floor_waggle,
}
