# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Door specials: open, close, raise, locked doors (T7.2).

Port of p_doors.c.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACUNIT, c_div
from pyhexen.sim.info_data import SfxEnum
from pyhexen.sim.message import set_message
from pyhexen.sim.sound import start_door_sequence, start_sound, stop_sequence
from pyhexen.sim.specials.dispatcher import (
    find_lowest_ceiling_surrounding,
    find_sectors_from_tag,
)
from pyhexen.sim.specials.floors import MoveResult, t_move_plane
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Line, Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class DoorType(IntEnum):
    DREV_NORMAL = 0
    DREV_CLOSE30THENOPEN = 1
    DREV_CLOSE = 2
    DREV_OPEN = 3
    DREV_RAISEIN5MINS = 4


VDOORWAIT = 150  # tics to stay open (4.2 seconds)

KEY_NAMES: tuple[str, ...] = (
    "STEEL KEY",
    "CAVE KEY",
    "AXE KEY",
    "FIRE KEY",
    "EMERALD KEY",
    "DUNGEON KEY",
    "SILVER KEY",
    "RUSTED KEY",
    "HORN KEY",
    "SWAMP KEY",
    "CASTLE KEY",
)


class VerticalDoor(Thinker):
    """Port of vldoor_t and T_VerticalDoor (p_doors.c)."""

    def __init__(
        self,
        sector: Sector,
        type_: DoorType,
        speed: int,
        topwait: int = VDOORWAIT,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.type = type_
        self.topheight = find_lowest_ceiling_surrounding(sector) - (4 * FRACUNIT)
        self.speed = speed
        self.topwait = topwait
        self.topcountdown = 0

        if type_ in (DoorType.DREV_RAISEIN5MINS, DoorType.DREV_CLOSE):
            self.direction = -1
        else:
            self.direction = 1

    def tick(self, sim: GameSim) -> None:
        """Port of T_VerticalDoor (p_doors.c)."""
        if self.direction == 0:  # WAITING
            self.topcountdown -= 1
            if self.topcountdown == 0:
                if self.type == DoorType.DREV_NORMAL:
                    self.direction = -1  # time to go back down
                    start_door_sequence(sim, self.sector)
                elif self.type == DoorType.DREV_CLOSE30THENOPEN:
                    self.direction = 1
        elif self.direction == 2:  # INITIAL WAIT
            self.topcountdown -= 1
            if self.topcountdown == 0 and self.type == DoorType.DREV_RAISEIN5MINS:
                self.direction = 1
                self.type = DoorType.DREV_NORMAL
        elif self.direction == -1:  # DOWN
            res = t_move_plane(
                sim, self.sector, self.speed, self.sector.floor_height, 0, 1, self.direction
            )
            if res == MoveResult.RES_PASTDEST:
                stop_sequence(sim, self.sector)
                if self.type in (DoorType.DREV_NORMAL, DoorType.DREV_CLOSE):
                    self._finish(sim)
                elif self.type == DoorType.DREV_CLOSE30THENOPEN:
                    self.direction = 0
                    self.topcountdown = 35 * 30
            elif res == MoveResult.RES_CRUSHED and self.type != DoorType.DREV_CLOSE:
                self.direction = 1  # DREV_CLOSE: don't go back up
        elif self.direction == 1:  # UP
            res = t_move_plane(sim, self.sector, self.speed, self.topheight, 0, 1, self.direction)
            if res == MoveResult.RES_PASTDEST:
                stop_sequence(sim, self.sector)
                if self.type == DoorType.DREV_NORMAL:
                    self.direction = 0  # wait at top
                    self.topcountdown = self.topwait
                elif self.type in (DoorType.DREV_CLOSE30THENOPEN, DoorType.DREV_OPEN):
                    self._finish(sim)

    def _finish(self, sim: GameSim) -> None:
        self.sector.special_data = None
        sim.tag_finished(self.sector.tag)
        self.removed = True

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "VerticalDoor",
            self.sector.index if self.sector is not None else -1,
            int(self.type),
            self.topheight,
            self.speed,
            self.topwait,
            self.topcountdown,
            self.direction,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "VerticalDoor",
            "sector_index": self.sector.index if self.sector is not None else -1,
            "door_type": int(self.type),
            "speed": self.speed,
            "topheight": self.topheight,
            "topwait": self.topwait,
            "topcountdown": self.topcountdown,
            "direction": self.direction,
        }


def ev_do_door(sim: GameSim, line: Line | None, args: tuple[int, ...], type_: DoorType) -> bool:
    """Port of EV_DoDoor (p_doors.c): move every idle sector tagged `args[0]`.

    Tag 0 matches untagged sectors, as P_FindSectorFromTag does; the Door_Open/Raise
    specials send tag 0 to EV_VerticalDoor first.
    """
    assert sim.level is not None
    speed = c_div(args[1] * FRACUNIT, 8)
    started = False
    for sec in find_sectors_from_tag(sim.level, args[0]):
        if sec.special_data is not None:
            continue
        started = True
        door = VerticalDoor(sec, type_, speed, topwait=args[2])
        if type_ == DoorType.DREV_CLOSE:
            door.direction = -1
        elif type_ == DoorType.DREV_CLOSE30THENOPEN:
            door.topheight = sec.ceiling_height
            door.direction = -1
        elif type_ in (DoorType.DREV_NORMAL, DoorType.DREV_OPEN):
            door.direction = 1
        sec.special_data = door
        sim.thinkers.add(door)
        start_door_sequence(sim, sec)
    return started


def ev_vertical_door(sim: GameSim, line: Line | None, mo: Any) -> bool:
    """Port of EV_VerticalDoor (p_doors.c): open the door behind `line` (tag 0)."""
    if line is None or line.back_sector is None:
        return False
    sec = line.back_sector
    if sec.special_data is not None:
        return False  # the sector already has an active thinker
    if line.special == 11:
        door_type = DoorType.DREV_OPEN
        line.special = 0
    else:
        door_type = DoorType.DREV_NORMAL
    door = VerticalDoor(sec, door_type, line.args[1] * (FRACUNIT // 8), topwait=line.args[2])
    door.direction = 1
    sim.thinkers.add(door)
    sec.special_data = door
    start_door_sequence(sim, sec)
    return True


def special_door_close(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Door_Close (10)."""
    return ev_do_door(sim, line, args, DoorType.DREV_CLOSE)


def special_door_open(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Door_Open (11)."""
    if not args[0]:
        return ev_vertical_door(sim, line, mo)
    return ev_do_door(sim, line, args, DoorType.DREV_OPEN)


def special_door_raise(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Door_Raise (12)."""
    if not args[0]:
        return ev_vertical_door(sim, line, mo)
    return ev_do_door(sim, line, args, DoorType.DREV_NORMAL)


def checked_locked_door(sim: GameSim, mo: Any, lock: int) -> bool:
    """Port of CheckedLockedDoor (p_spec.c); also the lock check in ACS_LockedExecute."""
    if mo is None or mo.player is None:
        return False
    if not lock:
        return True
    if not (mo.player.keys & (1 << (lock - 1))):
        key_name = KEY_NAMES[lock - 1] if lock - 1 < len(KEY_NAMES) else f"KEY {lock}"
        set_message(sim, mo.player, f"YOU NEED THE {key_name}\n", True)
        start_sound(sim, mo, int(SfxEnum.SFX_DOOR_LOCKED))
        return False
    return True


def special_door_locked_raise(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Door_LockedRaise (13)."""
    if not checked_locked_door(sim, mo, args[3]):
        return False
    if not args[0]:
        return ev_vertical_door(sim, line, mo)
    return ev_do_door(sim, line, args, DoorType.DREV_NORMAL)


SPECIAL_HANDLERS: dict[int, Any] = {
    10: special_door_close,
    11: special_door_open,
    12: special_door_raise,
    13: special_door_locked_raise,
}
