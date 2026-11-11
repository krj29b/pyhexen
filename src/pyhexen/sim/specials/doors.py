# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Door specials: open, close, raise, locked doors (T7.2).

Port of p_doors.c.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACUNIT
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
        if self.direction == 1:  # UP
            res = t_move_plane(
                sim,
                self.sector,
                self.speed,
                self.topheight,
                crush=0,
                floor_or_ceiling=1,
                direction=1,
            )
            if res == MoveResult.RES_PASTDEST:
                stop_sequence(sim, self.sector)
                if self.type == DoorType.DREV_OPEN:
                    self.removed = True
                    if self.sector is not None:
                        self.sector.special_data = None
                        sim.tag_finished(self.sector.tag)
                else:
                    self.direction = 0
                    self.topcountdown = self.topwait

        elif self.direction == -1:  # DOWN
            res = t_move_plane(
                sim,
                self.sector,
                self.speed,
                self.sector.floor_height,
                crush=0,
                floor_or_ceiling=1,
                direction=-1,
            )
            if res == MoveResult.RES_PASTDEST:
                stop_sequence(sim, self.sector)
                self.removed = True
                if self.sector is not None:
                    self.sector.special_data = None
                    sim.tag_finished(self.sector.tag)
            elif res == MoveResult.RES_CRUSHED:
                # Bounce back open on obstruction
                self.direction = 1

        elif self.direction == 0:  # WAITING
            self.topcountdown -= 1
            if self.topcountdown <= 0:
                self.direction = -1  # time to go back down
                start_door_sequence(sim, self.sector)

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


def ev_do_door(
    sim: GameSim,
    line: Line | None,
    type_: DoorType,
    speed: int,
    tag: int = 0,
    topwait: int = VDOORWAIT,
) -> bool:
    assert sim.level is not None
    if tag == 0 and line is not None and line.back_sector is not None:
        sec = line.back_sector
        door = VerticalDoor(sec, type_, speed, topwait=topwait)
        sec.special_data = door
        sim.thinkers.add(door)
        start_door_sequence(sim, sec)
        return True

    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        door = VerticalDoor(sec, type_, speed, topwait=topwait)
        sec.special_data = door
        sim.thinkers.add(door)
        start_door_sequence(sim, sec)
    return True


def special_door_close(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte = args[0], args[1]
    return ev_do_door(sim, line, DoorType.DREV_CLOSE, (speed_byte * FRACUNIT) >> 3, tag)


def special_door_open(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte = args[0], args[1]
    return ev_do_door(sim, line, DoorType.DREV_OPEN, (speed_byte * FRACUNIT) >> 3, tag)


def special_door_raise(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, speed_byte, wait_tics = args[0], args[1], args[2]
    wait = wait_tics if wait_tics > 0 else VDOORWAIT
    return ev_do_door(
        sim, line, DoorType.DREV_NORMAL, (speed_byte * FRACUNIT) >> 3, tag, topwait=wait
    )


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
    tag, speed_byte, wait_tics, key = args[0], args[1], args[2], args[3]
    if not checked_locked_door(sim, mo, key):
        return False
    wait = wait_tics if wait_tics > 0 else VDOORWAIT
    return ev_do_door(
        sim, line, DoorType.DREV_NORMAL, (speed_byte * FRACUNIT) >> 3, tag, topwait=wait
    )


SPECIAL_HANDLERS: dict[int, Any] = {
    10: special_door_close,
    11: special_door_open,
    12: special_door_raise,
    13: special_door_locked_raise,
}
