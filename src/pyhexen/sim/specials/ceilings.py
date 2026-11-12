# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Ceiling movers and crushers (T7.3, T15.8).

Port of p_ceilng.c.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACUNIT, c_div
from pyhexen.sim.sound import start_platform_sequence, stop_sequence
from pyhexen.sim.specials.dispatcher import (
    find_highest_ceiling_surrounding,
    find_sectors_from_tag,
)
from pyhexen.sim.specials.floors import MoveResult, t_move_plane
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

MAXCEILINGS = 30


class CeilingType(IntEnum):
    """Port of ceiling_e (p_spec.h)."""

    CLEV_LOWERTOFLOOR = 0
    CLEV_RAISETOHIGHEST = 1
    CLEV_LOWERANDCRUSH = 2
    CLEV_CRUSHANDRAISE = 3
    CLEV_LOWERBYVALUE = 4
    CLEV_RAISEBYVALUE = 5
    CLEV_CRUSHRAISEANDSTAY = 6
    CLEV_MOVETOVALUETIMES8 = 7


class CeilingMove(Thinker):
    """Port of ceiling_t and T_MoveCeiling (p_ceilng.c)."""

    def __init__(self, sector: Sector, ceiling_type: int, speed: int = 0) -> None:
        super().__init__()
        self.sector = sector
        self.type = int(ceiling_type)
        self.speed = speed
        self.topheight = 0
        self.bottomheight = 0
        self.direction = 0
        self.crush = 0
        self.tag = sector.tag

    def tick(self, sim: GameSim) -> None:
        if self.direction == 1:  # UP
            res = t_move_plane(sim, self.sector, self.speed, self.topheight, 0, 1, self.direction)
            if res == MoveResult.RES_PASTDEST:
                stop_sequence(sim, self.sector)
                if self.type == CeilingType.CLEV_CRUSHANDRAISE:
                    self.direction = -1
                    self.speed = self.speed * 2
                else:
                    remove_active_ceiling(sim, self)
        elif self.direction == -1:  # DOWN
            res = t_move_plane(
                sim, self.sector, self.speed, self.bottomheight, self.crush, 1, self.direction
            )
            if res == MoveResult.RES_PASTDEST:
                stop_sequence(sim, self.sector)
                if self.type in (
                    CeilingType.CLEV_CRUSHANDRAISE,
                    CeilingType.CLEV_CRUSHRAISEANDSTAY,
                ):
                    self.direction = 1
                    self.speed = c_div(self.speed, 2)
                else:
                    remove_active_ceiling(sim, self)
            # RES_CRUSHED: vanilla's slow-down is commented out; the crusher keeps its speed.

    _FIELDS = ("speed", "topheight", "bottomheight", "direction", "crush", "tag")

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "CeilingMove",
            self.sector.index if self.sector is not None else -1,
            self.type,
            *(getattr(self, f) for f in self._FIELDS),
        )

    def to_state(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "CeilingMove",
            "sector_index": self.sector.index if self.sector is not None else -1,
            "ceiling_type": self.type,
        }
        data.update({f: getattr(self, f) for f in self._FIELDS})
        return data

    def load_fields(self, entry: dict[str, Any]) -> None:
        """Restore the fields written by `to_state`."""
        for f in self._FIELDS:
            if f in entry:
                setattr(self, f, int(entry[f]))


def add_active_ceiling(sim: GameSim, ceiling: CeilingMove) -> None:
    """Port of P_AddActiveCeiling (p_ceilng.c).

    A 31st ceiling is not recorded, as in vanilla; it then never ends on its own.
    """
    slots = sim.active_ceilings
    for i, slot in enumerate(slots):
        if slot is None:
            slots[i] = ceiling
            return


def remove_active_ceiling(sim: GameSim, ceiling: CeilingMove) -> None:
    """Port of P_RemoveActiveCeiling (p_ceilng.c)."""
    slots = sim.active_ceilings
    for i, slot in enumerate(slots):
        if slot is ceiling:
            ceiling.sector.special_data = None
            ceiling.removed = True
            sim.tag_finished(ceiling.sector.tag)
            slots[i] = None
            break


def ev_do_ceiling(sim: GameSim, args: Sequence[int], ceiling_type: int) -> bool:
    """Port of EV_DoCeiling (p_ceilng.c): start a ceiling in every idle sector tagged `args[0]`."""
    assert sim.level is not None
    rtn = False
    for sec in find_sectors_from_tag(sim.level, args[0]):
        if sec.special_data is not None:
            continue

        rtn = True
        ceiling = CeilingMove(sec, ceiling_type, args[1] * (FRACUNIT // 8))
        sim.thinkers.add(ceiling)
        sec.special_data = ceiling

        if ceiling_type == CeilingType.CLEV_CRUSHRAISEANDSTAY:
            ceiling.crush = args[2]  # arg[2] = crushing value
            ceiling.topheight = sec.ceiling_height
            ceiling.bottomheight = sec.floor_height + 8 * FRACUNIT
            ceiling.direction = -1
        elif ceiling_type in (
            CeilingType.CLEV_CRUSHANDRAISE,
            CeilingType.CLEV_LOWERANDCRUSH,
            CeilingType.CLEV_LOWERTOFLOOR,
        ):
            # Vanilla falls through these three cases.
            if ceiling_type == CeilingType.CLEV_CRUSHANDRAISE:
                ceiling.topheight = sec.ceiling_height
            if ceiling_type != CeilingType.CLEV_LOWERTOFLOOR:
                ceiling.crush = args[2]  # arg[2] = crushing value
            ceiling.bottomheight = sec.floor_height
            if ceiling_type != CeilingType.CLEV_LOWERTOFLOOR:
                ceiling.bottomheight += 8 * FRACUNIT
            ceiling.direction = -1
        elif ceiling_type == CeilingType.CLEV_RAISETOHIGHEST:
            ceiling.topheight = find_highest_ceiling_surrounding(sec)
            ceiling.direction = 1
        elif ceiling_type == CeilingType.CLEV_LOWERBYVALUE:
            ceiling.bottomheight = sec.ceiling_height - args[2] * FRACUNIT
            ceiling.direction = -1
        elif ceiling_type == CeilingType.CLEV_RAISEBYVALUE:
            ceiling.topheight = sec.ceiling_height + args[2] * FRACUNIT
            ceiling.direction = 1
        elif ceiling_type == CeilingType.CLEV_MOVETOVALUETIMES8:
            dest_height = args[2] * FRACUNIT * 8
            if args[3]:
                dest_height = -dest_height
            if sec.ceiling_height <= dest_height:
                ceiling.direction = 1
                ceiling.topheight = dest_height
                if sec.ceiling_height == dest_height:
                    rtn = False
            else:
                ceiling.direction = -1
                ceiling.bottomheight = dest_height
        else:
            rtn = False
        add_active_ceiling(sim, ceiling)
        if rtn:
            start_platform_sequence(sim, sec)
    return rtn


def ev_ceiling_crush_stop(sim: GameSim, args: Sequence[int]) -> bool:
    """Port of EV_CeilingCrushStop (p_ceilng.c): remove the first ceiling tagged `args[0]`."""
    slots = sim.active_ceilings
    for i, ceiling in enumerate(slots):
        if ceiling is not None and ceiling.tag == args[0]:
            stop_sequence(sim, ceiling.sector)
            ceiling.sector.special_data = None
            ceiling.removed = True
            sim.tag_finished(ceiling.sector.tag)
            slots[i] = None
            return True
    return False


def _ceiling_special(ceiling_type: CeilingType) -> Any:
    def handler(sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any) -> bool:
        return ev_do_ceiling(sim, args, ceiling_type)

    handler.__doc__ = f"EV_DoCeiling(line, args, {ceiling_type.name}) (p_spec.c)."
    return handler


special_ceiling_lower_by_value = _ceiling_special(CeilingType.CLEV_LOWERBYVALUE)
special_ceiling_raise_by_value = _ceiling_special(CeilingType.CLEV_RAISEBYVALUE)
special_ceiling_crush_and_raise = _ceiling_special(CeilingType.CLEV_CRUSHANDRAISE)


def special_ceiling_crush_stop(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Ceiling_CrushStop (44)."""
    return ev_ceiling_crush_stop(sim, args)


SPECIAL_HANDLERS: dict[int, Any] = {
    40: special_ceiling_lower_by_value,
    41: special_ceiling_raise_by_value,
    42: special_ceiling_crush_and_raise,
    43: _ceiling_special(CeilingType.CLEV_LOWERANDCRUSH),
    44: special_ceiling_crush_stop,
    45: _ceiling_special(CeilingType.CLEV_CRUSHRAISEANDSTAY),
    69: _ceiling_special(CeilingType.CLEV_MOVETOVALUETIMES8),
}
