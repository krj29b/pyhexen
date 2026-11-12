# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Platforms (lifts) (T7.3, T15.8).

Port of p_plats.c.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.sound import start_platform_sequence, stop_sequence
from pyhexen.sim.specials.dispatcher import (
    find_highest_floor_surrounding,
    find_lowest_floor_surrounding,
    find_sectors_from_tag,
)
from pyhexen.sim.specials.floors import MoveResult, t_move_plane
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

MAXPLATS = 30


class PlatStatus(IntEnum):
    """Port of plat_e (p_spec.h)."""

    PLAT_UP = 0
    PLAT_DOWN = 1
    PLAT_WAITING = 2


class PlatType(IntEnum):
    """Port of plattype_e (p_spec.h)."""

    PLAT_PERPETUALRAISE = 0
    PLAT_DOWNWAITUPSTAY = 1
    PLAT_DOWNBYVALUEWAITUPSTAY = 2
    PLAT_UPWAITDOWNSTAY = 3
    PLAT_UPBYVALUEWAITDOWNSTAY = 4


class Plat(Thinker):
    """Port of plat_t and T_PlatRaise (p_plats.c)."""

    def __init__(self, sector: Sector, plat_type: int, speed: int = 0, tag: int = 0) -> None:
        super().__init__()
        self.sector = sector
        self.type = int(plat_type)
        self.speed = speed
        self.tag = tag
        self.low = 0
        self.high = 0
        self.wait = 0
        self.count = 0
        self.status = int(PlatStatus.PLAT_UP)
        self.crush = 0

    def tick(self, sim: GameSim) -> None:
        if self.status == PlatStatus.PLAT_UP:
            res = t_move_plane(sim, self.sector, self.speed, self.high, self.crush, 0, 1)
            if res == MoveResult.RES_CRUSHED and not self.crush:
                self.count = self.wait
                self.status = PlatStatus.PLAT_DOWN
                start_platform_sequence(sim, self.sector)
            elif res == MoveResult.RES_PASTDEST:
                self.count = self.wait
                self.status = PlatStatus.PLAT_WAITING
                stop_sequence(sim, self.sector)
                if self.type in (
                    PlatType.PLAT_DOWNWAITUPSTAY,
                    PlatType.PLAT_DOWNBYVALUEWAITUPSTAY,
                ):
                    remove_active_plat(sim, self)
        elif self.status == PlatStatus.PLAT_DOWN:
            res = t_move_plane(sim, self.sector, self.speed, self.low, 0, 0, -1)
            if res == MoveResult.RES_PASTDEST:
                self.count = self.wait
                self.status = PlatStatus.PLAT_WAITING
                if self.type in (
                    PlatType.PLAT_UPWAITDOWNSTAY,
                    PlatType.PLAT_UPBYVALUEWAITDOWNSTAY,
                ):
                    remove_active_plat(sim, self)
                stop_sequence(sim, self.sector)
        elif self.status == PlatStatus.PLAT_WAITING:
            # `if (!--plat->count)`: a wait of 0 counts down from -1 and never restarts.
            self.count -= 1
            if not self.count:
                if self.sector.floor_height == self.low:
                    self.status = PlatStatus.PLAT_UP
                else:
                    self.status = PlatStatus.PLAT_DOWN
                start_platform_sequence(sim, self.sector)

    _FIELDS = ("speed", "tag", "low", "high", "wait", "count", "status", "crush")

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "Plat",
            self.sector.index if self.sector is not None else -1,
            self.type,
            *(getattr(self, f) for f in self._FIELDS),
        )

    def to_state(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "Plat",
            "sector_index": self.sector.index if self.sector is not None else -1,
            "plat_type": self.type,
        }
        data.update({f: getattr(self, f) for f in self._FIELDS})
        return data

    def load_fields(self, entry: dict[str, Any]) -> None:
        """Restore the fields written by `to_state`."""
        for f in self._FIELDS:
            if f in entry:
                setattr(self, f, int(entry[f]))


def add_active_plat(sim: GameSim, plat: Plat) -> None:
    """Port of P_AddActivePlat (p_plats.c)."""
    slots = sim.active_plats
    for i, slot in enumerate(slots):
        if slot is None:
            slots[i] = plat
            return
    raise RuntimeError("P_AddActivePlat: no more plats!")


def remove_active_plat(sim: GameSim, plat: Plat) -> None:
    """Port of P_RemoveActivePlat (p_plats.c)."""
    slots = sim.active_plats
    for i, slot in enumerate(slots):
        if slot is plat:
            plat.sector.special_data = None
            sim.tag_finished(plat.sector.tag)
            plat.removed = True
            slots[i] = None
            return
    raise RuntimeError("P_RemoveActivePlat: can't find plat!")


def ev_do_plat(sim: GameSim, args: Sequence[int], plat_type: int) -> bool:
    """Port of EV_DoPlat (p_plats.c): start a platform in every idle sector tagged `args[0]`.

    Vanilla's unused `amount` parameter is left out.
    """
    assert sim.level is not None
    rtn = False
    for sec in find_sectors_from_tag(sim.level, args[0]):
        if sec.special_data is not None:
            continue

        rtn = True
        plat = Plat(sec, plat_type, args[1] * (FRACUNIT // 8), args[0])
        sim.thinkers.add(plat)
        sec.special_data = plat
        plat.wait = args[2]

        if plat_type == PlatType.PLAT_DOWNWAITUPSTAY:
            plat.low = min(find_lowest_floor_surrounding(sec) + 8 * FRACUNIT, sec.floor_height)
            plat.high = sec.floor_height
            plat.status = PlatStatus.PLAT_DOWN
        elif plat_type == PlatType.PLAT_DOWNBYVALUEWAITUPSTAY:
            plat.low = min(sec.floor_height - args[3] * 8 * FRACUNIT, sec.floor_height)
            plat.high = sec.floor_height
            plat.status = PlatStatus.PLAT_DOWN
        elif plat_type == PlatType.PLAT_UPWAITDOWNSTAY:
            plat.high = max(find_highest_floor_surrounding(sec), sec.floor_height)
            plat.low = sec.floor_height
            plat.status = PlatStatus.PLAT_UP
        elif plat_type == PlatType.PLAT_UPBYVALUEWAITDOWNSTAY:
            plat.high = max(sec.floor_height + args[3] * 8 * FRACUNIT, sec.floor_height)
            plat.low = sec.floor_height
            plat.status = PlatStatus.PLAT_UP
        elif plat_type == PlatType.PLAT_PERPETUALRAISE:
            plat.low = min(find_lowest_floor_surrounding(sec) + 8 * FRACUNIT, sec.floor_height)
            plat.high = max(find_highest_floor_surrounding(sec), sec.floor_height)
            plat.status = sim.rng.p_random() & 1
        add_active_plat(sim, plat)
        start_platform_sequence(sim, sec)
    return rtn


def ev_stop_plat(sim: GameSim, args: Sequence[int]) -> None:
    """Port of EV_StopPlat (p_plats.c), bugs included.

    Vanilla does not look for the tag: it overwrites the tag of the plat in the first
    slot with `args[0]` and, if that is non-zero, removes that plat (without stopping
    its sound sequence). With `args[0] == 0` it zeroes tags slot by slot. Vanilla
    dereferences an empty slot here (a crash on most systems); this port stops instead.
    """
    slots = sim.active_plats
    for i, plat in enumerate(slots):
        if plat is None:
            return
        plat.tag = args[0]
        if plat.tag != 0:
            plat.sector.special_data = None
            sim.tag_finished(plat.sector.tag)
            plat.removed = True
            slots[i] = None
            return


def _plat_special(plat_type: PlatType) -> Any:
    def handler(sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any) -> bool:
        return ev_do_plat(sim, args, plat_type)

    handler.__doc__ = f"EV_DoPlat(line, args, {plat_type.name}, 0) (p_spec.c)."
    return handler


special_plat_down_wait_up_stay = _plat_special(PlatType.PLAT_DOWNWAITUPSTAY)


def special_plat_stop(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Plat_Stop (61). Never reports success, as in P_ExecuteLineSpecial."""
    ev_stop_plat(sim, args)
    return False


SPECIAL_HANDLERS: dict[int, Any] = {
    60: _plat_special(PlatType.PLAT_PERPETUALRAISE),
    61: special_plat_stop,
    62: special_plat_down_wait_up_stay,
    63: _plat_special(PlatType.PLAT_DOWNBYVALUEWAITUPSTAY),
    64: _plat_special(PlatType.PLAT_UPWAITDOWNSTAY),
    65: _plat_special(PlatType.PLAT_UPBYVALUEWAITDOWNSTAY),
}
