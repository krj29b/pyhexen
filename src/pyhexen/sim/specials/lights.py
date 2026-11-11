# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sector lighting specials 110-116 (T7.6, redone in T15.4).

Port of light_t, T_Light and EV_SpawnLight (p_lights.c), in 16.16 fixed point.
Phased lights (T_Phase) live in `sector_specials.py`.

Vanilla quirk kept on purpose: T_Light adds the 16.16 step to the integer light
level and shifts the fraction away every tic, so the fraction never accumulates.
A positive step below 1.0 never brightens, and a negative one darkens by at least
1 per tic (the shift rounds toward minus infinity).
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.fixed import FRACBITS, fixed_div, to_int16, to_int32
from pyhexen.sim.specials.dispatcher import (
    find_sectors_from_tag,
    notify_sector_changed,
)
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class LightType(IntEnum):
    """Port of lighttype_t (p_local.h)."""

    LITE_RAISEBYVALUE = 0
    LITE_LOWERBYVALUE = 1
    LITE_CHANGETOVALUE = 2
    LITE_FADE = 3
    LITE_GLOW = 4
    LITE_FLICKER = 5
    LITE_STROBE = 6


class Light(Thinker):
    """Port of light_t and T_Light (p_lights.c).

    Field meaning depends on `type`, as in vanilla:
    - FADE: value1 = destination level, value2 = 16.16 step, tics2 = +1 brighter / -1 darker.
    - GLOW: value1 = upper, value2 = lower, tics1 = 16.16 step, tics2 = direction.
    - FLICKER: value1 = upper, value2 = lower.
    - STROBE: value1 = upper, value2 = lower, tics1 / tics2 = upper / lower tics.
    `count` delays the next change (flicker and strobe).
    """

    __slots__ = ("count", "sector", "tics1", "tics2", "type", "value1", "value2")

    def __init__(
        self,
        sector: Sector,
        light_type: LightType,
        *,
        value1: int = 0,
        value2: int = 0,
        tics1: int = 0,
        tics2: int = 0,
        count: int = 0,
    ) -> None:
        super().__init__()
        self.sector: Sector = sector
        self.type: LightType = LightType(light_type)
        self.value1: int = value1
        self.value2: int = value2
        self.tics1: int = tics1
        self.tics2: int = tics2
        self.count: int = count

    def tick(self, sim: GameSim) -> None:
        """Port of T_Light (p_lights.c)."""
        if self.count:
            self.count -= 1
            return
        sec = self.sector
        lt = self.type
        if lt == LightType.LITE_FADE:
            sec.light_level = _add_step(sec.light_level, self.value2)
            if self.tics2 == 1:
                if sec.light_level >= self.value1:
                    sec.light_level = self.value1
                    sim.thinkers.remove(self)
            elif sec.light_level <= self.value1:
                sec.light_level = self.value1
                sim.thinkers.remove(self)
        elif lt == LightType.LITE_GLOW:
            sec.light_level = _add_step(sec.light_level, self.tics1)
            if self.tics2 == 1:
                if sec.light_level >= self.value1:
                    sec.light_level = self.value1
                    self.tics1 = to_int32(-self.tics1)
                    self.tics2 = -1  # reverse direction
            elif sec.light_level <= self.value2:
                sec.light_level = self.value2
                self.tics1 = to_int32(-self.tics1)
                self.tics2 = 1  # reverse direction
        elif lt == LightType.LITE_FLICKER:
            if sec.light_level == self.value1:
                sec.light_level = self.value2
                self.count = (sim.rng.p_random() & 7) + 1
            else:
                sec.light_level = self.value1
                self.count = (sim.rng.p_random() & 31) + 1
        elif lt == LightType.LITE_STROBE:
            if sec.light_level == self.value1:
                sec.light_level = self.value2
                self.count = self.tics2
            else:
                sec.light_level = self.value1
                self.count = self.tics1
        else:
            return
        notify_sector_changed(sim, sec)

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "Light",
            self.sector.index if self.sector is not None else -1,
            int(self.type),
            self.value1,
            self.value2,
            self.tics1,
            self.tics2,
            self.count,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "Light",
            "sector_index": self.sector.index if self.sector is not None else -1,
            "light_type": int(self.type),
            "value1": self.value1,
            "value2": self.value2,
            "tics1": self.tics1,
            "tics2": self.tics2,
            "count": self.count,
        }

    @classmethod
    def from_state(cls, sector: Sector, entry: dict[str, Any]) -> Light:
        return cls(
            sector,
            LightType(entry.get("light_type", LightType.LITE_FADE)),
            value1=int(entry.get("value1", 0)),
            value2=int(entry.get("value2", 0)),
            tics1=int(entry.get("tics1", 0)),
            tics2=int(entry.get("tics2", 0)),
            count=int(entry.get("count", 0)),
        )


def _add_step(level: int, step: int) -> int:
    """`((lightlevel << FRACBITS) + step) >> FRACBITS`, stored in a short (sector_t.lightlevel)."""
    return to_int16(to_int32((level << FRACBITS) + step) >> FRACBITS)


def ev_spawn_light(sim: GameSim, args: tuple[int, ...], light_type: LightType) -> bool:
    """Port of EV_SpawnLight (p_lights.c)."""
    assert sim.level is not None
    arg1, arg2, arg3, arg4 = args[1], args[2], args[3], args[4]
    rtn = False
    for sec in find_sectors_from_tag(sim.level, args[0]):
        rtn = True
        light: Light | None = None
        if light_type == LightType.LITE_RAISEBYVALUE:
            sec.light_level = min(to_int16(sec.light_level + arg1), 255)
        elif light_type == LightType.LITE_LOWERBYVALUE:
            sec.light_level = max(to_int16(sec.light_level - arg1), 0)
        elif light_type == LightType.LITE_CHANGETOVALUE:
            sec.light_level = max(0, min(255, arg1))
        elif light_type == LightType.LITE_FADE:
            light = Light(
                sec,
                light_type,
                value1=arg1,  # destination lightlevel
                value2=fixed_div((arg1 - sec.light_level) << FRACBITS, arg2 << FRACBITS),
                tics2=1 if sec.light_level <= arg1 else -1,
            )
        elif light_type == LightType.LITE_GLOW:
            light = Light(
                sec,
                light_type,
                value1=arg1,  # upper lightlevel
                value2=arg2,  # lower lightlevel
                tics1=fixed_div((arg1 - sec.light_level) << FRACBITS, arg3 << FRACBITS),
                tics2=1 if sec.light_level <= arg1 else -1,
            )
        elif light_type == LightType.LITE_FLICKER:
            light = Light(sec, light_type, value1=arg1, value2=arg2)
            sec.light_level = arg1
            light.count = (sim.rng.p_random() & 64) + 1
        elif light_type == LightType.LITE_STROBE:
            light = Light(
                sec, light_type, value1=arg1, value2=arg2, tics1=arg3, tics2=arg4, count=arg3
            )
            sec.light_level = arg1
        notify_sector_changed(sim, sec)
        if light is not None:
            sim.thinkers.add(light)
    return rtn


def _special(light_type: LightType) -> Any:
    def handler(sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any) -> bool:
        return ev_spawn_light(sim, args, light_type)

    handler.__name__ = f"special_{light_type.name.lower()}"
    return handler


SPECIAL_HANDLERS: dict[int, Any] = {
    110: _special(LightType.LITE_RAISEBYVALUE),
    111: _special(LightType.LITE_LOWERBYVALUE),
    112: _special(LightType.LITE_CHANGETOVALUE),
    113: _special(LightType.LITE_FADE),
    114: _special(LightType.LITE_GLOW),
    115: _special(LightType.LITE_FLICKER),
    116: _special(LightType.LITE_STROBE),
}
