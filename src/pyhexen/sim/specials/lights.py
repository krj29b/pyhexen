# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sector lighting specials (T7.6).

Port of p_lights.c.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.sim.specials.dispatcher import (
    find_sectors_from_tag,
    notify_sector_changed,
)
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class LightFade(Thinker):
    """Port of light_t and T_Light (p_lights.c)."""

    def __init__(self, sector: Sector, target: int, tics: int) -> None:
        super().__init__()
        self.sector = sector
        self.target = max(0, min(255, target))
        self.tics = max(1, tics)
        self.step = (self.target - sector.light_level) / float(self.tics)
        self.current = float(sector.light_level)

    def tick(self, sim: GameSim) -> None:
        self.tics -= 1
        if self.tics <= 0:
            self.sector.light_level = self.target
            self.removed = True
        else:
            self.current += self.step
            self.sector.light_level = int(self.current)
        notify_sector_changed(sim, self.sector)


class LightGlow(Thinker):
    """Port of glow_t and T_Glow (p_lights.c)."""

    def __init__(self, sector: Sector, min_light: int, max_light: int, speed: int = 2) -> None:
        super().__init__()
        self.sector = sector
        self.min_light = min_light
        self.max_light = max_light
        self.speed = speed
        self.direction = 1

    def tick(self, sim: GameSim) -> None:
        new_light = self.sector.light_level + self.direction * self.speed
        if new_light >= self.max_light:
            new_light = self.max_light
            self.direction = -1
        elif new_light <= self.min_light:
            new_light = self.min_light
            self.direction = 1
        self.sector.light_level = new_light
        notify_sector_changed(sim, self.sector)


def special_light_change_to_value(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, light = args[0], args[1]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        sec.light_level = max(0, min(255, light))
        notify_sector_changed(sim, sec)
    return True


def special_light_fade(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, dest_light, tics = args[0], args[1], args[2]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        fader = LightFade(sec, dest_light, max(1, tics))
        sim.thinkers.add(fader)
    return True


def special_light_glow(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    tag, upper, lower, speed = args[0], args[1], args[2], args[3]
    assert sim.level is not None
    sectors = find_sectors_from_tag(sim.level, tag)
    if not sectors:
        return False
    for sec in sectors:
        glower = LightGlow(sec, min_light=lower, max_light=upper, speed=max(1, speed))
        sim.thinkers.add(glower)
    return True


SPECIAL_HANDLERS: dict[int, Any] = {
    112: special_light_change_to_value,
    113: special_light_fade,
    114: special_light_glow,
}
