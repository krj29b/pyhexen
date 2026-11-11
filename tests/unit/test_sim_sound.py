# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sim-side sound requests (sim/sound.py) and mover sound sequences (T15.2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.events import SequenceEvent, SoundEvent
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.sound import (
    SEQ_PLATFORM,
    SoundPoint,
    start_sound,
    stop_sound,
)
from pyhexen.sim.specials.ceilings import special_ceiling_lower_by_value
from pyhexen.sim.specials.floors import special_floor_lower_by_value
from pyhexen.sim.specials.pillars import special_pillar_build
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _sim() -> GameSim:
    wb = WadBuilder("IWAD")
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 256, 256, floor_height=64, ceiling_height=192, tag=7)
    mb.add_thing(x=64, y=64, type=1)
    wb.add_map(mb)
    sim = GameSim(level=load_level(WadStack(wb.to_bytes()), "MAP01"))
    sim.level.sectors[0].seq_type = 2  # type: ignore[union-attr]
    return sim


def test_start_sound_carries_origin_and_position() -> None:
    sim = _sim()
    got: list[SoundEvent] = []
    sim.events.subscribe(SoundEvent, got.append)
    point = SoundPoint(10 * FRACUNIT, 20 * FRACUNIT)
    start_sound(sim, point, 33, volume=90)
    start_sound(sim, None, "DoorLocked")
    sector = sim.level.sectors[0]  # type: ignore[union-attr]
    start_sound(sim, sector, 33)
    stop_sound(sim, point)
    start_sound(sim, point, 0)  # SFX_NONE: ignored

    assert got[0].origin is point and (got[0].x, got[0].y, got[0].volume) == (
        10 * FRACUNIT,
        20 * FRACUNIT,
        90,
    )
    assert got[1].origin is None  # at the listener
    assert (got[2].x, got[2].y) == (sector.sound_org_x, sector.sound_org_y)
    assert got[3].stop and got[3].origin is point
    assert len(got) == 4


Mover = Callable[[GameSim, int, tuple[int, ...], Any, Any], bool]


@pytest.mark.parametrize(
    "special",
    [special_floor_lower_by_value, special_ceiling_lower_by_value, special_pillar_build],
)
def test_movers_start_and_stop_their_platform_sequence(special: Mover) -> None:
    sim = _sim()
    seqs: list[SequenceEvent] = []
    sim.events.subscribe(SequenceEvent, seqs.append)
    assert special(sim, 0, (7, 64, 16, 0, 0), None, None)
    sector = sim.level.sectors[0]  # type: ignore[union-attr]
    assert [(e.origin, e.sequence, e.stop) for e in seqs] == [(sector, SEQ_PLATFORM + 2, False)]
    for _ in range(200):
        sim.tick()
    assert seqs[-1].stop and seqs[-1].origin is sector
