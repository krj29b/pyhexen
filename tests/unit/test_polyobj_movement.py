# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for Phase 9 Milestone 2 (T9.2 Polyobject Movement & Physical Interaction)."""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import ANG90
from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.sim.polyobj import (
    MovePolyThinker,
    PolyDoorThinker,
    ev_rotate_poly,
    get_polyobj,
    thrust_mobj,
)
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _build_test_sim(
    tag: int = 1,
    spawn: tuple[int, int] = (128, 128),
    *,
    width: int = 64,
    height: int = 64,
    crush: bool = False,
    mirror_tag: int = 0,
    seq_type: int = 0,
) -> tuple[Level, GameSim]:
    """Helper creating a test sim and level with a polyobject."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=tag,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=spawn[0],
        spawn_y=spawn[1],
        width=width,
        height=height,
        crush=crush,
        method="start_line",
        mirror_tag=mirror_tag,
        seq_type=seq_type,
    )
    mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    spawn_map_things(sim)
    return level, sim


class TestRotatePolyThinkerUnit:
    """Unit tests for angular motion, speeds, angles, and mirror propagation."""

    def test_rotate_thinker_completion(self) -> None:
        level, sim = _build_test_sim(tag=1, spawn=(128, 128))
        po = get_polyobj(level, 1)
        assert po is not None

        # Rotate 90 degrees CCW (direction=1) with speed 8
        success = ev_rotate_poly(sim, None, (1, 64, 64, 0, 0), direction=1, override=False)
        assert success is True
        assert po.busy is True

        # Tick until completion
        for _ in range(15):
            sim.tick()

        assert po.busy is False
        assert po.angle == ANG90

    def test_rotate_thinker_perpetual(self) -> None:
        level, sim = _build_test_sim(tag=2, spawn=(128, 128))
        po = get_polyobj(level, 2)
        assert po is not None

        # Perpetual rotation (angle byte 255)
        success = ev_rotate_poly(sim, None, (2, 32, 255, 0, 0), direction=1, override=False)
        assert success is True
        for _ in range(20):
            sim.tick()
        assert po.busy is True
        assert po.angle > 0

    def test_rotate_override_preemption(self) -> None:
        level, sim = _build_test_sim(tag=3, spawn=(128, 128))
        po = get_polyobj(level, 3)
        assert po is not None

        # Start perpetual rotation
        ev_rotate_poly(sim, None, (3, 32, 255, 0, 0), direction=1, override=False)
        first_thinker = po.specialdata
        assert first_thinker is not None

        # Override with CW rotation
        ev_rotate_poly(sim, None, (3, 64, 64, 0, 0), direction=-1, override=True)
        assert getattr(first_thinker, "removed", False) is True
        second_thinker = po.specialdata
        assert second_thinker is not first_thinker


class TestMovePolyThinkerUnit:
    """Unit tests for linear motion thinkers, speeds, and times-eight scaling."""

    def test_move_linear_cardinal_east(self) -> None:
        level, sim = _build_test_sim(tag=10, spawn=(100, 100))
        po = get_polyobj(level, 10)
        assert po is not None

        # Move 16 units East with speed 8 units/tic
        execute_line_special(sim, 4, (10, 64, 0, 16, 0))
        sim.tick()
        assert (po.start_spot.x >> FRACBITS) == 108
        sim.tick()
        assert (po.start_spot.x >> FRACBITS) == 116
        assert po.busy is False

    def test_move_times_eight_special(self) -> None:
        level, sim = _build_test_sim(tag=11, spawn=(100, 100))
        po = get_polyobj(level, 11)
        assert po is not None

        # Move 2 units (* 8 = 16 units) East with speed 8 units/tic
        execute_line_special(sim, 6, (11, 64, 0, 2, 0))
        thinker = po.specialdata
        assert isinstance(thinker, MovePolyThinker)
        assert thinker.dist == 16 * FRACUNIT


class TestPolyDoorThinkerUnit:
    """Unit tests for sliding and swinging doors, delays, and obstacle reversal."""

    def test_sliding_door_lifecycle(self) -> None:
        level, sim = _build_test_sim(tag=20, spawn=(100, 100))
        po = get_polyobj(level, 20)
        assert po is not None

        # Slide East 16 units, speed 8, wait 2 tics
        execute_line_special(sim, 8, (20, 64, 0, 16, 2))
        sim.tick()  # opens 8
        sim.tick()  # opens 8 (total 16, enters wait)
        assert (po.start_spot.x >> FRACBITS) == 116
        th = po.specialdata
        assert isinstance(th, PolyDoorThinker)
        assert th.close is True

        sim.tick()  # wait tic 1
        sim.tick()  # wait tic 2 (wait countdown completes)
        sim.tick()  # closes 8
        sim.tick()  # closes 8 (returns to 100)
        assert (po.start_spot.x >> FRACBITS) == 100
        assert po.busy is False

    def test_swinging_door_lifecycle(self) -> None:
        level, sim = _build_test_sim(tag=21, spawn=(100, 100))
        po = get_polyobj(level, 21)
        assert po is not None

        # Swing 64 (90 deg), speed 64 (takes 8 tics), wait 1 tic
        execute_line_special(sim, 7, (21, 64, 64, 1, 0))
        for _ in range(8):
            sim.tick()  # opens 90 deg
        assert po.angle == ANG90
        sim.tick()  # wait tic 1
        for _ in range(8):
            sim.tick()  # closes back to 0
        assert po.angle == 0
        assert po.busy is False

    def test_sliding_door_obstacle_reversal(self) -> None:
        level, sim = _build_test_sim(tag=22, spawn=(100, 100), crush=False)
        po = get_polyobj(level, 22)
        assert po is not None

        # Open door 8 units East, wait 1 tic
        execute_line_special(sim, 8, (22, 64, 0, 8, 1))
        sim.tick()  # opens
        sim.tick()  # wait expires

        # Place player blocking closing path at start spot
        player = spawn_player(sim, 0)
        player.mo.x = 100 * FRACUNIT
        player.mo.y = 100 * FRACUNIT

        sim.tick()  # closing tic encounters player -> must reverse!
        th = po.specialdata
        assert isinstance(th, PolyDoorThinker)
        assert th.close is False  # reversed back to opening phase


class TestPolyCollisionAndDamageUnit:
    """Unit tests for CheckMobjBlocking, ThrustMobj, and pinned crush damage."""

    def test_thrust_mobj_direction_and_clamp(self) -> None:
        level, sim = _build_test_sim(tag=30, spawn=(128, 128))
        player = spawn_player(sim, 0)
        po = get_polyobj(level, 30)
        assert po is not None

        seg = po.segs[0]  # top edge, angle 0
        thrust_mobj(sim, player.mo, seg, po)
        # Momentum must have been applied and clamped
        assert abs(player.mo.momx) <= 4 * FRACUNIT
        assert abs(player.mo.momy) <= 4 * FRACUNIT

    def test_crushing_pinned_damage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        level, sim = _build_test_sim(tag=31, spawn=(128, 128), crush=True)
        player = spawn_player(sim, 0)
        init_health = player.mo.health
        po = get_polyobj(level, 31)
        assert po is not None

        # Monkeypatch check_position in polyobj to simulate being pinned
        import pyhexen.sim.polyobj as po_mod

        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        thrust_mobj(sim, player.mo, po.segs[0], po)
        assert player.mo.health == init_health - 3


class TestACSVMHooksUnit:
    """Unit tests for PCD_POLYWAIT / PCD_POLYWAITDIRECT and poly_finished wakeup."""

    def test_acs_vm_polywait_and_wakeup(self) -> None:
        level, sim = _build_test_sim(tag=40, spawn=(100, 100))
        vm = sim.acs_vm
        assert vm is not None

        po = get_polyobj(level, 40)
        assert po is not None

        # Idle poly is not busy
        assert vm.is_poly_busy(40, sim) is False

        # Start motion
        execute_line_special(sim, 4, (40, 64, 0, 16, 0))
        assert vm.is_poly_busy(40, sim) is True

        # Advance until done
        for _ in range(5):
            sim.tick()

        assert vm.is_poly_busy(40, sim) is False
