# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Adversarial boundary stress tests for polyobject simulation (Milestone 2 challenge suite).

Targets boundary conditions, collision edge cases, thrust momentum bounds [0x10000, 0x40000],
3-point crushing damage invariants, door reversals and distance recalculation,
circular mirror links, and thinker preemption mechanics.
"""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import ANG90, ANG_MAX
from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag
from pyhexen.sim.mobj import Mobj, spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.sim.polyobj import (
    MovePolyThinker,
    PolyDoorThinker,
    PolyDoorType,
    RotatePolyThinker,
    ev_move_poly,
    ev_open_poly_door,
    ev_rotate_poly,
    get_polyobj,
    po_move_polyobj,
    thrust_mobj,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _build_boundary_sim(
    tag: int = 1,
    spawn: tuple[int, int] = (128, 128),
    *,
    width: int = 64,
    height: int = 64,
    crush: bool = False,
    mirror_tag: int = 0,
    seq_type: int = 0,
    room_size: tuple[int, int] = (512, 512),
) -> tuple[Level, GameSim]:
    """Helper building a single-polyobject test map and GameSim."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=room_size[0], height=room_size[1])
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


def _build_multi_poly_sim(
    poly_cfgs: list[dict],
) -> tuple[Level, GameSim]:
    """Helper building a multi-polyobject test map with separate rooms for subsector compliance."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    for cfg in poly_cfgs:
        sx, sy = cfg["spawn"]
        # In Raven Hexen each polyobject must spawn in its own subsector/sector
        mb.add_room(x=sx - 128, y=sy - 128, width=256, height=256)
        mb.add_polyobject_box(
            tag=cfg["tag"],
            anchor_x=cfg.get("anchor_x", -2000 - cfg["tag"] * 500),
            anchor_y=cfg.get("anchor_y", -2000),
            spawn_x=sx,
            spawn_y=sy,
            width=cfg.get("width", 64),
            height=cfg.get("height", 64),
            crush=cfg.get("crush", False),
            method=cfg.get("method", "start_line"),
            mirror_tag=cfg.get("mirror_tag", 0),
            seq_type=cfg.get("seq_type", 0),
        )
    first_spawn = poly_cfgs[0]["spawn"]
    mb.add_thing(tid=1, x=first_spawn[0], y=first_spawn[1] - 40, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    spawn_map_things(sim)
    return level, sim


class TestThrustMomentumBounds:
    """Stress tests verifying thrust momentum is strictly bounded in [0x10000, 0x40000]."""

    def test_thrust_momentum_lower_bound_zero_and_minimal_speed(self) -> None:
        """Verify impulse force clamps to 1.0 FRACUNIT (0x10000) for zero or low speed.

        Note: Raven's finecosine[0] == 65535, so FixedMul(0x10000, 65535) yields 0xffff.
        """
        level, sim = _build_boundary_sim(tag=10, spawn=(128, 128))
        po = get_polyobj(level, 10)
        assert po is not None
        seg = po.segs[0]  # East edge

        player = spawn_player(sim, 0)
        # 1. speed = 0 in MovePolyThinker
        th_zero = MovePolyThinker(poly=po, speed=0, dist=100, angle=0, x_speed=0, y_speed=0)
        po.specialdata = th_zero
        player.mo.momx = 0
        player.mo.momy = 0
        thrust_mobj(sim, player.mo, seg, po)

        impulse = max(abs(player.mo.momx), abs(player.mo.momy))
        assert impulse in (0xFFFF, 0x10000), f"Expected [0xffff, 0x10000], got {hex(impulse)}"

        # 2. speed = 1 fixed point unit
        th_one = MovePolyThinker(poly=po, speed=1, dist=100, angle=0, x_speed=0, y_speed=0)
        po.specialdata = th_one
        player.mo.momx = 0
        player.mo.momy = 0
        thrust_mobj(sim, player.mo, seg, po)
        impulse = max(abs(player.mo.momx), abs(player.mo.momy))
        assert impulse in (0xFFFF, 0x10000)

    def test_thrust_momentum_upper_bound_extreme_speeds(self) -> None:
        """Verify impulse force strictly clamps to 4.0 FRACUNIT (0x40000) for huge speeds.

        Note: With finecosine[0] == 65535, FixedMul(0x40000, 65535) yields 0x3fffc.
        """
        level, sim = _build_boundary_sim(tag=11, spawn=(128, 128))
        po = get_polyobj(level, 11)
        assert po is not None
        seg = po.segs[0]
        player = spawn_player(sim, 0)

        extreme_speeds = [
            32 * FRACUNIT,
            64 * FRACUNIT,
            1000 * FRACUNIT,
            0x10000000,
        ]
        for sp in extreme_speeds:
            th = MovePolyThinker(poly=po, speed=sp, dist=100, angle=0, x_speed=0, y_speed=0)
            po.specialdata = th
            player.mo.momx = 0
            player.mo.momy = 0
            thrust_mobj(sim, player.mo, seg, po)
            impulse = max(abs(player.mo.momx), abs(player.mo.momy))
            assert impulse in (
                0x3FFFC,
                0x40000,
            ), f"Speed {sp} upper clamp discrepancy: {hex(impulse)}"

    def test_thrust_momentum_sweep_spectrum(self) -> None:
        """Sweep linear speeds and verify every impulse is strictly in [0xffff, 0x40000]."""
        level, sim = _build_boundary_sim(tag=12, spawn=(128, 128))
        po = get_polyobj(level, 12)
        assert po is not None
        seg = po.segs[0]
        player = spawn_player(sim, 0)

        for test_speed in range(0, 50 * FRACUNIT, 2 * FRACUNIT):
            th = MovePolyThinker(poly=po, speed=test_speed, dist=100, angle=0, x_speed=0, y_speed=0)
            po.specialdata = th
            player.mo.momx = 0
            player.mo.momy = 0
            thrust_mobj(sim, player.mo, seg, po)
            impulse = max(abs(player.mo.momx), abs(player.mo.momy))
            assert 0xFFFF <= impulse <= 0x40000, (
                f"Speed {test_speed} yielded out-of-bounds impulse: {hex(impulse)}"
            )

    def test_thrust_momentum_none_specialdata_default(self) -> None:
        """When polyobject has no active thinker, thrust force defaults to 1.0 (0xffff)."""
        level, sim = _build_boundary_sim(tag=13, spawn=(128, 128))
        po = get_polyobj(level, 13)
        assert po is not None
        po.specialdata = None
        seg = po.segs[0]

        player = spawn_player(sim, 0)
        player.mo.momx = 0
        player.mo.momy = 0
        thrust_mobj(sim, player.mo, seg, po)
        impulse = max(abs(player.mo.momx), abs(player.mo.momy))
        assert impulse in (0xFFFF, 0x10000)

    def test_thrust_rotation_thinker_clamping(self) -> None:
        """Verify RotatePolyThinker uses speed >> 8 clamped to [0x10000, 0x40000]."""
        level, sim = _build_boundary_sim(tag=14, spawn=(128, 128))
        po = get_polyobj(level, 14)
        assert po is not None
        seg = po.segs[0]
        player = spawn_player(sim, 0)

        # 1. Angular speed = 0 -> clamped to 1.0
        po.specialdata = RotatePolyThinker(poly=po, speed=0, dist=100)
        player.mo.momx = 0
        player.mo.momy = 0
        thrust_mobj(sim, player.mo, seg, po)
        assert max(abs(player.mo.momx), abs(player.mo.momy)) in (0xFFFF, 0x10000)

        # 2. Angular speed = 0x2000000 -> 0x2000000 >> 8 = 0x20000 (2.0 FRACUNIT)
        po.specialdata = RotatePolyThinker(poly=po, speed=0x2000000, dist=100)
        player.mo.momx = 0
        player.mo.momy = 0
        thrust_mobj(sim, player.mo, seg, po)
        assert max(abs(player.mo.momx), abs(player.mo.momy)) in (0x1FFFE, 0x20000)

        # 3. Angular speed = 0x10000000 -> clamped to 4.0 FRACUNIT
        po.specialdata = RotatePolyThinker(poly=po, speed=0x10000000, dist=100)
        player.mo.momx = 0
        player.mo.momy = 0
        thrust_mobj(sim, player.mo, seg, po)
        assert max(abs(player.mo.momx), abs(player.mo.momy)) in (0x3FFFC, 0x40000)


class TestCrushingDamagePinnedVsFree:
    """Stress tests verifying 3-point crushing damage invariants."""

    def test_crush_damage_free_actor_zero_damage(self) -> None:
        """When poly.crush is True but actor has free room to move, 0 damage is dealt."""
        level, sim = _build_boundary_sim(tag=20, spawn=(128, 128), crush=True)
        po = get_polyobj(level, 20)
        assert po is not None
        player = spawn_player(sim, 0)
        init_health = player.mo.health

        player.mo.x = 256 * FRACUNIT
        player.mo.y = 256 * FRACUNIT
        thrust_mobj(sim, player.mo, po.segs[0], po)

        assert player.mo.health == init_health, f"Expected no damage, got health {player.mo.health}"
        assert player.damagecount == 0

    def test_crush_damage_pinned_actor_exact_three_points(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When pinned against an obstacle, actor receives exactly 3 points crushing damage."""
        level, sim = _build_boundary_sim(tag=21, spawn=(128, 128), crush=True)
        po = get_polyobj(level, 21)
        assert po is not None
        player = spawn_player(sim, 0)
        init_health = player.mo.health

        import pyhexen.sim.polyobj as po_mod

        # Simulate pinned condition where check_position fails
        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        thrust_mobj(sim, player.mo, po.segs[0], po)

        assert player.mo.health == init_health - 3
        assert player.health == init_health - 3
        assert player.damagecount == 3

    def test_crush_damage_accumulation_over_multiple_tics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Damage accumulates strictly as 3 points per pinned tic."""
        level, sim = _build_boundary_sim(tag=22, spawn=(128, 128), crush=True)
        po = get_polyobj(level, 22)
        assert po is not None
        player = spawn_player(sim, 0)
        init_health = player.mo.health

        import pyhexen.sim.polyobj as po_mod

        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        for i in range(1, 6):
            thrust_mobj(sim, player.mo, po.segs[0], po)
            assert player.mo.health == init_health - (3 * i)
            assert player.damagecount == 3 * i

    def test_non_crushing_poly_pinned_actor_zero_damage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When poly.crush is False, even a pinned actor receives 0 damage."""
        level, sim = _build_boundary_sim(tag=23, spawn=(128, 128), crush=False)
        po = get_polyobj(level, 23)
        assert po is not None
        assert po.crush is False
        player = spawn_player(sim, 0)
        init_health = player.mo.health

        import pyhexen.sim.polyobj as po_mod

        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        thrust_mobj(sim, player.mo, po.segs[0], po)

        assert player.mo.health == init_health
        assert player.damagecount == 0

    def test_crush_monster_pinned_damage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Solid shootable monster (non-player) also takes 3 points damage when pinned."""
        level, sim = _build_boundary_sim(tag=24, spawn=(128, 128), crush=True)
        po = get_polyobj(level, 24)
        assert po is not None

        monster = Mobj(
            x=128 * FRACUNIT,
            y=128 * FRACUNIT,
            flags=MobjFlag.MF_SOLID | MobjFlag.MF_SHOOTABLE,
            health=50,
        )

        import pyhexen.sim.polyobj as po_mod

        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        thrust_mobj(sim, monster, po.segs[0], po)
        assert monster.health == 47

    def test_crush_dead_actor_no_further_damage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dead actor (health <= 0) does not suffer further negative health accumulation."""
        level, sim = _build_boundary_sim(tag=25, spawn=(128, 128), crush=True)
        po = get_polyobj(level, 25)
        assert po is not None
        player = spawn_player(sim, 0)
        player.mo.health = 0

        import pyhexen.sim.polyobj as po_mod

        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        thrust_mobj(sim, player.mo, po.segs[0], po)
        assert player.mo.health == 0


class TestExtremeActorPositionsAndEnclosureBugs:
    """Stress tests verifying collision detection at geometrical boundaries."""

    def test_actor_corner_clipping(self) -> None:
        """Actor diagonally overlapping the polyobject corner vertex is blocked."""
        # Top-right corner of 64x64 box at (128, 128) is (160, 160)
        level, sim = _build_boundary_sim(tag=31, spawn=(128, 128))
        po = get_polyobj(level, 31)
        assert po is not None

        player = spawn_player(sim, 0)
        player.mo.x = 164 * FRACUNIT
        player.mo.y = 164 * FRACUNIT
        player.mo.radius = 16 * FRACUNIT

        blocked = not po_move_polyobj(sim, po, 8 * FRACUNIT, 0)
        assert blocked is True

    def test_actor_grazing_corner_outside_no_block(self) -> None:
        """Actor positioned strictly outside the polyobject does not block motion."""
        # Top-right corner is (160, 160)
        level, sim = _build_boundary_sim(tag=32, spawn=(128, 128))
        po = get_polyobj(level, 32)
        assert po is not None

        player = spawn_player(sim, 0)
        player.mo.x = 180 * FRACUNIT
        player.mo.y = 180 * FRACUNIT
        player.mo.radius = 16 * FRACUNIT

        success = po_move_polyobj(sim, po, -8 * FRACUNIT, 0)
        assert success is True

    def test_actor_inside_convex_polyobject_blocks_motion(self) -> None:
        """Actor placed entirely inside a convex polyobject is detected and blocks motion."""
        level, sim = _build_boundary_sim(tag=33, spawn=(128, 128))
        po = get_polyobj(level, 33)
        assert po is not None

        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        player.mo.radius = 8 * FRACUNIT

        blocked = not po_move_polyobj(sim, po, 8 * FRACUNIT, 0)
        assert blocked is True

    def test_actor_crossing_single_front_seg_receives_single_thrust_and_3pt_damage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Actor crossing only the leading edge seg receives a single thrust and 3 damage."""
        level, sim = _build_boundary_sim(tag=34, spawn=(128, 128), crush=True)
        po = get_polyobj(level, 34)
        assert po is not None

        player = spawn_player(sim, 0)
        # Place center outside AABB (y=170 > 164), with radius 16 crossing only y=160/164
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 170 * FRACUNIT
        player.mo.radius = 16 * FRACUNIT
        init_health = player.mo.health

        import pyhexen.sim.polyobj as po_mod

        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        po_move_polyobj(sim, po, 0, 4 * FRACUNIT)

        damage_dealt = init_health - player.mo.health
        assert damage_dealt == 3, (
            f"Expected 3 points crushing damage from single seg, got {damage_dealt}"
        )

    def test_quadruple_damage_bug_when_actor_is_inside_poly_aabb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EMPIRICAL FINDING: Actor enclosed within poly AABB triggers all 4 segs (12 damage).

        In reference po_man.c:1048-1106, CheckMobjBlocking tests seg linedef crossing only.
        In pyhexen/sim/polyobj.py:725-728, adding `is_inside_poly` causes all 4 segs in the
        polyobject to each independently call thrust_mobj and inflict 3 damage in a single tic,
        yielding 3 * 4 = 12 damage.
        """
        level, sim = _build_boundary_sim(tag=35, spawn=(128, 128), crush=True)
        po = get_polyobj(level, 35)
        assert po is not None
        assert len(po.segs) == 4

        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        init_health = player.mo.health

        import pyhexen.sim.polyobj as po_mod

        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        po_move_polyobj(sim, po, 0, 8 * FRACUNIT)
        damage_taken = init_health - player.mo.health
        assert damage_taken == 12, (
            f"Defect confirmed: Expected 12 damage (4 segs * 3), got {damage_taken}"
        )

    def test_thrust_momentum_cancellation_bug_on_enclosed_actor(self) -> None:
        """EMPIRICAL FINDING: Opposing thrusts from all segs cancel to (0, 0) on enclosed actor.

        Because all 4 segs (North, East, South, West) detect collision simultaneously via
        `is_inside_poly`, they impart opposing impulse vectors that sum to zero, leaving the
        actor trapped with zero net velocity.
        """
        level, sim = _build_boundary_sim(tag=36, spawn=(128, 128))
        po = get_polyobj(level, 36)
        assert po is not None

        player = spawn_player(sim, 0)
        player.mo.x = 128 * FRACUNIT
        player.mo.y = 128 * FRACUNIT
        player.mo.momx = 0
        player.mo.momy = 0

        po_move_polyobj(sim, po, 0, 8 * FRACUNIT)
        assert player.mo.momx == 0 and player.mo.momy == 0


class TestDoorReversalsAndStateTransitions:
    """Stress tests for sliding/swinging door obstacle collision, reversal, and crushing."""

    def test_sliding_door_mid_close_blocking_reversal_remainder_and_speed(self) -> None:
        """Non-crushing sliding door blocked mid-close reverses distance remainder and speed."""
        level, sim = _build_boundary_sim(tag=40, spawn=(100, 100), crush=False)
        po = get_polyobj(level, 40)
        assert po is not None

        # Door opens 64 units East (speed 8 units/tic = 8 tics), wait 2 tics
        ev_open_poly_door(sim, None, (40, 64, 0, 64, 2), PolyDoorType.PODOOR_SLIDE)
        for _ in range(8):
            sim.tick()
        assert (po.start_spot.x >> FRACBITS) == 164

        sim.tick()
        sim.tick()

        th = po.specialdata
        assert isinstance(th, PolyDoorThinker)
        assert th.close is True

        # Let door close for 3 tics (moves 24 units West, 40 units remaining)
        for _ in range(3):
            sim.tick()
        assert (po.start_spot.x >> FRACBITS) == 140
        assert th.dist == 40 * FRACUNIT

        # Place player in path to block the 4th closing step
        player = spawn_player(sim, 0)
        player.mo.x = 100 * FRACUNIT
        player.mo.y = 100 * FRACUNIT
        player.mo.radius = 16 * FRACUNIT

        sim.tick()  # Encounters player -> must reverse!

        assert th.close is False, "Door did not reverse close flag to False"
        assert th.dist == 24 * FRACUNIT, f"Expected 24 FRACUNIT remainder, got {th.dist}"
        assert th.x_speed == 8 * FRACUNIT

        # Move player out of way
        player.mo.x = 400 * FRACUNIT
        player.mo.y = 400 * FRACUNIT

        # Ticking 3 tics returns door to fully open position (164)
        for _ in range(3):
            sim.tick()
        assert (po.start_spot.x >> FRACBITS) == 164

        # Door re-enters wait state
        assert th.close is True
        assert th.dist == 64 * FRACUNIT

    def test_swinging_door_mid_close_blocking_reversal(self) -> None:
        """Non-crushing swinging door blocked mid-close reverses angle and speed."""
        level, sim = _build_boundary_sim(tag=41, spawn=(100, 100), crush=False)
        po = get_polyobj(level, 41)
        assert po is not None

        # Swing 64 BAM (90 deg), speed 64 (takes 8 tics), wait 2 tics
        ev_open_poly_door(sim, None, (41, 64, 64, 2, 0), PolyDoorType.PODOOR_SWING)
        for _ in range(8):
            sim.tick()
        assert po.angle == ANG90

        sim.tick()
        sim.tick()
        th = po.specialdata
        assert isinstance(th, PolyDoorThinker)
        assert th.close is True

        # Close for 3 tics
        for _ in range(3):
            sim.tick()

        # Place player blocking swing path
        player = spawn_player(sim, 0)
        player.mo.x = 100 * FRACUNIT
        player.mo.y = 100 * FRACUNIT
        player.mo.radius = 16 * FRACUNIT

        sim.tick()  # Encounters player -> reverse!
        assert th.close is False
        assert th.speed > 0  # Reopened to CCW direction

    def test_crushing_door_does_not_reverse_and_continues_motion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Crushing door does NOT reverse when blocked; it continues crushing and closes."""
        level, sim = _build_boundary_sim(tag=42, spawn=(100, 100), crush=True)
        po = get_polyobj(level, 42)
        assert po is not None
        assert po.crush is True

        ev_open_poly_door(sim, None, (42, 64, 0, 16, 1), PolyDoorType.PODOOR_SLIDE)
        sim.tick()
        sim.tick()
        sim.tick()  # Wait 1 tic expires

        th = po.specialdata
        assert isinstance(th, PolyDoorThinker)
        assert th.close is True

        # Place player in closing path
        player = spawn_player(sim, 0)
        player.mo.x = 100 * FRACUNIT
        player.mo.y = 100 * FRACUNIT
        player.mo.radius = 16 * FRACUNIT

        import pyhexen.sim.polyobj as po_mod

        # Simulate player pinned against a wall
        monkeypatch.setattr(po_mod, "check_position", lambda *args: False)

        sim.tick()  # Blocked!
        # Crusher door MUST NOT reverse
        assert th.close is True, "Crusher door should not reverse on collision"

        # Simulate obstacle clearing
        player.mo.x = 400 * FRACUNIT
        player.mo.y = 400 * FRACUNIT
        monkeypatch.undo()

        # Door must resume and complete closing
        for _ in range(3):
            sim.tick()
        assert (po.start_spot.x >> FRACBITS) == 100
        assert po.busy is False

    def test_sliding_door_blocked_on_first_closing_tic(self) -> None:
        """Door blocked on very first closing step reverses cleanly without lockup."""
        level, sim = _build_boundary_sim(tag=43, spawn=(100, 100), crush=False)
        po = get_polyobj(level, 43)
        assert po is not None

        ev_open_poly_door(sim, None, (43, 64, 0, 16, 1), PolyDoorType.PODOOR_SLIDE)
        sim.tick()
        sim.tick()
        sim.tick()  # wait expires

        # Place player blocking immediately
        player = spawn_player(sim, 0)
        player.mo.x = 116 * FRACUNIT
        player.mo.y = 100 * FRACUNIT

        sim.tick()  # Blocked on first closing tic
        th = po.specialdata
        assert isinstance(th, PolyDoorThinker)
        assert th.close is False


class TestCircularMirrorLinks:
    """Stress tests for circular mirror topologies ensuring no recursion or lockup."""

    def test_circular_mirror_2_cycle_rotation(self) -> None:
        """Circular mirror link A -> B -> A terminates cleanly in ev_rotate_poly."""
        poly_cfgs = [
            {"tag": 1, "spawn": (100, 100), "mirror_tag": 2},
            {"tag": 2, "spawn": (400, 100), "mirror_tag": 1},
        ]
        level, sim = _build_multi_poly_sim(poly_cfgs)
        p1 = get_polyobj(level, 1)
        p2 = get_polyobj(level, 2)
        assert p1 is not None and p2 is not None

        # Rotate poly 1: must mirror poly 2 and stop (no infinite loop)
        success = ev_rotate_poly(sim, None, (1, 64, 64, 0, 0), direction=1, override=False)
        assert success is True
        assert p1.busy is True
        assert p2.busy is True

        th1 = p1.specialdata
        th2 = p2.specialdata
        assert isinstance(th1, RotatePolyThinker)
        assert isinstance(th2, RotatePolyThinker)
        assert th1.speed > 0
        assert th2.speed < 0

        # Tick to completion
        for _ in range(15):
            sim.tick()

        assert p1.busy is False
        assert p2.busy is False
        assert p1.angle == ANG90
        assert p2.angle == (ANG_MAX - ANG90 + 1) & ANG_MAX

    def test_circular_mirror_3_cycle_linear_motion(self) -> None:
        """Circular mirror link A -> B -> C -> A terminates cleanly in ev_move_poly."""
        poly_cfgs = [
            {"tag": 10, "spawn": (100, 100), "mirror_tag": 11},
            {"tag": 11, "spawn": (400, 100), "mirror_tag": 12},
            {"tag": 12, "spawn": (700, 100), "mirror_tag": 10},
        ]
        level, sim = _build_multi_poly_sim(poly_cfgs)
        p1 = get_polyobj(level, 10)
        p2 = get_polyobj(level, 11)
        p3 = get_polyobj(level, 12)
        assert p1 is not None and p2 is not None and p3 is not None

        # Move poly 10 East
        success = ev_move_poly(sim, None, (10, 64, 0, 16, 0), times_eight=False, override=False)
        assert success is True
        assert p1.busy is True
        assert p2.busy is True
        assert p3.busy is True

        th1 = p1.specialdata
        th2 = p2.specialdata
        th3 = p3.specialdata
        assert th1.x_speed > 0
        assert th2.x_speed < 0
        assert th3.x_speed > 0

        for _ in range(5):
            sim.tick()

        assert p1.busy is False
        assert p2.busy is False
        assert p3.busy is False

    def test_self_mirror_link_no_lockup(self) -> None:
        """Polyobject with mirror pointing to itself terminates without duplicate thinkers."""
        poly_cfgs = [
            {"tag": 20, "spawn": (100, 100), "mirror_tag": 20},
        ]
        level, sim = _build_multi_poly_sim(poly_cfgs)
        p1 = get_polyobj(level, 20)
        assert p1 is not None

        success = ev_rotate_poly(sim, None, (20, 64, 64, 0, 0), direction=1, override=False)
        assert success is True
        for _ in range(15):
            sim.tick()
        assert p1.busy is False

    def test_circular_mirror_sliding_doors(self) -> None:
        """Circular mirror link on sliding door attaches thinkers to both and terminates."""
        poly_cfgs = [
            {"tag": 30, "spawn": (100, 100), "mirror_tag": 31},
            {"tag": 31, "spawn": (400, 100), "mirror_tag": 30},
        ]
        level, sim = _build_multi_poly_sim(poly_cfgs)
        p1 = get_polyobj(level, 30)
        p2 = get_polyobj(level, 31)
        assert p1 is not None and p2 is not None

        success = ev_open_poly_door(sim, None, (30, 64, 0, 16, 1), PolyDoorType.PODOOR_SLIDE)
        assert success is True
        assert p1.busy is True
        assert p2.busy is True

        th1 = p1.specialdata
        th2 = p2.specialdata
        assert th1.x_speed > 0
        assert th2.x_speed < 0


class TestThinkerPreemption:
    """Stress tests for thinker override and preemption."""

    def test_rotate_preempted_by_move_override(self) -> None:
        """Active RotatePolyThinker is preempted and marked removed by ev_move_poly."""
        level, sim = _build_boundary_sim(tag=50, spawn=(128, 128))
        po = get_polyobj(level, 50)
        assert po is not None

        # Start perpetual rotation
        ev_rotate_poly(sim, None, (50, 32, 255, 0, 0), direction=1, override=False)
        rot_th = po.specialdata
        assert isinstance(rot_th, RotatePolyThinker)
        assert not getattr(rot_th, "removed", False)

        # Preempt with linear move East
        ev_move_poly(sim, None, (50, 64, 0, 16, 0), times_eight=False, override=True)
        assert getattr(rot_th, "removed", False) is True
        move_th = po.specialdata
        assert isinstance(move_th, MovePolyThinker)

        init_angle = po.angle
        sim.tick()
        assert po.angle == init_angle
        assert (po.start_spot.x >> FRACBITS) == 136

    def test_move_preempted_by_rotate_override(self) -> None:
        """Active MovePolyThinker is preempted and marked removed by ev_rotate_poly."""
        level, sim = _build_boundary_sim(tag=51, spawn=(128, 128))
        po = get_polyobj(level, 51)
        assert po is not None

        ev_move_poly(sim, None, (51, 64, 0, 64, 0), times_eight=False, override=False)
        move_th = po.specialdata
        assert isinstance(move_th, MovePolyThinker)

        ev_rotate_poly(sim, None, (51, 64, 64, 0, 0), direction=1, override=True)
        assert getattr(move_th, "removed", False) is True
        rot_th = po.specialdata
        assert isinstance(rot_th, RotatePolyThinker)

    def test_door_preempted_by_move_override(self) -> None:
        """Active PolyDoorThinker is preempted by ev_move_poly with override=True."""
        level, sim = _build_boundary_sim(tag=52, spawn=(128, 128))
        po = get_polyobj(level, 52)
        assert po is not None

        ev_open_poly_door(sim, None, (52, 64, 0, 64, 5), PolyDoorType.PODOOR_SLIDE)
        door_th = po.specialdata
        assert isinstance(door_th, PolyDoorThinker)

        ev_move_poly(sim, None, (52, 64, 0, 16, 0), times_eight=False, override=True)
        assert getattr(door_th, "removed", False) is True
        assert isinstance(po.specialdata, MovePolyThinker)

    def test_no_preemption_when_override_false(self) -> None:
        """When override=False, triggering a busy polyobject returns False and keeps old thinker."""
        level, sim = _build_boundary_sim(tag=53, spawn=(128, 128))
        po = get_polyobj(level, 53)
        assert po is not None

        ev_move_poly(sim, None, (53, 64, 0, 64, 0), times_eight=False, override=False)
        orig_th = po.specialdata

        success = ev_rotate_poly(sim, None, (53, 64, 64, 0, 0), direction=1, override=False)
        assert success is False
        assert po.specialdata is orig_th
        assert not getattr(orig_th, "removed", False)

    def test_preemption_propagates_to_mirrored_polyobjects(self) -> None:
        """Preemption with override=True removes and replaces thinkers on mirrored polyobjects."""
        poly_cfgs = [
            {"tag": 60, "spawn": (100, 100), "mirror_tag": 61},
            {"tag": 61, "spawn": (400, 100), "mirror_tag": 60},
        ]
        level, sim = _build_multi_poly_sim(poly_cfgs)
        p1 = get_polyobj(level, 60)
        p2 = get_polyobj(level, 61)
        assert p1 is not None and p2 is not None

        # Start linear move on both
        ev_move_poly(sim, None, (60, 64, 0, 64, 0), times_eight=False, override=False)
        m1 = p1.specialdata
        m2 = p2.specialdata

        # Preempt both with rotation
        ev_rotate_poly(sim, None, (60, 64, 64, 0, 0), direction=1, override=True)
        assert getattr(m1, "removed", False) is True
        assert getattr(m2, "removed", False) is True
        assert isinstance(p1.specialdata, RotatePolyThinker)
        assert isinstance(p2.specialdata, RotatePolyThinker)
