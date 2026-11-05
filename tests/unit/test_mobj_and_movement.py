# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for Wave 3 Track C: T5.3 (mobj), T5.4 (map thing spawning),
T6.2 (maputl), and T6.3 (collision & movement).
"""

from __future__ import annotations

from dataclasses import replace

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACBITS, FRACUNIT, fixed_mul
from pyhexen.sim.actions import _ACTIONS, action
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    MOBJINFO,
    MobjFlag,
    MobjFlag2,
    MobjType,
    StateNum,
)
from pyhexen.sim.map import STEPHEIGHT, check_position, slide_move, try_move
from pyhexen.sim.maputl import (
    PT_ADDLINES,
    PT_ADDTHINGS,
    PT_EARLYOUT,
    DivLine,
    Intercept,
    box_on_line_side,
    intercept_vector,
    line_opening,
    path_traverse,
    point_on_line_side,
)
from pyhexen.sim.mobj import (
    FLOATRANDZ,
    MTF_AMBUSH,
    MTF_CLERIC,
    MTF_DORMANT,
    MTF_EASY,
    MTF_FIGHTER,
    MTF_GCOOP,
    MTF_GDEATHMATCH,
    MTF_GSINGLE,
    MTF_HARD,
    MTF_MAGE,
    MTF_NORMAL,
    ONCEILINGZ,
    ONFLOORZ,
    Mobj,
    remove_mobj,
    set_mobj_state,
    spawn_map_things,
    spawn_mobj,
)
from pyhexen.sim.movement import (
    GRAVITY,
    MAXMOVE,
    ORIG_FRICTION,
    STOPSPEED,
    p_xy_movement,
    p_z_movement,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import (
    ML_BLOCKING,
    ML_SPAC_SHIFT,
    ML_TWOSIDED,
    SPAC_CROSS,
    Level,
    SlopeType,
    load_level,
)


def _build_level(mb: MapBuilder, tmp_path) -> Level:
    wad_path = WadBuilder("IWAD").add_map(mb).write(tmp_path / f"{mb.name}.wad")
    with WadStack(wad_path) as wad:
        return load_level(wad, mb.name)


def test_spawn_and_remove_mobj_linking(tmp_path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256, floor_height=16, ceiling_height=128)
    level = _build_level(mb, tmp_path)
    sim = GameSim(level=level)

    # Spawn a fighter player mobj on floor
    mo1 = spawn_mobj(
        sim,
        x=64 * FRACUNIT,
        y=64 * FRACUNIT,
        z=ONFLOORZ,
        mobj_type=MobjType.MT_PLAYER_FIGHTER,
    )
    assert mo1.z == 16 * FRACUNIT
    assert mo1.floorz == 16 * FRACUNIT
    assert mo1.ceilingz == 128 * FRACUNIT
    assert mo1.prev_x == mo1.x
    assert mo1.prev_y == mo1.y
    assert mo1.prev_z == mo1.z
    assert mo1.sector is level.sectors[0]
    assert level.sectors[0].thing_list is mo1

    # Spawn second mobj on ceiling in same sector/block
    mo2 = spawn_mobj(
        sim,
        x=80 * FRACUNIT,
        y=80 * FRACUNIT,
        z=ONCEILINGZ,
        mobj_type=MobjType.MT_ETTIN,
    )
    assert mo2.z == 128 * FRACUNIT - mo2.info.height
    # Sector list is LIFO: mo2 -> mo1
    assert level.sectors[0].thing_list is mo2
    assert mo2.snext is mo1
    assert mo1.sprev is mo2

    # Spawn FLOATRANDZ mobj
    mo3 = spawn_mobj(
        sim,
        x=96 * FRACUNIT,
        y=96 * FRACUNIT,
        z=FLOATRANDZ,
        mobj_type=MobjType.MT_ETTIN,
    )
    assert mo3.floorz <= mo3.z <= mo3.ceilingz - mo3.info.height

    # Remove mo2 and verify sector and blockmap lists remain intact
    remove_mobj(sim, mo2)
    assert mo2.removed is True
    assert level.sectors[0].thing_list is mo3
    assert mo3.snext is mo1
    assert mo1.sprev is mo3

    # Remove remaining mobjs
    remove_mobj(sim, mo3)
    remove_mobj(sim, mo1)
    assert level.sectors[0].thing_list is None


def test_set_mobj_state_zero_tic_chain_and_actions(tmp_path) -> None:
    # Custom zero-tic chain test using patched states in a temporary state sequence
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim = GameSim(level=level)

    mo = spawn_mobj(sim, 64 * FRACUNIT, 64 * FRACUNIT, ONFLOORZ, MobjType.MT_ETTIN)

    call_log: list[str] = []

    @action("A_ChainStep1")
    def _step1(sim_arg: GameSim, mobj_arg: Mobj) -> None:
        assert sim_arg is sim
        assert mobj_arg is mo
        call_log.append("step1")

    @action("A_ChainStep2")
    def _step2(mobj_arg: Mobj) -> None:
        assert mobj_arg is mo
        call_log.append("step2")

    @action("A_ChainFinal")
    def _step_final(sim_arg: GameSim, mobj_arg: Mobj) -> None:
        _ = (sim_arg, mobj_arg)
        call_log.append("final")

    import pyhexen.sim.mobj as mobj_mod

    orig_states = mobj_mod.STATES
    s1_idx = int(StateNum.S_ETTIN_LOOK1)
    s2_idx = int(StateNum.S_ETTIN_LOOK2)
    s3_idx = int(StateNum.S_ETTIN_CHASE1)

    mutable_states = list(orig_states)
    mutable_states[s1_idx] = replace(
        orig_states[s1_idx], tics=0, action_name="A_ChainStep1", next_state=s2_idx
    )
    mutable_states[s2_idx] = replace(
        orig_states[s2_idx], tics=0, action_name="A_ChainStep2", next_state=s3_idx
    )
    mutable_states[s3_idx] = replace(
        orig_states[s3_idx], tics=5, action_name="A_ChainFinal", next_state=s1_idx
    )
    mobj_mod.STATES = tuple(mutable_states)

    try:
        alive = set_mobj_state(sim, mo, s1_idx)
        assert alive is True
        # Both 0-tic states executed immediately and stopped on s3_idx (tics=5)
        assert call_log == ["step1", "step2", "final"]
        assert mo.state_num == s3_idx
        assert mo.tics == 5

        # Tick 4 times: still in s3_idx with tics=1
        for _ in range(4):
            sim.tick()
        assert mo.state_num == s3_idx
        assert mo.tics == 1

        # 5th tick reaches tics=0 and loops back through s1 -> s2 -> s3
        sim.tick()
        assert call_log == ["step1", "step2", "final", "step1", "step2", "final"]
        assert mo.state_num == s3_idx
        assert mo.tics == 5

        # Setting S_NULL removes the mobj and returns False
        assert set_mobj_state(sim, mo, StateNum.S_NULL) is False
        assert mo.removed is True
    finally:
        mobj_mod.STATES = orig_states
        _ACTIONS.pop("A_ChainStep1", None)
        _ACTIONS.pop("A_ChainStep2", None)
        _ACTIONS.pop("A_ChainFinal", None)


def test_spawn_map_things_filters_and_tids(tmp_path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512, floor_height=0, ceiling_height=128)

    ettin_doomednum = MOBJINFO[MobjType.MT_ETTIN].doomednum
    assert ettin_doomednum == 10030

    # Player 1 start (type 1)
    mb.add_thing(x=64, y=64, type=1, flags=0x07E7, args=(0, 0, 0, 0, 0))
    # Deathmatch start (type 11)
    mb.add_thing(x=96, y=64, type=11, flags=0x07E7)
    # Polyobj anchor (type 3000) - deferred
    mb.add_thing(x=128, y=64, type=3000, flags=0x07E7)

    # Fighter + Easy only Ettin with TID=10, Ambush
    mb.add_thing(
        x=160,
        y=160,
        type=ettin_doomednum,
        tid=10,
        angle=90,
        flags=MTF_GSINGLE | MTF_FIGHTER | MTF_EASY | MTF_AMBUSH,
    )
    # Cleric + Normal only Ettin with TID=20, Dormant
    mb.add_thing(
        x=220,
        y=160,
        type=ettin_doomednum,
        tid=20,
        angle=180,
        flags=MTF_GSINGLE | MTF_CLERIC | MTF_NORMAL | MTF_DORMANT,
    )
    # Mage + Hard only Ettin with TID=30
    mb.add_thing(
        x=280,
        y=160,
        type=ettin_doomednum,
        tid=30,
        flags=MTF_GSINGLE | MTF_MAGE | MTF_HARD,
    )
    # Coop-only Ettin
    mb.add_thing(
        x=340,
        y=160,
        type=ettin_doomednum,
        tid=40,
        flags=MTF_GCOOP | MTF_FIGHTER | MTF_EASY,
    )
    # Deathmatch-only Ettin
    mb.add_thing(
        x=400,
        y=160,
        type=ettin_doomednum,
        tid=50,
        flags=MTF_GDEATHMATCH | MTF_FIGHTER | MTF_EASY,
    )

    level = _build_level(mb, tmp_path)

    # Case 1: Single-player Fighter (class=0), Easy (skill=1)
    sim_f_easy = GameSim(level=level, player_class=0, skill=1)
    spawned_f = spawn_map_things(sim_f_easy)
    assert len(spawned_f) == 1
    assert (0, 0) in sim_f_easy.player_starts
    assert len(sim_f_easy.deathmatch_starts) == 1
    assert sim_f_easy.find_mobjs_by_tid(10) == spawned_f
    assert bool(spawned_f[0].flags & MobjFlag.MF_AMBUSH)
    assert sim_f_easy.find_mobjs_by_tid(20) == []

    # Case 2: Single-player Cleric (class=1), Normal (skill=2)
    sim_c_norm = GameSim(level=level, player_class=1, skill=2)
    spawned_c = spawn_map_things(sim_c_norm)
    assert len(spawned_c) == 1
    assert sim_c_norm.find_mobjs_by_tid(20) == spawned_c
    assert bool(spawned_c[0].flags2 & MobjFlag2.MF2_DORMANT)
    assert spawned_c[0].tics == -1

    # Case 3: Single-player Mage (class=2), Hard (skill=3)
    sim_m_hard = GameSim(level=level, player_class=2, skill=3)
    spawned_m = spawn_map_things(sim_m_hard)
    assert len(spawned_m) == 1
    assert sim_m_hard.find_mobjs_by_tid(30) == spawned_m


def test_maputl_geometry_and_path_traverse(tmp_path) -> None:
    mb = MapBuilder("MAP01")
    _, _, portal_idx = mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=128,
        width2=128,
        height=128,
        floor1=0,
        ceil1=128,
        floor2=24,
        ceil2=96,
    )
    level = _build_level(mb, tmp_path)
    portal = level.lines[portal_idx]

    # Portal is at x=128, directed from (128, 128) down to (128, 0)
    assert portal.slopetype == SlopeType.ST_VERTICAL
    # Point at x=64 is on front (right) side = 0; x=192 is on back (left) side = 1
    assert point_on_line_side(64 * FRACUNIT, 64 * FRACUNIT, portal) == 0
    assert point_on_line_side(192 * FRACUNIT, 64 * FRACUNIT, portal) == 1

    # Box entirely on front side -> 0
    box_front = [80 << FRACBITS, 48 << FRACBITS, 48 << FRACBITS, 80 << FRACBITS]
    assert box_on_line_side(box_front, portal) == 0
    # Box entirely on back side -> 1
    box_back = [80 << FRACBITS, 48 << FRACBITS, 160 << FRACBITS, 192 << FRACBITS]
    assert box_on_line_side(box_back, portal) == 1
    # Box straddling x=128 -> -1
    box_cross = [80 << FRACBITS, 48 << FRACBITS, 112 << FRACBITS, 144 << FRACBITS]
    assert box_on_line_side(box_cross, portal) == -1

    # Line opening across portal (floor1=0, ceil1=128 vs floor2=24, ceil2=96)
    opening = line_opening(portal)
    assert opening.opentop == 96 * FRACUNIT
    assert opening.openbottom == 24 * FRACUNIT
    assert opening.lowfloor == 0
    assert opening.openrange == 72 * FRACUNIT

    # Intercept vector: horizontal trace from (0, 64) to (256, 64) crossing x=128 at frac=0.5
    v_trace = DivLine(x=0, y=64 * FRACUNIT, dx=256 * FRACUNIT, dy=0)
    v_line = DivLine(x=128 * FRACUNIT, y=128 * FRACUNIT, dx=0, dy=-128 * FRACUNIT)
    frac = intercept_vector(v_trace, v_line)
    assert frac == FRACUNIT // 2

    # Path traverse across both rooms with a thing placed in room 2
    sim = GameSim(level=level)
    target = spawn_mobj(sim, 192 * FRACUNIT, 64 * FRACUNIT, ONFLOORZ, MobjType.MT_ETTIN)

    visited: list[Intercept] = []
    ok = path_traverse(
        level,
        32 * FRACUNIT,
        64 * FRACUNIT,
        224 * FRACUNIT,
        64 * FRACUNIT,
        PT_ADDLINES | PT_ADDTHINGS,
        lambda in_: visited.append(in_) or True,
    )
    assert ok is True
    assert len(visited) == 2
    # Portal line at x=128 comes before the Ettin at x=192
    assert visited[0].is_aline is True
    assert visited[0].line is portal
    assert visited[1].is_aline is False
    assert visited[1].thing is target
    assert visited[0].frac < visited[1].frac

    # PT_EARLYOUT stops immediately when crossing a 1-sided outer wall
    visited_early: list[Intercept] = []
    ok_early = path_traverse(
        level,
        64 * FRACUNIT,
        64 * FRACUNIT,
        300 * FRACUNIT,
        64 * FRACUNIT,
        PT_ADDLINES | PT_EARLYOUT,
        lambda in_: visited_early.append(in_) or True,
    )
    assert ok_early is False


def test_step_up_ledge_and_doorway_clearance(tmp_path) -> None:
    # Test 1: 24-unit step is climbable
    mb_ok = MapBuilder("MAP01")
    mb_ok.add_two_sector_rooms(
        x=0, y=0, width1=128, width2=128, height=128, floor1=0, ceil1=128, floor2=24, ceil2=128
    )
    level_ok = _build_level(mb_ok, tmp_path)
    sim_ok = GameSim(level=level_ok)
    mo_ok = spawn_mobj(sim_ok, 110 * FRACUNIT, 64 * FRACUNIT, ONFLOORZ, MobjType.MT_PLAYER_FIGHTER)
    assert STEPHEIGHT == 24 * FRACUNIT
    assert try_move(sim_ok, mo_ok, 130 * FRACUNIT, 64 * FRACUNIT) is True
    assert mo_ok.floorz == 24 * FRACUNIT

    # Test 2: 25-unit ledge is too high to step up
    mb_high = MapBuilder("MAP02")
    mb_high.add_two_sector_rooms(
        x=0, y=0, width1=128, width2=128, height=128, floor1=0, ceil1=128, floor2=25, ceil2=128
    )
    level_high = _build_level(mb_high, tmp_path)
    sim_high = GameSim(level=level_high)
    mo_high = spawn_mobj(
        sim_high, 110 * FRACUNIT, 64 * FRACUNIT, ONFLOORZ, MobjType.MT_PLAYER_FIGHTER
    )
    assert try_move(sim_high, mo_high, 130 * FRACUNIT, 64 * FRACUNIT) is False

    # Test 3: Low ceiling doorway (height 48 < player height 64) blocks movement
    mb_lowceil = MapBuilder("MAP03")
    mb_lowceil.add_two_sector_rooms(
        x=0, y=0, width1=128, width2=128, height=128, floor1=0, ceil1=128, floor2=0, ceil2=48
    )
    level_lowceil = _build_level(mb_lowceil, tmp_path)
    sim_lowceil = GameSim(level=level_lowceil)
    mo_lowceil = spawn_mobj(
        sim_lowceil, 110 * FRACUNIT, 64 * FRACUNIT, ONFLOORZ, MobjType.MT_PLAYER_FIGHTER
    )
    assert mo_lowceil.height == 64 * FRACUNIT
    assert try_move(sim_lowceil, mo_lowceil, 130 * FRACUNIT, 64 * FRACUNIT) is False


def test_wall_sliding_and_solid_thing_collision(tmp_path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=256, height=256)
    level = _build_level(mb, tmp_path)
    sim = GameSim(level=level)

    # Place player near the east wall (x=256, player radius=16 -> max x is 240)
    player_mo = spawn_mobj(sim, 232 * FRACUNIT, 64 * FRACUNIT, ONFLOORZ, MobjType.MT_PLAYER_FIGHTER)
    assert bool(player_mo.flags2 & MobjFlag2.MF2_SLIDE)

    # Diagonal velocity into the east wall (+X blocked, +Y free)
    player_mo.momx = 16 * FRACUNIT
    player_mo.momy = 12 * FRACUNIT
    old_y = player_mo.y

    p_xy_movement(sim, player_mo)
    # Player should have slid north (+Y) along the east wall without penetrating x > 240
    assert player_mo.x <= 240 * FRACUNIT
    assert player_mo.y > old_y

    # Solid mobj collision
    blocker = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, ONFLOORZ, MobjType.MT_ETTIN)
    mover = spawn_mobj(sim, 80 * FRACUNIT, 128 * FRACUNIT, ONFLOORZ, MobjType.MT_ETTIN)
    assert check_position(sim, mover, 120 * FRACUNIT, 128 * FRACUNIT) is False
    assert sim.map_state.blocking_mobj is blocker

    # Direct slide_move call test
    player_mo.x = 232 * FRACUNIT
    player_mo.y = 100 * FRACUNIT
    player_mo.momx = 16 * FRACUNIT
    player_mo.momy = 10 * FRACUNIT
    slide_move(sim, player_mo)
    assert player_mo.y > 100 * FRACUNIT


def test_gravity_and_ground_friction(tmp_path) -> None:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512, floor_height=0, ceiling_height=256)
    level = _build_level(mb, tmp_path)
    sim = GameSim(level=level)

    # Spawn player in mid-air at z=64
    mo = spawn_mobj(sim, 128 * FRACUNIT, 128 * FRACUNIT, 64 * FRACUNIT, MobjType.MT_PLAYER_FIGHTER)
    assert mo.z == 64 * FRACUNIT
    assert mo.momz == 0

    # First airborne tic with momz == 0 applies -2 * GRAVITY
    p_z_movement(sim, mo)
    assert mo.momz == -2 * GRAVITY
    # Second airborne tic subtracts GRAVITY (-3 * GRAVITY)
    p_z_movement(sim, mo)
    assert mo.momz == -3 * GRAVITY

    # Step until landing on floor (z == 0)
    for _ in range(20):
        p_z_movement(sim, mo)
    assert mo.z == 0
    assert mo.momz == 0

    # Test ground friction in p_xy_movement
    mo.momx = 4 * FRACUNIT
    mo.momy = 0
    p_xy_movement(sim, mo)
    assert mo.momx == fixed_mul(4 * FRACUNIT, ORIG_FRICTION)

    # Large momentum (> MAXMOVE) is clamped to MAXMOVE and split into half-steps
    mo.x = 64 * FRACUNIT
    mo.momx = 40 * FRACUNIT
    p_xy_movement(sim, mo)
    assert mo.x == 64 * FRACUNIT + MAXMOVE

    # Below STOPSPEED snaps to 0
    mo.momx = STOPSPEED - 1
    mo.momy = 0
    p_xy_movement(sim, mo)
    assert mo.momx == 0


def test_crossed_special_line_spechit_collection(tmp_path) -> None:
    mb = MapBuilder("MAP01")
    _, _, portal_idx = mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=128,
        width2=128,
        height=128,
        portal_flags=ML_TWOSIDED | (SPAC_CROSS << ML_SPAC_SHIFT),
        portal_special=80,  # ACS_Execute
        portal_args=(1, 0, 0, 0, 0),
    )
    level = _build_level(mb, tmp_path)
    sim = GameSim(level=level)

    mo = spawn_mobj(sim, 120 * FRACUNIT, 64 * FRACUNIT, ONFLOORZ, MobjType.MT_PLAYER_FIGHTER)
    assert sim.activated_lines == []

    # Move across the portal at x=128 from x=120 to x=136
    moved = try_move(sim, mo, 136 * FRACUNIT, 64 * FRACUNIT)
    assert moved is True
    assert len(sim.map_state.crossed_lines) == 1
    assert sim.map_state.crossed_lines[0].index == portal_idx
    assert len(sim.activated_lines) == 1
    line, actor, spac = sim.activated_lines[0]
    assert line.index == portal_idx
    assert line.special == 80
    assert actor is mo
    assert spac == SPAC_CROSS
    assert not bool(ML_BLOCKING & line.flags)
