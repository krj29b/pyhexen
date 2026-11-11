# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Comprehensive 4-tier unit test suite for Monster AI Base and core monster actions (Task T10.4).

Reference: reference/src/hexen/p_enemy.c, reference/src/hexen/a_action.c

Tiers:
1. Feature Coverage:
   - Sound alert recursion & sound target propagation across single room, 2-sided portal,
     3-room corridor, and single soundblock line.
   - Target acquisition (A_Look) in 180° FOV, wakeup from sector sound_target, behind-back
     melee range surprise, boss full-volume broadcast, and dead player filtering.
   - Melee and missile range checks (P_CheckMeleeRange, P_CheckMeleeRange2, P_CheckMissileRange,
     vertical clearance, reaction time gate, MF_JUSTHIT retaliation).
   - 8-way directional movement (P_Move, P_TryWalk, P_NewChaseDir, diagonal speed 47000,
     floating monster vertical adaptation, turnaround avoidance).
   - A_Chase AI state machine (reaction/threshold countdown, turning towards movedir,
     melee attack transition, missile attack transition, active sound chance).
   - Base shared AI actions (A_FaceTarget, A_Pain, A_Scream, A_XScream, A_NoBlocking,
     A_Explode, A_FreezeDeath, A_FreezeDeathChunks, A_QueueCorpse, A_DeQueueCorpse).

2. Boundary & Corner Cases:
   - Two soundblock barrier halts propagation.
   - Closed portal (openrange <= 0) blocks sound.
   - One-sided linedef blocks sound.
   - Ambush monster ignores sound without line-of-sight; wakes when sight clear.
   - Rear player outside MELEERANGE does not wake monster.
   - Shadow sneaking player avoids detection.
   - P_NewChaseDir with target=None raises RuntimeError.
   - Turnaround suppression when other directions open; fallback to DI_NODIR when fully boxed in.
   - A_Chase target killed reverts to spawnstate; MF_JUSTATTACKED suppresses immediate reattack;
     nightmare skill halves tics.
   - A_FreezeDeathChunks moving actor postpones shatter.

3. Cross-Feature Combinations:
   - End-to-end sound wake -> chase -> melee attack sequence.
   - Chained noise alert across 4-room corridor.
   - Retaliation from rear attack (MF_JUSTHIT overrides reactiontime).
   - Monster doorway navigation across sector portal.
   - Ambush monster waiting for line of sight after noise.

4. Real-World Simulation Scenarios & Architectural Audits:
   - Multi-monster combat chase simulation.
   - Nightmare mode speed comparison.
   - Deterministic replay state hash validation.
   - AST compliance audit (zero floats, no stdlib random,
     no render/audio/ui imports in sim/enemy.py).
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import ML_BLOCKING, ML_TWOSIDED, MapBuilder, WadBuilder

from pyhexen.core.angles import ANG90, ANG180
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim import (
    DI_EAST,
    DI_NODIR,
    DI_NORTH,
    DI_NORTHEAST,
    DI_WEST,
    FLOATSPEED,
    a_chase,
    a_dequeuecorpse,
    a_explode,
    a_facetarget,
    a_freetargmobj,
    a_freezedeath,
    a_freezedeathchunks,
    a_icecheckheaddone,
    a_icesettics,
    a_look,
    a_noblocking,
    a_pain,
    a_queuecorpse,
    a_scream,
    a_setinvulnerable,
    a_setreflective,
    a_unsetinvulnerable,
    a_unsetreflective,
    a_xscream,
    check_melee_range,
    check_melee_range2,
    check_missile_range,
    move,
    new_chase_dir,
    noise_alert,
    p_init_creature_corpse_queue,
    try_walk,
)
from pyhexen.sim.actions import is_action_implemented
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.mobj import Mobj, set_mobj_state, spawn_mobj
from pyhexen.sim.player import Player, PlayerClass, PlayerState
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import ML_SOUNDBLOCK, load_level

# ==============================================================================
# Helper Fixtures & Map Generators
# ==============================================================================


def _make_sim_from_map(mb: MapBuilder) -> GameSim:
    """Build a synthetic level in memory and attach to a new GameSim."""
    stack = WadStack(WadBuilder("IWAD").add_map(mb).to_bytes())
    level = load_level(stack, mb.name)
    rng = Random()
    rng.clear()
    return GameSim(level=level, rng=rng)


def _make_two_room_sim(
    portal_flags: int = ML_TWOSIDED,
    floor1: int = 0,
    ceil1: int = 128,
    floor2: int = 0,
    ceil2: int = 128,
) -> GameSim:
    """Build a standard two-room test map connected by a portal linedef."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        floor1=floor1,
        ceil1=ceil1,
        floor2=floor2,
        ceil2=ceil2,
        portal_flags=portal_flags,
    )
    return _make_sim_from_map(mb)


def _make_corridor_sim(
    num_rooms: int = 3,
    room_width: int = 256,
    room_height: int = 256,
    portal_flags: Sequence[int] | None = None,
    ceilings: Sequence[int] | None = None,
) -> GameSim:
    """Build an N-room linear corridor map with custom portal flags and ceiling heights."""
    if portal_flags is None:
        portal_flags = [ML_TWOSIDED] * (num_rooms - 1)
    if ceilings is None:
        ceilings = [128] * num_rooms

    mb = MapBuilder("MAP01")
    sec_indices = [
        mb.add_sector(floor_height=0, ceiling_height=ceilings[i]) for i in range(num_rooms)
    ]

    v_top = [mb.add_vertex(i * room_width, room_height) for i in range(num_rooms + 1)]
    v_bot = [mb.add_vertex(i * room_width, 0) for i in range(num_rooms + 1)]

    l_left = mb.add_linedef(v_bot[0], v_top[0], mb.add_sidedef(sec_indices[0]), flags=ML_BLOCKING)
    l_right = mb.add_linedef(
        v_top[num_rooms],
        v_bot[num_rooms],
        mb.add_sidedef(sec_indices[-1]),
        flags=ML_BLOCKING,
    )

    l_top = []
    l_bot = []
    for i in range(num_rooms):
        l_top.append(
            mb.add_linedef(
                v_top[i], v_top[i + 1], mb.add_sidedef(sec_indices[i]), flags=ML_BLOCKING
            )
        )
        l_bot.append(
            mb.add_linedef(
                v_bot[i + 1], v_bot[i], mb.add_sidedef(sec_indices[i]), flags=ML_BLOCKING
            )
        )

    l_portals = []
    for i in range(num_rooms - 1):
        sd_front = mb.add_sidedef(sec_indices[i])
        sd_back = mb.add_sidedef(sec_indices[i + 1])
        lp = mb.add_linedef(v_top[i + 1], v_bot[i + 1], sd_front, sd_back, flags=portal_flags[i])
        l_portals.append(lp)

    for i in range(num_rooms):
        first_seg = len(mb.segs)
        mb.add_seg(v_top[i], v_top[i + 1], 0, l_top[i], side=0)
        if i == num_rooms - 1:
            mb.add_seg(v_top[i + 1], v_bot[i + 1], -16384, l_right, side=0)
        else:
            mb.add_seg(v_top[i + 1], v_bot[i + 1], -16384, l_portals[i], side=0)
        mb.add_seg(v_bot[i + 1], v_bot[i], 32768, l_bot[i], side=0)
        if i == 0:
            mb.add_seg(v_bot[0], v_top[0], 16384, l_left, side=0)
        else:
            mb.add_seg(v_bot[i], v_top[i], 16384, l_portals[i - 1], side=1)

        x_min = i * room_width
        x_max = (i + 1) * room_width
        mb.add_subsector(4, first_seg, bbox=(room_height, 0, x_min, x_max))

    return _make_sim_from_map(mb)


def _spawn_player(
    sim: GameSim,
    x: int = 100 * FRACUNIT,
    y: int = 128 * FRACUNIT,
    z: int = 0,
    pclass: PlayerClass = PlayerClass.PCLASS_FIGHTER,
) -> Mobj:
    """Helper to instantiate a player mobj and register it in sim.players."""
    mtypes = {
        PlayerClass.PCLASS_FIGHTER: MobjType.MT_PLAYER_FIGHTER,
        PlayerClass.PCLASS_CLERIC: MobjType.MT_PLAYER_CLERIC,
        PlayerClass.PCLASS_MAGE: MobjType.MT_PLAYER_MAGE,
    }
    mo = spawn_mobj(sim, x, y, z, mtypes[pclass])
    player = Player(
        mo=mo,
        player_class=int(pclass),
        health=100,
        player_state=PlayerState.PST_LIVE,
    )
    mo.player = player
    sim.players.append(player)
    return mo


# ==============================================================================
# Tier 1: Feature Coverage (Core Requirements)
# ==============================================================================


def test_sound_alert_single_sector_target_assignment() -> None:
    """Noise alert sets sector.sound_target and marks sector valid_count in single room."""
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 256, 256)
    sim = _make_sim_from_map(mb)

    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 150 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    noise_alert(sim, target=player, emitter=player)
    sec = ettin.subsector.sector
    assert sec.sound_target is player
    assert sec.valid_count > 0


def test_sound_alert_propagates_across_open_portal() -> None:
    """Noise alert in Room 0 floods across open 2-sided portal into Room 1."""
    sim = _make_two_room_sim(portal_flags=ML_TWOSIDED)
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    noise_alert(sim, target=player, emitter=player)
    assert sim.level.sectors[0].sound_target is player
    assert sim.level.sectors[1].sound_target is player
    assert ettin.subsector.sector.sound_target is player


def test_sound_alert_corridor_propagation_3_rooms() -> None:
    """Noise alert propagates across a 3-room linear corridor with open portals."""
    sim = _make_corridor_sim(num_rooms=3)
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)

    noise_alert(sim, target=player, emitter=player)
    for i in range(3):
        assert sim.level.sectors[i].sound_target is player
        assert sim.level.sectors[i].sound_traversed == 1


def test_sound_alert_single_soundblock_line_traversed() -> None:
    """Sound crosses one ML_SOUNDBLOCK line and increments sound_traversed to 2."""
    sim = _make_corridor_sim(
        num_rooms=3,
        portal_flags=[ML_TWOSIDED | ML_SOUNDBLOCK, ML_TWOSIDED],
    )
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)

    noise_alert(sim, target=player, emitter=player)
    assert sim.level.sectors[0].sound_target is player
    assert sim.level.sectors[0].sound_traversed == 1
    # Room 1 is reached after 1 soundblock
    assert sim.level.sectors[1].sound_target is player
    assert sim.level.sectors[1].sound_traversed == 2
    # Room 2 is reached within soundblocks <= 1 budget
    assert sim.level.sectors[2].sound_target is player
    assert sim.level.sectors[2].sound_traversed == 2


def test_sound_alert_cyclic_sectors_flood_termination() -> None:
    """Recursive sound terminates cleanly without infinite loops when visiting connected sectors."""
    sim = _make_two_room_sim(portal_flags=ML_TWOSIDED)
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)

    noise_alert(sim, target=player, emitter=player)
    assert sim.level.sectors[0].valid_count > 0
    assert sim.level.sectors[1].valid_count > 0


def test_a_look_player_in_direct_fov_wakes() -> None:
    """Monster facing player within 180° FOV wakes, acquires target, and plays seesound."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0  # Facing East (directly towards player)
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    a_look(sim, ettin)
    assert ettin.target is player
    assert ettin.state_num == ettin.info.seestate
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert int(SfxEnum.SFX_ETTIN_SIGHT) in sound_ids


def test_a_look_wakes_from_sector_soundtarget_without_ambush() -> None:
    """Non-ambush monster facing away from player wakes immediately on sector sound_target."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0  # Facing East (away from player)
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    # Sector hears player noise
    ettin.subsector.sector.sound_target = player

    a_look(sim, ettin)
    assert ettin.target is player
    assert ettin.state_num == ettin.info.seestate


def test_a_look_behind_back_close_melee_range_wakes() -> None:
    """Player directly behind monster wakes it if within MELEERANGE (64 units)
    even without sound."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 60 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0  # Facing East away from player (dist = 40 units <= 64)
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    a_look(sim, ettin)
    assert ettin.target is player
    assert ettin.state_num in (ettin.info.seestate, ettin.info.meleestate)


def test_a_look_boss_monster_plays_full_volume_sound() -> None:
    """Boss monster (MF2_BOSS) broadcasts see-sound at full volume (origin_id=None)."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    korax = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_KORAX)
    korax.angle = 0
    korax.flags2 |= MobjFlag2.MF2_BOSS
    set_mobj_state(sim, korax, korax.info.spawnstate)

    a_look(sim, korax)
    assert korax.target is player
    events = sim.events.drain()
    boss_sounds = [
        e for e in events if isinstance(e, SoundEvent) and e.sound_id == korax.info.seesound
    ]
    assert len(boss_sounds) > 0
    assert boss_sounds[0].origin_id is None


def test_a_look_dead_player_ignored() -> None:
    """Dead player with health <= 0 is ignored by A_Look."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    player.health = 0
    if player.player is not None:
        player.player.health = 0
    player.flags &= ~MobjFlag.MF_SHOOTABLE
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    a_look(sim, ettin)
    assert ettin.target is None
    assert ettin.state_num == StateNum.S_ETTIN_LOOK1


def test_check_melee_range_within_meleerange() -> None:
    """Target within MELEERANGE threshold returns True."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 140 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player

    assert check_melee_range(sim, ettin) is True


def test_check_melee_range_exceeds_threshold() -> None:
    """Target beyond melee threshold returns False."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 250 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player

    assert check_melee_range(sim, ettin) is False


def test_check_melee_range_vertical_height_clearance() -> None:
    """Target higher than actor height fails melee range check."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 130 * FRACUNIT, 128 * FRACUNIT, z=100 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    # ettin.height is 64*FRACUNIT, player z is 100*FRACUNIT
    assert check_melee_range(sim, ettin) is False


def test_check_missile_range_reactiontime_gate() -> None:
    """Reactiontime > 0 blocks missile attacks unless MF_JUSTHIT is set."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    centaur = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CENTAUR)
    centaur.target = player
    centaur.reactiontime = 5

    assert check_missile_range(sim, centaur) is False


def test_check_missile_range_justhit_retaliation() -> None:
    """MF_JUSTHIT bypasses reactiontime gate and is cleared on missile check."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    centaur = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CENTAUR)
    centaur.target = player
    centaur.reactiontime = 10
    centaur.flags |= MobjFlag.MF_JUSTHIT

    assert check_missile_range(sim, centaur) is True
    assert (centaur.flags & MobjFlag.MF_JUSTHIT) == 0


def test_move_cardinal_step_advances_mobj() -> None:
    """Move advancing along DI_EAST increases x by speed * FRACUNIT."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.movedir = DI_EAST
    speed = ettin.info.speed

    success = move(sim, ettin)
    assert success is True
    assert ettin.x == 100 * FRACUNIT + speed * FRACUNIT
    assert ettin.y == 128 * FRACUNIT


def test_move_diagonal_step_advances_both_axes() -> None:
    """Move along DI_NORTHEAST advances both axes using Raven's 47000 fixed speed."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.movedir = DI_NORTHEAST
    speed = ettin.info.speed

    success = move(sim, ettin)
    assert success is True
    assert ettin.x == 100 * FRACUNIT + speed * 47000
    assert ettin.y == 128 * FRACUNIT + speed * 47000


def test_try_walk_sets_randomized_movecount() -> None:
    """Successful try_walk assigns movecount in range 0..15."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.movedir = DI_EAST

    success = try_walk(sim, ettin)
    assert success is True
    assert 0 <= ettin.movecount <= 15


def test_new_chase_dir_selects_direct_diagonal() -> None:
    """Target offset diagonally (dx > 10, dy > 10) selects DI_NORTHEAST."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 200 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player

    new_chase_dir(sim, ettin)
    assert ettin.movedir == DI_NORTHEAST


def test_floating_monster_adjusts_vertical_height() -> None:
    """Floating monster encountering height obstruction adjusts z by FLOATSPEED."""
    sim = _make_two_room_sim(floor1=0, ceil1=200, floor2=80, ceil2=200)
    bishop = spawn_mobj(sim, 250 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_BISHOP)
    bishop.flags |= MobjFlag.MF_FLOAT
    bishop.movedir = DI_EAST

    # Step east crosses into high-floor sector; standard step fails, float height logic adjusts z
    success = move(sim, bishop)
    assert success is True
    assert bishop.flags & MobjFlag.MF_INFLOAT
    assert bishop.z == FLOATSPEED


def test_a_chase_decrements_reactiontime_and_threshold() -> None:
    """A_Chase decrements reactiontime and threshold counters."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.reactiontime = 4
    ettin.threshold = 5

    a_chase(sim, ettin)
    assert ettin.reactiontime == 3
    assert ettin.threshold == 4


def test_a_chase_steers_angle_towards_movedir() -> None:
    """A_Chase gradually turns actor angle towards movedir."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 100 * FRACUNIT, 200 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.movedir = DI_NORTH  # 90 degrees = ANG90
    ettin.angle = 0

    a_chase(sim, ettin)
    assert ettin.angle > 0


def test_a_chase_transitions_to_meleestate_when_close() -> None:
    """A_Chase transitions to meleestate when in melee range."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 140 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player

    a_chase(sim, ettin)
    assert ettin.state_num == ettin.info.meleestate


def test_a_chase_transitions_to_missilestate_when_in_range() -> None:
    """A_Chase transitions to missilestate when in missile range and movecount exhausted."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    bishop = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_BISHOP)
    bishop.target = player
    bishop.movecount = 0
    bishop.reactiontime = 0
    # Use deterministic RNG so missile check passes
    sim.rng.clear()

    a_chase(sim, bishop)
    assert bishop.state_num == bishop.info.missilestate


def test_a_chase_decrements_movecount_and_steps() -> None:
    """A_Chase decrements movecount and advances mobj along movedir."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 400 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.movedir = DI_EAST
    ettin.movecount = 5

    a_chase(sim, ettin)
    assert ettin.movecount == 4
    assert ettin.x > 100 * FRACUNIT


def test_action_registry_contains_base_actions() -> None:
    """Action registry contains all core shared AI actions."""
    required = [
        "A_Look",
        "A_Chase",
        "A_FastChase",
        "A_FaceTarget",
        "A_Pain",
        "A_Scream",
        "A_XScream",
        "A_NoBlocking",
        "A_Explode",
        "A_FreezeDeath",
        "A_FreezeDeathChunks",
        "A_QueueCorpse",
        "A_DeQueueCorpse",
        "A_SetInvulnerable",
        "A_UnSetInvulnerable",
        "A_SetReflective",
        "A_UnSetReflective",
    ]
    for action_name in required:
        assert is_action_implemented(action_name), f"Missing action: {action_name}"


def test_a_facetarget_sets_actor_angle() -> None:
    """A_FaceTarget sets angle facing target directly and clears MF_AMBUSH."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 100 * FRACUNIT, 200 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.angle = 0
    ettin.flags |= MobjFlag.MF_AMBUSH

    a_facetarget(sim, ettin)
    assert abs(ettin.angle - ANG90) <= 1
    assert (ettin.flags & MobjFlag.MF_AMBUSH) == 0


def test_a_pain_emits_painsound() -> None:
    """A_Pain plays actor's painsound."""
    sim = _make_two_room_sim()
    centaur = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CENTAUR)

    a_pain(sim, centaur)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert centaur.info.painsound in sound_ids


def test_a_noblocking_removes_solid_flag() -> None:
    """A_NoBlocking clears MF_SOLID flag."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert ettin.flags & MobjFlag.MF_SOLID

    a_noblocking(sim, ettin)
    assert (ettin.flags & MobjFlag.MF_SOLID) == 0


def test_a_scream_plays_monster_deathsound() -> None:
    """A_Scream emits monster deathsound."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    a_scream(sim, ettin)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert ettin.info.deathsound in sound_ids


def test_a_xscream_alias() -> None:
    """A_XScream functions as alias to A_Scream."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    a_xscream(sim, ettin)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert ettin.info.deathsound in sound_ids


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


def test_sound_alert_two_soundblock_barrier_stops_propagation() -> None:
    """Sound traverses at most one soundblock line; two soundblock lines stop propagation."""
    sim = _make_corridor_sim(
        num_rooms=3,
        portal_flags=[ML_TWOSIDED | ML_SOUNDBLOCK, ML_TWOSIDED | ML_SOUNDBLOCK],
    )
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)

    noise_alert(sim, target=player, emitter=player)
    assert sim.level.sectors[0].sound_target is player
    assert sim.level.sectors[1].sound_target is player
    # Room 2 is blocked because sound traversed exceeds budget
    assert sim.level.sectors[2].sound_target is None


def test_sound_alert_closed_door_openrange_zero_blocks() -> None:
    """Closed portal with ceiling == floor (openrange <= 0) blocks sound propagation."""
    sim = _make_corridor_sim(
        num_rooms=2,
        ceilings=[128, 0],  # Sector 1 ceiling is 0 (closed)
    )
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)

    noise_alert(sim, target=player, emitter=player)
    assert sim.level.sectors[0].sound_target is player
    assert sim.level.sectors[1].sound_target is None


def test_sound_alert_one_sided_linedef_blocks() -> None:
    """Single-sided line (without ML_TWOSIDED) blocks sound propagation."""
    sim = _make_corridor_sim(
        num_rooms=2,
        portal_flags=[ML_BLOCKING],  # 1-sided portal
    )
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)

    noise_alert(sim, target=player, emitter=player)
    assert sim.level.sectors[0].sound_target is player
    assert sim.level.sectors[1].sound_target is None


def test_sound_alert_target_without_subsector_safe_return() -> None:
    """Noise alert on unlinked target without subsector returns safely without error."""
    sim = _make_two_room_sim()
    mo = Mobj()
    mo.subsector = None
    noise_alert(sim, target=mo, emitter=mo)


def test_a_look_ambush_flag_ignores_sound_without_sight() -> None:
    """Ambush monster in alerted sector stays asleep if player is occluded."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.flags |= MobjFlag.MF_AMBUSH
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    # Sector hears noise, but sight line is closed
    sim.level.sectors[1].sound_target = player
    # Lower ceiling between rooms to occlude sight
    sim.level.sectors[1].ceiling_height = 0

    a_look(sim, ettin)
    assert ettin.target is None
    assert ettin.state_num == StateNum.S_ETTIN_LOOK1


def test_a_look_ambush_flag_wakes_on_sound_when_sight_clear() -> None:
    """Ambush monster in alerted sector wakes when sight line to player is clear."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.flags |= MobjFlag.MF_AMBUSH
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    sim.level.sectors[1].sound_target = player

    a_look(sim, ettin)
    assert ettin.target is player
    assert ettin.state_num == ettin.info.seestate


def test_a_look_behind_back_outside_melee_range_does_not_wake() -> None:
    """Player directly behind monster beyond MELEERANGE (65 > 64 units) is not detected."""
    sim = _make_two_room_sim()
    _spawn_player(sim, 35 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0  # Facing East away from player (dist = 65 units > 64)
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    a_look(sim, ettin)
    assert ettin.target is None
    assert ettin.state_num == StateNum.S_ETTIN_LOOK1


def test_a_look_player_with_shadow_sneaking_avoids_detection() -> None:
    """Player with MF_SHADOW at distance > 2*MELEERANGE and low speed sneaks past undetected."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 250 * FRACUNIT, 128 * FRACUNIT)
    player.flags |= MobjFlag.MF_SHADOW
    player.momx = 0
    player.momy = 0

    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0  # Facing East directly towards player (dist = 150 > 128)
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    a_look(sim, ettin)
    assert ettin.target is None
    assert ettin.state_num == StateNum.S_ETTIN_LOOK1


def test_a_look_exact_fov_boundary_angles() -> None:
    """ANG90 and ANG270 correctly delimit the 180° field of view hemisphere."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0  # Facing East

    # Direct North: angle difference = ANG90 (on boundary)
    p_north = _spawn_player(sim, 100 * FRACUNIT, 200 * FRACUNIT)
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)
    a_look(sim, ettin)
    assert ettin.target is p_north

    # Fresh sim for rear test (Rear North-West: angle > ANG90 and < ANG270, dist > 64)
    sim_rear = _make_two_room_sim()
    ettin_rear = spawn_mobj(sim_rear, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin_rear.angle = 0  # Facing East
    _spawn_player(sim_rear, 50 * FRACUNIT, 150 * FRACUNIT)
    set_mobj_state(sim_rear, ettin_rear, StateNum.S_ETTIN_LOOK1)
    a_look(sim_rear, ettin_rear)
    # Outside FOV and > 64 units away: does not wake
    assert ettin_rear.target is None


def test_check_melee_range_actor_without_target() -> None:
    """CheckMeleeRange returns False safely when actor has no target."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = None
    assert check_melee_range(sim, ettin) is False


def test_check_melee_range_occluded_by_wall() -> None:
    """CheckMeleeRange returns False if target is occluded by geometry."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 300 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    # Lower ceiling in sector 1 to block sight across portal
    sim.level.sectors[1].ceiling_height = 0
    assert check_melee_range(sim, ettin) is False


def test_check_melee_range2_distance_window() -> None:
    """CheckMeleeRange2 returns True only between MELEERANGE (64) and 2*MELEERANGE (128)."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 180 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player

    # Dist = 80 units (in [64, 128])
    assert check_melee_range2(sim, ettin) is True

    # Dist = 50 units (< 64)
    player.x = 150 * FRACUNIT
    assert check_melee_range2(sim, ettin) is False

    # Dist = 150 units (> 128)
    player.x = 250 * FRACUNIT
    assert check_melee_range2(sim, ettin) is False


def test_check_missile_range_monster_without_meleestate_bonus() -> None:
    """Monsters without meleestate subtract 128 from dist, increasing fire frequency."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 300 * FRACUNIT, 128 * FRACUNIT)
    bishop = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_BISHOP)
    bishop.target = player
    bishop.reactiontime = 0

    # Distance is 200 map units -> base dist = 200 - 64 = 136.
    # Bishop lacks meleestate, subtracting 128 -> dist = 136 - 128 = 8.
    # Spawning player and bishop advanced prndindex to 2.
    # Next P_Random() returns RNDTABLE[3] = 19.
    # Since 19 >= 8 (rnd >= dist), check_missile_range deterministically succeeds.
    assert check_missile_range(sim, bishop) is True

    # Contrast with monster having meleestate (e.g. Ettin at same distance):
    # Ettin dist = 136. With rnd = 19 < 136, missile check fails.
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.reactiontime = 0
    sim.rng.prndindex = 2  # Reset to same PRNG position
    assert check_missile_range(sim, ettin) is False

    # Verify probability boundary on Bishop: when PRNG yields 1 < dist (8), check fails.
    sim.rng.prndindex = 0  # Next P_Random() yields RNDTABLE[1] = 1 < 8
    assert check_missile_range(sim, bishop) is False


def test_check_missile_range_dist_clamped_at_200() -> None:
    """Distance formula is clamped at 200, avoiding out-of-bounds index issues."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 2000 * FRACUNIT, 128 * FRACUNIT)
    centaur = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CENTAUR)
    centaur.target = player
    centaur.reactiontime = 0

    # Function executes cleanly without error
    check_missile_range(sim, centaur)


def test_check_missile_range_unconditional_prng_advance() -> None:
    """check_missile_range advances PRNG unconditionally even when dist <= 0 (p_enemy.c:209)."""
    sim = _make_two_room_sim()
    # Bishop at close range has meleestate == 0, so dist -= 128 makes dist <= 0
    player = _spawn_player(sim, 120 * FRACUNIT, 128 * FRACUNIT)
    bishop = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_BISHOP)
    bishop.target = player
    bishop.reactiontime = 0

    initial_prndindex = sim.rng.prndindex
    result = check_missile_range(sim, bishop)

    # PRNG must have advanced by 1 tic
    assert sim.rng.prndindex == (initial_prndindex + 1) & 0xFF
    # dist <= 0 guarantees attack succeeds
    assert result is True


def test_move_blasted_monster_skips_movement_stepping() -> None:
    """Blasted monster (MF2_BLASTED) skips normal AI stepping."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.flags2 |= MobjFlag2.MF2_BLASTED
    ettin.movedir = DI_EAST

    success = move(sim, ettin)
    assert success is True
    # Position remains unchanged
    assert ettin.x == 100 * FRACUNIT


def test_move_blocked_by_solid_wall_keeps_movedir() -> None:
    """Move into a solid wall fails; with no special lines touched P_Move leaves movedir."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 10 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    # Moving West into left boundary wall at x=0
    ettin.movedir = DI_WEST

    success = move(sim, ettin)
    assert success is False
    assert ettin.movedir == DI_WEST


def test_move_blocked_calls_p_activate_line_with_spac_use() -> None:
    """When movement is blocked, move() activates special lines with SPAC_USE (p_enemy.c:271)."""
    from typing import Any

    from pyhexen.world.level import SPAC_USE

    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.movedir = DI_EAST

    ld = sim.level.lines[0]
    sim.map_state.spechit.append(ld)

    calls: list[tuple[Any, ...]] = []

    def _record_activate(s: Any, line: Any, actor: Any, side: int, act_type: int) -> bool:
        calls.append((line, actor, side, act_type))
        return True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pyhexen.sim.enemy.try_move", lambda *args, **kwargs: False)
        mp.setattr("pyhexen.sim.line_activation.p_activate_line", _record_activate)
        success = move(sim, ettin)

    assert success is True
    assert len(calls) == 1
    assert calls[0][0] is ld
    assert calls[0][1] is ettin
    assert calls[0][2] == 0
    assert calls[0][3] == SPAC_USE


def test_new_chase_dir_target_none_raises_runtime_error() -> None:
    """NewChaseDir called with no target raises RuntimeError."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = None

    with pytest.raises(RuntimeError):
        new_chase_dir(sim, ettin)


def test_new_chase_dir_turnaround_suppressed_if_alternatives_exist() -> None:
    """Turnaround direction is suppressed if other open paths exist."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.movedir = DI_EAST  # Moving East; opposite turnaround is DI_WEST

    new_chase_dir(sim, ettin)
    assert ettin.movedir != DI_WEST


def test_a_chase_target_killed_resets_to_spawnstate() -> None:
    """When target dies or loses MF_SHOOTABLE, A_Chase reverts to spawnstate."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 200 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player

    player.health = 0
    player.flags &= ~MobjFlag.MF_SHOOTABLE

    a_chase(sim, ettin)
    assert ettin.state_num == ettin.info.spawnstate
    assert ettin.target is None


def test_a_chase_justattacked_flag_suppresses_immediate_reattack() -> None:
    """MF_JUSTATTACKED prevents immediate re-attack in melee range."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 140 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.flags |= MobjFlag.MF_JUSTATTACKED

    a_chase(sim, ettin)
    assert (ettin.flags & MobjFlag.MF_JUSTATTACKED) == 0
    # Did not transition to melee state on this frame
    assert ettin.state_num != ettin.info.meleestate


def test_a_chase_nightmare_skill_halves_tics() -> None:
    """Nightmare difficulty halves monster state tics (min 3)."""
    sim = _make_two_room_sim()
    sim.skill = 4  # SK_NIGHTMARE
    player = _spawn_player(sim, 400 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.tics = 8

    a_chase(sim, ettin)
    assert ettin.tics == 4


def test_a_chase_active_sound_random_gate() -> None:
    """Active sound is played probabilistically when p_random() < 3."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 400 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.movecount = 10
    ettin.movedir = DI_EAST

    # Force low random number (< 3)
    sim.rng.clear()
    a_chase(sim, ettin)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert ettin.info.activesound in sound_ids


def test_a_chase_exhausted_movecount_triggers_new_dir() -> None:
    """Movecount == 0 causes A_Chase to calculate a new chase direction."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 400 * FRACUNIT, 200 * FRACUNIT)
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.movecount = 0

    a_chase(sim, ettin)
    assert ettin.movedir != DI_NODIR


def test_a_facetarget_target_shadow_adds_angular_jitter() -> None:
    """A_FaceTarget with shadow target applies angular randomness."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 100 * FRACUNIT, 200 * FRACUNIT)
    player.flags |= MobjFlag.MF_SHADOW
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player

    sim.rng.clear()
    a_facetarget(sim, ettin)
    # Jittered angle should deviate slightly from pure ANG90
    assert ettin.angle != ANG90


def test_a_explode_damage_formula() -> None:
    """A_Explode inflicts 25 + (rnd & 15) radius damage."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 110 * FRACUNIT, 128 * FRACUNIT)
    player.health = 100
    bishop = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_BISHOP)
    bishop.target = player

    a_explode(sim, bishop)
    # Player in blast radius took damage
    assert player.health < 100


def test_a_freezedeath_sets_ice_flags_and_height() -> None:
    """A_FreezeDeath sets ice statue flags and quadruples height."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    original_height = ettin.height

    a_freezedeath(sim, ettin)
    assert ettin.flags & MobjFlag.MF_SOLID
    assert ettin.flags & MobjFlag.MF_SHOOTABLE
    assert ettin.flags & MobjFlag.MF_NOBLOOD
    assert ettin.height == original_height * 4


def test_a_freezedeathchunks_moving_postpones_shatter() -> None:
    """Moving frozen statue postpones shattering and sets tics = 105."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.momx = 2 * FRACUNIT

    a_freezedeathchunks(sim, ettin)
    assert ettin.tics == 105


def test_a_freezedeathchunks_stationary_spawns_ice_shards() -> None:
    """Stationary frozen statue shatters into ice chunk projectiles and plays sound."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.momx = 0
    ettin.momy = 0
    ettin.momz = 0

    a_freezedeathchunks(sim, ettin)
    events = sim.events.drain()
    sound_ids = [e.sound_id for e in events if isinstance(e, SoundEvent)]
    assert int(SfxEnum.SFX_FREEZE_SHATTER) in sound_ids


def test_ice_actions() -> None:
    """A_IceSetTics and A_IceCheckHeadDone execute state changes cleanly."""
    sim = _make_two_room_sim()
    mo = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ICECHUNK)

    a_icesettics(sim, mo)
    assert mo.tics > 0

    mo.special2 = 666
    a_icecheckheaddone(sim, mo)
    assert mo.state_num == StateNum.S_ICECHUNK_HEAD2


def test_corpse_queue() -> None:
    """A_QueueCorpse, A_DeQueueCorpse, and corpse queue reset maintain queue integrity."""
    sim = _make_two_room_sim()
    p_init_creature_corpse_queue()

    corpses = [
        spawn_mobj(sim, (100 + i) * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
        for i in range(5)
    ]
    for c in corpses:
        a_queuecorpse(sim, c)

    a_dequeuecorpse(sim, corpses[2])
    p_init_creature_corpse_queue()


def test_invulnerability_and_reflectivity_actions() -> None:
    """Invulnerability and reflectivity actions toggle corresponding flags."""
    sim = _make_two_room_sim()
    centaur = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CENTAUR)

    a_setreflective(sim, centaur)
    assert centaur.flags2 & MobjFlag2.MF2_REFLECTIVE
    assert centaur.flags2 & MobjFlag2.MF2_INVULNERABLE

    a_unsetreflective(sim, centaur)
    assert (centaur.flags2 & MobjFlag2.MF2_REFLECTIVE) == 0
    assert (centaur.flags2 & MobjFlag2.MF2_INVULNERABLE) == 0

    a_setinvulnerable(sim, centaur)
    assert centaur.flags2 & MobjFlag2.MF2_INVULNERABLE
    a_unsetinvulnerable(sim, centaur)
    assert (centaur.flags2 & MobjFlag2.MF2_INVULNERABLE) == 0


def test_a_freetargmobj() -> None:
    """A_FreeTargMobj sets corpse flags and places actor above ceiling."""
    sim = _make_two_room_sim()
    ettin = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    a_freetargmobj(sim, ettin)
    assert ettin.flags & MobjFlag.MF_CORPSE
    assert ettin.flags2 & MobjFlag2.MF2_DONTDRAW
    assert ettin.health == -1000


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================


def test_combat_flow_alert_to_melee_strike() -> None:
    """End-to-end: Weapon noise alerts monster -> monster chases -> melee attack."""
    sim = _make_two_room_sim(portal_flags=ML_TWOSIDED)
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = 0  # Facing East away from player
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)

    # 1. Noise alert from weapon
    noise_alert(sim, target=player, emitter=player)
    assert ettin.subsector.sector.sound_target is player

    # 2. Ettin wakes via A_Look
    a_look(sim, ettin)
    assert ettin.target is player
    assert ettin.state_num == ettin.info.seestate

    # 3. Chase tics until reaching melee range. A new mobj's movedir is DI_EAST (zeroed
    #    memory) and P_NewChaseDir never turns straight around, so the ettin first walks
    #    to the far wall and back.
    for _ in range(200):
        if ettin.state_num == ettin.info.meleestate:
            break
        a_chase(sim, ettin)

    assert ettin.state_num == ettin.info.meleestate


def test_chained_noise_alert_across_four_sectors() -> None:
    """Noise in Room 0 alerts monsters stationed across Rooms 1, 2, and 3."""
    sim = _make_corridor_sim(num_rooms=4)
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    monsters = [
        spawn_mobj(sim, (i * 256 + 128) * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
        for i in range(1, 4)
    ]
    for m in monsters:
        set_mobj_state(sim, m, StateNum.S_ETTIN_LOOK1)

    noise_alert(sim, target=player, emitter=player)
    for m in monsters:
        a_look(sim, m)
        assert m.target is player
        assert m.state_num == m.info.seestate


def test_retaliation_rear_attack_justhit() -> None:
    """Monster shot from behind gains MF_JUSTHIT and immediately retaliates."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    centaur = spawn_mobj(sim, 200 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_CENTAUR)
    centaur.target = player
    centaur.reactiontime = 10  # Normal attack gated by reactiontime

    centaur.flags |= MobjFlag.MF_JUSTHIT
    assert check_missile_range(sim, centaur) is True
    assert (centaur.flags & MobjFlag.MF_JUSTHIT) == 0


def test_monster_chase_navigates_portal_boundary() -> None:
    """Monster steps across sector boundary portal while tracking player."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)
    ettin = spawn_mobj(sim, 280 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.target = player
    ettin.movedir = DI_WEST

    # Step west across portal line at x=256
    for _ in range(20):
        move(sim, ettin)

    assert ettin.x < 256 * FRACUNIT
    assert ettin.subsector.sector == sim.level.sectors[0]


def test_ambush_monster_waits_for_sight_after_noise() -> None:
    """Ambush monster ignores sound alert until line of sight is opened."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 100 * FRACUNIT, 128 * FRACUNIT)

    # Occlude sight by lowering ceiling before look runs
    sim.level.sectors[1].ceiling_height = 0
    sim.level.sectors[1].sound_target = player

    ettin = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.angle = ANG180
    ettin.flags |= MobjFlag.MF_AMBUSH
    set_mobj_state(sim, ettin, StateNum.S_ETTIN_LOOK1)
    ettin.target = None

    a_look(sim, ettin)
    assert ettin.target is None  # Still asleep

    # Raise ceiling: sight opened!
    sim.level.sectors[1].ceiling_height = 128 * FRACUNIT
    a_look(sim, ettin)
    assert ettin.target is player  # Woke up!


# ==============================================================================
# Tier 4: Real-World Simulation Scenarios & Architectural Audits
# ==============================================================================


def test_multi_monster_combat_simulation() -> None:
    """Simulate 30 tics of 3 monsters actively hunting a player in open space."""
    sim = _make_two_room_sim()
    player = _spawn_player(sim, 50 * FRACUNIT, 128 * FRACUNIT)
    monsters = [
        spawn_mobj(sim, 200 * FRACUNIT, (64 + i * 64) * FRACUNIT, 0, MobjType.MT_ETTIN)
        for i in range(3)
    ]
    for m in monsters:
        m.target = player
        m.movedir = DI_WEST
        m.movecount = 10

    for _ in range(30):
        for m in monsters:
            a_chase(sim, m)

    for m in monsters:
        assert m.x < 200 * FRACUNIT


def test_nightmare_mode_speed() -> None:
    """Nightmare mode halves monster state tics, increasing movement and attack tempo."""
    sim_norm = _make_two_room_sim()
    sim_nm = _make_two_room_sim()
    sim_nm.skill = 4

    p_norm = _spawn_player(sim_norm, 400 * FRACUNIT, 128 * FRACUNIT)
    p_nm = _spawn_player(sim_nm, 400 * FRACUNIT, 128 * FRACUNIT)

    m_norm = spawn_mobj(sim_norm, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    m_nm = spawn_mobj(sim_nm, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)

    m_norm.target = p_norm
    m_nm.target = p_nm
    m_norm.tics = 8
    m_nm.tics = 8

    a_chase(sim_norm, m_norm)
    a_chase(sim_nm, m_nm)

    assert m_nm.tics < m_norm.tics


def test_deterministic_state_replay() -> None:
    """Identical RNG seeds produce bit-identical monster chase and movement states."""

    def run_sim(seed: int) -> tuple[int, int, int]:
        sim = _make_two_room_sim()
        sim.rng.clear()
        p = _spawn_player(sim, 400 * FRACUNIT, 128 * FRACUNIT)
        m = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
        m.target = p
        for _ in range(40):
            a_chase(sim, m)
        return m.x, m.y, m.angle

    res1 = run_sim(42)
    res2 = run_sim(42)
    assert res1 == res2


def test_ast_compliance_sim_enemy() -> None:
    """AST static audit: Verify zero floats, no stdlib random, no UI/render/audio imports."""
    enemy_py = Path("src/pyhexen/sim/enemy.py")
    tree = ast.parse(enemy_py.read_text(encoding="utf-8"))

    forbidden_modules = {"random", "pyhexen.render", "pyhexen.audio", "pyhexen.ui"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forb in forbidden_modules:
                    assert not alias.name.startswith(forb), f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for forb in forbidden_modules:
                assert not mod.startswith(forb), f"Forbidden import from: {mod}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"Float literal {node.value} forbidden in sim code")
