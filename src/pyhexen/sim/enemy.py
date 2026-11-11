# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Common enemy AI, sound propagation, and core monster actions.

Port of p_enemy.c from Chocolate Hexen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import (
    ANG90,
    ANG270,
    ANGLETOFINESHIFT,
    approx_distance,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACBITS, FRACUNIT, fixed_div, fixed_mul, to_int32, to_uint32
from pyhexen.sim.actions import action
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType, SfxEnum, StateNum
from pyhexen.sim.map import try_move
from pyhexen.sim.maputl import line_opening
from pyhexen.sim.mobj import Mobj, set_mobj_state
from pyhexen.sim.sight import check_sight
from pyhexen.sim.sound import start_sound as _emit_sound_event
from pyhexen.sim.sound import stop_sound as _stop_sound
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.world.level import ML_SOUNDBLOCK, ML_TWOSIDED, SPAC_USE, Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

# 8 cardinal/diagonal directions and no-direction
DI_EAST: int = 0
DI_NORTHEAST: int = 1
DI_NORTH: int = 2
DI_NORTHWEST: int = 3
DI_WEST: int = 4
DI_SOUTHWEST: int = 5
DI_SOUTH: int = 6
DI_SOUTHEAST: int = 7
DI_NODIR: int = 8

# Fixed-point 16.16 step velocity multipliers (Raven's 47000 for diagonals)
XSPEED: tuple[int, ...] = (FRACUNIT, 47000, 0, -47000, -FRACUNIT, -47000, 0, 47000)
YSPEED: tuple[int, ...] = (0, 47000, FRACUNIT, 47000, 0, -47000, -FRACUNIT, -47000)

OPPOSITE: tuple[int, ...] = (
    DI_WEST,
    DI_SOUTHWEST,
    DI_SOUTH,
    DI_SOUTHEAST,
    DI_EAST,
    DI_NORTHEAST,
    DI_NORTH,
    DI_NORTHWEST,
    DI_NODIR,
)

DIAGS: tuple[int, ...] = (
    DI_NORTHWEST,
    DI_NORTHEAST,
    DI_SOUTHWEST,
    DI_SOUTHEAST,
)

MELEERANGE: int = 64 * FRACUNIT
FLOATSPEED: int = 4 * FRACUNIT
MONS_LOOK_RANGE: int = 16 * 64 * FRACUNIT
MONS_LOOK_LIMIT: int = 64
CORPSEQUEUESIZE: int = 64
CLASS_BOSS_STRAFE_RANGE: int = 64 * 10 * FRACUNIT

_corpse_queue: list[Mobj | None] = [None] * CORPSEQUEUESIZE
_corpse_queue_slot: int = 0
_sound_valid_count: int = 0


def _is_sim(obj: Any) -> bool:
    """Return True if obj is a GameSim instance without runtime type import."""
    return obj is not None and type(obj).__name__ == "GameSim"


# =============================================================================
# Sound Emission Helpers
# =============================================================================


def start_sound(
    sim: GameSim | None,
    actor: Mobj | None,
    sound_id: int | SfxEnum | None,
    *,
    full_volume: bool = False,
) -> None:
    """Emit a sound event through the simulation event bus."""
    if sim is None or sound_id is None:
        return
    s_id = int(sound_id)
    if s_id == int(SfxEnum.SFX_NONE):
        return
    _emit_sound_event(sim, None if (full_volume or actor is None) else actor, s_id)


def stop_sound(sim: GameSim | None, actor: Mobj | None) -> None:
    """Emit a sound stop event through the simulation event bus."""
    if sim is None or actor is None:
        return
    _stop_sound(sim, actor)


# =============================================================================
# Sound Propagation (R1)
# =============================================================================


def recursive_sound(
    sec: Sector,
    soundblocks: int,
    sound_target: Mobj,
    valid_count: int,
    sim: GameSim | None = None,
) -> None:
    """Flood connected sectors with sound alert.

    Port of P_RecursiveSound (p_enemy.c).
    """
    if sec.valid_count == valid_count and sec.sound_traversed <= soundblocks + 1:
        return

    sec.valid_count = valid_count
    sec.sound_traversed = soundblocks + 1
    sec.sound_target = sound_target

    for check in sec.lines:
        if not (check.flags & ML_TWOSIDED):
            continue
        opening = line_opening(check)
        if opening.openrange <= 0:
            continue
        other = check.back_sector if check.front_sector is sec else check.front_sector
        if other is None:
            continue
        if check.flags & ML_SOUNDBLOCK:
            if not soundblocks:
                recursive_sound(other, 1, sound_target, valid_count, sim)
        else:
            recursive_sound(other, soundblocks, sound_target, valid_count, sim)


def noise_alert(
    arg1: Any = None,
    arg2: Any = None,
    arg3: Any = None,
    *,
    target: Mobj | None = None,
    emitter: Mobj | None = None,
    sim: GameSim | None = None,
) -> None:
    """Alert monsters in connected sectors to sound emitted by target.

    Port of P_NoiseAlert (p_enemy.c).
    """
    global _sound_valid_count

    # Parameter normalization supporting (target, emitter, sim) or (sim, target, emitter)
    if arg1 is not None and type(arg1).__name__ == "GameSim":
        sim = arg1
        if target is None:
            target = arg2
        if emitter is None:
            emitter = arg3
    elif isinstance(arg1, Mobj):
        if target is None:
            target = arg1
        if emitter is None:
            emitter = arg2
        if sim is None and arg3 is not None and type(arg3).__name__ == "GameSim":
            sim = arg3

    if emitter is None and target is not None:
        emitter = target
    if target is None and emitter is not None:
        target = emitter
    if target is None or emitter is None:
        return
    if emitter.subsector is None or emitter.subsector.sector is None:
        return

    if (
        sim is not None
        and sim.level is not None
        and getattr(sim.level, "blockmap", None) is not None
    ):
        valid_count = sim.level.blockmap.next_valid_count()
    else:
        _sound_valid_count += 1
        valid_count = _sound_valid_count

    recursive_sound(
        emitter.subsector.sector,
        soundblocks=0,
        sound_target=target,
        valid_count=valid_count,
        sim=sim,
    )


# =============================================================================
# Target Acquisition & Senses (R2)
# =============================================================================


def _get_player_candidates(sim: GameSim | None) -> list[tuple[Any, Mobj]]:
    """Return list of (player_or_mobj, mobj) pairs for active players."""
    candidates: list[tuple[Any, Mobj]] = []
    if sim is None:
        return candidates

    if sim.players:
        for p in sim.players:
            if p is not None and p.mo is not None:
                candidates.append((p, p.mo))
        if candidates:
            return candidates

    for th in sim.thinkers:
        if isinstance(th, Mobj):
            if th.player is not None:
                candidates.append((th.player, th))
            elif th.type in (
                MobjType.MT_PLAYER_FIGHTER,
                MobjType.MT_PLAYER_CLERIC,
                MobjType.MT_PLAYER_MAGE,
                MobjType.MT_PIGPLAYER,
            ):
                candidates.append((th, th))
    return candidates


def look_for_monsters(actor: Mobj, sim: GameSim | None = None) -> bool:
    """Scan for other hostile monsters to fight when single player is dead.

    Port of P_LookForMonsters (p_enemy.c).
    """
    if sim is None:
        return False

    candidates = _get_player_candidates(sim)
    if not candidates:
        return False
    _, player_mo = candidates[0]
    if not check_sight(sim, player_mo, actor):
        return False

    count = 0
    for th in sim.thinkers:
        if not isinstance(th, Mobj):
            continue
        mo = th
        if not (mo.flags & MobjFlag.MF_COUNTKILL) or mo is actor or mo.health <= 0:
            continue
        if approx_distance(actor.x - mo.x, actor.y - mo.y) > MONS_LOOK_RANGE:
            continue
        if sim.rng.p_random() < 16:
            continue
        count += 1
        if count > MONS_LOOK_LIMIT:
            return False
        if not check_sight(sim, actor, mo):
            continue
        if actor.type == MobjType.MT_MINOTAUR and mo.type == MobjType.MT_MINOTAUR:
            continue
        actor.target = mo
        return True
    return False


def look_for_players(
    arg1: Any,
    arg2: Any = None,
    arg3: Any = None,
) -> bool:
    """Scan active players within line of sight and 180° FOV.

    Port of P_LookForPlayers (p_enemy.c).
    """
    # Parameter normalization: (actor, allaround=False, sim=None) or (sim, actor, allaround=False)
    if type(arg1).__name__ == "GameSim":
        sim = arg1
        actor = arg2
        allaround = bool(arg3)
    else:
        actor = arg1
        allaround = bool(arg2)
        sim = arg3

    if actor is None:
        return False

    candidates = _get_player_candidates(sim)
    if not candidates:
        return False

    is_netgame = getattr(sim, "netgame", False) if sim is not None else False
    if not is_netgame:
        first_player, first_mo = candidates[0]
        first_health = getattr(first_player, "health", getattr(first_mo, "health", 0))
        if getattr(first_mo, "health", 1) <= 0 or (first_mo.flags & MobjFlag.MF_SHOOTABLE) == 0:
            first_health = 0
        if first_health <= 0:
            return look_for_monsters(actor, sim)

    num_players = len(candidates)
    c = 0
    stop = (actor.lastlook - 1) & 3

    for _ in range(4):
        idx = actor.lastlook
        actor.lastlook = (actor.lastlook + 1) & 3

        if idx >= num_players:
            continue

        c += 1
        p_obj, pmo = candidates[idx]
        health = getattr(p_obj, "health", getattr(pmo, "health", 0))
        if getattr(pmo, "health", 1) <= 0 or (pmo.flags & MobjFlag.MF_SHOOTABLE) == 0:
            health = 0
        if health <= 0:
            if c >= 2 or idx == stop:
                break
            continue

        if not check_sight(sim, actor, pmo):
            if c >= 2 or idx == stop:
                break
            continue

        if not allaround:
            an = to_uint32(point_to_angle2(actor.x, actor.y, pmo.x, pmo.y) - actor.angle)
            if ANG90 < an < ANG270:
                dist = approx_distance(pmo.x - actor.x, pmo.y - actor.y)
                if dist > MELEERANGE:
                    if c >= 2 or idx == stop:
                        break
                    continue

        if pmo.flags & MobjFlag.MF_SHADOW:
            dist = approx_distance(pmo.x - actor.x, pmo.y - actor.y)
            speed = approx_distance(pmo.momx, pmo.momy)
            if dist > 2 * MELEERANGE and speed < 5 * FRACUNIT:
                return False
            if sim is not None and sim.rng.p_random() < 225:
                return False

        if actor.type == MobjType.MT_MINOTAUR:
            if actor.special1 is p_obj or actor.special1 is pmo:
                if c >= 2 or idx == stop:
                    break
                continue

        actor.target = pmo
        return True

    return False


def _wake_up(actor: Mobj, sim: GameSim | None) -> None:
    """Transition waking monster to seestate and play seesound."""
    if actor.info.seesound:
        if actor.flags2 & MobjFlag2.MF2_BOSS:
            start_sound(sim, None, actor.info.seesound, full_volume=True)
        else:
            start_sound(sim, actor, actor.info.seesound)
    if sim is not None:
        set_mobj_state(sim, actor, actor.info.seestate)
    else:
        actor.state_num = actor.info.seestate


# =============================================================================
# Range Checking Routines (R3)
# =============================================================================


def check_melee_range(arg1: Any, arg2: Any = None) -> bool:
    """Verify horizontal distance, line of sight, and vertical bounds for melee attack.

    Port of P_CheckMeleeRange (p_enemy.c).
    """
    if type(arg1).__name__ == "GameSim":
        sim = arg1
        actor = arg2
    else:
        actor = arg1
        sim = arg2

    if actor is None or actor.target is None:
        return False
    mo = actor.target
    dist = approx_distance(mo.x - actor.x, mo.y - actor.y)
    if dist >= MELEERANGE:
        return False
    if not check_sight(sim, actor, mo):
        return False
    if mo.z > actor.z + actor.height:
        return False
    elif actor.z > mo.z + mo.height:
        return False
    return True


def check_melee_range2(arg1: Any, arg2: Any = None) -> bool:
    """Verify secondary extended-reach melee range (64 <= dist < 128).

    Port of P_CheckMeleeRange2 (p_enemy.c).
    """
    if type(arg1).__name__ == "GameSim":
        sim = arg1
        actor = arg2
    else:
        actor = arg1
        sim = arg2

    if actor is None or actor.target is None:
        return False
    mo = actor.target
    dist = approx_distance(mo.x - actor.x, mo.y - actor.y)
    if dist >= MELEERANGE * 2 or dist < MELEERANGE:
        return False
    if not check_sight(sim, actor, mo):
        return False
    if mo.z > actor.z + actor.height:
        return False
    elif actor.z > mo.z + mo.height:
        return False
    return True


def check_missile_range(arg1: Any, arg2: Any = None) -> bool:
    """Check line of sight and distance probability curve for missile attack.

    Port of P_CheckMissileRange (p_enemy.c).
    """
    if type(arg1).__name__ == "GameSim":
        sim = arg1
        actor = arg2
    else:
        actor = arg1
        sim = arg2

    if actor is None or actor.target is None:
        return False

    if not check_sight(sim, actor, actor.target):
        return False

    if actor.flags & MobjFlag.MF_JUSTHIT:
        actor.flags &= ~MobjFlag.MF_JUSTHIT
        return True

    if actor.reactiontime:
        return False

    dist = (approx_distance(actor.x - actor.target.x, actor.y - actor.target.y) >> FRACBITS) - 64
    if not actor.info.meleestate:
        dist -= 128
    if dist > 200:
        dist = 200

    rnd = sim.rng.p_random() if sim is not None else 0
    if rnd < dist:
        return False
    return True


# =============================================================================
# Monster Movement & Directional Chase (R4)
# =============================================================================


def move(arg1: Any, arg2: Any = None) -> bool:
    """Advance actor one step along current movedir.

    Port of P_Move (p_enemy.c).
    """
    if type(arg1).__name__ == "GameSim":
        sim = arg1
        actor = arg2
    else:
        actor = arg1
        sim = arg2

    if actor is None:
        return False

    if actor.flags2 & MobjFlag2.MF2_BLASTED:
        return True
    if actor.movedir == DI_NODIR:
        return False

    tryx = actor.x + actor.info.speed * XSPEED[actor.movedir]
    tryy = actor.y + actor.info.speed * YSPEED[actor.movedir]

    if sim is None:
        actor.x = tryx
        actor.y = tryy
        return True

    if not try_move(sim, actor, tryx, tryy):
        if (actor.flags & MobjFlag.MF_FLOAT) and sim.map_state.floatok:
            if actor.z < sim.map_state.tmfloorz:
                actor.z += FLOATSPEED
            else:
                actor.z -= FLOATSPEED
            actor.flags |= MobjFlag.MF_INFLOAT
            return True

        actor.movedir = DI_NODIR
        spechits = list(sim.map_state.spechit) or list(sim.map_state.last_spechit)
        good = False
        from pyhexen.sim.line_activation import p_activate_line

        while spechits:
            ld = spechits.pop()
            if p_activate_line(sim, ld, actor, 0, SPAC_USE):
                good = True
        return good

    actor.flags &= ~MobjFlag.MF_INFLOAT
    if not (actor.flags & MobjFlag.MF_FLOAT):
        actor.z = actor.floorz
    return True


def try_walk(arg1: Any, arg2: Any = None) -> bool:
    """Attempt movement and reset movecount to random 0..15 on success.

    Port of P_TryWalk (p_enemy.c).
    """
    if type(arg1).__name__ == "GameSim":
        sim = arg1
        actor = arg2
    else:
        actor = arg1
        sim = arg2

    if not move(actor, sim):
        return False
    if sim is not None:
        actor.movecount = sim.rng.p_random() & 15
    return True


def new_chase_dir(arg1: Any, arg2: Any = None) -> None:
    """Select 8-way chase direction avoiding turnaround, prioritizing diagonals.

    Port of P_NewChaseDir (p_enemy.c).
    """
    if type(arg1).__name__ == "GameSim":
        sim = arg1
        actor = arg2
    else:
        actor = arg1
        sim = arg2

    if actor is None or actor.target is None:
        raise RuntimeError("P_NewChaseDir: called with no target")

    olddir = actor.movedir
    turnaround = OPPOSITE[olddir] if olddir < len(OPPOSITE) else DI_NODIR

    deltax = actor.target.x - actor.x
    deltay = actor.target.y - actor.y

    if deltax > 10 * FRACUNIT:
        d1 = DI_EAST
    elif deltax < -10 * FRACUNIT:
        d1 = DI_WEST
    else:
        d1 = DI_NODIR

    if deltay < -10 * FRACUNIT:
        d2 = DI_SOUTH
    elif deltay > 10 * FRACUNIT:
        d2 = DI_NORTH
    else:
        d2 = DI_NODIR

    # Try direct diagonal route
    if d1 != DI_NODIR and d2 != DI_NODIR:
        diag_idx = ((1 if deltay < 0 else 0) << 1) + (1 if deltax > 0 else 0)
        actor.movedir = DIAGS[diag_idx]
        if actor.movedir != turnaround and try_walk(actor, sim):
            return

    # Try randomized other directions
    rnd = sim.rng.p_random() if sim is not None else 0
    if rnd > 200 or abs(deltay) > abs(deltax):
        d1, d2 = d2, d1

    if d1 == turnaround:
        d1 = DI_NODIR
    if d2 == turnaround:
        d2 = DI_NODIR

    if d1 != DI_NODIR:
        actor.movedir = d1
        if try_walk(actor, sim):
            return

    if d2 != DI_NODIR:
        actor.movedir = d2
        if try_walk(actor, sim):
            return

    # Try old direction
    if olddir != DI_NODIR:
        actor.movedir = olddir
        if try_walk(actor, sim):
            return

    # Sweep remaining directions
    sweep_rnd = sim.rng.p_random() if sim is not None else 0
    if sweep_rnd & 1:
        for tdir in range(DI_EAST, DI_SOUTHEAST + 1):
            if tdir != turnaround:
                actor.movedir = tdir
                if try_walk(actor, sim):
                    return
    else:
        for tdir in range(DI_SOUTHEAST, DI_EAST - 1, -1):
            if tdir != turnaround:
                actor.movedir = tdir
                if try_walk(actor, sim):
                    return

    # Fallback to turnaround
    if turnaround != DI_NODIR:
        actor.movedir = turnaround
        if try_walk(actor, sim):
            return

    actor.movedir = DI_NODIR


# =============================================================================
# Shared Base AI Actions (R5)
# =============================================================================


@action("A_Look")
def a_look(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Idle scanning for sound alert or sight wakeup.

    Port of A_Look (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return

    actor.threshold = 0
    targ = None
    if actor.subsector is not None and actor.subsector.sector is not None:
        sec = actor.subsector.sector
        targ = getattr(sec, "sound_target", None) or getattr(sec, "soundtarget", None)

    if targ is not None and bool(targ.flags & MobjFlag.MF_SHOOTABLE):
        if actor.flags & MobjFlag.MF_AMBUSH:
            if check_sight(sim, actor, targ):
                actor.target = targ
                _wake_up(actor, sim)
                return
        else:
            actor.target = targ
            _wake_up(actor, sim)
            return

    if not look_for_players(actor, False, sim):
        return

    _wake_up(actor, sim)


@action("A_Chase")
def a_chase(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Core monster chase state machine ticker.

    Port of A_Chase (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return

    if actor.reactiontime > 0:
        actor.reactiontime -= 1

    if actor.threshold > 0:
        actor.threshold -= 1

    # Nightmare difficulty speedup
    skill_val = getattr(sim, "gameskill", getattr(sim, "skill", 2)) if sim is not None else 2
    if int(skill_val) == 4:
        actor.tics -= actor.tics // 2
        if actor.tics < 3:
            actor.tics = 3

    # Turn towards movement direction (angular increments of ANG45)
    if actor.movedir < 8:
        actor.angle &= (7 << 29) & 0xFFFFFFFF
        target_angle = (actor.movedir << 29) & 0xFFFFFFFF
        delta = to_int32(actor.angle - target_angle)
        if delta > 0:
            actor.angle = to_uint32(actor.angle - (ANG90 // 2))
        elif delta < 0:
            actor.angle = to_uint32(actor.angle + (ANG90 // 2))

    # Target loss or invalidation
    if (
        actor.target is None
        or not (actor.target.flags & MobjFlag.MF_SHOOTABLE)
        or actor.target.health <= 0
    ):
        if look_for_players(actor, True, sim):
            return
        actor.target = None
        if sim is not None:
            set_mobj_state(sim, actor, actor.info.spawnstate)
        else:
            actor.state_num = actor.info.spawnstate
        return

    # Don't attack twice consecutively
    if actor.flags & MobjFlag.MF_JUSTATTACKED:
        actor.flags &= ~MobjFlag.MF_JUSTATTACKED
        if int(skill_val) != 4:
            new_chase_dir(actor, sim)
            return

    # 1. Melee attack transition
    if actor.info.meleestate and check_melee_range(actor, sim):
        if actor.info.attacksound:
            start_sound(sim, actor, actor.info.attacksound)
        if sim is not None:
            set_mobj_state(sim, actor, actor.info.meleestate)
        else:
            actor.state_num = actor.info.meleestate
        return

    # 2. Missile attack transition
    if actor.info.missilestate:
        missile_ok = True
        if int(skill_val) < 4 and actor.movecount > 0:
            missile_ok = False
        elif not check_missile_range(actor, sim):
            missile_ok = False

        if missile_ok:
            if sim is not None:
                set_mobj_state(sim, actor, actor.info.missilestate)
            else:
                actor.state_num = actor.info.missilestate
            actor.flags |= MobjFlag.MF_JUSTATTACKED
            return

    # 3. Retargeting in netgame if target lost line-of-sight
    is_netgame = getattr(sim, "netgame", False) if sim is not None else False
    if is_netgame and not actor.threshold and not check_sight(sim, actor, actor.target):
        if look_for_players(actor, True, sim):
            return

    # 4. Chase stepping
    actor.movecount -= 1
    if actor.movecount < 0 or not move(actor, sim):
        new_chase_dir(actor, sim)

    # 5. Active sound playback
    if actor.info.activesound and sim is not None and sim.rng.p_random() < 3:
        if actor.type == MobjType.MT_BISHOP and sim.rng.p_random() < 128:
            start_sound(sim, actor, actor.info.seesound)
        elif actor.type == MobjType.MT_PIG:
            pig_sfx = int(SfxEnum.SFX_PIG_ACTIVE1) + (sim.rng.p_random() & 1)
            start_sound(sim, actor, pig_sfx)
        elif actor.flags2 & MobjFlag2.MF2_BOSS:
            start_sound(sim, None, actor.info.activesound, full_volume=True)
        else:
            start_sound(sim, actor, actor.info.activesound)


@action("A_FastChase")
def a_fastchase(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Boss chase routine with lateral strafing bursts.

    Port of A_FastChase (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return

    if actor.reactiontime > 0:
        actor.reactiontime -= 1

    if actor.threshold > 0:
        actor.threshold -= 1

    skill_val = getattr(sim, "gameskill", getattr(sim, "skill", 2)) if sim is not None else 2
    if int(skill_val) == 4:
        actor.tics -= actor.tics // 2
        if actor.tics < 3:
            actor.tics = 3

    if actor.movedir < 8:
        actor.angle &= (7 << 29) & 0xFFFFFFFF
        target_angle = (actor.movedir << 29) & 0xFFFFFFFF
        delta = to_int32(actor.angle - target_angle)
        if delta > 0:
            actor.angle = to_uint32(actor.angle - (ANG90 // 2))
        elif delta < 0:
            actor.angle = to_uint32(actor.angle + (ANG90 // 2))

    if (
        actor.target is None
        or not (actor.target.flags & MobjFlag.MF_SHOOTABLE)
        or actor.target.health <= 0
    ):
        if look_for_players(actor, True, sim):
            return
        if sim is not None:
            set_mobj_state(sim, actor, actor.info.spawnstate)
        else:
            actor.state_num = actor.info.spawnstate
        return

    if actor.flags & MobjFlag.MF_JUSTATTACKED:
        actor.flags &= ~MobjFlag.MF_JUSTATTACKED
        if int(skill_val) != 4:
            new_chase_dir(actor, sim)
            return

    # Strafe
    strafe_time = int(actor.special2) if isinstance(actor.special2, int) else 0
    if strafe_time > 0:
        actor.special2 = strafe_time - 1
    else:
        target = actor.target
        actor.special2 = 0
        actor.momx = 0
        actor.momy = 0
        dist = approx_distance(actor.x - target.x, actor.y - target.y)
        if dist < CLASS_BOSS_STRAFE_RANGE and sim is not None:
            if sim.rng.p_random() < 100:
                ang = point_to_angle2(actor.x, actor.y, target.x, target.y)
                if sim.rng.p_random() < 128:
                    ang = to_uint32(ang + ANG90)
                else:
                    ang = to_uint32(ang - ANG90)
                fine = ang >> ANGLETOFINESHIFT
                actor.momx = fixed_mul(13 * FRACUNIT, fine_cosine(fine))
                actor.momy = fixed_mul(13 * FRACUNIT, fine_sine(fine))
                actor.special2 = 3

    if actor.info.missilestate:
        missile_ok = True
        if int(skill_val) < 4 and actor.movecount > 0:
            missile_ok = False
        elif not check_missile_range(actor, sim):
            missile_ok = False

        if missile_ok:
            if sim is not None:
                set_mobj_state(sim, actor, actor.info.missilestate)
            else:
                actor.state_num = actor.info.missilestate
            actor.flags |= MobjFlag.MF_JUSTATTACKED
            return

    is_netgame = getattr(sim, "netgame", False) if sim is not None else False
    if is_netgame and not actor.threshold and not check_sight(sim, actor, actor.target):
        if look_for_players(actor, True, sim):
            return

    if not (isinstance(actor.special2, int) and actor.special2 > 0):
        actor.movecount -= 1
        if actor.movecount < 0 or not move(actor, sim):
            new_chase_dir(actor, sim)


@action("A_FaceTarget")
def a_facetarget(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Snap actor angle toward target, applying shadow jitter if target is cloaked.

    Port of A_FaceTarget (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None:
        return
    actor.flags &= ~MobjFlag.MF_AMBUSH
    actor.angle = to_uint32(point_to_angle2(actor.x, actor.y, actor.target.x, actor.target.y))
    if actor.target.flags & MobjFlag.MF_SHADOW:
        sub_rnd = sim.rng.p_subrandom() if sim is not None else 0
        actor.angle = to_uint32(actor.angle + (sub_rnd << 21))


@action("A_Pain")
def a_pain(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Play monster pain sound effect.

    Port of A_Pain (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None and actor.info.painsound:
        start_sound(sim, actor, actor.info.painsound)


@action("A_Scream")
def a_scream(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Play monster death sound or contextual player scream.

    Port of A_Scream (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    stop_sound(sim, actor)
    if actor.player is not None:
        if getattr(actor.player, "morphTics", 0) > 0:
            start_sound(sim, actor, actor.info.deathsound)
        else:
            if actor.momz <= -39 * FRACUNIT:
                sound = int(SfxEnum.SFX_PLAYER_FALLING_SPLAT)
            elif actor.health > -50:
                pclass = getattr(actor.player, "player_class", getattr(actor.player, "class_", 0))
                if pclass == 0:
                    sound = int(SfxEnum.SFX_PLAYER_FIGHTER_NORMAL_DEATH)
                elif pclass == 1:
                    sound = int(SfxEnum.SFX_PLAYER_CLERIC_NORMAL_DEATH)
                elif pclass == 2:
                    sound = int(SfxEnum.SFX_PLAYER_MAGE_NORMAL_DEATH)
                else:
                    sound = 0
            elif actor.health > -100:
                pclass = getattr(actor.player, "player_class", getattr(actor.player, "class_", 0))
                if pclass == 0:
                    sound = int(SfxEnum.SFX_PLAYER_FIGHTER_CRAZY_DEATH)
                elif pclass == 1:
                    sound = int(SfxEnum.SFX_PLAYER_CLERIC_CRAZY_DEATH)
                elif pclass == 2:
                    sound = int(SfxEnum.SFX_PLAYER_MAGE_CRAZY_DEATH)
                else:
                    sound = 0
            else:
                pclass = getattr(actor.player, "player_class", getattr(actor.player, "class_", 0))
                if pclass == 0:
                    base = int(SfxEnum.SFX_PLAYER_FIGHTER_EXTREME1_DEATH)
                elif pclass == 1:
                    base = int(SfxEnum.SFX_PLAYER_CLERIC_EXTREME1_DEATH)
                elif pclass == 2:
                    base = int(SfxEnum.SFX_PLAYER_MAGE_EXTREME1_DEATH)
                else:
                    base = 0
                r = (sim.rng.p_random() % 3) if (sim is not None and base != 0) else 0
                sound = base + r
            if sound:
                start_sound(sim, actor, sound)
    else:
        if actor.info.deathsound:
            start_sound(sim, actor, actor.info.deathsound)


@action("A_XScream")
def a_xscream(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Extreme gib death scream (alias to A_Scream in Hexen)."""
    a_scream(actor, sim)


@action("A_NoBlocking")
def a_noblocking(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Clear MF_SOLID flag on dying actor.

    Port of A_NoBlocking (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags &= ~MobjFlag.MF_SOLID


@action("A_Explode")
def a_explode(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Radius damage dispatcher for explosions.

    Port of A_Explode (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return

    damage = 128
    distance = 128
    damage_self = True

    if actor.type == MobjType.MT_FIREBOMB:
        actor.z += 32 * FRACUNIT
        actor.flags &= ~MobjFlag.MF_SHADOW
    elif actor.type == MobjType.MT_MNTRFX2:
        damage = 24
    elif actor.type == MobjType.MT_BISHOP:
        rnd = sim.rng.p_random() if sim is not None else 0
        damage = 25 + (rnd & 15)
    elif actor.type == MobjType.MT_HAMMER_MISSILE:
        damage = 128
        damage_self = False
    elif actor.type == MobjType.MT_FSWORD_MISSILE:
        damage = 64
        damage_self = False
    elif actor.type == MobjType.MT_CIRCLEFLAME:
        damage = 20
        damage_self = False
    elif actor.type in (
        MobjType.MT_SORCBALL1,
        MobjType.MT_SORCBALL2,
        MobjType.MT_SORCBALL3,
    ):
        distance = 255
        damage = 255
        actor.args = (1, actor.args[1], actor.args[2], actor.args[3], actor.args[4])
    elif actor.type == MobjType.MT_SORCFX1:
        damage = 30
    elif actor.type == MobjType.MT_SORCFX4:
        damage = 20
    elif actor.type == MobjType.MT_TREEDESTRUCTIBLE:
        damage = 10
    elif actor.type == MobjType.MT_DRAGON_FX2:
        damage = 80
        damage_self = False
    elif actor.type == MobjType.MT_MSTAFF_FX:
        damage = 64
        distance = 192
        damage_self = False
    elif actor.type == MobjType.MT_MSTAFF_FX2:
        damage = 80
        distance = 192
        damage_self = False
    elif actor.type == MobjType.MT_POISONCLOUD:
        damage = 4
        distance = 40
    elif actor.type in (MobjType.MT_ZXMAS_TREE, MobjType.MT_ZSHRUB2):
        damage = 30
        distance = 64

    if sim is not None:
        from pyhexen.sim.attacks import radius_attack

        radius_attack(sim, actor, actor.target, damage, distance, damage_self)


@action("A_FreezeDeath")
def a_freezedeath(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Freeze actor into solid ice statue with random duration.

    Port of A_FreezeDeath (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    r1 = sim.rng.p_random() if sim is not None else 0
    r2 = sim.rng.p_random() if sim is not None else 0
    actor.tics = 75 + r1 + r2
    actor.flags |= MobjFlag.MF_SOLID | MobjFlag.MF_SHOOTABLE | MobjFlag.MF_NOBLOOD
    actor.flags2 |= (
        MobjFlag2.MF2_PUSHABLE
        | MobjFlag2.MF2_TELESTOMP
        | MobjFlag2.MF2_PASSMOBJ
        | MobjFlag2.MF2_SLIDE
    )
    actor.height <<= 2
    start_sound(sim, actor, SfxEnum.SFX_FREEZE_DEATH)

    if actor.player is not None:
        p = actor.player
        p.damagecount = 0
        p.bonuscount = 0
        if hasattr(p, "poisoncount"):
            p.poisoncount = 0
    elif (actor.flags & MobjFlag.MF_COUNTKILL) and actor.special:
        if sim is not None:
            execute_line_special(
                sim,
                actor.special,
                actor.args,
                None,
                actor,
            )


@action("A_IceSetTics")
def a_icesettics(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Randomize ice chunk duration based on terrain floor flat.

    Port of A_IceSetTics (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    rnd = sim.rng.p_random() if sim is not None else 0
    actor.tics = 70 + (rnd & 63)
    if actor.subsector is not None and actor.subsector.sector is not None:
        sec = actor.subsector.sector
        floor_pic = getattr(sec, "floor_pic", "").upper()
        if "LAVA" in floor_pic:
            actor.tics >>= 2
        elif "ICE" in floor_pic:
            actor.tics <<= 1


@action("A_IceCheckHeadDone")
def a_icecheckheaddone(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Advance player ice head chunk state when special2 is 666.

    Port of A_IceCheckHeadDone (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None and actor.special2 == 666:
        if sim is not None:
            set_mobj_state(sim, actor, StateNum.S_ICECHUNK_HEAD2)
        else:
            actor.state_num = StateNum.S_ICECHUNK_HEAD2


@action("A_FreezeDeathChunks")
def a_freezedeathchunks(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Shatter stationary frozen statue into ice chunk projectiles.

    Port of A_FreezeDeathChunks (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if actor.momx != 0 or actor.momy != 0 or actor.momz != 0:
        actor.tics = 105
        return
    start_sound(sim, actor, SfxEnum.SFX_FREEZE_SHATTER)

    if sim is not None:
        from pyhexen.sim.mobj import spawn_mobj

        # Burst 1
        num_chunks1 = 12 + (sim.rng.p_random() & 15)
        for _ in range(num_chunks1 + 1):
            r1 = sim.rng.p_random()
            r2 = sim.rng.p_random()
            r3 = sim.rng.p_random()
            cx = actor.x + (((r3 - 128) * actor.radius) >> 7)
            cy = actor.y + (((r2 - 128) * actor.radius) >> 7)
            cz = actor.z + ((r1 * actor.height) // 255)
            mo = spawn_mobj(sim, cx, cy, cz, MobjType.MT_ICECHUNK)
            state_offset = sim.rng.p_random() % 3
            set_mobj_state(sim, mo, mo.info.spawnstate + state_offset)
            h = actor.height if actor.height > 0 else 1
            mo.momz = fixed_div(mo.z - actor.z, h) << 2
            mo.momx = sim.rng.p_subrandom() << (FRACBITS - 7)
            mo.momy = sim.rng.p_subrandom() << (FRACBITS - 7)
            a_icesettics(mo, sim)

        # Burst 2
        num_chunks2 = 12 + (sim.rng.p_random() & 15)
        for _ in range(num_chunks2 + 1):
            r1 = sim.rng.p_random()
            r2 = sim.rng.p_random()
            r3 = sim.rng.p_random()
            cx = actor.x + (((r3 - 128) * actor.radius) >> 7)
            cy = actor.y + (((r2 - 128) * actor.radius) >> 7)
            cz = actor.z + ((r1 * actor.height) // 255)
            mo = spawn_mobj(sim, cx, cy, cz, MobjType.MT_ICECHUNK)
            state_offset = sim.rng.p_random() % 3
            set_mobj_state(sim, mo, mo.info.spawnstate + state_offset)
            h = actor.height if actor.height > 0 else 1
            mo.momz = fixed_div(mo.z - actor.z, h) << 2
            mo.momx = sim.rng.p_subrandom() << (FRACBITS - 7)
            mo.momy = sim.rng.p_subrandom() << (FRACBITS - 7)
            a_icesettics(mo, sim)

        if actor.player is not None:
            mo = spawn_mobj(sim, actor.x, actor.y, actor.z + 41 * FRACUNIT, MobjType.MT_ICECHUNK)
            set_mobj_state(sim, mo, StateNum.S_ICECHUNK_HEAD)
            h = actor.height if actor.height > 0 else 1
            mo.momz = fixed_div(mo.z - actor.z, h) << 2
            mo.momx = sim.rng.p_subrandom() << (FRACBITS - 7)
            mo.momy = sim.rng.p_subrandom() << (FRACBITS - 7)
            mo.flags2 |= MobjFlag2.MF2_ICEDAMAGE
            mo.flags2 &= ~MobjFlag2.MF2_FLOORCLIP
            mo.player = actor.player
            actor.player = None
            mo.health = actor.health
            mo.angle = actor.angle
            mo.player.mo = mo

        set_mobj_state(sim, actor, StateNum.S_FREETARGMOBJ)
        actor.flags2 |= MobjFlag2.MF2_DONTDRAW


@action("A_FreeTargMobj")
def a_freetargmobj(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Free mobj from interactions and hide above ceiling.

    Port of A_FreeTargMobj (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    actor.momx = 0
    actor.momy = 0
    actor.momz = 0
    actor.z = actor.ceilingz + 4 * FRACUNIT
    actor.flags &= ~(
        MobjFlag.MF_SHOOTABLE
        | MobjFlag.MF_FLOAT
        | MobjFlag.MF_SKULLFLY
        | MobjFlag.MF_SOLID
        | MobjFlag.MF_COUNTKILL
    )
    actor.flags |= MobjFlag.MF_CORPSE | MobjFlag.MF_DROPOFF | MobjFlag.MF_NOGRAVITY
    actor.flags2 &= ~(MobjFlag2.MF2_PASSMOBJ | MobjFlag2.MF2_LOGRAV)
    actor.flags2 |= MobjFlag2.MF2_DONTDRAW
    actor.player = None
    actor.health = -1000


@action("A_QueueCorpse")
def a_queuecorpse(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Add corpse to 64-slot circular queue, removing oldest on overflow.

    Port of A_QueueCorpse (p_enemy.c).
    """
    global _corpse_queue_slot
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if _corpse_queue_slot >= CORPSEQUEUESIZE:
        old_corpse = _corpse_queue[_corpse_queue_slot % CORPSEQUEUESIZE]
        if old_corpse is not None and sim is not None:
            from pyhexen.sim.mobj import remove_mobj

            remove_mobj(sim, old_corpse)
    _corpse_queue[_corpse_queue_slot % CORPSEQUEUESIZE] = actor
    _corpse_queue_slot += 1


@action("A_DeQueueCorpse")
def a_dequeuecorpse(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Remove corpse from circular queue.

    Port of A_DeQueueCorpse (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    for slot in range(CORPSEQUEUESIZE):
        if _corpse_queue[slot] is actor:
            _corpse_queue[slot] = None
            break


def p_init_creature_corpse_queue() -> None:
    """Reset the creature corpse queue.

    Port of P_InitCreatureCorpseQueue (p_enemy.c).
    """
    global _corpse_queue_slot
    for i in range(CORPSEQUEUESIZE):
        _corpse_queue[i] = None
    _corpse_queue_slot = 0


@action("A_SetInvulnerable")
def a_setinvulnerable(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Set MF2_INVULNERABLE flag.

    Port of A_SetInvulnerable (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags2 |= MobjFlag2.MF2_INVULNERABLE


@action("A_UnSetInvulnerable")
def a_unsetinvulnerable(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Clear MF2_INVULNERABLE flag.

    Port of A_UnSetInvulnerable (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags2 &= ~MobjFlag2.MF2_INVULNERABLE


@action("A_SetReflective")
def a_setreflective(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Set MF2_REFLECTIVE flag (and invulnerability for Centaurs).

    Port of A_SetReflective (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    actor.flags2 |= MobjFlag2.MF2_REFLECTIVE
    if actor.type in (MobjType.MT_CENTAUR, MobjType.MT_CENTAURLEADER):
        a_setinvulnerable(actor, sim)


@action("A_UnSetReflective")
def a_unsetreflective(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Clear MF2_REFLECTIVE flag (and invulnerability for Centaurs).

    Port of A_UnSetReflective (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    actor.flags2 &= ~MobjFlag2.MF2_REFLECTIVE
    if actor.type in (MobjType.MT_CENTAUR, MobjType.MT_CENTAURLEADER):
        a_unsetinvulnerable(actor, sim)


def p_update_morphed_monster(actor: Mobj, tics: int, sim: GameSim | None = None) -> bool:
    """Count down morph timer and attempt unmorphing back to original form.

    Port of P_UpdateMorphedMonster (p_enemy.c).
    """
    if isinstance(actor.special1, int):
        actor.special1 -= tics
        if actor.special1 > 0:
            return False
    else:
        return False

    mo_type = int(actor.special2) if isinstance(actor.special2, int) else 0
    if mo_type in (
        MobjType.MT_WRAITHB,
        MobjType.MT_SERPENT,
        MobjType.MT_SERPENTLEADER,
        MobjType.MT_MINOTAUR,
    ):
        return False

    if sim is None or sim.level is None:
        return False

    from pyhexen.sim.map import check_position
    from pyhexen.sim.mobj import remove_mobj, spawn_mobj

    unmorphed = spawn_mobj(sim, actor.x, actor.y, actor.z, mo_type)
    if not check_position(sim, unmorphed, actor.x, actor.y):
        remove_mobj(sim, unmorphed)
        actor.special1 = 5 * 35
        return False

    unmorphed.angle = actor.angle
    unmorphed.target = actor.target
    remove_mobj(sim, actor)
    return True


@action("A_PigLook")
def a_piglook(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Morph check before A_Look.

    Port of A_PigLook (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if p_update_morphed_monster(actor, 10, sim):
        return
    a_look(actor, sim)


@action("A_PigChase")
def a_pigchase(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Morph check before A_Chase.

    Port of A_PigChase (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if p_update_morphed_monster(actor, 3, sim):
        return
    a_chase(actor, sim)


@action("A_PigAttack")
def a_pigattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Pig melee attack.

    Port of A_PigAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if p_update_morphed_monster(actor, 18, sim):
        return
    if actor.target is None:
        return
    if check_melee_range(actor, sim):
        rnd = sim.rng.p_random() if sim is not None else 0
        damage = 2 + (rnd & 1)
        if sim is not None:
            from pyhexen.sim.inter import damage_mobj

            damage_mobj(sim, target=actor.target, inflictor=actor, source=actor, damage=damage)
        start_sound(sim, actor, SfxEnum.SFX_PIG_ATTACK)


@action("A_PigPain")
def a_pigpain(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Pig pain with upward hop momentum.

    Port of A_PigPain (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    a_pain(actor, sim)
    if actor.z <= actor.floorz:
        actor.momz = (7 * FRACUNIT) // 2


@action("A_CheckFloor")
def a_checkfloor(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Snap falling object to floor and transition to deathstate on impact.

    Port of A_CheckFloor (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if actor.z <= actor.floorz:
        actor.z = actor.floorz
        actor.flags2 &= ~MobjFlag2.MF2_LOGRAV
        if sim is not None:
            set_mobj_state(sim, actor, actor.info.deathstate)
        else:
            actor.state_num = actor.info.deathstate


__all__ = [
    "CLASS_BOSS_STRAFE_RANGE",
    "CORPSEQUEUESIZE",
    "DIAGS",
    "DI_EAST",
    "DI_NODIR",
    "DI_NORTH",
    "DI_NORTHEAST",
    "DI_NORTHWEST",
    "DI_SOUTH",
    "DI_SOUTHEAST",
    "DI_SOUTHWEST",
    "DI_WEST",
    "FLOATSPEED",
    "MELEERANGE",
    "MONS_LOOK_LIMIT",
    "MONS_LOOK_RANGE",
    "OPPOSITE",
    "XSPEED",
    "YSPEED",
    "a_chase",
    "a_checkfloor",
    "a_dequeuecorpse",
    "a_explode",
    "a_facetarget",
    "a_fastchase",
    "a_freezedeath",
    "a_freezedeathchunks",
    "a_freetargmobj",
    "a_icecheckheaddone",
    "a_icesettics",
    "a_look",
    "a_noblocking",
    "a_pain",
    "a_pigattack",
    "a_pigchase",
    "a_piglook",
    "a_pigpain",
    "a_queuecorpse",
    "a_scream",
    "a_setinvulnerable",
    "a_setreflective",
    "a_unsetinvulnerable",
    "a_unsetreflective",
    "a_xscream",
    "check_melee_range",
    "check_melee_range2",
    "check_missile_range",
    "look_for_monsters",
    "look_for_players",
    "move",
    "new_chase_dir",
    "noise_alert",
    "p_init_creature_corpse_queue",
    "p_update_morphed_monster",
    "recursive_sound",
    "start_sound",
    "stop_sound",
    "try_walk",
]
