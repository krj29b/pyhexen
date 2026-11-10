# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.
"""Boss monster actions: Death Wyvern, Maulotaur, Heresiarch, Class Bosses, and Korax.

Ports action functions and boss helpers from Chocolate Doom's `src/hexen/p_enemy.c`,
`src/hexen/p_pspr.c`, `src/hexen/p_mobj.c`, `src/hexen/p_map.c`, `src/hexen/p_maputl.c`,
and `src/hexen/p_telept.c` for Task T10.5c.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import (
    ANG1,
    ANG45,
    ANG90,
    ANGLETOFINESHIFT,
    FINEMASK,
    approx_distance,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import (
    FRACBITS,
    FRACUNIT,
    c_div,
    fixed_mul,
    to_int32,
    to_uint32,
)
from pyhexen.sim.actions import action
from pyhexen.sim.actions.monsters_common import (
    FLOAT_BOB_OFFSETS,
    FLOOR_WATER,
    get_thing_floor_type,
    hitdice,
)
from pyhexen.sim.attacks import (
    check_missile_spawn,
    face_mobj,
    seeker_missile,
    spawn_missile,
    spawn_missile_angle,
)
from pyhexen.sim.enemy import a_chase, check_melee_range, look_for_players
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.map import check_position, teleport_move
from pyhexen.sim.mobj import (
    ONFLOORZ,
    Mobj,
    find_mobj_from_tid,
    remove_mobj,
    set_mobj_state,
    set_mobj_state_nf,
    spawn_mobj,
)
from pyhexen.sim.sight import check_sight
from pyhexen.sim.specials.dispatcher import execute_line_special

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

# =============================================================================
# Constants
# =============================================================================

MAULATORTICS: int = 25 * 35
MINOTAUR_LOOK_DIST: int = 16 * 54 * FRACUNIT

SORCBALL_INITIAL_SPEED: int = 7
SORCBALL_TERMINAL_SPEED: int = 25
SORCBALL_SPEED_ROTATIONS: int = 5
SORC_DEFENSE_TIME: int = 255
SORC_DEFENSE_HEIGHT: int = 45
BOUNCE_TIME_UNIT: int = 35 // 2
SORCFX4_RAPIDFIRE_TIME: int = 6 * 3
SORCFX4_SPREAD_ANGLE: int = 20

SORC_STOPPED: int = 0
SORC_NORMAL: int = 1
SORC_ACCELERATE: int = 2
SORC_DECELERATE: int = 3

BALL1_ANGLEOFFSET: int = 0
BALL2_ANGLEOFFSET: int = (ANG_MAX_32 := 0xFFFFFFFF) // 3
BALL3_ANGLEOFFSET: int = (ANG_MAX_32 // 3) * 2

KORAX_SPIRIT_LIFETIME: int = 5 * 35
KORAX_COMMAND_HEIGHT: int = 120
KORAX_COMMAND_OFFSET: int = 27

KORAX_TID: int = 245
KORAX_FIRST_TELEPORT_TID: int = 248
KORAX_TELEPORT_TID: int = 249


# =============================================================================
# Shared Helpers
# =============================================================================


def _get_arg(actor: Mobj, idx: int) -> int:
    """Safely read actor.args[idx]."""
    if idx < len(actor.args):
        return int(actor.args[idx])
    return 0


def _set_arg(actor: Mobj, idx: int, val: int) -> None:
    """Safely mutate actor.args[idx] even when actor.args is a tuple."""
    args_list = list(actor.args)
    while len(args_list) <= idx:
        args_list.append(0)
    args_list[idx] = int(val)
    actor.args = tuple(args_list[:5])  # type: ignore[assignment]


def _resolve_mobj(val: Any) -> Mobj | None:
    """Resolve an Mobj stored directly or wrapped in a union-like attribute."""
    if isinstance(val, Mobj):
        return val
    m = getattr(val, "m", None)
    if isinstance(m, Mobj):
        return m
    return None


def _emit_sound(
    sim: GameSim,
    actor: Mobj | None,
    sound_id: int | SfxEnum,
    *,
    full_volume: bool = False,
) -> None:
    """Emit a SoundEvent on sim.events."""
    sid = int(sound_id)
    if sid <= 0:
        return
    origin_id = None if (full_volume or actor is None) else actor.tid
    x = actor.x if actor is not None else 0
    y = actor.y if actor is not None else 0
    z = actor.z if actor is not None else 0
    sim.events.emit(
        SoundEvent(
            sound_id=sid,
            origin_id=origin_id,
            x=x,
            y=y,
            z=z,
        )
    )


def p_test_mobj_location(sim: GameSim, mobj: Mobj) -> bool:
    """Port of P_TestMobjLocation (p_map.c:729).

    Returns True if the mobj is not blocked by anything at its current
    location, otherwise returns False.
    """
    flags = mobj.flags
    mobj.flags &= ~MobjFlag.MF_PICKUP
    if check_position(sim, mobj, mobj.x, mobj.y):
        mobj.flags = flags
        if mobj.z < mobj.floorz or mobj.z + mobj.height > mobj.ceilingz:
            return False
        return True
    mobj.flags = flags
    return False


def rough_block_check(sim: GameSim, mo: Mobj, index: int) -> Mobj | None:
    """Port of RoughBlockCheck (p_maputl.c:1007)."""
    if sim.level is None:
        return None
    bm = sim.level.blockmap
    if index < 0 or index >= len(bm.block_links):
        return None

    link: Mobj | None = bm.block_links[index]
    master = _resolve_mobj(mo.special1) if mo.type == MobjType.MT_MINOTAUR else None

    while link is not None:
        if mo.player is not None:
            if (link is mo) or not (
                (link.flags & MobjFlag.MF_COUNTKILL) or (link.player is not None and link is not mo)
            ):
                link = link.bnext
                continue
            if not (link.flags & MobjFlag.MF_SHOOTABLE):
                link = link.bnext
                continue
            if link.flags2 & MobjFlag2.MF2_DORMANT:
                link = link.bnext
                continue
            if link.type == MobjType.MT_MINOTAUR and _resolve_mobj(link.special1) is mo:
                link = link.bnext
                continue
            if sim.netgame and not sim.deathmatch and link.player is not None:
                link = link.bnext
                continue
            if check_sight(sim, mo, link):
                return link
            link = link.bnext
        elif mo.type == MobjType.MT_MINOTAUR:
            if link is mo or not (
                (link.flags & MobjFlag.MF_COUNTKILL)
                or (link.player is not None and link is not master)
            ):
                link = link.bnext
                continue
            if not (link.flags & MobjFlag.MF_SHOOTABLE):
                link = link.bnext
                continue
            if link.flags2 & MobjFlag2.MF2_DORMANT:
                link = link.bnext
                continue
            if link.type == MobjType.MT_MINOTAUR and _resolve_mobj(link.special1) is _resolve_mobj(
                mo.special1
            ):
                link = link.bnext
                continue
            if sim.netgame and not sim.deathmatch and link.player is not None:
                link = link.bnext
                continue
            if check_sight(sim, mo, link):
                return link
            link = link.bnext
        elif mo.type == MobjType.MT_MSTAFF_FX2:
            if link is mo or not (
                (link.flags & MobjFlag.MF_COUNTKILL)
                or (link.player is not None and link is not mo.target)
            ):
                link = link.bnext
                continue
            if not (link.flags & MobjFlag.MF_SHOOTABLE):
                link = link.bnext
                continue
            if link.flags2 & MobjFlag2.MF2_DORMANT:
                link = link.bnext
                continue
            if link.type == MobjType.MT_MINOTAUR and _resolve_mobj(link.special1) is mo.target:
                link = link.bnext
                continue
            if sim.netgame and not sim.deathmatch and link.player is not None:
                link = link.bnext
                continue
            if check_sight(sim, mo, link):
                angle = to_uint32(
                    point_to_angle2(
                        mo.target.x if mo.target is not None else mo.x,
                        mo.target.y if mo.target is not None else mo.y,
                        link.x,
                        link.y,
                    )
                    - (mo.target.angle if mo.target is not None else mo.angle)
                )
                angle >>= 24
                if angle > 226 or angle < 30:
                    return link
            link = link.bnext
        else:  # Spirits (MT_HOLY_FX)
            if link is mo or not (
                (link.flags & MobjFlag.MF_COUNTKILL)
                or (link.player is not None and link is not mo.target)
            ):
                link = link.bnext
                continue
            if not (link.flags & MobjFlag.MF_SHOOTABLE):
                link = link.bnext
                continue
            if link.flags2 & MobjFlag2.MF2_DORMANT:
                link = link.bnext
                continue
            if link.type == MobjType.MT_MINOTAUR and _resolve_mobj(link.special1) is mo.target:
                link = link.bnext
                continue
            if sim.netgame and not sim.deathmatch and link.player is not None:
                link = link.bnext
                continue
            if link is mo.target:
                link = link.bnext
                continue
            if check_sight(sim, mo, link):
                return link
            link = link.bnext

    return None


def rough_monster_search(sim: GameSim, mo: Mobj, distance: int) -> Mobj | None:
    """Port of P_RoughMonsterSearch (p_maputl.c:824).

    Searches nearby blockmap cells in expanding square rings up to `distance`.
    """
    if sim.level is None:
        return None
    bm = sim.level.blockmap
    bmapwidth = bm.width
    bmapheight = bm.height
    if bmapwidth <= 0 or bmapheight <= 0:
        return None

    start_x = bm.world_to_block_x(mo.x)
    start_y = bm.world_to_block_y(mo.y)

    if 0 <= start_x < bmapwidth and 0 <= start_y < bmapheight:
        target = rough_block_check(sim, mo, start_y * bmapwidth + start_x)
        if target is not None:
            return target

    for count in range(1, distance + 1):
        block_x = start_x - count
        block_y = start_y - count

        if block_y < 0:
            block_y = 0
        elif block_y >= bmapheight:
            block_y = bmapheight - 1
        if block_x < 0:
            block_x = 0
        elif block_x >= bmapwidth:
            block_x = bmapwidth - 1

        block_index = block_y * bmapwidth + block_x
        first_stop = start_x + count
        if first_stop < 0:
            continue
        if first_stop >= bmapwidth:
            first_stop = bmapwidth - 1

        second_stop = start_y + count
        if second_stop < 0:
            continue
        if second_stop >= bmapheight:
            second_stop = bmapheight - 1

        third_stop = second_stop * bmapwidth + block_x
        second_stop = second_stop * bmapwidth + first_stop
        first_stop += block_y * bmapwidth
        final_stop = block_index

        # Trace the first block section (along the top)
        for _ in range(block_index, first_stop + 1):
            target = rough_block_check(sim, mo, block_index)
            if target is not None:
                return target
            block_index += 1

        # Trace the second block section (right edge)
        block_index -= 1
        while block_index <= second_stop:
            target = rough_block_check(sim, mo, block_index)
            if target is not None:
                return target
            block_index += bmapwidth

        # Trace the third block section (bottom edge)
        block_index -= bmapwidth
        while block_index >= third_stop:
            target = rough_block_check(sim, mo, block_index)
            if target is not None:
                return target
            block_index -= 1

        # Trace the final block section (left edge)
        block_index += 1
        while block_index > final_stop:
            target = rough_block_check(sim, mo, block_index)
            if target is not None:
                return target
            block_index -= bmapwidth

    return None


def p_teleport(
    sim: GameSim,
    thing: Mobj,
    x: int,
    y: int,
    angle: int,
    use_fog: bool = True,
) -> bool:
    """Port of P_Teleport (p_telept.c:50)."""
    oldx = thing.x
    oldy = thing.y
    oldz = thing.z

    if not teleport_move(sim, thing, x, y, False):
        return False

    if thing.player is not None:
        if thing.player.powers[3] and thing.z != thing.floorz:  # pw_flight
            thing.z = thing.floorz
        else:
            thing.z = thing.floorz
            thing.player.viewheight = getattr(thing.player, "viewheight", 48 * FRACUNIT)
            thing.player.deltaviewheight = 0
            thing.player.viewz = thing.z + thing.player.viewheight
            thing.player.lookdir = 0
    elif thing.flags & MobjFlag.MF_MISSILE:
        thing.z = thing.floorz
    else:
        thing.z = thing.floorz

    # Spawn teleport fog at source and destination
    if use_fog:
        fog_delta = 0 if (thing.flags & MobjFlag.MF_MISSILE) else 24 * FRACUNIT
        fog = spawn_mobj(sim, oldx, oldy, oldz + fog_delta, MobjType.MT_TFOG)
        _emit_sound(sim, fog, SfxEnum.SFX_TELEPORT)
        an = ((angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        fog2 = spawn_mobj(
            sim,
            x + 20 * fine_cosine(an),
            y + 20 * fine_sine(an),
            thing.z + fog_delta,
            MobjType.MT_TFOG,
        )
        _emit_sound(sim, fog2, SfxEnum.SFX_TELEPORT)
        if thing.player is not None and not thing.player.powers[1]:  # pw_weaponlevel2
            thing.reactiontime = 18

    thing.angle = to_uint32(angle)
    if thing.flags2 & MobjFlag2.MF2_FLOORCLIP:
        if thing.z == thing.floorz and get_thing_floor_type(thing) > FLOOR_WATER:
            thing.floorclip = 10 * FRACUNIT
        else:
            thing.floorclip = 0

    if thing.flags & MobjFlag.MF_MISSILE:
        fine_an = ((angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        thing.momx = fixed_mul(thing.info.speed, fine_cosine(fine_an))
        thing.momy = fixed_mul(thing.info.speed, fine_sine(fine_an))
    elif use_fog:
        thing.momx = 0
        thing.momy = 0
        thing.momz = 0

    return True


def spawn_korax_missile(
    sim: GameSim,
    x: int,
    y: int,
    z: int,
    source: Mobj,
    dest: Mobj,
    mtype: int | MobjType,
) -> Mobj | None:
    """Port of P_SpawnKoraxMissile (p_mobj.c:2438)."""
    th = spawn_mobj(sim, x, y, z, mtype)
    if th.info.seesound:
        _emit_sound(sim, th, th.info.seesound)
    th.target = source
    an = point_to_angle2(x, y, dest.x, dest.y)
    if dest.flags & MobjFlag.MF_SHADOW:
        an = to_uint32(an + (sim.rng.p_subrandom() << 21))
    th.angle = to_uint32(an)
    fine_an = (th.angle >> ANGLETOFINESHIFT) & FINEMASK
    th.momx = fixed_mul(th.info.speed, fine_cosine(fine_an))
    th.momy = fixed_mul(th.info.speed, fine_sine(fine_an))
    dist = approx_distance(dest.x - x, dest.y - y)
    denom = th.info.speed if th.info.speed != 0 else 1
    dist = c_div(dist, denom)
    if dist < 1:
        dist = 1
    th.momz = c_div(dest.z - z + (30 * FRACUNIT), dist)
    if check_missile_spawn(sim, th):
        return th
    return None


# =============================================================================
# 1. Death Wyvern (MT_DRAGON) — p_enemy.c:2830-3112
# =============================================================================


def dragon_seek(sim: GameSim, actor: Mobj, thresh: int, turn_max: int) -> None:
    """Port of DragonSeek (p_enemy.c:2851)."""
    target = actor.tracer
    if target is None:
        return

    direction, delta = face_mobj(actor, target)
    if delta > thresh:
        delta >>= 1
        if delta > turn_max:
            delta = turn_max

    if direction:
        actor.angle = to_uint32(actor.angle + delta)
    else:
        actor.angle = to_uint32(actor.angle - delta)

    angle_fine = (actor.angle >> ANGLETOFINESHIFT) & FINEMASK
    actor.momx = fixed_mul(actor.info.speed, fine_cosine(angle_fine))
    actor.momy = fixed_mul(actor.info.speed, fine_sine(angle_fine))

    dist = c_div(
        approx_distance(target.x - actor.x, target.y - actor.y),
        actor.info.speed if actor.info.speed != 0 else 1,
    )
    if actor.z + actor.height < target.z or target.z + target.height < actor.z:
        if dist < 1:
            dist = 1
        actor.momz = c_div(target.z - actor.z, dist)
    else:
        actor.momz = 0

    if (
        actor.target is not None
        and (actor.target.flags & MobjFlag.MF_SHOOTABLE)
        and (sim.rng.p_random() < 64)
    ):
        # Attack the destination mobj if it's attackable, or attack actor->target
        old_target: Mobj | None = None
        angle_to_target = point_to_angle2(actor.x, actor.y, actor.target.x, actor.target.y)
        if abs(to_int32(actor.angle - angle_to_target)) < (ANG45 // 2):
            if check_sight(sim, actor, actor.target):
                spawn_missile(sim, actor, actor.target, MobjType.MT_DRAGON_FX)
                _emit_sound(sim, actor, SfxEnum.SFX_DRAGON_ATTACK)
        elif target.flags & MobjFlag.MF_SHOOTABLE:
            old_target = actor.target
            actor.target = target
            if abs(to_int32(actor.angle - angle_to_target)) < (ANG45 // 2):
                if check_sight(sim, actor, actor.target):
                    spawn_missile(sim, actor, actor.target, MobjType.MT_DRAGON_FX)
                    _emit_sound(sim, actor, SfxEnum.SFX_DRAGON_ATTACK)
            actor.target = old_target

    if dist < 4:
        # Hit the target thing
        if actor.target is not None and sim.rng.p_random() < 200:
            best_actor: Mobj | None = None
            best_angle = 0xFFFFFFFF
            angle_to_target = point_to_angle2(actor.x, actor.y, actor.target.x, actor.target.y)
            for i in range(5):
                arg_tid = _get_arg(target, i)
                if not arg_tid:
                    continue
                for mo in find_mobj_from_tid(sim, arg_tid):
                    angle_to_spot = point_to_angle2(actor.x, actor.y, mo.x, mo.y)
                    diff = abs(to_int32(angle_to_spot - angle_to_target))
                    if diff < best_angle:
                        best_angle = diff
                        best_actor = mo
            if best_actor is not None:
                actor.tracer = best_actor
        else:
            # Pick a random non-zero arg from target.args
            non_zero_args = [_get_arg(target, i) for i in range(5) if _get_arg(target, i) != 0]
            if non_zero_args:
                i = 0
                while True:
                    idx = (sim.rng.p_random() >> 2) % 5
                    chosen_tid = _get_arg(target, idx)
                    if chosen_tid != 0:
                        break
                    i += 1
                    if i > 100:
                        chosen_tid = non_zero_args[0]
                        break
                matches = find_mobj_from_tid(sim, chosen_tid)
                if matches:
                    actor.tracer = matches[0]


@action("A_DragonInitFlight")
def a_dragoninitflight(sim: GameSim, actor: Mobj) -> None:
    """Port of A_DragonInitFlight (p_enemy.c:2959)."""
    tid = actor.tid
    while True:
        matches = find_mobj_from_tid(sim, tid)
        found: Mobj | None = None
        for mo in matches:
            if mo.type == MobjType.MT_MAPSPOT:
                found = mo
                break
        if found is not None:
            actor.tracer = found
            break
        if tid == 0:
            set_mobj_state(sim, actor, actor.info.spawnstate)
            return
        tid = 0


@action("A_DragonFlight")
def a_dragonflight(sim: GameSim, actor: Mobj) -> None:
    """Port of A_DragonFlight (p_enemy.c:2989)."""
    dragon_seek(sim, actor, 4 * ANG1, 8 * ANG1)
    if actor.target is not None:
        if not (actor.target.flags & MobjFlag.MF_SHOOTABLE):
            actor.target = None
            return
    else:
        look_for_players(sim, actor, True)
        return

    angle = point_to_angle2(actor.x, actor.y, actor.target.x, actor.target.y)
    if abs(to_int32(actor.angle - angle)) < (ANG45 // 2) and check_melee_range(sim, actor):
        damage = hitdice(sim, 8)
        damage_mobj(sim, actor.target, actor, actor, damage)
        _emit_sound(sim, actor, SfxEnum.SFX_DRAGON_ATTACK)
    elif abs(to_int32(actor.angle - angle)) <= ANG1 * 20:
        set_mobj_state(sim, actor, actor.info.missilestate)
        _emit_sound(sim, actor, SfxEnum.SFX_DRAGON_ATTACK)


@action("A_DragonFlap")
def a_dragonflap(sim: GameSim, actor: Mobj) -> None:
    """Port of A_DragonFlap (p_enemy.c:3025)."""
    a_dragonflight(sim, actor)
    if sim.rng.p_random() < 240:
        _emit_sound(sim, actor, SfxEnum.SFX_DRAGON_WINGFLAP)
    else:
        _emit_sound(sim, actor, actor.info.activesound)


@action("A_DragonAttack")
def a_dragonattack(sim: GameSim, actor: Mobj) -> None:
    """Port of A_DragonAttack (p_enemy.c:3045)."""
    if actor.target is None:
        return
    spawn_missile(sim, actor, actor.target, MobjType.MT_DRAGON_FX)


@action("A_DragonFX2")
def a_dragonfx2(sim: GameSim, actor: Mobj) -> None:
    """Port of A_DragonFX2 (p_enemy.c:3059)."""
    delay = 16 + (sim.rng.p_random() >> 3)
    for i in range(1 + (sim.rng.p_random() & 3)):
        x = actor.x + ((sim.rng.p_random() - 128) << 14)
        y = actor.y + ((sim.rng.p_random() - 128) << 14)
        z = actor.z + ((sim.rng.p_random() - 128) << 12)
        mo = spawn_mobj(sim, x, y, z, MobjType.MT_DRAGON_FX2)
        mo.tics = delay + (sim.rng.p_random() & 3) * i * 2
        mo.target = actor.target


@action("A_DragonPain")
def a_dragonpain(sim: GameSim, actor: Mobj) -> None:
    """Port of A_DragonPain (p_enemy.c:3083)."""
    if actor.info.painsound:
        _emit_sound(sim, actor, actor.info.painsound)
    if actor.tracer is None:
        set_mobj_state(sim, actor, StateNum.S_DRAGON_INIT)


@action("A_DragonCheckCrash")
def a_dragoncheckcrash(sim: GameSim, actor: Mobj) -> None:
    """Port of A_DragonCheckCrash (p_enemy.c:3100)."""
    if actor.z <= actor.floorz:
        set_mobj_state(sim, actor, StateNum.S_DRAGON_CRASH1)


# =============================================================================
# 2. Maulotaur (MT_MINOTAUR, MT_MNTRFX1..3) — p_enemy.c:1085-1513
# =============================================================================


@action("A_MinotaurFade0")
def a_minotaurfade0(actor: Mobj) -> None:
    """Port of A_MinotaurFade0 (p_enemy.c:1085)."""
    actor.flags &= ~MobjFlag.MF_ALTSHADOW
    actor.flags |= MobjFlag.MF_SHADOW


@action("A_MinotaurFade1")
def a_minotaurfade1(actor: Mobj) -> None:
    """Port of A_MinotaurFade1 (p_enemy.c:1098)."""
    actor.flags &= ~MobjFlag.MF_SHADOW
    actor.flags |= MobjFlag.MF_ALTSHADOW


@action("A_MinotaurFade2")
def a_minotaurfade2(actor: Mobj) -> None:
    """Port of A_MinotaurFade2 (p_enemy.c:1112)."""
    actor.flags &= ~MobjFlag.MF_SHADOW
    actor.flags &= ~MobjFlag.MF_ALTSHADOW


def _get_minotaur_starttime(mo: Mobj) -> int:
    """Extract Minotaur summon starttime from mo.starttime or mo.args."""
    st = getattr(mo, "starttime", None)
    if isinstance(st, int):
        return st
    a0 = _get_arg(mo, 0)
    a1 = _get_arg(mo, 1)
    a2 = _get_arg(mo, 2)
    a3 = _get_arg(mo, 3)
    if 0 <= a0 < 256 and (a1 or a2 or a3):
        return a0 | ((a1 & 0xFF) << 8) | ((a2 & 0xFF) << 16) | ((a3 & 0xFF) << 24)
    return a0


def check_minotaur_age(sim: GameSim, mo: Mobj) -> bool:
    """Port of CheckMinotaurAge (p_enemy.c:1203).

    Returns False and kills `mo` if its summon duration has expired.
    """
    starttime = _get_minotaur_starttime(mo)
    if sim.level_time - starttime >= MAULATORTICS:
        damage_mobj(sim, mo, None, None, 10000)
        return False
    return True


@action("A_MinotaurRoam")
def a_minotaurroam(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurRoam (p_enemy.c:1129)."""
    actor.flags &= ~MobjFlag.MF_SHADOW
    actor.flags &= ~MobjFlag.MF_ALTSHADOW

    if not check_minotaur_age(sim, actor):
        return

    if sim.rng.p_random() < 30:
        a_minotaurlook(sim, actor)
        if actor.target is not None:
            return

    if sim.rng.p_random() < 6:
        master = _resolve_mobj(actor.special1)
        if master is not None:
            actor.movedir = point_to_angle2(actor.x, actor.y, master.x, master.y) >> 29
    from pyhexen.sim.enemy import move, new_chase_dir

    if not move(actor, sim):
        new_chase_dir(actor, sim)


@action("A_MinotaurLook")
def a_minotaurlook(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurLook (p_enemy.c:1186)."""
    if not check_minotaur_age(sim, actor):
        return

    actor.target = None
    master = _resolve_mobj(actor.special1)

    # First check players in deathmatch
    if sim.deathmatch:
        for player in sim.players:
            if not getattr(player, "in_game", True):
                continue
            mo = player.mo
            if mo is None or mo is master or mo.health <= 0:
                continue
            dist = approx_distance(actor.x - mo.x, actor.y - mo.y)
            if dist > MINOTAUR_LOOK_DIST:
                continue
            actor.target = mo
            break

    if actor.target is None:
        if master is not None and master.health > 0 and master.player is not None:
            actor.target = rough_monster_search(sim, master, 20)
        else:
            actor.target = rough_monster_search(sim, actor, 20)

    if actor.target is None:
        for th in sim.thinkers:
            if not isinstance(th, Mobj):
                continue
            if not (th.flags & MobjFlag.MF_COUNTKILL):
                continue
            if th.health <= 0:
                continue
            if not (th.flags & MobjFlag.MF_SHOOTABLE):
                continue
            dist = approx_distance(actor.x - th.x, actor.y - th.y)
            if dist > MINOTAUR_LOOK_DIST:
                continue
            if th is master or th is actor:
                continue
            if th.type == MobjType.MT_MINOTAUR and _resolve_mobj(th.special1) is master:
                continue
            actor.target = th
            break

    if actor.target is not None:
        set_mobj_state_nf(sim, actor, StateNum.S_MNTR_WALK1)
    else:
        set_mobj_state_nf(sim, actor, StateNum.S_MNTR_ROAM1)


@action("A_MinotaurChase")
def a_minotaurchase(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurChase (p_enemy.c:1265)."""
    actor.flags &= ~MobjFlag.MF_SHADOW
    actor.flags &= ~MobjFlag.MF_ALTSHADOW

    if not check_minotaur_age(sim, actor):
        return

    if sim.rng.p_random() < 30:
        a_minotaurlook(sim, actor)

    if (
        actor.target is None
        or actor.target.health <= 0
        or not (actor.target.flags & MobjFlag.MF_SHOOTABLE)
    ):
        set_mobj_state(sim, actor, StateNum.S_MNTR_LOOK1)
        return

    actor.reactiontime = 0

    # Check melee attack
    if actor.info.meleestate and check_melee_range(sim, actor):
        if actor.info.attacksound:
            _emit_sound(sim, actor, actor.info.attacksound)
        set_mobj_state(sim, actor, actor.info.meleestate)
        return

    # Check missile attack
    if actor.info.missilestate:
        from pyhexen.sim.enemy import check_missile_range

        if check_missile_range(sim, actor):
            set_mobj_state(sim, actor, actor.info.missilestate)
            return

    from pyhexen.sim.enemy import move, new_chase_dir

    if not move(actor, sim):
        new_chase_dir(actor, sim)

    if actor.info.activesound and sim.rng.p_random() < 6:
        _emit_sound(sim, actor, actor.info.activesound)


@action("A_MinotaurAtk1")
def a_minotauratk1(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurAtk1 (p_enemy.c:1329)."""
    if actor.target is None:
        return
    _emit_sound(sim, actor, SfxEnum.SFX_MAULATOR_HAMMER_SWING)
    if check_melee_range(sim, actor):
        damage_mobj(sim, actor.target, actor, actor, hitdice(sim, 4))


@action("A_MinotaurDecide")
def a_minotaurdecide(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurDecide (p_enemy.c:1351)."""
    target = actor.target
    if target is None:
        return

    dist = approx_distance(actor.x - target.x, actor.y - target.y)
    if (
        target.z + target.height > actor.z
        and target.z + target.height < actor.z + actor.height
        and dist < 16 * 64 * FRACUNIT
        and dist > 1 * 64 * FRACUNIT
        and sim.rng.p_random() < 230
    ):
        # Charge attack
        set_mobj_state_nf(sim, actor, StateNum.S_MNTR_ATK4_1)
        actor.flags |= MobjFlag.MF_SKULLFLY
        from pyhexen.sim.enemy import a_facetarget

        a_facetarget(sim, actor)
        angle = (actor.angle >> ANGLETOFINESHIFT) & FINEMASK
        speed = 23 * FRACUNIT
        actor.momx = fixed_mul(speed, fine_cosine(angle))
        actor.momy = fixed_mul(speed, fine_sine(angle))
        _set_arg(actor, 4, 35 // 2)
        actor.special2 = 35 // 2
    elif target.z == target.floorz and dist < 9 * 64 * FRACUNIT and sim.rng.p_random() < 100:
        # Floor fire attack
        set_mobj_state(sim, actor, StateNum.S_MNTR_ATK3_1)
        actor.special2 = 0
    else:
        from pyhexen.sim.enemy import a_facetarget

        a_facetarget(sim, actor)


@action("A_MinotaurCharge")
def a_minotaurcharge(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurCharge (p_enemy.c:1395)."""
    if actor.target is None:
        return
    a4 = _get_arg(actor, 4)
    if a4 > 0:
        puff = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_PUNCHPUFF)
        puff.momz = 2 * FRACUNIT
        _set_arg(actor, 4, a4 - 1)
    else:
        actor.flags &= ~MobjFlag.MF_SKULLFLY
        set_mobj_state(sim, actor, actor.info.seestate)


@action("A_MinotaurAtk2")
def a_minotauratk2(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurAtk2 (p_enemy.c:1423). Swing spread fire attack."""
    target = actor.target
    if target is None:
        return
    _emit_sound(sim, actor, SfxEnum.SFX_MAULATOR_HAMMER_SWING)
    if check_melee_range(sim, actor):
        damage_mobj(sim, target, actor, actor, hitdice(sim, 3))
        return

    mo = spawn_missile(sim, actor, target, MobjType.MT_MNTRFX1)
    if mo is not None:
        momz = mo.momz
        angle = mo.angle
        spawn_missile_angle(sim, actor, MobjType.MT_MNTRFX1, to_uint32(angle - (ANG45 // 8)), momz)
        spawn_missile_angle(sim, actor, MobjType.MT_MNTRFX1, to_uint32(angle + (ANG45 // 8)), momz)
        spawn_missile_angle(sim, actor, MobjType.MT_MNTRFX1, to_uint32(angle - (ANG45 // 16)), momz)
        spawn_missile_angle(sim, actor, MobjType.MT_MNTRFX1, to_uint32(angle + (ANG45 // 16)), momz)


@action("A_MinotaurAtk3")
def a_minotauratk3(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MinotaurAtk3 (p_enemy.c:1459). Floor fire."""
    if actor.target is None:
        return
    if check_melee_range(sim, actor):
        damage_mobj(sim, actor.target, actor, actor, hitdice(sim, 3))
        player = actor.target.player
        if player is not None:
            player.deltaviewheight = -16 * FRACUNIT
    else:
        mo = spawn_missile(sim, actor, actor.target, MobjType.MT_MNTRFX2)
        if mo is not None:
            _emit_sound(sim, mo, SfxEnum.SFX_MAULATOR_HAMMER_HIT)

    if sim.rng.p_random() < 192 and int(actor.special2 or 0) == 0:
        set_mobj_state(sim, actor, StateNum.S_MNTR_ATK3_4)
        actor.special2 = 1


@action("A_MntrFloorFire")
def a_mntrfloorfire(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MntrFloorFire (p_enemy.c:1497)."""
    r1 = sim.rng.p_subrandom()
    r2 = sim.rng.p_subrandom()
    actor.z = actor.floorz
    x = actor.x + (r2 << 10)
    y = actor.y + (r1 << 10)
    mo = spawn_mobj(sim, x, y, ONFLOORZ, MobjType.MT_MNTRFX3)
    mo.target = actor.target
    mo.momx = 1  # Force block checking
    check_missile_spawn(sim, mo)


@action("A_SmokePuffExit")
def a_smokepuffexit(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SmokePuffExit (p_enemy.c:1519)."""
    spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_MNTRSMOKEEXIT)


# =============================================================================
# 3. Heresiarch (MT_SORCBOSS, MT_SORCBALL1..3, MT_SORCFX1..4) — p_enemy.c:3914-4578
# =============================================================================


@action("A_SorcSpinBalls")
def a_sorcspinballs(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcSpinBalls (p_enemy.c:3959)."""
    a_slowballs(actor)
    _set_arg(actor, 0, 0)  # Currently no defense
    _set_arg(actor, 3, SORC_NORMAL)
    _set_arg(actor, 4, SORCBALL_INITIAL_SPEED)
    actor.special1 = ANG1
    z = actor.z - actor.floorclip + actor.info.height

    mo = spawn_mobj(sim, actor.x, actor.y, z, MobjType.MT_SORCBALL1)
    mo.target = actor
    mo.special2 = BALL1_ANGLEOFFSET
    actor.args = tuple(list(actor.args)[:5])  # type: ignore[assignment]

    mo = spawn_mobj(sim, actor.x, actor.y, z, MobjType.MT_SORCBALL2)
    mo.target = actor
    mo.special2 = BALL2_ANGLEOFFSET

    mo = spawn_mobj(sim, actor.x, actor.y, z, MobjType.MT_SORCBALL3)
    mo.target = actor
    mo.special2 = BALL3_ANGLEOFFSET


@action("A_SorcBallOrbit")
def a_sorcballorbit(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcBallOrbit (p_enemy.c:3996)."""
    parent = actor.target
    if parent is None:
        return

    dist = parent.radius - (actor.radius << 1)
    prevangle = int(actor.special1 or 0)

    if parent.health <= 0:
        set_mobj_state(sim, actor, actor.info.painstate)

    mode = _get_arg(parent, 3)
    if mode == SORC_STOPPED:
        pass
    elif mode == SORC_NORMAL:
        a_sorcupdateballangle(actor)
    elif mode == SORC_ACCELERATE:
        a_accelballs(sim, actor)
    elif mode == SORC_DECELERATE:
        a_decelballs(actor)

    angle = (
        (to_uint32(int(parent.special1 or 0) + int(actor.special2 or 0))) >> ANGLETOFINESHIFT
    ) & FINEMASK
    actor.angle = to_uint32(angle << ANGLETOFINESHIFT)

    if (
        mode == SORC_ACCELERATE
        and _get_arg(parent, 4) >= SORCBALL_TERMINAL_SPEED
        and actor.type == int(_get_arg(parent, 2))
        and not (parent.health <= 0)
    ):
        parent_fine = ((parent.angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        if abs(angle - parent_fine) < (30 << 5):
            # Can stop now - cast spell
            _set_arg(parent, 3, SORC_STOPPED)
            set_mobj_state(sim, parent, StateNum.S_SORC_ATK4)

    # Set coordinates
    x = parent.x + fixed_mul(dist, fine_cosine(angle))
    y = parent.y + fixed_mul(dist, fine_sine(angle))
    z = parent.z - parent.floorclip + parent.info.height
    if _get_arg(parent, 0):  # Defense active
        z -= SORC_DEFENSE_HEIGHT * FRACUNIT

    actor.x = x
    actor.y = y
    actor.z = z
    actor.special1 = angle
    _ = prevangle


@action("A_SpeedBalls")
def a_speedballs(actor: Mobj) -> None:
    """Port of A_SpeedBalls (p_enemy.c:4082)."""
    _set_arg(actor, 3, SORC_ACCELERATE)
    _set_arg(actor, 2, int(MobjType.MT_SORCBALL1) + (actor.lastlook % 3))


@action("A_SlowBalls")
def a_slowballs(actor: Mobj) -> None:
    """Port of A_SlowBalls (p_enemy.c:4095)."""
    _set_arg(actor, 3, SORC_DECELERATE)


@action("A_StopBalls")
def a_stopballs(actor: Mobj) -> None:
    """Port of A_StopBalls (p_enemy.c:4109)."""
    parent = actor.target
    if parent is None:
        return
    _set_arg(parent, 3, SORC_STOPPED)
    _set_arg(parent, 1, 0)


@action("A_AccelBalls")
def a_accelballs(sim: GameSim, actor: Mobj) -> None:
    """Port of A_AccelBalls (p_enemy.c:4124)."""
    sorc = actor.target
    if sorc is None:
        return
    speed = _get_arg(sorc, 4)
    if speed < SORCBALL_TERMINAL_SPEED:
        speed += 1
        _set_arg(sorc, 4, speed)
    else:
        _set_arg(sorc, 4, SORCBALL_TERMINAL_SPEED)
        if speed >= SORCBALL_TERMINAL_SPEED:
            # Start rotation counter
            rotations = _get_arg(sorc, 1)
            if rotations == 0:
                _set_arg(sorc, 1, SORCBALL_SPEED_ROTATIONS)
    a_sorcupdateballangle(actor)
    _ = sim


@action("A_DecelBalls")
def a_decelballs(actor: Mobj) -> None:
    """Port of A_DecelBalls (p_enemy.c:4152)."""
    sorc = actor.target
    if sorc is None:
        return
    speed = _get_arg(sorc, 4)
    if speed > SORCBALL_INITIAL_SPEED:
        speed -= 1
        _set_arg(sorc, 4, speed)
    else:
        _set_arg(sorc, 4, SORCBALL_INITIAL_SPEED)
        _set_arg(sorc, 3, SORC_NORMAL)
    a_sorcupdateballangle(actor)


@action("A_SorcUpdateBallAngle")
def a_sorcupdateballangle(actor: Mobj) -> None:
    """Port of A_SorcUpdateBallAngle (p_enemy.c:4170)."""
    parent = actor.target
    if parent is None:
        return
    if actor.type == MobjType.MT_SORCBALL1:
        parent.special1 = to_uint32(int(parent.special1 or 0) + ANG1 * _get_arg(parent, 4))


@action("A_CastSorcererSpell")
def a_castsorcererspell(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CastSorcererSpell (p_enemy.c:4187)."""
    spell = _get_arg(actor, 2)
    _emit_sound(sim, None, SfxEnum.SFX_SORCERER_SPELLCAST, full_volume=True)

    # Put sorcerer into throw spell animation
    if spell == int(MobjType.MT_SORCBALL1) and actor.health > (actor.info.spawnhealth // 3):
        spell = int(MobjType.MT_SORCBALL2)
        _set_arg(actor, 2, spell)

    z = actor.z - actor.floorclip + actor.info.height - 30 * FRACUNIT

    if spell == int(MobjType.MT_SORCBALL1):
        # Summon Bishops
        mo = spawn_mobj(sim, actor.x, actor.y, z, MobjType.MT_SORCFX3)
        mo.target = actor
        mo.angle = actor.angle
        _set_arg(actor, 3, SORC_DECELERATE)
    elif spell == int(MobjType.MT_SORCBALL2):
        # Defensive reflective shield
        z = actor.z - actor.floorclip + SORC_DEFENSE_HEIGHT * FRACUNIT
        mo = spawn_mobj(sim, actor.x, actor.y, z, MobjType.MT_SORCFX2)
        actor.flags2 |= MobjFlag2.MF2_REFLECTIVE | MobjFlag2.MF2_INVULNERABLE
        _set_arg(actor, 0, SORC_DEFENSE_TIME)
        if mo is not None:
            mo.target = actor
        _set_arg(actor, 3, SORC_DECELERATE)
    elif spell == int(MobjType.MT_SORCBALL3):
        # Offensive spell
        if sim.rng.p_random() < 128:
            set_mobj_state(sim, actor, StateNum.S_SORC_ATTACK1)
        else:
            set_mobj_state(sim, actor, StateNum.S_SORC_ATTACK4)


@action("A_SorcOffense1")
def a_sorcoffense1(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcOffense1 (p_enemy.c:4260)."""
    if actor.target is None:
        return
    ang1 = to_uint32(actor.angle + ANG1 * 70)
    ang2 = to_uint32(actor.angle - ANG1 * 70)
    mo = spawn_missile_angle(sim, actor, MobjType.MT_SORCFX1, ang1, 0)
    if mo is not None:
        mo.target = actor
        mo.special1 = actor.target
        _set_arg(mo, 4, BOUNCE_TIME_UNIT)
        _set_arg(mo, 3, 15)
    mo = spawn_missile_angle(sim, actor, MobjType.MT_SORCFX1, ang2, 0)
    if mo is not None:
        mo.target = actor
        mo.special1 = actor.target
        _set_arg(mo, 4, BOUNCE_TIME_UNIT)
        _set_arg(mo, 3, 15)


@action("A_SorcOffense2")
def a_sorcoffense2(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcOffense2 (p_enemy.c:4289)."""
    if actor.target is None:
        return
    dest = actor.target
    index = (_get_arg(actor, 4) << 5) & FINEMASK
    _set_arg(actor, 4, (_get_arg(actor, 4) + 15) & 255)
    delta = fine_sine(index) * SORCFX4_SPREAD_ANGLE
    delta = (delta >> FRACBITS) * ANG1
    ang1 = to_uint32(actor.angle + delta)
    mo = spawn_missile_angle(sim, actor, MobjType.MT_SORCFX4, ang1, 0)
    if mo is not None:
        mo.special2 = 35 * 5 // 2
        dist = approx_distance(dest.x - mo.x, dest.y - mo.y)
        denom = mo.info.speed if mo.info.speed != 0 else 1
        dist = c_div(dist, denom)
        if dist < 1:
            dist = 1
        mo.momz = c_div(dest.z - mo.z, dist)


@action("A_SorcBossAttack")
def a_sorcbossattack(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcBossAttack (p_enemy.c:4325)."""
    _set_arg(actor, 4, 0)
    actor.special1 = to_uint32(actor.angle + ANG90)
    a_speedballs(actor)
    _ = sim


@action("A_SpawnFizzle")
def a_spawnfizzle(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SpawnFizzle (p_enemy.c:4340)."""
    dist = 5 * FRACUNIT
    angle = (actor.angle >> ANGLETOFINESHIFT) & FINEMASK
    speed = actor.info.speed
    x = actor.x + fixed_mul(dist, fine_cosine(angle))
    y = actor.y + fixed_mul(dist, fine_sine(angle))
    z = actor.z - actor.floorclip + (actor.height >> 1)
    for _ in range(5):
        mo = spawn_mobj(sim, x, y, z, MobjType.MT_SORCSPARK1)
        r_ang = (angle + ((sim.rng.p_random() % 5) << 1)) & FINEMASK
        mo.momx = fixed_mul(sim.rng.p_random() % speed, fine_cosine(r_ang))
        mo.momy = fixed_mul(sim.rng.p_random() % speed, fine_sine(r_ang))
        mo.momz = FRACUNIT * 2


@action("A_SorcFX1Seek")
def a_sorcfx1seek(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcFX1Seek (p_enemy.c:4372)."""
    a_bouncecheck(sim, actor)
    seeker_missile(actor, ANG1 * 2, ANG1 * 6, sim)


@action("A_SorcFX2Split")
def a_sorcfx2split(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcFX2Split (p_enemy.c:4389)."""
    parent = actor.target
    if parent is None:
        remove_mobj(sim, actor)
        return
    angle = parent.angle
    for _ in range(8):
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_SORCFX2_T1)
        mo.target = parent
        _set_arg(mo, 0, 0)  # Orbit direction = CCW
        mo.special1 = angle  # Set angle
        set_mobj_state_nf(sim, mo, StateNum.S_SORCFX2_ORBIT1)

        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_SORCFX2_T1)
        mo.target = parent
        _set_arg(mo, 0, 1)  # Orbit direction = CW
        mo.special1 = angle
        set_mobj_state_nf(sim, mo, StateNum.S_SORCFX2_ORBIT1)

        angle = to_uint32(angle + ANG45)

    set_mobj_state_nf(sim, actor, StateNum.S_NULL)


@action("A_SorcFX2Orbit")
def a_sorcfx2orbit(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcFX2Orbit (p_enemy.c:4425)."""
    parent = actor.target
    if parent is None or parent.health <= 0 or not _get_arg(parent, 0):
        set_mobj_state_nf(sim, actor, actor.info.deathstate)
        if parent is not None:
            parent.flags2 &= ~(MobjFlag2.MF2_REFLECTIVE | MobjFlag2.MF2_INVULNERABLE)
        return

    if _get_arg(actor, 0) and (_get_arg(parent, 0) - 1 <= 0):
        set_mobj_state_nf(sim, actor, actor.info.deathstate)
        parent.flags2 &= ~(MobjFlag2.MF2_REFLECTIVE | MobjFlag2.MF2_INVULNERABLE)
        _set_arg(parent, 0, 0)
        return

    dist = parent.info.radius
    if _get_arg(actor, 0):  # CW
        actor.special1 = to_uint32(int(actor.special1 or 0) + ANG1 * 10)
        angle = (int(actor.special1) >> ANGLETOFINESHIFT) & FINEMASK
        x = parent.x + fixed_mul(dist, fine_cosine(angle))
        y = parent.y + fixed_mul(dist, fine_sine(angle))
        z = parent.z - parent.floorclip + SORC_DEFENSE_HEIGHT * FRACUNIT
        z += fixed_mul(15 * FRACUNIT, fine_cosine(angle))
        spawn_mobj(sim, x, y, z, MobjType.MT_SORCFX2_T1)
        _set_arg(parent, 0, _get_arg(parent, 0) - 1)
    else:  # CCW
        actor.special1 = to_uint32(int(actor.special1 or 0) - ANG1 * 10)
        angle = (int(actor.special1) >> ANGLETOFINESHIFT) & FINEMASK
        x = parent.x + fixed_mul(dist, fine_cosine(angle))
        y = parent.y + fixed_mul(dist, fine_sine(angle))
        z = parent.z - parent.floorclip + 20 * FRACUNIT
        z += fixed_mul(20 * FRACUNIT, fine_sine(angle))
        spawn_mobj(sim, x, y, z, MobjType.MT_SORCFX2_T1)

    actor.x = x
    actor.y = y
    actor.z = z


@action("A_SorcererBishopEntry")
def a_sorcererbishopentry(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcererBishopEntry (p_enemy.c:4473)."""
    spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_SORCFX3_EXPLOSION)
    _emit_sound(sim, actor, actor.info.seesound)


@action("A_SpawnBishop")
def a_spawnbishop(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SpawnBishop (p_enemy.c:4485)."""
    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_BISHOP)
    if not p_test_mobj_location(sim, mo):
        set_mobj_state(sim, mo, StateNum.S_NULL)
    elif actor.target is not None:
        mo.target = actor.target.target
    set_mobj_state_nf(sim, actor, StateNum.S_NULL)


@action("A_SorcFX4Check")
def a_sorcfx4check(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcFX4Check (p_enemy.c:4525)."""
    s2 = int(actor.special2 or 0)
    actor.special2 = s2 - 1
    if s2 <= 0:
        set_mobj_state_nf(sim, actor, actor.info.deathstate)


@action("A_SorcBallPop")
def a_sorcballpop(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SorcBallPop (p_enemy.c:4525)."""
    _emit_sound(sim, None, SfxEnum.SFX_SORCERER_BALLPOP, full_volume=True)
    actor.flags &= ~MobjFlag.MF_NOGRAVITY
    actor.flags2 |= MobjFlag2.MF2_LOGRAV
    actor.momx = ((sim.rng.p_random() % 10) - 5) << FRACBITS
    actor.momy = ((sim.rng.p_random() % 10) - 5) << FRACBITS
    actor.momz = (2 + (sim.rng.p_random() % 3)) << FRACBITS
    actor.special2 = 4 * FRACUNIT
    _set_arg(actor, 4, BOUNCE_TIME_UNIT)
    _set_arg(actor, 3, 5)


@action("A_BounceCheck")
def a_bouncecheck(sim: GameSim, actor: Mobj) -> None:
    """Port of A_BounceCheck (p_enemy.c:4544)."""
    a4 = _get_arg(actor, 4)
    _set_arg(actor, 4, a4 - 1)
    if a4 <= 0:
        a3 = _get_arg(actor, 3)
        _set_arg(actor, 3, a3 - 1)
        if a3 <= 0:
            set_mobj_state(sim, actor, actor.info.deathstate)
            if actor.type in (
                MobjType.MT_SORCBALL1,
                MobjType.MT_SORCBALL2,
                MobjType.MT_SORCBALL3,
            ):
                _emit_sound(sim, None, SfxEnum.SFX_SORCERER_BIGBALLEXPLODE, full_volume=True)
            else:
                _emit_sound(sim, actor, SfxEnum.SFX_SORCERER_BALLEXPLODE)
        else:
            _set_arg(actor, 4, BOUNCE_TIME_UNIT)


# =============================================================================
# 4. Three Class Bosses & Weapons/Projectiles — p_enemy.c:4583-4770, p_pspr.c
# =============================================================================


@action("A_FighterAttack")
def a_fighterattack(sim: GameSim, actor: Mobj) -> None:
    """Port of A_FighterAttack (p_enemy.c:4618)."""
    if actor.target is None:
        return
    a_fswordattack2(sim, actor)


@action("A_FSwordAttack2")
def a_fswordattack2(sim: GameSim, actor: Mobj) -> None:
    """Port of A_FSwordAttack2 (p_pspr.c:840)."""
    angle = actor.angle
    for offset in (
        -(ANG45 // 4),
        -(ANG45 // 8),
        0,
        ANG45 // 8,
        ANG45 // 4,
    ):
        mo = spawn_missile_angle(
            sim,
            actor,
            MobjType.MT_FSWORD_MISSILE,
            to_uint32(angle + offset),
            0,
        )
        if mo is not None and actor.target is not None:
            dist = approx_distance(actor.target.x - mo.x, actor.target.y - mo.y)
            denom = mo.info.speed if mo.info.speed != 0 else 1
            dist = c_div(dist, denom)
            if dist < 1:
                dist = 1
            mo.momz = c_div(actor.target.z - actor.z, dist)
    _emit_sound(sim, actor, SfxEnum.SFX_FIGHTER_SWORD_FIRE)


@action("A_FSwordFlames")
def a_fswordflames(sim: GameSim, actor: Mobj) -> None:
    """Port of A_FSwordFlames (p_pspr.c:859)."""
    for _ in range(1 + (sim.rng.p_random() & 3)):
        x = actor.x + ((sim.rng.p_random() - 128) << 12)
        y = actor.y + ((sim.rng.p_random() - 128) << 12)
        z = actor.z + ((sim.rng.p_random() - 128) << 11)
        spawn_mobj(sim, x, y, z, MobjType.MT_FSWORD_FLAME)


@action("A_ClericAttack")
def a_clericattack(sim: GameSim, actor: Mobj) -> None:
    """Port of A_ClericAttack (p_enemy.c:4630)."""
    if actor.target is None:
        return
    a_cholyattack3(sim, actor)


@action("A_CHolyAttack3")
def a_cholyattack3(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CHolyAttack3 (p_pspr.c:1835)."""
    spawn_missile(sim, actor, actor.target, MobjType.MT_HOLY_MISSILE)
    _emit_sound(sim, actor, SfxEnum.SFX_CHOLY_FIRE)


@action("A_CHolyAttack2")
def a_cholyattack2(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CHolyAttack2 (p_pspr.c:1856). Spawns 4 spirits from MT_HOLY_MISSILE."""
    for j in range(4):
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_HOLY_FX)
        if j == 0:
            mo.special2 = sim.rng.p_random() & 7
        elif j == 1:
            mo.special2 = 32 + (sim.rng.p_random() & 7)
        elif j == 2:
            mo.special2 = (32 + (sim.rng.p_random() & 7)) << 16
        else:
            r = sim.rng.p_random()
            mo.special2 = ((32 + (r & 7)) << 16) + 32 + (sim.rng.p_random() & 7)

        mo.z = actor.z
        mo.angle = to_uint32(actor.angle + (ANG45 + ANG45 // 2) - ANG45 * j)
        fine_an = (mo.angle >> ANGLETOFINESHIFT) & FINEMASK
        mo.momx = fixed_mul(mo.info.speed, fine_cosine(fine_an))
        mo.momy = fixed_mul(mo.info.speed, fine_sine(fine_an))
        mo.target = actor.target
        _set_arg(mo, 0, 10)  # Initial turn value
        _set_arg(mo, 1, 0)  # Initial look angle
        if sim.deathmatch:
            mo.health = 85

        if actor.tracer is not None:
            mo.special1 = actor.tracer
            mo.flags |= MobjFlag.MF_NOCLIP | MobjFlag.MF_SKULLFLY
            mo.flags &= ~MobjFlag.MF_MISSILE

        tail = spawn_mobj(sim, mo.x, mo.y, mo.z, MobjType.MT_HOLY_TAIL)
        tail.special2 = mo
        for _ in range(1, 3):
            next_tail = spawn_mobj(sim, mo.x, mo.y, mo.z, MobjType.MT_HOLY_TAIL)
            set_mobj_state(sim, next_tail, next_tail.info.spawnstate + 1)
            tail.special1 = next_tail
            tail = next_tail
        tail.special1 = None


def choly_find_target(sim: GameSim, actor: Mobj) -> None:
    """Port of CHolyFindTarget (p_pspr.c:1879)."""
    target = rough_monster_search(sim, actor, 6)
    if target is not None:
        actor.special1 = target
        actor.flags |= MobjFlag.MF_NOCLIP | MobjFlag.MF_SKULLFLY
        actor.flags &= ~MobjFlag.MF_MISSILE


def choly_seeker_missile(sim: GameSim, actor: Mobj, thresh: int, turn_max: int) -> None:
    """Port of CHolySeekerMissile (p_pspr.c:1906)."""
    target = _resolve_mobj(actor.special1)
    if target is None:
        return
    if not (target.flags & MobjFlag.MF_SHOOTABLE) or (
        not (target.flags & MobjFlag.MF_COUNTKILL) and target.player is None
    ):
        actor.special1 = None
        actor.flags &= ~(MobjFlag.MF_NOCLIP | MobjFlag.MF_SKULLFLY)
        actor.flags |= MobjFlag.MF_MISSILE
        choly_find_target(sim, actor)
        return

    direction, delta = face_mobj(actor, target)
    if delta > thresh:
        delta >>= 1
        if delta > turn_max:
            delta = turn_max
    if direction:
        actor.angle = to_uint32(actor.angle + delta)
    else:
        actor.angle = to_uint32(actor.angle - delta)

    fine_an = (actor.angle >> ANGLETOFINESHIFT) & FINEMASK
    actor.momx = fixed_mul(actor.info.speed, fine_cosine(fine_an))
    actor.momy = fixed_mul(actor.info.speed, fine_sine(fine_an))

    if actor.z + actor.height < target.z or target.z + target.height < actor.z:
        dist = approx_distance(target.x - actor.x, target.y - actor.y)
        denom = actor.info.speed if actor.info.speed != 0 else 1
        dist = c_div(dist, denom)
        if dist < 1:
            dist = 1
        actor.momz = c_div((target.z + (target.height >> 1)) - actor.z, dist)


def choly_weave(sim: GameSim, actor: Mobj) -> None:
    """Port of CHolyWeave (p_pspr.c:1975)."""
    weave_xy = int(actor.special2 or 0) >> 16
    weave_z = int(actor.special2 or 0) & 0xFFFF
    angle = (to_uint32(actor.angle + ANG90) >> ANGLETOFINESHIFT) & FINEMASK
    new_x = actor.x - fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 2)
    new_y = actor.y - fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 2)
    weave_xy = (weave_xy + (sim.rng.p_random() % 5)) & 63
    new_x += fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 2)
    new_y += fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 2)
    from pyhexen.sim.map import try_move

    try_move(sim, actor, new_x, new_y)
    actor.z -= FLOAT_BOB_OFFSETS[weave_z & 63] << 1
    weave_z = (weave_z + (sim.rng.p_random() % 5)) & 63
    actor.z += FLOAT_BOB_OFFSETS[weave_z] << 1
    actor.special2 = weave_z + (weave_xy << 16)


@action("A_CHolySeek")
def a_cholyseek(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CHolySeek (p_pspr.c:2025)."""
    actor.health -= 1
    if actor.health <= 0:
        actor.momx >>= 2
        actor.momy >>= 2
        actor.momz = 0
        set_mobj_state(sim, actor, actor.info.deathstate)
        actor.tics -= sim.rng.p_random() & 3
        return

    if _resolve_mobj(actor.special1) is not None:
        choly_seeker_missile(
            sim,
            actor,
            _get_arg(actor, 0) * ANG1,
            _get_arg(actor, 0) * ANG1 * 2,
        )
        if not ((sim.level_time + 7) & 15):
            _set_arg(actor, 0, 5 + (sim.rng.p_random() // 20))
    else:
        choly_find_target(sim, actor)

    choly_weave(sim, actor)


def choly_tail_follow(actor: Mobj, dist: int) -> None:
    """Port of CHolyTailFollow (p_pspr.c:2102)."""
    child = _resolve_mobj(actor.special1)
    if child is not None:
        an = point_to_angle2(actor.x, actor.y, child.x, child.y)
        fine_an = ((an & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        child.x = actor.x + fixed_mul(dist, fine_cosine(fine_an))
        child.y = actor.y + fixed_mul(dist, fine_sine(fine_an))
        child.z = actor.z - 2 * FRACUNIT
        choly_tail_follow(child, dist - FRACUNIT)


def choly_tail_remove(sim: GameSim, actor: Mobj) -> None:
    """Port of CHolyTailRemove (p_pspr.c:2131)."""
    child = _resolve_mobj(actor.special1)
    if child is not None:
        choly_tail_remove(sim, child)
    remove_mobj(sim, actor)


@action("A_CHolyTail")
def a_cholytail(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CHolyTail (p_pspr.c:2145)."""
    parent = _resolve_mobj(actor.special2)
    if parent is not None and not parent.removed:
        if parent.state_num >= parent.info.deathstate:
            choly_tail_remove(sim, actor)
            return
        fine_an = ((parent.angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        actor.x = parent.x - fixed_mul(14 * FRACUNIT, fine_cosine(fine_an))
        actor.y = parent.y - fixed_mul(14 * FRACUNIT, fine_sine(fine_an))
        actor.z = parent.z - 2 * FRACUNIT
        choly_tail_follow(actor, 10 * FRACUNIT)
    else:
        choly_tail_remove(sim, actor)


@action("A_CHolyCheckScream")
def a_cholycheckscream(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CHolyCheckScream (p_pspr.c:2192)."""
    a_cholyseek(sim, actor)
    if sim.rng.p_random() < 20:
        _emit_sound(sim, actor, SfxEnum.SFX_SPIRIT_ACTIVE)
    if _resolve_mobj(actor.special1) is None:
        choly_find_target(sim, actor)


@action("A_CHolySpawnPuff")
def a_cholyspawnpuff(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CHolySpawnPuff (p_pspr.c:2212)."""
    spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_HOLY_MISSILE_PUFF)


@action("A_MageAttack")
def a_mageattack(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MageAttack (p_enemy.c:4642)."""
    if actor.target is None:
        return
    a_mstaffattack2(sim, actor)


def mstaff_spawn2(sim: GameSim, source: Mobj, angle: int) -> None:
    """Port of MStaffSpawn2 (p_pspr.c:1275)."""
    mo = spawn_missile_angle(sim, source, MobjType.MT_MSTAFF_FX2, to_uint32(angle), 0)
    if mo is not None:
        mo.target = source
        mo.special1 = rough_monster_search(sim, mo, 10)


@action("A_MStaffAttack2")
def a_mstaffattack2(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MStaffAttack2 (p_pspr.c:1291)."""
    angle = actor.angle
    mstaff_spawn2(sim, actor, angle)
    mstaff_spawn2(sim, actor, to_uint32(angle - ANG1 * 5))
    mstaff_spawn2(sim, actor, to_uint32(angle + ANG1 * 5))
    _emit_sound(sim, actor, SfxEnum.SFX_MAGE_STAFF_FIRE)


@action("A_MStaffWeave")
def a_mstaffweave(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MStaffWeave (p_pspr.c:1224)."""
    weave_xy = int(actor.special2 or 0) >> 16
    weave_z = int(actor.special2 or 0) & 0xFFFF
    angle = (to_uint32(actor.angle + ANG90) >> ANGLETOFINESHIFT) & FINEMASK
    new_x = actor.x - fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 2)
    new_y = actor.y - fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 2)
    weave_xy = (weave_xy + 3) & 63
    new_x += fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 2)
    new_y += fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 2)
    from pyhexen.sim.map import try_move

    try_move(sim, actor, new_x, new_y)
    actor.z -= FLOAT_BOB_OFFSETS[weave_z & 63] << 1
    weave_z = (weave_z + 3) & 63
    actor.z += FLOAT_BOB_OFFSETS[weave_z] << 1
    if actor.z <= actor.floorz:
        actor.z = actor.floorz + FRACUNIT
    actor.special2 = weave_z + (weave_xy << 16)


@action("A_MStaffTrack")
def a_mstafftrack(sim: GameSim, actor: Mobj) -> None:
    """Port of A_MStaffTrack (p_pspr.c:1259)."""
    if _resolve_mobj(actor.special1) is None and (sim.rng.p_random() < 50):
        actor.special1 = rough_monster_search(sim, actor, 10)
    seeker_missile(actor, ANG1 * 2, ANG1 * 10, sim)


@action("A_ClassBossHealth")
def a_classbosshealth(sim: GameSim, actor: Mobj) -> None:
    """Port of A_ClassBossHealth (p_enemy.c:4654)."""
    if sim.netgame and not sim.deathmatch:
        if not int(actor.special1 or 0):
            actor.health *= 5
            actor.special1 = 1


@action("A_SkullPop")
def a_skullpop(sim: GameSim, actor: Mobj) -> None:
    """Port of A_SkullPop (p_enemy.c:1783)."""
    if actor.player is None:
        return
    actor.flags &= ~MobjFlag.MF_SOLID
    mo = spawn_mobj(sim, actor.x, actor.y, actor.z + 48 * FRACUNIT, MobjType.MT_BLOODYSKULL)
    mo.momx = sim.rng.p_subrandom() << 9
    mo.momy = sim.rng.p_subrandom() << 9
    mo.momz = FRACUNIT * 2 + (sim.rng.p_random() << 6)
    player = actor.player
    actor.player = None
    actor.special1 = getattr(player, "player_class", getattr(player, "pclass", 0))
    mo.player = player
    mo.health = actor.health
    mo.angle = actor.angle
    player.mo = mo
    player.lookdir = 0
    player.damagecount = 32


@action("A_CheckSkullFloor")
def a_checkskullfloor(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CheckSkullFloor (p_enemy.c:1817)."""
    if actor.z <= actor.floorz:
        set_mobj_state(sim, actor, StateNum.S_BLOODYSKULLX1)
        _emit_sound(sim, actor, SfxEnum.SFX_DRIP)


@action("A_CheckSkullDone")
def a_checkskulldone(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CheckSkullDone (p_enemy.c:1832)."""
    if int(actor.special2 or 0) == 666:
        set_mobj_state(sim, actor, StateNum.S_BLOODYSKULLX2)


@action("A_CheckBurnGone")
def a_checkburngone(sim: GameSim, actor: Mobj) -> None:
    """Port of A_CheckBurnGone (p_enemy.c:1846)."""
    if int(actor.special2 or 0) == 666:
        set_mobj_state(sim, actor, StateNum.S_PLAY_FDTH20)


# =============================================================================
# 5. Korax (MT_KORAX, MT_KORAX_SPIRIT1..6, MT_KORAX_BOLT) — p_enemy.c:4915-5428
# =============================================================================


@action("A_KoraxChase")
def a_koraxchase(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KoraxChase (p_enemy.c:4939)."""
    if not int(actor.special2 or 0) and sim.netgame and not sim.deathmatch:
        actor.health *= 5
        actor.special2 = 1

    a_chase(sim, actor)

    if sim.rng.p_random() < 30:
        set_mobj_state(sim, actor, StateNum.S_KORAX_MISSILE1)
    elif sim.rng.p_random() < 30:
        _emit_sound(sim, None, SfxEnum.SFX_KORAX_ACTIVE, full_volume=True)

    # Teleport away when health drops below half
    if actor.health < (actor.info.spawnhealth >> 1):
        if not int(actor.special1 or 0):
            spots = find_mobj_from_tid(sim, KORAX_FIRST_TELEPORT_TID)
            if spots:
                p_teleport(sim, actor, spots[0].x, spots[0].y, spots[0].angle, False)
            actor.special1 = 1


@action("A_KoraxStep")
def a_koraxstep(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KoraxStep (p_enemy.c:4979)."""
    _emit_sound(sim, actor, SfxEnum.SFX_KORAX_STEP)


@action("A_KoraxStep2")
def a_koraxstep2(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KoraxStep2 (p_enemy.c:4989)."""
    _emit_sound(sim, actor, SfxEnum.SFX_KORAX_STEP)
    a_chase(sim, actor)


def kspirit_init(spirit: Mobj, korax: Mobj) -> None:
    """Port of KSpiritInit (p_enemy.c:5065)."""
    spirit.health = KORAX_SPIRIT_LIFETIME
    spirit.tracer = korax
    spirit.special1 = (32 + (korax.lastlook & 7)) * FRACUNIT  # Floatbob index in args[0], or weave
    _set_arg(spirit, 0, 10)  # Turn value
    _set_arg(spirit, 1, 0)  # Initial look angle
    # Tailor tail for spirit
    spirit.special2 = 32 + (32 << 16)


@action("A_KoraxBonePop")
def a_koraxbonepop(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KoraxBonePop (p_enemy.c:5035)."""
    for i in range(6):
        ang = to_uint32((0xFFFFFFFF // 6) * i)
        mo = spawn_missile_angle(sim, actor, MobjType.MT_KORAX_SPIRIT1 + i, ang, 5 * FRACUNIT)
        if mo is not None:
            kspirit_init(mo, actor)


@action("A_KoraxDecide")
def a_koraxdecide(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KoraxDecide (p_enemy.c:5090)."""
    if sim.rng.p_random() < 220:
        set_mobj_state(sim, actor, StateNum.S_KORAX_ATTACK1)
    else:
        set_mobj_state(sim, actor, StateNum.S_KORAX_COMMAND1)


def _korax_fire(sim: GameSim, actor: Mobj, mtype: int | MobjType) -> None:
    """Port of KoraxFire1..6 (p_enemy.c:5254-5303). Spawns 6 projectiles from Korax's 6 arms."""
    if actor.target is None:
        return
    arm_Extension: tuple[tuple[int, int], ...] = (
        (40, -55),
        (55, -40),
        (43, -29),
        (40, 55),
        (55, 40),
        (43, 29),
    )
    arm_Height: tuple[int, ...] = (108, 82, 54, 108, 82, 54)

    ang = (actor.angle >> ANGLETOFINESHIFT) & FINEMASK
    cos_a = fine_cosine(ang)
    sin_a = fine_sine(ang)
    ang_perp = (to_uint32(actor.angle - ANG90) >> ANGLETOFINESHIFT) & FINEMASK
    cos_p = fine_cosine(ang_perp)
    sin_p = fine_sine(ang_perp)

    for i in range(6):
        ext_fwd, ext_side = arm_Extension[i]
        x = actor.x + fixed_mul(ext_fwd * FRACUNIT, cos_a) + fixed_mul(ext_side * FRACUNIT, cos_p)
        y = actor.y + fixed_mul(ext_fwd * FRACUNIT, sin_a) + fixed_mul(ext_side * FRACUNIT, sin_p)
        z = actor.z - actor.floorclip + arm_Height[i] * FRACUNIT
        spawn_korax_missile(sim, x, y, z, actor, actor.target, mtype)


def korax_fire1(sim: GameSim, actor: Mobj) -> None:
    """Port of KoraxFire1 (p_enemy.c:5254) — Wraith FX1."""
    _korax_fire(sim, actor, MobjType.MT_WRAITHFX1)


def korax_fire2(sim: GameSim, actor: Mobj) -> None:
    """Port of KoraxFire2 (p_enemy.c:5263) — Demon FX1."""
    _korax_fire(sim, actor, MobjType.MT_DEMONFX1)


def korax_fire3(sim: GameSim, actor: Mobj) -> None:
    """Port of KoraxFire3 (p_enemy.c:5272) — Demon2 FX1."""
    _korax_fire(sim, actor, MobjType.MT_DEMON2FX1)


def korax_fire4(sim: GameSim, actor: Mobj) -> None:
    """Port of KoraxFire4 (p_enemy.c:5281) — FireDemon FX3."""
    _korax_fire(sim, actor, MobjType.MT_FIREDEMON_FX3)


def korax_fire5(sim: GameSim, actor: Mobj) -> None:
    """Port of KoraxFire5 (p_enemy.c:5290) — Centaur FX."""
    _korax_fire(sim, actor, MobjType.MT_CENTAUR_FX)


def korax_fire6(sim: GameSim, actor: Mobj) -> None:
    """Port of KoraxFire6 (p_enemy.c:5299) — Serpent FX."""
    _korax_fire(sim, actor, MobjType.MT_SERPENTFX)


@action("A_KoraxMissile")
def a_koraxmissile(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KoraxMissile (p_enemy.c:5127)."""
    if actor.target is None:
        return
    fire_type = sim.rng.p_random() % 6
    _emit_sound(sim, actor, SfxEnum.SFX_KORAX_ATTACK)
    if fire_type == 0:
        korax_fire1(sim, actor)
    elif fire_type == 1:
        korax_fire2(sim, actor)
    elif fire_type == 2:
        korax_fire3(sim, actor)
    elif fire_type == 3:
        korax_fire4(sim, actor)
    elif fire_type == 4:
        korax_fire5(sim, actor)
    else:
        korax_fire6(sim, actor)


@action("A_KoraxCommand")
def a_koraxcommand(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KoraxCommand (p_enemy.c:5177)."""
    _emit_sound(sim, actor, SfxEnum.SFX_KORAX_COMMAND)

    # Shoot stream of lightning to ceiling
    ang = (to_uint32(actor.angle - ANG90) >> ANGLETOFINESHIFT) & FINEMASK
    x = actor.x + KORAX_COMMAND_OFFSET * fine_cosine(ang)
    y = actor.y + KORAX_COMMAND_OFFSET * fine_sine(ang)
    z = actor.z + KORAX_COMMAND_HEIGHT * FRACUNIT
    spawn_mobj(sim, x, y, z, MobjType.MT_KORAX_BOLT)

    # Trigger Korax command scripts (250..254)
    num_commands = 4 if (actor.health > (actor.info.spawnhealth >> 1)) else 5
    cmd = sim.rng.p_random() % num_commands
    if cmd == 0:
        execute_line_special(sim, 80, [250, 0, 0, 0, 0], None, actor, side=0)
    elif cmd == 1:
        execute_line_special(sim, 80, [251, 0, 0, 0, 0], None, actor, side=0)
    elif cmd == 2:
        execute_line_special(sim, 80, [252, 0, 0, 0, 0], None, actor, side=0)
    elif cmd == 3:
        execute_line_special(sim, 80, [253, 0, 0, 0, 0], None, actor, side=0)
    else:
        execute_line_special(sim, 80, [254, 0, 0, 0, 0], None, actor, side=0)


def a_kspiritweave(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KSpiritWeave (p_enemy.c:5316)."""
    weave_xy = int(actor.special2 or 0) >> 16
    weave_z = int(actor.special2 or 0) & 0xFFFF
    angle = (to_uint32(actor.angle + ANG90) >> ANGLETOFINESHIFT) & FINEMASK
    new_x = actor.x - fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 2)
    new_y = actor.y - fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy & 63] << 2)
    weave_xy = (weave_xy + (sim.rng.p_random() % 5)) & 63
    new_x += fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 2)
    new_y += fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy] << 2)
    from pyhexen.sim.map import try_move

    try_move(sim, actor, new_x, new_y)
    actor.z -= FLOAT_BOB_OFFSETS[weave_z & 63] << 1
    weave_z = (weave_z + (sim.rng.p_random() % 5)) & 63
    actor.z += FLOAT_BOB_OFFSETS[weave_z] << 1
    actor.special2 = weave_z + (weave_xy << 16)


def a_kspiritseeker(sim: GameSim, actor: Mobj, thresh: int, turn_max: int) -> None:
    """Port of A_KSpiritSeeker (p_enemy.c:5350)."""
    target = actor.tracer
    if target is None:
        return
    direction, delta = face_mobj(actor, target)
    if delta > thresh:
        delta >>= 1
        if delta > turn_max:
            delta = turn_max
    if direction:
        actor.angle = to_uint32(actor.angle + delta)
    else:
        actor.angle = to_uint32(actor.angle - delta)

    fine_an = (actor.angle >> ANGLETOFINESHIFT) & FINEMASK
    actor.momx = fixed_mul(actor.info.speed, fine_cosine(fine_an))
    actor.momy = fixed_mul(actor.info.speed, fine_sine(fine_an))

    if not (sim.level_time & 15):
        new_z = target.z + (sim.rng.p_random() << FRACBITS)
        dist = approx_distance(target.x - actor.x, target.y - actor.y)
        denom = actor.info.speed if actor.info.speed != 0 else 1
        dist = c_div(dist, denom)
        if dist < 1:
            dist = 1
        actor.momz = c_div(new_z - actor.z, dist)
    _ = sim


@action("A_KSpiritRoam")
def a_kspiritroam(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KSpiritRoam (p_enemy.c:5380)."""
    if actor.health <= 0:
        _emit_sound(sim, actor, SfxEnum.SFX_SPIRIT_DIE)
        set_mobj_state(sim, actor, StateNum.S_KSPIRIT_DEATH1)
        return
    actor.health -= 1
    if actor.tracer is not None:
        a_kspiritseeker(
            sim,
            actor,
            _get_arg(actor, 0) * ANG1,
            _get_arg(actor, 0) * ANG1 * 2,
        )
    a_kspiritweave(sim, actor)
    if sim.rng.p_random() < 50:
        _emit_sound(sim, actor, SfxEnum.SFX_SPIRIT_ACTIVE)


@action("A_KBolt")
def a_kbolt(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KBolt (p_enemy.c:5401)."""
    s1 = int(actor.special1 or 0)
    if s1 <= 0:
        remove_mobj(sim, actor)
    else:
        actor.special1 = s1 - 1


@action("A_KBoltRaise")
def a_kboltraise(sim: GameSim, actor: Mobj) -> None:
    """Port of A_KBoltRaise (p_enemy.c:5417)."""
    z = actor.z + KORAX_COMMAND_HEIGHT * FRACUNIT
    if (z + KORAX_COMMAND_HEIGHT * FRACUNIT) < actor.ceilingz:
        mo = spawn_mobj(sim, actor.x, actor.y, z, MobjType.MT_KORAX_BOLT)
        mo.special1 = 15


__all__ = [
    "KORAX_COMMAND_HEIGHT",
    "KORAX_COMMAND_OFFSET",
    "KORAX_FIRST_TELEPORT_TID",
    "KORAX_SPIRIT_LIFETIME",
    "KORAX_TELEPORT_TID",
    "KORAX_TID",
    "MAULATORTICS",
    "MINOTAUR_LOOK_DIST",
    "SORCBALL_INITIAL_SPEED",
    "SORCBALL_SPEED_ROTATIONS",
    "SORCBALL_TERMINAL_SPEED",
    "SORC_ACCELERATE",
    "SORC_DECELERATE",
    "SORC_DEFENSE_HEIGHT",
    "SORC_DEFENSE_TIME",
    "SORC_NORMAL",
    "SORC_STOPPED",
    "a_accelballs",
    "a_bouncecheck",
    "a_castsorcererspell",
    "a_cholyattack2",
    "a_cholyattack3",
    "a_cholycheckscream",
    "a_cholyseek",
    "a_cholyspawnpuff",
    "a_cholytail",
    "a_classbosshealth",
    "a_clericattack",
    "a_decelballs",
    "a_dragonattack",
    "a_dragoncheckcrash",
    "a_dragonflap",
    "a_dragonflight",
    "a_dragonfx2",
    "a_dragoninitflight",
    "a_dragonpain",
    "a_fighterattack",
    "a_fswordattack2",
    "a_fswordflames",
    "a_kbolt",
    "a_kboltraise",
    "a_koraxbonepop",
    "a_koraxchase",
    "a_koraxcommand",
    "a_koraxdecide",
    "a_koraxmissile",
    "a_koraxstep",
    "a_koraxstep2",
    "a_kspiritroam",
    "a_kspiritseeker",
    "a_kspiritweave",
    "a_mageattack",
    "a_minotauratk1",
    "a_minotauratk2",
    "a_minotauratk3",
    "a_minotaurcharge",
    "a_minotaurchase",
    "a_minotaurdecide",
    "a_minotaurfade0",
    "a_minotaurfade1",
    "a_minotaurfade2",
    "a_minotaurlook",
    "a_minotaurroam",
    "a_mntrfloorfire",
    "a_mstaffattack2",
    "a_mstafftrack",
    "a_mstaffweave",
    "a_slowballs",
    "a_smokepuffexit",
    "a_sorcballorbit",
    "a_sorcballpop",
    "a_sorcbossattack",
    "a_sorcererbishopentry",
    "a_sorcfx1seek",
    "a_sorcfx2orbit",
    "a_sorcfx2split",
    "a_sorcfx4check",
    "a_sorcoffense1",
    "a_sorcoffense2",
    "a_sorcspinballs",
    "a_sorcupdateballangle",
    "a_spawnbishop",
    "a_spawnfizzle",
    "a_speedballs",
    "a_stopballs",
    "check_minotaur_age",
    "choly_find_target",
    "choly_seeker_missile",
    "choly_tail_follow",
    "choly_tail_remove",
    "choly_weave",
    "dragon_seek",
    "korax_fire1",
    "korax_fire2",
    "korax_fire3",
    "korax_fire4",
    "korax_fire5",
    "korax_fire6",
    "kspirit_init",
    "mstaff_spawn2",
    "p_teleport",
    "p_test_mobj_location",
    "rough_block_check",
    "rough_monster_search",
    "spawn_korax_missile",
]
