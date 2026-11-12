# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Thing line specials: spawn, projectile, activate, deactivate, remove, destroy (T15.8).

Port of p_things.c, plus the specials of P_ExecuteLineSpecial (p_spec.c) that act on
things or the activator: DamageThing (73), Radius_Quake (120, A_LocalQuake in
a_action.c) and UsePuzzleItem (129, EV_LineSearchForPuzzleItem).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import ANGLETOFINESHIFT, fine_cosine, fine_sine
from pyhexen.core.fixed import fixed_mul
from pyhexen.sim.acs.vm import TRANSLATE_THING_TYPE
from pyhexen.sim.info_data import (
    MOBJINFO,
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.mobj import Mobj, remove_mobj, set_mobj_state, spawn_mobj
from pyhexen.sim.sound import start_sound

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import Line

TELEFOGHEIGHT = 32 << 16


def _mobjs_from_tid(sim: GameSim, tid: int) -> Iterator[Mobj]:
    """Walk the things with `tid` like a P_FindMobjFromTID loop (p_mobj.c).

    Things removed while the loop runs are skipped, as they have left the TID list.
    """
    for mobj in sim.find_mobjs_by_tid(tid):
        if not mobj.removed and mobj.tid == tid:
            yield mobj


def _translate_thing_type(sim: GameSim, index: int) -> int | None:
    """TranslateThingType[index], or None when -nomonsters forbids it.

    An index past the table (vanilla reads out of bounds) also gives None.
    """
    if not 0 <= index < len(TRANSLATE_THING_TYPE):
        return None
    mo_type = int(TRANSLATE_THING_TYPE[index])
    if sim.nomonsters and (MOBJINFO[mo_type].flags & MobjFlag.MF_COUNTKILL):
        return None  # Don't spawn monsters if -nomonsters
    return mo_type


def ev_thing_projectile(sim: GameSim, args: Sequence[int], gravity: bool) -> bool:
    """Port of EV_ThingProjectile (p_things.c).

    `args`: tid of the spots, thing type, angle (byte), speed and vertical speed (1/8 units).
    """
    from pyhexen.sim.attacks import check_missile_spawn

    mo_type = _translate_thing_type(sim, args[1])
    if mo_type is None:
        return False
    success = False
    angle = (args[2] << 24) & 0xFFFFFFFF
    fine_angle = angle >> ANGLETOFINESHIFT
    speed = args[3] << 13
    vspeed = args[4] << 13
    for mobj in _mobjs_from_tid(sim, args[0]):
        new_mobj = spawn_mobj(sim, mobj.x, mobj.y, mobj.z, mo_type)
        if new_mobj.info.seesound:
            start_sound(sim, new_mobj, new_mobj.info.seesound)
        new_mobj.target = mobj  # Originator
        new_mobj.angle = angle
        new_mobj.momx = fixed_mul(speed, fine_cosine(fine_angle))
        new_mobj.momy = fixed_mul(speed, fine_sine(fine_angle))
        new_mobj.momz = vspeed
        new_mobj.flags2 |= MobjFlag2.MF2_DROPPED  # Don't respawn
        if gravity:
            new_mobj.flags &= ~MobjFlag.MF_NOGRAVITY
            new_mobj.flags2 |= MobjFlag2.MF2_LOGRAV
        if check_missile_spawn(sim, new_mobj):
            success = True
    return success


def ev_thing_spawn(sim: GameSim, args: Sequence[int], fog: bool) -> bool:
    """Port of EV_ThingSpawn (p_things.c).

    `args`: tid of the spots, thing type, angle (byte). A thing that does not fit is removed.
    """
    from pyhexen.sim.actions.monsters_bosses import p_test_mobj_location

    mo_type = _translate_thing_type(sim, args[1])
    if mo_type is None:
        return False
    success = False
    angle = (args[2] << 24) & 0xFFFFFFFF
    for mobj in _mobjs_from_tid(sim, args[0]):
        if MOBJINFO[mo_type].flags2 & MobjFlag2.MF2_FLOATBOB:
            z = mobj.z - mobj.floorz
        else:
            z = mobj.z
        new_mobj = spawn_mobj(sim, mobj.x, mobj.y, z, mo_type)
        if not p_test_mobj_location(sim, new_mobj):
            remove_mobj(sim, new_mobj)  # Didn't fit
        else:
            new_mobj.angle = angle
            if fog:
                fog_mobj = spawn_mobj(sim, mobj.x, mobj.y, mobj.z + TELEFOGHEIGHT, MobjType.MT_TFOG)
                start_sound(sim, fog_mobj, int(SfxEnum.SFX_TELEPORT))
            new_mobj.flags2 |= MobjFlag2.MF2_DROPPED  # Don't respawn
            if new_mobj.flags2 & MobjFlag2.MF2_FLOATBOB:
                new_mobj.special1 = new_mobj.z - new_mobj.floorz
            success = True
    return success


def _activate_thing(sim: GameSim, mobj: Mobj) -> bool:
    """Port of ActivateThing (p_things.c)."""
    from pyhexen.sim.inter import damage_mobj

    if mobj.flags & MobjFlag.MF_COUNTKILL:  # Monster
        if mobj.flags2 & MobjFlag2.MF2_DORMANT:
            mobj.flags2 &= ~MobjFlag2.MF2_DORMANT
            mobj.tics = 1
            return True
        return False
    mtype = mobj.type
    if mtype in (MobjType.MT_ZTWINEDTORCH, MobjType.MT_ZTWINEDTORCH_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZTWINEDTORCH_1)
        start_sound(sim, mobj, int(SfxEnum.SFX_IGNITE))
    elif mtype in (MobjType.MT_ZWALLTORCH, MobjType.MT_ZWALLTORCH_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZWALLTORCH1)
        start_sound(sim, mobj, int(SfxEnum.SFX_IGNITE))
    elif mtype == MobjType.MT_ZGEMPEDESTAL:
        set_mobj_state(sim, mobj, StateNum.S_ZGEMPEDESTAL2)
    elif mtype == MobjType.MT_ZWINGEDSTATUENOSKULL:
        set_mobj_state(sim, mobj, StateNum.S_ZWINGEDSTATUENOSKULL2)
    elif mtype in (MobjType.MT_THRUSTFLOOR_UP, MobjType.MT_THRUSTFLOOR_DOWN):
        if mobj.args[0] == 0:
            start_sound(sim, mobj, int(SfxEnum.SFX_THRUSTSPIKE_LOWER))
            mobj.flags2 &= ~MobjFlag2.MF2_DONTDRAW
            if mobj.args[1]:
                set_mobj_state(sim, mobj, StateNum.S_BTHRUSTRAISE1)
            else:
                set_mobj_state(sim, mobj, StateNum.S_THRUSTRAISE1)
    elif mtype in (MobjType.MT_ZFIREBULL, MobjType.MT_ZFIREBULL_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZFIREBULL_BIRTH)
        start_sound(sim, mobj, int(SfxEnum.SFX_IGNITE))
    elif mtype == MobjType.MT_ZBELL:
        if mobj.health > 0:
            damage_mobj(sim, mobj, None, None, 10)  # 'ring' the bell
    elif mtype in (MobjType.MT_ZCAULDRON, MobjType.MT_ZCAULDRON_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZCAULDRON1)
        start_sound(sim, mobj, int(SfxEnum.SFX_IGNITE))
    elif mtype == MobjType.MT_FLAME_SMALL:
        start_sound(sim, mobj, int(SfxEnum.SFX_IGNITE))
        set_mobj_state(sim, mobj, StateNum.S_FLAME_SMALL1)
    elif mtype == MobjType.MT_FLAME_LARGE:
        start_sound(sim, mobj, int(SfxEnum.SFX_IGNITE))
        set_mobj_state(sim, mobj, StateNum.S_FLAME_LARGE1)
    elif mtype == MobjType.MT_BAT_SPAWNER:
        set_mobj_state(sim, mobj, StateNum.S_SPAWNBATS1)
    else:
        return False
    return True


def _deactivate_thing(sim: GameSim, mobj: Mobj) -> bool:
    """Port of DeactivateThing (p_things.c)."""
    if mobj.flags & MobjFlag.MF_COUNTKILL:  # Monster
        if not (mobj.flags2 & MobjFlag2.MF2_DORMANT):
            mobj.flags2 |= MobjFlag2.MF2_DORMANT
            mobj.tics = -1
            return True
        return False
    mtype = mobj.type
    if mtype in (MobjType.MT_ZTWINEDTORCH, MobjType.MT_ZTWINEDTORCH_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZTWINEDTORCH_UNLIT)
    elif mtype in (MobjType.MT_ZWALLTORCH, MobjType.MT_ZWALLTORCH_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZWALLTORCH_U)
    elif mtype in (MobjType.MT_THRUSTFLOOR_UP, MobjType.MT_THRUSTFLOOR_DOWN):
        if mobj.args[0] == 1:
            start_sound(sim, mobj, int(SfxEnum.SFX_THRUSTSPIKE_RAISE))
            if mobj.args[1]:
                set_mobj_state(sim, mobj, StateNum.S_BTHRUSTLOWER)
            else:
                set_mobj_state(sim, mobj, StateNum.S_THRUSTLOWER)
    elif mtype in (MobjType.MT_ZFIREBULL, MobjType.MT_ZFIREBULL_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZFIREBULL_DEATH)
    elif mtype in (MobjType.MT_ZCAULDRON, MobjType.MT_ZCAULDRON_UNLIT):
        set_mobj_state(sim, mobj, StateNum.S_ZCAULDRON_U)
    elif mtype == MobjType.MT_FLAME_SMALL:
        set_mobj_state(sim, mobj, StateNum.S_FLAME_SDORM1)
    elif mtype == MobjType.MT_FLAME_LARGE:
        set_mobj_state(sim, mobj, StateNum.S_FLAME_LDORM1)
    elif mtype == MobjType.MT_BAT_SPAWNER:
        set_mobj_state(sim, mobj, StateNum.S_SPAWNBATS_OFF)
    else:
        return False
    return True


def ev_thing_activate(sim: GameSim, tid: int) -> bool:
    """Port of EV_ThingActivate (p_things.c)."""
    success = False
    for mobj in _mobjs_from_tid(sim, tid):
        if _activate_thing(sim, mobj):
            success = True
    return success


def ev_thing_deactivate(sim: GameSim, tid: int) -> bool:
    """Port of EV_ThingDeactivate (p_things.c)."""
    success = False
    for mobj in _mobjs_from_tid(sim, tid):
        if _deactivate_thing(sim, mobj):
            success = True
    return success


def ev_thing_remove(sim: GameSim, tid: int) -> bool:
    """Port of EV_ThingRemove (p_things.c). A bridge dissolves instead, and ends the search."""
    from pyhexen.sim.actions.environment import a_bridgeremove

    success = False
    for mobj in _mobjs_from_tid(sim, tid):
        if mobj.type == MobjType.MT_BRIDGE:
            a_bridgeremove(mobj, sim)
            return True
        remove_mobj(sim, mobj)
        success = True
    return success


def ev_thing_destroy(sim: GameSim, tid: int) -> bool:
    """Port of EV_ThingDestroy (p_things.c)."""
    from pyhexen.sim.inter import damage_mobj

    success = False
    for mobj in _mobjs_from_tid(sim, tid):
        if mobj.flags & MobjFlag.MF_SHOOTABLE:
            damage_mobj(sim, mobj, None, None, 10000)
            success = True
    return success


def a_local_quake(sim: GameSim, args: Sequence[int]) -> bool:
    """Port of A_LocalQuake (a_action.c): spawn a quake focus on every thing with tid `args[4]`.

    `args`: intensity, duration, damage radius, tremor radius, tid.
    """
    success = False
    for target in _mobjs_from_tid(sim, args[4]):
        focus = spawn_mobj(sim, target.x, target.y, target.z, MobjType.MT_QUAKE_FOCUS)
        # args[1] is halved: A_Quake decrements it every 2 tics.
        focus.args = (args[0], args[1] >> 1, args[2], args[3], args[4])
        success = True
    return success


def ev_line_search_for_puzzle_item(sim: GameSim, line: Line | None, mo: Mobj | None) -> bool:
    """Port of EV_LineSearchForPuzzleItem (p_spec.c).

    Uses the first puzzle item in the activator's inventory that matches the line's
    first argument. Vanilla needs a line; without one this reports failure.
    """
    from pyhexen.sim.artifacts import ArtiType, player_remove_artifact, use_artifact
    from pyhexen.sim.message import console_player

    if mo is None or line is None:
        return False
    player = mo.player
    if player is None:
        return False

    # Search player's inventory for puzzle items
    for i, slot in enumerate(player.inventory):
        arti = int(slot.type)
        type_ = arti - int(ArtiType.ARTI_FIRSTPUZZITEM)
        if type_ < 0:
            continue
        if type_ == line.args[0]:
            # A puzzle item was found for the line
            if use_artifact(sim, player, arti):
                player_remove_artifact(player, i)
                if player is console_player(sim):
                    start_sound(sim, None, int(SfxEnum.SFX_PUZZLE_SUCCESS))
                return True
    return False


def special_damage_thing(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Mobj | None
) -> bool:
    """DamageThing (73): hurt the activator; 0 damage guarantees a kill."""
    from pyhexen.sim.inter import damage_mobj

    if mo is not None:  # vanilla has no activator-less caller
        damage_mobj(sim, mo, None, None, args[0] if args[0] else 10000)
    return True


def special_radius_quake(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Radius_Quake (120)."""
    return a_local_quake(sim, args)


def special_use_puzzle_item(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """UsePuzzleItem (129)."""
    return ev_line_search_for_puzzle_item(sim, line, mo)


def special_thing_activate(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_Activate (130)."""
    return ev_thing_activate(sim, args[0])


def special_thing_deactivate(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_Deactivate (131)."""
    return ev_thing_deactivate(sim, args[0])


def special_thing_remove(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_Remove (132)."""
    return ev_thing_remove(sim, args[0])


def special_thing_destroy(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_Destroy (133)."""
    return ev_thing_destroy(sim, args[0])


def special_thing_projectile(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_Projectile (134)."""
    return ev_thing_projectile(sim, args, False)


def special_thing_spawn(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_Spawn (135)."""
    return ev_thing_spawn(sim, args, True)


def special_thing_projectile_gravity(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_ProjectileGravity (136)."""
    return ev_thing_projectile(sim, args, True)


def special_thing_spawn_no_fog(
    sim: GameSim, special: int, args: tuple[int, ...], line: Any, mo: Any
) -> bool:
    """Thing_SpawnNoFog (137)."""
    return ev_thing_spawn(sim, args, False)


SPECIAL_HANDLERS: dict[int, Any] = {
    73: special_damage_thing,
    120: special_radius_quake,
    129: special_use_puzzle_item,
    130: special_thing_activate,
    131: special_thing_deactivate,
    132: special_thing_remove,
    133: special_thing_destroy,
    134: special_thing_projectile,
    135: special_thing_spawn,
    136: special_thing_projectile_gravity,
    137: special_thing_spawn_no_fog,
}
