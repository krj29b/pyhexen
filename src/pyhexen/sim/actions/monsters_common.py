# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Monster action routines for common actors (Task T10.5a).

Covers:
- Ettin (A_EttinAttack, A_DropMace)
- Afrit / FireDemon (A_FiredSpawnRock, A_FiredRocks, A_FiredAttack, A_SmBounce,
  A_FiredChase, A_FiredSplotch)
- Centaur & Slaughtaur (A_CentaurAttack, A_CentaurAttack2, A_CentaurDropStuff,
  A_CentaurDefend)
- Chaos Serpent / Demon & Demon2 (A_DemonAttack1, A_DemonAttack2, A_DemonDeath,
  A_Demon2Death)
- Wendigo / IceGuy (A_IceGuyLook, A_IceGuyChase, A_IceGuyAttack,
  A_IceGuyMissilePuff, A_IceGuyDie, A_IceGuyMissileExplode)
- Stalker / Serpent (A_SerpentUnHide, A_SerpentHide, A_SerpentChase,
  A_SerpentRaiseHump, A_SerpentLowerHump, A_SerpentHumpDecide,
  A_SerpentBirthScream, A_SerpentDiveSound, A_SerpentWalk,
  A_SerpentCheckForAttack, A_SerpentChooseAttack, A_SerpentMeleeAttack,
  A_SerpentMissileAttack, A_SerpentHeadPop, A_SerpentSpawnGibs, A_FloatGib,
  A_SinkGib, A_DelayGib, A_SerpentHeadCheck)
- Shared state/sound helpers from a_action.c (A_HideThing, A_UnHideThing,
  A_SetShootable, A_UnSetShootable, A_SetAltShadow, A_ContMobjSound) and
  floor terrain splash helpers (get_thing_floor_type, hit_floor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import (
    ANG45,
    ANG90,
    ANGLETOFINESHIFT,
    approx_distance,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACBITS, FRACUNIT, fixed_mul, to_int32, to_uint32
from pyhexen.sim.actions import action
from pyhexen.sim.attacks import spawn_missile, spawn_missile_angle, spawn_missile_xyz
from pyhexen.sim.enemy import (
    _is_sim,
    a_chase,
    a_facetarget,
    a_freezedeathchunks,
    a_look,
    a_unsetinvulnerable,
    check_melee_range,
    check_melee_range2,
    check_missile_range,
    look_for_players,
    move,
    new_chase_dir,
    noise_alert,
    start_sound,
)
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType, SfxEnum, StateNum
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.map import try_move
from pyhexen.sim.mobj import FLOAT_BOB_OFFSETS, ONFLOORZ, Mobj, set_mobj_state, spawn_mobj
from pyhexen.sim.sight import check_sight

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

# Floor terrain constants from p_local.h
FLOOR_SOLID: int = 0
FLOOR_ICE: int = 1
FLOOR_LIQUID: int = 2
FLOOR_WATER: int = 3
FLOOR_LAVA: int = 4
FLOOR_SLUDGE: int = 5

SMALLSPLASHCLIP: int = 12 << FRACBITS
FIREDEMON_ATTACK_RANGE: int = 64 * 8 * FRACUNIT

# Compile-time C constant (fixed_t)(-0.3 * FRACUNIT) in A_IceGuyMissileExplode
ICEGUY_FX2_MOMZ: int = -19660

TERRAIN_TYPE_MAP: dict[str, int] = {
    "X_005": FLOOR_WATER,
    "X_001": FLOOR_LAVA,
    "X_009": FLOOR_SLUDGE,
    "F_033": FLOOR_ICE,
}


def hitdice(sim: GameSim | None, a: int) -> int:
    """Compute Hexen HITDICE(a): ((1 + (P_Random() & 7)) * a).

    Port of HITDICE macro (m_random.h).
    """
    rnd = sim.rng.p_random() if sim is not None else 0
    return (1 + (rnd & 7)) * a


def face_movement_direction(actor: Mobj) -> None:
    """Snap actor.angle to match its 8-way movedir.

    Port of FaceMovementDirection (p_enemy.c).
    """
    if 0 <= actor.movedir < 8:
        actor.angle = (actor.movedir << 29) & 0xFFFFFFFF


def get_thing_floor_type(thing: Mobj) -> int:
    """Return the terrain type (FLOOR_*) beneath `thing`.

    Port of P_GetThingFloorType (p_mobj.c).
    """
    if thing.floorpic:
        if isinstance(thing.floorpic, str):
            return TERRAIN_TYPE_MAP.get(thing.floorpic.upper(), FLOOR_SOLID)
        if isinstance(thing.floorpic, int) and thing.floorpic > 0:
            return thing.floorpic
    if thing.subsector is not None and thing.subsector.sector is not None:
        sec = thing.subsector.sector
        fpic = getattr(sec, "floor_pic", None) or getattr(sec, "floorpic", "")
        if isinstance(fpic, str):
            return TERRAIN_TYPE_MAP.get(fpic.upper(), FLOOR_SOLID)
        if isinstance(fpic, int):
            return fpic
    return FLOOR_SOLID


def hit_floor(sim: GameSim | None, thing: Mobj) -> int:
    """Handle floor impact splashes and hazards when `thing` hits a liquid floor.

    Port of P_HitFloor (p_mobj.c).
    """
    if thing.subsector is not None and thing.subsector.sector is not None:
        sec = thing.subsector.sector
        sec_fh = getattr(sec, "floor_height", getattr(sec, "floorheight", thing.floorz))
        if thing.floorz != sec_fh:
            return FLOOR_SOLID

    if thing.type in (
        MobjType.MT_LEAF1,
        MobjType.MT_LEAF2,
        MobjType.MT_SPLASH,
        MobjType.MT_SLUDGECHUNK,
    ):
        return FLOOR_SOLID

    smallsplash = thing.info.mass < 10
    floor_type = get_thing_floor_type(thing)

    if floor_type == FLOOR_WATER:
        if sim is not None:
            if smallsplash:
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_SPLASHBASE)
                if mo is not None:
                    mo.floorclip += SMALLSPLASHCLIP
                start_sound(sim, mo, SfxEnum.SFX_AMBIENT10)
            else:
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_SPLASH)
                mo.target = thing
                mo.momx = sim.rng.p_subrandom() << 8
                mo.momy = sim.rng.p_subrandom() << 8
                mo.momz = 2 * FRACUNIT + (sim.rng.p_random() << 8)
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_SPLASHBASE)
                if thing.player is not None:
                    noise_alert(sim, thing, thing)
                start_sound(sim, mo, SfxEnum.SFX_WATER_SPLASH)
        return FLOOR_WATER

    if floor_type == FLOOR_LAVA:
        if sim is not None:
            if smallsplash:
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_LAVASPLASH)
                if mo is not None:
                    mo.floorclip += SMALLSPLASHCLIP
            else:
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_LAVASMOKE)
                mo.momz = FRACUNIT + (sim.rng.p_random() << 7)
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_LAVASPLASH)
                if thing.player is not None:
                    noise_alert(sim, thing, thing)
            start_sound(sim, mo, SfxEnum.SFX_LAVA_SIZZLE)
            if thing.player is not None and (getattr(sim, "level_time", 0) & 31):
                lava_inflictor = Mobj(
                    type=MobjType.MT_CIRCLEFLAME,
                    flags2=int(MobjFlag2.MF2_FIREDAMAGE | MobjFlag2.MF2_NODMGTHRUST),
                )
                damage_mobj(sim, thing, lava_inflictor, None, 5)
        return FLOOR_LAVA

    if floor_type == FLOOR_SLUDGE:
        if sim is not None:
            if smallsplash:
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_SLUDGESPLASH)
                if mo is not None:
                    mo.floorclip += SMALLSPLASHCLIP
            else:
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_SLUDGECHUNK)
                mo.target = thing
                mo.momx = sim.rng.p_subrandom() << 8
                mo.momy = sim.rng.p_subrandom() << 8
                mo.momz = FRACUNIT + (sim.rng.p_random() << 8)
                mo = spawn_mobj(sim, thing.x, thing.y, ONFLOORZ, MobjType.MT_SLUDGESPLASH)
                if thing.player is not None:
                    noise_alert(sim, thing, thing)
            start_sound(sim, mo, SfxEnum.SFX_SLUDGE_GLOOP)
        return FLOOR_SLUDGE

    return FLOOR_SOLID


# =============================================================================
# Shared State & Sound Actions (a_action.c)
# =============================================================================


@action("A_HideThing")
def a_hidething(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Set MF2_DONTDRAW on actor.

    Port of A_HideThing (a_action.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags2 |= MobjFlag2.MF2_DONTDRAW


@action("A_UnHideThing")
def a_unhidething(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Clear MF2_DONTDRAW on actor.

    Port of A_UnHideThing (a_action.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags2 &= ~MobjFlag2.MF2_DONTDRAW


@action("A_SetShootable")
def a_setshootable(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Clear MF2_NONSHOOTABLE and set MF_SHOOTABLE on actor.

    Port of A_SetShootable (a_action.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags2 &= ~MobjFlag2.MF2_NONSHOOTABLE
        actor.flags |= MobjFlag.MF_SHOOTABLE


@action("A_UnSetShootable")
def a_unsetshootable(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Set MF2_NONSHOOTABLE and clear MF_SHOOTABLE on actor.

    Port of A_UnSetShootable (a_action.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags2 |= MobjFlag2.MF2_NONSHOOTABLE
        actor.flags &= ~MobjFlag.MF_SHOOTABLE


@action("A_SetAltShadow")
def a_setaltshadow(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Clear MF_SHADOW and set MF_ALTSHADOW on actor.

    Port of A_SetAltShadow (a_action.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.flags &= ~MobjFlag.MF_SHADOW
        actor.flags |= MobjFlag.MF_ALTSHADOW


@action("A_ContMobjSound")
def a_contmobjsound(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Play continuous projectile/effect sound based on actor type.

    Port of A_ContMobjSound (a_action.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if actor.type == MobjType.MT_SERPENTFX:
        start_sound(sim, actor, SfxEnum.SFX_SERPENTFX_CONTINUOUS)
    elif actor.type == MobjType.MT_HAMMER_MISSILE:
        start_sound(sim, actor, SfxEnum.SFX_FIGHTER_HAMMER_CONTINUOUS)
    elif actor.type == MobjType.MT_QUAKE_FOCUS:
        start_sound(sim, actor, SfxEnum.SFX_EARTHQUAKE)


# =============================================================================
# Ettin Actions (p_enemy.c)
# =============================================================================


@action("A_EttinAttack")
def a_ettinattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Ettin melee mace swing dealing HITDICE(2) damage.

    Port of A_EttinAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if check_melee_range(actor, sim) and actor.target is not None:
        dmg = hitdice(sim, 2)
        if sim is not None:
            damage_mobj(sim, actor.target, actor, actor, dmg)


@action("A_DropMace")
def a_dropmace(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn dropped Ettin mace on extreme death.

    Port of A_DropMace (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return
    mo = spawn_mobj(
        sim,
        actor.x,
        actor.y,
        actor.z + (actor.height >> 1),
        MobjType.MT_ETTIN_MACE,
    )
    if mo is not None:
        r1 = sim.rng.p_random()
        r2 = sim.rng.p_random()
        r3 = sim.rng.p_random()
        mo.momx = (r1 - 128) << 11
        mo.momy = (r2 - 128) << 11
        mo.momz = FRACUNIT * 10 + (r3 << 10)
        mo.target = actor


# =============================================================================
# Afrit / FireDemon Actions (p_enemy.c)
# =============================================================================


@action("A_FiredSpawnRock")
def a_firedspawnrock(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn a single falling fire demon rock chunk and clear actor strafe/attack state.

    Port of A_FiredSpawnRock (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    rock_types = (
        MobjType.MT_FIREDEMON_FX1,
        MobjType.MT_FIREDEMON_FX2,
        MobjType.MT_FIREDEMON_FX3,
        MobjType.MT_FIREDEMON_FX4,
        MobjType.MT_FIREDEMON_FX5,
    )
    rtype = rock_types[sim.rng.p_random() % 5]

    x = actor.x + ((sim.rng.p_random() - 128) << 12)
    y = actor.y + ((sim.rng.p_random() - 128) << 12)
    z = actor.z + (sim.rng.p_random() << 11)
    mo = spawn_mobj(sim, x, y, z, rtype)
    if mo is not None:
        mo.target = actor
        mo.momx = (sim.rng.p_random() - 128) << 10
        mo.momy = (sim.rng.p_random() - 128) << 10
        mo.momz = sim.rng.p_random() << 10
        mo.special1 = 2  # Number bounces

    actor.special2 = 0
    actor.flags &= ~MobjFlag.MF_JUSTATTACKED


@action("A_FiredRocks")
def a_firedrocks(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn 5 fire demon rocks during wakeup/unencase sequence.

    Port of A_FiredRocks (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    for _ in range(5):
        a_firedspawnrock(actor, sim)


@action("A_FiredAttack")
def a_firedattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Fire Afrit projectile (MT_FIREDEMON_FX6) at target.

    Port of A_FiredAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None or sim is None:
        return
    mo = spawn_missile(sim, actor, actor.target, MobjType.MT_FIREDEMON_FX6)
    if mo is not None:
        start_sound(sim, actor, SfxEnum.SFX_FIRED_ATTACK)


@action("A_SmBounce")
def a_smbounce(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Apply small upward and horizontal bounce momentum to rock debris.

    Port of A_SmBounce (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    actor.z = actor.floorz + FRACUNIT
    r1 = sim.rng.p_random() if sim is not None else 0
    r2 = sim.rng.p_random() if sim is not None else 0
    r3 = sim.rng.p_random() if sim is not None else 0
    actor.momz = (2 * FRACUNIT) + (r1 << 10)
    actor.momx = (r2 % 3) << FRACBITS
    actor.momy = (r3 % 3) << FRACBITS


@action("A_FiredChase")
def a_firedchase(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Afrit chase AI with vertical float bobbing, minimum floor hover, and lateral strafing.

    Port of A_FiredChase (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return

    weaveindex = actor.special1 if isinstance(actor.special1, int) else 0
    target = actor.target

    if actor.reactiontime > 0:
        actor.reactiontime -= 1
    if actor.threshold > 0:
        actor.threshold -= 1

    # Float up and down
    actor.z += FLOAT_BOB_OFFSETS[weaveindex & 63]
    actor.special1 = (weaveindex + 2) & 63

    # Ensure it stays above minimum height (floorz + 64)
    if actor.z < actor.floorz + (64 * FRACUNIT):
        actor.z += 2 * FRACUNIT

    if target is None or not (target.flags & MobjFlag.MF_SHOOTABLE) or target.health <= 0:
        look_for_players(actor, True, sim)
        return

    # Strafe
    strafe_tics = actor.special2 if isinstance(actor.special2, int) else 0
    if strafe_tics > 0:
        actor.special2 = strafe_tics - 1
    else:
        actor.special2 = 0
        actor.momx = 0
        actor.momy = 0
        dist = approx_distance(actor.x - target.x, actor.y - target.y)
        if dist < FIREDEMON_ATTACK_RANGE:
            rnd = sim.rng.p_random() if sim is not None else 255
            if rnd < 30:
                ang = point_to_angle2(actor.x, actor.y, target.x, target.y)
                dir_rnd = sim.rng.p_random() if sim is not None else 0
                if dir_rnd < 128:
                    ang = to_uint32(ang + ANG90)
                else:
                    ang = to_uint32(ang - ANG90)
                fine = ang >> ANGLETOFINESHIFT
                actor.momx = fixed_mul(8 * FRACUNIT, fine_cosine(fine))
                actor.momy = fixed_mul(8 * FRACUNIT, fine_sine(fine))
                actor.special2 = 3  # strafe time

    face_movement_direction(actor)

    # Normal movement when not strafing
    if not (isinstance(actor.special2, int) and actor.special2 > 0):
        actor.movecount -= 1
        if actor.movecount < 0 or not move(actor, sim):
            new_chase_dir(actor, sim)

    # Do missile attack
    if not (actor.flags & MobjFlag.MF_JUSTATTACKED):
        if check_missile_range(actor, sim):
            atk_rnd = sim.rng.p_random() if sim is not None else 255
            if atk_rnd < 20:
                if sim is not None:
                    set_mobj_state(sim, actor, actor.info.missilestate)
                else:
                    actor.state_num = actor.info.missilestate
                actor.flags |= MobjFlag.MF_JUSTATTACKED
                return
    else:
        actor.flags &= ~MobjFlag.MF_JUSTATTACKED

    # Active sound
    if actor.info.activesound and sim is not None and sim.rng.p_random() < 3:
        start_sound(sim, actor, actor.info.activesound)


@action("A_FiredSplotch")
def a_firedsplotch(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn two burning splotch gibs when an Afrit dies.

    Port of A_FiredSplotch (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_FIREDEMON_SPLOTCH1)
    if mo is not None:
        r1 = sim.rng.p_random()
        r2 = sim.rng.p_random()
        r3 = sim.rng.p_random()
        mo.momx = (r1 - 128) << 11
        mo.momy = (r2 - 128) << 11
        mo.momz = FRACUNIT * 3 + (r3 << 10)

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_FIREDEMON_SPLOTCH2)
    if mo is not None:
        r1 = sim.rng.p_random()
        r2 = sim.rng.p_random()
        r3 = sim.rng.p_random()
        mo.momx = (r1 - 128) << 11
        mo.momy = (r2 - 128) << 11
        mo.momz = FRACUNIT * 3 + (r3 << 10)


# =============================================================================
# Centaur & Slaughtaur Actions (p_enemy.c)
# =============================================================================


@action("A_CentaurAttack")
def a_centaurattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Centaur melee shield/sword bash dealing (P_Random() % 7 + 3) damage.

    Port of A_CentaurAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None:
        return
    if check_melee_range(actor, sim):
        rnd = sim.rng.p_random() if sim is not None else 0
        dmg = (rnd % 7) + 3
        if sim is not None:
            damage_mobj(sim, actor.target, actor, actor, dmg)


@action("A_CentaurAttack2")
def a_centaurattack2(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Slaughtaur shield projectile attack (MT_CENTAUR_FX).

    Port of A_CentaurAttack2 (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None:
        return
    if sim is not None:
        spawn_missile(sim, actor, actor.target, MobjType.MT_CENTAUR_FX)
    start_sound(sim, actor, SfxEnum.SFX_CENTAURLEADER_ATTACK)


@action("A_CentaurDropStuff")
def a_centaurdropstuff(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Drop Centaur shield and sword debris on extreme death.

    Port of A_CentaurDropStuff (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(
        sim,
        actor.x,
        actor.y,
        actor.z + 45 * FRACUNIT,
        MobjType.MT_CENTAUR_SHIELD,
    )
    if mo is not None:
        angle = to_uint32(actor.angle + ANG90)
        fine = angle >> ANGLETOFINESHIFT
        r_z = sim.rng.p_random()
        r_x = sim.rng.p_random()
        r_y = sim.rng.p_random()
        mo.momz = FRACUNIT * 8 + (r_z << 10)
        mo.momx = fixed_mul(((r_x - 128) << 11) + FRACUNIT, fine_cosine(fine))
        mo.momy = fixed_mul(((r_y - 128) << 11) + FRACUNIT, fine_sine(fine))
        mo.target = actor

    mo = spawn_mobj(
        sim,
        actor.x,
        actor.y,
        actor.z + 45 * FRACUNIT,
        MobjType.MT_CENTAUR_SWORD,
    )
    if mo is not None:
        angle = to_uint32(actor.angle - ANG90)
        fine = angle >> ANGLETOFINESHIFT
        r_z = sim.rng.p_random()
        r_x = sim.rng.p_random()
        r_y = sim.rng.p_random()
        mo.momz = FRACUNIT * 8 + (r_z << 10)
        mo.momx = fixed_mul(((r_x - 128) << 11) + FRACUNIT, fine_cosine(fine))
        mo.momy = fixed_mul(((r_y - 128) << 11) + FRACUNIT, fine_sine(fine))
        mo.target = actor


@action("A_CentaurDefend")
def a_centaurdefend(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Centaur defensive shield stance: face target and counter-attack in melee range.

    Port of A_CentaurDefend (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    a_facetarget(actor, sim)
    if check_melee_range(actor, sim):
        rnd = sim.rng.p_random() if sim is not None else 255
        if rnd < 32:
            a_unsetinvulnerable(actor, sim)
            if sim is not None:
                set_mobj_state(sim, actor, actor.info.meleestate)
            else:
                actor.state_num = actor.info.meleestate


# =============================================================================
# Chaos Serpent / Demon & Demon2 Actions (p_enemy.c)
# =============================================================================


@action("A_DemonAttack1")
def a_demonattack1(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Chaos Serpent melee bite dealing HITDICE(2) damage.

    Port of A_DemonAttack1 (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if check_melee_range(actor, sim) and actor.target is not None:
        dmg = hitdice(sim, 2)
        if sim is not None:
            damage_mobj(sim, actor.target, actor, actor, dmg)


@action("A_DemonAttack2")
def a_demonattack2(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Chaos Serpent fireball/poison breath missile attack (MT_DEMONFX1 / MT_DEMON2FX1).

    Port of A_DemonAttack2 (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None or sim is None:
        return
    if actor.type == MobjType.MT_DEMON:
        fireball = MobjType.MT_DEMONFX1
    else:
        fireball = MobjType.MT_DEMON2FX1
    mo = spawn_missile(sim, actor, actor.target, fireball)
    if mo is not None:
        mo.z += 30 * FRACUNIT
        start_sound(sim, actor, SfxEnum.SFX_DEMON_MISSILE_FIRE)


def _spawn_demon_chunks(
    sim: GameSim,
    actor: Mobj,
    chunk_types: tuple[MobjType, MobjType, MobjType, MobjType, MobjType],
) -> None:
    """Spawn 5 Demon/Demon2 body chunks on extreme death."""
    for idx, ctype in enumerate(chunk_types):
        mo = spawn_mobj(sim, actor.x, actor.y, actor.z + 45 * FRACUNIT, ctype)
        if mo is not None:
            if idx == 0:
                angle = to_uint32(actor.angle + ANG90)
            else:
                angle = to_uint32(actor.angle - ANG90)
            fine = angle >> ANGLETOFINESHIFT
            r_x = sim.rng.p_random()
            r_y = sim.rng.p_random()
            mo.momz = 8 * FRACUNIT
            mo.momx = fixed_mul((r_x << 10) + FRACUNIT, fine_cosine(fine))
            mo.momy = fixed_mul((r_y << 10) + FRACUNIT, fine_sine(fine))
            mo.target = actor


@action("A_DemonDeath")
def a_demondeath(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn green Chaos Serpent chunks (MT_DEMONCHUNK1..5) on extreme death.

    Port of A_DemonDeath (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return
    _spawn_demon_chunks(
        sim,
        actor,
        (
            MobjType.MT_DEMONCHUNK1,
            MobjType.MT_DEMONCHUNK2,
            MobjType.MT_DEMONCHUNK3,
            MobjType.MT_DEMONCHUNK4,
            MobjType.MT_DEMONCHUNK5,
        ),
    )


@action("A_Demon2Death")
def a_demon2death(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn brown Chaos Serpent chunks (MT_DEMON2CHUNK1..5) on extreme death.

    Port of A_Demon2Death (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return
    _spawn_demon_chunks(
        sim,
        actor,
        (
            MobjType.MT_DEMON2CHUNK1,
            MobjType.MT_DEMON2CHUNK2,
            MobjType.MT_DEMON2CHUNK3,
            MobjType.MT_DEMON2CHUNK4,
            MobjType.MT_DEMON2CHUNK5,
        ),
    )


# =============================================================================
# Wendigo / IceGuy Actions (p_enemy.c)
# =============================================================================


@action("A_IceGuyLook")
def a_iceguylook(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Wendigo idle look routine with ambient frost wisp spawning.

    Port of A_IceGuyLook (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    a_look(actor, sim)
    if sim is not None and sim.rng.p_random() < 64:
        dist = ((sim.rng.p_random() - 128) * actor.radius) >> 7
        an = to_uint32(actor.angle + ANG90) >> ANGLETOFINESHIFT
        wisp_type = MobjType(int(MobjType.MT_ICEGUY_WISP1) + (sim.rng.p_random() & 1))
        spawn_mobj(
            sim,
            actor.x + fixed_mul(dist, fine_cosine(an)),
            actor.y + fixed_mul(dist, fine_sine(an)),
            actor.z + 60 * FRACUNIT,
            wisp_type,
        )


@action("A_IceGuyChase")
def a_iceguychase(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Wendigo chase routine with trailing frost wisps matching actor momentum.

    Port of A_IceGuyChase (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    a_chase(actor, sim)
    if sim is not None and sim.rng.p_random() < 128:
        dist = ((sim.rng.p_random() - 128) * actor.radius) >> 7
        an = to_uint32(actor.angle + ANG90) >> ANGLETOFINESHIFT
        wisp_type = MobjType(int(MobjType.MT_ICEGUY_WISP1) + (sim.rng.p_random() & 1))
        mo = spawn_mobj(
            sim,
            actor.x + fixed_mul(dist, fine_cosine(an)),
            actor.y + fixed_mul(dist, fine_sine(an)),
            actor.z + 60 * FRACUNIT,
            wisp_type,
        )
        if mo is not None:
            mo.momx = actor.momx
            mo.momy = actor.momy
            mo.momz = actor.momz
            mo.target = actor


@action("A_IceGuyAttack")
def a_iceguyattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Wendigo dual ice projectile volley from left and right hands.

    Port of A_IceGuyAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None:
        return
    if sim is not None:
        an1 = to_uint32(actor.angle + ANG90) >> ANGLETOFINESHIFT
        spawn_missile_xyz(
            sim,
            actor.x + fixed_mul(actor.radius >> 1, fine_cosine(an1)),
            actor.y + fixed_mul(actor.radius >> 1, fine_sine(an1)),
            actor.z + 40 * FRACUNIT,
            actor,
            actor.target,
            MobjType.MT_ICEGUY_FX,
        )
        an2 = to_uint32(actor.angle - ANG90) >> ANGLETOFINESHIFT
        spawn_missile_xyz(
            sim,
            actor.x + fixed_mul(actor.radius >> 1, fine_cosine(an2)),
            actor.y + fixed_mul(actor.radius >> 1, fine_sine(an2)),
            actor.z + 40 * FRACUNIT,
            actor,
            actor.target,
            MobjType.MT_ICEGUY_FX,
        )
    start_sound(sim, actor, actor.info.attacksound)


@action("A_IceGuyMissilePuff")
def a_iceguymissilepuff(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn trailing puff for Wendigo ice missile (MT_ICEFX_PUFF).

    Port of A_IceGuyMissilePuff (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return
    spawn_mobj(sim, actor.x, actor.y, actor.z + 2 * FRACUNIT, MobjType.MT_ICEFX_PUFF)


@action("A_IceGuyDie")
def a_iceguydie(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Shatter Wendigo immediately into ice chunks on death.

    Port of A_IceGuyDie (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    actor.momx = 0
    actor.momy = 0
    actor.momz = 0
    actor.height <<= 2
    a_freezedeathchunks(actor, sim)


@action("A_IceGuyMissileExplode")
def a_iceguymissileexplode(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Explode Wendigo ice projectile into 8 radial shard missiles (MT_ICEGUY_FX2).

    Port of A_IceGuyMissileExplode (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return
    for i in range(8):
        mo = spawn_missile_angle(
            sim,
            actor,
            MobjType.MT_ICEGUY_FX2,
            to_uint32(i * ANG45),
            ICEGUY_FX2_MOMZ,
        )
        if mo is not None:
            mo.target = actor.target


# =============================================================================
# Stalker / Serpent Actions (p_enemy.c)
# =============================================================================


@action("A_SerpentUnHide")
def a_serpentunhide(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Make Stalker visible with 24-unit submerged floorclip.

    Port of A_SerpentUnHide (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    actor.flags2 &= ~MobjFlag2.MF2_DONTDRAW
    actor.floorclip = 24 * FRACUNIT


@action("A_SerpentHide")
def a_serpenthide(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Hide submerged Stalker and reset floorclip to 0.

    Port of A_SerpentHide (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    actor.flags2 |= MobjFlag2.MF2_DONTDRAW
    actor.floorclip = 0


def _get_sector_floor_id(actor: Mobj) -> Any:
    """Return sector floor flat identifier for submerged Stalker boundary checks."""
    if actor.subsector is not None and actor.subsector.sector is not None:
        sec = actor.subsector.sector
        return getattr(sec, "floor_pic", getattr(sec, "floorpic", None))
    return None


@action("A_SerpentChase")
def a_serpentchase(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Submerged Stalker chase AI staying confined to its current liquid flat type.

    Port of A_SerpentChase (p_enemy.c).
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

    if actor.info.meleestate and check_melee_range(actor, sim):
        if actor.info.attacksound:
            start_sound(sim, actor, actor.info.attacksound)
        if sim is not None:
            set_mobj_state(sim, actor, actor.info.meleestate)
        else:
            actor.state_num = actor.info.meleestate
        return

    is_netgame = getattr(sim, "netgame", False) if sim is not None else False
    if is_netgame and not actor.threshold and not check_sight(sim, actor, actor.target):
        if look_for_players(actor, True, sim):
            return

    old_x = actor.x
    old_y = actor.y
    old_floor = _get_sector_floor_id(actor)
    actor.movecount -= 1
    if actor.movecount < 0 or not move(actor, sim):
        new_chase_dir(actor, sim)
    new_floor = _get_sector_floor_id(actor)
    if new_floor != old_floor:
        if sim is not None:
            try_move(sim, actor, old_x, old_y)
        else:
            actor.x = old_x
            actor.y = old_y
        new_chase_dir(actor, sim)

    if actor.info.activesound and sim is not None and sim.rng.p_random() < 3:
        start_sound(sim, actor, actor.info.activesound)


@action("A_SerpentRaiseHump")
def a_serpentraisehump(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Raise Stalker hump above liquid surface by reducing floorclip by 4 units.

    Port of A_SerpentRaiseHump (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.floorclip -= 4 * FRACUNIT


@action("A_SerpentLowerHump")
def a_serpentlowerhump(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Lower Stalker hump back into liquid surface by increasing floorclip by 4 units.

    Port of A_SerpentLowerHump (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.floorclip += 4 * FRACUNIT


@action("A_SerpentHumpDecide")
def a_serpenthumpdecide(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Decide whether Stalker raises hump or (if leader) surfaces for missile attack.

    Port of A_SerpentHumpDecide (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if actor.type == MobjType.MT_SERPENTLEADER:
        if sim.rng.p_random() > 30:
            return
        if sim.rng.p_random() < 40:
            set_mobj_state(sim, actor, StateNum.S_SERPENT_SURFACE1)
            return
    elif sim.rng.p_random() > 3:
        return

    if not check_melee_range(actor, sim):
        if actor.type == MobjType.MT_SERPENTLEADER and sim.rng.p_random() < 128:
            set_mobj_state(sim, actor, StateNum.S_SERPENT_SURFACE1)
        else:
            set_mobj_state(sim, actor, StateNum.S_SERPENT_HUMP1)
            start_sound(sim, actor, SfxEnum.SFX_SERPENT_ACTIVE)


@action("A_SerpentBirthScream")
def a_serpentbirthscream(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Play Stalker surfacing birth sound (SFX_SERPENT_BIRTH).

    Port of A_SerpentBirthScream (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        start_sound(sim, actor, SfxEnum.SFX_SERPENT_BIRTH)


@action("A_SerpentDiveSound")
def a_serpentdivesound(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Play Stalker diving splash sound (SFX_SERPENT_ACTIVE).

    Port of A_SerpentDiveSound (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        start_sound(sim, actor, SfxEnum.SFX_SERPENT_ACTIVE)


@action("A_SerpentWalk")
def a_serpentwalk(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Surfaced Stalker walk/chase step entering S_SERPENT_ATK1 on melee range.

    Port of A_SerpentWalk (p_enemy.c).
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

    if actor.info.meleestate and check_melee_range(actor, sim):
        if actor.info.attacksound:
            start_sound(sim, actor, actor.info.attacksound)
        if sim is not None:
            set_mobj_state(sim, actor, StateNum.S_SERPENT_ATK1)
        else:
            actor.state_num = StateNum.S_SERPENT_ATK1
        return

    is_netgame = getattr(sim, "netgame", False) if sim is not None else False
    if is_netgame and not actor.threshold and not check_sight(sim, actor, actor.target):
        if look_for_players(actor, True, sim):
            return

    actor.movecount -= 1
    if actor.movecount < 0 or not move(actor, sim):
        new_chase_dir(actor, sim)


@action("A_SerpentCheckForAttack")
def a_serpentcheckforattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Evaluate surfaced Stalker attack/walk state transition.

    Port of A_SerpentCheckForAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None:
        return

    if actor.type == MobjType.MT_SERPENTLEADER:
        if not check_melee_range(actor, sim):
            if sim is not None:
                set_mobj_state(sim, actor, StateNum.S_SERPENT_ATK1)
            else:
                actor.state_num = StateNum.S_SERPENT_ATK1
            return

    if check_melee_range2(actor, sim):
        if sim is not None:
            set_mobj_state(sim, actor, StateNum.S_SERPENT_WALK1)
        else:
            actor.state_num = StateNum.S_SERPENT_WALK1
    elif check_melee_range(actor, sim):
        rnd = sim.rng.p_random() if sim is not None else 255
        target_state = StateNum.S_SERPENT_WALK1 if rnd < 32 else StateNum.S_SERPENT_ATK1
        if sim is not None:
            set_mobj_state(sim, actor, target_state)
        else:
            actor.state_num = target_state


@action("A_SerpentChooseAttack")
def a_serpentchooseattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Switch Stalker Leader to S_SERPENT_MISSILE1 if target is outside melee range.

    Port of A_SerpentChooseAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None or check_melee_range(actor, sim):
        return
    if actor.type == MobjType.MT_SERPENTLEADER:
        if sim is not None:
            set_mobj_state(sim, actor, StateNum.S_SERPENT_MISSILE1)
        else:
            actor.state_num = StateNum.S_SERPENT_MISSILE1


@action("A_SerpentMeleeAttack")
def a_serpentmeleeattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Stalker melee claw/bite dealing HITDICE(5) damage with follow-up check.

    Port of A_SerpentMeleeAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None:
        return
    if check_melee_range(actor, sim):
        dmg = hitdice(sim, 5)
        if sim is not None:
            damage_mobj(sim, actor.target, actor, actor, dmg)
        start_sound(sim, actor, SfxEnum.SFX_SERPENT_MELEEHIT)
    rnd = sim.rng.p_random() if sim is not None else 255
    if rnd < 96:
        a_serpentcheckforattack(actor, sim)


@action("A_SerpentMissileAttack")
def a_serpentmissileattack(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Stalker Leader acid projectile attack (MT_SERPENTFX).

    Port of A_SerpentMissileAttack (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or actor.target is None or sim is None:
        return
    spawn_missile(sim, actor, actor.target, MobjType.MT_SERPENTFX)


@action("A_SerpentHeadPop")
def a_serpentheadpop(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn severed Stalker head (MT_SERPENT_HEAD) on death.

    Port of A_SerpentHeadPop (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return
    spawn_mobj(sim, actor.x, actor.y, actor.z + 45 * FRACUNIT, MobjType.MT_SERPENT_HEAD)


@action("A_SerpentSpawnGibs")
def a_serpentspawngibs(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Spawn 3 floating Stalker gibs (MT_SERPENT_GIB1..3).

    Port of A_SerpentSpawnGibs (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    for gib_type in (
        MobjType.MT_SERPENT_GIB1,
        MobjType.MT_SERPENT_GIB2,
        MobjType.MT_SERPENT_GIB3,
    ):
        r1 = sim.rng.p_random()
        r2 = sim.rng.p_random()
        mo = spawn_mobj(
            sim,
            actor.x + ((r2 - 128) << 12),
            actor.y + ((r1 - 128) << 12),
            actor.floorz + FRACUNIT,
            gib_type,
        )
        if mo is not None:
            mo.momx = (sim.rng.p_random() - 128) << 6
            mo.momy = (sim.rng.p_random() - 128) << 6
            mo.floorclip = 6 * FRACUNIT


@action("A_FloatGib")
def a_floatgib(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Raise floating gib by decrementing floorclip by 1 unit.

    Port of A_FloatGib (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.floorclip -= FRACUNIT


@action("A_SinkGib")
def a_sinkgib(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Sink gib by incrementing floorclip by 1 unit.

    Port of A_SinkGib (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None:
        actor.floorclip += FRACUNIT


@action("A_DelayGib")
def a_delaygib(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Randomize gib sink delay by subtracting (P_Random() >> 2) from tics.

    Port of A_DelayGib (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is not None and sim is not None:
        actor.tics -= sim.rng.p_random() >> 2


@action("A_SerpentHeadCheck")
def a_serpentheadcheck(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Check if falling Stalker head hit liquid (splash + remove) or solid floor.

    Port of A_SerpentHeadCheck (p_enemy.c).
    """
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return
    if actor.z <= actor.floorz:
        if get_thing_floor_type(actor) >= FLOOR_LIQUID:
            hit_floor(sim, actor)
            if sim is not None:
                set_mobj_state(sim, actor, StateNum.S_NULL)
            else:
                actor.state_num = StateNum.S_NULL
        else:
            if sim is not None:
                set_mobj_state(sim, actor, StateNum.S_SERPENT_HEAD_X1)
            else:
                actor.state_num = StateNum.S_SERPENT_HEAD_X1


__all__ = [
    "FIREDEMON_ATTACK_RANGE",
    "FLOOR_ICE",
    "FLOOR_LAVA",
    "FLOOR_LIQUID",
    "FLOOR_SLUDGE",
    "FLOOR_SOLID",
    "FLOOR_WATER",
    "ICEGUY_FX2_MOMZ",
    "SMALLSPLASHCLIP",
    "TERRAIN_TYPE_MAP",
    "a_centaurattack",
    "a_centaurattack2",
    "a_centaurdefend",
    "a_centaurdropstuff",
    "a_contmobjsound",
    "a_delaygib",
    "a_demon2death",
    "a_demonattack1",
    "a_demonattack2",
    "a_demondeath",
    "a_dropmace",
    "a_ettinattack",
    "a_firedattack",
    "a_firedchase",
    "a_firedrocks",
    "a_firedspawnrock",
    "a_firedsplotch",
    "a_floatgib",
    "a_hidething",
    "a_iceguyattack",
    "a_iceguychase",
    "a_iceguydie",
    "a_iceguylook",
    "a_iceguymissileexplode",
    "a_iceguymissilepuff",
    "a_serpentbirthscream",
    "a_serpentchase",
    "a_serpentcheckforattack",
    "a_serpentchooseattack",
    "a_serpentdivesound",
    "a_serpentheadcheck",
    "a_serpentheadpop",
    "a_serpenthide",
    "a_serpenthumpdecide",
    "a_serpentlowerhump",
    "a_serpentmeleeattack",
    "a_serpentmissileattack",
    "a_serpentraisehump",
    "a_serpentspawngibs",
    "a_serpentunhide",
    "a_serpentwalk",
    "a_setaltshadow",
    "a_setshootable",
    "a_sinkgib",
    "a_smbounce",
    "a_unhidething",
    "a_unsetshootable",
    "face_movement_direction",
    "get_thing_floor_type",
    "hit_floor",
    "hitdice",
]
