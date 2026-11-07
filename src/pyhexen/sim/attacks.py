# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Combat attacks: hitscan raycasting, projectile spawning & physics, homing, and radius damage.

Port of P_AimLineAttack, PTR_AimTraverse, P_LineAttack, PTR_ShootTraverse,
P_SpawnPuff, P_BloodSplatter, P_BloodSplatter2, P_RipperBlood, P_SpawnMissile,
P_SpawnMissileXYZ, P_SpawnMissileAngle, P_SpawnMissileAngleSpeed,
P_CheckMissileSpawn, P_FaceMobj, P_SeekerMissile, and P_RadiusAttack
(p_map.c, p_mobj.c, p_enemy.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG180,
    ANG_MAX,
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
    fixed_div,
    fixed_mul,
)
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
)
from pyhexen.sim.maputl import (
    PT_ADDLINES,
    PT_ADDTHINGS,
    DivLine,
    Intercept,
    _traverse_intercepts,
    intercept_vector,
    line_opening,
    path_traverse,
    point_on_divline_side,
)
from pyhexen.sim.mobj import (
    ONFLOORZ,
    Mobj,
    spawn_mobj,
)
from pyhexen.world.blockmap import MAPBLOCKSHIFT
from pyhexen.world.level import (
    ML_TWOSIDED,
    SPAC_IMPACT,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

MAXRADIUS: int = 32 * FRACUNIT

# View slope limit constants matching Hexen reference (100 * FRACUNIT / 160)
TOPSLOPE: int = 40960
BOTTOMSLOPE: int = -40960

__all__ = [
    "BOTTOMSLOPE",
    "MAXRADIUS",
    "TOPSLOPE",
    "aim_line_attack",
    "check_missile_spawn",
    "check_sight",
    "face_mobj",
    "line_attack",
    "radius_attack",
    "seeker_missile",
    "spawn_blood_splatter",
    "spawn_blood_splatter2",
    "spawn_missile",
    "spawn_missile_angle",
    "spawn_missile_angle_speed",
    "spawn_missile_xyz",
    "spawn_puff",
    "spawn_ripper_blood",
]


def face_mobj(source: Mobj, target: Mobj) -> tuple[int, int]:
    """Port of P_FaceMobj (p_mobj.c).

    Returns (dir, delta) where dir is 1 for clockwise, 0 for counter-clockwise,
    and delta is the angular difference in BAM units.
    """
    angle1 = source.angle & 0xFFFFFFFF
    angle2 = point_to_angle2(source.x, source.y, target.x, target.y) & 0xFFFFFFFF
    if angle2 > angle1:
        diff = angle2 - angle1
        if diff > ANG180:
            delta = (ANG_MAX - diff) & 0xFFFFFFFF
            return (0, delta)
        else:
            return (1, diff)
    else:
        diff = angle1 - angle2
        if diff > ANG180:
            delta = (ANG_MAX - diff) & 0xFFFFFFFF
            return (1, delta)
        else:
            return (0, diff)


def seeker_missile(
    actor: Mobj,
    thresh: int,
    turn_max: int,
    sim: GameSim | None = None,
) -> bool:
    """Port of P_SeekerMissile (p_mobj.c).

    Adjusts actor heading toward target stored in actor.special1 (or actor.target).
    Returns True if target was tracked, False otherwise.
    """
    target = actor.special1 if isinstance(actor.special1, Mobj) else None
    if target is None and isinstance(actor.target, Mobj):
        target = actor.target

    if target is None:
        return False

    if not (target.flags & MobjFlag.MF_SHOOTABLE):
        # Target died or lost shootability
        actor.special1 = None
        return False

    dir_turn, delta = face_mobj(actor, target)
    if delta > thresh:
        delta >>= 1
        if delta > turn_max:
            delta = turn_max

    if dir_turn:
        actor.angle = (actor.angle + delta) & 0xFFFFFFFF
    else:
        actor.angle = (actor.angle - delta) & 0xFFFFFFFF

    fine = (actor.angle >> ANGLETOFINESHIFT) & FINEMASK
    actor.momx = fixed_mul(actor.info.speed, fine_cosine(fine))
    actor.momy = fixed_mul(actor.info.speed, fine_sine(fine))

    if actor.z + actor.height < target.z or target.z + target.height < actor.z:
        dist = approx_distance(target.x - actor.x, target.y - actor.y)
        if actor.info.speed > 0:
            dist = dist // actor.info.speed
        else:
            dist = 1
        if dist < 1:
            dist = 1
        target_midz = target.z + (target.height >> 1)
        actor_midz = actor.z + (actor.height >> 1)
        actor.momz = (target_midz - actor_midz) // dist

    return True


def check_missile_spawn(sim: GameSim, missile: Mobj) -> bool:
    """Port of P_CheckMissileSpawn (p_mobj.c).

    Advances missile by half momentum. If blocked by geometry/actors, explodes it and returns False.
    """
    new_x = missile.x + (missile.momx >> 1)
    new_y = missile.y + (missile.momy >> 1)
    missile.z += missile.momz >> 1

    from pyhexen.sim.map import try_move
    from pyhexen.sim.movement import explode_missile

    if not try_move(sim, missile, new_x, new_y):
        explode_missile(sim, missile)
        return False
    return True


def spawn_missile(
    sim: GameSim,
    source: Mobj,
    dest: Mobj,
    missile_type: MobjType | int,
) -> Mobj | None:
    """Port of P_SpawnMissile (p_mobj.c).

    Spawns projectile at source aiming toward dest. Returns missile Mobj,
    or None if it exploded immediately.
    """
    mtype = MobjType(missile_type)
    match mtype:
        case MobjType.MT_MNTRFX1:
            z = source.z + 40 * FRACUNIT
        case MobjType.MT_MNTRFX2:
            z = ONFLOORZ + source.floorclip
        case MobjType.MT_CENTAUR_FX:
            z = source.z + 45 * FRACUNIT
        case MobjType.MT_ICEGUY_FX:
            z = source.z + 40 * FRACUNIT
        case MobjType.MT_HOLY_MISSILE:
            z = source.z + 40 * FRACUNIT
        case _:
            z = source.z + 32 * FRACUNIT

    z -= source.floorclip
    th = spawn_mobj(sim, source.x, source.y, z, mtype)
    if th.info.seesound:
        sim.events.emit(
            SoundEvent(
                sound_id=int(th.info.seesound),
                origin_id=th.tid,
                x=th.x,
                y=th.y,
                z=th.z,
            )
        )

    th.target = source
    an = point_to_angle2(source.x, source.y, dest.x, dest.y)
    if dest.flags & MobjFlag.MF_SHADOW:
        an = (an + (sim.rng.p_subrandom() << 21)) & 0xFFFFFFFF
    th.angle = an

    an_fine = (an >> ANGLETOFINESHIFT) & FINEMASK
    th.momx = fixed_mul(th.info.speed, fine_cosine(an_fine))
    th.momy = fixed_mul(th.info.speed, fine_sine(an_fine))

    dist = approx_distance(dest.x - source.x, dest.y - source.y)
    if th.info.speed > 0:
        dist = dist // th.info.speed
    else:
        dist = 1
    if dist < 1:
        dist = 1

    th.momz = (dest.z - source.z) // dist
    return th if check_missile_spawn(sim, th) else None


def spawn_missile_xyz(
    sim: GameSim,
    x: int,
    y: int,
    z: int,
    source: Mobj,
    dest: Mobj,
    missile_type: MobjType | int,
) -> Mobj | None:
    """Port of P_SpawnMissileXYZ (p_mobj.c)."""
    mtype = MobjType(missile_type)
    spawn_z = z - source.floorclip
    th = spawn_mobj(sim, x, y, spawn_z, mtype)
    if th.info.seesound:
        sim.events.emit(
            SoundEvent(
                sound_id=int(th.info.seesound),
                origin_id=th.tid,
                x=th.x,
                y=th.y,
                z=th.z,
            )
        )

    th.target = source
    an = point_to_angle2(source.x, source.y, dest.x, dest.y)
    if dest.flags & MobjFlag.MF_SHADOW:
        an = (an + (sim.rng.p_subrandom() << 21)) & 0xFFFFFFFF
    th.angle = an

    an_fine = (an >> ANGLETOFINESHIFT) & FINEMASK
    th.momx = fixed_mul(th.info.speed, fine_cosine(an_fine))
    th.momy = fixed_mul(th.info.speed, fine_sine(an_fine))

    dist = approx_distance(dest.x - source.x, dest.y - source.y)
    if th.info.speed > 0:
        dist = dist // th.info.speed
    else:
        dist = 1
    if dist < 1:
        dist = 1

    th.momz = (dest.z - source.z) // dist
    return th if check_missile_spawn(sim, th) else None


def spawn_missile_angle(
    sim: GameSim,
    source: Mobj,
    missile_type: MobjType | int,
    angle: int,
    vz: int,
) -> Mobj | None:
    """Port of P_SpawnMissileAngle (p_mobj.c).

    Spawns projectile at source with explicit BAM angle and vertical velocity vz.
    """
    mtype = MobjType(missile_type)
    match mtype:
        case MobjType.MT_MNTRFX1:
            z = source.z + 40 * FRACUNIT
        case MobjType.MT_MNTRFX2:
            z = ONFLOORZ + source.floorclip
        case MobjType.MT_ICEGUY_FX2:
            z = source.z + 3 * FRACUNIT
        case MobjType.MT_MSTAFF_FX2:
            z = source.z + 40 * FRACUNIT
        case _:
            z = source.z + 32 * FRACUNIT

    z -= source.floorclip
    mo = spawn_mobj(sim, source.x, source.y, z, mtype)
    if mo.info.seesound:
        sim.events.emit(
            SoundEvent(
                sound_id=int(mo.info.seesound),
                origin_id=mo.tid,
                x=mo.x,
                y=mo.y,
                z=mo.z,
            )
        )

    mo.target = source
    mo.angle = angle & 0xFFFFFFFF
    an_fine = (mo.angle >> ANGLETOFINESHIFT) & FINEMASK
    mo.momx = fixed_mul(mo.info.speed, fine_cosine(an_fine))
    mo.momy = fixed_mul(mo.info.speed, fine_sine(an_fine))
    mo.momz = vz
    return mo if check_missile_spawn(sim, mo) else None


def spawn_missile_angle_speed(
    sim: GameSim,
    source: Mobj,
    missile_type: MobjType | int,
    angle: int,
    vz: int,
    speed: int,
) -> Mobj | None:
    """Port of P_SpawnMissileAngleSpeed (p_mobj.c)."""
    mtype = MobjType(missile_type)
    z = source.z - source.floorclip
    mo = spawn_mobj(sim, source.x, source.y, z, mtype)
    mo.target = source
    mo.angle = angle & 0xFFFFFFFF
    an_fine = (mo.angle >> ANGLETOFINESHIFT) & FINEMASK
    mo.momx = fixed_mul(speed, fine_cosine(an_fine))
    mo.momy = fixed_mul(speed, fine_sine(an_fine))
    mo.momz = vz
    return mo if check_missile_spawn(sim, mo) else None


def spawn_puff(
    sim: GameSim,
    x: int,
    y: int,
    z: int,
    puff_type: MobjType | int = MobjType.MT_PUNCHPUFF,
    linetarget: Mobj | None = None,
) -> Mobj:
    """Port of P_SpawnPuff (p_mobj.c)."""
    mtype = MobjType(puff_type)
    jitter_z = z + (sim.rng.p_subrandom() << 10)
    puff = spawn_mobj(sim, x, y, jitter_z, mtype)

    if linetarget is not None and puff.info.seesound:
        sim.events.emit(
            SoundEvent(
                sound_id=int(puff.info.seesound),
                origin_id=puff.tid,
                x=puff.x,
                y=puff.y,
                z=puff.z,
            )
        )
    elif puff.info.attacksound:
        sim.events.emit(
            SoundEvent(
                sound_id=int(puff.info.attacksound),
                origin_id=puff.tid,
                x=puff.x,
                y=puff.y,
                z=puff.z,
            )
        )

    match mtype:
        case MobjType.MT_PUNCHPUFF:
            puff.momz = FRACUNIT
        case MobjType.MT_HAMMERPUFF:
            puff.momz = (4 * FRACUNIT) // 5
        case _:
            pass

    return puff


def spawn_blood_splatter(
    sim: GameSim,
    x: int,
    y: int,
    z: int,
    originator: Mobj,
) -> Mobj:
    """Port of P_BloodSplatter (p_mobj.c). Spawns MT_BLOODSPLATTER."""
    mo = spawn_mobj(sim, x, y, z, MobjType.MT_BLOODSPLATTER)
    mo.target = originator
    mo.momx = sim.rng.p_subrandom() << 10
    mo.momy = sim.rng.p_subrandom() << 10
    mo.momz = 3 * FRACUNIT
    return mo


def spawn_blood_splatter2(
    sim: GameSim,
    x: int,
    y: int,
    z: int,
    originator: Mobj,
) -> Mobj:
    """Port of P_BloodSplatter2 (p_mobj.c). Spawns MT_AXEBLOOD."""
    r1 = sim.rng.p_random()
    r2 = sim.rng.p_random()
    offset_x = (r2 - 128) << 11
    offset_y = (r1 - 128) << 11
    mo = spawn_mobj(sim, x + offset_x, y + offset_y, z, MobjType.MT_AXEBLOOD)
    mo.target = originator
    return mo


def spawn_ripper_blood(sim: GameSim, mo: Mobj) -> Mobj:
    """Port of P_RipperBlood (p_mobj.c). Spawns MT_BLOOD with halved momentum."""
    offset_x = sim.rng.p_subrandom() << 12
    offset_y = sim.rng.p_subrandom() << 12
    offset_z = sim.rng.p_subrandom() << 12
    th = spawn_mobj(sim, mo.x + offset_x, mo.y + offset_y, mo.z + offset_z, MobjType.MT_BLOOD)
    th.momx = mo.momx >> 1
    th.momy = mo.momy >> 1
    th.tics += sim.rng.p_random() & 3
    return th


def aim_line_attack(
    sim: GameSim,
    t1: Mobj,
    angle: int,
    distance: int,
) -> int:
    """Port of P_AimLineAttack (p_map.c).

    Traces raycast from t1 along BAM angle up to distance. Returns slope (16.16 fixed-point)
    and stores target in sim.linetarget and sim.map_state.linetarget. Returns 0 if no target.
    """
    angle_fine = ((angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
    x2 = t1.x + fixed_mul(distance, fine_cosine(angle_fine))
    y2 = t1.y + fixed_mul(distance, fine_sine(angle_fine))
    shootz = t1.z + (t1.height >> 1) + 8 * FRACUNIT
    topslope = TOPSLOPE
    bottomslope = BOTTOMSLOPE
    attackrange = distance

    linetarget: Mobj | None = None
    aimslope: int = 0

    def aim_traverse(in_intercept: Intercept) -> bool:
        nonlocal topslope, bottomslope, linetarget, aimslope

        if in_intercept.is_aline:
            li = in_intercept.line
            if li is None or not (li.flags & ML_TWOSIDED):
                return False  # 1-sided line blocks aim

            open_info = line_opening(li)
            if open_info.openbottom >= open_info.opentop:
                return False  # door closed completely

            dist = fixed_mul(attackrange, in_intercept.frac)
            if dist <= 0:
                dist = 1

            if (
                li.front_sector is not None
                and li.back_sector is not None
                and li.front_sector.floor_height != li.back_sector.floor_height
            ):
                slope = fixed_div(open_info.openbottom - shootz, dist)
                if slope > bottomslope:
                    bottomslope = slope

            if (
                li.front_sector is not None
                and li.back_sector is not None
                and li.front_sector.ceiling_height != li.back_sector.ceiling_height
            ):
                slope = fixed_div(open_info.opentop - shootz, dist)
                if slope < topslope:
                    topslope = slope

            if topslope <= bottomslope:
                return False

            return True

        th = in_intercept.thing
        if th is None or th is t1:
            return True
        if not (th.flags & MobjFlag.MF_SHOOTABLE):
            return True
        if th.player is not None and sim.netgame and not sim.deathmatch:
            return True

        dist = fixed_mul(attackrange, in_intercept.frac)
        if dist <= 0:
            dist = 1

        thingtopslope = fixed_div(th.z + th.height - shootz, dist)
        if thingtopslope < bottomslope:
            return True  # shot over the thing
        thingbottomslope = fixed_div(th.z - shootz, dist)
        if thingbottomslope > topslope:
            return True  # shot under the thing

        if thingtopslope > topslope:
            thingtopslope = topslope
        if thingbottomslope < bottomslope:
            thingbottomslope = bottomslope

        aimslope = (thingtopslope + thingbottomslope) // 2
        linetarget = th
        return False

    if sim.level is not None:
        path_traverse(sim.level, t1.x, t1.y, x2, y2, PT_ADDLINES | PT_ADDTHINGS, aim_traverse)
    else:
        # Standalone sim without level geometry: build thing intercepts directly
        trace = DivLine(x=t1.x, y=t1.y, dx=x2 - t1.x, dy=y2 - t1.y)
        intercepts: list[Intercept] = []
        tracepositive = (trace.dx ^ trace.dy) > 0
        for th in sim.thinkers:
            if not isinstance(th, Mobj) or th is t1:
                continue
            if tracepositive:
                tx1 = th.x - th.radius
                ty1 = th.y + th.radius
                tx2 = th.x + th.radius
                ty2 = th.y - th.radius
            else:
                tx1 = th.x - th.radius
                ty1 = th.y - th.radius
                tx2 = th.x + th.radius
                ty2 = th.y + th.radius
            s1 = point_on_divline_side(tx1, ty1, trace)
            s2 = point_on_divline_side(tx2, ty2, trace)
            if s1 == s2:
                continue
            dl = DivLine(x=tx1, y=ty1, dx=tx2 - tx1, dy=ty2 - ty1)
            frac = intercept_vector(trace, dl)
            if 0 <= frac <= FRACUNIT:
                intercepts.append(Intercept(frac=frac, is_aline=False, thing=th))
        _traverse_intercepts(intercepts, FRACUNIT, aim_traverse)

    sim.linetarget = linetarget
    if hasattr(sim, "map_state"):
        sim.map_state.linetarget = linetarget

    if linetarget is not None:
        return aimslope
    return 0


def line_attack(
    sim: GameSim,
    t1: Mobj,
    angle: int,
    distance: int,
    slope: int,
    damage: int,
    puff_type: MobjType | int = MobjType.MT_PUNCHPUFF,
) -> None:
    """Port of P_LineAttack (p_map.c).

    Fires hitscan ray along angle and slope. Inflicts damage via damage_mobj, activates impact
    lines, spawns puffs and blood splatters, and emits sound events.
    """
    m_puff_type = MobjType(puff_type)
    angle_fine = ((angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
    x2 = t1.x + fixed_mul(distance, fine_cosine(angle_fine))
    y2 = t1.y + fixed_mul(distance, fine_sine(angle_fine))
    shootz = t1.z + (t1.height >> 1) + 8 * FRACUNIT - t1.floorclip
    attackrange = distance
    aimslope = slope

    from pyhexen.sim.inter import damage_mobj
    from pyhexen.sim.line_activation import p_activate_line

    def shoot_traverse(in_intercept: Intercept) -> bool:
        if in_intercept.is_aline:
            li = in_intercept.line
            if li is None:
                return False

            if li.special:
                p_activate_line(sim, li, t1, side=0, activation_type=SPAC_IMPACT)

            if not (li.flags & ML_TWOSIDED):
                return _hit_line(in_intercept, li)

            open_info = line_opening(li)
            dist = fixed_mul(attackrange, in_intercept.frac)
            if dist <= 0:
                dist = 1

            if (
                li.front_sector is not None
                and li.back_sector is not None
                and li.front_sector.floor_height != li.back_sector.floor_height
            ):
                slope_check = fixed_div(open_info.openbottom - shootz, dist)
                if slope_check > aimslope:
                    return _hit_line(in_intercept, li)

            if (
                li.front_sector is not None
                and li.back_sector is not None
                and li.front_sector.ceiling_height != li.back_sector.ceiling_height
            ):
                slope_check = fixed_div(open_info.opentop - shootz, dist)
                if slope_check < aimslope:
                    return _hit_line(in_intercept, li)

            return True

        th = in_intercept.thing
        if th is None or th is t1:
            return True
        if not (th.flags & MobjFlag.MF_SHOOTABLE):
            return True

        dist = fixed_mul(attackrange, in_intercept.frac)
        if dist <= 0:
            dist = 1

        thingtopslope = fixed_div(th.z + th.height - shootz, dist)
        if thingtopslope < aimslope:
            return True  # shot over the thing
        thingbottomslope = fixed_div(th.z - shootz, dist)
        if thingbottomslope > aimslope:
            return True  # shot under the thing

        # Hit thing! Setback = 10 * FRACUNIT / attackrange
        setback = fixed_div(10 * FRACUNIT, attackrange) if attackrange > 0 else 0
        frac = in_intercept.frac - setback
        hit_x = t1.x + fixed_mul(x2 - t1.x, frac)
        hit_y = t1.y + fixed_mul(y2 - t1.y, frac)
        hit_z = shootz + fixed_mul(aimslope, fixed_mul(frac, attackrange))

        sim.linetarget = th
        if hasattr(sim, "map_state"):
            sim.map_state.linetarget = th

        spawn_puff(sim, hit_x, hit_y, hit_z, puff_type=m_puff_type, linetarget=th)

        if damage > 0:
            if not (th.flags & MobjFlag.MF_NOBLOOD) and not (
                th.flags2 & MobjFlag2.MF2_INVULNERABLE
            ):
                if m_puff_type in (MobjType.MT_AXEPUFF, MobjType.MT_AXEPUFF_GLOW):
                    spawn_blood_splatter2(sim, hit_x, hit_y, hit_z, th)
                if sim.rng.p_random() < 192:
                    spawn_blood_splatter(sim, hit_x, hit_y, hit_z, th)

            if m_puff_type == MobjType.MT_FLAMEPUFF2:
                lava = Mobj(
                    type=MobjType.MT_CIRCLEFLAME,
                    flags2=MobjFlag2.MF2_FIREDAMAGE | MobjFlag2.MF2_NODMGTHRUST,
                )
                damage_mobj(sim, target=th, inflictor=lava, source=t1, damage=damage)
            else:
                damage_mobj(sim, target=th, inflictor=t1, source=t1, damage=damage)

        return False

    def _hit_line(in_intercept: Intercept, li) -> bool:
        # Hit line setback = 4 * FRACUNIT / attackrange
        setback = fixed_div(4 * FRACUNIT, attackrange) if attackrange > 0 else 0
        frac = in_intercept.frac - setback
        hit_x = t1.x + fixed_mul(x2 - t1.x, frac)
        hit_y = t1.y + fixed_mul(y2 - t1.y, frac)
        hit_z = shootz + fixed_mul(aimslope, fixed_mul(frac, attackrange))

        if li.front_sector is not None and li.front_sector.ceiling_pic.startswith("F_SKY"):
            if hit_z > li.front_sector.ceiling_height:
                return False  # Sky absorption
            if li.back_sector is not None and li.back_sector.ceiling_pic.startswith("F_SKY"):
                return False  # Sky hack wall

        sim.linetarget = None
        if hasattr(sim, "map_state"):
            sim.map_state.linetarget = None
        spawn_puff(sim, hit_x, hit_y, hit_z, puff_type=m_puff_type, linetarget=None)
        return False

    if sim.level is not None:
        unobstructed = path_traverse(
            sim.level, t1.x, t1.y, x2, y2, PT_ADDLINES | PT_ADDTHINGS, shoot_traverse
        )
    else:
        trace = DivLine(x=t1.x, y=t1.y, dx=x2 - t1.x, dy=y2 - t1.y)
        intercepts = []
        tracepositive = (trace.dx ^ trace.dy) > 0
        for th in sim.thinkers:
            if not isinstance(th, Mobj) or th is t1:
                continue
            if tracepositive:
                tx1 = th.x - th.radius
                ty1 = th.y + th.radius
                tx2 = th.x + th.radius
                ty2 = th.y - th.radius
            else:
                tx1 = th.x - th.radius
                ty1 = th.y - th.radius
                tx2 = th.x + th.radius
                ty2 = th.y + th.radius
            s1 = point_on_divline_side(tx1, ty1, trace)
            s2 = point_on_divline_side(tx2, ty2, trace)
            if s1 == s2:
                continue
            dl = DivLine(x=tx1, y=ty1, dx=tx2 - tx1, dy=ty2 - ty1)
            frac = intercept_vector(trace, dl)
            if 0 <= frac <= FRACUNIT:
                intercepts.append(Intercept(frac=frac, is_aline=False, thing=th))
        unobstructed = _traverse_intercepts(intercepts, FRACUNIT, shoot_traverse)

    if unobstructed:
        # Ray reached full distance without hit
        if m_puff_type == MobjType.MT_PUNCHPUFF:
            sim.events.emit(
                SoundEvent(
                    sound_id=int(SfxEnum.SFX_FIGHTER_PUNCH_MISS),
                    origin_id=t1.tid,
                    x=t1.x,
                    y=t1.y,
                    z=t1.z,
                )
            )
        elif m_puff_type in (
            MobjType.MT_HAMMERPUFF,
            MobjType.MT_AXEPUFF,
            MobjType.MT_AXEPUFF_GLOW,
        ):
            sim.events.emit(
                SoundEvent(
                    sound_id=int(SfxEnum.SFX_FIGHTER_HAMMER_MISS),
                    origin_id=t1.tid,
                    x=t1.x,
                    y=t1.y,
                    z=t1.z,
                )
            )
        elif m_puff_type == MobjType.MT_FLAMEPUFF:
            spawn_puff(
                sim,
                x2,
                y2,
                shootz + fixed_mul(slope, distance),
                puff_type=m_puff_type,
                linetarget=None,
            )


def check_sight(sim: GameSim, t1: Mobj, t2: Mobj) -> bool:
    """Port of P_CheckSight (p_sight.c).

    Returns True if an unobstructed line of sight exists between t1 and t2.
    """
    if sim.level is None:
        return True

    x1, y1 = t1.x, t1.y
    z1 = t1.z + (t1.height >> 1)
    x2, y2 = t2.x, t2.y
    z2 = t2.z + (t2.height >> 1)

    if x1 == x2 and y1 == y2:
        return True

    sight_blocked = False

    def sight_traverse(in_intercept: Intercept) -> bool:
        nonlocal sight_blocked
        if not in_intercept.is_aline:
            return True

        li = in_intercept.line
        if li is None:
            return True

        if not (li.flags & ML_TWOSIDED):
            sight_blocked = True
            return False  # 1-sided line blocks vision

        open_info = line_opening(li)
        if open_info.openbottom >= open_info.opentop:
            sight_blocked = True
            return False  # closed portal/door

        ray_z = z1 + fixed_mul(z2 - z1, in_intercept.frac)
        if ray_z < open_info.openbottom or ray_z > open_info.opentop:
            sight_blocked = True
            return False  # ray clipped by floor/ceiling step

        return True

    path_traverse(sim.level, x1, y1, x2, y2, PT_ADDLINES, sight_traverse)
    return not sight_blocked


def radius_attack(
    sim: GameSim,
    spot: Mobj,
    source: Mobj | None,
    damage: int,
    distance: int | None = None,
    damage_source: bool = False,
) -> None:
    """Port of P_RadiusAttack and PIT_RadiusAttack (p_map.c).

    Inflicts distance-falloff splash damage to shootable actors in surrounding blocks
    via damage_mobj.
    """
    if damage <= 0:
        return

    if distance is None:
        distance = damage

    if distance <= 0:
        return

    from pyhexen.sim.inter import damage_mobj

    targets: list[Mobj] = []
    if sim.level is not None and sim.level.blockmap is not None:
        bm = sim.level.blockmap
        dist_box = (distance + MAXRADIUS) << FRACBITS
        yh = (spot.y + dist_box - bm.org_y) >> MAPBLOCKSHIFT
        yl = (spot.y - dist_box - bm.org_y) >> MAPBLOCKSHIFT
        xh = (spot.x + dist_box - bm.org_x) >> MAPBLOCKSHIFT
        xl = (spot.x - dist_box - bm.org_x) >> MAPBLOCKSHIFT

        yl = max(0, min(bm.height - 1, yl))
        yh = max(0, min(bm.height - 1, yh))
        xl = max(0, min(bm.width - 1, xl))
        xh = max(0, min(bm.width - 1, xh))

        visited: set[int] = set()
        for by in range(yl, yh + 1):
            for bx in range(xl, xh + 1):
                idx = by * bm.width + bx
                mo = bm.block_links[idx]
                while mo is not None:
                    if id(mo) not in visited:
                        visited.add(id(mo))
                        targets.append(mo)
                    mo = mo.bnext
    else:
        for th in sim.thinkers:
            if isinstance(th, Mobj):
                targets.append(th)

    for thing in targets:
        if not (thing.flags & MobjFlag.MF_SHOOTABLE):
            continue

        if not damage_source and thing is source:
            continue

        # Vertical culling: abs(dz >> FRACBITS) > 2 * distance
        if abs((thing.z - spot.z) >> FRACBITS) > 2 * distance:
            continue

        dx = abs(thing.x - spot.x)
        dy = abs(thing.y - spot.y)
        chebyshev = dx if dx > dy else dy
        dist = (chebyshev - thing.radius) >> FRACBITS
        if dist < 0:
            dist = 0
        if dist >= distance:
            continue

        if check_sight(sim, thing, spot):
            calculated_damage = c_div(damage * (distance - dist), distance) + 1
            if thing.player is not None:
                calculated_damage >>= 2

            if calculated_damage > 0:
                damage_mobj(
                    sim,
                    target=thing,
                    inflictor=spot,
                    source=source,
                    damage=calculated_damage,
                )
