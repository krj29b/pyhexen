# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Cleric weapon action functions (Task T10.8).

Ports `A_CMaceAttack`, `A_CStaffCheck`, `A_CStaffAttack`, `A_CStaffMissileSlither`,
`A_CStaffInitBlink`, `A_CStaffCheckBlink`, `A_CFlameAttack`, `A_CFlamePuff`,
`A_CFlameMissile`, `A_CFlameRotate`, `A_CHolyAttack`, and `A_CHolyPalette`
from Chocolate Doom's `src/hexen/p_pspr.c`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG45,
    ANG90,
    ANG180,
    ANGLETOFINESHIFT,
    FINEMASK,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACUNIT, fixed_mul, to_uint32
from pyhexen.sim.actions import action
from pyhexen.sim.attacks import (
    aim_line_attack,
    line_attack,
    spawn_player_missile,
    spm_angle,
)
from pyhexen.sim.enemy import MELEERANGE
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType, SfxEnum, StateNum
from pyhexen.sim.map import try_move
from pyhexen.sim.mobj import FLOAT_BOB_OFFSETS, Mobj, set_mobj_state, spawn_mobj
from pyhexen.sim.movement import p_blaster_mobj_thinker
from pyhexen.sim.psprites import (
    MANA_1,
    MANA_2,
    WEAPON_MANA_USE,
    PspDef,
    _emit_sound,
    _resolve_player_and_psp,
    adjust_player_angle,
    p_set_psprite,
    ps_weapon,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.player import Player

FLAMESPEED: int = 29491  # 0.45 * FRACUNIT
FLAMEROTSPEED: int = 2 * FRACUNIT
STARTHOLYPAL: int = 22


@action("A_CMaceAttack")
def a_cmaceattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CMaceAttack (p_pspr.c:1467).

    Mace of Contrition melee attack (damage 25..40, range 2 * MELEERANGE).
    Searches 16 angle offsets in +/- 45-degree cone; strikes walls if no actor hit.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    damage = 25 + (sim.rng.p_random() & 15)
    puff_type = MobjType.MT_HAMMERPUFF

    for i in range(16):
        # Positive offset
        angle = to_uint32(pmo.angle + i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, 2 * MELEERANGE)
        if sim.linetarget is not None:
            line_attack(sim, pmo, angle, 2 * MELEERANGE, slope, damage, puff_type)
            adjust_player_angle(sim, pmo)
            return

        # Negative offset
        angle = to_uint32(pmo.angle - i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, 2 * MELEERANGE)
        if sim.linetarget is not None:
            line_attack(sim, pmo, angle, 2 * MELEERANGE, slope, damage, puff_type)
            adjust_player_angle(sim, pmo)
            return

    # Didn't find any creatures, so try to strike any walls
    pmo.special1 = None
    angle = pmo.angle
    slope = aim_line_attack(sim, pmo, angle, MELEERANGE)
    line_attack(sim, pmo, angle, MELEERANGE, slope, damage, puff_type)


@action("A_CStaffCheck")
def a_cstaffcheck(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CStaffCheck (p_pspr.c:1515).

    Serpent Staff melee drain check (range 1.5 * MELEERANGE).
    Drains health (capped at 100) and advances psprite to S_CSTAFFATK2_1.
    If no target in range, psprite will naturally proceed to A_CStaffAttack.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    damage = 20 + (sim.rng.p_random() & 15)
    puff_type = MobjType.MT_CSTAFFPUFF
    staff_range = (3 * MELEERANGE) // 2

    for i in range(3):
        # Positive offset
        angle = to_uint32(pmo.angle + i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, staff_range)
        linetarget = sim.linetarget
        if linetarget is not None:
            line_attack(sim, pmo, angle, staff_range, slope, damage, puff_type)
            pmo.angle = point_to_angle2(pmo.x, pmo.y, linetarget.x, linetarget.y)
            if (
                linetarget.player is not None or bool(linetarget.flags & MobjFlag.MF_COUNTKILL)
            ) and not (linetarget.flags2 & (MobjFlag2.MF2_DORMANT | MobjFlag2.MF2_INVULNERABLE)):
                new_life = player.health + (damage >> 3)
                if new_life > 100:
                    new_life = 100
                player.health = new_life
                pmo.health = new_life
                p_set_psprite(sim, player, ps_weapon, StateNum.S_CSTAFFATK2_1)
            player.mana[MANA_1] -= WEAPON_MANA_USE[player.player_class][player.readyweapon]
            break

        # Negative offset
        angle = to_uint32(pmo.angle - i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, staff_range)
        linetarget = sim.linetarget
        if linetarget is not None:
            line_attack(sim, pmo, angle, staff_range, slope, damage, puff_type)
            pmo.angle = point_to_angle2(pmo.x, pmo.y, linetarget.x, linetarget.y)
            if linetarget.player is not None or bool(linetarget.flags & MobjFlag.MF_COUNTKILL):
                new_life = player.health + (damage >> 4)
                if new_life > 100:
                    new_life = 100
                player.health = new_life
                pmo.health = new_life
                p_set_psprite(sim, player, ps_weapon, StateNum.S_CSTAFFATK2_1)
            player.mana[MANA_1] -= WEAPON_MANA_USE[player.player_class][player.readyweapon]
            break


@action("A_CStaffAttack")
def a_cstaffattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CStaffAttack (p_pspr.c:1575).

    Fires two Serpent Staff missiles angled slightly left and right.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    player.mana[MANA_1] -= WEAPON_MANA_USE[player.player_class][player.readyweapon]

    mo1 = spm_angle(sim, pmo, MobjType.MT_CSTAFF_MISSILE, to_uint32(pmo.angle - (ANG45 // 15)))
    if mo1 is not None:
        mo1.special2 = 32

    mo2 = spm_angle(sim, pmo, MobjType.MT_CSTAFF_MISSILE, to_uint32(pmo.angle + (ANG45 // 15)))
    if mo2 is not None:
        mo2.special2 = 0

    _emit_sound(sim, pmo, SfxEnum.SFX_CLERIC_CSTAFF_FIRE)


@action("A_CStaffMissileSlither")
def a_cstaffmissileslither(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CStaffMissileSlither (p_pspr.c:1601).

    Applies sinusoidal bobbing perpendicular to missile trajectory.
    """
    weave_xy = actor.special2 if isinstance(actor.special2, int) else 0
    angle = ((actor.angle + ANG90) >> ANGLETOFINESHIFT) & FINEMASK

    new_x = actor.x - fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy])
    new_y = actor.y - fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy])

    weave_xy = (weave_xy + 3) & 63

    new_x += fixed_mul(fine_cosine(angle), FLOAT_BOB_OFFSETS[weave_xy])
    new_y += fixed_mul(fine_sine(angle), FLOAT_BOB_OFFSETS[weave_xy])

    try_move(sim, actor, new_x, new_y)
    actor.special2 = weave_xy


@action("A_CStaffInitBlink")
def a_cstaffinitblink(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CStaffInitBlink (p_pspr.c:1624).

    Initializes eye blink countdown timer on player mo.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    if player.mo is not None:
        player.mo.special1 = (sim.rng.p_random() >> 1) + 20


@action("A_CStaffCheckBlink")
def a_cstaffcheckblink(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CStaffCheckBlink (p_pspr.c:1635).

    Decrements blink countdown timer; triggers S_CSTAFFBLINK1 on expiry.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    timer = (pmo.special1 if isinstance(pmo.special1, int) else 1) - 1
    pmo.special1 = timer
    if timer <= 0:
        p_set_psprite(sim, player, ps_weapon, StateNum.S_CSTAFFBLINK1)
        pmo.special1 = (sim.rng.p_random() + 50) >> 2


@action("A_CFlameAttack")
def a_cflameattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CFlameAttack (p_pspr.c:1653).

    Fires Firestorm missile with sub-stepping blaster thinker.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    mo = spawn_player_missile(sim, pmo, MobjType.MT_CFLAME_MISSILE)
    if mo is not None:
        mo.thinker_func = p_blaster_mobj_thinker
        mo.special1 = 2

    player.mana[MANA_2] -= WEAPON_MANA_USE[player.player_class][player.readyweapon]
    _emit_sound(sim, pmo, SfxEnum.SFX_CLERIC_FLAME_FIRE)


@action("A_CFlamePuff")
def a_cflamepuff(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CFlamePuff (p_pspr.c:1674).

    Clears MF2_DONTDRAW, zeroes momentum, and plays explosion sound.
    """
    actor.flags2 &= ~MobjFlag2.MF2_DONTDRAW
    actor.momx = 0
    actor.momy = 0
    actor.momz = 0
    _emit_sound(sim, actor, SfxEnum.SFX_CLERIC_FLAME_EXPLODE)


@action("A_CFlameMissile")
def a_cflamemissile(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CFlameMissile (p_pspr.c:1689).

    Un-hides flame puff, plays explosion sound, and if hitting a shootable actor,
    spawns an 8-point orbiting circle of flames around the victim.
    """
    actor.flags2 &= ~MobjFlag2.MF2_DONTDRAW
    _emit_sound(sim, actor, SfxEnum.SFX_CLERIC_FLAME_EXPLODE)

    blocking = sim.map_state.blocking_mobj if hasattr(sim, "map_state") else None
    if blocking is not None and bool(blocking.flags & MobjFlag.MF_SHOOTABLE):
        dist = blocking.radius + 18 * FRACUNIT
        for i in range(4):
            an = ((i * ANG45) >> ANGLETOFINESHIFT) & FINEMASK

            # Positive offset flame
            mo = spawn_mobj(
                sim,
                blocking.x + fixed_mul(dist, fine_cosine(an)),
                blocking.y + fixed_mul(dist, fine_sine(an)),
                blocking.z + 5 * FRACUNIT,
                MobjType.MT_CIRCLEFLAME,
            )
            if mo is not None:
                mo.angle = to_uint32(an << ANGLETOFINESHIFT)
                mo.target = actor.target
                mo.momx = fixed_mul(FLAMESPEED, fine_cosine(an))
                mo.special1 = mo.momx
                mo.momy = fixed_mul(FLAMESPEED, fine_sine(an))
                mo.special2 = mo.momy
                mo.tics -= sim.rng.p_random() & 3

            # Negative offset flame (180 opposite)
            mo = spawn_mobj(
                sim,
                blocking.x - fixed_mul(dist, fine_cosine(an)),
                blocking.y - fixed_mul(dist, fine_sine(an)),
                blocking.z + 5 * FRACUNIT,
                MobjType.MT_CIRCLEFLAME,
            )
            if mo is not None:
                mo.angle = to_uint32(ANG180 + (an << ANGLETOFINESHIFT))
                mo.target = actor.target
                mo.momx = fixed_mul(-FLAMESPEED, fine_cosine(an))
                mo.special1 = mo.momx
                mo.momy = fixed_mul(-FLAMESPEED, fine_sine(an))
                mo.special2 = mo.momy
                mo.tics -= sim.rng.p_random() & 3

        set_mobj_state(sim, actor, StateNum.S_FLAMEPUFF2_1)


@action("A_CFlameRotate")
def a_cflamerotate(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CFlameRotate (p_pspr.c:1824).

    Orbiting circle flame tangential rotation and angle step.
    """
    an = (((actor.angle + ANG90) & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
    sp1 = actor.special1 if isinstance(actor.special1, int) else 0
    sp2 = actor.special2 if isinstance(actor.special2, int) else 0
    actor.momx = sp1 + fixed_mul(FLAMEROTSPEED, fine_cosine(an))
    actor.momy = sp2 + fixed_mul(FLAMEROTSPEED, fine_sine(an))
    actor.angle = to_uint32(actor.angle + (ANG90 // 15))


@action("A_CHolyAttack")
def a_cholyattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CHolyAttack (p_pspr.c:1923).

    Wraithverge fires holy missile, costs 18 Blue and 18 Green Mana,
    and flashes palette to STARTHOLYPAL (22).
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    cost = WEAPON_MANA_USE[player.player_class][player.readyweapon]
    player.mana[MANA_1] -= cost
    player.mana[MANA_2] -= cost

    spawn_player_missile(sim, pmo, MobjType.MT_HOLY_MISSILE)

    player.damagecount = 0
    player.bonuscount = 0
    player.palette = STARTHOLYPAL

    _emit_sound(sim, pmo, SfxEnum.SFX_CHOLY_FIRE)


@action("A_CHolyPalette")
def a_cholypalette(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_CHolyPalette (p_pspr.c:1944).

    Fades palette flash across S_CHOLYATK_6..9 back to normal palette (0).
    """
    sim, player, psp = _resolve_player_and_psp(sim, player_or_mo, psp)
    cur_state = int(psp.state_num if psp is not None else StateNum.S_CHOLYATK_6)
    pal = STARTHOLYPAL + cur_state - int(StateNum.S_CHOLYATK_6)
    if pal == STARTHOLYPAL + 3:
        pal = 0
    player.palette = pal
