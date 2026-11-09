# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Fighter weapon action functions (Task T10.7).

Ports `A_FPunchAttack`, `A_FAxeAttack`, `A_FHammerAttack`, `A_FHammerThrow`,
and `A_FSwordAttack` from Chocolate Doom's `src/hexen/p_pspr.c`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import ANG45
from pyhexen.core.fixed import FRACUNIT, to_uint32
from pyhexen.sim.actions import action
from pyhexen.sim.actions.monsters_casters import thrust_mobj
from pyhexen.sim.attacks import (
    aim_line_attack,
    line_attack,
    spawn_player_missile,
    spm_angle_xyz,
)
from pyhexen.sim.enemy import MELEERANGE
from pyhexen.sim.info_data import MobjFlag, MobjType, SfxEnum, StateNum
from pyhexen.sim.mobj import Mobj
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

HAMMER_RANGE: int = MELEERANGE + (MELEERANGE // 2)
AXERANGE: int = (9 * MELEERANGE) // 4  # 2.25 * MELEERANGE = 144 * FRACUNIT


@action("A_FPunchAttack")
def a_fpunchattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_FPunchAttack (p_pspr.c:1308).

    Fighter's Spiked Gauntlets melee combo: every 3rd consecutive hit deals
    double damage with 6*FRACUNIT knockback and transitions to S_PUNCHATK2_1.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    damage = 40 + (sim.rng.p_random() & 15)
    power = 2 * FRACUNIT
    puff_type = MobjType.MT_PUNCHPUFF
    hit = False

    for i in range(16):
        angle = to_uint32(pmo.angle + i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, 2 * MELEERANGE)
        if sim.linetarget is not None:
            pmo.special1 = int(pmo.special1) + 1
            if pmo.special1 == 3:
                damage <<= 1
                power = 6 * FRACUNIT
                puff_type = MobjType.MT_HAMMERPUFF
            line_attack(sim, pmo, angle, 2 * MELEERANGE, slope, damage, puff_type=puff_type)
            target = sim.linetarget
            if target is not None and (
                (target.flags & MobjFlag.MF_COUNTKILL) or target.player is not None
            ):
                thrust_mobj(target, angle, power)
            adjust_player_angle(sim, pmo)
            hit = True
            break

        angle = to_uint32(pmo.angle - i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, 2 * MELEERANGE)
        if sim.linetarget is not None:
            pmo.special1 = int(pmo.special1) + 1
            if pmo.special1 == 3:
                damage <<= 1
                power = 6 * FRACUNIT
                puff_type = MobjType.MT_HAMMERPUFF
            line_attack(sim, pmo, angle, 2 * MELEERANGE, slope, damage, puff_type=puff_type)
            target = sim.linetarget
            if target is not None and (
                (target.flags & MobjFlag.MF_COUNTKILL) or target.player is not None
            ):
                thrust_mobj(target, angle, power)
            adjust_player_angle(sim, pmo)
            hit = True
            break

    if not hit:
        # Didn't find any creatures, so try to strike any walls
        pmo.special1 = 0
        angle = pmo.angle
        slope = aim_line_attack(sim, pmo, angle, MELEERANGE)
        line_attack(sim, pmo, angle, MELEERANGE, slope, damage, puff_type=puff_type)

    if pmo.special1 == 3:
        pmo.special1 = 0
        p_set_psprite(sim, player, ps_weapon, StateNum.S_PUNCHATK2_1)
        _emit_sound(sim, pmo, SfxEnum.SFX_FIGHTER_GRUNT)


@action("A_FAxeAttack")
def a_faxeattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_FAxeAttack (p_pspr.c:1386).

    Fighter's Timon's Axe: deals 2x damage with glowing puff and deducts Blue Mana
    on creature hit when Blue Mana > 0; transitions to non-glowing S_FAXEATK_5 when
    mana runs out.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    r = sim.rng.p_random()
    damage = 40 + (r & 15) + (sim.rng.p_random() & 7)
    power = 0
    if player.mana[MANA_1] > 0:
        damage <<= 1
        power = 6 * FRACUNIT
        puff_type = MobjType.MT_AXEPUFF_GLOW
        use_mana = 1
    else:
        puff_type = MobjType.MT_AXEPUFF
        use_mana = 0

    hit = False
    for i in range(16):
        angle = to_uint32(pmo.angle + i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, AXERANGE)
        if sim.linetarget is not None:
            line_attack(sim, pmo, angle, AXERANGE, slope, damage, puff_type=puff_type)
            target = sim.linetarget
            if target is not None and (
                (target.flags & MobjFlag.MF_COUNTKILL) or target.player is not None
            ):
                thrust_mobj(target, angle, power)
            adjust_player_angle(sim, pmo)
            use_mana += 1
            hit = True
            break

        angle = to_uint32(pmo.angle - i * (ANG45 // 16))
        slope = aim_line_attack(sim, pmo, angle, AXERANGE)
        if sim.linetarget is not None:
            line_attack(sim, pmo, angle, AXERANGE, slope, damage, puff_type=puff_type)
            target = sim.linetarget
            if target is not None and (target.flags & MobjFlag.MF_COUNTKILL):
                thrust_mobj(target, angle, power)
            adjust_player_angle(sim, pmo)
            use_mana += 1
            hit = True
            break

    if not hit:
        # Didn't find any creatures, so try to strike any walls
        pmo.special1 = 0
        angle = pmo.angle
        slope = aim_line_attack(sim, pmo, angle, MELEERANGE)
        line_attack(sim, pmo, angle, MELEERANGE, slope, damage, puff_type=puff_type)

    if use_mana == 2:
        pclass = player.player_class & 3
        wp = player.readyweapon & 3
        player.mana[MANA_1] -= WEAPON_MANA_USE[pclass][wp]
        if player.mana[MANA_1] <= 0:
            p_set_psprite(sim, player, ps_weapon, StateNum.S_FAXEATK_5)


@action("A_FHammerAttack")
def a_fhammerattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_FHammerAttack (p_pspr.c:724).

    Fighter's Hammer of Retribution melee check: if a target or wall is hit in
    HAMMER_RANGE, does not throw a hammer (`pmo.special1 = 0`); if nothing is hit
    in melee and Green Mana is sufficient, sets `pmo.special1 = 1` so `A_FHammerThrow`
    launches an exploding hammer missile.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    damage = 60 + (sim.rng.p_random() & 63)
    power = 10 * FRACUNIT
    puff_type = MobjType.MT_HAMMERPUFF
    hit = False

    for i in range(16):
        angle = to_uint32(pmo.angle + i * (ANG45 // 32))
        slope = aim_line_attack(sim, pmo, angle, HAMMER_RANGE)
        if sim.linetarget is not None:
            line_attack(sim, pmo, angle, HAMMER_RANGE, slope, damage, puff_type=puff_type)
            adjust_player_angle(sim, pmo)
            target = sim.linetarget
            if target is not None and (
                (target.flags & MobjFlag.MF_COUNTKILL) or target.player is not None
            ):
                thrust_mobj(target, angle, power)
            pmo.special1 = 0  # Don't throw a hammer
            hit = True
            break

        angle = to_uint32(pmo.angle - i * (ANG45 // 32))
        slope = aim_line_attack(sim, pmo, angle, HAMMER_RANGE)
        if sim.linetarget is not None:
            line_attack(sim, pmo, angle, HAMMER_RANGE, slope, damage, puff_type=puff_type)
            adjust_player_angle(sim, pmo)
            target = sim.linetarget
            if target is not None and (
                (target.flags & MobjFlag.MF_COUNTKILL) or target.player is not None
            ):
                thrust_mobj(target, angle, power)
            pmo.special1 = 0  # Don't throw a hammer
            hit = True
            break

    if not hit:
        # Didn't find any targets in meleerange, so set to throw out a hammer unless a wall was hit
        angle = pmo.angle
        slope = aim_line_attack(sim, pmo, angle, HAMMER_RANGE)
        puff = line_attack(sim, pmo, angle, HAMMER_RANGE, slope, damage, puff_type=puff_type)
        pmo.special1 = 0 if puff is not None else 1

    pclass = player.player_class & 3
    wp = player.readyweapon & 3
    if player.mana[MANA_2] < WEAPON_MANA_USE[pclass][wp]:
        pmo.special1 = 0


@action("A_FHammerThrow")
def a_fhammerthrow(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_FHammerThrow (p_pspr.c:793)."""
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None or not pmo.special1:
        return

    pclass = player.player_class & 3
    wp = player.readyweapon & 3
    player.mana[MANA_2] -= WEAPON_MANA_USE[pclass][wp]
    mo = spawn_player_missile(sim, pmo, MobjType.MT_HAMMER_MISSILE)
    if mo is not None:
        mo.special1 = 0


@action("A_FSwordAttack")
def a_fswordattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_FSwordAttack (p_pspr.c:815).

    Fighter's Quietus: deducts both Blue and Green Mana and fires a 5-projectile spread
    of `MT_FSWORD_MISSILE`.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    pclass = player.player_class & 3
    wp = player.readyweapon & 3
    cost = WEAPON_MANA_USE[pclass][wp]
    player.mana[MANA_1] -= cost
    player.mana[MANA_2] -= cost

    spm_angle_xyz(
        sim,
        pmo,
        pmo.x,
        pmo.y,
        pmo.z - 10 * FRACUNIT,
        MobjType.MT_FSWORD_MISSILE,
        to_uint32(pmo.angle + ANG45 // 4),
    )
    spm_angle_xyz(
        sim,
        pmo,
        pmo.x,
        pmo.y,
        pmo.z - 5 * FRACUNIT,
        MobjType.MT_FSWORD_MISSILE,
        to_uint32(pmo.angle + ANG45 // 8),
    )
    spm_angle_xyz(
        sim,
        pmo,
        pmo.x,
        pmo.y,
        pmo.z,
        MobjType.MT_FSWORD_MISSILE,
        pmo.angle,
    )
    spm_angle_xyz(
        sim,
        pmo,
        pmo.x,
        pmo.y,
        pmo.z + 5 * FRACUNIT,
        MobjType.MT_FSWORD_MISSILE,
        to_uint32(pmo.angle - ANG45 // 8),
    )
    spm_angle_xyz(
        sim,
        pmo,
        pmo.x,
        pmo.y,
        pmo.z + 10 * FRACUNIT,
        MobjType.MT_FSWORD_MISSILE,
        to_uint32(pmo.angle - ANG45 // 4),
    )
    _emit_sound(sim, pmo, SfxEnum.SFX_FIGHTER_SWORD_FIRE)
