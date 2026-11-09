# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Mage weapon action functions (Task T10.9).

Ports `A_MWandAttack`, `A_FireConePL1`, `A_ShedShard`, `A_LightningReady`,
`A_LightningClip`, `A_LightningZap`, `A_MLightningAttack`, `A_MLightningAttack2`,
`A_ZapMimic`, `A_LastZap`, `A_LightningRemove`, `A_MStaffAttack`, and `A_MStaffPalette`
from Chocolate Doom's `src/hexen/p_pspr.c`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG1,
    ANG45,
    ANG90,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACBITS, FRACUNIT, to_uint32
from pyhexen.sim.actions import action
from pyhexen.sim.actions.monsters_bosses import rough_monster_search
from pyhexen.sim.attacks import (
    aim_line_attack,
    spawn_missile_angle_speed,
    spawn_player_missile,
    spm_angle,
)
from pyhexen.sim.enemy import MELEERANGE
from pyhexen.sim.info_data import MobjFlag2, MobjType, SfxEnum, StateNum
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.mobj import Mobj, set_mobj_state, spawn_mobj
from pyhexen.sim.movement import explode_missile, p_blaster_mobj_thinker, thrust_mobj
from pyhexen.sim.psprites import (
    MANA_1,
    MANA_2,
    WEAPON_MANA_USE,
    PspDef,
    _emit_sound,
    _resolve_player_and_psp,
    a_weaponready,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.player import Player

ZAGSPEED: int = FRACUNIT
STARTSCOURGEPAL: int = 25

SHARDSPAWN_LEFT: int = 1
SHARDSPAWN_RIGHT: int = 2
SHARDSPAWN_UP: int = 4
SHARDSPAWN_DOWN: int = 8


@action("A_MWandAttack")
def a_mwandattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_MWandAttack (p_pspr.c:880).

    Sapphire Wand primary attack (fires high-speed MT_MWAND_MISSILE).
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    mo = spawn_player_missile(sim, pmo, MobjType.MT_MWAND_MISSILE)
    if mo is not None:
        mo.thinker_func = p_blaster_mobj_thinker

    _emit_sound(sim, pmo, SfxEnum.SFX_MAGE_WAND_FIRE)


@action("A_LightningReady")
def a_lightningready(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_LightningReady (p_pspr.c:900).

    Lightning weapon idle sound and ready state.
    """
    sim, player, psp = _resolve_player_and_psp(sim, player_or_mo, psp)
    a_weaponready(sim, player, psp)
    if sim.rng.p_random() < 160 and player.mo is not None:
        _emit_sound(sim, player.mo, SfxEnum.SFX_MAGE_LIGHTNING_READY)


@action("A_LightningClip")
def a_lightningclip(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_LightningClip (p_pspr.c:917).

    Clips floor/ceiling lightning heads and applies horizontal zig-zag and target tracking.
    """
    target: Mobj | None = None
    if actor.type == MobjType.MT_LIGHTNING_FLOOR:
        actor.z = actor.floorz
        c_mo = actor.special2 if isinstance(actor.special2, Mobj) else None
        target = c_mo.special1 if c_mo is not None and isinstance(c_mo.special1, Mobj) else None
    elif actor.type == MobjType.MT_LIGHTNING_CEILING:
        actor.z = actor.ceilingz - actor.height
        target = actor.special1 if isinstance(actor.special1, Mobj) else None

    if actor.type == MobjType.MT_LIGHTNING_FLOOR:
        c_mo = actor.special2 if isinstance(actor.special2, Mobj) else None
        zig_zag = sim.rng.p_random()
        sp1 = actor.special1 if isinstance(actor.special1, int) else 0

        if (zig_zag > 128 and sp1 < 2) or sp1 < -2:
            thrust_mobj(actor, to_uint32(actor.angle + ANG90), ZAGSPEED)
            if c_mo is not None:
                thrust_mobj(c_mo, to_uint32(actor.angle + ANG90), ZAGSPEED)
            actor.special1 = sp1 + 1
        else:
            thrust_mobj(actor, to_uint32(actor.angle - ANG90), ZAGSPEED)
            if c_mo is not None:
                thrust_mobj(c_mo, to_uint32(c_mo.angle - ANG90), ZAGSPEED)
            actor.special1 = sp1 - 1

    if target is not None:
        if target.health <= 0:
            explode_missile(sim, actor)
        else:
            actor.angle = point_to_angle2(actor.x, actor.y, target.x, target.y)
            actor.momx = 0
            actor.momy = 0
            thrust_mobj(actor, actor.angle, actor.info.speed >> 1)


@action("A_LightningZap")
def a_lightningzap(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_LightningZap (p_pspr.c:979).

    Deducts lightning health, spawns MT_LIGHTNING_ZAP discharges, and plays sound.
    """
    a_lightningclip(sim, actor, psp)

    actor.health -= 8
    if actor.health <= 0:
        set_mobj_state(sim, actor, actor.info.deathstate)
        return

    delta_z = 10 * FRACUNIT if actor.type == MobjType.MT_LIGHTNING_FLOOR else -10 * FRACUNIT
    r1 = sim.rng.p_random()
    r2 = sim.rng.p_random()

    spawn_x = actor.x + (((r2 - 128) * actor.radius) >> 8)
    spawn_y = actor.y + (((r1 - 128) * actor.radius) >> 8)
    spawn_z = actor.z + delta_z

    mo = spawn_mobj(sim, spawn_x, spawn_y, spawn_z, MobjType.MT_LIGHTNING_ZAP)
    if mo is not None:
        mo.special2 = actor
        mo.momx = actor.momx
        mo.momy = actor.momy
        mo.target = actor.target
        mo.momz = 20 * FRACUNIT if actor.type == MobjType.MT_LIGHTNING_FLOOR else -20 * FRACUNIT

    if actor.type == MobjType.MT_LIGHTNING_FLOOR and sim.rng.p_random() < 160:
        _emit_sound(sim, actor, SfxEnum.SFX_MAGE_LIGHTNING_CONTINUOUS)


@action("A_MLightningAttack2")
def a_mlightningattack2(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_MLightningAttack2 (p_pspr.c:1053).

    Spawns linked floor and ceiling lightning heads.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    fmo = spawn_player_missile(sim, pmo, MobjType.MT_LIGHTNING_FLOOR)
    cmo = spawn_player_missile(sim, pmo, MobjType.MT_LIGHTNING_CEILING)

    if fmo is not None:
        fmo.special1 = None
        fmo.special2 = cmo
        a_lightningzap(sim, fmo, psp)

    if cmo is not None:
        cmo.special1 = None  # Target mobj that it will track
        cmo.special2 = fmo
        a_lightningzap(sim, cmo, psp)

    _emit_sound(sim, pmo, SfxEnum.SFX_MAGE_LIGHTNING_FIRE)


@action("A_MLightningAttack")
def a_mlightningattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_MLightningAttack (p_pspr.c:1080).

    Fires Arc of Death lightning weapon, deducting Green Mana.
    """
    sim, player, psp = _resolve_player_and_psp(sim, player_or_mo, psp)
    a_mlightningattack2(sim, player, psp)
    cost = WEAPON_MANA_USE[player.player_class][player.readyweapon]
    player.mana[MANA_2] -= cost


@action("A_ZapMimic")
def a_zapmimic(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_ZapMimic (p_pspr.c:1092).

    Syncs zap discharge momentum with its parent lightning head.
    """
    parent = actor.special2 if isinstance(actor.special2, Mobj) else None
    if parent is not None:
        if parent.state_num >= int(parent.info.deathstate) or parent.state_num == int(
            StateNum.S_FREETARGMOBJ
        ):
            explode_missile(sim, actor)
        else:
            actor.momx = parent.momx
            actor.momy = parent.momy


@action("A_LastZap")
def a_lastzap(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_LastZap (p_pspr.c:1118).

    Spawns explosive terminal zap upward.
    """
    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_LIGHTNING_ZAP)
    if mo is not None:
        set_mobj_state(sim, mo, StateNum.S_LIGHTNING_ZAP_X1)
        mo.momz = 40 * FRACUNIT


@action("A_LightningRemove")
def a_lightningremove(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_LightningRemove (p_pspr.c:1136).

    Breaks the floor/ceiling lightning link and explodes the counterpart.
    """
    counterpart = actor.special2 if isinstance(actor.special2, Mobj) else None
    if counterpart is not None:
        counterpart.special2 = None
        explode_missile(sim, counterpart)


def _mstaff_spawn(sim: GameSim, pmo: Mobj, angle: int) -> None:
    """Helper for Bloodscourge projectile spawning."""
    mo = spm_angle(sim, pmo, MobjType.MT_MSTAFF_FX2, to_uint32(angle))
    if mo is not None:
        mo.target = pmo
        mo.special1 = rough_monster_search(sim, mo, 10)


@action("A_MStaffAttack")
def a_mstaffattack(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_MStaffAttack (p_pspr.c:1172).

    Bloodscourge fires 3 tracking projectiles, costs 15 Blue and 15 Green Mana,
    and flashes palette to STARTSCOURGEPAL (25).
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    cost = WEAPON_MANA_USE[player.player_class][player.readyweapon]
    player.mana[MANA_1] -= cost
    player.mana[MANA_2] -= cost

    angle = pmo.angle
    _mstaff_spawn(sim, pmo, angle)
    _mstaff_spawn(sim, pmo, angle - ANG1 * 5)
    _mstaff_spawn(sim, pmo, angle + ANG1 * 5)

    _emit_sound(sim, pmo, SfxEnum.SFX_MAGE_STAFF_FIRE)

    player.damagecount = 0
    player.bonuscount = 0
    player.palette = STARTSCOURGEPAL


@action("A_MStaffPalette")
def a_mstaffpalette(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_MStaffPalette (p_pspr.c:1202).

    Fades Bloodscourge palette flash across S_MSTAFFATK_2..5 back to normal (0).
    """
    sim, player, psp = _resolve_player_and_psp(sim, player_or_mo, psp)
    cur_state = int(psp.state_num if psp is not None else StateNum.S_MSTAFFATK_2)
    pal = STARTSCOURGEPAL + cur_state - int(StateNum.S_MSTAFFATK_2)
    if pal == STARTSCOURGEPAL + 3:
        pal = 0
    player.palette = pal


@action("A_FireConePL1")
def a_fireconepl1(
    sim: GameSim,
    player_or_mo: Player | Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_FireConePL1 (p_pspr.c:2258).

    Frost Shards melee ice damage check (range MELEERANGE); if no target in melee,
    fires MT_SHARDFX1 spreading projectile.
    """
    sim, player, _ = _resolve_player_and_psp(sim, player_or_mo, psp)
    pmo = player.mo
    if pmo is None:
        return

    cost = WEAPON_MANA_USE[player.player_class][player.readyweapon]
    player.mana[MANA_1] -= cost
    _emit_sound(sim, pmo, SfxEnum.SFX_MAGE_SHARDS_FIRE)

    damage = 90 + (sim.rng.p_random() & 15)
    conedone = False

    for i in range(16):
        angle = to_uint32(pmo.angle + i * (ANG45 // 16))
        aim_line_attack(sim, pmo, angle, MELEERANGE)
        linetarget = sim.linetarget
        if linetarget is not None:
            pmo.flags2 |= MobjFlag2.MF2_ICEDAMAGE
            damage_mobj(sim, linetarget, pmo, pmo, damage)
            pmo.flags2 &= ~MobjFlag2.MF2_ICEDAMAGE
            conedone = True
            break

    if not conedone:
        mo = spawn_player_missile(sim, pmo, MobjType.MT_SHARDFX1)
        if mo is not None:
            mo.special1 = SHARDSPAWN_LEFT | SHARDSPAWN_DOWN | SHARDSPAWN_UP | SHARDSPAWN_RIGHT
            mo.special2 = 3
            mo.target = pmo
            mo.args = (3, mo.args[1], mo.args[2], mo.args[3], mo.args[4])


@action("A_ShedShard")
def a_shedshard(
    sim: GameSim,
    actor: Mobj,
    psp: PspDef | None = None,
) -> None:
    """Port of A_ShedShard (p_pspr.c:2300).

    Spawns branching daughter shards in reproduction stages.
    """
    spawndir = actor.special1 if isinstance(actor.special1, int) else 0
    spermcount = actor.special2 if isinstance(actor.special2, int) else 0

    if spermcount <= 0:
        return

    actor.special2 = 0
    spermcount -= 1

    # Spawn new missile in set directions
    if spawndir & SHARDSPAWN_LEFT:
        mo = spawn_missile_angle_speed(
            sim,
            actor,
            MobjType.MT_SHARDFX1,
            to_uint32(actor.angle + (ANG45 // 9)),
            0,
            (20 + 2 * spermcount) << FRACBITS,
        )
        if mo is not None:
            mo.special1 = SHARDSPAWN_LEFT
            mo.special2 = spermcount
            mo.momz = actor.momz
            mo.target = actor.target
            arg0 = 2 if spermcount == 3 else 0
            mo.args = (arg0, mo.args[1], mo.args[2], mo.args[3], mo.args[4])

    if spawndir & SHARDSPAWN_RIGHT:
        mo = spawn_missile_angle_speed(
            sim,
            actor,
            MobjType.MT_SHARDFX1,
            to_uint32(actor.angle - (ANG45 // 9)),
            0,
            (20 + 2 * spermcount) << FRACBITS,
        )
        if mo is not None:
            mo.special1 = SHARDSPAWN_RIGHT
            mo.special2 = spermcount
            mo.momz = actor.momz
            mo.target = actor.target
            arg0 = 2 if spermcount == 3 else 0
            mo.args = (arg0, mo.args[1], mo.args[2], mo.args[3], mo.args[4])

    if spawndir & SHARDSPAWN_UP:
        mo = spawn_missile_angle_speed(
            sim,
            actor,
            MobjType.MT_SHARDFX1,
            actor.angle,
            0,
            (15 + 2 * spermcount) << FRACBITS,
        )
        if mo is not None:
            mo.momz = actor.momz
            mo.z += 8 * FRACUNIT
            if spermcount & 1:
                mo.special1 = SHARDSPAWN_UP | SHARDSPAWN_LEFT | SHARDSPAWN_RIGHT
            else:
                mo.special1 = SHARDSPAWN_UP
            mo.special2 = spermcount
            mo.target = actor.target
            arg0 = 2 if spermcount == 3 else 0
            mo.args = (arg0, mo.args[1], mo.args[2], mo.args[3], mo.args[4])

    if spawndir & SHARDSPAWN_DOWN:
        mo = spawn_missile_angle_speed(
            sim,
            actor,
            MobjType.MT_SHARDFX1,
            actor.angle,
            0,
            (15 + 2 * spermcount) << FRACBITS,
        )
        if mo is not None:
            mo.momz = actor.momz
            mo.z -= 4 * FRACUNIT
            if spermcount & 1:
                mo.special1 = SHARDSPAWN_DOWN | SHARDSPAWN_LEFT | SHARDSPAWN_RIGHT
            else:
                mo.special1 = SHARDSPAWN_DOWN
            mo.special2 = spermcount
            mo.target = actor.target
            arg0 = 2 if spermcount == 3 else 0
            mo.args = (arg0, mo.args[1], mo.args[2], mo.args[3], mo.args[4])
