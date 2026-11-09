# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Combat damage, armor absorption, pain response, and death handling.

Port of P_DamageMobj, P_KillMobj, P_FallingDamage, P_PoisonDamage,
P_PoisonPlayer, and ActiveMinotaur (p_inter.c).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import (
    ANG180,
    ANGLETOFINESHIFT,
    FINEMASK,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div, fixed_div, fixed_mul, to_uint32
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    StateNum,
)
from pyhexen.sim.mobj import Mobj, set_mobj_state
from pyhexen.sim.player import (
    CF_GODMODE,
    PW_FLIGHT,
    PW_INVULNERABILITY,
    Player,
    PlayerClass,
    PlayerState,
)
from pyhexen.sim.specials.dispatcher import execute_line_special

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

# Armor slot indices (p_inter.c, h2def.h)
ARMOR_ARMOR: int = 0  # Mesh Armor
ARMOR_SHIELD: int = 1  # Falcon Shield
ARMOR_HELMET: int = 2  # Platinum Helm
ARMOR_AMULET: int = 3  # Amulet of Warding
NUMARMOR: int = 4

# Combat and weapon constants (h2def.h, p_local.h)
BASETHRESHOLD: int = 100
FF_FULLBRIGHT: int = 0x8000
WP_FOURTH: int = 3

# Fixed-point armor durability increment points added upon pickup
# [Fighter, Cleric, Mage, Pig] x [Mesh, Shield, Helm, Amulet] (p_inter.c:27-32)
ARMOR_INCREMENT: tuple[tuple[int, int, int, int], ...] = (
    (25 * FRACUNIT, 20 * FRACUNIT, 15 * FRACUNIT, 5 * FRACUNIT),  # PCLASS_FIGHTER
    (10 * FRACUNIT, 25 * FRACUNIT, 5 * FRACUNIT, 20 * FRACUNIT),  # PCLASS_CLERIC
    (5 * FRACUNIT, 15 * FRACUNIT, 10 * FRACUNIT, 25 * FRACUNIT),  # PCLASS_MAGE
    (0, 0, 0, 0),  # PCLASS_PIG
)

# Innate base armor absorption percentage (fixed-point) (p_inter.c:34-35)
AUTO_ARMOR_SAVE: tuple[int, ...] = (
    15 * FRACUNIT,  # Fighter: 15% innate
    10 * FRACUNIT,  # Cleric:  10% innate
    5 * FRACUNIT,  # Mage:     5% innate
    0,  # Pig:      0% innate
)

# Maximum armor factor (p_inter.c)
ARMOR_MAX: tuple[int, ...] = (20, 18, 16, 1)

# Mapping of monster mobj types to frozen ice sculpture states (p_inter.c:1424-1466)
MONSTER_ICE_STATES: dict[int, StateNum] = {
    int(MobjType.MT_BISHOP): StateNum.S_BISHOP_ICE,
    int(MobjType.MT_CENTAUR): StateNum.S_CENTAUR_ICE,
    int(MobjType.MT_CENTAURLEADER): StateNum.S_CENTAUR_ICE,
    int(MobjType.MT_DEMON): StateNum.S_DEMON_ICE,
    int(MobjType.MT_DEMON2): StateNum.S_DEMON_ICE,
    int(MobjType.MT_SERPENT): StateNum.S_SERPENT_ICE,
    int(MobjType.MT_SERPENTLEADER): StateNum.S_SERPENT_ICE,
    int(MobjType.MT_WRAITH): StateNum.S_WRAITH_ICE,
    int(MobjType.MT_WRAITHB): StateNum.S_WRAITH_ICE,
    int(MobjType.MT_ETTIN): StateNum.S_ETTIN_ICE1,
    int(MobjType.MT_FIREDEMON): StateNum.S_FIRED_ICE1,
    int(MobjType.MT_FIGHTER_BOSS): StateNum.S_FIGHTER_ICE,
    int(MobjType.MT_CLERIC_BOSS): StateNum.S_CLERIC_ICE,
    int(MobjType.MT_MAGE_BOSS): StateNum.S_MAGE_ICE,
    int(MobjType.MT_PIG): StateNum.S_PIG_ICE,
}

# Mapping of player classes to frozen ice sculpture states (p_inter.c:1371-1387)
PLAYER_ICE_STATES: dict[int, StateNum] = {
    int(PlayerClass.PCLASS_FIGHTER): StateNum.S_FPLAY_ICE,
    int(PlayerClass.PCLASS_CLERIC): StateNum.S_CPLAY_ICE,
    int(PlayerClass.PCLASS_MAGE): StateNum.S_MPLAY_ICE,
    int(PlayerClass.PCLASS_PIG): StateNum.S_PIG_ICE,
}

# Mapping of player classes to flame death states and burn scream sounds (p_inter.c:1349-1365)
PLAYER_BURN_INFO: dict[int, tuple[StateNum, SfxEnum]] = {
    int(PlayerClass.PCLASS_FIGHTER): (
        StateNum.S_PLAY_F_FDTH1,
        SfxEnum.SFX_PLAYER_FIGHTER_BURN_DEATH,
    ),
    int(PlayerClass.PCLASS_CLERIC): (
        StateNum.S_PLAY_C_FDTH1,
        SfxEnum.SFX_PLAYER_CLERIC_BURN_DEATH,
    ),
    int(PlayerClass.PCLASS_MAGE): (
        StateNum.S_PLAY_M_FDTH1,
        SfxEnum.SFX_PLAYER_MAGE_BURN_DEATH,
    ),
}

# Mapping of boss types to flame death states and burn scream sounds (p_inter.c:1395-1411)
BOSS_BURN_INFO: dict[int, tuple[StateNum, SfxEnum]] = {
    int(MobjType.MT_FIGHTER_BOSS): (
        StateNum.S_PLAY_F_FDTH1,
        SfxEnum.SFX_PLAYER_FIGHTER_BURN_DEATH,
    ),
    int(MobjType.MT_CLERIC_BOSS): (
        StateNum.S_PLAY_C_FDTH1,
        SfxEnum.SFX_PLAYER_CLERIC_BURN_DEATH,
    ),
    int(MobjType.MT_MAGE_BOSS): (
        StateNum.S_PLAY_M_FDTH1,
        SfxEnum.SFX_PLAYER_MAGE_BURN_DEATH,
    ),
}


@dataclass(slots=True, frozen=True)
class DamageEvent:
    """Emitted when an actor takes damage or dies."""

    target: Mobj
    inflictor: Mobj | None = None
    source: Mobj | None = None
    damage: int = 0
    damage_type: str = "normal"  # "normal", "fire", "ice"


def _resolve_mobj(obj: Any) -> Mobj | None:
    """Resolve an actor instance to an Mobj."""
    if obj is None:
        return None
    if isinstance(obj, Mobj):
        return obj
    if hasattr(obj, "mo") and isinstance(obj.mo, Mobj):
        return obj.mo
    return None


def auto_use_health(sim: GameSim, player: Player, save_health: int) -> None:
    """Port of P_AutoUseHealth (p_inter.c)."""
    from pyhexen.sim.artifacts import auto_use_health as _auto_use_health

    _auto_use_health(sim, player, save_health)


def morph_player(sim: GameSim, player: Player) -> None:
    """Port of P_MorphPlayer (p_inter.c)."""
    from pyhexen.sim.artifacts import morph_player as _morph_player

    _morph_player(sim, player)


def morph_monster(sim: GameSim, monster: Mobj) -> None:
    """Port of P_MorphMonster (p_inter.c)."""
    from pyhexen.sim.artifacts import morph_monster as _morph_monster

    _morph_monster(sim, monster)


def teleport_other(sim: GameSim, victim: Mobj) -> None:
    """Port of P_TeleportOther (p_inter.c)."""
    from pyhexen.sim.artifacts import teleport_other as _teleport_other

    _teleport_other(sim, victim)


def minotaur_slam(sim: GameSim, source: Mobj, target: Mobj) -> None:
    """Port of P_MinotaurSlam (p_inter.c:1513)."""
    angle = (
        (point_to_angle2(source.x, source.y, target.x, target.y) & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    ) & FINEMASK
    thrust = 16 * FRACUNIT + (sim.rng.p_random() << 10)
    target.momx += fixed_mul(thrust, fine_cosine(angle))
    target.momy += fixed_mul(thrust, fine_sine(angle))
    dmg = ((sim.rng.p_random() & 7) + 1) * 4
    damage_mobj(sim, target, None, source, dmg)
    if target.player is not None:
        target.reactiontime = 14 + (sim.rng.p_random() & 7)


def drop_weapon(sim: GameSim, player: Player) -> None:
    """Port of P_DropWeapon (p_pspr.c)."""
    from pyhexen.sim.psprites import p_drop_weapon

    p_drop_weapon(sim, player)


def active_minotaur(sim: GameSim, player: Player) -> bool:
    """Port of ActiveMinotaur (p_inter.c).

    Return True if the player has an active summoned minotaur.
    """
    for th in sim.thinkers:
        if isinstance(th, Mobj) and th.type == MobjType.MT_MINOTAUR and th.health > 0:
            m = getattr(th, "special1", None)
            master_mo = getattr(m, "m", m)
            if master_mo is player.mo:
                return True
    return False


def poison_player(player: Player, poisoner: Mobj | None, poison: int) -> None:
    """Port of P_PoisonPlayer (p_inter.c)."""
    if (player.cheats & CF_GODMODE) or (
        len(player.powers) > PW_INVULNERABILITY and player.powers[PW_INVULNERABILITY]
    ):
        return
    player.poisoncount += poison
    if hasattr(player, "poisoner"):
        player.poisoner = poisoner
    if player.poisoncount > 100:
        player.poisoncount = 100


def poison_damage(
    sim: GameSim,
    player: Player,
    source: Mobj | None,
    damage: int,
    play_pain_sound: bool,
) -> None:
    """Port of P_PoisonDamage (p_inter.c).

    Applies unarmored poison damage directly to the player.
    """
    target = player.mo
    if target is None or target.health <= 0:
        return
    if (target.flags2 & MobjFlag2.MF2_INVULNERABLE) and damage < 10000:
        return
    if sim.skill == 0:
        damage >>= 1
    if damage < 1000 and (
        (player.cheats & CF_GODMODE)
        or (len(player.powers) > PW_INVULNERABILITY and player.powers[PW_INVULNERABILITY])
    ):
        return

    if damage >= player.health and (sim.skill == 0 or sim.deathmatch) and not player.morph_tics:
        auto_use_health(sim, player, damage - player.health + 1)

    player.health -= damage
    if player.health < 0:
        player.health = 0
    player.attacker = source

    target.health -= damage
    if target.health <= 0:
        target.special1 = damage
        if source is not None and not player.morph_tics:
            if (
                (source.flags2 & MobjFlag2.MF2_FIREDAMAGE)
                and (target.health > -50)
                and (damage > 25)
            ):
                target.flags2 |= MobjFlag2.MF2_FIREDAMAGE
            if source.flags2 & MobjFlag2.MF2_ICEDAMAGE:
                target.flags2 |= MobjFlag2.MF2_ICEDAMAGE
        kill_mobj(sim, source, target)
        return

    if (sim.level_time & 63 == 0) and play_pain_sound:
        if target.info.painstate != StateNum.S_NULL:
            set_mobj_state(sim, target, target.info.painstate)


def falling_damage(sim: GameSim, player: Player) -> None:
    """Port of P_FallingDamage (p_inter.c).

    Computes falling impact damage from downward vertical momentum.
    """
    mo = player.mo
    if mo is None:
        return
    mom = abs(mo.momz)
    dist = fixed_mul(mom, (16 * FRACUNIT) // 23)

    if mom >= 63 * FRACUNIT:
        damage_mobj(sim, mo, None, None, 10000)
        return

    damage = ((fixed_mul(dist, dist) // 10) >> FRACBITS) - 24
    if mo.momz > -39 * FRACUNIT and damage > mo.health and mo.health != 1:
        damage = mo.health - 1

    if damage > 0:
        sim.events.emit(
            SoundEvent(
                sound_id=int(SfxEnum.SFX_PLAYER_LAND),
                origin_id=mo.tid,
                x=mo.x,
                y=mo.y,
                z=mo.z,
            )
        )
        damage_mobj(sim, mo, None, None, damage)


def damage_mobj(
    sim: GameSim,
    target: Mobj,
    inflictor: Mobj | None,
    source: Mobj | None,
    damage: int,
) -> None:
    """Port of P_DamageMobj (p_inter.c).

    Apply combat damage, calculate class-based armor absorption, evaluate
    momentum thrust physics, and trigger pain or death states.
    """
    resolved_target = _resolve_mobj(target)
    if resolved_target is None:
        return
    target = resolved_target

    inflictor = _resolve_mobj(inflictor)
    source = _resolve_mobj(source)

    if damage < 0:
        damage = 0

    if not (target.flags & MobjFlag.MF_SHOOTABLE):
        return

    if target.health <= 0:
        if inflictor is not None and (inflictor.flags2 & MobjFlag2.MF2_ICEDAMAGE):
            return
        elif target.flags & MobjFlag.MF_ICECORPSE:
            target.tics = 1
            target.momx = 0
            target.momy = 0
        return

    if (target.flags2 & MobjFlag2.MF2_INVULNERABLE) and damage < 10000:
        if target.player is not None:
            return
        if inflictor is not None:
            if inflictor.type not in (
                MobjType.MT_HOLY_FX,
                MobjType.MT_POISONCLOUD,
                MobjType.MT_FIREBOMB,
            ):
                return
        else:
            return

    if target.player is not None:
        if damage < 1000 and (
            (target.player.cheats & CF_GODMODE)
            or (
                len(target.player.powers) > PW_INVULNERABILITY
                and target.player.powers[PW_INVULNERABILITY]
            )
        ):
            return

    if target.flags & MobjFlag.MF_SKULLFLY:
        target.momx = 0
        target.momy = 0
        target.momz = 0

    if target.flags2 & MobjFlag2.MF2_DORMANT:
        return

    if damage <= 0:
        return

    player = target.player
    if player is not None and sim.skill == 0:
        damage >>= 1

    # Special inflictor types
    if inflictor is not None:
        itype = inflictor.type
        if itype == MobjType.MT_EGGFX:
            if player is not None:
                morph_player(sim, player)
            else:
                morph_monster(sim, target)
            return
        elif itype in (
            MobjType.MT_TELOTHER_FX1,
            MobjType.MT_TELOTHER_FX2,
            MobjType.MT_TELOTHER_FX3,
            MobjType.MT_TELOTHER_FX4,
            MobjType.MT_TELOTHER_FX5,
        ):
            if (
                (target.flags & MobjFlag.MF_COUNTKILL)
                and (target.type != MobjType.MT_SERPENT)
                and (target.type != MobjType.MT_SERPENTLEADER)
                and not (target.flags2 & MobjFlag2.MF2_BOSS)
            ):
                teleport_other(sim, target)
            return
        elif itype == MobjType.MT_MINOTAUR:
            if inflictor.flags & MobjFlag.MF_SKULLFLY:
                minotaur_slam(sim, inflictor, target)
                return
        elif itype == MobjType.MT_BISH_FX:
            damage >>= 1
        elif itype == MobjType.MT_SHARDFX1:
            s2 = getattr(inflictor, "special2", 0)
            if s2 == 3:
                damage <<= 3
            elif s2 == 2:
                damage <<= 2
            elif s2 == 1:
                damage <<= 1
        elif itype == MobjType.MT_CSTAFF_MISSILE:
            if player is not None:
                poison_player(player, source, 20)
                damage >>= 1
        elif itype == MobjType.MT_ICEGUY_FX2:
            damage >>= 1
        elif itype == MobjType.MT_POISONDART:
            if player is not None:
                poison_player(player, source, 20)
                damage >>= 1
        elif itype == MobjType.MT_POISONCLOUD:
            if player is not None:
                if player.poisoncount < 4:
                    poison_damage(
                        sim,
                        player,
                        source,
                        15 + (sim.rng.p_random() & 15),
                        False,
                    )
                    poison_player(player, source, 50)
                    sim.events.emit(
                        SoundEvent(
                            sound_id=int(SfxEnum.SFX_PLAYER_POISONCOUGH),
                            origin_id=target.tid,
                            x=target.x,
                            y=target.y,
                            z=target.z,
                        )
                    )
                return
            elif not (target.flags & MobjFlag.MF_COUNTKILL):
                return
        elif itype == MobjType.MT_FSWORD_MISSILE:
            if player is not None:
                damage -= damage >> 2

    # Momentum thrust physics
    if (
        inflictor is not None
        and (source is None or getattr(source, "player", None) is None)
        and not (inflictor.flags2 & MobjFlag2.MF2_NODMGTHRUST)
    ):
        mass = target.info.mass
        if mass <= 0:
            mass = 100
        thrust = c_div(damage * (FRACUNIT >> 3) * 150, mass)
        ang = point_to_angle2(inflictor.x, inflictor.y, target.x, target.y)
        if (
            damage < 40
            and damage > target.health
            and (target.z - inflictor.z > 64 * FRACUNIT)
            and (sim.rng.p_random() & 1)
        ):
            ang = to_uint32(ang + ANG180)
            thrust *= 4
        fine = ((ang & 0xFFFFFFFF) >> ANGLETOFINESHIFT) & FINEMASK
        target.momx += fixed_mul(thrust, fine_cosine(fine))
        target.momy += fixed_mul(thrust, fine_sine(fine))

    # Player class armor absorption
    if player is not None:
        pclass = player.player_class
        pclass_idx = pclass if 0 <= pclass < len(AUTO_ARMOR_SAVE) else 0
        saved_percent = (
            AUTO_ARMOR_SAVE[pclass_idx]
            + player.armor_points[ARMOR_ARMOR]
            + player.armor_points[ARMOR_SHIELD]
            + player.armor_points[ARMOR_HELMET]
            + player.armor_points[ARMOR_AMULET]
        )
        if saved_percent > 0:
            if saved_percent > 100 * FRACUNIT:
                saved_percent = 100 * FRACUNIT
            for i in range(NUMARMOR):
                if player.armor_points[i] > 0:
                    deduction = fixed_div(
                        fixed_mul(
                            damage << FRACBITS,
                            ARMOR_INCREMENT[pclass_idx][i],
                        ),
                        300 * FRACUNIT,
                    )
                    player.armor_points[i] -= deduction
                    if player.armor_points[i] < 2 * FRACUNIT:
                        player.armor_points[i] = 0
            saved = fixed_div(
                fixed_mul(damage << FRACBITS, saved_percent),
                100 * FRACUNIT,
            )
            if saved > saved_percent * 2:
                saved = saved_percent * 2
            damage -= saved >> FRACBITS

        if damage >= player.health and (sim.skill == 0 or sim.deathmatch) and not player.morph_tics:
            auto_use_health(sim, player, damage - player.health + 1)

        player.health -= damage
        if player.health < 0:
            player.health = 0
        player.attacker = source
        player.damagecount += damage
        if player.damagecount > 100:
            player.damagecount = 100

    # Apply health reduction
    target.health -= damage
    damage_type = "normal"
    if target.health <= 0:
        if inflictor is not None:
            if inflictor.flags2 & MobjFlag2.MF2_FIREDAMAGE:
                if player is not None and not player.morph_tics:
                    if target.health > -50 and damage > 25:
                        target.flags2 |= MobjFlag2.MF2_FIREDAMAGE
                        damage_type = "fire"
                else:
                    target.flags2 |= MobjFlag2.MF2_FIREDAMAGE
                    damage_type = "fire"
            elif inflictor.flags2 & MobjFlag2.MF2_ICEDAMAGE:
                target.flags2 |= MobjFlag2.MF2_ICEDAMAGE
                damage_type = "ice"

        if source is not None and source.type == MobjType.MT_MINOTAUR:
            master = getattr(source, "special1", None)
            master_mo = getattr(master, "m", master)
            if (
                isinstance(master_mo, Mobj)
                and master_mo.player is not None
                and master_mo.player.mo is master_mo
            ):
                source = master_mo

        if (
            source is not None
            and getattr(source, "player", None) is not None
            and getattr(source.player, "readyweapon", -1) == WP_FOURTH
        ):
            target.health = -5000

        sim.events.emit(
            DamageEvent(
                target=target,
                inflictor=inflictor,
                source=source,
                damage=damage,
                damage_type=damage_type,
            )
        )
        kill_mobj(sim, source, target)
        return

    # Pain chance and pain state transitions
    if sim.rng.p_random() < target.info.painchance and not (target.flags & MobjFlag.MF_SKULLFLY):
        if inflictor is not None and (
            MobjType.MT_LIGHTNING_FLOOR <= inflictor.type <= MobjType.MT_LIGHTNING_ZAP
        ):
            if sim.rng.p_random() < 96:
                target.flags |= MobjFlag.MF_JUSTHIT
                if target.info.painstate != StateNum.S_NULL:
                    set_mobj_state(sim, target, target.info.painstate)
            else:
                target.frame |= FF_FULLBRIGHT
                if (
                    (target.flags & MobjFlag.MF_COUNTKILL)
                    and (sim.rng.p_random() < 128)
                    and target.type
                    in (
                        MobjType.MT_CENTAUR,
                        MobjType.MT_CENTAURLEADER,
                        MobjType.MT_ETTIN,
                    )
                ):
                    sim.events.emit(
                        SoundEvent(
                            sound_id=int(SfxEnum.SFX_PUPPYBEAT),
                            origin_id=target.tid,
                            x=target.x,
                            y=target.y,
                            z=target.z,
                        )
                    )
        else:
            target.flags |= MobjFlag.MF_JUSTHIT
            if target.info.painstate != StateNum.S_NULL:
                set_mobj_state(sim, target, target.info.painstate)
            if (
                inflictor is not None
                and inflictor.type == MobjType.MT_POISONCLOUD
                and (target.flags & MobjFlag.MF_COUNTKILL)
                and (sim.rng.p_random() < 128)
                and target.type
                in (
                    MobjType.MT_CENTAUR,
                    MobjType.MT_CENTAURLEADER,
                    MobjType.MT_ETTIN,
                )
            ):
                sim.events.emit(
                    SoundEvent(
                        sound_id=int(SfxEnum.SFX_PUPPYBEAT),
                        origin_id=target.tid,
                        x=target.x,
                        y=target.y,
                        z=target.z,
                    )
                )

    target.reactiontime = 0
    if (
        not target.threshold
        and source is not None
        and not (source.flags2 & MobjFlag2.MF2_BOSS)
        and target.type != MobjType.MT_BISHOP
        and target.type != MobjType.MT_MINOTAUR
    ):
        if (target.type == MobjType.MT_CENTAUR and source.type == MobjType.MT_CENTAURLEADER) or (
            target.type == MobjType.MT_CENTAURLEADER and source.type == MobjType.MT_CENTAUR
        ):
            sim.events.emit(
                DamageEvent(
                    target=target,
                    inflictor=inflictor,
                    source=source,
                    damage=damage,
                    damage_type=damage_type,
                )
            )
            return

        target.target = source
        target.threshold = BASETHRESHOLD
        if target.state_num == target.info.spawnstate and target.info.seestate != StateNum.S_NULL:
            set_mobj_state(sim, target, target.info.seestate)

    sim.events.emit(
        DamageEvent(
            target=target,
            inflictor=inflictor,
            source=source,
            damage=damage,
            damage_type=damage_type,
        )
    )


def kill_mobj(sim: GameSim, source: Mobj | None, target: Mobj) -> None:
    """Port of P_KillMobj (p_inter.c).

    Kills the target actor, updates corpse flags, executes monster specials,
    and transitions actor into appropriate death sequence.
    """
    resolved_target = _resolve_mobj(target)
    if resolved_target is None:
        return
    target = resolved_target
    source = _resolve_mobj(source)

    target.flags &= ~(
        MobjFlag.MF_SHOOTABLE | MobjFlag.MF_FLOAT | MobjFlag.MF_SKULLFLY | MobjFlag.MF_NOGRAVITY
    )
    target.flags |= MobjFlag.MF_CORPSE | MobjFlag.MF_DROPOFF
    target.flags2 &= ~MobjFlag2.MF2_PASSMOBJ
    target.height >>= 2
    target.target = None

    # Initiate monster death actions / specials
    if (
        (target.flags & MobjFlag.MF_COUNTKILL) or (target.type == MobjType.MT_ZBELL)
    ) and target.special:
        if target.type == MobjType.MT_SORCBOSS:
            if sim.acs_vm is not None:
                sim.acs_vm.start_script(
                    target.special,
                    args=(0, 0, 0),
                    activator=target,
                    line=None,
                    side=0,
                    sim=sim,
                )
        else:
            execute_line_special(
                sim,
                target.special,
                target.args,
                line=None,
                mo=target,
            )

    # Check for frag changes in deathmatch/netgame
    if source is not None and source.player is not None and target.player is not None:
        if hasattr(target.player, "frags") and isinstance(target.player.frags, list):
            try:
                target_idx = sim.players.index(target.player)
                if target is source:
                    target.player.frags[target_idx] -= 1
                else:
                    source.player.frags[target_idx] += 1
            except (ValueError, IndexError):
                pass
    elif target.player is not None and source is None:
        if hasattr(target.player, "frags") and isinstance(target.player.frags, list):
            try:
                target_idx = sim.players.index(target.player)
                target.player.frags[target_idx] -= 1
            except (ValueError, IndexError):
                pass

    # Player death processing
    player = target.player
    if player is not None:
        target.flags &= ~MobjFlag.MF_SOLID
        target.flags2 &= ~MobjFlag2.MF2_FLY
        if len(player.powers) > PW_FLIGHT:
            player.powers[PW_FLIGHT] = 0
        player.player_state = PlayerState.PST_DEAD
        drop_weapon(sim, player)

        # Player flame death
        if target.flags2 & MobjFlag2.MF2_FIREDAMAGE:
            burn_info = PLAYER_BURN_INFO.get(int(player.player_class))
            if burn_info is not None:
                st, sfx = burn_info
                sim.events.emit(
                    SoundEvent(
                        sound_id=int(sfx),
                        origin_id=target.tid,
                        x=target.x,
                        y=target.y,
                        z=target.z,
                    )
                )
                set_mobj_state(sim, target, st)
                return

        # Player ice death
        if target.flags2 & MobjFlag2.MF2_ICEDAMAGE:
            target.flags &= ~(7 << 26)  # MF_TRANSLATION
            target.flags |= MobjFlag.MF_ICECORPSE
            ice_st = PLAYER_ICE_STATES.get(int(player.player_class))
            if ice_st is not None:
                set_mobj_state(sim, target, ice_st)
                return

    # Monster / Boss flame death
    if target.flags2 & MobjFlag2.MF2_FIREDAMAGE:
        boss_burn = BOSS_BURN_INFO.get(int(target.type))
        if boss_burn is not None:
            st, sfx = boss_burn
            sim.events.emit(
                SoundEvent(
                    sound_id=int(sfx),
                    origin_id=target.tid,
                    x=target.x,
                    y=target.y,
                    z=target.z,
                )
            )
            set_mobj_state(sim, target, st)
            return
        elif target.type == MobjType.MT_TREEDESTRUCTIBLE:
            set_mobj_state(sim, target, StateNum.S_ZTREEDES_X1)
            target.height = 24 * FRACUNIT
            sim.events.emit(
                SoundEvent(
                    sound_id=int(SfxEnum.SFX_TREE_EXPLODE),
                    origin_id=target.tid,
                    x=target.x,
                    y=target.y,
                    z=target.z,
                )
            )
            return

    # Monster ice death
    if target.flags2 & MobjFlag2.MF2_ICEDAMAGE:
        target.flags |= MobjFlag.MF_ICECORPSE
        m_ice_st = MONSTER_ICE_STATES.get(int(target.type))
        if m_ice_st is not None:
            set_mobj_state(sim, target, m_ice_st)
            return
        else:
            target.flags &= ~MobjFlag.MF_ICECORPSE

    # Minotaur master handling
    if target.type == MobjType.MT_MINOTAUR:
        master = getattr(target, "special1", None)
        if master is not None:
            master_mo = getattr(master, "m", master)
            if (
                isinstance(master_mo, Mobj)
                and master_mo.health > 0
                and master_mo.player is not None
            ):
                if not active_minotaur(sim, master_mo.player):
                    if len(master_mo.player.powers) > 8:
                        master_mo.player.powers[8] = 0  # pw_minotaur
    elif target.type == MobjType.MT_TREEDESTRUCTIBLE:
        target.height = 24 * FRACUNIT

    # Extreme vs Normal Death
    if (
        target.health < -(target.info.spawnhealth >> 1)
        and target.info.xdeathstate != StateNum.S_NULL
    ):
        set_mobj_state(sim, target, target.info.xdeathstate)
    else:
        if (
            target.type == MobjType.MT_FIREDEMON
            and target.z <= target.floorz + 2 * FRACUNIT
            and target.info.xdeathstate != StateNum.S_NULL
        ):
            set_mobj_state(sim, target, target.info.xdeathstate)
        else:
            set_mobj_state(sim, target, target.info.deathstate)

    if not target.removed:
        target.tics -= sim.rng.p_random() & 3


# C-style function name aliases for cross-referencing Chocolate Doom reference
P_DamageMobj = damage_mobj
P_KillMobj = kill_mobj
P_FallingDamage = falling_damage
p_falling_damage = falling_damage
P_PoisonDamage = poison_damage
p_poison_damage = poison_damage
P_PoisonPlayer = poison_player
p_poison_player = poison_player
