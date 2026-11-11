# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Items and inventory system: pickups, artifacts, keys, puzzle items, and artifact actions.

Port of P_TouchSpecialThing, P_GiveArtifact, P_GiveBody, P_GiveArmor, P_GiveKey,
P_GivePower, P_UseArtifact, P_PlayerUseArtifact, P_PlayerNextArtifact,
P_PlayerRemoveArtifact, P_AutoUseHealth, P_MorphPlayer, P_UndoPlayerMorph,
P_MorphMonster, P_BlastRadius, P_BlastMobj, P_HealRadius, P_ArtiTele,
P_ArtiTeleportOther, P_TeleportOther, and P_UsePuzzleItem
(p_inter.c, p_user.c, p_map.c, a_action.c).
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import (
    ANG45,
    ANGLETOFINESHIFT,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACBITS, FRACUNIT, abs32, c_div, fixed_mul, to_uint32
from pyhexen.sim.actions import action
from pyhexen.sim.attacks import (
    check_missile_spawn,
    spawn_player_missile,
    spm_angle,
)
from pyhexen.sim.enemy import a_explode, noise_alert
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    SpriteNum,
    StateNum,
)
from pyhexen.sim.inter import (
    ARMOR_AMULET,
    ARMOR_ARMOR,
    ARMOR_HELMET,
    ARMOR_INCREMENT,
    ARMOR_MAX,
    ARMOR_SHIELD,
    AUTO_ARMOR_SAVE,
    NUMARMOR,
)
from pyhexen.sim.maputl import (
    PT_ADDLINES,
    PT_ADDTHINGS,
    line_opening,
    path_traverse,
    point_on_line_side,
)
from pyhexen.sim.message import set_message, set_yellow_message
from pyhexen.sim.mobj import (
    FLOAT_BOB_OFFSETS,
    Mobj,
    remove_mobj,
    set_mobj_state,
    spawn_mobj,
)
from pyhexen.sim.movement import thrust_mobj
from pyhexen.sim.psprites import (
    MANA_1,
    MANA_2,
    WP_SECOND,
    WP_THIRD,
    give_mana,
    give_weapon,
    give_weapon_piece,
    p_activate_morph_weapon,
    p_post_morph_weapon,
    p_set_psprite_nf,
)
from pyhexen.sim.sound import start_sound
from pyhexen.sim.specials.dispatcher import execute_line_special

if TYPE_CHECKING:
    from pyhexen.core.ticcmd import Intercept
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.player import Player


# =============================================================================
# Constants & Enums
# =============================================================================

BONUSADD: int = 6
TELEFOGHEIGHT: int = 32 * FRACUNIT
MAXHEALTH: int = 100
MAXMORPHHEALTH: int = 30
MAX_MANA: int = 200

INVULNTICS: int = 30 * 35
INFRATICS: int = 120 * 35
FLIGHTTICS: int = 60 * 35
SPEEDTICS: int = 45 * 35
MORPHTICS: int = 40 * 35
MAULATORTICS: int = 25 * 35
BLINKTHRESHOLD: int = 4 * 35

BLAST_RADIUS_DIST: int = 255 * FRACUNIT
BLAST_SPEED: int = 20 * FRACUNIT
BLAST_FULLSTRENGTH: int = 255
HEAL_RADIUS_DIST: int = 255 * FRACUNIT
USERANGE: int = 64 * FRACUNIT
USE_PUZZLE_ITEM_SPECIAL: int = 129


class ArtiType(IntEnum):
    """Port of artitype_t (h2def.h)."""

    ARTI_NONE = 0
    ARTI_INVULNERABILITY = 1
    ARTI_HEALTH = 2
    ARTI_SUPERHEALTH = 3
    ARTI_HEALINGRADIUS = 4
    ARTI_SUMMON = 5
    ARTI_TORCH = 6
    ARTI_EGG = 7
    ARTI_FLY = 8
    ARTI_BLASTRADIUS = 9
    ARTI_POISONBAG = 10
    ARTI_TELEPORTOTHER = 11
    ARTI_SPEED = 12
    ARTI_BOOSTMANA = 13
    ARTI_BOOSTARMOR = 14
    ARTI_TELEPORT = 15

    # Puzzle artifacts
    ARTI_FIRSTPUZZITEM = 16
    ARTI_PUZZSKULL = 16
    ARTI_PUZZGEMBIG = 17
    ARTI_PUZZGEMRED = 18
    ARTI_PUZZGEMGREEN1 = 19
    ARTI_PUZZGEMGREEN2 = 20
    ARTI_PUZZGEMBLUE1 = 21
    ARTI_PUZZGEMBLUE2 = 22
    ARTI_PUZZBOOK1 = 23
    ARTI_PUZZBOOK2 = 24
    ARTI_PUZZSKULL2 = 25
    ARTI_PUZZFWEAPON = 26
    ARTI_PUZZCWEAPON = 27
    ARTI_PUZZMWEAPON = 28
    ARTI_PUZZGEAR1 = 29
    ARTI_PUZZGEAR2 = 30
    ARTI_PUZZGEAR3 = 31
    ARTI_PUZZGEAR4 = 32
    NUMARTIFACTS = 33


class PowerType(IntEnum):
    """Port of powertype_t (h2def.h)."""

    PW_NONE = 0
    PW_INVULNERABILITY = 1
    PW_ALLMAP = 2
    PW_INFRARED = 3
    PW_FLIGHT = 4
    PW_SHIELD = 5
    PW_HEALTH2 = 6
    PW_SPEED = 7
    PW_MINOTAUR = 8
    NUMPOWERS = 9


ARTIFACT_MESSAGES: tuple[str, ...] = (
    "",
    "ICON OF THE DEFENDER",
    "QUARTZ FLASK",
    "MYSTIC URN",
    "MYSTIC AMBIT INCANT",
    "DARK SERVANT",
    "TORCH",
    "PORKALATOR",
    "WINGS OF WRATH",
    "DISC OF REPULSION",
    "FLECHETTE",
    "BANISHMENT DEVICE",
    "BOOTS OF SPEED",
    "KRATER OF MIGHT",
    "DRAGONSKIN BRACERS",
    "CHAOS DEVICE",
    "YORICK'S SKULL",
    "HEART OF D'SPARIL",
    "RUBY PLANET",
    "EMERALD PLANET",
    "EMERALD PLANET",
    "SAPPHIRE PLANET",
    "SAPPHIRE PLANET",
    "DAEMON CODEX",
    "LIBER OSCURA",
    "FLAME MASK",
    "GLAIVE SEAL",
    "HOLY RELIC",
    "SIGIL OF THE MAGUS",
    "CLOCK GEAR",
    "CLOCK GEAR",
    "CLOCK GEAR",
    "CLOCK GEAR",
)

TEXT_KEY_MESSAGES: tuple[str, ...] = (
    "STEEL KEY",
    "CAVE KEY",
    "AXE KEY",
    "FIRE KEY",
    "EMERALD KEY",
    "DUNGEON KEY",
    "SILVER KEY",
    "RUSTED KEY",
    "HORN KEY",
    "SWAMP KEY",
    "CASTLE KEY",
)


# =============================================================================
# Sound & Helper Functions
# =============================================================================


def _is_sim(arg: Any) -> bool:
    """Detect whether arg behaves as a GameSim context."""
    return hasattr(arg, "level") and hasattr(arg, "events")


def aprox_distance(dx: int, dy: int) -> int:
    """Port of P_AproxDistance (p_maputl.c:862)."""
    adx = abs32(dx)
    ady = abs32(dy)
    if adx < ady:
        return adx + ady - (adx >> 1)
    return adx + ady - (ady >> 1)


def _start_sound(
    sim: GameSim,
    actor: Mobj | None,
    sound_id: int | SfxEnum,
    *,
    full_volume: bool = False,
) -> None:
    """Emit a SoundEvent on sim.events."""
    sid = int(sound_id)
    if sid <= 0 or sid == int(SfxEnum.SFX_NONE):
        return
    start_sound(sim, None if full_volume else actor, sid)


def active_minotaur(sim: GameSim, master: Player) -> Mobj | None:
    """Search active mobjs for an active minotaur bound to `master`.

    Port of ActiveMinotaur (p_inter.c:1248).
    """
    for mo in sim.thinkers:
        if (
            isinstance(mo, Mobj)
            and mo.type == MobjType.MT_MINOTAUR
            and getattr(mo, "special1", None) is master.mo
        ):
            return mo
    return None


# =============================================================================
# Core Stat & Inventory Modifications
# =============================================================================


def give_body(player: Player, num: int) -> bool:
    """Port of P_GiveBody (p_inter.c:535). Returns False if body isn't needed."""
    max_health = MAXMORPHHEALTH if player.morph_tics else MAXHEALTH
    if player.health >= max_health:
        return False
    player.health = min(max_health, player.health + num)
    if player.mo is not None:
        player.mo.health = player.health
    return True


def give_armor(player: Player, armortype: int, amount: int) -> bool:
    """Port of P_GiveArmor (p_inter.c:565).

    amount == -1: sets to class increment hit value if current is lower.
    amount > 0: adds amount * 5 * FRACUNIT up to ARMOR_MAX.
    """
    pclass = player.player_class
    if pclass >= len(ARMOR_INCREMENT):
        pclass = 0

    if amount == -1:
        hits = ARMOR_INCREMENT[pclass][armortype]
        if player.armor_points[armortype] >= hits:
            return False
        player.armor_points[armortype] = hits
        return True

    hits = amount * 5 * FRACUNIT
    total_armor = sum(player.armor_points) + AUTO_ARMOR_SAVE[pclass]
    if total_armor < ARMOR_MAX[pclass] * 5 * FRACUNIT:
        player.armor_points[armortype] += hits
        return True
    return False


def give_key(player: Player, key: int) -> bool:
    """Port of P_GiveKey (p_inter.c:608). Returns False if player already has key."""
    if player.keys & (1 << key):
        return False
    player.bonuscount += BONUSADD
    player.keys |= 1 << key
    return True


def give_power(sim: GameSim, player: Player, power: int) -> bool:
    """Port of P_GivePower (p_inter.c:627). Returns True if power accepted."""
    mo = player.mo
    if power == PowerType.PW_INVULNERABILITY:
        if player.powers[power] > BLINKTHRESHOLD:
            return False
        player.powers[power] = INVULNTICS
        if mo is not None:
            mo.flags2 |= MobjFlag2.MF2_INVULNERABLE
            if player.player_class == 2:  # PCLASS_MAGE
                mo.flags2 |= MobjFlag2.MF2_REFLECTIVE
        return True

    if power == PowerType.PW_FLIGHT:
        if player.powers[power] > BLINKTHRESHOLD:
            return False
        player.powers[power] = FLIGHTTICS
        if mo is not None:
            mo.flags2 |= MobjFlag2.MF2_FLY
            mo.flags |= MobjFlag.MF_NOGRAVITY
            if mo.z <= mo.floorz:
                player.flyheight = 10
        return True

    if power == PowerType.PW_INFRARED:
        if player.powers[power] > BLINKTHRESHOLD:
            return False
        player.powers[power] = INFRATICS
        return True

    if power == PowerType.PW_SPEED:
        if player.powers[power] > BLINKTHRESHOLD:
            return False
        player.powers[power] = SPEEDTICS
        return True

    if power == PowerType.PW_MINOTAUR:
        player.powers[power] = MAULATORTICS
        return True

    if player.powers[power]:
        return False
    player.powers[power] = 1
    return True


def give_artifact(player: Player, arti: int, mo: Mobj | None = None) -> bool:
    """Port of P_GiveArtifact (p_inter.c:789). Returns True if accepted."""
    from pyhexen.sim.player import InventorySlot

    # Find existing slot
    slot_idx = -1
    for idx, slot in enumerate(player.inventory):
        if slot.type == arti:
            slot_idx = idx
            break

    if slot_idx == -1:
        # Insert before first puzzle item if this is a standard artifact
        insert_idx = len(player.inventory)
        if arti < ArtiType.ARTI_FIRSTPUZZITEM:
            for idx, slot in enumerate(player.inventory):
                if slot.type >= ArtiType.ARTI_FIRSTPUZZITEM:
                    insert_idx = idx
                    break
        new_slot = InventorySlot(type=arti, count=1)
        player.inventory.insert(insert_idx, new_slot)
    else:
        if player.inventory[slot_idx].count >= 25:
            return False
        player.inventory[slot_idx].count += 1

    if not player.artifact_count:
        player.ready_artifact = arti
        player.inv_ptr = 0

    player.artifact_count += 1
    return True


def set_dormant_artifact(sim: GameSim, arti: Mobj) -> None:
    """Port of SetDormantArtifact (p_inter.c:865)."""
    arti.flags &= ~MobjFlag.MF_SPECIAL
    if sim.deathmatch and not (arti.flags2 & MobjFlag2.MF2_DROPPED):
        if arti.type == MobjType.MT_ARTIINVULNERABILITY:
            set_mobj_state(sim, arti, StateNum.S_DORMANTARTI3_1)
        elif arti.type in (MobjType.MT_SUMMONMAULATOR, MobjType.MT_ARTIFLY):
            set_mobj_state(sim, arti, StateNum.S_DORMANTARTI2_1)
        else:
            set_mobj_state(sim, arti, StateNum.S_DORMANTARTI1_1)
    else:
        set_mobj_state(sim, arti, StateNum.S_DEADARTI1)


def try_pickup_artifact(
    sim: GameSim,
    player: Player,
    artifact_type: int,
    artifact: Mobj,
) -> None:
    """Port of TryPickupArtifact (p_inter.c:709)."""
    if give_artifact(player, artifact_type, artifact):
        if artifact.special:
            execute_line_special(
                sim,
                artifact.special,
                artifact.args,
                None,
                player.mo,
                side=0,
            )
            artifact.special = 0

        player.bonuscount += BONUSADD
        msg = ARTIFACT_MESSAGES[artifact_type] if artifact_type < len(ARTIFACT_MESSAGES) else ""

        if artifact_type < ArtiType.ARTI_FIRSTPUZZITEM:
            set_dormant_artifact(sim, artifact)
            _start_sound(sim, artifact, SfxEnum.SFX_PICKUP_ARTIFACT)
            if msg:
                set_message(sim, player, msg, False)
        else:
            _start_sound(sim, player.mo, SfxEnum.SFX_PICKUP_ITEM)
            if msg:
                set_message(sim, player, msg, True)  # puzzle item: ultimate
            if not sim.netgame or sim.deathmatch:
                remove_mobj(sim, artifact)


def player_next_artifact(player: Player) -> None:
    """Port of P_PlayerNextArtifact (p_user.c:1326). Cycles active readyArtifact."""
    if not player.inventory:
        player.ready_artifact = ArtiType.ARTI_NONE
        player.inv_ptr = 0
        return

    player.inv_ptr -= 1
    if player.inv_ptr < 0:
        player.inv_ptr = len(player.inventory) - 1
    player.ready_artifact = player.inventory[player.inv_ptr].type


def player_remove_artifact(player: Player, slot: int) -> None:
    """Port of P_PlayerRemoveArtifact (p_user.c:1361). Consumes one artifact from slot."""
    if slot < 0 or slot >= len(player.inventory):
        return

    player.artifact_count -= 1
    player.inventory[slot].count -= 1
    if player.inventory[slot].count <= 0:
        player.inventory.pop(slot)
        if player.inv_ptr >= len(player.inventory):
            player.inv_ptr = max(0, len(player.inventory) - 1)
        if player.inventory:
            player.ready_artifact = player.inventory[player.inv_ptr].type
        else:
            player.ready_artifact = ArtiType.ARTI_NONE


# =============================================================================
# Artifact Usage & Effects
# =============================================================================


def blast_mobj(sim: GameSim, source: Mobj, victim: Mobj, strength: int = 255) -> None:
    """Port of P_BlastMobj (p_user.c:1102)."""
    angle = (
        point_to_angle2(source.x, source.y, victim.x, victim.y) & 0xFFFFFFFF
    ) >> ANGLETOFINESHIFT
    if strength < BLAST_FULLSTRENGTH:
        victim.momx = fixed_mul(strength, fine_cosine(angle))
        victim.momy = fixed_mul(strength, fine_sine(angle))
        if victim.player is None:
            victim.flags2 |= MobjFlag2.MF2_SLIDE | MobjFlag2.MF2_BLASTED
    else:
        if victim.flags & MobjFlag.MF_MISSILE:
            if victim.type in (
                MobjType.MT_SORCBALL1,
                MobjType.MT_SORCBALL2,
                MobjType.MT_SORCBALL3,
            ):
                return
            if victim.type == MobjType.MT_MSTAFF_FX2:
                victim.special1 = victim.target
                victim.target = source

        if victim.type == MobjType.MT_HOLY_FX and victim.special1 is source:
            victim.special1 = victim.target
            victim.target = source

        victim.momx = fixed_mul(BLAST_SPEED, fine_cosine(angle))
        victim.momy = fixed_mul(BLAST_SPEED, fine_sine(angle))

        ang = (
            point_to_angle2(victim.x, victim.y, source.x, source.y) & 0xFFFFFFFF
        ) >> ANGLETOFINESHIFT
        px = victim.x + fixed_mul(victim.radius + FRACUNIT, fine_cosine(ang))
        py = victim.y + fixed_mul(victim.radius + FRACUNIT, fine_sine(ang))
        pz = victim.z - victim.floorclip + (victim.height >> 1)
        puff = spawn_mobj(sim, px, py, pz, MobjType.MT_BLASTEFFECT)
        if puff is not None:
            puff.momx = victim.momx
            puff.momy = victim.momy

        if victim.flags & MobjFlag.MF_MISSILE:
            victim.momz = 8 * FRACUNIT
            if puff is not None:
                puff.momz = victim.momz
        else:
            mass = victim.info.mass if victim.info.mass else 100
            victim.momz = c_div(1000, mass) << FRACBITS

        if victim.player is None:
            victim.flags2 |= MobjFlag2.MF2_SLIDE | MobjFlag2.MF2_BLASTED


def blast_radius(sim: GameSim, player: Player) -> None:
    """Port of P_BlastRadius (p_user.c:1190). Repels nearby enemies and missiles."""
    pmo = player.mo
    if pmo is None:
        return

    _start_sound(sim, pmo, SfxEnum.SFX_ARTIFACT_BLAST)
    noise_alert(sim, pmo, pmo)

    for mo in list(sim.thinkers):
        if not isinstance(mo, Mobj):
            continue
        if mo is pmo or (mo.flags2 & MobjFlag2.MF2_BOSS):
            continue
        if mo.type in (
            MobjType.MT_POISONCLOUD,
            MobjType.MT_HOLY_FX,
        ) or (mo.flags & MobjFlag.MF_ICECORPSE):
            pass
        elif (mo.flags & MobjFlag.MF_COUNTKILL) and mo.health <= 0:
            continue
        elif (
            not (mo.flags & MobjFlag.MF_COUNTKILL)
            and mo.player is None
            and not (mo.flags & MobjFlag.MF_MISSILE)
        ):
            continue

        if mo.flags2 & MobjFlag2.MF2_DORMANT:
            continue
        if mo.type == MobjType.MT_WRAITHB and (mo.flags2 & MobjFlag2.MF2_DONTDRAW):
            continue
        if mo.type in (
            MobjType.MT_SPLASHBASE,
            MobjType.MT_SPLASH,
            MobjType.MT_SERPENT,
            MobjType.MT_SERPENTLEADER,
        ):
            continue

        dist = aprox_distance(pmo.x - mo.x, pmo.y - mo.y)
        if dist > BLAST_RADIUS_DIST:
            continue
        blast_mobj(sim, pmo, mo, BLAST_FULLSTRENGTH)


def heal_radius(sim: GameSim, player: Player) -> bool:
    """Port of P_HealRadius (p_user.c:1255). Mystic Ambit Incant area buff."""
    pmo = player.mo
    if pmo is None:
        return False

    effective = False
    for other_p in sim.players:
        mo = other_p.mo
        if mo is None or mo.health <= 0:
            continue
        dist = aprox_distance(pmo.x - mo.x, pmo.y - mo.y)
        if dist > HEAL_RADIUS_DIST:
            continue

        pclass = player.player_class
        if pclass == 0:  # Fighter: radius armor boost
            if (
                give_armor(other_p, ARMOR_ARMOR, 1)
                or give_armor(other_p, ARMOR_SHIELD, 1)
                or give_armor(other_p, ARMOR_HELMET, 1)
                or give_armor(other_p, ARMOR_AMULET, 1)
            ):
                effective = True
                _start_sound(sim, mo, SfxEnum.SFX_MYSTICINCANT)
        elif pclass == 1:  # Cleric: radius heal
            amt = 50 + (sim.rng.p_random() % 50)
            if give_body(other_p, amt):
                effective = True
                _start_sound(sim, mo, SfxEnum.SFX_MYSTICINCANT)
        elif pclass == 2:  # Mage: radius mana boost
            amt = 50 + (sim.rng.p_random() % 50)
            g1 = give_mana(sim, other_p, MANA_1, amt)
            g2 = give_mana(sim, other_p, MANA_2, amt)
            if g1 or g2:
                effective = True
                _start_sound(sim, mo, SfxEnum.SFX_MYSTICINCANT)

    return effective


def p_arti_tele(sim: GameSim, player: Player) -> None:
    """Port of P_ArtiTele (p_user.c:964). Chaos Device teleport."""
    from pyhexen.sim.actions.monsters_bosses import p_teleport

    pmo = player.mo
    if pmo is None:
        return

    mthing = sim.player_starts.get((0, 0))
    if mthing is not None:
        dest_x = mthing.x << 16
        dest_y = mthing.y << 16
        dest_angle = to_uint32(ANG45 * c_div(mthing.angle, 45))
    else:
        dest_x = 0
        dest_y = 0
        dest_angle = 0

    p_teleport(sim, pmo, dest_x, dest_y, dest_angle, use_fog=True)
    if player.morph_tics:
        undo_player_morph(sim, player)


def p_arti_teleport_other(sim: GameSim, player: Player) -> None:
    """Port of P_ArtiTeleportOther (p_user.c:1001). Banishment Device missile."""
    pmo = player.mo
    if pmo is None:
        return
    mo = spawn_player_missile(sim, pmo, MobjType.MT_TELOTHER_FX1)
    if mo is not None:
        mo.target = pmo


def teleport_other(sim: GameSim, victim: Mobj) -> None:
    """Port of P_TeleportOther (p_user.c:1062). Teleports victim to start spot."""
    from pyhexen.sim.actions.monsters_bosses import p_teleport

    if victim.player is not None:
        mthing = sim.player_starts.get((0, 0))
        if mthing is not None:
            dest_x = mthing.x << 16
            dest_y = mthing.y << 16
            dest_angle = to_uint32(ANG45 * c_div(mthing.angle, 45))
            p_teleport(sim, victim, dest_x, dest_y, dest_angle, use_fog=True)
    else:
        if (victim.flags & MobjFlag.MF_COUNTKILL) and victim.special:
            execute_line_special(
                sim,
                victim.special,
                victim.args,
                None,
                victim,
                side=0,
            )
            victim.special = 0

        mthing = sim.player_starts.get((0, 0))
        if mthing is not None:
            dest_x = mthing.x << 16
            dest_y = mthing.y << 16
            dest_angle = to_uint32(ANG45 * c_div(mthing.angle, 45))
            p_teleport(sim, victim, dest_x, dest_y, dest_angle, use_fog=True)


def morph_player(sim: GameSim, player: Player) -> bool:
    """Port of P_MorphPlayer (p_inter.c:1540). Transforms player into a pig."""
    if player.powers[PowerType.PW_INVULNERABILITY]:
        return False
    if player.morph_tics:
        return False

    pmo = player.mo
    if pmo is None:
        return False

    x = pmo.x
    y = pmo.y
    z = pmo.z
    angle = pmo.angle
    old_flags2 = pmo.flags2

    fog = spawn_mobj(sim, x, y, z + TELEFOGHEIGHT, MobjType.MT_TFOG)
    if fog is not None:
        _start_sound(sim, fog, SfxEnum.SFX_TELEPORT)

    beast_mo = spawn_mobj(sim, x, y, z, MobjType.MT_PIGPLAYER)
    beast_mo.special1 = player.readyweapon
    beast_mo.angle = angle
    beast_mo.player = player
    player.health = beast_mo.health = MAXMORPHHEALTH
    player.mo = beast_mo
    player.armor_points = [0, 0, 0, 0]
    player.player_class = 3  # PCLASS_PIG
    if old_flags2 & MobjFlag2.MF2_FLY:
        beast_mo.flags2 |= MobjFlag2.MF2_FLY
        beast_mo.flags |= MobjFlag.MF_NOGRAVITY

    player.morph_tics = MORPHTICS
    p_activate_morph_weapon(sim, player)
    remove_mobj(sim, pmo)
    return True


def undo_player_morph(sim: GameSim, player: Player) -> bool:
    """Port of P_UndoPlayerMorph (p_user.c:498). Reverts pig form back to normal player."""
    from pyhexen.sim.actions.monsters_bosses import p_test_mobj_location

    pmo = player.mo
    if pmo is None:
        return False

    x = pmo.x
    y = pmo.y
    z = pmo.z
    angle = pmo.angle
    weapon = int(pmo.special1) if isinstance(pmo.special1, int) else 0
    old_flags2 = pmo.flags2

    orig_class = getattr(player, "original_class", 0)
    class_mobj_types = (
        MobjType.MT_PLAYER_FIGHTER,
        MobjType.MT_PLAYER_CLERIC,
        MobjType.MT_PLAYER_MAGE,
    )
    mo_type = class_mobj_types[orig_class if orig_class < len(class_mobj_types) else 0]

    pmo.flags &= ~MobjFlag.MF_SOLID
    mo = spawn_mobj(sim, x, y, z, mo_type)
    if not p_test_mobj_location(sim, mo):
        pmo.flags |= MobjFlag.MF_SOLID
        remove_mobj(sim, mo)
        player.morph_tics = 2 * 35
        return False

    mo.angle = angle
    mo.player = player
    mo.reactiontime = 18
    if old_flags2 & MobjFlag2.MF2_FLY:
        mo.flags2 |= MobjFlag2.MF2_FLY
        mo.flags |= MobjFlag.MF_NOGRAVITY

    player.morph_tics = 0
    player.player_class = orig_class
    player.health = mo.health = MAXHEALTH
    player.mo = mo

    fine = (angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    fog = spawn_mobj(
        sim,
        x + 20 * fine_cosine(fine),
        y + 20 * fine_sine(fine),
        z + TELEFOGHEIGHT,
        MobjType.MT_TFOG,
    )
    if fog is not None:
        _start_sound(sim, fog, SfxEnum.SFX_TELEPORT)

    p_post_morph_weapon(sim, player, weapon)
    remove_mobj(sim, pmo)
    return True


def morph_monster(sim: GameSim, actor: Mobj) -> bool:
    """Port of P_MorphMonster (p_inter.c:1591). Transforms monster into a pig."""
    if actor.player is not None:
        return False
    if not (actor.flags & MobjFlag.MF_COUNTKILL):
        return False
    if actor.flags2 & MobjFlag2.MF2_BOSS:
        return False
    if actor.type in (
        MobjType.MT_PIG,
        MobjType.MT_FIGHTER_BOSS,
        MobjType.MT_CLERIC_BOSS,
        MobjType.MT_MAGE_BOSS,
    ):
        return False

    old_type = actor.type
    x = actor.x
    y = actor.y
    z = actor.z
    angle = actor.angle
    target = actor.target
    tid = actor.tid
    special = actor.special
    args = actor.args
    flags = actor.flags

    fog = spawn_mobj(sim, x, y, z + TELEFOGHEIGHT, MobjType.MT_TFOG)
    if fog is not None:
        _start_sound(sim, fog, SfxEnum.SFX_TELEPORT)

    pig = spawn_mobj(sim, x, y, z, MobjType.MT_PIG)
    pig.special2 = int(old_type)
    pig.special1 = MORPHTICS + sim.rng.p_random()
    pig.flags |= flags & MobjFlag.MF_SHADOW
    pig.target = target
    pig.angle = angle
    pig.tid = tid
    pig.special = special
    pig.args = args

    if old_type == MobjType.MT_MINOTAUR and actor.special1 is not None:
        master = actor.special1
        if hasattr(master, "health") and master.health > 0 and getattr(master, "player", None):
            if not active_minotaur(sim, master.player):
                master.player.powers[PowerType.PW_MINOTAUR] = 0

    remove_mobj(sim, actor)
    return True


def p_morph_player_think(sim: GameSim, player: Player) -> None:
    """Port of P_MorphPlayerThink (p_user.c:444). Periodic pig sounds and snout sniff."""
    if player.morph_tics & 15:
        return
    pmo = player.mo
    if pmo is None:
        return

    if pmo.momx == 0 and pmo.momy == 0 and sim.rng.p_random() < 64:
        p_set_psprite_nf(player, 0, StateNum.S_SNOUTATK2)
        _start_sound(sim, pmo, SfxEnum.SFX_PIG_ACTIVE1)
        return

    if sim.rng.p_random() < 48:
        sfx = SfxEnum.SFX_PIG_ACTIVE1 if sim.rng.p_random() < 128 else SfxEnum.SFX_PIG_ACTIVE2
        _start_sound(sim, pmo, sfx)


def auto_use_health(sim: GameSim, player: Player, save_health: int) -> None:
    """Port of P_AutoUseHealth (p_inter.c:1659). Auto-uses flasks or urns to prevent death."""
    normal_slot = -1
    normal_count = 0
    super_slot = -1
    super_count = 0

    for idx, slot in enumerate(player.inventory):
        if slot.type == ArtiType.ARTI_HEALTH:
            normal_slot = idx
            normal_count = slot.count
        elif slot.type == ArtiType.ARTI_SUPERHEALTH:
            super_slot = idx
            super_count = slot.count

    if getattr(sim, "skill", 2) == 0 and (normal_count * 25 >= save_health):
        count = (save_health + 24) // 25
        for _ in range(count):
            player.health += 25
            player_remove_artifact(player, normal_slot)
    elif super_count * 100 >= save_health:
        count = (save_health + 99) // 100
        for _ in range(count):
            player.health += 100
            player_remove_artifact(player, super_slot)
    elif getattr(sim, "skill", 2) == 0 and (super_count * 100 + normal_count * 25 >= save_health):
        count = (save_health + 24) // 25
        save_health -= count * 25
        for _ in range(count):
            player.health += 25
            player_remove_artifact(player, normal_slot)
        count = (save_health + 99) // 100
        for _ in range(count):
            player.health += 100
            for s, slot in enumerate(player.inventory):
                if slot.type == ArtiType.ARTI_SUPERHEALTH:
                    player_remove_artifact(player, s)
                    break

    if player.mo is not None:
        player.mo.health = player.health


def use_puzzle_item(sim: GameSim, player: Player, item_type: int) -> bool:
    """Port of P_UsePuzzleItem (p_map.c:2112). Uses puzzle item on linedef or mobj."""
    pmo = player.mo
    if pmo is None or sim.level is None:
        return False

    angle = (pmo.angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    x1 = pmo.x
    y1 = pmo.y
    x2 = x1 + (USERANGE >> FRACBITS) * fine_cosine(angle)
    y2 = y1 + (USERANGE >> FRACBITS) * fine_sine(angle)

    activated = False

    def _ptr_puzzle_traverse(intercept: Intercept) -> bool:
        nonlocal activated
        if intercept.is_aline:
            li = intercept.line
            if li is None or li.special != USE_PUZZLE_ITEM_SPECIAL:
                if li is not None:
                    opening = line_opening(li)
                    if opening.openrange <= 0:
                        fail_sfx = (
                            SfxEnum.SFX_PUZZLE_FAIL_FIGHTER
                            if player.player_class == 0
                            else (
                                SfxEnum.SFX_PUZZLE_FAIL_CLERIC
                                if player.player_class == 1
                                else SfxEnum.SFX_PUZZLE_FAIL_MAGE
                            )
                        )
                        _start_sound(sim, pmo, fail_sfx)
                        return False
                return True

            if point_on_line_side(pmo.x, pmo.y, li) == 1:
                return False
            if item_type != li.args[0]:
                return False

            execute_line_special(
                sim,
                80,  # ACS_Execute
                (li.args[1], 0, li.args[2], li.args[3], li.args[4]),
                li,
                pmo,
                side=0,
            )
            li.special = 0
            activated = True
            return False

        # Mobj check
        target_mo = intercept.thing
        if target_mo is None or target_mo.special != USE_PUZZLE_ITEM_SPECIAL:
            return True
        arg0 = target_mo.args[0] if len(target_mo.args) > 0 else 0
        if item_type != arg0:
            return True

        arg1 = target_mo.args[1] if len(target_mo.args) > 1 else 0
        arg2 = target_mo.args[2] if len(target_mo.args) > 2 else 0
        arg3 = target_mo.args[3] if len(target_mo.args) > 3 else 0
        arg4 = target_mo.args[4] if len(target_mo.args) > 4 else 0
        execute_line_special(
            sim,
            80,
            (arg1, 0, arg2, arg3, arg4),
            None,
            pmo,
            side=0,
        )
        target_mo.special = 0
        activated = True
        return False

    path_traverse(sim.level, x1, y1, x2, y2, PT_ADDLINES | PT_ADDTHINGS, _ptr_puzzle_traverse)
    return activated


def use_artifact(sim: GameSim, player: Player, arti: int) -> bool:
    """Port of P_UseArtifact (p_user.c:1446). Returns True if effect activated."""
    pmo = player.mo
    if pmo is None:
        return False

    if arti == ArtiType.ARTI_INVULNERABILITY:
        return give_power(sim, player, PowerType.PW_INVULNERABILITY)
    if arti == ArtiType.ARTI_HEALTH:
        return give_body(player, 25)
    if arti == ArtiType.ARTI_SUPERHEALTH:
        return give_body(player, 100)
    if arti == ArtiType.ARTI_HEALINGRADIUS:
        return heal_radius(sim, player)
    if arti == ArtiType.ARTI_TORCH:
        return give_power(sim, player, PowerType.PW_INFRARED)
    if arti == ArtiType.ARTI_EGG:
        spawn_player_missile(sim, pmo, MobjType.MT_EGGFX)
        spm_angle(sim, pmo, MobjType.MT_EGGFX, to_uint32(pmo.angle - (ANG45 // 6)))
        spm_angle(sim, pmo, MobjType.MT_EGGFX, to_uint32(pmo.angle + (ANG45 // 6)))
        spm_angle(sim, pmo, MobjType.MT_EGGFX, to_uint32(pmo.angle - (ANG45 // 3)))
        spm_angle(sim, pmo, MobjType.MT_EGGFX, to_uint32(pmo.angle + (ANG45 // 3)))
        return True
    if arti == ArtiType.ARTI_FLY:
        return give_power(sim, player, PowerType.PW_FLIGHT)
    if arti == ArtiType.ARTI_SUMMON:
        mo = spawn_player_missile(sim, pmo, MobjType.MT_SUMMON_FX)
        if mo is not None:
            mo.target = pmo
            mo.special1 = pmo
            mo.momz = 5 * FRACUNIT
        return True
    if arti == ArtiType.ARTI_TELEPORT:
        p_arti_tele(sim, player)
        return True
    if arti == ArtiType.ARTI_TELEPORTOTHER:
        p_arti_teleport_other(sim, player)
        return True
    if arti == ArtiType.ARTI_POISONBAG:
        angle = (pmo.angle & 0xFFFFFFFF) >> ANGLETOFINESHIFT
        if player.player_class == 1:  # Cleric
            bmo = spawn_mobj(
                sim,
                pmo.x + 16 * fine_cosine(angle),
                pmo.y + 24 * fine_sine(angle),
                pmo.z - pmo.floorclip + 8 * FRACUNIT,
                MobjType.MT_POISONBAG,
            )
            if bmo is not None:
                bmo.target = pmo
        elif player.player_class == 2:  # Mage
            bmo = spawn_mobj(
                sim,
                pmo.x + 16 * fine_cosine(angle),
                pmo.y + 24 * fine_sine(angle),
                pmo.z - pmo.floorclip + 8 * FRACUNIT,
                MobjType.MT_FIREBOMB,
            )
            if bmo is not None:
                bmo.target = pmo
        else:  # Fighter / Pig
            bmo = spawn_mobj(
                sim,
                pmo.x,
                pmo.y,
                pmo.z - pmo.floorclip + 35 * FRACUNIT,
                MobjType.MT_THROWINGBOMB,
            )
            if bmo is not None:
                bmo.angle = to_uint32(pmo.angle + (((sim.rng.p_random() & 7) - 4) << 24))
                bmo.momz = 4 * FRACUNIT + (player.lookdir << (FRACBITS - 4))
                bmo.z += player.lookdir << (FRACBITS - 4)
                thrust_mobj(bmo, bmo.angle, bmo.info.speed)
                bmo.momx += pmo.momx >> 1
                bmo.momy += pmo.momy >> 1
                bmo.target = pmo
                bmo.tics -= sim.rng.p_random() & 3
                check_missile_spawn(sim, bmo)
        return True
    if arti == ArtiType.ARTI_SPEED:
        return give_power(sim, player, PowerType.PW_SPEED)
    if arti == ArtiType.ARTI_BOOSTMANA:
        g1 = give_mana(sim, player, MANA_1, MAX_MANA)
        g2 = give_mana(sim, player, MANA_2, MAX_MANA)
        return g1 or g2
    if arti == ArtiType.ARTI_BOOSTARMOR:
        count = 0
        for i in range(NUMARMOR):
            count += int(give_armor(player, i, 1))
        return count > 0
    if arti == ArtiType.ARTI_BLASTRADIUS:
        blast_radius(sim, player)
        return True

    # Puzzle artifacts
    if arti >= ArtiType.ARTI_FIRSTPUZZITEM:
        if use_puzzle_item(sim, player, arti - ArtiType.ARTI_FIRSTPUZZITEM):
            return True
        set_yellow_message(sim, player, "YOU CANNOT USE THIS HERE", False)
        return False

    return False


def player_use_artifact(sim: GameSim, player: Player, arti: int) -> bool:
    """Port of P_PlayerUseArtifact (p_user.c:1405). Checks inventory, uses and removes."""
    for idx, slot in enumerate(player.inventory):
        if slot.type == arti:
            if use_artifact(sim, player, arti):
                player_remove_artifact(player, idx)
                sfx = (
                    SfxEnum.SFX_ARTIFACT_USE
                    if arti < ArtiType.ARTI_FIRSTPUZZITEM
                    else SfxEnum.SFX_PUZZLE_SUCCESS
                )
                _start_sound(sim, player.mo, sfx)
                return True
            if arti < ArtiType.ARTI_FIRSTPUZZITEM:
                player_next_artifact(player)
            return False
    return False


# =============================================================================
# Collision & Special Thing Pickup Dispatch
# =============================================================================


def touch_special_thing(sim: GameSim, special: Mobj, toucher: Mobj) -> None:
    """Port of P_TouchSpecialThing (p_inter.c:934). Dispatches all item pickups."""
    player = toucher.player
    if player is None:
        return

    delta = special.z - toucher.z
    if delta > toucher.height or delta < -32 * FRACUNIT:
        return
    if toucher.health <= 0:
        return

    sound = SfxEnum.SFX_PICKUP_ITEM
    sp = special.sprite

    # Health Potion
    if sp == SpriteNum.SPR_PTN1:
        if not give_body(player, 10):
            return
        set_message(sim, player, "CRYSTAL VIAL", False)

    # Armor pickups
    elif sp == SpriteNum.SPR_ARM1:
        if not give_armor(player, ARMOR_ARMOR, -1):
            return
        set_message(sim, player, "MESH ARMOR", False)
    elif sp == SpriteNum.SPR_ARM2:
        if not give_armor(player, ARMOR_SHIELD, -1):
            return
        set_message(sim, player, "FALCON SHIELD", False)
    elif sp == SpriteNum.SPR_ARM3:
        if not give_armor(player, ARMOR_HELMET, -1):
            return
        set_message(sim, player, "PLATINUM HELMET", False)
    elif sp == SpriteNum.SPR_ARM4:
        if not give_armor(player, ARMOR_AMULET, -1):
            return
        set_message(sim, player, "AMULET OF WARDING", False)

    # Keys
    elif SpriteNum.SPR_KEY1 <= sp <= SpriteNum.SPR_KEYB:
        key_idx = sp - SpriteNum.SPR_KEY1
        if not give_key(player, key_idx):
            return
        sound = SfxEnum.SFX_PICKUP_KEY
        if key_idx < len(TEXT_KEY_MESSAGES):
            set_message(sim, player, TEXT_KEY_MESSAGES[key_idx], True)
        if special.special:
            execute_line_special(
                sim,
                special.special,
                special.args,
                None,
                toucher,
                side=0,
            )
            special.special = 0
        if not sim.netgame:
            remove_mobj(sim, special)
        _start_sound(sim, toucher, sound)
        return

    # Standard Artifacts
    elif sp == SpriteNum.SPR_PTN2:
        try_pickup_artifact(sim, player, ArtiType.ARTI_HEALTH, special)
        return
    elif sp == SpriteNum.SPR_SOAR:
        try_pickup_artifact(sim, player, ArtiType.ARTI_FLY, special)
        return
    elif sp == SpriteNum.SPR_INVU:
        try_pickup_artifact(sim, player, ArtiType.ARTI_INVULNERABILITY, special)
        return
    elif sp == SpriteNum.SPR_SUMN:
        try_pickup_artifact(sim, player, ArtiType.ARTI_SUMMON, special)
        return
    elif sp == SpriteNum.SPR_PORK:
        try_pickup_artifact(sim, player, ArtiType.ARTI_EGG, special)
        return
    elif sp == SpriteNum.SPR_SPHL:
        try_pickup_artifact(sim, player, ArtiType.ARTI_SUPERHEALTH, special)
        return
    elif sp == SpriteNum.SPR_HRAD:
        try_pickup_artifact(sim, player, ArtiType.ARTI_HEALINGRADIUS, special)
        return
    elif sp == SpriteNum.SPR_TRCH:
        try_pickup_artifact(sim, player, ArtiType.ARTI_TORCH, special)
        return
    elif sp == SpriteNum.SPR_ATLP:
        try_pickup_artifact(sim, player, ArtiType.ARTI_TELEPORT, special)
        return
    elif sp == SpriteNum.SPR_TELO:
        try_pickup_artifact(sim, player, ArtiType.ARTI_TELEPORTOTHER, special)
        return
    elif sp == SpriteNum.SPR_PSBG:
        try_pickup_artifact(sim, player, ArtiType.ARTI_POISONBAG, special)
        return
    elif sp == SpriteNum.SPR_SPED:
        try_pickup_artifact(sim, player, ArtiType.ARTI_SPEED, special)
        return
    elif sp == SpriteNum.SPR_BMAN:
        try_pickup_artifact(sim, player, ArtiType.ARTI_BOOSTMANA, special)
        return
    elif sp == SpriteNum.SPR_BRAC:
        try_pickup_artifact(sim, player, ArtiType.ARTI_BOOSTARMOR, special)
        return
    elif sp == SpriteNum.SPR_BLST:
        try_pickup_artifact(sim, player, ArtiType.ARTI_BLASTRADIUS, special)
        return

    # Puzzle Artifacts
    elif sp == SpriteNum.SPR_ASKU:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZSKULL, special)
        return
    elif sp == SpriteNum.SPR_ABGM:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEMBIG, special)
        return
    elif sp == SpriteNum.SPR_AGMR:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEMRED, special)
        return
    elif sp == SpriteNum.SPR_AGMG:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEMGREEN1, special)
        return
    elif sp == SpriteNum.SPR_AGG2:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEMGREEN2, special)
        return
    elif sp == SpriteNum.SPR_AGMB:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEMBLUE1, special)
        return
    elif sp == SpriteNum.SPR_AGB2:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEMBLUE2, special)
        return
    elif sp == SpriteNum.SPR_ABK1:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZBOOK1, special)
        return
    elif sp == SpriteNum.SPR_ABK2:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZBOOK2, special)
        return
    elif sp == SpriteNum.SPR_ASK2:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZSKULL2, special)
        return
    elif sp == SpriteNum.SPR_AFWP:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZFWEAPON, special)
        return
    elif sp == SpriteNum.SPR_ACWP:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZCWEAPON, special)
        return
    elif sp == SpriteNum.SPR_AMWP:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZMWEAPON, special)
        return
    elif sp == SpriteNum.SPR_AGER:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEAR1, special)
        return
    elif sp == SpriteNum.SPR_AGR2:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEAR2, special)
        return
    elif sp == SpriteNum.SPR_AGR3:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEAR3, special)
        return
    elif sp == SpriteNum.SPR_AGR4:
        try_pickup_artifact(sim, player, ArtiType.ARTI_PUZZGEAR4, special)
        return

    # Mana Pickups
    elif sp == SpriteNum.SPR_MAN1:
        if not give_mana(sim, player, MANA_1, 15):
            return
        set_message(sim, player, "BLUE MANA", False)
    elif sp == SpriteNum.SPR_MAN2:
        if not give_mana(sim, player, MANA_2, 15):
            return
        set_message(sim, player, "GREEN MANA", False)
    elif sp == SpriteNum.SPR_MAN3:
        g1 = give_mana(sim, player, MANA_1, 20)
        g2 = give_mana(sim, player, MANA_2, 20)
        if not (g1 or g2):
            return
        set_message(sim, player, "COMBINED MANA", False)

    # Weapons
    elif sp == SpriteNum.SPR_WMCS:
        give_weapon(sim, player, 2, WP_SECOND, special, "FROST SHARDS")
        return
    elif sp == SpriteNum.SPR_WMLG:
        give_weapon(sim, player, 2, WP_THIRD, special, "ARC OF DEATH")
        return
    elif sp == SpriteNum.SPR_WFAX:
        give_weapon(sim, player, 0, WP_SECOND, special, "TIMON'S AXE")
        return
    elif sp == SpriteNum.SPR_WFHM:
        give_weapon(sim, player, 0, WP_THIRD, special, "HAMMER OF RETRIBUTION")
        return
    elif sp == SpriteNum.SPR_WCSS:
        give_weapon(sim, player, 1, WP_SECOND, special, "SERPENT STAFF")
        return
    elif sp == SpriteNum.SPR_WCFM:
        give_weapon(sim, player, 1, WP_THIRD, special, "FIRESTORM")
        return

    # Fourth Weapon Pieces
    elif sp == SpriteNum.SPR_WFR1:
        give_weapon_piece(sim, player, 0, 1, special)
        return
    elif sp == SpriteNum.SPR_WFR2:
        give_weapon_piece(sim, player, 0, 2, special)
        return
    elif sp == SpriteNum.SPR_WFR3:
        give_weapon_piece(sim, player, 0, 4, special)
        return
    elif sp == SpriteNum.SPR_WCH1:
        give_weapon_piece(sim, player, 1, 1, special)
        return
    elif sp == SpriteNum.SPR_WCH2:
        give_weapon_piece(sim, player, 1, 2, special)
        return
    elif sp == SpriteNum.SPR_WCH3:
        give_weapon_piece(sim, player, 1, 4, special)
        return
    elif sp == SpriteNum.SPR_WMS1:
        give_weapon_piece(sim, player, 2, 1, special)
        return
    elif sp == SpriteNum.SPR_WMS2:
        give_weapon_piece(sim, player, 2, 2, special)
        return
    elif sp == SpriteNum.SPR_WMS3:
        give_weapon_piece(sim, player, 2, 4, special)
        return
    else:
        return

    if special.special:
        execute_line_special(
            sim,
            special.special,
            special.args,
            None,
            toucher,
            side=0,
        )
        special.special = 0

    remove_mobj(sim, special)
    player.bonuscount += BONUSADD
    _start_sound(sim, toucher, sound)


# =============================================================================
# Action Functions
# =============================================================================


@action("A_Summon")
def a_summon(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_Summon (a_action.c:615). Summons Dark Servant minotaur."""
    from pyhexen.sim.actions.monsters_bosses import p_test_mobj_location

    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_MINOTAUR)
    if mo is not None:
        if not p_test_mobj_location(sim, mo) or actor.special1 is None:
            remove_mobj(sim, mo)
            ret_mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_SUMMONMAULATOR)
            if ret_mo is not None:
                ret_mo.flags2 |= MobjFlag2.MF2_DROPPED
            return

        lt = sim.level_time
        mo.args = (lt & 0xFF, (lt >> 8) & 0xFF, (lt >> 16) & 0xFF, (lt >> 24) & 0xFF, 0)
        master = actor.special1
        if isinstance(master, Mobj) and (master.flags & MobjFlag.MF_CORPSE):
            mo.special1 = None
        else:
            mo.special1 = master
            if isinstance(master, Mobj) and master.player is not None:
                give_power(sim, master.player, PowerType.PW_MINOTAUR)

        spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_MNTRSMOKE)


@action("A_PoisonBagInit")
def a_poisonbaginit(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_PoisonBagInit (a_action.c:744). Initializes poison cloud."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z + 28 * FRACUNIT, MobjType.MT_POISONCLOUD)
    if mo is not None:
        mo.momx = 1
        mo.special1 = 24 + (sim.rng.p_random() & 7)
        mo.special2 = 0
        mo.target = actor.target
        mo.radius = 20 * FRACUNIT
        mo.height = 30 * FRACUNIT
        mo.flags &= ~MobjFlag.MF_NOCLIP


@action("A_PoisonBagCheck")
def a_poisonbagcheck(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_PoisonBagCheck (a_action.c:769). Ticks down poison cloud."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    val = int(actor.special1) if isinstance(actor.special1, int) else 0
    val -= 1
    actor.special1 = val
    if val <= 0:
        set_mobj_state(sim, actor, StateNum.S_POISONCLOUD_X1)


@action("A_PoisonBagDamage")
def a_poisonbagdamage(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_PoisonBagDamage (a_action.c:787). Applies splash damage and bobs z."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    a_explode(actor, sim)
    bob_idx = int(actor.special2) if isinstance(actor.special2, int) else 0
    actor.z += FLOAT_BOB_OFFSETS[bob_idx & 63] >> 4
    actor.special2 = (bob_idx + 1) & 63


@action("A_PoisonShroom")
def a_poisonshroom(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_PoisonShroom (a_action.c:804). Randomizes shroom duration."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return
    actor.tics = 128 + (sim.rng.p_random() << 1)


@action("A_CheckThrowBomb")
def a_checkthrowbomb(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_CheckThrowBomb (a_action.c:815). Flechette throwing bomb ground settle."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    if (
        abs32(actor.momx) < (3 * FRACUNIT >> 1)
        and abs32(actor.momy) < (3 * FRACUNIT >> 1)
        and actor.momz < 2 * FRACUNIT
        and actor.state_num == int(StateNum.S_THROWINGBOMB6)
    ):
        set_mobj_state(sim, actor, StateNum.S_THROWINGBOMB7)
        actor.z = actor.floorz
        actor.momz = 0
        actor.flags2 &= ~MobjFlag2.MF2_FLOORBOUNCE
        actor.flags &= ~MobjFlag.MF_MISSILE

    actor.health -= 1
    if actor.health <= 0:
        set_mobj_state(sim, actor, actor.info.deathstate)


@action("A_CheckTeleRing")
def a_checktelering(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_CheckTeleRing (a_action.c:1011). Lifetime countdown for teleport rings."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    val = int(actor.special1) if isinstance(actor.special1, int) else 0
    val -= 1
    actor.special1 = val
    if val <= 0:
        set_mobj_state(sim, actor, actor.info.deathstate)


@action("A_SpeedFade")
def a_speedfade(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_SpeedFade (p_user.c:1638). Shadow ghost fade for boots of speed."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None:
        return

    actor.flags |= MobjFlag.MF_SHADOW
    actor.flags &= ~MobjFlag.MF_ALTSHADOW
    if actor.target is not None:
        actor.sprite = actor.target.sprite


@action("A_RestoreArtifact")
def a_restoreartifact(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_RestoreArtifact (p_inter.c:895). Respawns dormant artifact."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.flags |= MobjFlag.MF_SPECIAL
    set_mobj_state(sim, actor, actor.info.spawnstate)
    _start_sound(sim, actor, SfxEnum.SFX_RESPAWN)


@action("A_RestoreSpecialThing1")
def a_restorespecialthing1(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_RestoreSpecialThing1 (p_inter.c:910). Makes special thing visible again."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.flags2 &= ~MobjFlag2.MF2_DONTDRAW
    _start_sound(sim, actor, SfxEnum.SFX_RESPAWN)


@action("A_RestoreSpecialThing2")
def a_restorespecialthing2(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_RestoreSpecialThing2 (p_inter.c:918). Re-enables special thing pickup."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    actor.flags |= MobjFlag.MF_SPECIAL
    set_mobj_state(sim, actor, actor.info.spawnstate)


@action("A_TeloSpawnA")
def a_telospawna(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_TeloSpawnA (a_action.c:947). Spawns MT_TELOTHER_FX2."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_TELOTHER_FX2)
    if mo is not None:
        mo.special1 = 1
        mo.angle = actor.angle
        mo.target = actor.target
        mo.momx = actor.momx >> 1
        mo.momy = actor.momy >> 1
        mo.momz = actor.momz >> 1


@action("A_TeloSpawnB")
def a_telospawnb(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_TeloSpawnB (a_action.c:963). Spawns MT_TELOTHER_FX3."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_TELOTHER_FX3)
    if mo is not None:
        mo.special1 = 1
        mo.angle = actor.angle
        mo.target = actor.target
        mo.momx = actor.momx >> 1
        mo.momy = actor.momy >> 1
        mo.momz = actor.momz >> 1


@action("A_TeloSpawnC")
def a_telospawnc(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_TeloSpawnC (a_action.c:979). Spawns MT_TELOTHER_FX4."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_TELOTHER_FX4)
    if mo is not None:
        mo.special1 = 1
        mo.angle = actor.angle
        mo.target = actor.target
        mo.momx = actor.momx >> 1
        mo.momy = actor.momy >> 1
        mo.momz = actor.momz >> 1


@action("A_TeloSpawnD")
def a_telospawnd(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_TeloSpawnD (a_action.c:995). Spawns MT_TELOTHER_FX5."""
    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = spawn_mobj(sim, actor.x, actor.y, actor.z, MobjType.MT_TELOTHER_FX5)
    if mo is not None:
        mo.special1 = 1
        mo.angle = actor.angle
        mo.target = actor.target
        mo.momx = actor.momx >> 1
        mo.momy = actor.momy >> 1
        mo.momz = actor.momz >> 1


@action("A_SoAExplode")
def a_soaexplode(actor: Mobj | GameSim, sim: GameSim | Mobj | None = None) -> None:
    """Port of A_SoAExplode (a_action.c:1155). Suit of armor explosion."""
    from pyhexen.sim.acs.vm import TRANSLATE_THING_TYPE

    if _is_sim(actor):
        sim, actor = actor, sim
    if actor is None or sim is None:
        return

    mo = None
    for i in range(10):
        r1 = sim.rng.p_random()
        r2 = sim.rng.p_random()
        r3 = sim.rng.p_random()
        mo = spawn_mobj(
            sim,
            actor.x + ((r3 - 128) << 12),
            actor.y + ((r2 - 128) << 12),
            actor.z + c_div(r1 * actor.height, 256),
            MobjType.MT_ZARMORCHUNK,
        )
        set_mobj_state(sim, mo, mo.info.spawnstate + i)
        mo.momz = ((sim.rng.p_random() & 7) + 5) * FRACUNIT
        mo.momx = sim.rng.p_subrandom() << (FRACBITS - 6)
        mo.momy = sim.rng.p_subrandom() << (FRACBITS - 6)

    arg0 = actor.args[0] if len(actor.args) > 0 else 0
    if arg0 and arg0 < len(TRANSLATE_THING_TYPE):
        spawn_mobj(sim, actor.x, actor.y, actor.z, TRANSLATE_THING_TYPE[arg0])

    if mo is not None:
        _start_sound(sim, mo, SfxEnum.SFX_SUITOFARMOR_BREAK)
    remove_mobj(sim, actor)


__all__ = [
    "ArtiType",
    "PowerType",
    "give_body",
    "give_armor",
    "give_key",
    "give_power",
    "give_artifact",
    "try_pickup_artifact",
    "player_next_artifact",
    "player_remove_artifact",
    "player_use_artifact",
    "use_artifact",
    "auto_use_health",
    "morph_player",
    "undo_player_morph",
    "morph_monster",
    "p_morph_player_think",
    "teleport_other",
    "p_arti_tele",
    "p_arti_teleport_other",
    "blast_mobj",
    "blast_radius",
    "heal_radius",
    "use_puzzle_item",
    "touch_special_thing",
    "set_dormant_artifact",
    "active_minotaur",
    "a_summon",
    "a_poisonbaginit",
    "a_poisonbagcheck",
    "a_poisonbagdamage",
    "a_poisonshroom",
    "a_checkthrowbomb",
    "a_checktelering",
    "a_speedfade",
    "a_restoreartifact",
    "a_restorespecialthing1",
    "a_restorespecialthing2",
    "a_telospawna",
    "a_telospawnb",
    "a_telospawnc",
    "a_telospawnd",
    "a_soaexplode",
]
