# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for items, inventory, artifacts, pickups, and pig morphing (T10.10)."""

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.artifacts import (
    ArtiType,
    PowerType,
    a_checktelering,
    a_checkthrowbomb,
    a_poisonbagcheck,
    a_poisonbaginit,
    a_poisonshroom,
    a_restoreartifact,
    a_restorespecialthing1,
    a_restorespecialthing2,
    a_soaexplode,
    a_speedfade,
    blast_radius,
    give_armor,
    give_artifact,
    give_body,
    give_key,
    give_power,
    heal_radius,
    morph_monster,
    morph_player,
    player_next_artifact,
    player_remove_artifact,
    player_use_artifact,
    touch_special_thing,
    undo_player_morph,
    use_artifact,
    use_puzzle_item,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    SpriteNum,
    StateNum,
)
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.mobj import Mobj, spawn_map_things, spawn_mobj
from pyhexen.sim.player import Player, PlayerClass
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _make_test_sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 1024, 1024)
    # Player 1 start
    mb.add_thing(tid=0, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    spawn_map_things(sim)
    return sim


def test_give_body_and_armor() -> None:
    player = Player(health=50, player_class=int(PlayerClass.PCLASS_FIGHTER))
    assert give_body(player, 25) is True
    assert player.health == 75

    # Cap at MAXHEALTH
    assert give_body(player, 50) is True
    assert player.health == 100

    # Overcap rejected
    assert give_body(player, 10) is False
    assert player.health == 100

    # Armor increments
    assert give_armor(player, 0, -1) is True
    assert player.armor_points[0] > 0
    # Duplicate same or lower rejected
    assert give_armor(player, 0, -1) is False

    # Armor bonus amount
    prev_armor = player.armor_points[1]
    assert give_armor(player, 1, 1) is True
    assert player.armor_points[1] == prev_armor + 5 * FRACUNIT


def test_keys_and_powers() -> None:
    sim = _make_test_sim()
    pmo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=pmo, health=100)
    pmo.player = player

    # Give key
    assert give_key(player, 0) is True
    assert player.keys & 1
    # Duplicate key
    assert give_key(player, 0) is False

    # Give power: Flight
    assert give_power(sim, player, PowerType.PW_FLIGHT) is True
    assert player.powers[PowerType.PW_FLIGHT] > 0
    assert bool(pmo.flags2 & MobjFlag2.MF2_FLY)
    assert bool(pmo.flags & MobjFlag.MF_NOGRAVITY)

    # Invulnerability
    assert give_power(sim, player, PowerType.PW_INVULNERABILITY) is True
    assert bool(pmo.flags2 & MobjFlag2.MF2_INVULNERABLE)


def test_inventory_storage_and_ordering() -> None:
    player = Player()

    # Pick up non-puzzle artifacts and puzzle items
    assert give_artifact(player, ArtiType.ARTI_PUZZSKULL) is True
    assert player.ready_artifact == ArtiType.ARTI_PUZZSKULL
    assert len(player.inventory) == 1

    # Standard artifact should insert before puzzle artifacts
    assert give_artifact(player, ArtiType.ARTI_HEALTH) is True
    assert player.inventory[0].type == ArtiType.ARTI_HEALTH
    assert player.inventory[1].type == ArtiType.ARTI_PUZZSKULL
    assert player.artifact_count == 2

    # Stacking
    assert give_artifact(player, ArtiType.ARTI_HEALTH) is True
    assert player.inventory[0].count == 2
    assert player.artifact_count == 3

    # Next artifact cycling
    player_next_artifact(player)
    assert player.ready_artifact in (ArtiType.ARTI_HEALTH, ArtiType.ARTI_PUZZSKULL)

    # Removal
    player_remove_artifact(player, 0)
    assert player.inventory[0].count == 1
    player_remove_artifact(player, 0)
    # Slot compacted
    assert len(player.inventory) == 1
    assert player.inventory[0].type == ArtiType.ARTI_PUZZSKULL


def test_touch_special_thing_pickups() -> None:
    sim = _make_test_sim()
    toucher = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=toucher, health=80)
    toucher.player = player

    # Health potion
    potion = spawn_mobj(sim, 0, 0, 0, MobjType.MT_HEALINGBOTTLE)
    potion.sprite = SpriteNum.SPR_PTN1
    touch_special_thing(sim, potion, toucher)
    assert player.health == 90

    # Armor
    arm = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARMOR_1)
    arm.sprite = SpriteNum.SPR_ARM1
    touch_special_thing(sim, arm, toucher)
    assert player.armor_points[0] > 0

    # Artifact (Quartz Flask)
    flask = spawn_mobj(sim, 0, 0, 0, MobjType.MT_HEALTHFLASK)
    flask.sprite = SpriteNum.SPR_PTN2
    touch_special_thing(sim, flask, toucher)
    assert any(slot.type == ArtiType.ARTI_HEALTH for slot in player.inventory)

    # Mana pickup
    mana_mo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_MANA1)
    mana_mo.sprite = SpriteNum.SPR_MAN1
    touch_special_thing(sim, mana_mo, toucher)
    assert player.mana[0] == 15


def test_use_artifacts() -> None:
    sim = _make_test_sim()
    pmo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=pmo, health=50, player_class=int(PlayerClass.PCLASS_FIGHTER))
    pmo.player = player

    give_artifact(player, ArtiType.ARTI_HEALTH)
    assert player_use_artifact(sim, player, ArtiType.ARTI_HEALTH) is True
    assert player.health == 75

    give_artifact(player, ArtiType.ARTI_SUPERHEALTH)
    assert player_use_artifact(sim, player, ArtiType.ARTI_SUPERHEALTH) is True
    assert player.health == 100

    # Wings of Wrath
    give_artifact(player, ArtiType.ARTI_FLY)
    assert player_use_artifact(sim, player, ArtiType.ARTI_FLY) is True
    assert player.powers[PowerType.PW_FLIGHT] > 0

    # Boots of Speed
    give_artifact(player, ArtiType.ARTI_SPEED)
    assert player_use_artifact(sim, player, ArtiType.ARTI_SPEED) is True
    assert player.powers[PowerType.PW_SPEED] > 0

    # Disc of repulsion
    enemy = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_ETTIN)
    enemy.flags |= MobjFlag.MF_COUNTKILL
    blast_radius(sim, player)
    assert enemy.momx > 0 or bool(enemy.flags2 & MobjFlag2.MF2_BLASTED)


def test_flechette_per_class() -> None:
    sim = _make_test_sim()

    # Cleric flechette: MT_POISONBAG
    c_mo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_CLERIC)
    c_player = Player(mo=c_mo, player_class=int(PlayerClass.PCLASS_CLERIC))
    c_mo.player = c_player
    assert use_artifact(sim, c_player, ArtiType.ARTI_POISONBAG) is True
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    assert any(m.type == MobjType.MT_POISONBAG for m in mobjs)

    # Mage flechette: MT_FIREBOMB
    m_mo = spawn_mobj(sim, 100 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_MAGE)
    m_player = Player(mo=m_mo, player_class=int(PlayerClass.PCLASS_MAGE))
    m_mo.player = m_player
    assert use_artifact(sim, m_player, ArtiType.ARTI_POISONBAG) is True
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    assert any(m.type == MobjType.MT_FIREBOMB for m in mobjs)

    # Fighter flechette: MT_THROWINGBOMB
    f_mo = spawn_mobj(sim, 200 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    f_player = Player(mo=f_mo, player_class=int(PlayerClass.PCLASS_FIGHTER))
    f_mo.player = f_player
    assert use_artifact(sim, f_player, ArtiType.ARTI_POISONBAG) is True
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    assert any(m.type == MobjType.MT_THROWINGBOMB for m in mobjs)


def test_heal_radius_per_class() -> None:
    sim = _make_test_sim()
    p1_mo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    p1 = Player(mo=p1_mo, player_class=int(PlayerClass.PCLASS_FIGHTER))
    p1_mo.player = p1

    p2_mo = spawn_mobj(sim, 50 * FRACUNIT, 0, 0, MobjType.MT_PLAYER_CLERIC)
    p2 = Player(mo=p2_mo, player_class=int(PlayerClass.PCLASS_CLERIC), health=50)
    p2_mo.player = p2

    sim.players = [p1, p2]

    # Fighter heal_radius gives armor boost to p2
    assert heal_radius(sim, p1) is True
    assert sum(p2.armor_points) > 0


def test_auto_use_health_on_lethal_damage() -> None:
    sim = _make_test_sim()
    sim.skill = 0  # sk_baby auto-health
    pmo = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=pmo, health=10)
    pmo.player = player
    sim.players = [player]

    # Give Mystic Urn
    give_artifact(player, ArtiType.ARTI_SUPERHEALTH)

    # Lethal damage (50 damage to 10 hp)
    damage_mobj(sim, pmo, None, None, 50)

    # Urn consumed, player survived!
    assert player.health > 0
    assert not any(slot.type == ArtiType.ARTI_SUPERHEALTH for slot in player.inventory)


def test_morph_player_and_undo() -> None:
    sim = _make_test_sim()
    pmo = spawn_mobj(sim, 256 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=pmo, health=100, player_class=int(PlayerClass.PCLASS_FIGHTER))
    pmo.player = player

    # Morph to pig
    assert morph_player(sim, player) is True
    assert player.player_class == int(PlayerClass.PCLASS_PIG)
    assert player.morph_tics > 0
    assert player.mo.type == MobjType.MT_PIGPLAYER

    # Undo morph
    assert undo_player_morph(sim, player) is True
    assert player.player_class == int(PlayerClass.PCLASS_FIGHTER)
    assert player.morph_tics == 0
    assert player.mo.type == MobjType.MT_PLAYER_FIGHTER


def test_morph_monster() -> None:
    sim = _make_test_sim()
    ettin = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
    ettin.flags |= MobjFlag.MF_COUNTKILL

    assert morph_monster(sim, ettin) is True
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    pig = [m for m in mobjs if m.type == MobjType.MT_PIG][0]
    assert pig.special2 == int(MobjType.MT_ETTIN)
    assert pig.special1 > 0


def test_use_puzzle_item() -> None:
    sim = _make_test_sim()
    pmo = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=pmo, player_class=int(PlayerClass.PCLASS_FIGHTER))
    pmo.player = player

    # Face south (270 deg) towards line at y=0
    from pyhexen.core.angles import ANG270

    pmo.angle = ANG270
    pmo.y = 32 * FRACUNIT  # within 64 units of line at y=0

    # Set line special to 129 matching puzzle item 0 (arti_puzzskull)
    line = sim.level.lines[2]
    line.special = 129
    line.args = (0, 1, 0, 0, 0)

    assert use_puzzle_item(sim, player, 0) is True
    assert line.special == 0  # Used!


def test_artifact_actions() -> None:
    sim = _make_test_sim()

    # A_PoisonShroom
    shroom = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ZPOISONSHROOM)
    a_poisonshroom(shroom, sim)
    assert shroom.tics >= 128

    # A_PoisonBagInit & Check & Damage
    bag = spawn_mobj(sim, 0, 0, 0, MobjType.MT_POISONBAG)
    a_poisonbaginit(bag, sim)
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    cloud = [m for m in mobjs if m.type == MobjType.MT_POISONCLOUD][0]
    assert cloud.special1 > 0

    cloud.special1 = 1
    a_poisonbagcheck(cloud, sim)
    assert cloud.state_num == int(StateNum.S_POISONCLOUD_X1)

    # A_CheckThrowBomb
    bomb = spawn_mobj(sim, 0, 0, 0, MobjType.MT_THROWINGBOMB)
    bomb.health = 1
    a_checkthrowbomb(bomb, sim)
    assert bomb.state_num == int(bomb.info.deathstate)

    # A_CheckTeleRing
    ring = spawn_mobj(sim, 0, 0, 0, MobjType.MT_TELOTHER_FX2)
    ring.special1 = 1
    a_checktelering(ring, sim)
    assert ring.state_num == int(ring.info.deathstate)

    # A_SpeedFade
    ghost = spawn_mobj(sim, 0, 0, 0, MobjType.MT_PLAYER_SPEED)
    a_speedfade(ghost, sim)
    assert bool(ghost.flags & MobjFlag.MF_SHADOW)

    # A_RestoreArtifact
    arti = spawn_mobj(sim, 0, 0, 0, MobjType.MT_HEALTHFLASK)
    arti.flags &= ~MobjFlag.MF_SPECIAL
    a_restoreartifact(arti, sim)
    assert bool(arti.flags & MobjFlag.MF_SPECIAL)

    # A_RestoreSpecialThing1 & 2
    thing = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ARMOR_1)
    thing.flags2 |= MobjFlag2.MF2_DONTDRAW
    thing.flags &= ~MobjFlag.MF_SPECIAL
    a_restorespecialthing1(thing, sim)
    assert not (thing.flags2 & MobjFlag2.MF2_DONTDRAW)
    a_restorespecialthing2(thing, sim)
    assert bool(thing.flags & MobjFlag.MF_SPECIAL)

    # A_SoAExplode
    armor_suit = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ZSUITOFARMOR)
    a_soaexplode(armor_suit, sim)
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    assert any(m.type == MobjType.MT_ZARMORCHUNK for m in mobjs)
