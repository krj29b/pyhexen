# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for map transitions, carry-over rules, and spawn positioning (T13.1)."""

from __future__ import annotations

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.events import MapExit, SoundEvent
from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.artifacts import ArtiType
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import SfxEnum
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import (
    PW_FLIGHT,
    PW_INVULNERABILITY,
    PW_SPEED,
    InventorySlot,
    Player,
    PlayerClass,
    PlayerState,
    spawn_player,
)
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.sim.transitions import (
    execute_map_transition,
    player_exit_map,
    resolve_target_map,
    spawn_player_at_transition,
)
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _create_test_sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms()
    # Player 1 start in room 1 (arg1=0)
    mb.add_thing(tid=10, x=128, y=128, z=0, angle=0, type=1)
    # Player 1 start for position 2 in room 2 (arg1=2)
    mb.add_thing(tid=11, x=384, y=128, z=0, angle=90, type=1, args=(2, 0, 0, 0, 0))
    # Destination thing with TID 20 in room 2 (type 14 = MT_TELEPORTMAN)
    mb.add_thing(tid=20, x=384, y=128, z=0, angle=180, type=14)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    spawn_map_things(sim)
    return sim


def test_resolve_target_map() -> None:
    """Verify target map resolution with direct numbers, warptrans, and negative numbers."""
    table = MapInfoTable()
    table[1] = MapInfo(map_num=1, name="Seven Portals", lump_name="MAP01", cluster=1, warp_trans=1)
    table[13] = MapInfo(map_num=13, name="Shadow Wood", lump_name="MAP13", cluster=2, warp_trans=7)
    table[27] = MapInfo(map_num=27, name="Necropolis", lump_name="MAP27", cluster=3, warp_trans=13)

    # Endgame / victory preserved
    assert resolve_target_map(-1, table) == -1
    assert resolve_target_map(0, table) == 0

    # Warp translation lookup: warp number 7 -> MAP13, warp number 13 -> MAP27
    assert resolve_target_map(7, table) == 13
    assert resolve_target_map(13, table) == 27

    # Direct map number lookup
    assert resolve_target_map(1, table) == 1

    # Unmapped number fallback
    assert resolve_target_map(99, table) == 99
    assert resolve_target_map(5, None) == 5


def test_teleport_specials_70_and_71() -> None:
    """Verify intra-level teleportation specials 70 (with fog) and 71 (no fog)."""
    sim = _create_test_sim()
    player = spawn_player(sim, player_num=0)
    assert player.mo is not None
    player.mo.x = 0
    player.mo.y = 0

    sound_events: list[SoundEvent] = []
    sim.events.subscribe(SoundEvent, lambda ev: sound_events.append(ev))

    # Special 70: Teleport with fog to TID 20
    ok = execute_line_special(sim, 70, (20, 0, 0, 0, 0), mo=player.mo, side=0)
    assert ok
    dest = sim.find_mobjs_by_tid(20)[0]
    assert (player.mo.x, player.mo.y) == (dest.x, dest.y)
    assert player.mo.reactiontime == 18
    # Sound and fog spawned
    assert any(ev.sound_id == int(SfxEnum.SFX_TELEPORT) for ev in sound_events)

    # Special 71: Teleport without fog
    player.mo.reactiontime = 0
    player.mo.momx = 100 * FRACUNIT
    ok2 = execute_line_special(sim, 71, (20, 0, 0, 0, 0), mo=player.mo, side=0)
    assert ok2
    # Without fog, momentum is not zeroed and reaction time is not set to 18
    assert player.mo.reactiontime == 0

    # Crossing back side (side 1) fails
    assert not execute_line_special(sim, 70, (20, 0, 0, 0, 0), mo=player.mo, side=1)

    # Dead player cannot teleport
    player.player_state = PlayerState.PST_DEAD
    assert not execute_line_special(sim, 70, (20, 0, 0, 0, 0), mo=player.mo, side=0)


def test_teleport_new_map_and_endgame_specials() -> None:
    """Verify specials 74 (Teleport_NewMap), 75 (Teleport_EndGame), 243, and 244."""
    sim = _create_test_sim()
    player = spawn_player(sim, player_num=0)
    assert player.mo is not None

    exits: list[MapExit] = []
    sim.events.subscribe(MapExit, lambda ev: exits.append(ev))

    # Special 74: Teleport_NewMap(map=3, pos=1)
    ok = execute_line_special(sim, 74, (3, 1, 0, 0, 0), mo=player.mo, side=0)
    assert ok
    assert len(exits) == 1
    assert exits[0].next_map == 3
    assert exits[0].position == 1

    # Special 75: Teleport_EndGame in singleplayer
    exits.clear()
    ok = execute_line_special(sim, 75, (0, 0, 0, 0, 0), mo=player.mo, side=0)
    assert ok
    assert len(exits) == 1
    assert exits[0].next_map == -1
    assert exits[0].position == -1

    # Special 75 in deathmatch
    sim.deathmatch = True
    exits.clear()
    ok = execute_line_special(sim, 75, (0, 0, 0, 0, 0), mo=player.mo, side=0)
    assert ok
    assert exits[0].next_map == 1
    assert exits[0].position == 0
    sim.deathmatch = False

    # Specials 243 (Exit_Normal) and 244 (Exit_Secret)
    exits.clear()
    sim.map_num = 2
    execute_line_special(sim, 243, (0, 0, 0, 0, 0), mo=player.mo, side=0)
    assert len(exits) == 1
    assert exits[0].next_map == 3
    assert not exits[0].secret

    exits.clear()
    execute_line_special(sim, 244, (1, 0, 0, 0, 0), mo=player.mo, side=0)
    assert len(exits) == 1
    assert exits[0].secret


def test_player_carryover_same_cluster() -> None:
    """Verify that within the same cluster, keys, flight, and puzzle items are preserved."""
    player = Player(
        player_class=int(PlayerClass.PCLASS_CLERIC),
        health=85,
        armor_points=[5, 10, 15, 20],
        mana=[45, 60],
        keys=0b1011,  # Holds 3 keys
        readyweapon=2,
        weaponowned=[True, True, True, False],
        inventory=[
            InventorySlot(type=int(ArtiType.ARTI_HEALTH), count=3),
            InventorySlot(type=int(ArtiType.ARTI_FLY), count=2),
            InventorySlot(type=int(ArtiType.ARTI_PUZZSKULL), count=1),
        ],
        artifact_count=6,
    )
    player.powers[PW_FLIGHT] = 500
    player.powers[PW_INVULNERABILITY] = 300
    player.lookdir = 15
    player.extralight = 2
    player.fixedcolormap = 1

    player_exit_map(player, old_cluster=1, new_cluster=1)

    # Core stats preserved
    assert player.player_class == int(PlayerClass.PCLASS_CLERIC)
    assert player.health == 85
    assert player.armor_points == [5, 10, 15, 20]
    assert player.mana == [45, 60]
    assert player.readyweapon == 2
    assert player.weaponowned == [True, True, True, False]

    # Keys, puzzle items, flight preserved in same cluster
    assert player.keys == 0b1011
    assert player.powers[PW_FLIGHT] == 500
    assert any(s.type == int(ArtiType.ARTI_FLY) for s in player.inventory)
    assert any(s.type == int(ArtiType.ARTI_PUZZSKULL) for s in player.inventory)
    assert len(player.inventory) == 3

    # Temporary non-flight power stripped
    assert player.powers[PW_INVULNERABILITY] == 0

    # Visual states reset
    assert player.lookdir == 0
    assert player.extralight == 0
    assert player.fixedcolormap == 0


def test_player_carryover_new_cluster() -> None:
    """Verify that across cluster boundaries, keys, flight, and puzzle items are wiped."""
    player = Player(
        player_class=int(PlayerClass.PCLASS_MAGE),
        original_class=int(PlayerClass.PCLASS_MAGE),
        health=72,
        armor_points=[2, 4, 6, 8],
        mana=[100, 50],
        keys=0b111111,  # Multiple keys
        readyweapon=1,
        weaponowned=[True, True, False, False],
        inventory=[
            InventorySlot(type=int(ArtiType.ARTI_SUPERHEALTH), count=2),
            InventorySlot(type=int(ArtiType.ARTI_FLY), count=3),
            InventorySlot(type=int(ArtiType.ARTI_PUZZGEAR1), count=1),
            InventorySlot(type=int(ArtiType.ARTI_TORCH), count=1),
        ],
        artifact_count=7,
        morph_tics=200,
    )
    # Temporarily turned into a pig
    player.player_class = int(PlayerClass.PCLASS_PIG)
    player.powers[PW_FLIGHT] = 1000
    player.powers[PW_SPEED] = 400

    player_exit_map(player, old_cluster=1, new_cluster=2)

    # Stats and class restored
    assert player.player_class == int(PlayerClass.PCLASS_MAGE)
    assert player.morph_tics == 0
    assert player.health == 72
    assert player.armor_points == [2, 4, 6, 8]
    assert player.mana == [100, 50]
    assert player.weaponowned == [True, True, False, False]

    # Keys wiped across cluster change
    assert player.keys == 0

    # Flight power wiped
    assert player.powers[PW_FLIGHT] == 0
    assert player.powers[PW_SPEED] == 0

    # Flight artifact and puzzle item wiped from inventory; standard artifacts kept
    remaining_types = [slot.type for slot in player.inventory]
    assert int(ArtiType.ARTI_SUPERHEALTH) in remaining_types
    assert int(ArtiType.ARTI_TORCH) in remaining_types
    assert int(ArtiType.ARTI_FLY) not in remaining_types
    assert int(ArtiType.ARTI_PUZZGEAR1) not in remaining_types
    assert player.artifact_count == 3


def test_execute_map_transition() -> None:
    """Verify execute_map_transition outcome, cluster change detection, and victory finale."""
    sim = _create_test_sim()
    sim.map_num = 1
    player = spawn_player(sim, player_num=0)
    player.keys = 0b11

    table = MapInfoTable()
    table[1] = MapInfo(map_num=1, name="Hub 1 A", lump_name="MAP01", cluster=1, warp_trans=1)
    table[2] = MapInfo(map_num=2, name="Hub 1 B", lump_name="MAP02", cluster=1, warp_trans=2)
    table[13] = MapInfo(map_num=13, name="Hub 2 A", lump_name="MAP13", cluster=2, warp_trans=7)

    # Transition to Map 2 (same cluster)
    res1 = execute_map_transition(sim, target_map=2, position=0, mapinfo_table=table)
    assert not res1.cluster_changed
    assert res1.target_map == 2
    assert not res1.victory
    assert player.keys == 0b11

    # Transition to Map 13 (different cluster). The target is a real map number,
    # never a warp number: G_Completed(args[0], ...) does no translation.
    res2 = execute_map_transition(sim, target_map=13, position=1, mapinfo_table=table)
    assert res2.cluster_changed
    assert res2.target_map == 13
    assert res2.position == 1
    assert not res2.victory
    assert player.keys == 0  # Keys cleared between clusters

    # Victory / endgame transition (-1, -1)
    res_vic = execute_map_transition(sim, target_map=-1, position=-1, mapinfo_table=table)
    assert res_vic.victory
    assert sim.paused


def test_spawn_player_at_transition_position() -> None:
    """Verify player spawn positioning at destination map spot."""
    sim = _create_test_sim()
    player = spawn_player(sim, player_num=0)
    assert player.mo is not None

    # Spawn at position 2 (arg1=2 start spot at x=384, y=128, angle=90)
    spawn_player_at_transition(sim, player, player_num=0, position=2)

    assert player.mo.x == 384 * FRACUNIT
    assert player.mo.y == 128 * FRACUNIT
    assert player.player_state == PlayerState.PST_LIVE


def test_game_timer_triggers_exit() -> None:
    """Verify that sim.timer_game expiring triggers an exit_normal MapExit event."""
    sim = _create_test_sim()
    sim.map_num = 4
    sim.timer_game = 2

    exits: list[MapExit] = []
    sim.events.subscribe(MapExit, lambda ev: exits.append(ev))

    sim.tick()
    assert len(exits) == 0
    assert sim.timer_game == 1

    sim.tick()
    assert len(exits) == 1
    assert exits[0].next_map == 5
