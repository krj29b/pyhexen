# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T13.4 Save/load UI, file management, and IWAD checksum validation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.artifacts import ArtiType, give_artifact
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.hub import HubManager
from pyhexen.sim.player import spawn_player
from pyhexen.sim.savegame import (
    NUM_SAVE_SLOTS,
    SaveChecksumError,
    SaveError,
    SaveGameManager,
    SaveVersionError,
    compute_wad_checksum,
)
from pyhexen.sim.specials.doors import DoorType, VerticalDoor
from pyhexen.ui.menu import MenuManager
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _create_synthetic_test_env(tmp_path: Path) -> tuple[WadStack, SaveGameManager]:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)

    mb1 = MapBuilder("MAP01")
    mb1.add_two_sector_rooms()
    mb1.add_thing(tid=1, x=128, y=128, z=0, angle=0, type=1)
    wb.add_map(mb1)

    raw_wad = wb.to_bytes()
    stack = WadStack(raw_wad)
    checksum = compute_wad_checksum(raw_wad)
    save_mgr = SaveGameManager(save_dir=tmp_path / "saves", iwad_checksum=checksum)
    return stack, save_mgr


def test_save_and_load_across_restart(tmp_path: Path) -> None:
    """Acceptance test: saving and loading across a restart works."""
    stack, save_mgr = _create_synthetic_test_env(tmp_path)

    def load_level_cb(map_num: int) -> Level:
        return load_level(stack, f"MAP{map_num:02d}")

    # 1. Initialize session 1 (pre-restart)
    level1 = load_level_cb(1)
    sim1 = GameSim(level=level1)
    sim1.map_num = 1
    sim1.cluster = 1
    player1 = spawn_player(sim1, player_num=0)
    player1.health = 92
    player1.keys = 0b0101
    give_artifact(player1, ArtiType.ARTI_INVULNERABILITY)

    # Move a door
    sec = level1.sectors[1]
    sec.ceiling_height = sec.floor_height
    door = VerticalDoor(sec, DoorType.DREV_NORMAL, speed=2 * FRACUNIT, topwait=150)
    door.direction = 1
    sec.special_data = door
    sim1.thinkers.add(door)

    # Run for 20 tics
    for _ in range(20):
        sim1.tick()

    hub1 = HubManager(current_cluster=1)
    hub1.world_vars = [11, 22, 33]
    hub1.snapshots[2] = b"MAP02_DUMMY_SNAPSHOT"

    pre_save_hash = sim1.state_hash()

    # 2. Save game to slot 1
    save_path = save_mgr.save_game(slot=1, description="Hub Checkpoint", sim=sim1, hub=hub1)
    assert save_path.is_file()

    # Verify slot info
    slots = save_mgr.list_slots()
    assert len(slots) == NUM_SAVE_SLOTS
    assert slots[1].occupied
    assert slots[1].description == "Hub Checkpoint"
    assert slots[1].map_num == 1
    assert not slots[0].occupied

    # 3. Simulate Restart: completely clear in-memory state and create fresh instances
    del sim1, hub1, player1, level1

    new_sim = GameSim()
    new_hub = HubManager()

    # Load game from slot 1
    save_mgr.load_game(
        slot=1,
        sim=new_sim,
        hub=new_hub,
        load_level_cb=load_level_cb,
    )

    # 4. Verify restored state
    assert new_sim.map_num == 1
    assert new_sim.cluster == 1
    assert len(new_sim.players) == 1
    p_restored = new_sim.players[0]
    assert p_restored.health == 92
    assert p_restored.keys == 0b0101
    assert any(slot.type == ArtiType.ARTI_INVULNERABILITY for slot in p_restored.inventory)
    assert p_restored.mo is not None
    assert p_restored.mo.player is p_restored

    # Verify door position and state hash match pre-save
    assert new_sim.level.sectors[1].ceiling_height == sec.ceiling_height
    assert new_sim.state_hash() == pre_save_hash

    # Verify hub state
    assert new_hub.current_cluster == 1
    assert new_hub.world_vars == [11, 22, 33]
    assert new_hub.snapshots.get(2) == b"MAP02_DUMMY_SNAPSHOT"

    # Continue ticking post-load
    for _ in range(10):
        new_sim.tick()


def test_iwad_checksum_validation(tmp_path: Path) -> None:
    """Loading a save created with a different IWAD checksum raises SaveChecksumError."""
    stack, save_mgr = _create_synthetic_test_env(tmp_path)
    sim = GameSim(level=load_level(stack, "MAP01"))
    spawn_player(sim, player_num=0)

    # Save with checksum A
    save_mgr.save_game(slot=0, description="Test", sim=sim, iwad_checksum="CHKSUM_AAA")

    # Attempt to load with checksum B
    fresh_sim = GameSim()
    with pytest.raises(SaveChecksumError, match="IWAD checksum mismatch"):
        save_mgr.load_game(
            slot=0,
            sim=fresh_sim,
            expected_checksum="CHKSUM_BBB",
            validate_checksum=True,
        )


def test_save_version_and_magic_validation(tmp_path: Path) -> None:
    """Corrupted version or magic header raises appropriate SaveError."""
    stack, save_mgr = _create_synthetic_test_env(tmp_path)
    sim = GameSim(level=load_level(stack, "MAP01"))
    spawn_player(sim, player_num=0)

    path = save_mgr.save_game(slot=0, description="Version Test", sim=sim)

    # Read and corrupt version
    with gzip.open(path, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
    data["header"]["version"] = 999
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))

    fresh_sim = GameSim()
    with pytest.raises(SaveVersionError, match="Incompatible save version 999"):
        save_mgr.load_game(slot=0, sim=fresh_sim, validate_checksum=False)

    # Corrupt magic
    data["header"]["version"] = 1
    data["header"]["magic"] = "CORRUPT_MAGIC"
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))

    with pytest.raises(SaveError, match="File is not a valid PyHexen save game"):
        save_mgr.load_game(slot=0, sim=fresh_sim, validate_checksum=False)


def test_menu_ui_synchronization(tmp_path: Path) -> None:
    """MenuManager save/load slots and callbacks sync properly with SaveGameManager."""
    stack, save_mgr = _create_synthetic_test_env(tmp_path)
    level = load_level(stack, "MAP01")
    current_sim = GameSim(level=level)
    current_hub = HubManager(current_cluster=1)
    spawn_player(current_sim, player_num=0)

    menu = MenuManager()

    # Pre-save slot 3
    save_mgr.save_game(slot=3, description="Slot Three Save", sim=current_sim, hub=current_hub)

    # Sync menu
    save_mgr.sync_menu(
        menu=menu,
        sim_getter=lambda: current_sim,
        hub_getter=lambda: current_hub,
        load_level_cb=lambda num: level,
    )

    # Check menu slots synced
    assert menu.slot_texts[3] == "Slot Three Save"
    assert menu.slot_status[3] is True
    assert menu.slot_status[0] is False

    # Simulate saving via menu callback on slot 0
    assert menu.on_save_game is not None
    menu.on_save_game(0, "Menu Saved Slot 0")

    assert menu.slot_texts[0] == "Menu Saved Slot 0"
    assert menu.slot_status[0] is True
    assert save_mgr.get_slot_path(0).is_file()

    # Simulate loading via menu callback on slot 0
    loaded_flag = False

    def on_loaded(sim: GameSim, hub: HubManager) -> None:
        nonlocal loaded_flag
        loaded_flag = True

    save_mgr.sync_menu(
        menu=menu,
        sim_getter=lambda: current_sim,
        hub_getter=lambda: current_hub,
        on_loaded_cb=on_loaded,
        load_level_cb=lambda num: level,
    )

    assert menu.on_load_game is not None
    menu.on_load_game(0)
    assert loaded_flag

    # Delete slot
    assert save_mgr.delete_slot(0)
    assert not save_mgr.get_slot_path(0).exists()
