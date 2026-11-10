# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T13.3 Hub persistence and level snapshot management."""

from __future__ import annotations

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.artifacts import ArtiType, give_artifact
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.hub import HubManager
from pyhexen.sim.player import PW_FLIGHT, spawn_player
from pyhexen.sim.specials.doors import DoorType, VerticalDoor
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _build_synthetic_hub_wad() -> tuple[WadStack, MapInfoTable]:
    """Build a synthetic WAD containing MAP01, MAP02 (Cluster 1) and MAP03 (Cluster 2)."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)

    # MAP01: two-room layout with a doorway sector between them
    mb1 = MapBuilder("MAP01")
    mb1.add_two_sector_rooms()
    mb1.add_thing(tid=1, x=128, y=128, z=0, angle=0, type=1)  # Player start 0
    wb.add_map(mb1)

    # MAP02: single room
    mb2 = MapBuilder("MAP02")
    mb2.add_room(x=0, y=0, width=512, height=512)
    mb2.add_thing(tid=1, x=256, y=256, z=0, angle=0, type=1)  # Player start 0
    wb.add_map(mb2)

    # MAP03: new cluster room
    mb3 = MapBuilder("MAP03")
    mb3.add_room(x=0, y=0, width=512, height=512)
    mb3.add_thing(tid=1, x=256, y=256, z=0, angle=0, type=1)  # Player start 0
    wb.add_map(mb3)

    raw_wad = wb.to_bytes()
    stack = WadStack(raw_wad)

    mapinfo = MapInfoTable(
        {
            1: MapInfo(map_num=1, name="MAP01", lump_name="MAP01", cluster=1),
            2: MapInfo(map_num=2, name="MAP02", lump_name="MAP02", cluster=1),
            3: MapInfo(map_num=3, name="MAP03", lump_name="MAP03", cluster=2),
        }
    )

    return stack, mapinfo


def test_two_map_hub_door_left_open_stays_open() -> None:
    """Acceptance: synthetic two-map hub checks that a door left open stays open."""
    wad, mapinfo = _build_synthetic_hub_wad()

    def load_level_cb(map_num: int) -> Level:
        return load_level(wad, f"MAP{map_num:02d}")

    hub = HubManager(current_cluster=1)

    # 1. Start in MAP01
    level1 = load_level_cb(1)
    sim1 = GameSim(level=level1)
    sim1.map_num = 1
    sim1.cluster = 1
    spawn_player(sim1, player_num=0)

    # Door is in sector 1; starts closed
    door_sec = level1.sectors[1]
    door_sec.ceiling_height = door_sec.floor_height
    closed_height = door_sec.ceiling_height

    # Raise door
    door = VerticalDoor(door_sec, DoorType.DREV_NORMAL, speed=2 * FRACUNIT, topwait=150)
    door.topheight = 128 * FRACUNIT
    door.direction = 1
    door_sec.special_data = door
    sim1.thinkers.add(door)

    # Advance simulation until door is open
    for _ in range(40):
        sim1.tick()
    open_height = door_sec.ceiling_height
    assert open_height > closed_height

    # 2. Transition from MAP01 to MAP02 (same cluster)
    sim2, res1 = hub.transition_map(
        sim1,
        target_map=2,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )
    assert res1.target_map == 2
    assert not res1.cluster_changed
    assert sim2.map_num == 2
    assert hub.has_map(1)  # MAP01 is stored in hub snapshot

    # While in MAP02, player explores and queues a deferred script for MAP01
    sim2.acs_store.queue_script(map_num=1, script_num=5, args=(42, 84, 126))
    assert len(sim2.acs_store.entries) == 1

    # Advance MAP02 simulation
    for _ in range(15):
        sim2.tick()

    # 3. Transition back from MAP02 to MAP01 (re-entry)
    sim1_back, res2 = hub.transition_map(
        sim2,
        target_map=1,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )
    assert res2.target_map == 1
    assert not res2.cluster_changed
    assert sim1_back.map_num == 1
    assert hub.has_map(2)  # MAP02 was snapshotted on leaving

    # CRITICAL ACCEPTANCE CHECK: The door left open in MAP01 stays open!
    restored_door_sec = sim1_back.level.sectors[1]
    assert restored_door_sec.ceiling_height == open_height
    assert restored_door_sec.ceiling_height > closed_height

    # Check that deferred script for MAP01 was processed
    assert len(sim1_back.acs_store.entries) == 0


def test_hub_monster_state_preserved_across_hub_reentry() -> None:
    """Damage inflicted on monsters in a level persists when returning to that level in a hub."""
    wad, mapinfo = _build_synthetic_hub_wad()

    def load_level_cb(map_num: int) -> Level:
        return load_level(wad, f"MAP{map_num:02d}")

    hub = HubManager(current_cluster=1)

    sim1 = GameSim(level=load_level_cb(1))
    sim1.map_num = 1
    sim1.cluster = 1
    spawn_player(sim1, player_num=0)

    # Spawn an Ettin with tid 55
    from pyhexen.sim.mobj import Mobj

    ettin = Mobj()
    ettin.tid = 55
    ettin.health = 45  # Damaged
    sim1.thinkers.add(ettin)
    sim1.tid_map[55] = [ettin]

    # Travel to MAP02
    sim2, _ = hub.transition_map(
        sim1,
        target_map=2,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )

    # Travel back to MAP01
    sim1_back, _ = hub.transition_map(
        sim2,
        target_map=1,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )

    restored_ettin = next(t for t in sim1_back.thinkers if isinstance(t, Mobj) and t.tid == 55)
    assert restored_ettin.health == 45


def test_cluster_change_clears_hub_and_applies_rules() -> None:
    """Changing clusters clears hub map snapshots, clears ACS store, and strips keys/flight."""
    wad, mapinfo = _build_synthetic_hub_wad()

    def load_level_cb(map_num: int) -> Level:
        return load_level(wad, f"MAP{map_num:02d}")

    hub = HubManager(current_cluster=1)

    # 1. Start in MAP01 (Cluster 1)
    sim1 = GameSim(level=load_level_cb(1))
    sim1.map_num = 1
    sim1.cluster = 1
    player = spawn_player(sim1, player_num=0)
    player.keys = 0b0111  # 3 keys
    player.powers[PW_FLIGHT] = 1000
    give_artifact(player, ArtiType.ARTI_FLY)
    give_artifact(player, ArtiType.ARTI_INVULNERABILITY)

    # Queue an ACS store script for MAP99 (deferred for another map in the cluster)
    sim1.acs_store.queue_script(map_num=99, script_num=10, args=(1, 2, 3))

    # Move to MAP02 (same cluster) to establish hub snapshots
    sim2, res1 = hub.transition_map(
        sim1,
        target_map=2,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )
    assert hub.has_map(1)
    # The script for MAP99 remains in the store across same-cluster transitions
    assert len(sim2.acs_store.entries) == 1

    # 2. Transition from MAP02 (Cluster 1) to MAP03 (Cluster 2)
    sim3, res2 = hub.transition_map(
        sim2,
        target_map=3,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )
    assert res2.cluster_changed
    assert res2.new_cluster == 2
    assert hub.current_cluster == 2

    # Hub snapshots MUST be cleared upon entering a new cluster
    assert not hub.has_map(1)
    assert not hub.has_map(2)

    # ACS store MUST be cleared on cluster change
    assert len(sim3.acs_store.entries) == 0

    # Cluster carry-over rules: keys and flight power are stripped, standard items kept
    p3 = sim3.players[0]
    assert p3.keys == 0
    assert p3.powers[PW_FLIGHT] == 0
    # Flight artifact stripped
    assert not any(slot.type == ArtiType.ARTI_FLY for slot in p3.inventory)
    # Invulnerability artifact kept
    assert any(slot.type == ArtiType.ARTI_INVULNERABILITY for slot in p3.inventory)


def _fresh_setup_rng_state(wad: WadStack, map_num: int) -> tuple[int, int]:
    """RNG state right after a clean level setup (G_InitNew + P_SetupLevel)."""
    sim = GameSim(level=load_level(wad, f"MAP{map_num:02d}"))
    sim.rng.p_random()  # make sure setup_level really clears it
    sim.setup_level()
    return sim.rng.get_state()


def test_map_entry_rng_and_revisit_player_start_match_vanilla() -> None:
    """New maps clear the RNG (G_InitNew); revisits do not restore the archived RNG
    (SV_LoadMap does not archive it) and still know the map's player starts."""
    wad, mapinfo = _build_synthetic_hub_wad()

    def load_level_cb(map_num: int) -> Level:
        return load_level(wad, f"MAP{map_num:02d}")

    hub = HubManager(current_cluster=1)
    sim1 = GameSim(level=load_level_cb(1))
    sim1.map_num = 1
    sim1.setup_level(player_nums=(0,))
    for _ in range(50):
        sim1.rng.p_random()

    sim2, _ = hub.transition_map(
        sim1, target_map=2, position=0, mapinfo_table=mapinfo, load_level_cb=load_level_cb
    )
    assert sim2.rng.get_state() == _fresh_setup_rng_state(wad, 2)
    for _ in range(30):
        sim2.rng.p_random()

    sim1_back, _ = hub.transition_map(
        sim2, target_map=1, position=0, mapinfo_table=mapinfo, load_level_cb=load_level_cb
    )
    assert sim1_back.rng.get_state() == _fresh_setup_rng_state(wad, 1)
    assert (0, 0) in sim1_back.player_starts
    mo = sim1_back.players[0].mo
    assert mo is not None
    assert (mo.x // FRACUNIT, mo.y // FRACUNIT) == (128, 128)
    # Only the restored thinkers remain: nothing from the throwaway setup is linked.
    assert mo in sim1_back.thinkers
