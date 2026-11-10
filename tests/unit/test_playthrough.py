# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for full playthrough verification and hub progression (T14.5, Milestone M6)."""

from __future__ import annotations

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.hub import HubManager
from pyhexen.sim.mobj import spawn_map_things
from pyhexen.sim.player import spawn_player
from pyhexen.wad.mapinfo import MapInfo, MapInfoTable
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _build_multi_hub_synthetic_wad() -> tuple[WadStack, MapInfoTable]:
    """Build a multi-hub synthetic WAD with 3 maps across 2 clusters."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)

    # MAP01: Hub 1 Room
    mb1 = MapBuilder("MAP01")
    mb1.add_two_sector_rooms(light1=120, light2=180)
    mb1.add_thing(tid=1, x=128, y=128, z=0, angle=0, type=1)
    wb.add_map(mb1)

    # MAP02: Hub 1 Secondary
    mb2 = MapBuilder("MAP02")
    mb2.add_room(x=0, y=0, width=512, height=512)
    mb2.add_polyobject_box(1, -2000, -2000, 100, 100)
    mb2.add_thing(tid=1, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb2)

    # MAP03: Hub 2 (Cluster 2) Room
    mb3 = MapBuilder("MAP03")
    mb3.add_room(x=0, y=0, width=512, height=512)
    mb3.add_thing(tid=1, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb3)

    raw_bytes = wb.to_bytes()
    stack = WadStack(raw_bytes)
    mapinfo = MapInfoTable(
        {
            1: MapInfo(map_num=1, name="HUB1_MAIN", lump_name="MAP01", cluster=1),
            2: MapInfo(map_num=2, name="HUB1_SUB", lump_name="MAP02", cluster=1),
            3: MapInfo(map_num=3, name="HUB2_MAIN", lump_name="MAP03", cluster=2),
        }
    )
    return stack, mapinfo


def test_full_playthrough_all_classes_sim() -> None:
    """Verify that all 3 player classes (Fighter, Cleric, Mage) can spawn and simulate."""
    stack, _ = _build_multi_hub_synthetic_wad()
    level = load_level(stack, "MAP01")

    for pclass in [0, 1, 2]:
        sim = GameSim(level=level, skill=2, player_class=pclass)
        spawn_map_things(sim)
        player = spawn_player(sim, 0)
        assert player is not None
        assert player.player_class == pclass

        # Advance 35 tics
        for _ in range(35):
            sim.tick()
        assert sim.level_time == 35


def test_hub_progression_and_cluster_boundary_transitions() -> None:
    """Verify in-hub round trip and cluster boundary transition."""
    stack, mapinfo = _build_multi_hub_synthetic_wad()

    def load_level_cb(map_num: int) -> Level:
        return load_level(stack, f"MAP{map_num:02d}")

    hub = HubManager(current_cluster=1)

    # 1. Start in MAP01 (Cluster 1)
    level1 = load_level_cb(1)
    sim = GameSim(level=level1, skill=2, player_class=0)
    sim.map_num = 1
    sim.cluster = 1
    spawn_map_things(sim)
    spawn_player(sim, 0)

    # 2. In-hub transition to MAP02
    sim2, res1 = hub.transition_map(
        sim,
        target_map=2,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )
    assert res1.target_map == 2
    assert not res1.cluster_changed
    assert sim2.cluster == 1
    assert hub.has_map(1)

    # Advance 35 tics in MAP02
    for _ in range(35):
        sim2.tick()

    # 3. Return to MAP01 (state restored from snapshot)
    sim3, res2 = hub.transition_map(
        sim2,
        target_map=1,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )
    assert res2.target_map == 1
    assert not res2.cluster_changed
    assert hub.has_map(2)

    # 4. Cluster boundary transition to MAP03 (Cluster 2)
    sim4, res3 = hub.transition_map(
        sim3,
        target_map=3,
        position=0,
        mapinfo_table=mapinfo,
        load_level_cb=load_level_cb,
    )
    assert res3.target_map == 3
    assert res3.cluster_changed
    assert sim4.cluster == 2
    # Cluster change clears old cluster snapshots
    assert not hub.has_map(1)
    assert not hub.has_map(2)
