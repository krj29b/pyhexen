# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for status bar, inventory bar, and HUD overlay (T12.2)."""

from __future__ import annotations

import numpy as np

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.events import EventBus
from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.sim.artifacts import ArtiType, PowerType
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag2, MobjType
from pyhexen.sim.mobj import spawn_map_things, spawn_mobj
from pyhexen.sim.player import InventorySlot, Player, PlayerClass
from pyhexen.ui.fonts import FontSet
from pyhexen.ui.statusbar import (
    INVENTORY_TIMEOUT_TICS,
    NUMBONUSPALS,
    NUMPOISONPALS,
    NUMREDPALS,
    STARTBONUSPALS,
    STARTICEPAL,
    STARTPOISONPALS,
    STARTREDPALS,
    StatusBar,
    get_sb_palette,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _make_test_sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 1024, 1024)
    mb.add_thing(tid=0, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    sim.rng = Random(0x12345678)
    sim.events = EventBus()
    spawn_map_things(sim)
    return sim


def test_statusbar_init_and_fallback_patches() -> None:
    sb = StatusBar()
    assert sb.health_marker == 100
    assert sb.inventory is False
    assert sb.curpos == 0
    assert sb.inv_ptr == 0

    # Fallback patch retrieval
    h2top = sb.get_patch("H2TOP")
    assert h2top.width == 320
    assert h2top.height == 66

    statbar = sb.get_patch("STATBAR")
    assert statbar.width == 244
    assert statbar.height == 38

    chain = sb.get_patch("CHAIN")
    assert chain.width == 196
    assert chain.height == 7


def test_health_marker_ticker() -> None:
    sb = StatusBar()
    player = Player(health=50)

    # Health decreases from 100 to 50 smoothly
    sb.tick(player)
    assert sb.health_marker < 100
    assert sb.health_marker >= 94  # clamped max delta is 6

    # Multiple ticks should reach 50
    for _ in range(50):
        sb.tick(player)
    assert sb.health_marker == 50

    # Health increases from 50 to 90
    player.health = 90
    sb.tick(player)
    assert sb.health_marker > 50
    assert sb.health_marker <= 56


def test_draw_common_bar() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    sb = StatusBar()

    sb.draw_common_bar(overlay)
    # Status bar area (y >= 134) should have non-zero pixels
    assert np.any(overlay.mask[134:, :])
    assert np.any(overlay.indices[134:, :])


def test_draw_main_bar_health_and_mana() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    sb = StatusBar()
    player = Player(
        health=85,
        mana=[150, 75],
        player_class=int(PlayerClass.PCLASS_FIGHTER),
    )
    player.readyweapon = 1  # Timon's Axe (Mana 1 active)

    sb.draw_main_bar(overlay, player)
    # Check that mask has drawn pixels in the status bar area
    assert np.any(overlay.mask[162:198, 38:282])

    # Red health test (< 25)
    overlay.clear()
    player.health = 15
    sb.health_marker = 15
    sb.draw_main_bar(overlay, player)
    assert np.any(overlay.mask[176:192, 40:60])


def test_draw_weapon_pieces() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    sb = StatusBar()
    player = Player(
        pieces=3,  # Pieces 1 and 2
        player_class=int(PlayerClass.PCLASS_CLERIC),
    )

    sb.draw_weapon_pieces(overlay, player)
    assert np.any(overlay.mask[162:194, 190:240])

    # Full weapon assembled
    overlay.clear()
    player.pieces = 7
    sb.draw_weapon_pieces(overlay, player)
    assert np.any(overlay.mask[162:194, 190:240])


def test_inventory_navigation_and_timeout() -> None:
    sb = StatusBar()
    player = Player()
    player.inventory = [
        InventorySlot(type=ArtiType.ARTI_HEALTH, count=5),
        InventorySlot(type=ArtiType.ARTI_SUPERHEALTH, count=2),
        InventorySlot(type=ArtiType.ARTI_TORCH, count=1),
        InventorySlot(type=ArtiType.ARTI_BLASTRADIUS, count=3),
        InventorySlot(type=ArtiType.ARTI_SPEED, count=1),
        InventorySlot(type=ArtiType.ARTI_FLY, count=2),
        InventorySlot(type=ArtiType.ARTI_POISONBAG, count=4),
        InventorySlot(type=ArtiType.ARTI_BOOSTMANA, count=1),
    ]

    # Open inventory
    sb.toggle_inventory()
    assert sb.inventory is True
    assert sb.inventory_tics == INVENTORY_TIMEOUT_TICS

    # Navigate right through inventory
    sb.inventory_right(player)
    assert sb.inv_ptr == 1
    assert sb.curpos == 1
    assert player.ready_artifact == ArtiType.ARTI_SUPERHEALTH

    # Scroll past 7 slots
    for _ in range(8):
        sb.inventory_right(player)
    assert sb.inv_ptr == 7
    assert sb.curpos == 6

    # Navigate left back to slot 0
    for _ in range(10):
        sb.inventory_left(player)
    assert sb.inv_ptr == 0
    assert sb.curpos == 0
    assert player.ready_artifact == ArtiType.ARTI_HEALTH

    # Test auto-close timeout
    sb.inventory_tics = 1
    sb.tick(player)
    assert sb.inventory is False


def test_inventory_use_artifact() -> None:
    sim = _make_test_sim()
    pmo = spawn_mobj(sim, 256 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=pmo, health=50, player_class=int(PlayerClass.PCLASS_FIGHTER))
    pmo.player = player
    player.inventory = [InventorySlot(type=ArtiType.ARTI_HEALTH, count=2)]

    sb = StatusBar()
    sb.inv_ptr = 0

    # Using Quartz Flask restores 25 health and starts flash
    success = sb.use_selected_artifact(sim, player)
    assert success is True
    assert player.health == 75
    assert player.inventory[0].count == 1
    assert sb.artifact_flash == 4


def test_draw_inventory_bar() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    sb = StatusBar()
    sb.inventory = True
    player = Player()
    player.inventory = [
        InventorySlot(type=ArtiType.ARTI_HEALTH, count=3),
        InventorySlot(type=ArtiType.ARTI_TORCH, count=1),
    ]

    sb.draw_inventory_bar(overlay, player, level_time=10)
    assert np.any(overlay.mask[162:198, 38:282])


def test_draw_key_bar() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    sb = StatusBar()
    sb.automap_active = True
    player = Player(
        keys=0b101,  # Key 1 and Key 3
        armor_points=[50 * FRACUNIT, 25 * FRACUNIT, 0, 0],
    )

    sb.draw_key_bar(overlay, player)
    assert np.any(overlay.mask[162:198, 38:282])


def test_fullscreen_hud() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    sb = StatusBar()
    sb.fullscreen = True
    player = Player(
        health=100,
        ready_artifact=int(ArtiType.ARTI_HEALTH),
        inventory=[InventorySlot(type=ArtiType.ARTI_HEALTH, count=3)],
    )

    sb.draw(overlay, player, level_time=0)
    # Large health number at bottom left
    assert np.any(overlay.mask[175:200, 0:50])
    # Ready artifact icon at bottom right
    assert np.any(overlay.mask[165:200, 270:320])

    # Floating inventory in fullscreen mode
    overlay.clear()
    sb.inventory = True
    sb.draw(overlay, player, level_time=0)
    assert np.any(overlay.mask[165:200, 45:280])


def test_animated_icons() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    sb = StatusBar()
    player = Player()
    # Activate Wings of Wrath and Boots of Speed
    player.powers[PowerType.PW_FLIGHT] = 1000
    player.powers[PowerType.PW_SPEED] = 1000

    sb.draw_animated_icons(overlay, player, level_time=15)
    # Flight icon at (20, 19)
    assert np.any(overlay.mask[19:51, 20:52])
    # Speed icon at (60, 19)
    assert np.any(overlay.mask[19:51, 60:92])


def test_palette_flash() -> None:
    player = Player()
    assert get_sb_palette(player) == 0

    # Damage flash
    player.damagecount = 10
    pal = get_sb_palette(player)
    assert STARTREDPALS <= pal < STARTREDPALS + NUMREDPALS

    # Bonus flash
    player.damagecount = 0
    player.bonuscount = 15
    pal = get_sb_palette(player)
    assert STARTBONUSPALS <= pal < STARTBONUSPALS + NUMBONUSPALS

    # Poison flash
    player.bonuscount = 0
    player.poisoncount = 20
    pal = get_sb_palette(player)
    assert STARTPOISONPALS <= pal < STARTPOISONPALS + NUMPOISONPALS

    # Ice frozen flash
    player.poisoncount = 0
    pmo = spawn_mobj(_make_test_sim(), 0, 0, 0, MobjType.MT_PLAYER_FIGHTER)
    pmo.flags2 |= MobjFlag2.MF2_ICEDAMAGE
    player.mo = pmo
    assert get_sb_palette(player) == STARTICEPAL
