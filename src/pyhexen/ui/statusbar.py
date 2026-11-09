# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Status bar and inventory bar system (T12.2).

Port of SB_Init, SB_Ticker, SB_Drawer, SB_Responder, and SB_PaletteFlash (sb_bar.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.sim.artifacts import (
    BLINKTHRESHOLD,
    PowerType,
    player_use_artifact,
)
from pyhexen.sim.info_data import MobjFlag2
from pyhexen.sim.inter import (
    ARMOR_AMULET,
    ARMOR_ARMOR,
    ARMOR_HELMET,
    ARMOR_SHIELD,
    AUTO_ARMOR_SAVE,
    NUMARMOR,
)
from pyhexen.sim.player import Player
from pyhexen.ui.fonts import FontSet
from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.wadfile import WadFile, WadStack

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

# Constants from p_local.h / sb_bar.c
STARTREDPALS: int = 1
NUMREDPALS: int = 8
STARTBONUSPALS: int = 9
NUMBONUSPALS: int = 4
STARTPOISONPALS: int = 13
NUMPOISONPALS: int = 8
STARTICEPAL: int = 21

INVENTORY_TIMEOUT_TICS: int = 5 * 35  # 5 seconds auto-close
MAX_MANA: int = 200

# 4th Weapon piece bitmasks and x-offsets by class
WPIECE1: int = 1
WPIECE2: int = 2
WPIECE3: int = 4

PIECE_X: tuple[tuple[int, int, int], ...] = (
    (190, 225, 234),  # Fighter
    (190, 212, 225),  # Cleric
    (190, 205, 224),  # Mage
    (0, 0, 0),  # Pig
)

# Artifact lump name mapping (sb_bar.c:patcharti)
PATCH_ARTI: tuple[str, ...] = (
    "ARTIBOX",  # 0: none
    "ARTIINVU",  # 1: invulnerability
    "ARTIPTN2",  # 2: health
    "ARTISPHL",  # 3: superhealth
    "ARTIHRAD",  # 4: healing radius
    "ARTISUMN",  # 5: summon maulator
    "ARTITRCH",  # 6: torch
    "ARTIPORK",  # 7: egg (porkalator)
    "ARTISOAR",  # 8: fly
    "ARTIBLST",  # 9: blast radius
    "ARTIPSBG",  # 10: poison bag
    "ARTITELO",  # 11: teleport other
    "ARTISPED",  # 12: speed
    "ARTIBMAN",  # 13: boost mana
    "ARTIBRAC",  # 14: boost armor
    "ARTIATLP",  # 15: teleport
    "ARTISKLL",  # 16: arti_puzzskull
    "ARTIBGEM",  # 17: arti_puzzgembig
    "ARTIGEMR",  # 18: arti_puzzgemred
    "ARTIGEMG",  # 19: arti_puzzgemgreen1
    "ARTIGMG2",  # 20: arti_puzzgemgreen2
    "ARTIGEMB",  # 21: arti_puzzgemblue1
    "ARTIGMB2",  # 22: arti_puzzgemblue2
    "ARTIBOK1",  # 23: arti_puzzbook1
    "ARTIBOK2",  # 24: arti_puzzbook2
    "ARTISKL2",  # 25: arti_puzzskull2
    "ARTIFWEP",  # 26: arti_puzzfweapon
    "ARTICWEP",  # 27: arti_puzzcweapon
    "ARTIMWEP",  # 28: arti_puzzmweapon
    "ARTIGEAR",  # 29: arti_puzzgear1
    "ARTIGER2",  # 30: arti_puzzgear2
    "ARTIGER3",  # 31: arti_puzzgear3
    "ARTIGER4",  # 32: arti_puzzgear4
)


def _make_synthetic_picture(
    width: int,
    height: int,
    color: int = 1,
    *,
    left: int = 0,
    top: int = 0,
) -> Picture:
    """Create a placeholder picture for synthetic / headless tests."""
    indices = np.full((height, width), color, dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    return Picture(
        width=width,
        height=height,
        left=left,
        top=top,
        indices=indices,
        mask=mask,
    )


def get_sb_palette(player: Player) -> int:
    """Calculate the active palette flash index (0..27).

    Port of SB_PaletteFlash (sb_bar.c:892).
    """
    if player.poisoncount > 0:
        pal = (player.poisoncount + 7) >> 3
        if pal >= NUMPOISONPALS:
            pal = NUMPOISONPALS - 1
        return pal + STARTPOISONPALS
    elif player.damagecount > 0:
        pal = (player.damagecount + 7) >> 3
        if pal >= NUMREDPALS:
            pal = NUMREDPALS - 1
        return pal + STARTREDPALS
    elif player.bonuscount > 0:
        pal = (player.bonuscount + 7) >> 3
        if pal >= NUMBONUSPALS:
            pal = NUMBONUSPALS - 1
        return pal + STARTBONUSPALS
    elif player.mo is not None and bool(player.mo.flags2 & MobjFlag2.MF2_ICEDAMAGE):
        return STARTICEPAL
    return 0


class StatusBar:
    """Hexen status bar, inventory bar, and HUD overlay manager.

    Port of `sb_bar.c`. Manages health chain animation, mana vials,
    armor points, fourth weapon pieces, inventory selection, fullscreen HUD,
    animated power icons, and palette flash feedback.
    """

    def __init__(
        self,
        wad: WadStack | WadFile | None = None,
        fonts: FontSet | None = None,
    ) -> None:
        self.wad: WadStack | WadFile | None = wad
        self.fonts: FontSet = FontSet.from_wad(wad) if fonts is None else fonts

        self.health_marker: int = 100
        self.inventory: bool = False
        self.curpos: int = 0
        self.inv_ptr: int = 0
        self.inventory_tics: int = 0
        self.artifact_flash: int = 0

        self.fullscreen: bool = False
        self.automap_active: bool = False

        self._patch_cache: dict[str, Picture] = {}

    def get_patch(self, name: str) -> Picture:
        """Fetch and cache a decoded `Picture` patch, falling back to a synthetic one."""
        key = name.upper()
        if key in self._patch_cache:
            return self._patch_cache[key]

        pic: Picture | None = None
        if self.wad is not None:
            try:
                data = self.wad.read_lump(key)
                pic = decode_picture(data)
            except Exception:
                pic = None

        if pic is None:
            pic = self._make_fallback_patch(key)

        self._patch_cache[key] = pic
        return pic

    def _make_fallback_patch(self, name: str) -> Picture:
        """Generate a proportional synthetic picture when a lump is missing."""
        n = name.upper()
        if n in ("H2TOP", "H2BAR"):
            return _make_synthetic_picture(320, 66, 16)
        if n in ("STATBAR", "KEYBAR", "INVBAR"):
            return _make_synthetic_picture(244, 38, 24)
        if n in ("CHAIN",):
            return _make_synthetic_picture(196, 7, 32)
        if n in ("LIFEGEM",):
            return _make_synthetic_picture(16, 7, 176)
        if n in ("LFEDGE",):
            return _make_synthetic_picture(28, 7, 20)
        if n in ("RTEDGE",):
            return _make_synthetic_picture(43, 7, 20)
        if n.startswith("MANAVIAL"):
            return _make_synthetic_picture(8, 24, 180)
        if n.startswith("MANA"):
            return _make_synthetic_picture(16, 16, 190)
        if n.startswith("INRED"):
            return _make_synthetic_picture(8, 16, 175)
        if n.startswith("IN") and len(n) == 3 and n[2].isdigit():
            return _make_synthetic_picture(8, 16, 220)
        if n.startswith("SMALLIN"):
            return _make_synthetic_picture(4, 7, 210)
        if n in ("SELECTBOX", "WEAPONSLOT", "WEAPONFULL"):
            return _make_synthetic_picture(32, 32, 230)
        if n.startswith("PIECE"):
            return _make_synthetic_picture(15, 30, 200)
        if n.startswith("KEYSLOT") or n.startswith("ARMSLOT"):
            return _make_synthetic_picture(16, 16, 205)
        if (
            n.startswith("SPFLY")
            or n.startswith("SPED")
            or n.startswith("DEFN")
            or n.startswith("MINO")
        ):
            return _make_synthetic_picture(32, 32, 195)
        if n.startswith("USEARTI"):
            return _make_synthetic_picture(32, 32, 245)
        if n.startswith("ARTI"):
            return _make_synthetic_picture(32, 32, 160)
        return _make_synthetic_picture(16, 16, 128)

    # =========================================================================
    # Ticker & State Update
    # =========================================================================

    def tick(self, player: Player) -> None:
        """Advance status bar timers and interpolate health display (port of SB_Ticker)."""
        cur_health = max(0, player.health)
        if cur_health < self.health_marker:
            delta = (self.health_marker - cur_health) >> 2
            delta = max(1, min(6, delta))
            self.health_marker -= delta
        elif cur_health > self.health_marker:
            delta = (cur_health - self.health_marker) >> 2
            delta = max(1, min(6, delta))
            self.health_marker += delta

        if self.artifact_flash > 0:
            self.artifact_flash -= 1

        if self.inventory_tics > 0:
            self.inventory_tics -= 1
            if self.inventory_tics <= 0:
                self.inventory = False

        # Constrain inventory pointers
        inv_len = len(player.inventory)
        if inv_len == 0:
            self.inv_ptr = 0
            self.curpos = 0
        else:
            if self.inv_ptr >= inv_len:
                self.inv_ptr = inv_len - 1
            if self.curpos > self.inv_ptr:
                self.curpos = self.inv_ptr
            if self.curpos > 6:
                self.curpos = 6

    # =========================================================================
    # Input Handling / Responder
    # =========================================================================

    def toggle_inventory(self) -> None:
        """Toggle inventory bar visibility."""
        self.inventory = not self.inventory
        if self.inventory:
            self.inventory_tics = INVENTORY_TIMEOUT_TICS

    def inventory_left(self, player: Player) -> None:
        """Move inventory selection cursor left."""
        self.inventory_tics = INVENTORY_TIMEOUT_TICS
        if not self.inventory:
            self.inventory = True
            return

        inv_len = len(player.inventory)
        if inv_len == 0:
            return

        if self.inv_ptr > 0:
            self.inv_ptr -= 1
            if self.inv_ptr < 6:
                self.curpos = max(0, self.curpos - 1)
        player.ready_artifact = player.inventory[self.inv_ptr].type

    def inventory_right(self, player: Player) -> None:
        """Move inventory selection cursor right."""
        self.inventory_tics = INVENTORY_TIMEOUT_TICS
        if not self.inventory:
            self.inventory = True
            return

        inv_len = len(player.inventory)
        if inv_len == 0:
            return

        if self.inv_ptr + 1 < inv_len:
            self.inv_ptr += 1
            if self.curpos < 6:
                self.curpos += 1
        player.ready_artifact = player.inventory[self.inv_ptr].type

    def use_selected_artifact(self, sim: GameSim, player: Player) -> bool:
        """Activate the currently selected artifact."""
        if not player.inventory or self.inv_ptr >= len(player.inventory):
            return False

        slot = player.inventory[self.inv_ptr]
        arti_type = slot.type
        if arti_type <= 0:
            return False

        success = player_use_artifact(sim, player, arti_type)
        if success:
            self.artifact_flash = 4  # 4-frame flash
            # Update pointers if slot count reached 0 and was removed
            if self.inv_ptr >= len(player.inventory):
                self.inv_ptr = max(0, len(player.inventory) - 1)
            if self.curpos > self.inv_ptr:
                self.curpos = self.inv_ptr
        return success

    def handle_key(self, key: str | int, sim: GameSim, player: Player) -> bool:
        """Process an input key event (port of SB_Responder)."""
        if key in ("\\", "inv_toggle"):
            self.toggle_inventory()
            return True
        elif key in ("[", "inv_left"):
            self.inventory_left(player)
            return True
        elif key in ("]", "inv_right"):
            self.inventory_right(player)
            return True
        elif key in ("\r", "\n", "enter", "inv_use"):
            return self.use_selected_artifact(sim, player)
        return False

    # =========================================================================
    # Number Drawing Helpers
    # =========================================================================

    def draw_i_number(
        self,
        overlay: Overlay2D,
        val: int,
        x: int,
        y: int,
        *,
        red: bool = False,
    ) -> None:
        """Draw a three-digit integer (port of DrINumber / DrRedINumber)."""
        prefix = "INRED" if red else "IN"

        if val < 0:
            val = min(99, -val)
            neg_pic = self.get_patch("NEGATIVE")
            if val > 9:
                overlay.draw_picture(self.get_patch(f"{prefix}{val // 10}"), x + 8, y)
                overlay.draw_picture(neg_pic, x, y)
            else:
                overlay.draw_picture(neg_pic, x + 8, y)
            overlay.draw_picture(self.get_patch(f"{prefix}{val % 10}"), x + 16, y)
            return

        oldval = val
        if val > 99:
            overlay.draw_picture(self.get_patch(f"{prefix}{val // 100}"), x, y)
        val = val % 100
        if val > 9 or oldval > 99:
            overlay.draw_picture(self.get_patch(f"{prefix}{val // 10}"), x + 8, y)
        val = val % 10
        overlay.draw_picture(self.get_patch(f"{prefix}{val}"), x + 16, y)

    def draw_small_number(self, overlay: Overlay2D, val: int, x: int, y: int) -> None:
        """Draw small two/three digit count (port of DrSmallNumber)."""
        if val <= 0:
            return
        val = val % 1000
        if val > 99:
            overlay.draw_picture(self.get_patch(f"SMALLIN{val // 100}"), x, y)
            overlay.draw_picture(self.get_patch(f"SMALLIN{(val % 100) // 10}"), x + 4, y)
        elif val > 9:
            overlay.draw_picture(self.get_patch(f"SMALLIN{val // 10}"), x + 4, y)
        val = val % 10
        overlay.draw_picture(self.get_patch(f"SMALLIN{val}"), x + 8, y)

    def draw_b_number(self, overlay: Overlay2D, val: int, x: int, y: int) -> None:
        """Draw large Font B number (port of DrBNumber)."""
        val = max(0, val)
        text = str(val)
        overlay.draw_text_b(text, x, y)

    # =========================================================================
    # Component Renderers
    # =========================================================================

    def draw_common_bar(self, overlay: Overlay2D) -> None:
        """Draw base status bar background and health chain (port of DrawCommonBar)."""
        overlay.draw_picture(self.get_patch("H2TOP"), 0, 134)

        health_pos = max(0, min(100, self.health_marker))
        chain_x = 28 + (((health_pos * 196) // 100) % 9)
        gem_x = 7 + ((health_pos * 11) // 5)

        overlay.draw_picture(self.get_patch("CHAIN"), chain_x, 193)
        overlay.draw_picture(self.get_patch("LIFEGEM"), gem_x, 193)
        overlay.draw_picture(self.get_patch("LFEDGE"), 0, 193)
        overlay.draw_picture(self.get_patch("RTEDGE"), 277, 193)

    def draw_main_bar(self, overlay: Overlay2D, player: Player) -> None:
        """Draw main status bar interface (port of DrawMainBar)."""
        overlay.draw_picture(self.get_patch("STATBAR"), 38, 162)

        # 1. Health digits
        hlth = max(0, min(100, self.health_marker))
        if hlth >= 25:
            self.draw_i_number(overlay, hlth, 40, 176, red=False)
        else:
            self.draw_i_number(overlay, hlth, 40, 176, red=True)

        # 2. Ready Artifact
        if self.artifact_flash > 0:
            flash_idx = min(3, 4 - self.artifact_flash)
            flash_letter = chr(ord("A") + flash_idx)
            overlay.draw_picture(self.get_patch("ARTICLEAR"), 144, 160)
            overlay.draw_picture(self.get_patch(f"USEARTI{flash_letter}"), 148, 164)
        elif player.ready_artifact > 0 and player.ready_artifact < len(PATCH_ARTI):
            overlay.draw_picture(self.get_patch("ARTICLEAR"), 144, 160)
            overlay.draw_picture(self.get_patch(PATCH_ARTI[player.ready_artifact]), 143, 163)
            # Count
            count = 0
            if player.inventory and self.inv_ptr < len(player.inventory):
                count = player.inventory[self.inv_ptr].count
            if count > 1:
                self.draw_small_number(overlay, count, 162, 184)

        # 3. Mana counts and vials
        m1 = player.mana[0] if len(player.mana) > 0 else 0
        m2 = player.mana[1] if len(player.mana) > 1 else 0

        self.draw_small_number(overlay, m1, 79, 181)
        self.draw_small_number(overlay, m2, 111, 181)

        rw = player.readyweapon
        # Mana 1 icon & vial
        m1_dim = (m1 == 0) or (rw in (0, 2))  # First or third weapon
        overlay.draw_picture(self.get_patch("MANADIM1" if m1_dim else "MANABRIGHT1"), 77, 164)
        vial1_patch = self.get_patch("MANAVIALDIM1" if (rw == 0) else "MANAVIAL1")
        overlay.draw_picture(vial1_patch, 94, 164)

        # Clear liquid above fill line (columns 95, 96, 97 from y=165 to 186)
        empty_h1 = 22 - (22 * min(MAX_MANA, m1)) // MAX_MANA
        if empty_h1 > 0:
            overlay.fill_rect(95, 165, 3, empty_h1, color_index=0)

        # Mana 2 icon & vial
        m2_dim = (m2 == 0) or (rw in (0, 1))  # First or second weapon
        overlay.draw_picture(self.get_patch("MANADIM2" if m2_dim else "MANABRIGHT2"), 110, 164)
        vial2_patch = self.get_patch("MANAVIALDIM2" if (rw in (0, 1)) else "MANAVIAL2")
        overlay.draw_picture(vial2_patch, 102, 164)

        empty_h2 = 22 - (22 * min(MAX_MANA, m2)) // MAX_MANA
        if empty_h2 > 0:
            overlay.fill_rect(103, 165, 3, empty_h2, color_index=0)

        # 4. Armor points
        pclass = min(2, max(0, player.player_class))
        tot_armor = (
            AUTO_ARMOR_SAVE[pclass]
            + player.armor_points[ARMOR_ARMOR]
            + player.armor_points[ARMOR_SHIELD]
            + player.armor_points[ARMOR_HELMET]
            + player.armor_points[ARMOR_AMULET]
        )
        armor_val = tot_armor // (5 * FRACUNIT)
        self.draw_i_number(overlay, armor_val, 250, 176)

        # 5. Weapon pieces
        self.draw_weapon_pieces(overlay, player)

    def draw_weapon_pieces(self, overlay: Overlay2D, player: Player) -> None:
        """Draw fourth weapon assembly progress (port of DrawWeaponPieces)."""
        pieces = player.pieces
        if pieces == 7:
            overlay.draw_picture(self.get_patch("WEAPONFULL"), 190, 162)
            return

        overlay.draw_picture(self.get_patch("WEAPONSLOT"), 190, 162)
        pclass = min(2, max(0, player.player_class))
        coords = PIECE_X[pclass]

        if pieces & WPIECE1:
            overlay.draw_picture(self.get_patch("PIECE1"), coords[0], 162)
        if pieces & WPIECE2:
            overlay.draw_picture(self.get_patch("PIECE2"), coords[1], 162)
        if pieces & WPIECE3:
            overlay.draw_picture(self.get_patch("PIECE3"), coords[2], 162)

    def draw_inventory_bar(
        self,
        overlay: Overlay2D,
        player: Player,
        level_time: int = 0,
    ) -> None:
        """Draw 7-slot inventory popup bar (port of DrawInventoryBar)."""
        overlay.draw_picture(self.get_patch("INVBAR"), 38, 162)

        x_start = self.inv_ptr - self.curpos
        inv_slots = player.inventory

        for i in range(7):
            idx = x_start + i
            if 0 <= idx < len(inv_slots):
                slot = inv_slots[idx]
                if slot.type > 0 and slot.type < len(PATCH_ARTI):
                    overlay.draw_picture(
                        self.get_patch(PATCH_ARTI[slot.type]),
                        50 + i * 31,
                        163,
                    )
                    if slot.count > 1:
                        self.draw_small_number(overlay, slot.count, 68 + i * 31, 185)

        # Selection cursor box
        overlay.draw_picture(self.get_patch("SELECTBOX"), 50 + self.curpos * 31, 163)

        # Left / Right scroll indicator gems
        if x_start > 0:
            gem_patch = self.get_patch("INVLFGEM1" if not (level_time & 4) else "INVLFGEM2")
            overlay.draw_picture(gem_patch, 42, 163)

        if len(inv_slots) - x_start > 7:
            gem_patch = self.get_patch("INVRTGEM1" if not (level_time & 4) else "INVRTGEM2")
            overlay.draw_picture(gem_patch, 269, 163)

    def draw_key_bar(self, overlay: Overlay2D, player: Player) -> None:
        """Draw automap key and armor inventory bar (port of DrawKeyBar)."""
        overlay.draw_picture(self.get_patch("KEYBAR"), 38, 162)

        # Keys 1..11
        x_pos = 46
        for k in range(11):
            if player.keys & (1 << k) and x_pos <= 126:
                overlay.draw_picture(self.get_patch(f"KEYSLOT{k + 1}"), x_pos, 164)
                x_pos += 20

        # Armor pieces
        for i in range(NUMARMOR):
            pts = player.armor_points[i]
            if pts > 0:
                overlay.draw_picture(
                    self.get_patch(f"ARMSLOT{i + 1}"),
                    150 + 31 * i,
                    164,
                )

    def draw_fullscreen_stuff(
        self,
        overlay: Overlay2D,
        player: Player,
        level_time: int = 0,
    ) -> None:
        """Draw minimal HUD when running in fullscreen mode (port of DrawFullScreenStuff)."""
        # Big health at bottom-left
        health_val = max(0, player.health)
        self.draw_b_number(overlay, health_val, 5, 180)

        if not self.inventory:
            # Ready artifact icon at bottom-right
            if player.ready_artifact > 0 and player.ready_artifact < len(PATCH_ARTI):
                overlay.draw_picture(
                    self.get_patch("ARTIBOX"),
                    286,
                    170,
                    translucent=True,
                )
                overlay.draw_picture(
                    self.get_patch(PATCH_ARTI[player.ready_artifact]),
                    284,
                    169,
                )
                count = 0
                if player.inventory and self.inv_ptr < len(player.inventory):
                    count = player.inventory[self.inv_ptr].count
                if count > 1:
                    self.draw_small_number(overlay, count, 302, 192)
        else:
            # Fullscreen floating inventory boxes
            x_start = self.inv_ptr - self.curpos
            inv_slots = player.inventory

            for i in range(7):
                overlay.draw_picture(
                    self.get_patch("ARTIBOX"),
                    50 + i * 31,
                    168,
                    translucent=True,
                )
                idx = x_start + i
                if 0 <= idx < len(inv_slots):
                    slot = inv_slots[idx]
                    if slot.type > 0 and slot.type < len(PATCH_ARTI):
                        overlay.draw_picture(
                            self.get_patch(PATCH_ARTI[slot.type]),
                            49 + i * 31,
                            167,
                        )
                        if slot.count > 1:
                            self.draw_small_number(
                                overlay,
                                slot.count,
                                66 + i * 31,
                                188,
                            )

            overlay.draw_picture(
                self.get_patch("SELECTBOX"),
                50 + self.curpos * 31,
                167,
            )
            if x_start > 0:
                gem = self.get_patch("INVLFGEM1" if not (level_time & 4) else "INVLFGEM2")
                overlay.draw_picture(gem, 40, 167)
            if len(inv_slots) - x_start > 7:
                gem = self.get_patch("INVRTGEM1" if not (level_time & 4) else "INVRTGEM2")
                overlay.draw_picture(gem, 268, 167)

    def draw_animated_icons(
        self,
        overlay: Overlay2D,
        player: Player,
        level_time: int = 0,
    ) -> None:
        """Draw active powerup rotating icons at top of screen (port of DrawAnimatedIcons)."""
        frame = (level_time // 3) & 15

        # Flight (Wings of Wrath)
        flight_tics = player.powers[PowerType.PW_FLIGHT]
        if flight_tics > 0:
            if flight_tics > BLINKTHRESHOLD or not (flight_tics & 16):
                overlay.draw_picture(self.get_patch(f"SPFLY{frame}"), 20, 19)

        # Speed (Boots of Speed)
        speed_tics = player.powers[PowerType.PW_SPEED]
        if speed_tics > 0:
            if speed_tics > BLINKTHRESHOLD or not (speed_tics & 16):
                overlay.draw_picture(self.get_patch(f"SPED{frame}"), 60, 19)

        # Invulnerability (Icon of the Defender)
        invul_tics = player.powers[PowerType.PW_INVULNERABILITY]
        if invul_tics > 0:
            if invul_tics > BLINKTHRESHOLD or not (invul_tics & 16):
                overlay.draw_picture(self.get_patch(f"DEFN{frame}"), 260, 19)

        # Minotaur active (Dark Servant)
        mino_tics = player.powers[PowerType.PW_MINOTAUR]
        if mino_tics > 0:
            if mino_tics > BLINKTHRESHOLD or not (mino_tics & 16):
                overlay.draw_picture(self.get_patch(f"MINO{frame}"), 300, 19)

    # =========================================================================
    # Main Drawer
    # =========================================================================

    def draw(
        self,
        overlay: Overlay2D,
        player: Player,
        level_time: int = 0,
    ) -> None:
        """Render status bar, inventory, or fullscreen HUD (port of SB_Drawer)."""
        if self.fullscreen and not self.automap_active:
            self.draw_fullscreen_stuff(overlay, player, level_time)
        else:
            self.draw_common_bar(overlay)
            if self.inventory:
                self.draw_inventory_bar(overlay, player, level_time)
            elif self.automap_active:
                self.draw_key_bar(overlay, player)
            else:
                self.draw_main_bar(overlay, player)

        self.draw_animated_icons(overlay, player, level_time)


__all__ = [
    "INVENTORY_TIMEOUT_TICS",
    "MAX_MANA",
    "NUMBONUSPALS",
    "NUMPOISONPALS",
    "NUMREDPALS",
    "PATCH_ARTI",
    "PIECE_X",
    "STARTBONUSPALS",
    "STARTICEPAL",
    "STARTPOISONPALS",
    "STARTREDPALS",
    "WPIECE1",
    "WPIECE2",
    "WPIECE3",
    "StatusBar",
    "get_sb_palette",
]
