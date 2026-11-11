# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Hexen menu system (T12.4).

Port of MN_Init, MN_Drawer, MN_Ticker, MN_Responder, MN_ActivateMenu,
and MN_DeactivateMenu (mn_menu.c).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame

from pyhexen.app.config import Config, save_config
from pyhexen.app.input import KeyBindings
from pyhexen.render.overlay2d import VIRTUAL_HEIGHT, Overlay2D
from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.wadfile import WadFile, WadStack

if TYPE_CHECKING:
    pass

# Constants from mn_menu.c
ITEM_HEIGHT: int = 20
SELECTOR_XOFFSET: int = -28
SELECTOR_YOFFSET: int = -1
SLOTTEXTLEN: int = 16
ASCII_CURSOR: str = "["

COMMON_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (640, 400),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 800),
    (1920, 1080),
)

LEFT_DIR: int = 0
RIGHT_DIR: int = 1

# Sound identifiers from Hexen
SFX_PLATFORM_STOP: str = "PlatformStop"
SFX_HAMMER_HIT: str = "FighterHammerHitWall"
SFX_DOOR_CLOSE: str = "DoorCloseLight"
SFX_PICKUP_KEY: str = "PickupKey"
SFX_CHAT: str = "Chat"


class MenuType(IntEnum):
    """Menu screen identifiers matching Hexen MenuType_t."""

    MAIN = 0
    CLASS = 1
    SKILL = 2
    OPTIONS = 3
    OPTIONS2 = 4
    FILES = 5
    LOAD = 6
    SAVE = 7
    CONTROLS = 8
    VIDEO = 9
    NONE = 10


class ItemType(IntEnum):
    """Menu item types matching Hexen ItemType_t."""

    EMPTY = 0
    EFUNC = 1
    LRFUNC = 2
    SETMENU = 3
    INERT = 4


class PromptType(IntEnum):
    """Confirmation prompt modal dialogs (typeofask)."""

    NONE = 0
    QUIT = 1
    END_GAME = 2
    QUICKSAVE = 3
    QUICKLOAD = 4
    SUICIDE = 5


PROMPT_MESSAGES: dict[PromptType, str] = {
    PromptType.QUIT: "ARE YOU SURE YOU WANT TO QUIT?",
    PromptType.END_GAME: "ARE YOU SURE YOU WANT TO END THE GAME?",
    PromptType.QUICKSAVE: "DO YOU WANT TO QUICKSAVE THE GAME NAMED",
    PromptType.QUICKLOAD: "DO YOU WANT TO QUICKLOAD THE GAME NAMED",
    PromptType.SUICIDE: "ARE YOU SURE YOU WANT TO SUICIDE?",
}

SKILL_NAMES_BY_CLASS: tuple[tuple[str, ...], ...] = (
    ("SQUIRE", "KNIGHT", "WARRIOR", "BERSERKER", "TITAN"),  # Fighter
    ("ALTAR BOY", "ACOLYTE", "PRIEST", "CARDINAL", "POPE"),  # Cleric
    ("APPRENTICE", "ENCHANTER", "SORCERER", "WARLOCK", "ARCHIMAGE"),  # Mage
)

CLASS_BOX_LUMPS: tuple[str, ...] = ("M_FBOX", "M_CBOX", "M_MBOX")
CLASS_WALK_PREFIXES: tuple[str, ...] = ("M_FWALK", "M_CWALK", "M_MWALK")
MAULO_FRAMES: tuple[str, ...] = (
    "FBULA0",
    "FBULB0",
    "FBULC0",
    "FBULD0",
    "FBULE0",
    "FBULF0",
    "FBULG0",
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


def _make_fallback_patch(name: str) -> Picture:
    """Deterministic synthetic fallback for missing menu graphic lumps."""
    upper = name.upper()
    if upper == "M_HTIC":
        return _make_synthetic_picture(144, 52, color=160, left=0, top=-3)
    if upper in ("M_SLCTR1", "M_SLCTR2"):
        return _make_synthetic_picture(22, 15, color=180, left=-1, top=-5)
    if upper == "M_SLDLT":
        return _make_synthetic_picture(23, 19, color=130, left=-9, top=0)
    if upper in ("M_SLDMD1", "M_SLDMD2"):
        return _make_synthetic_picture(8, 13, color=140, left=0, top=-1)
    if upper == "M_SLDRT":
        return _make_synthetic_picture(23, 19, color=130, left=0, top=0)
    if upper == "M_SLDKB":
        return _make_synthetic_picture(13, 13, color=200, left=6, top=6)
    if upper == "M_FSLOT":
        return _make_synthetic_picture(190, 18, color=60, left=-1, top=-1)
    if upper in ("M_FBOX", "M_CBOX", "M_MBOX"):
        return _make_synthetic_picture(112, 136, color=80, left=0, top=0)
    if any(upper.startswith(prefix) for prefix in ("M_FWALK", "M_CWALK", "M_MWALK")):
        return _make_synthetic_picture(44, 66, color=150, left=-8, top=-4)
    if upper.startswith("FBUL"):
        return _make_synthetic_picture(32, 32, color=210, left=16, top=30)
    return _make_synthetic_picture(16, 16, color=120)


@dataclass(slots=True)
class MenuItem:
    """A single item inside a menu."""

    item_type: ItemType
    text: str | None = None
    target_menu: MenuType = MenuType.NONE
    option: int = 0
    slider_width: int = 0
    slider_min: int = 0
    slider_max: int = 0
    value_getter: Callable[[], int] | None = None
    value_setter: Callable[[int], None] | None = None


@dataclass(slots=True)
class Menu:
    """Menu container with items and navigation state."""

    menu_type: MenuType
    x: int
    y: int
    items: list[MenuItem] = field(default_factory=list)
    cur_pos: int = 0
    old_pos: int = 0
    prev_menu: MenuType = MenuType.NONE
    title: str | None = None


class MenuManager:
    """Complete Hexen menu subsystem (T12.4).

    Manages main menu, class select, skill select, options, controls/key rebinding,
    save/load slots, and modal confirmation prompts.
    """

    def __init__(
        self,
        wad: WadStack | WadFile | None = None,
        config: Config | None = None,
        bindings: KeyBindings | None = None,
        config_path: Path | str | None = None,
        on_config_changed: Callable[[Config], None] | None = None,
    ) -> None:
        self.wad: WadStack | WadFile | None = wad
        self.config: Config = config if config is not None else Config()
        self.config_path: Path | str | None = config_path
        self.bindings: KeyBindings = bindings if bindings is not None else KeyBindings()
        self._patch_cache: dict[str, Picture] = {}

        # Apply saved key bindings if present in config
        if hasattr(self.config.controls, "bindings") and self.config.controls.bindings:
            for act_name, code in self.config.controls.bindings.items():
                self._assign_rebound_key(act_name, code, persist=False)

        # Menu state
        self.active: bool = False
        self.current_menu_type: MenuType = MenuType.MAIN
        self.prompt_type: PromptType = PromptType.NONE
        self.info_type: int = 0
        self.menu_time: int = 0
        self.messages_on: bool = True
        self.selected_class: int = 0  # 0: Fighter, 1: Cleric, 2: Mage
        self.selected_skill: int = 2  # medium default

        # Save/load slots (6 slots)
        self.slot_texts: list[str] = [""] * 6
        self.slot_status: list[bool] = [False] * 6
        self.current_slot: int = 0
        self.editing_slot: bool = False
        self._old_slot_text: str = ""
        self.quicksave_slot: int = -1
        self.quickload_slot: int = -1

        # Key rebinding state
        self.rebinding_action: str | None = None

        # Callbacks
        self.on_new_game: Callable[[int, int], None] | None = None
        self.on_load_game: Callable[[int], None] | None = None
        self.on_save_game: Callable[[int, str], None] | None = None
        self.on_quit_game: Callable[[], None] | None = None
        self.on_end_game: Callable[[], None] | None = None
        self.on_suicide: Callable[[], None] | None = None
        self.on_sound: Callable[[str], None] | None = None
        # Player message (P_SetMessage text, ultimate flag) and the messageson toggle.
        self.on_message: Callable[[str, bool], None] | None = None
        self.on_messages_toggled: Callable[[bool], None] | None = None
        self.on_config_changed: Callable[[Config], None] | None = on_config_changed

        # Build menus
        self.menus: dict[MenuType, Menu] = self._build_menus()

    def get_patch(self, name: str) -> Picture:
        """Fetch a graphic patch from WAD with synthetic fallback."""
        name_upper = name.upper()
        if name_upper in self._patch_cache:
            return self._patch_cache[name_upper]

        if self.wad is not None:
            lump = self.wad.find(name_upper)
            if lump is not None:
                try:
                    pic = decode_picture(self.wad.get_bytes(name_upper))
                    self._patch_cache[name_upper] = pic
                    return pic
                except Exception:
                    pass

        fallback = _make_fallback_patch(name_upper)
        self._patch_cache[name_upper] = fallback
        return fallback

    def _play_sound(self, sound_name: str) -> None:
        """Trigger a menu sound effect."""
        if self.on_sound is not None:
            self.on_sound(sound_name)

    # -------------------------------------------------------------------------
    # Menu Initialization
    # -------------------------------------------------------------------------

    def _build_menus(self) -> dict[MenuType, Menu]:
        """Construct the menu hierarchy."""
        menus: dict[MenuType, Menu] = {}

        # 1. Main Menu (x=110, y=56)
        menus[MenuType.MAIN] = Menu(
            menu_type=MenuType.MAIN,
            x=110,
            y=56,
            items=[
                MenuItem(ItemType.SETMENU, "NEW GAME", target_menu=MenuType.CLASS),
                MenuItem(ItemType.SETMENU, "OPTIONS", target_menu=MenuType.OPTIONS),
                MenuItem(ItemType.SETMENU, "GAME FILES", target_menu=MenuType.FILES),
                MenuItem(ItemType.EFUNC, "INFO", option=0),
                MenuItem(ItemType.EFUNC, "QUIT GAME", option=0),
            ],
            prev_menu=MenuType.NONE,
        )

        # 2. Class Menu (x=66, y=66)
        menus[MenuType.CLASS] = Menu(
            menu_type=MenuType.CLASS,
            x=66,
            y=66,
            items=[
                MenuItem(ItemType.EFUNC, "FIGHTER", option=0),
                MenuItem(ItemType.EFUNC, "CLERIC", option=1),
                MenuItem(ItemType.EFUNC, "MAGE", option=2),
            ],
            prev_menu=MenuType.MAIN,
            title="CHOOSE CLASS:",
        )

        # 3. Skill Menu (x=120, y=44)
        menus[MenuType.SKILL] = Menu(
            menu_type=MenuType.SKILL,
            x=120,
            y=44,
            items=[
                MenuItem(ItemType.EFUNC, "SQUIRE", option=0),
                MenuItem(ItemType.EFUNC, "KNIGHT", option=1),
                MenuItem(ItemType.EFUNC, "WARRIOR", option=2),
                MenuItem(ItemType.EFUNC, "BERSERKER", option=3),
                MenuItem(ItemType.EFUNC, "TITAN", option=4),
            ],
            cur_pos=2,
            old_pos=2,
            prev_menu=MenuType.CLASS,
            title="CHOOSE SKILL LEVEL:",
        )

        # 4. Options Menu (x=88, y=30)
        menus[MenuType.OPTIONS] = Menu(
            menu_type=MenuType.OPTIONS,
            x=88,
            y=30,
            items=[
                MenuItem(ItemType.EFUNC, "END GAME", option=0),
                MenuItem(ItemType.EFUNC, "MESSAGES : ", option=0),
                MenuItem(
                    ItemType.LRFUNC,
                    "MOUSE SENSITIVITY",
                    slider_width=10,
                    slider_min=0,
                    slider_max=9,
                    value_getter=self._get_mouse_sensitivity,
                    value_setter=self._set_mouse_sensitivity,
                ),
                MenuItem(ItemType.EMPTY),
                MenuItem(ItemType.SETMENU, "MORE...", target_menu=MenuType.OPTIONS2),
                MenuItem(ItemType.SETMENU, "CONTROLS...", target_menu=MenuType.CONTROLS),
                MenuItem(ItemType.SETMENU, "VIDEO OPTIONS...", target_menu=MenuType.VIDEO),
                MenuItem(
                    ItemType.LRFUNC,
                    "INVERT MOUSE : ",
                    slider_width=0,
                    slider_min=0,
                    slider_max=1,
                    value_getter=self._get_invert_mouse,
                    value_setter=self._set_invert_mouse,
                ),
            ],
            prev_menu=MenuType.MAIN,
        )

        # 5. Options2 Menu (x=90, y=20)
        menus[MenuType.OPTIONS2] = Menu(
            menu_type=MenuType.OPTIONS2,
            x=90,
            y=20,
            items=[
                MenuItem(
                    ItemType.LRFUNC,
                    "SCREEN SIZE",
                    slider_width=9,
                    slider_min=0,
                    slider_max=8,
                    value_getter=self._get_screen_size,
                    value_setter=self._set_screen_size,
                ),
                MenuItem(ItemType.EMPTY),
                MenuItem(
                    ItemType.LRFUNC,
                    "SFX VOLUME",
                    slider_width=16,
                    slider_min=0,
                    slider_max=15,
                    value_getter=self._get_sfx_volume,
                    value_setter=self._set_sfx_volume,
                ),
                MenuItem(ItemType.EMPTY),
                MenuItem(
                    ItemType.LRFUNC,
                    "MUSIC VOLUME",
                    slider_width=16,
                    slider_min=0,
                    slider_max=15,
                    value_getter=self._get_music_volume,
                    value_setter=self._set_music_volume,
                ),
                MenuItem(ItemType.EMPTY),
                MenuItem(ItemType.SETMENU, "VIDEO OPTIONS...", target_menu=MenuType.VIDEO),
            ],
            prev_menu=MenuType.OPTIONS,
        )

        # Video Options Menu (x=40, y=24)
        menus[MenuType.VIDEO] = Menu(
            menu_type=MenuType.VIDEO,
            x=40,
            y=24,
            items=[
                MenuItem(
                    ItemType.LRFUNC,
                    "RESOLUTION",
                    slider_width=0,
                    slider_min=0,
                    slider_max=len(COMMON_RESOLUTIONS) - 1,
                    value_getter=self._get_resolution_idx,
                    value_setter=self._set_resolution_idx,
                ),
                MenuItem(
                    ItemType.LRFUNC,
                    "FULLSCREEN",
                    slider_width=0,
                    slider_min=0,
                    slider_max=1,
                    value_getter=self._get_fullscreen,
                    value_setter=self._set_fullscreen,
                ),
                MenuItem(
                    ItemType.LRFUNC,
                    "FIELD OF VIEW",
                    slider_width=0,
                    slider_min=0,
                    slider_max=10,
                    value_getter=self._get_fov_slider,
                    value_setter=self._set_fov_slider,
                ),
                MenuItem(
                    ItemType.LRFUNC,
                    "LOW-RES MODE",
                    slider_width=0,
                    slider_min=0,
                    slider_max=1,
                    value_getter=self._get_lowres,
                    value_setter=self._set_lowres,
                ),
                MenuItem(
                    ItemType.LRFUNC,
                    "PIXEL ASPECT",
                    slider_width=0,
                    slider_min=0,
                    slider_max=1,
                    value_getter=self._get_aspect,
                    value_setter=self._set_aspect,
                ),
                MenuItem(
                    ItemType.LRFUNC,
                    "TRUE PITCH",
                    slider_width=0,
                    slider_min=0,
                    slider_max=1,
                    value_getter=self._get_true_pitch,
                    value_setter=self._set_true_pitch,
                ),
                MenuItem(
                    ItemType.LRFUNC,
                    "FILTERING",
                    slider_width=0,
                    slider_min=0,
                    slider_max=1,
                    value_getter=self._get_filter,
                    value_setter=self._set_filter,
                ),
                MenuItem(ItemType.EMPTY),
                MenuItem(ItemType.EFUNC, "SAVE CONFIG", option=0),
            ],
            prev_menu=MenuType.OPTIONS,
        )

        # 6. Files Menu (x=110, y=60)
        menus[MenuType.FILES] = Menu(
            menu_type=MenuType.FILES,
            x=110,
            y=60,
            items=[
                MenuItem(ItemType.SETMENU, "LOAD GAME", target_menu=MenuType.LOAD),
                MenuItem(ItemType.SETMENU, "SAVE GAME", target_menu=MenuType.SAVE),
            ],
            prev_menu=MenuType.MAIN,
        )

        # 7. Load Menu (x=70, y=30)
        menus[MenuType.LOAD] = Menu(
            menu_type=MenuType.LOAD,
            x=70,
            y=30,
            items=[MenuItem(ItemType.EFUNC, option=i) for i in range(6)],
            prev_menu=MenuType.FILES,
            title="LOAD GAME",
        )

        # 8. Save Menu (x=70, y=30)
        menus[MenuType.SAVE] = Menu(
            menu_type=MenuType.SAVE,
            x=70,
            y=30,
            items=[MenuItem(ItemType.EFUNC, option=i) for i in range(6)],
            prev_menu=MenuType.FILES,
            title="SAVE GAME",
        )

        # 9. Controls Menu (x=60, y=20)
        menus[MenuType.CONTROLS] = Menu(
            menu_type=MenuType.CONTROLS,
            x=60,
            y=20,
            items=[
                MenuItem(ItemType.EFUNC, "FORWARD", option=0),
                MenuItem(ItemType.EFUNC, "BACKWARD", option=1),
                MenuItem(ItemType.EFUNC, "STRAFE LEFT", option=2),
                MenuItem(ItemType.EFUNC, "STRAFE RIGHT", option=3),
                MenuItem(ItemType.EFUNC, "TURN LEFT", option=4),
                MenuItem(ItemType.EFUNC, "TURN RIGHT", option=5),
                MenuItem(ItemType.EFUNC, "FIRE", option=6),
                MenuItem(ItemType.EFUNC, "USE", option=7),
                MenuItem(ItemType.EFUNC, "JUMP", option=8),
                MenuItem(ItemType.EFUNC, "SPEED", option=9),
                MenuItem(ItemType.EFUNC, "USE ARTIFACT", option=10),
                MenuItem(ItemType.EFUNC, "PREV ARTIFACT", option=11),
                MenuItem(ItemType.EFUNC, "NEXT ARTIFACT", option=12),
            ],
            prev_menu=MenuType.OPTIONS,
            title="KEY CONTROLS",
        )

        return menus

    # -------------------------------------------------------------------------
    # Value Getters & Setters
    # -------------------------------------------------------------------------

    def _get_mouse_sensitivity(self) -> int:
        sens = getattr(self.config.controls, "mouse_sensitivity", 1.0)
        return int(max(0, min(9, round(sens * 9))))

    def _set_mouse_sensitivity(self, val: int) -> None:
        self.config.controls.mouse_sensitivity = max(0.0, min(1.0, val / 9.0))

    def _get_screen_size(self) -> int:
        return 8  # Full screen view

    def _set_screen_size(self, val: int) -> None:
        pass

    def _get_sfx_volume(self) -> int:
        vol = getattr(self.config.audio, "sfx_volume", 0.8)
        return int(max(0, min(15, round(vol * 15))))

    def _set_sfx_volume(self, val: int) -> None:
        self.config.audio.sfx_volume = max(0.0, min(1.0, val / 15.0))

    def _get_music_volume(self) -> int:
        vol = getattr(self.config.audio, "music_volume", 0.7)
        return int(max(0, min(15, round(vol * 15))))

    def _set_music_volume(self, val: int) -> None:
        self.config.audio.music_volume = max(0.0, min(1.0, val / 15.0))

    def _get_invert_mouse(self) -> int:
        return 1 if getattr(self.config.controls, "invert_mouse", False) else 0

    def _set_invert_mouse(self, val: int) -> None:
        self.config.controls.invert_mouse = bool(val)

    def _get_resolution_idx(self) -> int:
        cur = (self.config.video.width, self.config.video.height)
        if cur in COMMON_RESOLUTIONS:
            return COMMON_RESOLUTIONS.index(cur)
        return min(
            range(len(COMMON_RESOLUTIONS)),
            key=lambda i: (
                abs(COMMON_RESOLUTIONS[i][0] - cur[0]) + abs(COMMON_RESOLUTIONS[i][1] - cur[1])
            ),
        )

    def _set_resolution_idx(self, val: int) -> None:
        idx = max(0, min(len(COMMON_RESOLUTIONS) - 1, val))
        w, h = COMMON_RESOLUTIONS[idx]
        self.config.video.width = w
        self.config.video.height = h

    def _get_fullscreen(self) -> int:
        return 1 if self.config.video.fullscreen else 0

    def _set_fullscreen(self, val: int) -> None:
        self.config.video.fullscreen = bool(val)

    def _get_fov_slider(self) -> int:
        fov = getattr(self.config.video, "fov", 90.0)
        return int(round(max(0.0, min(10.0, (fov - 70.0) / 50.0 * 10.0))))

    def _set_fov_slider(self, val: int) -> None:
        clamped = max(0, min(10, val))
        self.config.video.fov = 70.0 + (clamped / 10.0) * 50.0

    def _get_lowres(self) -> int:
        return 1 if self.config.video.authentic_lowres else 0

    def _set_lowres(self, val: int) -> None:
        self.config.video.authentic_lowres = bool(val)

    def _get_aspect(self) -> int:
        return 1 if self.config.video.authentic_aspect else 0

    def _set_aspect(self, val: int) -> None:
        self.config.video.authentic_aspect = bool(val)

    def _get_true_pitch(self) -> int:
        return 1 if self.config.video.true_pitch else 0

    def _set_true_pitch(self, val: int) -> None:
        self.config.video.true_pitch = bool(val)

    def _get_filter(self) -> int:
        return 1 if self.config.video.smooth_filter else 0

    def _set_filter(self, val: int) -> None:
        self.config.video.smooth_filter = bool(val)

    def save_config_to_disk(self) -> None:
        """Persist current configuration and key bindings to config.toml."""
        self.config.controls.bindings = {
            "forward": self.bindings.forward,
            "backward": self.bindings.backward,
            "strafe_left": self.bindings.strafe_left,
            "strafe_right": self.bindings.strafe_right,
            "turn_left": self.bindings.turn_left,
            "turn_right": self.bindings.turn_right,
            "fire": self.bindings.fire,
            "use": self.bindings.use,
            "jump": self.bindings.jump,
            "speed": self.bindings.speed,
            "use_artifact": self.bindings.use_artifact,
            "inv_left": self.bindings.inv_left,
            "inv_right": self.bindings.inv_right,
        }
        try:
            save_config(self.config, self.config_path)
        except Exception:
            pass

    def _notify_config_changed(self) -> None:
        if self.on_config_changed is not None:
            self.on_config_changed(self.config)

    # -------------------------------------------------------------------------
    # Navigation & State
    # -------------------------------------------------------------------------

    @property
    def current_menu(self) -> Menu:
        return self.menus[self.current_menu_type]

    def activate_menu(self) -> None:
        """Open the menu system (MN_ActivateMenu)."""
        if self.active:
            return
        self.active = True
        self.menu_time = 0
        self.prompt_type = PromptType.NONE
        self.info_type = 0
        self.editing_slot = False
        self.rebinding_action = None
        self.set_menu(MenuType.MAIN)
        self._play_sound(SFX_PLATFORM_STOP)

    def deactivate_menu(self) -> None:
        """Close the menu system (MN_DeactivateMenu)."""
        if not self.active and self.prompt_type == PromptType.NONE:
            return
        self.active = False
        self.prompt_type = PromptType.NONE
        self.info_type = 0
        self.editing_slot = False
        self.rebinding_action = None
        self._play_sound(SFX_PLATFORM_STOP)

    def set_menu(self, menu_type: MenuType) -> None:
        """Switch to a submenu (SetMenu)."""
        if menu_type == MenuType.NONE:
            self.deactivate_menu()
            return

        cur = self.current_menu
        cur.old_pos = cur.cur_pos

        self.current_menu_type = menu_type
        new_menu = self.menus[menu_type]

        # Refresh class-specific strings on entering Skill Menu
        if menu_type == MenuType.SKILL:
            names = SKILL_NAMES_BY_CLASS[self.selected_class]
            for idx, item in enumerate(new_menu.items):
                item.text = names[idx]
            # Offset based on class in original C
            if self.selected_class == 0:
                new_menu.x = 120
            elif self.selected_class == 1:
                new_menu.x = 116
            else:
                new_menu.x = 112

        new_menu.cur_pos = new_menu.old_pos

    def tick(self) -> None:
        """Advance menu timer by 1 tic (MN_Ticker)."""
        if self.active or self.prompt_type != PromptType.NONE:
            self.menu_time += 1

    # -------------------------------------------------------------------------
    # Input Handling (MN_Responder)
    # -------------------------------------------------------------------------

    def handle_key(self, key: int, char_typed: str | None = None) -> bool:
        """Process keyboard input event."""
        # 1. Info Screen eating keypresses
        if self.info_type > 0:
            if key == pygame.K_ESCAPE or self.info_type >= 3:
                self.info_type = 0
                self.deactivate_menu()
            else:
                self.info_type += 1
            self._play_sound(SFX_DOOR_CLOSE)
            return True

        # 2. Confirmation Prompt modal
        if self.prompt_type != PromptType.NONE:
            if key in (pygame.K_y, pygame.K_RETURN, pygame.K_KP_ENTER):
                self._confirm_prompt()
                return True
            elif key in (pygame.K_n, pygame.K_ESCAPE):
                self.prompt_type = PromptType.NONE
                return True
            return True

        # 3. Key rebinding mode
        if self.rebinding_action is not None:
            if key == pygame.K_ESCAPE:
                self.rebinding_action = None
                self._play_sound(SFX_PICKUP_KEY)
                return True
            self._assign_rebound_key(self.rebinding_action, key)
            self.rebinding_action = None
            self._play_sound(SFX_DOOR_CLOSE)
            return True

        # 4. Save game name editing (key steal)
        if self.editing_slot:
            return self._handle_slot_text_key(key, char_typed)

        # 5. Global Function key shortcuts when menu is not active
        if not self.active:
            if key == pygame.K_ESCAPE:
                self.activate_menu()
                return True
            if key == pygame.K_F1:  # Info
                self.info_type = 1
                self.active = True
                self._play_sound(SFX_DOOR_CLOSE)
                return True
            if key == pygame.K_F2:  # Save
                self.activate_menu()
                self.set_menu(MenuType.SAVE)
                return True
            if key == pygame.K_F3:  # Load
                self.activate_menu()
                self.set_menu(MenuType.LOAD)
                return True
            if key == pygame.K_F4:  # Volume
                self.activate_menu()
                self.set_menu(MenuType.OPTIONS2)
                return True
            if key == pygame.K_F5:  # Suicide
                self.prompt_type = PromptType.SUICIDE
                self._play_sound(SFX_CHAT)
                return True
            if key == pygame.K_F6:  # Quicksave
                if self.quicksave_slot >= 0:
                    self.prompt_type = PromptType.QUICKSAVE
                    self._play_sound(SFX_CHAT)
                else:
                    self.activate_menu()
                    self.set_menu(MenuType.SAVE)
                    self._send_message("CHOOSE A QUICKSAVE SLOT", True)
                return True
            if key == pygame.K_F7:  # End game
                self.prompt_type = PromptType.END_GAME
                self._play_sound(SFX_CHAT)
                return True
            if key == pygame.K_F8:  # Messages toggle
                self.toggle_messages()
                return True
            if key == pygame.K_F9:  # Quickload
                if self.quickload_slot >= 0:
                    self.prompt_type = PromptType.QUICKLOAD
                    self._play_sound(SFX_CHAT)
                else:
                    self.activate_menu()
                    self.set_menu(MenuType.LOAD)
                    self._send_message("CHOOSE A QUICKLOAD SLOT", True)
                return True
            if key == pygame.K_F10:  # Quit game
                self.prompt_type = PromptType.QUIT
                self._play_sound(SFX_CHAT)
                return True
            return False

        # 6. Active menu navigation
        menu = self.current_menu
        item = menu.items[menu.cur_pos] if menu.items else None

        if key == pygame.K_DOWN:
            self._move_cursor(1)
            return True
        elif key == pygame.K_UP:
            self._move_cursor(-1)
            return True
        elif key == pygame.K_LEFT:
            if item is not None and item.item_type == ItemType.LRFUNC:
                self._adjust_slider(item, -1)
                self._play_sound(SFX_PICKUP_KEY)
            return True
        elif key == pygame.K_RIGHT:
            if item is not None and item.item_type == ItemType.LRFUNC:
                self._adjust_slider(item, 1)
                self._play_sound(SFX_PICKUP_KEY)
            return True
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.select_current_item()
            return True
        elif key == pygame.K_ESCAPE:
            if menu.prev_menu != MenuType.NONE:
                self.set_menu(menu.prev_menu)
                self._play_sound(SFX_PICKUP_KEY)
            else:
                self.deactivate_menu()
            return True
        elif char_typed:
            # Jump by first letter
            upper_char = char_typed.upper()
            for idx, itm in enumerate(menu.items):
                if itm.item_type != ItemType.EMPTY and itm.text and itm.text.startswith(upper_char):
                    menu.cur_pos = idx
                    self._play_sound(SFX_HAMMER_HIT)
                    return True

        return False

    def toggle_messages(self) -> None:
        """Port of SCMessages (mn_menu.c): flip messageson and say so."""
        self.messages_on = not self.messages_on
        if self.on_messages_toggled is not None:
            self.on_messages_toggled(self.messages_on)
        self._send_message("MESSAGES ON" if self.messages_on else "MESSAGES OFF", True)
        self._play_sound(SFX_CHAT)

    def _send_message(self, text: str, ultimate: bool) -> None:
        if self.on_message is not None:
            self.on_message(text, ultimate)

    def _move_cursor(self, direction: int) -> None:
        """Move cursor up or down skipping empty items."""
        menu = self.current_menu
        if not menu.items:
            return
        count = len(menu.items)
        new_pos = menu.cur_pos
        for _ in range(count):
            new_pos = (new_pos + direction) % count
            if menu.items[new_pos].item_type != ItemType.EMPTY:
                menu.cur_pos = new_pos
                self._play_sound(SFX_HAMMER_HIT)
                return

    def _adjust_slider(self, item: MenuItem, step: int) -> None:
        """Adjust a slider value."""
        if item.value_getter is None or item.value_setter is None:
            return
        curr = item.value_getter()
        if item.slider_max == 1 and item.slider_min == 0:
            new_val = 1 if curr == 0 else 0
        elif item.text == "RESOLUTION":
            new_val = curr + step
            if new_val < item.slider_min:
                new_val = item.slider_max
            elif new_val > item.slider_max:
                new_val = item.slider_min
        else:
            new_val = max(item.slider_min, min(item.slider_max, curr + step))
        item.value_setter(new_val)
        self.save_config_to_disk()
        self._notify_config_changed()

    def select_current_item(self) -> None:
        """Activate the currently focused menu item."""
        menu = self.current_menu
        if not menu.items:
            return
        item = menu.items[menu.cur_pos]

        if item.item_type == ItemType.SETMENU:
            self._play_sound(SFX_DOOR_CLOSE)
            self.set_menu(item.target_menu)
        elif item.item_type == ItemType.LRFUNC:
            self._adjust_slider(item, 1)
            self._play_sound(SFX_PICKUP_KEY)
        elif item.item_type == ItemType.EFUNC:
            self._execute_item_action(menu.menu_type, menu.cur_pos, item)

    def _execute_item_action(self, menu_type: MenuType, pos: int, item: MenuItem) -> None:
        """Execute action for EFUNC items."""
        if menu_type == MenuType.MAIN:
            if pos == 3:  # INFO
                self.info_type = 1
                self._play_sound(SFX_DOOR_CLOSE)
            elif pos == 4:  # QUIT GAME
                self.prompt_type = PromptType.QUIT
                self.active = False
                self._play_sound(SFX_CHAT)

        elif menu_type == MenuType.CLASS:
            self.selected_class = item.option
            self._play_sound(SFX_DOOR_CLOSE)
            self.set_menu(MenuType.SKILL)

        elif menu_type == MenuType.SKILL:
            self.selected_skill = item.option
            self._play_sound(SFX_DOOR_CLOSE)
            self.deactivate_menu()
            if self.on_new_game is not None:
                self.on_new_game(self.selected_class, self.selected_skill)

        elif menu_type == MenuType.OPTIONS:
            if pos == 0:  # END GAME
                self.prompt_type = PromptType.END_GAME
                self.active = False
                self._play_sound(SFX_CHAT)
            elif pos == 1:  # MESSAGES
                self.toggle_messages()

        elif menu_type == MenuType.VIDEO:
            if pos == 8:  # SAVE CONFIG
                self.save_config_to_disk()
                self._notify_config_changed()
                self._play_sound(SFX_CHAT)

        elif menu_type == MenuType.LOAD:
            slot = pos
            if self.slot_status[slot]:
                self.quickload_slot = slot
                self.deactivate_menu()
                self._play_sound(SFX_DOOR_CLOSE)
                if self.on_load_game is not None:
                    self.on_load_game(slot)

        elif menu_type == MenuType.SAVE:
            slot = pos
            self.current_slot = slot
            self.editing_slot = True
            self._old_slot_text = self.slot_texts[slot]
            self._play_sound(SFX_DOOR_CLOSE)

        elif menu_type == MenuType.CONTROLS:
            action_name = item.text
            self.rebinding_action = action_name
            self._play_sound(SFX_DOOR_CLOSE)

    def _handle_slot_text_key(self, key: int, char_typed: str | None) -> bool:
        """Handle typing when editing save slot name."""
        slot = self.current_slot
        cur_text = self.slot_texts[slot]

        if key == pygame.K_BACKSPACE:
            if cur_text:
                self.slot_texts[slot] = cur_text[:-1]
            return True
        elif key == pygame.K_ESCAPE:
            self.slot_texts[slot] = self._old_slot_text
            self.editing_slot = False
            return True
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.editing_slot = False
            if self.slot_texts[slot].strip():
                self.slot_status[slot] = True
            self.quicksave_slot = slot
            self.deactivate_menu()
            self._play_sound(SFX_DOOR_CLOSE)
            if self.on_save_game is not None:
                self.on_save_game(slot, self.slot_texts[slot])
            return True
        elif char_typed and len(cur_text) < SLOTTEXTLEN:
            ch = char_typed.upper()
            if ch.isalnum() or ch in (" ", "-", "_", "!", ".", ","):
                self.slot_texts[slot] = cur_text + ch
                return True
        return True

    def _confirm_prompt(self) -> None:
        """Handle prompt confirmation ('Y' / Enter)."""
        prompt = self.prompt_type
        self.prompt_type = PromptType.NONE

        if prompt == PromptType.QUIT:
            if self.on_quit_game is not None:
                self.on_quit_game()
        elif prompt == PromptType.END_GAME:
            if self.on_end_game is not None:
                self.on_end_game()
        elif prompt == PromptType.SUICIDE:
            if self.on_suicide is not None:
                self.on_suicide()
        elif prompt == PromptType.QUICKSAVE:
            if self.quicksave_slot >= 0 and self.on_save_game is not None:
                self.on_save_game(self.quicksave_slot, self.slot_texts[self.quicksave_slot])
        elif prompt == PromptType.QUICKLOAD:
            if self.quickload_slot >= 0 and self.on_load_game is not None:
                self.on_load_game(self.quickload_slot)

    # -------------------------------------------------------------------------
    # Key Rebinding Helpers
    # -------------------------------------------------------------------------

    def get_action_key_name(self, action_name: str | None) -> str:
        """Return human-readable key name for a given control action."""
        if not action_name:
            return ""
        name_map: dict[str, int] = {
            "FORWARD": self.bindings.forward,
            "BACKWARD": self.bindings.backward,
            "STRAFE LEFT": self.bindings.strafe_left,
            "STRAFE RIGHT": self.bindings.strafe_right,
            "TURN LEFT": self.bindings.turn_left,
            "TURN RIGHT": self.bindings.turn_right,
            "FIRE": self.bindings.fire,
            "USE": self.bindings.use,
            "JUMP": self.bindings.jump,
            "SPEED": self.bindings.speed,
            "USE ARTIFACT": self.bindings.use_artifact,
            "PREV ARTIFACT": self.bindings.inv_left,
            "NEXT ARTIFACT": self.bindings.inv_right,
        }
        code = name_map.get(action_name, 0)
        try:
            return pygame.key.name(code).upper()
        except Exception:
            return str(code)

    def _assign_rebound_key(self, action_name: str, key_code: int, persist: bool = True) -> None:
        """Assign keycode to KeyBindings."""
        act = action_name.upper().replace("_", " ")
        if act == "FORWARD":
            self.bindings.forward = key_code
        elif act == "BACKWARD":
            self.bindings.backward = key_code
        elif act == "STRAFE LEFT":
            self.bindings.strafe_left = key_code
        elif act == "STRAFE RIGHT":
            self.bindings.strafe_right = key_code
        elif act == "TURN LEFT":
            self.bindings.turn_left = key_code
        elif act == "TURN RIGHT":
            self.bindings.turn_right = key_code
        elif act == "FIRE":
            self.bindings.fire = key_code
        elif act == "USE":
            self.bindings.use = key_code
        elif act == "JUMP":
            self.bindings.jump = key_code
        elif act == "SPEED":
            self.bindings.speed = key_code
        elif act in ("USE ARTIFACT",):
            self.bindings.use_artifact = key_code
        elif act in ("PREV ARTIFACT", "INV LEFT"):
            self.bindings.inv_left = key_code
        elif act in ("NEXT ARTIFACT", "INV RIGHT"):
            self.bindings.inv_right = key_code

        if persist:
            if hasattr(self.config.controls, "bindings"):
                self.config.controls.bindings[action_name] = key_code
            self.save_config_to_disk()
            self._notify_config_changed()

    # -------------------------------------------------------------------------
    # Mouse Navigation
    # -------------------------------------------------------------------------

    def handle_mouse_move(self, vx: int, vy: int) -> bool:
        """Update cursor selection based on virtual (320x200) mouse position."""
        if not self.active or self.prompt_type != PromptType.NONE or self.info_type > 0:
            return False

        menu = self.current_menu
        if not menu.items:
            return False

        for idx, item in enumerate(menu.items):
            if item.item_type == ItemType.EMPTY:
                continue
            item_y = menu.y + idx * ITEM_HEIGHT
            if item_y <= vy < item_y + ITEM_HEIGHT:
                if menu.cur_pos != idx:
                    menu.cur_pos = idx
                    self._play_sound(SFX_HAMMER_HIT)
                return True
        return False

    def handle_mouse_click(self, vx: int, vy: int, button: int = 1) -> bool:
        """Handle mouse click in virtual (320x200) coordinates."""
        if self.info_type > 0:
            return self.handle_key(pygame.K_RETURN)

        if self.prompt_type != PromptType.NONE:
            if button == 1:
                # Top half confirms, bottom half cancels
                if vy < 120:
                    self._confirm_prompt()
                else:
                    self.prompt_type = PromptType.NONE
                return True
            return False

        if not self.active:
            return False

        menu = self.current_menu
        if button == 1:
            # Check if clicked on a slider
            if 0 <= menu.cur_pos < len(menu.items):
                item = menu.items[menu.cur_pos]
                if item.item_type == ItemType.LRFUNC:
                    item_y = menu.y + menu.cur_pos * ITEM_HEIGHT
                    slider_x = menu.x + 24
                    slider_w = item.slider_width * 8
                    in_y = item_y <= vy < item_y + ITEM_HEIGHT
                    in_x = slider_x <= vx <= slider_x + slider_w
                    if in_y and in_x:
                        slot = max(item.slider_min, min(item.slider_max, (vx - slider_x) // 8))
                        if item.value_setter is not None:
                            item.value_setter(slot)
                            self._play_sound(SFX_PICKUP_KEY)
                        return True

            self.select_current_item()
            return True

        elif button == 3:  # Right click acts as back/escape
            return self.handle_key(pygame.K_ESCAPE)

        elif button == 4:  # Wheel up
            is_lr = (
                0 <= menu.cur_pos < len(menu.items)
                and menu.items[menu.cur_pos].item_type == ItemType.LRFUNC
            )
            if is_lr:
                return self.handle_key(pygame.K_RIGHT)
            return self.handle_key(pygame.K_UP)

        elif button == 5:  # Wheel down
            is_lr = (
                0 <= menu.cur_pos < len(menu.items)
                and menu.items[menu.cur_pos].item_type == ItemType.LRFUNC
            )
            if is_lr:
                return self.handle_key(pygame.K_LEFT)
            return self.handle_key(pygame.K_DOWN)

        return False

    # -------------------------------------------------------------------------
    # Rendering (MN_Drawer)
    # -------------------------------------------------------------------------

    def draw(self, overlay: Overlay2D) -> None:
        """Render the active menu or prompt to overlay (MN_Drawer)."""
        # 1. Info Screen
        if self.info_type > 0:
            self._draw_info_screen(overlay)
            return

        # 2. Confirmation Prompts
        if self.prompt_type != PromptType.NONE:
            self._draw_prompt(overlay)
            return

        if not self.active:
            return

        # 3. Specific menu decoration drawings
        menu = self.current_menu
        if menu.menu_type == MenuType.MAIN:
            self._draw_main_menu(overlay)
        elif menu.menu_type == MenuType.CLASS:
            self._draw_class_menu(overlay)
        elif menu.menu_type == MenuType.SKILL:
            self._draw_skill_menu(overlay)
        elif menu.menu_type == MenuType.OPTIONS:
            self._draw_options_menu(overlay)
        elif menu.menu_type == MenuType.OPTIONS2:
            self._draw_options2_menu(overlay)
        elif menu.menu_type in (MenuType.LOAD, MenuType.SAVE):
            self._draw_file_menu(overlay)
        elif menu.menu_type == MenuType.CONTROLS:
            self._draw_controls_menu(overlay)
        elif menu.menu_type == MenuType.VIDEO:
            self._draw_video_menu(overlay)

        # 4. Standard menu items
        x = menu.x
        y = menu.y
        for idx, itm in enumerate(menu.items):
            if itm.item_type != ItemType.EMPTY and itm.text:
                if menu.menu_type == MenuType.CONTROLS:
                    # Draw action and bound key
                    key_str = self.get_action_key_name(itm.text)
                    if self.rebinding_action == itm.text:
                        overlay.draw_text_a(f"{itm.text}: [PRESS KEY]", x, y, yellow=True)
                    else:
                        overlay.draw_text_a(f"{itm.text}: {key_str}", x, y)
                elif menu.menu_type not in (MenuType.LOAD, MenuType.SAVE):
                    overlay.draw_text_b(itm.text, x, y)

            if itm.item_type == ItemType.LRFUNC and itm.slider_width > 0:
                val = itm.value_getter() if itm.value_getter is not None else 0
                self._draw_slider(overlay, menu.x, menu.y, idx, itm.slider_width, val)

            y += ITEM_HEIGHT

        # 5. Selector skull cursor
        if menu.items and 0 <= menu.cur_pos < len(menu.items):
            cursor_name = "M_SLCTR1" if (self.menu_time & 16) else "M_SLCTR2"
            cursor_pic = self.get_patch(cursor_name)
            sel_x = menu.x + SELECTOR_XOFFSET
            sel_y = menu.y + (menu.cur_pos * ITEM_HEIGHT) + SELECTOR_YOFFSET
            overlay.draw_picture(cursor_pic, sel_x, sel_y)

    def _draw_main_menu(self, overlay: Overlay2D) -> None:
        """Draw Main Menu logo and animated burning skulls (DrawMainMenu)."""
        logo = self.get_patch("M_HTIC")
        overlay.draw_picture(logo, 88, 0)

        frame = (self.menu_time // 5) % 7
        skull1 = self.get_patch(MAULO_FRAMES[(frame + 2) % 7])
        skull2 = self.get_patch(MAULO_FRAMES[frame])
        overlay.draw_picture(skull1, 37, 80)
        overlay.draw_picture(skull2, 278, 80)

    def _draw_class_menu(self, overlay: Overlay2D) -> None:
        """Draw class selection preview box and animated walking actor (DrawClassMenu)."""
        overlay.draw_text_b("CHOOSE CLASS:", 34, 24)

        cls_idx = max(0, min(2, self.current_menu.cur_pos))
        box_pic = self.get_patch(CLASS_BOX_LUMPS[cls_idx])
        overlay.draw_picture(box_pic, 174, 8)

        walk_frame = ((self.menu_time >> 3) & 3) + 1
        walk_lump = f"{CLASS_WALK_PREFIXES[cls_idx]}{walk_frame}"
        walk_pic = self.get_patch(walk_lump)
        overlay.draw_picture(walk_pic, 174 + 24, 8 + 12)

    def _draw_skill_menu(self, overlay: Overlay2D) -> None:
        """Draw Skill Menu title (DrawSkillMenu)."""
        overlay.draw_text_b("CHOOSE SKILL LEVEL:", 74, 16)

    def _draw_options_menu(self, overlay: Overlay2D) -> None:
        """Draw Options Menu ON/OFF text (DrawOptionsMenu)."""
        msg_text = "ON" if self.messages_on else "OFF"
        overlay.draw_text_b(msg_text, 196, 50)
        inv_text = "ON" if self.config.controls.invert_mouse else "OFF"
        overlay.draw_text_b(inv_text, 196, 170)

    def _draw_options2_menu(self, overlay: Overlay2D) -> None:
        """Draw Options2 Menu headers."""
        pass

    def _draw_video_menu(self, overlay: Overlay2D) -> None:
        """Draw Video Menu title and option values."""
        overlay.draw_text_b("VIDEO OPTIONS", 160, 6, center=True)
        menu = self.current_menu
        val_x = 200

        # 0: Resolution
        w, h = COMMON_RESOLUTIONS[self._get_resolution_idx()]
        overlay.draw_text_b(f"{w}X{h}", val_x, menu.y)

        # 1: Fullscreen
        fs_str = "ON" if self._get_fullscreen() else "OFF"
        overlay.draw_text_b(fs_str, val_x, menu.y + ITEM_HEIGHT)

        # 2: FOV
        fov_val = int(round(self.config.video.fov))
        overlay.draw_text_b(str(fov_val), val_x, menu.y + 2 * ITEM_HEIGHT)

        # 3: Low-res mode
        lr_str = "ON" if self._get_lowres() else "OFF"
        overlay.draw_text_b(lr_str, val_x, menu.y + 3 * ITEM_HEIGHT)

        # 4: Pixel aspect
        pa_str = "ON" if self._get_aspect() else "OFF"
        overlay.draw_text_b(pa_str, val_x, menu.y + 4 * ITEM_HEIGHT)

        # 5: True pitch
        tp_str = "ON" if self._get_true_pitch() else "OFF"
        overlay.draw_text_b(tp_str, val_x, menu.y + 5 * ITEM_HEIGHT)

        # 6: Filtering
        flt_str = "SMOOTH" if self._get_filter() else "NEAREST"
        overlay.draw_text_b(flt_str, val_x, menu.y + 6 * ITEM_HEIGHT)

    def _draw_file_menu(self, overlay: Overlay2D) -> None:
        """Draw Save/Load slots with slot boxes and descriptions (DrawFileSlots)."""
        is_load = self.current_menu_type == MenuType.LOAD
        title = "LOAD GAME" if is_load else "SAVE GAME"
        w = overlay.fonts.font_b.text_width(title)
        overlay.draw_text_b(title, 160 - w // 2, 10)

        menu = self.current_menu
        x = menu.x
        y = menu.y
        slot_bg = self.get_patch("M_FSLOT")

        for idx in range(6):
            overlay.draw_picture(slot_bg, x, y)
            text = self.slot_texts[idx]
            if self.editing_slot and self.current_slot == idx:
                cursor_char = ASCII_CURSOR if (self.menu_time & 8) else ""
                overlay.draw_text_a(f"{text}{cursor_char}", x + 5, y + 5, yellow=True)
            elif self.slot_status[idx] and text:
                overlay.draw_text_a(text, x + 5, y + 5)
            y += ITEM_HEIGHT

    def _draw_controls_menu(self, overlay: Overlay2D) -> None:
        """Draw Controls Menu header."""
        overlay.draw_text_b("KEY CONTROLS", 160, 8, center=True)

    def _draw_slider(
        self,
        overlay: Overlay2D,
        menu_x: int,
        menu_y: int,
        item_index: int,
        width: int,
        slot: int,
    ) -> None:
        """Render a slider bar with left cap, notched segments, right cap, and knob (DrawSlider)."""
        x = menu_x + 24
        y = menu_y + 2 + (item_index * ITEM_HEIGHT)

        left_cap = self.get_patch("M_SLDLT")
        overlay.draw_picture(left_cap, x - 32, y)

        x2 = x
        for count in range(width, 0, -1):
            mid_name = "M_SLDMD1" if (count & 1) else "M_SLDMD2"
            mid_patch = self.get_patch(mid_name)
            overlay.draw_picture(mid_patch, x2, y)
            x2 += 8

        right_cap = self.get_patch("M_SLDRT")
        overlay.draw_picture(right_cap, x2, y)

        knob = self.get_patch("M_SLDKB")
        overlay.draw_picture(knob, x + 4 + slot * 8, y + 7)

    def _draw_prompt(self, overlay: Overlay2D) -> None:
        """Render confirmation modal question."""
        msg = PROMPT_MESSAGES.get(self.prompt_type, "")
        overlay.draw_text_a(msg, 160, 80, center=True, yellow=True)

        if self.prompt_type == PromptType.QUICKSAVE and self.quicksave_slot >= 0:
            slot_name = self.slot_texts[self.quicksave_slot]
            overlay.draw_text_a(f'"{slot_name}" ?', 160, 92, center=True)
        elif self.prompt_type == PromptType.QUICKLOAD and self.quickload_slot >= 0:
            slot_name = self.slot_texts[self.quickload_slot]
            overlay.draw_text_a(f'"{slot_name}" ?', 160, 92, center=True)

        overlay.draw_text_a("PRESS Y OR ENTER TO CONFIRM, N TO CANCEL", 160, 110, center=True)

    def _draw_info_screen(self, overlay: Overlay2D) -> None:
        """Render help/info screen page."""
        info_lump = f"HELP{self.info_type}" if self.info_type in (1, 2) else "CREDIT"
        patch = self.get_patch(info_lump)
        overlay.draw_picture(patch, 0, 0)
        overlay.draw_text_a(
            "PRESS ANY KEY FOR NEXT PAGE, ESC TO EXIT",
            160,
            VIRTUAL_HEIGHT - 12,
            center=True,
            yellow=True,
        )
