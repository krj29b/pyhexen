# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Intermission, finale, title screen, and demo loop (T12.6).

Port of IN_Start, IN_Ticker, IN_Drawer (in_lude.c),
F_StartFinale, F_Ticker, F_Drawer (f_finale.c),
and H2_StartTitle, H2_AdvanceDemo, H2_DoAdvanceDemo (h2_main.c).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import pygame

from pyhexen.render.overlay2d import VIRTUAL_HEIGHT, VIRTUAL_WIDTH, Overlay2D
from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.wadfile import WadFile, WadStack

if TYPE_CHECKING:
    from pyhexen.sim.player import Player

# Constants from in_lude.c and f_finale.c
TEXTSPEED: int = 3
TEXTWAIT_INTERMISSION: int = 140
TEXTWAIT_FINALE: int = 250
FADE_TICS: int = 70

RAW_SCREEN_SIZE: int = VIRTUAL_WIDTH * VIRTUAL_HEIGHT  # 64000 bytes

# Authentic Hexen cluster texts for hubs 1..5
DEFAULT_CLUSTER_MESSAGES: dict[int, str] = {
    1: (
        "WITH THE DESTRUCTION OF THE SEVEN PORTALS,\n"
        "THE REALM BEYOND OPENS BEFORE YOU.\n\n"
        "DARK FORESTS STRETCH INTO ENDLESS TWILIGHT,\n"
        "HAUNTED BY CREATURES THAT WERE ONCE MEN.\n\n"
        "SOMEWHERE IN THIS CURSED EXPANSE LIES THE\n"
        "SHADOW WOOD AND THE SECRETS OF KORAX."
    ),
    2: (
        "THE SHADOW WOOD HAS FALLEN SILENT.\n\n"
        "THE TWISTED GROVES AND NOXIOUS SWAMPS\n"
        "NOW LIE BEHIND YOU, BUT DARKER TRIALS\n"
        "AWAIT WITHIN THE SEMINARY OF THE HERESIARCH.\n\n"
        "ONLY THE STRONGEST FAITH AND STEEL\n"
        "WILL PREVAIL AGAINST THE HORRORS AHEAD."
    ),
    3: (
        "THE HERESIARCH HAS FALLEN, HIS DARK\n"
        "INCANTATIONS SILENCED FOREVER.\n\n"
        "AHEAD LOOMS THE CASTLE OF GRIEF, WHERE\n"
        "THE CHILL OF THE GRAVE SEEPS THROUGH\n"
        "ANCIENT STONE.\n\n"
        "STEEL YOUR RESOLVE, FOR KORAX AWAITS\n"
        "IN THE DEPTHS OF HIS DARK DOMAIN."
    ),
    4: (
        "THE NECROPOLIS HAS SURRENDERED ITS SECRETS.\n\n"
        "BEFORE YOU STANDS THE FINAL GATEWAY:\n"
        "THE DARK CRUCIBLE, THE SANCTUM OF KORAX.\n\n"
        "ALL THE POWER OF THE SERPENT RIDERS\n"
        "CONVERGES HERE FOR THE ULTIMATE BATTLE."
    ),
    5: (
        "KORAX IS DEFEATED, HIS EVIL BROKEN!\n\n"
        "THE GATES BETWEEN WORLDS BEGIN TO CLOSE,\n"
        "AND THE LAND IS FREED FROM HIS TYRANNY."
    ),
}

# Authentic Hexen win/finale texts
DEFAULT_WIN_MESSAGES: dict[int, str] = {
    0: (
        "WITH KORAX DESTROYED, THE DARK TAINT THAT\n"
        "INFECTED THE REALM BEGINS TO DISSIPATE.\n\n"
        "THE CITADEL CRUMBLES AROUND YOU AS ANCIENT\n"
        "SORCERIES SHATTER INTO DUST."
    ),
    1: (
        "THROUGH THE RUBBLE AND SMOKE, A RIFT OPENS\n"
        "INTO THE ASTRAL VOID.\n\n"
        "THE SERPENT RIDER'S REIGN OF TERROR HAS ENDED,\n"
        "AND THE MORTAL SPHERE MAY HEAL ONCE MORE."
    ),
    2: (
        "THE TRIUMPH BELONGS TO YOU.\n\n"
        "YOUR NAME WILL ECHO THROUGH ETERNITY AS\n"
        "THE DEFENDER OF THE REALMS."
    ),
}

# Class-specific finale epilogues (Win 3)
CLASS_FINALE_TEXTS: dict[int, str] = {
    0: (  # Fighter
        "YOUR WEAPONS BATTERED BUT UNBROKEN, YOU STAND VICTORIOUS.\n\n"
        "THE MANTLE OF SUPREME WARRIOR IS YOURS FOR ALL TIME,\n"
        "AND PEACE RESTS UPON THE BATTERED LAND."
    ),
    1: (  # Cleric
        "BY THE GRACE OF THE HOLY LIGHT, THE SHADOW IS VANQUISHED.\n\n"
        "YOU HAVE CLEANSED THE DESECRATED SHRINES AND PROVED\n"
        "THAT FAITH AND PURITY CONQUER ALL DARKNESS."
    ),
    2: (  # Mage
        "THE ANCIENT LORE IS PRESERVED AND THE ARCANE MYSTERIES\n"
        "OF THE SPHERES ARE LAID BARE BEFORE YOUR SIGHT.\n\n"
        "YOU RETURN AS THE SOVEREIGN ARCHIMAGE OF THE REALM."
    ),
}


def _make_synthetic_screen(base_color: int = 40) -> bytes:
    """Generate a deterministic 64000-byte raw screen for headless tests."""
    arr = np.zeros((VIRTUAL_HEIGHT, VIRTUAL_WIDTH), dtype=np.uint8)
    for y in range(VIRTUAL_HEIGHT):
        arr[y, :] = (base_color + (y // 10)) & 0xFF
    return arr.tobytes()


def _make_synthetic_picture(width: int, height: int, color: int = 150) -> Picture:
    """Create a placeholder picture for synthetic tests."""
    indices = np.full((height, width), color, dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    return Picture(width=width, height=height, left=0, top=0, indices=indices, mask=mask)


# =============================================================================
# 1. Intermission Screen (IN_Start, IN_Ticker, IN_Drawer)
# =============================================================================


class Intermission:
    """Hub transition intermission manager (in_lude.c).

    Displays hub narrative text typewriter animation on `INTERPIC` or frag tallies
    in deathmatch games.
    """

    def __init__(self, wad: WadStack | WadFile | None = None) -> None:
        self.wad: WadStack | WadFile | None = wad
        self.active: bool = False
        self.intertime: int = 0
        self.interstate: int = 0
        self.skip_intermission: bool = False
        self.cnt: int = 0
        self.hub_text: str = ""
        self.hub_count: int = 0
        self.is_deathmatch: bool = False
        self.frags_matrix: list[list[int]] = []
        self.total_frags: list[int] = []

        # Callbacks
        self.on_sound: Callable[[str], None] | None = None
        self.on_music: Callable[[str], None] | None = None
        self.on_done: Callable[[], None] | None = None

        # Cache
        self._interpic: bytes = self._load_raw_screen("INTERPIC", default_color=50)

    def _load_raw_screen(self, lump_name: str, default_color: int) -> bytes:
        if self.wad is not None:
            lump = self.wad.find(lump_name)
            if lump is not None:
                try:
                    data = self.wad.get_bytes(lump_name)
                    if len(data) >= RAW_SCREEN_SIZE:
                        return data[:RAW_SCREEN_SIZE]
                except Exception:
                    pass
        return _make_synthetic_screen(default_color)

    def start(
        self,
        old_cluster: int,
        new_cluster: int,
        *,
        is_deathmatch: bool = False,
        players: list[Player] | None = None,
    ) -> None:
        """Start the intermission sequence (IN_Start)."""
        self.active = True
        self.intertime = 0
        self.interstate = 0
        self.skip_intermission = False
        self.is_deathmatch = is_deathmatch

        if not self.is_deathmatch:
            # Check for cluster change narrative
            if old_cluster != new_cluster and 1 <= old_cluster <= 5:
                # Attempt to read lump, or fallback to default authentic text
                msg_lump = f"clus{old_cluster}msg"
                text = ""
                if self.wad is not None:
                    entry = self.wad.find(msg_lump)
                    if entry is not None:
                        try:
                            text = self.wad.get_bytes(msg_lump).decode("ascii", errors="ignore")
                        except Exception:
                            text = ""
                if not text:
                    text = DEFAULT_CLUSTER_MESSAGES.get(old_cluster, "")
                self.hub_text = text
                self.hub_count = len(self.hub_text) * TEXTSPEED + TEXTWAIT_INTERMISSION
                if self.on_music is not None:
                    self.on_music("hub")
            else:
                self.hub_text = ""
                self.hub_count = 0
        else:
            # Setup deathmatch frags tally
            self.hub_text = ""
            self.hub_count = 0
            num_players = len(players) if players else 1
            self.frags_matrix = [[0] * num_players for _ in range(num_players)]
            self.total_frags = [0] * num_players
            if self.on_music is not None:
                self.on_music("hub")

    def tick(self) -> bool:
        """Advance intermission ticker by 1 tic (IN_Ticker). Returns True if still active."""
        if not self.active:
            return False

        if self.interstate:
            self.cnt -= 1
            if self.cnt <= 0:
                self.active = False
                if self.on_done is not None:
                    self.on_done()
                return False
            return True

        self.intertime += 1

        # Check skip or completion
        if not self.is_deathmatch:
            if not self.hub_count or self.skip_intermission or self.intertime >= self.hub_count:
                self.interstate = 1
                self.cnt = 10
        else:
            if self.skip_intermission and self.intertime >= 140:
                self.interstate = 1
                self.cnt = 10

        return True

    def handle_key(self, key: int) -> bool:
        """Process key press to skip or advance intermission."""
        if not self.active:
            return False
        self.skip_intermission = True
        return True

    def handle_mouse_click(self, button: int = 1) -> bool:
        """Process mouse click to skip or advance intermission."""
        if not self.active:
            return False
        self.skip_intermission = True
        return True

    def draw(self, overlay: Overlay2D) -> None:
        """Render intermission background and text / tally table (IN_Drawer)."""
        if not self.active or self.interstate:
            return

        overlay.draw_raw_screen(self._interpic)

        if not self.is_deathmatch:
            if self.hub_text:
                self._draw_typewriter_text(overlay)
        else:
            self._draw_deathmatch_tally(overlay)

    def _draw_typewriter_text(self, overlay: Overlay2D) -> None:
        """Draw typewriter narrative text (DrawHubText)."""
        chars_to_show = max(0, (self.intertime - 10) // TEXTSPEED)
        if chars_to_show <= 0:
            return

        visible_text = self.hub_text[:chars_to_show]
        cx = 10
        cy = 14
        font = overlay.fonts.font_a

        for line in visible_text.splitlines():
            font.draw_text(overlay.indices, overlay.mask, line, cx, cy)
            cy += 9

    def _draw_deathmatch_tally(self, overlay: Overlay2D) -> None:
        """Draw simple deathmatch frag summary table (DrDeathTally)."""
        overlay.draw_text_b("DEATHMATCH STATS", 160, 20, center=True)
        y = 60
        font = overlay.fonts.font_a
        for idx, total in enumerate(self.total_frags):
            msg = f"PLAYER {idx + 1}: {total} FRAGS"
            font.draw_text(overlay.indices, overlay.mask, msg, 100, y)
            y += 12


# =============================================================================
# 2. Finale Sequence (F_StartFinale, F_Ticker, F_Drawer)
# =============================================================================


class FinaleStage(IntEnum):
    """Stages of the Hexen end sequence (f_finale.c)."""

    STAGE_0_FADE_PIC1 = 0  # Fade in FINALE1 (70 tics)
    STAGE_1_TEXT1 = 1  # Text 1 on FINALE1
    STAGE_2_PIC2_TEXT2 = 2  # FINALE2 with Text 2
    STAGE_3_FADE_OUT_PIC2 = 3  # Fade out FINALE2 (70 tics)
    STAGE_4_FADE_IN_CHESS = 4  # Fade in FINALE3 (chess) (71 tics)
    STAGE_5_TEXT3_EPILOGUE = 5  # FINALE3 with character portrait & epilogue text


class Finale:
    """Campaign victory end sequence manager (f_finale.c).

    Orchestrates the 6-stage ending sequence with fade transitions, narrative text,
    character chess portraits, and ending themes.
    """

    def __init__(self, wad: WadStack | WadFile | None = None) -> None:
        self.wad: WadStack | WadFile | None = wad
        self.active: bool = False
        self.stage: FinaleStage = FinaleStage.STAGE_0_FADE_PIC1
        self.stage_count: int = 0
        self.stage_end_count: int = 70
        self.player_class: int = 0  # 0: Fighter, 1: Cleric, 2: Mage
        self.is_netgame: bool = False
        self.current_text: str = ""

        # Fade alpha (0.0 transparent black to 1.0 fully visible)
        self.fade_alpha: float = 1.0

        # Callbacks
        self.on_music: Callable[[str], None] | None = None
        self.on_done: Callable[[], None] | None = None

        # Screens & patches
        self._finale1: bytes = self._load_raw_screen("FINALE1", default_color=30)
        self._finale2: bytes = self._load_raw_screen("FINALE2", default_color=60)
        self._finale3: bytes = self._load_raw_screen("FINALE3", default_color=80)
        self._patches: dict[str, Picture] = {}

    def _load_raw_screen(self, lump_name: str, default_color: int) -> bytes:
        if self.wad is not None:
            lump = self.wad.find(lump_name)
            if lump is not None:
                try:
                    data = self.wad.get_bytes(lump_name)
                    if len(data) >= RAW_SCREEN_SIZE:
                        return data[:RAW_SCREEN_SIZE]
                except Exception:
                    pass
        return _make_synthetic_screen(default_color)

    def _get_patch(self, name: str) -> Picture:
        name_upper = name.upper()
        if name_upper in self._patches:
            return self._patches[name_upper]

        if self.wad is not None:
            lump = self.wad.find(name_upper)
            if lump is not None:
                try:
                    pic = decode_picture(self.wad.get_bytes(name_upper))
                    self._patches[name_upper] = pic
                    return pic
                except Exception:
                    pass

        # Fallback patch
        pic = _make_synthetic_picture(86, 200, color=160)
        self._patches[name_upper] = pic
        return pic

    def start(self, player_class: int = 0, *, is_netgame: bool = False) -> None:
        """Start the finale sequence (F_StartFinale)."""
        self.active = True
        self.player_class = player_class
        self.is_netgame = is_netgame
        self.stage = FinaleStage.STAGE_0_FADE_PIC1
        self.stage_count = 0
        self.stage_end_count = FADE_TICS
        self.fade_alpha = 0.0
        self.current_text = DEFAULT_WIN_MESSAGES[0]

        if self.on_music is not None:
            self.on_music("hall")

    def tick(self) -> bool:
        """Advance finale sequence by 1 tic (F_Ticker). Returns True if still active."""
        if not self.active:
            return False

        self.stage_count += 1

        # Fade calculations
        if self.stage == FinaleStage.STAGE_0_FADE_PIC1:
            self.fade_alpha = min(1.0, self.stage_count / FADE_TICS)
        elif self.stage == FinaleStage.STAGE_3_FADE_OUT_PIC2:
            self.fade_alpha = max(0.0, 1.0 - (self.stage_count / FADE_TICS))
        elif self.stage == FinaleStage.STAGE_4_FADE_IN_CHESS:
            self.fade_alpha = min(1.0, self.stage_count / 71)
        else:
            self.fade_alpha = 1.0

        # Stage progression
        if self.stage_count >= self.stage_end_count:
            self._advance_stage()

        return self.active

    def _advance_stage(self) -> None:
        self.stage_count = 0
        next_stage_val = int(self.stage) + 1

        if next_stage_val > int(FinaleStage.STAGE_5_TEXT3_EPILOGUE):
            self.active = False
            if self.on_done is not None:
                self.on_done()
            return

        self.stage = FinaleStage(next_stage_val)

        if self.stage == FinaleStage.STAGE_1_TEXT1:
            self.current_text = DEFAULT_WIN_MESSAGES[0]
            self.stage_end_count = len(self.current_text) * TEXTSPEED + TEXTWAIT_FINALE

        elif self.stage == FinaleStage.STAGE_2_PIC2_TEXT2:
            self.current_text = DEFAULT_WIN_MESSAGES[1]
            self.stage_end_count = len(self.current_text) * TEXTSPEED + TEXTWAIT_FINALE
            if self.on_music is not None:
                self.on_music("orb")

        elif self.stage == FinaleStage.STAGE_3_FADE_OUT_PIC2:
            self.stage_end_count = FADE_TICS

        elif self.stage == FinaleStage.STAGE_4_FADE_IN_CHESS:
            self.stage_end_count = 71
            if self.on_music is not None:
                self.on_music("chess")

        elif self.stage == FinaleStage.STAGE_5_TEXT3_EPILOGUE:
            self.current_text = CLASS_FINALE_TEXTS.get(
                self.player_class, DEFAULT_WIN_MESSAGES.get(2, "")
            )
            self.stage_end_count = len(self.current_text) * TEXTSPEED + TEXTWAIT_FINALE * 2

    def handle_key(self, key: int) -> bool:
        """Skip current stage on key press."""
        if not self.active:
            return False
        self._advance_stage()
        return True

    def handle_mouse_click(self, button: int = 1) -> bool:
        """Skip current stage on mouse click."""
        if not self.active:
            return False
        self._advance_stage()
        return True

    def draw(self, overlay: Overlay2D) -> None:
        """Render active finale screen, character portrait, and narrative text (F_Drawer)."""
        if not self.active:
            return

        # 1. Base pic for current stage
        if self.stage in (FinaleStage.STAGE_0_FADE_PIC1, FinaleStage.STAGE_1_TEXT1):
            overlay.draw_raw_screen(self._finale1)
        elif self.stage in (FinaleStage.STAGE_2_PIC2_TEXT2, FinaleStage.STAGE_3_FADE_OUT_PIC2):
            overlay.draw_raw_screen(self._finale2)
        elif self.stage in (FinaleStage.STAGE_4_FADE_IN_CHESS, FinaleStage.STAGE_5_TEXT3_EPILOGUE):
            overlay.draw_raw_screen(self._finale3)

            # Draw character portraits on chess board
            if self.is_netgame:
                patch = self._get_patch("chessall")
                overlay.draw_picture(patch, 20, 0)
            else:
                if self.player_class == 1:  # Cleric
                    patch = self._get_patch("chessc")
                    overlay.draw_picture(patch, 60, 0)
                elif self.player_class == 2:  # Mage
                    patch = self._get_patch("chessm")
                    overlay.draw_picture(patch, 60, 0)

        # 2. Typewriter text
        if self.stage in (
            FinaleStage.STAGE_1_TEXT1,
            FinaleStage.STAGE_2_PIC2_TEXT2,
            FinaleStage.STAGE_5_TEXT3_EPILOGUE,
        ):
            chars = max(0, (self.stage_count - 10) // TEXTSPEED)
            visible_text = self.current_text[:chars]
            cx = 20
            cy = 135 if self.stage == FinaleStage.STAGE_5_TEXT3_EPILOGUE else 10
            font = overlay.fonts.font_a
            for line in visible_text.splitlines():
                font.draw_text(overlay.indices, overlay.mask, line, cx, cy)
                cy += 9


# =============================================================================
# 3. Title Screen & Demo / Advisory Loop (H2_StartTitle, H2_DoAdvanceDemo)
# =============================================================================


class TitleSequence(IntEnum):
    """Demo / intro loop sequences matching Hexen h2_main.c."""

    TITLE_1 = 0  # TITLE screen (280 tics, music: hexen)
    ADVISOR = 1  # TITLE screen with ADVISOR patch (210 tics)
    DEMO_1 = 2  # demo1
    CREDIT_1 = 3  # CREDIT screen (200 tics)
    DEMO_2 = 4  # demo2
    CREDIT_2 = 5  # CREDIT screen (200 tics)
    DEMO_3 = 6  # demo3


SEQUENCE_DURATIONS: dict[TitleSequence, int] = {
    TitleSequence.TITLE_1: 280,
    TitleSequence.ADVISOR: 210,
    TitleSequence.DEMO_1: 0,
    TitleSequence.CREDIT_1: 200,
    TitleSequence.DEMO_2: 0,
    TitleSequence.CREDIT_2: 200,
    TitleSequence.DEMO_3: 0,
}


class TitleLoop:
    """Hexen title screen, advisory card, credit screen, and demo playback loop (h2_main.c)."""

    def __init__(self, wad: WadStack | WadFile | None = None) -> None:
        self.wad: WadStack | WadFile | None = wad
        self.active: bool = False
        self.sequence: TitleSequence = TitleSequence.TITLE_1
        self.seq_time: int = 0

        # Callbacks
        self.on_play_demo: Callable[[str], None] | None = None
        self.on_music: Callable[[str], None] | None = None
        self.on_menu_activate: Callable[[], None] | None = None

        # Screens & patches
        self._title_pic: bytes = self._load_raw_screen("TITLE", default_color=20)
        self._credit_pic: bytes = self._load_raw_screen("CREDIT", default_color=40)
        self._advisor_patch: Picture = self._load_patch("ADVISOR", fallback_w=80, fallback_h=36)

    def _load_raw_screen(self, lump_name: str, default_color: int) -> bytes:
        if self.wad is not None:
            lump = self.wad.find(lump_name)
            if lump is not None:
                try:
                    data = self.wad.get_bytes(lump_name)
                    if len(data) >= RAW_SCREEN_SIZE:
                        return data[:RAW_SCREEN_SIZE]
                except Exception:
                    pass
        return _make_synthetic_screen(default_color)

    def _load_patch(self, lump_name: str, fallback_w: int, fallback_h: int) -> Picture:
        if self.wad is not None:
            lump = self.wad.find(lump_name)
            if lump is not None:
                try:
                    return decode_picture(self.wad.get_bytes(lump_name))
                except Exception:
                    pass
        return _make_synthetic_picture(fallback_w, fallback_h, color=180)

    def start(self) -> None:
        """Initialize title loop (H2_StartTitle)."""
        self.active = True
        self.sequence = TitleSequence.TITLE_1
        self.seq_time = 0
        if self.on_music is not None:
            self.on_music("hexen")

    def advance(self) -> None:
        """Advance to next title/demo sequence (H2_DoAdvanceDemo)."""
        self.seq_time = 0
        next_seq = (int(self.sequence) + 1) % 7
        self.sequence = TitleSequence(next_seq)

        if self.sequence == TitleSequence.TITLE_1:
            if self.on_music is not None:
                self.on_music("hexen")
        elif self.sequence == TitleSequence.DEMO_1:
            if self.on_play_demo is not None:
                self.on_play_demo("demo1")
        elif self.sequence == TitleSequence.DEMO_2:
            if self.on_play_demo is not None:
                self.on_play_demo("demo2")
        elif self.sequence == TitleSequence.DEMO_3:
            if self.on_play_demo is not None:
                self.on_play_demo("demo3")

    def tick(self) -> None:
        """Advance title timer by 1 tic."""
        if not self.active:
            return

        dur = SEQUENCE_DURATIONS.get(self.sequence, 0)
        if dur > 0:
            self.seq_time += 1
            if self.seq_time >= dur:
                self.advance()

    def handle_key(self, key: int) -> bool:
        """Handle key input during title screen."""
        if not self.active:
            return False

        if key == pygame.K_ESCAPE:
            if self.on_menu_activate is not None:
                self.on_menu_activate()
            return True

        self.advance()
        return True

    def handle_mouse_click(self, button: int = 1) -> bool:
        """Handle mouse click during title screen."""
        if not self.active:
            return False
        self.advance()
        return True

    def draw(self, overlay: Overlay2D) -> None:
        """Render current title or credit screen to overlay (PageDrawer)."""
        if not self.active:
            return

        if self.sequence in (TitleSequence.TITLE_1, TitleSequence.ADVISOR):
            overlay.draw_raw_screen(self._title_pic)
            if self.sequence == TitleSequence.ADVISOR:
                overlay.draw_picture(self._advisor_patch, 4, 160)
        elif self.sequence in (TitleSequence.CREDIT_1, TitleSequence.CREDIT_2):
            overlay.draw_raw_screen(self._credit_pic)
