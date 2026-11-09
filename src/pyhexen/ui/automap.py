# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Hexen automap system (T12.5).

Port of AM_LevelInit, AM_Start, AM_Stop, AM_Ticker, AM_Drawer, AM_Responder,
AM_drawWalls, AM_drawPlayers, AM_drawThings, and AM_drawGrid (am_map.c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

from pyhexen.core.angles import fine_cosine, fine_sine
from pyhexen.core.fixed import FRACBITS, FRACUNIT, fixed_div, fixed_mul
from pyhexen.render.overlay2d import VIRTUAL_HEIGHT, Overlay2D
from pyhexen.sim.artifacts import PowerType
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import ML_DONTDRAW, ML_MAPPED, ML_SECRET, Level, Line

if TYPE_CHECKING:
    from pyhexen.sim.mobj import Mobj
    from pyhexen.sim.player import Player

# Palette colors from am_map.h
PARCH: int = 103
WHITE: int = 32
WALLCOLORS: int = 96  # Red 1-sided walls
TSWALLCOLORS: int = 40  # Grey 2-sided lines with no height change (cheat)
FDWALLCOLORS: int = 112  # Brown floor height change
CDWALLCOLORS: int = 80  # Yellow ceiling height change
SECRET_WALL_COLOR: int = 0  # Black (when cheating)
GRIDCOLORS: int = 48  # Grey gridlines
GRAYS: int = 40
ALLMAP_LINE_COLOR: int = 43  # Unseen line with allmap powerup (GRAYS + 3)
BLOODRED: int = 177  # Inter-level exit line (special 74, 75)
BLUEKEY: int = 157  # Intra-level teleport line (special 70, 71)
YELLOWKEY: int = 137
GREENKEY: int = 198  # Locked door line (special 13, 83)
THINGCOLORS: int = 8  # Green monster/thing indicators

# Dimensions
FINIT_WIDTH: int = 320
FINIT_HEIGHT: int = 154  # Upper portion above the 46px KeyBar
SBAR_HEIGHT: int = 46

# Scaling and panning constants
PLAYER_RADIUS: int = 16 * FRACUNIT
F_PANINC: int = 4  # Pixels per pan step
ZOOM_IN_MUL: int = int(1.02 * FRACUNIT)
ZOOM_OUT_MUL: int = int(FRACUNIT / 1.02)

# Cohen-Sutherland outcodes
CS_INSIDE: int = 0
CS_LEFT: int = 1
CS_RIGHT: int = 2
CS_BOTTOM: int = 4
CS_TOP: int = 8

# Player arrow geometry from am_data.h
R_ARROW: int = (8 * PLAYER_RADIUS) // 7
PLAYER_ARROW: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((-R_ARROW + R_ARROW // 4, 0), (0, 0)),
    ((-R_ARROW + R_ARROW // 4, R_ARROW // 8), (R_ARROW, 0)),
    ((-R_ARROW + R_ARROW // 4, -R_ARROW // 8), (R_ARROW, 0)),
    ((-R_ARROW + R_ARROW // 4, -R_ARROW // 4), (-R_ARROW + R_ARROW // 4, R_ARROW // 4)),
    ((-R_ARROW + R_ARROW // 8, -R_ARROW // 4), (-R_ARROW + R_ARROW // 8, R_ARROW // 4)),
    ((-R_ARROW + R_ARROW // 8, -R_ARROW // 4), (-R_ARROW + R_ARROW // 4, -R_ARROW // 4)),
    ((-R_ARROW + R_ARROW // 8, R_ARROW // 4), (-R_ARROW + R_ARROW // 4, R_ARROW // 4)),
    ((-R_ARROW - R_ARROW // 4, R_ARROW // 8), (-R_ARROW - R_ARROW // 4, -R_ARROW // 8)),
    ((-R_ARROW - R_ARROW // 4, R_ARROW // 8), (-R_ARROW + R_ARROW // 8, R_ARROW // 8)),
    ((-R_ARROW - R_ARROW // 4, -R_ARROW // 8), (-R_ARROW + R_ARROW // 8, -R_ARROW // 8)),
)

# Thing indicator triangle (thintriangle_guy in am_data.h)
THING_TRIANGLE: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((-FRACUNIT // 2, int(-0.7 * FRACUNIT)), (FRACUNIT, 0)),
    ((FRACUNIT, 0), (-FRACUNIT // 2, int(0.7 * FRACUNIT))),
    ((-FRACUNIT // 2, int(0.7 * FRACUNIT)), (-FRACUNIT // 2, int(-0.7 * FRACUNIT))),
)


def _compute_outcode(x: int, y: int, w: int, h: int) -> int:
    code = CS_INSIDE
    if x < 0:
        code |= CS_LEFT
    elif x >= w:
        code |= CS_RIGHT
    if y < 0:
        code |= CS_TOP
    elif y >= h:
        code |= CS_BOTTOM
    return code


def clip_line(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> tuple[int, int, int, int] | None:
    """Cohen-Sutherland 2D line clipping against a [0..w-1] x [0..h-1] viewport."""
    code0 = _compute_outcode(x0, y0, w, h)
    code1 = _compute_outcode(x1, y1, w, h)

    while True:
        if not (code0 | code1):
            return x0, y0, x1, y1
        if code0 & code1:
            return None

        code_out = code0 if code0 else code1
        if code_out & CS_TOP:
            x = x0 + ((x1 - x0) * (0 - y0)) // (y1 - y0) if (y1 != y0) else x0
            y = 0
        elif code_out & CS_BOTTOM:
            x = x0 + ((x1 - x0) * (h - 1 - y0)) // (y1 - y0) if (y1 != y0) else x0
            y = h - 1
        elif code_out & CS_RIGHT:
            y = y0 + ((y1 - y0) * (w - 1 - x0)) // (x1 - x0) if (x1 != x0) else y0
            x = w - 1
        else:  # CS_LEFT
            y = y0 + ((y1 - y0) * (0 - x0)) // (x1 - x0) if (x1 != x0) else y0
            x = 0

        if code_out == code0:
            x0, y0 = x, y
            code0 = _compute_outcode(x0, y0, w, h)
        else:
            x1, y1 = x, y
            code1 = _compute_outcode(x1, y1, w, h)


def draw_line(
    canvas: np.ndarray,
    mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: int,
) -> None:
    """Bresenham line rasterizer writing to numpy canvas and mask buffers."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    max_h, max_w = canvas.shape
    col = color & 0xFF

    while True:
        if 0 <= x < max_w and 0 <= y < max_h:
            canvas[y, x] = col
            mask[y, x] = True

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def rotate_point(x: int, y: int, angle: int) -> tuple[int, int]:
    """Rotate a fixed-point coordinate pair by a 32-bit BAM angle."""
    cos_val = fine_cosine(angle)
    sin_val = fine_sine(angle)
    rx = fixed_mul(x, cos_val) - fixed_mul(y, sin_val)
    ry = fixed_mul(x, sin_val) + fixed_mul(y, cos_val)
    return rx, ry


class AutoMap:
    """Hexen 2D vector automap engine (T12.5).

    Supports classic parchment windowing, transparent 2D overlay, player arrow tracking,
    panning, zooming, grid display, line visibility rules, and cheat reveals.
    """

    def __init__(self, wad: WadStack | WadFile | None = None) -> None:
        self.wad: WadStack | WadFile | None = wad
        self.active: bool = False
        self.overlay_mode: bool = False  # Transparent HUD overlay vs classic parchment
        self.fullscreen_view: bool = False  # 320x200 vs 320x154
        self.follow_player: bool = True
        self.grid: bool = False
        self.cheating: int = 0  # 0: normal, 1: all lines, 2: all lines + actors

        # Window & viewport dimensions
        self.f_w: int = FINIT_WIDTH
        self.f_h: int = FINIT_HEIGHT
        self.map_x_start: int = 0
        self.map_y_start: int = 0

        # World bounds & scales in fixed-point
        self.min_x: int = 0
        self.max_x: int = 0
        self.min_y: int = 0
        self.max_y: int = 0
        self.scale_mtof: int = FRACUNIT // 4
        self.scale_ftom: int = 4 * FRACUNIT
        self.min_scale_mtof: int = FRACUNIT // 8
        self.max_scale_mtof: int = 4 * FRACUNIT

        # Map window position in fixed-point
        self.m_x: int = 0
        self.m_y: int = 0
        self.m_w: int = 0
        self.m_h: int = 0

        # Saved scale & loc for max zoom toggle
        self._saved_scale: int = self.scale_mtof
        self._saved_m_x: int = 0
        self._saved_m_y: int = 0
        self._is_max_zoom: bool = False

        # Raw parchment lump
        self._autopage: bytes | None = None
        self._load_autopage()

    def _load_autopage(self) -> None:
        """Load AUTOPAGE raw lump from WAD if available."""
        if self.wad is not None:
            lump = self.wad.find("AUTOPAGE")
            if lump is not None:
                try:
                    self._autopage = self.wad.get_bytes("AUTOPAGE")
                except Exception:
                    self._autopage = None

    def start(self, level: Level, player: Player) -> None:
        """Open the automap (AM_Start)."""
        self.active = True
        self.f_w = FINIT_WIDTH
        self.f_h = VIRTUAL_HEIGHT if self.fullscreen_view else FINIT_HEIGHT
        self._compute_level_bounds(level)

        # Initial scale
        self.scale_mtof = max(self.min_scale_mtof, int(self.min_scale_mtof / 0.7))
        if self.scale_mtof > self.max_scale_mtof:
            self.scale_mtof = self.min_scale_mtof
        self.scale_ftom = fixed_div(FRACUNIT, self.scale_mtof)

        # Center on player
        self.m_w = fixed_mul(self.f_w << FRACBITS, self.scale_ftom)
        self.m_h = fixed_mul(self.f_h << FRACBITS, self.scale_ftom)
        px = player.mo.x if player.mo is not None else 0
        py = player.mo.y if player.mo is not None else 0
        self.m_x = px - self.m_w // 2
        self.m_y = py - self.m_h // 2
        self.follow_player = True
        self._is_max_zoom = False

    def stop(self) -> None:
        """Close the automap (AM_Stop)."""
        self.active = False

    def toggle(self, level: Level, player: Player) -> None:
        """Toggle automap active state."""
        if self.active:
            self.stop()
        else:
            self.start(level, player)

    def _compute_level_bounds(self, level: Level) -> None:
        """Calculate bounding box and scale limits from level vertices."""
        if not level.vertexes:
            self.min_x = self.min_y = -1000 * FRACUNIT
            self.max_x = self.max_y = 1000 * FRACUNIT
        else:
            self.min_x = min(v.x for v in level.vertexes)
            self.max_x = max(v.x for v in level.vertexes)
            self.min_y = min(v.y for v in level.vertexes)
            self.max_y = max(v.y for v in level.vertexes)

        max_w = max(1, self.max_x - self.min_x)
        max_h = max(1, self.max_y - self.min_y)

        a = fixed_div(self.f_w << FRACBITS, max_w)
        b = fixed_div(self.f_h << FRACBITS, max_h)
        self.min_scale_mtof = min(a, b)
        self.max_scale_mtof = fixed_div(self.f_h << FRACBITS, 2 * PLAYER_RADIUS)
        if self.max_scale_mtof <= self.min_scale_mtof:
            self.max_scale_mtof = self.min_scale_mtof * 10

    def tick(self, player: Player) -> None:
        """Update window position and tracking per simulation tic (AM_Ticker)."""
        if not self.active:
            return

        if self.follow_player and player.mo is not None:
            self.m_x = player.mo.x - self.m_w // 2
            self.m_y = player.mo.y - self.m_h // 2

    # -------------------------------------------------------------------------
    # Input Handling (AM_Responder)
    # -------------------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input for automap navigation."""
        if not self.active:
            return False

        if key in (pygame.K_TAB, pygame.K_ESCAPE):
            self.stop()
            return True
        elif key == pygame.K_f:
            self.toggle_follow()
            return True
        elif key == pygame.K_g:
            self.toggle_grid()
            return True
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.zoom(1)
            return True
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.zoom(-1)
            return True
        elif key == pygame.K_0:
            self.toggle_max_zoom()
            return True
        elif key == pygame.K_o:
            self.toggle_overlay()
            return True
        elif key == pygame.K_LEFT:
            self.pan(-F_PANINC, 0)
            return True
        elif key == pygame.K_RIGHT:
            self.pan(F_PANINC, 0)
            return True
        elif key == pygame.K_UP:
            self.pan(0, F_PANINC)
            return True
        elif key == pygame.K_DOWN:
            self.pan(0, -F_PANINC)
            return True

        return False

    def handle_mouse_drag(self, dx: int, dy: int) -> None:
        """Pan automap by screen delta in pixels."""
        if not self.active:
            return
        # Invert dy because Y points up in map coordinates
        self.pan(-dx, dy)

    def handle_mouse_wheel(self, direction: int) -> None:
        """Zoom automap via mouse wheel (direction > 0 zooms in, < 0 zooms out)."""
        if not self.active:
            return
        self.zoom(direction)

    def pan(self, screen_dx: int, screen_dy: int) -> None:
        """Pan automap by screen pixel offsets."""
        self.follow_player = False
        map_dx = fixed_mul(screen_dx << FRACBITS, self.scale_ftom)
        map_dy = fixed_mul(screen_dy << FRACBITS, self.scale_ftom)
        self.m_x += map_dx
        self.m_y += map_dy

        # Clamp within map bounds
        half_w = self.m_w // 2
        half_h = self.m_h // 2
        self.m_x = max(self.min_x - half_w, min(self.max_x - half_w, self.m_x))
        self.m_y = max(self.min_y - half_h, min(self.max_y - half_h, self.m_y))

    def zoom(self, step: int) -> None:
        """Zoom in (step > 0) or out (step < 0)."""
        center_x = self.m_x + self.m_w // 2
        center_y = self.m_y + self.m_h // 2

        if step > 0:
            self.scale_mtof = fixed_mul(self.scale_mtof, ZOOM_IN_MUL)
        else:
            self.scale_mtof = fixed_mul(self.scale_mtof, ZOOM_OUT_MUL)

        self.scale_mtof = max(self.min_scale_mtof, min(self.max_scale_mtof, self.scale_mtof))
        self.scale_ftom = fixed_div(FRACUNIT, self.scale_mtof)

        self.m_w = fixed_mul(self.f_w << FRACBITS, self.scale_ftom)
        self.m_h = fixed_mul(self.f_h << FRACBITS, self.scale_ftom)
        self.m_x = center_x - self.m_w // 2
        self.m_y = center_y - self.m_h // 2
        self._is_max_zoom = False

    def toggle_max_zoom(self) -> None:
        """Toggle between maximum whole-map fit and previous zoom."""
        if not self._is_max_zoom:
            self._saved_scale = self.scale_mtof
            self._saved_m_x = self.m_x
            self._saved_m_y = self.m_y

            self.scale_mtof = self.min_scale_mtof
            self.scale_ftom = fixed_div(FRACUNIT, self.scale_mtof)
            self.m_w = fixed_mul(self.f_w << FRACBITS, self.scale_ftom)
            self.m_h = fixed_mul(self.f_h << FRACBITS, self.scale_ftom)
            self.m_x = (self.min_x + self.max_x) // 2 - self.m_w // 2
            self.m_y = (self.min_y + self.max_y) // 2 - self.m_h // 2
            self._is_max_zoom = True
        else:
            self.scale_mtof = self._saved_scale
            self.scale_ftom = fixed_div(FRACUNIT, self.scale_mtof)
            self.m_w = fixed_mul(self.f_w << FRACBITS, self.scale_ftom)
            self.m_h = fixed_mul(self.f_h << FRACBITS, self.scale_ftom)
            self.m_x = self._saved_m_x
            self.m_y = self._saved_m_y
            self._is_max_zoom = False

    def toggle_follow(self) -> None:
        """Toggle follow player mode."""
        self.follow_player = not self.follow_player

    def toggle_grid(self) -> None:
        """Toggle coordinate blockmap gridlines."""
        self.grid = not self.grid

    def toggle_overlay(self) -> None:
        """Toggle between parchment background and transparent 2D overlay."""
        self.overlay_mode = not self.overlay_mode

    # -------------------------------------------------------------------------
    # Coordinate Transforms
    # -------------------------------------------------------------------------

    def map_to_screen(self, wx: int, wy: int) -> tuple[int, int]:
        """Convert fixed-point map coordinates to frame-buffer pixel coordinates."""
        dx = wx - self.m_x
        dy = wy - self.m_y
        sx = (fixed_mul(dx, self.scale_mtof)) >> FRACBITS
        sy = self.f_h - ((fixed_mul(dy, self.scale_mtof)) >> FRACBITS)
        return sx, sy

    # -------------------------------------------------------------------------
    # Drawing (AM_Drawer)
    # -------------------------------------------------------------------------

    def draw(
        self,
        overlay: Overlay2D,
        level: Level,
        player: Player,
        actors: list[Mobj] | None = None,
    ) -> None:
        """Render the automap onto the 320x200 overlay canvas."""
        if not self.active:
            return

        # 1. Background
        if not self.overlay_mode:
            self._draw_background(overlay)

        # 2. Grid lines
        if self.grid:
            self._draw_grid(overlay)

        # 3. Geometry walls
        self._draw_walls(overlay, level, player)

        # 4. Player indicator
        self._draw_player(overlay, player)

        # 5. Actors / monsters (cheating == 2)
        if self.cheating == 2 and actors is not None:
            self._draw_things(overlay, actors)

        # 6. Map Name & World Timer
        self._draw_overlay_text(overlay, level, player)

    def _draw_background(self, overlay: Overlay2D) -> None:
        """Draw parchment background (AM_clearFB)."""
        if self._autopage is not None and len(self._autopage) >= self.f_w * 154:
            arr = np.frombuffer(self._autopage, dtype=np.uint8, count=self.f_w * self.f_h).reshape(
                (self.f_h, self.f_w)
            )
            overlay.indices[: self.f_h, : self.f_w] = arr
            overlay.mask[: self.f_h, : self.f_w] = True
        else:
            overlay.fill_rect(0, 0, self.f_w, self.f_h, color_index=PARCH)

    def _draw_grid(self, overlay: Overlay2D) -> None:
        """Draw blockmap alignment gridlines (AM_drawGrid)."""
        step = 128 << FRACBITS
        start_x = self.m_x - (self.m_x % step)
        end_x = self.m_x + self.m_w
        start_y = self.m_y - (self.m_y % step)
        end_y = self.m_y + self.m_h

        # Vertical lines
        x = start_x
        while x <= end_x:
            sx0, sy0 = self.map_to_screen(x, self.m_y)
            sx1, sy1 = self.map_to_screen(x, self.m_y + self.m_h)
            clipped = clip_line(sx0, sy0, sx1, sy1, self.f_w, self.f_h)
            if clipped:
                draw_line(
                    overlay.indices,
                    overlay.mask,
                    clipped[0],
                    clipped[1],
                    clipped[2],
                    clipped[3],
                    GRIDCOLORS,
                )
            x += step

        # Horizontal lines
        y = start_y
        while y <= end_y:
            sx0, sy0 = self.map_to_screen(self.m_x, y)
            sx1, sy1 = self.map_to_screen(self.m_x + self.m_w, y)
            clipped = clip_line(sx0, sy0, sx1, sy1, self.f_w, self.f_h)
            if clipped:
                draw_line(
                    overlay.indices,
                    overlay.mask,
                    clipped[0],
                    clipped[1],
                    clipped[2],
                    clipped[3],
                    GRIDCOLORS,
                )
            y += step

    def _draw_walls(self, overlay: Overlay2D, level: Level, player: Player) -> None:
        """Render linedefs according to visibility and special type rules (AM_drawWalls)."""
        has_allmap = bool(player.powers[int(PowerType.PW_ALLMAP)])

        for line in level.lines:
            color = self._classify_line_color(line, has_allmap)
            if color is None:
                continue

            sx0, sy0 = self.map_to_screen(line.v1.x, line.v1.y)
            sx1, sy1 = self.map_to_screen(line.v2.x, line.v2.y)
            clipped = clip_line(sx0, sy0, sx1, sy1, self.f_w, self.f_h)
            if clipped:
                draw_line(
                    overlay.indices,
                    overlay.mask,
                    clipped[0],
                    clipped[1],
                    clipped[2],
                    clipped[3],
                    color,
                )

    def _classify_line_color(self, line: Line, has_allmap: bool) -> int | None:
        """Determine color of linedef per Hexen rules, or None if hidden."""
        if (line.flags & ML_DONTDRAW) and not self.cheating:
            return None

        is_mapped = bool(line.flags & ML_MAPPED) or bool(self.cheating)

        if not is_mapped:
            if has_allmap:
                return ALLMAP_LINE_COLOR
            return None

        # 1-sided wall
        if line.back_sector is None:
            return WALLCOLORS

        # 2-sided line
        if line.flags & ML_SECRET:
            return SECRET_WALL_COLOR if self.cheating else WALLCOLORS

        # Locked doors (13 or 83)
        if line.special in (13, 83):
            return GREENKEY

        # Teleports
        if line.special in (70, 71):
            return BLUEKEY
        if line.special in (74, 75):
            return BLOODRED

        # Height changes
        front = line.front_sector
        back = line.back_sector
        if front is not None and back is not None:
            front_fh = getattr(front, "floor_height", getattr(front, "floorheight", 0))
            back_fh = getattr(back, "floor_height", getattr(back, "floorheight", 0))
            if front_fh != back_fh:
                return FDWALLCOLORS

            front_ch = getattr(front, "ceiling_height", getattr(front, "ceilingheight", 0))
            back_ch = getattr(back, "ceiling_height", getattr(back, "ceilingheight", 0))
            if front_ch != back_ch:
                return CDWALLCOLORS

        # 2-sided line with no height change
        if self.cheating:
            return TSWALLCOLORS

        return None

    def _draw_player(self, overlay: Overlay2D, player: Player) -> None:
        """Draw player directional arrow (AM_drawPlayers)."""
        if player.mo is None:
            return

        px = player.mo.x
        py = player.mo.y
        angle = player.mo.angle

        for (p0x, p0y), (p1x, p1y) in PLAYER_ARROW:
            rx0, ry0 = rotate_point(p0x, p0y, angle)
            rx1, ry1 = rotate_point(p1x, p1y, angle)

            sx0, sy0 = self.map_to_screen(px + rx0, py + ry0)
            sx1, sy1 = self.map_to_screen(px + rx1, py + ry1)

            clipped = clip_line(sx0, sy0, sx1, sy1, self.f_w, self.f_h)
            if clipped:
                draw_line(
                    overlay.indices,
                    overlay.mask,
                    clipped[0],
                    clipped[1],
                    clipped[2],
                    clipped[3],
                    WHITE,
                )

    def _draw_things(self, overlay: Overlay2D, actors: list[Mobj]) -> None:
        """Draw enemy and monster indicators when cheat is 2 (AM_drawThings)."""
        scale = 16

        for actor in actors:
            if actor.health <= 0:
                continue

            ax = actor.x
            ay = actor.y
            angle = actor.angle

            for (p0x, p0y), (p1x, p1y) in THING_TRIANGLE:
                rx0, ry0 = rotate_point(p0x * scale, p0y * scale, angle)
                rx1, ry1 = rotate_point(p1x * scale, p1y * scale, angle)

                sx0, sy0 = self.map_to_screen(ax + rx0, ay + ry0)
                sx1, sy1 = self.map_to_screen(ax + rx1, ay + ry1)

                clipped = clip_line(sx0, sy0, sx1, sy1, self.f_w, self.f_h)
                if clipped:
                    draw_line(
                        overlay.indices,
                        overlay.mask,
                        clipped[0],
                        clipped[1],
                        clipped[2],
                        clipped[3],
                        THINGCOLORS,
                    )

    def _draw_overlay_text(self, overlay: Overlay2D, level: Level, player: Player) -> None:
        """Draw map name and world timer."""
        # Map name at bottom-left
        map_title = level.name
        overlay.draw_text_a(map_title, 38, self.f_h - 10, yellow=True)

        # World timer at top-right
        world_timer = getattr(player, "world_timer", 0)
        total_sec = world_timer // 35
        hours = total_sec // 3600
        mins = (total_sec % 3600) // 60
        secs = total_sec % 60
        timer_str = f"{hours:02d} : {mins:02d} : {secs:02d}"
        overlay.draw_text_a(timer_str, 240, 8)
