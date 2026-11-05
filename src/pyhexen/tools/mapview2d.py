# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""2D interactive and headless map viewer (Milestone M1, T3.4)."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import pygame

from pyhexen.app.config import resolve_iwad
from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.wad.mapinfo import MapInfo, load_mapinfo
from pyhexen.wad.wadfile import WadStack, normalize_lump_name
from pyhexen.world.level import GET_SPAC, ML_TWOSIDED, Level, Line, MapThing, Sector, load_level

# Background & UI colors
COLOR_BG = (20, 22, 28)
COLOR_GRID = (34, 38, 48)
COLOR_ORIGIN = (55, 65, 85)
COLOR_TOOLTIP_BG = (16, 18, 24)
COLOR_TOOLTIP_BORDER = (110, 130, 165)
COLOR_TEXT = (230, 235, 245)
COLOR_HIGHLIGHT = (255, 255, 100)

# Line colors by type
COLOR_LINE_ONESIDED = (235, 75, 75)
COLOR_LINE_TWOSIDED_DIFF = (210, 155, 70)
COLOR_LINE_TWOSIDED_SAME = (115, 120, 130)
COLOR_LINE_SPECIAL = (235, 85, 225)
COLOR_LINE_POLYOBJ = (75, 245, 95)

# Thing category colors
COLOR_THING_PLAYER = (75, 215, 255)
COLOR_THING_POLYOBJ = (75, 245, 95)
COLOR_THING_SOUND = (180, 120, 255)
COLOR_THING_MONSTER = (255, 85, 85)
COLOR_THING_ITEM = (255, 215, 75)
COLOR_THING_DECORATION = (150, 195, 150)

POLYOBJ_LINE_SPECIALS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
PLAYER_THING_TYPES = frozenset({1, 2, 3, 4, 11, 9100, 9101, 9102, 9103})
POLYOBJ_THING_TYPES = frozenset({3000, 3001, 3002})

# Known Hexen monster doomednums for category coloring
MONSTER_THING_TYPES = frozenset(
    {
        31,  # Demon1 (Chaos Serpent)
        8080,  # Demon2 (Brown Chaos Serpent)
        10030,  # Ettin
        107,  # Centaur
        115,  # CentaurLeader (Slaughtaur)
        114,  # Bishop
        121,  # FireDemon (Afrit)
        34,  # Wraith (Reiver)
        10011,  # WraithBuried
        8020,  # IceGuy (Wendigo)
        254,  # Dragon (Death Wyvern)
        10080,  # Sorcerer (Heresiarch)
        10100,  # FighterBoss (Zedek)
        10101,  # ClericBoss (Traductus)
        10102,  # MageBoss (Menelkir)
        10200,  # Korax
        8065,  # Stalker
        8066,  # StalkerBoss
    }
)

ITEM_THING_TYPES = frozenset(
    {
        10,
        12,
        13,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        30,
        32,
        33,
        36,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        122,
        123,
        124,
        8000,
        8002,
        8003,
        8004,
        8029,
        8030,
        8031,
        8032,
        8033,
        8034,
        8035,
        8036,
        8037,
        8038,
        8039,
        8040,
        8041,
        9002,
        9003,
        9004,
        9005,
        9006,
        9007,
        9008,
        9009,
        9010,
        9011,
        9012,
        9014,
        9015,
        9016,
        9017,
    }
)


def classify_line_color(line: Line) -> tuple[int, int, int]:
    """Return the RGB display color for a `Line` based on its type and flags."""
    if line.special in POLYOBJ_LINE_SPECIALS:
        return COLOR_LINE_POLYOBJ
    if line.special != 0:
        return COLOR_LINE_SPECIAL
    if line.back_sector is None or not (line.flags & ML_TWOSIDED):
        return COLOR_LINE_ONESIDED
    if line.front_sector is not None and (
        line.front_sector.floor_height != line.back_sector.floor_height
        or line.front_sector.ceiling_height != line.back_sector.ceiling_height
    ):
        return COLOR_LINE_TWOSIDED_DIFF
    return COLOR_LINE_TWOSIDED_SAME


def classify_thing_category(thing: MapThing) -> tuple[str, tuple[int, int, int]]:
    """Return `(category_name, rgb_color)` for a `MapThing`."""
    t = thing.type
    if t in PLAYER_THING_TYPES:
        return ("player", COLOR_THING_PLAYER)
    if t in POLYOBJ_THING_TYPES:
        return ("polyobj", COLOR_THING_POLYOBJ)
    if 1400 <= t <= 1410:
        return ("sound", COLOR_THING_SOUND)
    if t in MONSTER_THING_TYPES:
        return ("monster", COLOR_THING_MONSTER)
    if t in ITEM_THING_TYPES:
        return ("item", COLOR_THING_ITEM)
    return ("decoration", COLOR_THING_DECORATION)


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


class MapView2D:
    """2D map renderer and interactive inspector for a Hexen `Level`."""

    def __init__(
        self,
        level: Level,
        *,
        width: int = 1280,
        height: int = 800,
        map_info: MapInfo | None = None,
    ) -> None:
        self.level = level
        self.width = width
        self.height = height
        self.map_info = map_info
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.scale: float = 1.0
        self.bounds: tuple[float, float, float, float] = self._compute_bounds()
        self.reset_view()

    def _compute_bounds(self) -> tuple[float, float, float, float]:
        if not self.level.vertexes:
            return (-128.0, -128.0, 128.0, 128.0)
        xs = [v.x / FRACUNIT for v in self.level.vertexes]
        ys = [v.y / FRACUNIT for v in self.level.vertexes]
        return (min(xs), min(ys), max(xs), max(ys))

    def reset_view(self) -> None:
        """Center and fit the entire level within the viewport."""
        min_x, min_y, max_x, max_y = self.bounds
        self.offset_x = 0.5 * (min_x + max_x)
        self.offset_y = 0.5 * (min_y + max_y)
        span_x = max(64.0, (max_x - min_x) * 1.15)
        span_y = max(64.0, (max_y - min_y) * 1.15)
        self.scale = min(self.width / span_x, self.height / span_y)

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world map coordinates `(wx, wy)` to screen pixel coordinates `(sx, sy)`."""
        sx = int(round(self.width * 0.5 + (wx - self.offset_x) * self.scale))
        sy = int(round(self.height * 0.5 - (wy - self.offset_y) * self.scale))
        return (sx, sy)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert screen pixel coordinates `(sx, sy)` to world map coordinates `(wx, wy)`."""
        wx = self.offset_x + (sx - self.width * 0.5) / self.scale
        wy = self.offset_y - (sy - self.height * 0.5) / self.scale
        return (wx, wy)

    def pan_pixels(self, dx_px: float, dy_px: float) -> None:
        """Pan the view by screen pixel deltas `(dx_px, dy_px)`."""
        self.offset_x -= dx_px / self.scale
        self.offset_y += dy_px / self.scale

    def zoom_at(self, factor: float, screen_pos: tuple[float, float] | None = None) -> None:
        """Zoom by `factor`, keeping `screen_pos` (or screen center) anchored in world space."""
        if screen_pos is not None:
            before_wx, before_wy = self.screen_to_world(screen_pos[0], screen_pos[1])
        else:
            before_wx, before_wy = self.offset_x, self.offset_y

        self.scale = max(0.05, min(64.0, self.scale * factor))

        if screen_pos is not None:
            after_wx, after_wy = self.screen_to_world(screen_pos[0], screen_pos[1])
            self.offset_x += before_wx - after_wx
            self.offset_y += before_wy - after_wy

    def pick_element(
        self,
        world_x: float,
        world_y: float,
        *,
        tolerance: float | None = None,
    ) -> MapThing | Line | Sector | None:
        """Pick the closest `MapThing`, `Line`, or containing `Sector` at `(world_x, world_y)`."""
        tol = tolerance if tolerance is not None else max(6.0, 10.0 / self.scale)

        # 1. Closest thing within tolerance
        best_thing: MapThing | None = None
        best_thing_dist = tol
        for thing in self.level.things:
            dist = math.hypot(world_x - thing.x, world_y - thing.y)
            if dist <= best_thing_dist:
                best_thing_dist = dist
                best_thing = thing
        if best_thing is not None:
            return best_thing

        # 2. Closest line within tolerance
        best_line: Line | None = None
        best_line_dist = tol
        for line in self.level.lines:
            x1 = line.v1.x / FRACUNIT
            y1 = line.v1.y / FRACUNIT
            x2 = line.v2.x / FRACUNIT
            y2 = line.v2.y / FRACUNIT
            dist = _point_to_segment_distance(world_x, world_y, x1, y1, x2, y2)
            if dist <= best_line_dist:
                best_line_dist = dist
                best_line = line
        if best_line is not None:
            return best_line

        # 3. Containing sector via BSP if inside map bounds
        min_x, min_y, max_x, max_y = self.bounds
        if min_x <= world_x <= max_x and min_y <= world_y <= max_y and self.level.subsectors:
            sub = self.level.point_in_subsector(
                int(round(world_x * FRACUNIT)),
                int(round(world_y * FRACUNIT)),
            )
            return sub.sector

        return None

    def format_tooltip(self, element: MapThing | Line | Sector) -> list[str]:
        """Format tooltip text lines for a picked `MapThing`, `Line`, or `Sector`."""
        if isinstance(element, MapThing):
            cat, _ = classify_thing_category(element)
            return [
                f"Thing (type={element.type}, category={cat})",
                f"Pos: ({element.x}, {element.y}, z={element.z}) Angle: {element.angle}",
                f"TID: {element.tid} Flags: 0x{element.flags:04X}",
                f"Special: {element.special} Args: {element.args}",
            ]
        if isinstance(element, Line):
            front_sec = element.front_sector.index if element.front_sector else None
            back_sec = element.back_sector.index if element.back_sector else None
            spac = GET_SPAC(element.flags)
            return [
                f"Line #{element.index} (sectors: {front_sec} / {back_sec})",
                f"Flags: 0x{element.flags:04X} (SPAC={spac})",
                f"Special: {element.special} Args: {element.args}",
            ]
        if isinstance(element, Sector):
            fh = element.floor_height >> FRACBITS
            ch = element.ceiling_height >> FRACBITS
            return [
                f"Sector #{element.index} (light={element.light_level})",
                f"Heights: floor={fh} ({element.floor_pic}), ceil={ch} ({element.ceiling_pic})",
                f"Special: {element.special} Tag: {element.tag}",
            ]
        return []

    def render_to_surface(
        self,
        surface: pygame.Surface,
        *,
        hover_world: tuple[float, float] | None = None,
    ) -> MapThing | Line | Sector | None:
        """Render the 2D map, grid, things, and optional hover tooltip onto `surface`."""
        self.width, self.height = surface.get_size()
        surface.fill(COLOR_BG)

        # Draw 128-unit grid
        grid_step = 128.0
        if self.scale * grid_step >= 12.0:
            left_wx, top_wy = self.screen_to_world(0, 0)
            right_wx, bottom_wy = self.screen_to_world(self.width, self.height)
            gx_start = math.floor(left_wx / grid_step) * grid_step
            gx = gx_start
            while gx <= right_wx:
                sx, _ = self.world_to_screen(gx, 0.0)
                col = COLOR_ORIGIN if abs(gx) < 1e-6 else COLOR_GRID
                pygame.draw.line(surface, col, (sx, 0), (sx, self.height), 1)
                gx += grid_step

            gy_start = math.floor(bottom_wy / grid_step) * grid_step
            gy = gy_start
            while gy <= top_wy:
                _, sy = self.world_to_screen(0.0, gy)
                col = COLOR_ORIGIN if abs(gy) < 1e-6 else COLOR_GRID
                pygame.draw.line(surface, col, (0, sy), (self.width, sy), 1)
                gy += grid_step

        picked = (
            self.pick_element(hover_world[0], hover_world[1]) if hover_world is not None else None
        )

        # Draw lines
        for line in self.level.lines:
            p1 = self.world_to_screen(line.v1.x / FRACUNIT, line.v1.y / FRACUNIT)
            p2 = self.world_to_screen(line.v2.x / FRACUNIT, line.v2.y / FRACUNIT)
            if picked is line:
                color = COLOR_HIGHLIGHT
                width = 3
            elif isinstance(picked, Sector) and (
                line.front_sector is picked or line.back_sector is picked
            ):
                color = COLOR_HIGHLIGHT
                width = 2
            else:
                color = classify_line_color(line)
                width = 2 if line.back_sector is None else 1
            pygame.draw.line(surface, color, p1, p2, width)

        # Draw things
        radius = max(3, min(8, int(round(8.0 * self.scale))))
        for thing in self.level.things:
            sx, sy = self.world_to_screen(float(thing.x), float(thing.y))
            if sx < -16 or sx > self.width + 16 or sy < -16 or sy > self.height + 16:
                continue
            _, color = classify_thing_category(thing)
            if picked is thing:
                pygame.draw.circle(surface, COLOR_HIGHLIGHT, (sx, sy), radius + 3, 2)
            pygame.draw.circle(surface, color, (sx, sy), radius, 0)
            rad_angle = math.radians(thing.angle)
            tx = int(round(sx + math.cos(rad_angle) * (radius + 4)))
            ty = int(round(sy - math.sin(rad_angle) * (radius + 4)))
            pygame.draw.line(surface, color, (sx, sy), (tx, ty), 1)

        # Draw HUD / tooltip if font system is available
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.SysFont("consolas", 14) or pygame.font.Font(None, 16)
            title = self.level.name
            if self.map_info is not None:
                title = f"{self.level.name}: {self.map_info.name} (Cluster {self.map_info.cluster})"
            header_surf = font.render(
                f"{title} | Lines: {len(self.level.lines)} | Sectors: {len(self.level.sectors)} | "
                f"Things: {len(self.level.things)}",
                True,
                COLOR_TEXT,
            )
            surface.blit(header_surf, (10, 8))

            if picked is not None:
                lines_txt = self.format_tooltip(picked)
                rendered = [font.render(line_str, True, COLOR_TEXT) for line_str in lines_txt]
                box_w = max(r.get_width() for r in rendered) + 16
                box_h = sum(r.get_height() + 2 for r in rendered) + 12
                box_rect = pygame.Rect(10, self.height - box_h - 10, box_w, box_h)
                pygame.draw.rect(surface, COLOR_TOOLTIP_BG, box_rect)
                pygame.draw.rect(surface, COLOR_TOOLTIP_BORDER, box_rect, 1)
                cur_y = box_rect.top + 6
                for r in rendered:
                    surface.blit(r, (box_rect.left + 8, cur_y))
                    cur_y += r.get_height() + 2
        except Exception:
            pass

        return picked


def _normalize_warp_name(warp: str) -> str:
    stripped = warp.strip()
    if stripped.isdigit():
        return f"MAP{int(stripped):02d}"
    return normalize_lump_name(stripped)


def run_interactive(viewer: MapView2D) -> int:
    """Run the interactive 2D map viewer window loop."""
    pygame.init()
    try:
        screen = pygame.display.set_mode((viewer.width, viewer.height), pygame.RESIZABLE)
        pygame.display.set_caption(f"PyHexen 2D Map Viewer - {viewer.level.name}")
        clock = pygame.time.Clock()
        dragging = False
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    viewer.width, viewer.height = event.w, event.h
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        viewer.reset_view()
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        viewer.zoom_at(1.25)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        viewer.zoom_at(0.8)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button in (1, 2, 3):
                        dragging = True
                    elif event.button == 4:
                        viewer.zoom_at(1.2, event.pos)
                    elif event.button == 5:
                        viewer.zoom_at(1.0 / 1.2, event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (1, 2, 3):
                        dragging = False
                elif event.type == pygame.MOUSEMOTION and dragging:
                    viewer.pan_pixels(event.rel[0], event.rel[1])
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_pos = pygame.mouse.get_pos()
                    factor = 1.2 if event.y > 0 else (1.0 / 1.2)
                    viewer.zoom_at(factor, mouse_pos)

            keys = pygame.key.get_pressed()
            pan_speed = 10.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                viewer.pan_pixels(pan_speed, 0.0)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                viewer.pan_pixels(-pan_speed, 0.0)
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                viewer.pan_pixels(0.0, pan_speed)
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                viewer.pan_pixels(0.0, -pan_speed)

            mx, my = pygame.mouse.get_pos()
            hover_world = viewer.screen_to_world(mx, my)
            viewer.render_to_surface(screen, hover_world=hover_world)
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyhexen.tools.mapview2d",
        description="2D map viewer for Hexen maps.",
    )
    parser.add_argument(
        "--iwad",
        type=str,
        default=None,
        help="Path to HEXEN.WAD (defaults to PYHEXEN_IWAD or config.toml)",
    )
    parser.add_argument(
        "--file",
        "-f",
        nargs="*",
        default=[],
        help="Optional PWAD file(s)",
    )
    parser.add_argument(
        "--warp",
        type=str,
        default="MAP01",
        help="Map name or number to view (e.g. MAP01 or 1)",
    )
    parser.add_argument(
        "--headless-out",
        type=str,
        default=None,
        help="Render the 2D view to a PNG file and exit without opening a window",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    iwad_path = resolve_iwad(args.iwad, must_exist=True)
    if iwad_path is None:
        print(
            "Error: No HEXEN.WAD found. Specify --iwad, PYHEXEN_IWAD, or config.toml.",
            file=sys.stderr,
        )
        return 1

    map_name = _normalize_warp_name(args.warp)

    with WadStack(iwad_path) as wad:
        for pwad in args.file:
            wad.add_wad(pwad)

        level = load_level(wad, map_name)
        map_info = None
        if "MAPINFO" in wad:
            info_table = load_mapinfo(wad)
            map_info = info_table.get(map_name)

        viewer = MapView2D(level, map_info=map_info)

        if args.headless_out:
            surface = pygame.Surface((viewer.width, viewer.height))
            viewer.render_to_surface(surface)
            out_path = Path(args.headless_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(surface, str(out_path))
            print(f"Saved 2D map view of {map_name} -> {out_path}")
            return 0

        return run_interactive(viewer)


if __name__ == "__main__":
    raise SystemExit(main())
