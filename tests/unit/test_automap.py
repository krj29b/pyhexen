# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T12.5: Automap system (am_map.c)."""

from __future__ import annotations

import numpy as np
import pygame
import pytest

from pyhexen.app.config import resolve_iwad
from pyhexen.core.angles import ANG90, ANG180
from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.sim.artifacts import PowerType
from pyhexen.sim.mobj import Mobj
from pyhexen.sim.player import Player
from pyhexen.ui.automap import (
    ALLMAP_LINE_COLOR,
    BLOODRED,
    BLUEKEY,
    CDWALLCOLORS,
    FDWALLCOLORS,
    GREENKEY,
    PARCH,
    SECRET_WALL_COLOR,
    THINGCOLORS,
    TSWALLCOLORS,
    WALLCOLORS,
    WHITE,
    AutoMap,
    clip_line,
    draw_line,
    rotate_point,
)
from pyhexen.ui.fonts import FontSet
from pyhexen.wad.wadfile import WadFile
from pyhexen.world.level import (
    ML_DONTDRAW,
    ML_MAPPED,
    ML_SECRET,
    Line,
    Sector,
    Side,
    SlopeType,
    Vertex,
)


def _make_dummy_vertex(x: int, y: int) -> Vertex:
    return Vertex(x=x << FRACBITS, y=y << FRACBITS)


def _make_dummy_sector(floorheight: int = 0, ceilingheight: int = 128) -> Sector:
    return Sector(
        index=0,
        floor_height=floorheight << FRACBITS,
        ceiling_height=ceilingheight << FRACBITS,
        floor_pic="F_001",
        ceiling_pic="C_001",
        light_level=160,
        special=0,
        tag=0,
        lines=[],
        sound_target=None,
        valid_count=0,
    )


def _make_dummy_line(
    v1: Vertex,
    v2: Vertex,
    flags: int = ML_MAPPED,
    special: int = 0,
    front_sector: Sector | None = None,
    back_sector: Sector | None = None,
) -> Line:
    front_side = (
        Side(
            index=0,
            texture_offset=0,
            row_offset=0,
            top_texture="",
            bottom_texture="",
            mid_texture="",
            sector=front_sector,
        )
        if front_sector
        else None
    )
    back_side = (
        Side(
            index=1,
            texture_offset=0,
            row_offset=0,
            top_texture="",
            bottom_texture="",
            mid_texture="",
            sector=back_sector,
        )
        if back_sector
        else None
    )

    return Line(
        index=0,
        v1=v1,
        v2=v2,
        dx=v2.x - v1.x,
        dy=v2.y - v1.y,
        flags=flags,
        special=special,
        args=(0, 0, 0, 0, 0),
        sidenum=(0, 1 if back_sector else -1),
        front_side=front_side,
        back_side=back_side,
        front_sector=front_sector,
        back_sector=back_sector,
        slopetype=SlopeType.ST_HORIZONTAL,
        bbox=(min(v1.x, v2.x), min(v1.y, v2.y), max(v1.x, v2.x), max(v1.y, v2.y)),
    )


class DummyLevel:
    """Minimal level container for automap testing."""

    def __init__(self, vertexes: list[Vertex], lines: list[Line], name: str = "MAP01") -> None:
        self.vertexes = vertexes
        self.lines = lines
        self.name = name


def test_cohen_sutherland_clipping() -> None:
    w, h = 320, 154

    # 1. Fully inside
    res = clip_line(10, 10, 50, 50, w, h)
    assert res == (10, 10, 50, 50)

    # 2. Fully outside (right)
    assert clip_line(350, 10, 400, 50, w, h) is None

    # 3. Fully outside (top)
    assert clip_line(10, -50, 50, -10, w, h) is None

    # 4. Clipping crossing left boundary (-20, 20) -> (40, 20)
    clipped = clip_line(-20, 20, 40, 20, w, h)
    assert clipped is not None
    assert clipped[0] == 0
    assert clipped[1] == 20
    assert clipped[2] == 40
    assert clipped[3] == 20

    # 5. Clipping crossing bottom boundary (10, 100) -> (10, 200)
    clipped2 = clip_line(10, 100, 10, 200, w, h)
    assert clipped2 is not None
    assert clipped2[0] == 10
    assert clipped2[1] == 100
    assert clipped2[2] == 10
    assert clipped2[3] == h - 1


def test_bresenham_line_drawing() -> None:
    canvas = np.zeros((100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)

    # Horizontal line
    draw_line(canvas, mask, 10, 20, 30, 20, 96)
    assert np.all(canvas[20, 10:31] == 96)
    assert np.all(mask[20, 10:31])

    # Vertical line
    draw_line(canvas, mask, 50, 10, 50, 25, 40)
    assert np.all(canvas[10:26, 50] == 40)
    assert np.all(mask[10:26, 50])


def test_rotate_point() -> None:
    # Rotate (10, 0) by 90 degrees CCW -> (0, 10)
    rx, ry = rotate_point(10 * FRACUNIT, 0, ANG90)
    assert abs(rx) < 500  # near 0
    assert abs(ry - 10 * FRACUNIT) < 500

    # Rotate (10, 0) by 180 degrees -> (-10, 0)
    rx, ry = rotate_point(10 * FRACUNIT, 0, ANG180)
    assert abs(rx - (-10 * FRACUNIT)) < 500
    assert abs(ry) < 500


def test_automap_initialization_and_start_stop() -> None:
    automap = AutoMap()
    assert not automap.active
    assert automap.follow_player
    assert not automap.grid
    assert automap.cheating == 0

    v1 = _make_dummy_vertex(-100, -100)
    v2 = _make_dummy_vertex(100, 100)
    sec = _make_dummy_sector()
    line = _make_dummy_line(v1, v2, front_sector=sec)
    lvl = DummyLevel([v1, v2], [line])

    plr = Player()
    plr.mo = Mobj(x=0, y=0, z=0)

    # Start
    automap.start(lvl, plr)
    assert automap.active
    assert automap.min_x == -100 << FRACBITS
    assert automap.max_x == 100 << FRACBITS

    # Center is on player (0, 0)
    center_x = automap.m_x + automap.m_w // 2
    center_y = automap.m_y + automap.m_h // 2
    assert abs(center_x) < 500
    assert abs(center_y) < 500

    # Stop
    automap.stop()
    assert not automap.active

    # Toggle
    automap.toggle(lvl, plr)
    assert automap.active


def test_follow_player_and_panning() -> None:
    v1 = _make_dummy_vertex(-500, -500)
    v2 = _make_dummy_vertex(500, 500)
    sec = _make_dummy_sector()
    lvl = DummyLevel([v1, v2], [_make_dummy_line(v1, v2, front_sector=sec)])

    plr = Player()
    plr.mo = Mobj(x=10 << FRACBITS, y=20 << FRACBITS)

    automap = AutoMap()
    automap.start(lvl, plr)
    assert automap.follow_player

    # Move player and tick
    plr.mo.x = 50 << FRACBITS
    plr.mo.y = 80 << FRACBITS
    automap.tick(plr)

    assert automap.m_x == (50 << FRACBITS) - automap.m_w // 2
    assert automap.m_y == (80 << FRACBITS) - automap.m_h // 2

    # Panning breaks follow mode
    automap.pan(10, 0)
    assert not automap.follow_player
    panned_mx = automap.m_x

    # Subsequent tick does not snap back to player
    plr.mo.x = 200 << FRACBITS
    automap.tick(plr)
    assert automap.m_x == panned_mx

    # Toggle follow restores tracking
    automap.toggle_follow()
    assert automap.follow_player
    automap.tick(plr)
    assert automap.m_x == (200 << FRACBITS) - automap.m_w // 2


def test_zoom_and_max_zoom() -> None:
    v1 = _make_dummy_vertex(-500, -500)
    v2 = _make_dummy_vertex(500, 500)
    sec = _make_dummy_sector()
    lvl = DummyLevel([v1, v2], [_make_dummy_line(v1, v2, front_sector=sec)])
    plr = Player()
    plr.mo = Mobj(x=0, y=0)

    automap = AutoMap()
    automap.start(lvl, plr)

    scale_orig = automap.scale_mtof
    automap.zoom(1)
    assert automap.scale_mtof > scale_orig

    automap.zoom(-1)
    assert abs(automap.scale_mtof - scale_orig) < 500

    scale_before_toggle = automap.scale_mtof
    # Max zoom toggle
    automap.toggle_max_zoom()
    assert automap._is_max_zoom
    assert automap.scale_mtof == automap.min_scale_mtof

    # Toggle back restores previous zoom
    automap.toggle_max_zoom()
    assert not automap._is_max_zoom
    assert automap.scale_mtof == scale_before_toggle


def test_line_classification_rules() -> None:
    automap = AutoMap()
    plr = Player()
    plr.powers[int(PowerType.PW_ALLMAP)] = 0

    v1 = _make_dummy_vertex(0, 0)
    v2 = _make_dummy_vertex(100, 0)
    sec1 = _make_dummy_sector(floorheight=0, ceilingheight=128)
    sec2 = _make_dummy_sector(floorheight=32, ceilingheight=128)
    sec3 = _make_dummy_sector(floorheight=0, ceilingheight=160)
    sec4 = _make_dummy_sector(floorheight=0, ceilingheight=128)

    # 1. Unmapped line -> None
    l_unmapped = _make_dummy_line(v1, v2, flags=0, front_sector=sec1)
    assert automap._classify_line_color(l_unmapped, has_allmap=False) is None

    # 2. Unmapped line with allmap powerup -> ALLMAP_LINE_COLOR
    assert automap._classify_line_color(l_unmapped, has_allmap=True) == ALLMAP_LINE_COLOR

    # 3. DONTDRAW line -> None (unless cheating)
    l_dontdraw = _make_dummy_line(v1, v2, flags=ML_MAPPED | ML_DONTDRAW, front_sector=sec1)
    assert automap._classify_line_color(l_dontdraw, has_allmap=False) is None
    automap.cheating = 1
    assert automap._classify_line_color(l_dontdraw, has_allmap=False) == WALLCOLORS
    automap.cheating = 0

    # 4. Mapped 1-sided wall -> WALLCOLORS (96)
    l_1sided = _make_dummy_line(v1, v2, flags=ML_MAPPED, front_sector=sec1, back_sector=None)
    assert automap._classify_line_color(l_1sided, has_allmap=False) == WALLCOLORS

    # 5. Mapped 2-sided with floor height change -> FDWALLCOLORS (112)
    l_floor = _make_dummy_line(v1, v2, flags=ML_MAPPED, front_sector=sec1, back_sector=sec2)
    assert automap._classify_line_color(l_floor, has_allmap=False) == FDWALLCOLORS

    # 6. Mapped 2-sided with ceiling height change -> CDWALLCOLORS (80)
    l_ceil = _make_dummy_line(v1, v2, flags=ML_MAPPED, front_sector=sec1, back_sector=sec3)
    assert automap._classify_line_color(l_ceil, has_allmap=False) == CDWALLCOLORS

    # 7. Mapped 2-sided with no height change -> hidden unless cheating (TSWALLCOLORS 40)
    l_same = _make_dummy_line(v1, v2, flags=ML_MAPPED, front_sector=sec1, back_sector=sec4)
    assert automap._classify_line_color(l_same, has_allmap=False) is None
    automap.cheating = 1
    assert automap._classify_line_color(l_same, has_allmap=False) == TSWALLCOLORS
    automap.cheating = 0

    # 8. Locked door (special 13) -> GREENKEY (198)
    l_lock = _make_dummy_line(
        v1, v2, flags=ML_MAPPED, special=13, front_sector=sec1, back_sector=sec4
    )
    assert automap._classify_line_color(l_lock, has_allmap=False) == GREENKEY

    # 9. Teleport intra-level (special 70) -> BLUEKEY (157)
    l_tele = _make_dummy_line(
        v1, v2, flags=ML_MAPPED, special=70, front_sector=sec1, back_sector=sec4
    )
    assert automap._classify_line_color(l_tele, has_allmap=False) == BLUEKEY

    # 10. Exit / inter-level teleport (special 74) -> BLOODRED (177)
    l_exit = _make_dummy_line(
        v1, v2, flags=ML_MAPPED, special=74, front_sector=sec1, back_sector=sec4
    )
    assert automap._classify_line_color(l_exit, has_allmap=False) == BLOODRED

    # 11. Secret door (ML_SECRET)
    l_secret = _make_dummy_line(
        v1, v2, flags=ML_MAPPED | ML_SECRET, front_sector=sec1, back_sector=sec4
    )
    assert automap._classify_line_color(l_secret, has_allmap=False) == WALLCOLORS
    automap.cheating = 1
    assert automap._classify_line_color(l_secret, has_allmap=False) == SECRET_WALL_COLOR


def test_keyboard_and_mouse_controls() -> None:
    automap = AutoMap()
    v1 = _make_dummy_vertex(-100, -100)
    v2 = _make_dummy_vertex(100, 100)
    sec = _make_dummy_sector()
    lvl = DummyLevel([v1, v2], [_make_dummy_line(v1, v2, front_sector=sec)])
    plr = Player()
    plr.mo = Mobj(x=0, y=0)

    # Inactive: ignores keys
    assert not automap.handle_key(pygame.K_f)

    # Start
    automap.start(lvl, plr)

    # Key F: toggle follow
    automap.handle_key(pygame.K_f)
    assert not automap.follow_player

    # Key G: toggle grid
    automap.handle_key(pygame.K_g)
    assert automap.grid

    # Key O: toggle overlay
    automap.handle_key(pygame.K_o)
    assert automap.overlay_mode

    # Arrow keys pan while held (follow mode is off now)
    orig_x = automap.m_x
    assert automap.handle_key(pygame.K_RIGHT)
    automap.tick(plr)
    assert automap.m_x > orig_x
    automap.handle_key_up(pygame.K_RIGHT)
    held_x = automap.m_x
    automap.tick(plr)
    assert automap.m_x == held_x

    # Escape is left for the menu
    assert not automap.handle_key(pygame.K_ESCAPE)

    # Mouse drag
    automap.handle_mouse_drag(10, 0)

    # Mouse wheel zoom
    scale_before = automap.scale_mtof
    automap.handle_mouse_wheel(1)
    assert automap.scale_mtof > scale_before

    # Tab stops automap
    automap.handle_key(pygame.K_TAB)
    assert not automap.active


def test_automap_render_draw() -> None:
    overlay = Overlay2D(FontSet.synthetic())
    automap = AutoMap()

    v1 = _make_dummy_vertex(-50, -50)
    v2 = _make_dummy_vertex(50, -50)
    v3 = _make_dummy_vertex(50, 50)
    v4 = _make_dummy_vertex(-50, 50)
    sec = _make_dummy_sector()

    lines = [
        _make_dummy_line(v1, v2, front_sector=sec),
        _make_dummy_line(v2, v3, front_sector=sec),
        _make_dummy_line(v3, v4, front_sector=sec),
        _make_dummy_line(v4, v1, front_sector=sec),
    ]
    lvl = DummyLevel([v1, v2, v3, v4], lines, name="MAP01")

    plr = Player()
    plr.mo = Mobj(x=0, y=0, angle=0)

    # Inactive: draws nothing
    overlay.clear()
    automap.draw(overlay, lvl, plr)
    assert not np.any(overlay.mask)

    # Active: draws parchment background, walls, player arrow, map name, timer
    automap.start(lvl, plr)
    overlay.clear()
    automap.draw(overlay, lvl, plr)
    assert np.any(overlay.mask)
    assert np.any(overlay.indices == PARCH)
    assert np.any(overlay.indices == WALLCOLORS)
    assert np.any(overlay.indices == WHITE)

    # Overlay mode (transparent): parchment background is NOT filled
    automap.overlay_mode = True
    overlay.clear()
    automap.draw(overlay, lvl, plr)
    assert np.any(overlay.mask)
    assert not np.any(overlay.indices == PARCH)

    # Grid enabled
    automap.grid = True
    overlay.clear()
    automap.draw(overlay, lvl, plr)
    assert np.any(overlay.mask)

    # Cheating 2: renders monsters/things
    automap.cheating = 2
    monster = Mobj(x=10 << FRACBITS, y=10 << FRACBITS, health=100, angle=ANG90)
    overlay.clear()
    automap.draw(overlay, lvl, plr, actors=[monster])
    assert np.any(overlay.indices == THINGCOLORS)


@pytest.mark.wad
def test_real_iwad_autopage() -> None:
    iwad_path = resolve_iwad()
    if not iwad_path or not iwad_path.is_file():
        pytest.skip("No real IWAD available")

    wad = WadFile(iwad_path)
    automap = AutoMap(wad=wad)
    assert automap._autopage is not None
    assert len(automap._autopage) >= 320 * 154
