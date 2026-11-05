# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T2.8 (wadinfo CLI) and T3.4 (2D map viewer / Milestone M1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame
import pytest
from tests.fixtures.wadbuilder import MapBuilder, PatchPlacement, TextureDef, WadBuilder

from pyhexen.tools import mapview2d, wadinfo
from pyhexen.tools.mapview2d import (
    COLOR_LINE_ONESIDED,
    COLOR_LINE_POLYOBJ,
    COLOR_LINE_SPECIAL,
    COLOR_LINE_TWOSIDED_DIFF,
    COLOR_LINE_TWOSIDED_SAME,
    MapView2D,
    classify_line_color,
    classify_thing_category,
)
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Line, MapThing, Sector, load_level


@pytest.fixture
def synthetic_m1_wad(tmp_path: Path) -> Path:
    """Create a synthetic IWAD on disk with graphics, MAPINFO, and MAP01."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(num_palettes=2)
    wb.add_colormap(num_rows=33, include_fogmap=True)

    # Patch lump inside P_START..P_END
    patch_pixels = np.arange(64, dtype=np.uint8).reshape((8, 8))
    wb.add_marker("P_START")
    wb.add_picture("PATCH01", patch_pixels, left=0, top=0)
    wb.add_marker("P_END")

    # Composite texture WALL01 using PATCH01
    wb.add_textures(
        [
            TextureDef(
                name="WALL01",
                width=8,
                height=8,
                patches=[PatchPlacement(originx=0, originy=0, patch=0)],
            )
        ],
        pnames=["PATCH01"],
    )

    # Flats inside F_START..F_END
    wb.add_marker("F_START")
    wb.add_flat("FLOOR01", fill=12)
    wb.add_flat("CEIL01", fill=34)
    wb.add_marker("F_END")

    # Sprite inside S_START..S_END
    wb.add_marker("S_START")
    wb.add_picture("TROOA0", patch_pixels, left=4, top=8)
    wb.add_marker("S_END")

    # MAPINFO lump
    mapinfo_text = """
    map 1 "WINNOWING HALL"
    warptrans 1
    next 2
    cluster 1
    sky1 SKY1 0
    lightning
    cdtrack 13
    """
    wb.add_lump("MAPINFO", mapinfo_text.encode("ascii"))

    # MAP01 with two connected rooms and several categorized things
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=0,
        y=0,
        width1=256,
        width2=256,
        height=256,
        floor1=0,
        ceil1=128,
        floor2=24,
        ceil2=128,
        portal_special=12,
        portal_args=(1, 16, 0, 0, 0),
    )
    mb.add_thing(x=64, y=128, type=1, angle=0)  # Player 1 start
    mb.add_thing(x=384, y=128, type=31, angle=180)  # Monster
    mb.add_thing(x=128, y=64, type=3000, angle=90)  # Polyobj anchor
    mb.add_thing(x=192, y=192, type=122, angle=45)  # Item
    wb.add_map(mb)

    wad_path = tmp_path / "m1_test.wad"
    return wb.write(wad_path)


def test_wadinfo_list_and_maps(synthetic_m1_wad: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = wadinfo.main(["--iwad", str(synthetic_m1_wad), "list"])
    assert rc == 0
    out_list = capsys.readouterr().out
    assert "PLAYPAL" in out_list
    assert "WALL01" not in out_list  # WALL01 is inside TEXTURE1, not a raw lump
    assert "TEXTURE1" in out_list
    assert "Sprites: 1 | Flats: 2 | Patches: 1 | Textures: 1" in out_list

    rc_maps = wadinfo.main(["--iwad", str(synthetic_m1_wad), "maps"])
    assert rc_maps == 0
    out_maps = capsys.readouterr().out
    assert "MAP01" in out_maps
    assert "WINNOWING HALL" in out_maps
    assert "SKY1" in out_maps


def test_wadinfo_extract(synthetic_m1_wad: Path, tmp_path: Path) -> None:
    extract_dir = tmp_path / "extracted"
    rc = wadinfo.main(
        ["--iwad", str(synthetic_m1_wad), "extract", "PLAYPAL", "--out", str(extract_dir)]
    )
    assert rc == 0
    lmp_file = extract_dir / "PLAYPAL.lmp"
    assert lmp_file.is_file()
    assert len(lmp_file.read_bytes()) == 2 * 768


def test_wadinfo_export_png(synthetic_m1_wad: Path, tmp_path: Path) -> None:
    png_dir = tmp_path / "pngs"
    for args, expected_name in [
        (["export-png", "texture", "WALL01", "--out", str(png_dir)], "WALL01.png"),
        (["export-png", "flat", "FLOOR01", "--out", str(png_dir)], "FLOOR01.png"),
        (["export-png", "patch", "PATCH01", "--out", str(png_dir)], "PATCH01.png"),
        (["export-png", "sprite", "TROOA0", "--out", str(png_dir)], "TROOA0.png"),
        (["export-png", "palette", "--out", str(png_dir)], "PLAYPAL.png"),
        (["export-png", "colormap", "--out", str(png_dir)], "COLORMAP.png"),
    ]:
        rc = wadinfo.main(["--iwad", str(synthetic_m1_wad), *args])
        assert rc == 0
        out_png = png_dir / expected_name
        assert out_png.is_file()
        raw = out_png.read_bytes()
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_mapview2d_transforms_and_classification(synthetic_m1_wad: Path) -> None:
    with WadStack(synthetic_m1_wad) as wad:
        level = load_level(wad, "MAP01")
    viewer = MapView2D(level, width=800, height=600)

    # Coordinate round-trip
    for wx, wy in [(0.0, 0.0), (128.5, 64.25), (512.0, 256.0)]:
        sx, sy = viewer.world_to_screen(wx, wy)
        rx, ry = viewer.screen_to_world(sx, sy)
        assert rx == pytest.approx(wx, abs=1.0 / viewer.scale + 1e-3)
        assert ry == pytest.approx(wy, abs=1.0 / viewer.scale + 1e-3)

    # Zoom at center preserves the world point at screen center
    cx, cy = 400.0, 300.0
    wx_before, wy_before = viewer.screen_to_world(cx, cy)
    viewer.zoom_at(2.0, (cx, cy))
    wx_after, wy_after = viewer.screen_to_world(cx, cy)
    assert wx_after == pytest.approx(wx_before, abs=1.0)
    assert wy_after == pytest.approx(wy_before, abs=1.0)

    # Pan by pixels shifts screen coordinates by (dx, dy)
    sx0, sy0 = viewer.world_to_screen(100.0, 100.0)
    viewer.pan_pixels(25.0, -15.0)
    sx1, sy1 = viewer.world_to_screen(100.0, 100.0)
    assert sx1 == sx0 + 25
    assert sy1 == sy0 - 15

    # Line colour classifications
    one_sided = level.lines[0]
    portal_special = level.lines[3]
    assert classify_line_color(one_sided) == COLOR_LINE_ONESIDED
    assert classify_line_color(portal_special) == COLOR_LINE_SPECIAL

    # Construct synthetic line variations for polyobj, height-change, and same-height
    poly_line = Line(
        index=99,
        v1=one_sided.v1,
        v2=one_sided.v2,
        dx=one_sided.dx,
        dy=one_sided.dy,
        flags=one_sided.flags,
        special=1,
        args=(1, 0, 0, 0, 0),
        sidenum=one_sided.sidenum,
        front_side=one_sided.front_side,
        back_side=None,
        front_sector=level.sectors[0],
        back_sector=None,
        slopetype=one_sided.slopetype,
        bbox=one_sided.bbox,
    )
    assert classify_line_color(poly_line) == COLOR_LINE_POLYOBJ

    height_change_line = Line(
        index=100,
        v1=portal_special.v1,
        v2=portal_special.v2,
        dx=portal_special.dx,
        dy=portal_special.dy,
        flags=portal_special.flags,
        special=0,
        args=(0, 0, 0, 0, 0),
        sidenum=portal_special.sidenum,
        front_side=portal_special.front_side,
        back_side=portal_special.back_side,
        front_sector=level.sectors[0],
        back_sector=level.sectors[1],
        slopetype=portal_special.slopetype,
        bbox=portal_special.bbox,
    )
    assert classify_line_color(height_change_line) == COLOR_LINE_TWOSIDED_DIFF

    same_height_line = Line(
        index=101,
        v1=portal_special.v1,
        v2=portal_special.v2,
        dx=portal_special.dx,
        dy=portal_special.dy,
        flags=portal_special.flags,
        special=0,
        args=(0, 0, 0, 0, 0),
        sidenum=portal_special.sidenum,
        front_side=portal_special.front_side,
        back_side=portal_special.back_side,
        front_sector=level.sectors[0],
        back_sector=level.sectors[0],
        slopetype=portal_special.slopetype,
        bbox=portal_special.bbox,
    )
    assert classify_line_color(same_height_line) == COLOR_LINE_TWOSIDED_SAME

    # Thing category classifications
    assert classify_thing_category(level.things[0])[0] == "player"
    assert classify_thing_category(level.things[1])[0] == "monster"
    assert classify_thing_category(level.things[2])[0] == "polyobj"
    assert classify_thing_category(level.things[3])[0] == "item"


def test_mapview2d_pick_element_and_headless_render(synthetic_m1_wad: Path, tmp_path: Path) -> None:
    with WadStack(synthetic_m1_wad) as wad:
        level = load_level(wad, "MAP01")
    viewer = MapView2D(level, width=640, height=400)

    # 1. Pick a MapThing near (64, 128)
    picked_thing = viewer.pick_element(64.0, 128.0)
    assert isinstance(picked_thing, MapThing)
    assert picked_thing.type == 1
    tooltip_thing = viewer.format_tooltip(picked_thing)
    assert any("category=player" in line for line in tooltip_thing)

    # 2. Pick a Line near the top wall of room 1 (x=100, y=256)
    picked_line = viewer.pick_element(100.0, 256.0)
    assert isinstance(picked_line, Line)
    tooltip_line = viewer.format_tooltip(picked_line)
    assert any("Line #" in line for line in tooltip_line)

    # 3. Pick a Sector in the interior of room 2 away from walls and things (x=450, y=64)
    picked_sector = viewer.pick_element(450.0, 64.0)
    assert isinstance(picked_sector, Sector)
    assert picked_sector is level.sectors[1]
    tooltip_sector = viewer.format_tooltip(picked_sector)
    assert any("Sector #1" in line for line in tooltip_sector)

    # Render to headless pygame Surface
    pygame.init()
    surface = pygame.Surface((640, 400))
    viewer.render_to_surface(surface, hover_world=(64.0, 128.0))

    # CLI headless PNG output
    out_png = tmp_path / "map01_headless.png"
    rc = mapview2d.main(
        [
            "--iwad",
            str(synthetic_m1_wad),
            "--warp",
            "MAP01",
            "--headless-out",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.is_file()
    assert out_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
