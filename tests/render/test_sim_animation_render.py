# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.6d: the renderer shows the sim's surface animation, sky scroll and lightning."""

from __future__ import annotations

import numpy as np

from fixtures.wadbuilder import MapBuilder, PatchPlacement, TextureDef, WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.specials.anim import load_surface_animator
from pyhexen.wad.mapinfo import MapInfo, load_mapinfo
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

MAPINFO_TEXT = b"""
map 1 "ONE"
warptrans 1
next 1
cluster 1
sky1 SKY1 2
"""


def _build_wad() -> WadBuilder:
    wb = WadBuilder("IWAD")
    wb.add_playpal(14)
    wb.add_colormap(33, include_fogmap=True)
    wb.add_lump("MAPINFO", MAPINFO_TEXT)
    wb.add_marker("P_START")
    wb.add_picture("WALL01", np.full((64, 64), 160, dtype=np.uint8))
    wb.add_picture("SKY1", np.full((128, 256), 90, dtype=np.uint8))
    wb.add_marker("P_END")
    wb.add_textures(
        [
            TextureDef(name="WALL01", width=64, height=64, patches=[PatchPlacement(patch=0)]),
            TextureDef(name="SKY1", width=256, height=128, patches=[PatchPlacement(patch=1)]),
        ],
        pnames=["WALL01", "SKY1"],
    )
    wb.add_marker("F_START")
    wb.add_flat("FLOORA1", fill=30)
    wb.add_flat("FLOORA2", fill=200)
    wb.add_flat("CEIL01", fill=50)
    wb.add_marker("F_END")
    wb.add_lump("ANIMDEFS", b"flat FLOORA1\n pic 1 tics 4\n pic 2 tics 4\n")
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 512, 512, floor_pic="FLOORA1", ceiling_pic="CEIL01", light=255)
    wb.add_map(mb)
    return wb


def test_renderer_shows_sim_animation_without_touching_the_rng() -> None:
    stack = WadStack(_build_wad().to_bytes())
    with stack:
        info = load_mapinfo(stack).get(1)
        assert info is not None and info.sky1_scroll_delta == 2 << 8
        sim = GameSim(level=load_level(stack, "MAP01"))
        sim.surface_animator = load_surface_animator(stack)
        sim.setup_level(sky1_scroll_delta=info.sky1_scroll_delta)

        with RenderContext(standalone=True, width=64, height=48) as ctx:
            _check_frames(ctx, stack, sim, info)


def _check_frames(ctx: RenderContext, stack: WadStack, sim: GameSim, info: MapInfo) -> None:
    renderer = WorldRenderer(ctx, wad=stack)
    renderer.load_map(stack, sim.level, info, event_bus=sim.events)
    camera = Camera()
    camera.x, camera.y, camera.z, camera.yaw = 256.0, 256.0, 41.0, 0.0
    camera.snap_interpolation()

    def floor_pixel() -> tuple[int, ...]:
        rng_before = sim.rng.prndindex
        renderer.render_frame(camera, sim)
        assert sim.rng.prndindex == rng_before  # the renderer never draws P_Random
        img = np.frombuffer(ctx.ctx.fbo.read(components=3), dtype=np.uint8)
        img = img.reshape((48, 64, 3))
        return tuple(int(c) for c in img[2, 32])  # bottom row: floor

    sim.tick([])  # first tic: every sequence moves to its first frame
    first = floor_pixel()
    for _ in range(4):
        sim.tick([])
    second = floor_pixel()
    assert second != first  # FLOORA1 -> FLOORA2 after 4 tics
    for _ in range(4):
        sim.tick([])
    assert floor_pixel() == first

    animator = sim.surface_animator
    assert animator.sky1_column_offset == 9 * (2 << 8)
    assert renderer.sky_renderer.sky1_column_offset == animator.sky1_column_offset
    assert not renderer.sky_renderer.use_sky2
    animator.sky_flash = True  # P_LightningFlash swaps in the map's second sky
    floor_pixel()
    assert renderer.sky_renderer.use_sky2
    assert not hasattr(renderer, "tick")


def test_setup_level_resets_sky_scroll() -> None:
    """R_InitSky: scroll deltas come from MAPINFO and the column offsets restart at 0."""
    stack = WadStack(_build_wad().to_bytes())
    with stack:
        sim = GameSim(level=load_level(stack, "MAP01"))
        sim.surface_animator.sky1_column_offset = 12345
        sim.setup_level(sky1_scroll_delta=3 * FRACUNIT, sky2_scroll_delta=FRACUNIT)
        animator = sim.surface_animator
        assert animator.sky1_column_offset == 0
        assert (animator.sky1_scroll_delta, animator.sky2_scroll_delta) == (3 * FRACUNIT, FRACUNIT)
        sim.tick([])
        assert animator.sky1_column_offset == 3 * FRACUNIT
        assert animator.sky2_column_offset == FRACUNIT
