# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T4.2 (GpuTextureManager), T4.5 (WorldShaderProgram), and T4.6 (SkyRenderer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from tests.fixtures.wadbuilder import PatchPlacement, TextureDef, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.render.context import RenderContext
from pyhexen.render.sky import (
    LIGHTNING_SPECIAL,
    LIGHTNING_SPECIAL2,
    LightningState,
    SkyRenderer,
)
from pyhexen.render.textures import (
    SURFACE_FLAT,
    SURFACE_MASKED_WALL,
    SURFACE_WALL,
    GpuTextureManager,
    WorldShaderProgram,
    compute_colormap_row,
)
from pyhexen.wad.wadfile import WadStack


@dataclass
class _DummySector:
    lightlevel: int
    special: int = 0
    ceilingpic: int = 0
    ceiling_flat: str = "CEIL01"


def _read_pixels(rctx: RenderContext) -> np.ndarray:
    target = rctx._fbo if rctx._fbo is not None else rctx.ctx.screen
    raw = target.read(components=4)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((rctx.height, rctx.width, 4))
    return np.flipud(arr).copy()


def _build_test_wad(tmp_path: Path) -> WadStack:
    builder = WadBuilder("IWAD")
    builder.add_playpal(14)
    builder.add_colormap(33, include_fogmap=True)

    # Patch 0: 8x8 filled with index 40
    p0 = np.full((8, 8), 40, dtype=np.uint8)
    # Patch 1: 8x8 filled with index 80
    p1 = np.full((8, 8), 80, dtype=np.uint8)
    # Patch 2: 8x8 filled with index 120, with bottom half transparent
    p2 = np.full((8, 8), 120, dtype=np.uint8)
    p2_mask = np.ones((8, 8), dtype=bool)
    p2_mask[4:, :] = False

    builder.add_picture("PATCH0", p0)
    builder.add_picture("PATCH1", p1)
    builder.add_picture("PATCH2", p2, mask=p2_mask)

    builder.add_textures(
        textures=[
            TextureDef("WALLA1", 8, 8, [PatchPlacement(0, 0, 0)]),
            TextureDef("WALLA2", 8, 8, [PatchPlacement(0, 0, 1)]),
            TextureDef("MASKWALL", 8, 8, [PatchPlacement(0, 0, 2)], masked=1),
        ],
        pnames=["PATCH0", "PATCH1", "PATCH2"],
    )

    builder.add_marker("F_START")
    builder.add_flat("FLATA1", fill=50)
    builder.add_flat("FLATA2", fill=90)
    builder.add_flat("FLATA3", fill=130)
    builder.add_marker("F_END")

    animdefs_txt = (
        "texture WALLA1\n"
        "pic 1 tics 3\n"
        "pic 2 tics 3\n"
        "\n"
        "flat FLATA1\n"
        "pic 1 tics 2\n"
        "pic 2 rand 4 8\n"
        "pic 3 tics 2\n"
    )
    builder.add_lump("ANIMDEFS", animdefs_txt.encode("ascii"))

    wad_path = builder.write(tmp_path / "test_render.wad")
    return WadStack(wad_path)


def test_gpu_texture_manager_uploads_and_animdefs(tmp_path: Path) -> None:
    wad = _build_test_wad(tmp_path)
    try:
        with RenderContext(width=64, height=64, standalone=True) as rctx:
            assert rctx.ctx is not None
            mgr = GpuTextureManager(rctx.ctx, wad)
            try:
                assert mgr.palette_tex.size == (256, 14)
                assert mgr.colormap_tex.size == (256, 33)
                assert mgr.fogmap_tex.size == (256, 33)

                # Initial translation tables are identity (slot 0 is "-" for wall textures)
                assert mgr.wall_translation == [0, 1, 2, 3]
                assert mgr.flat_translation == [0, 1, 2]

                t1 = mgr.get_wall_texture("WALLA1", translated=False)
                assert t1.width == 8
                assert t1.height == 8
                # Cached lookup returns the exact same object
                assert mgr.get_wall_texture(1, translated=False) is t1

                f0 = mgr.get_flat_texture("FLATA1", translated=False)
                assert f0.width == 64
                assert f0.height == 64

                # Tick 1 advances from initial frame (last) to frame 0 (pic 1 -> base_idx)
                rng = Random()
                mgr.update_animations(1, rng)
                assert mgr.wall_translation[1] == 1
                assert mgr.flat_translation[0] == 0

                # WALLA1 frame 0 has tics=3 -> ticks 2, 3 stay at 1; tick 4 switches to 2
                # FLATA1 frame 0 has tics=2 -> tick 2 stays at 0; tick 3 switches to 1 (rand 4..8)
                mgr.update_animations(2, rng)
                assert mgr.wall_translation[1] == 1
                assert mgr.flat_translation[0] == 0

                mgr.update_animations(3, rng)
                assert mgr.wall_translation[1] == 1
                assert mgr.flat_translation[0] == 1
                assert rng.prndindex == 1  # consumed exactly 1 P_Random call for rand 4..8!

                mgr.update_animations(4, rng)
                assert mgr.wall_translation[1] == 2
                assert mgr.get_wall_texture("WALLA1", translated=True).name == "WALLA2"
            finally:
                mgr.release()
    finally:
        wad.close()


def test_world_shader_palette_and_colormap_exact_match(tmp_path: Path) -> None:
    wad = _build_test_wad(tmp_path)
    try:
        with RenderContext(width=32, height=32, standalone=True) as rctx:
            assert rctx.ctx is not None
            mgr = GpuTextureManager(rctx.ctx, wad)
            shader = WorldShaderProgram(rctx.ctx)
            try:
                wall_tex = mgr.get_wall_texture("WALLA1", translated=False)  # index 40
                flat_tex = mgr.get_flat_texture("FLATA1", translated=False)  # index 50

                view = np.eye(4, dtype=np.float32)
                proj = np.eye(4, dtype=np.float32)
                proj[2, 2] = 0.0  # keep clip.z = 0 regardless of view_pos.z

                def _render_quad(
                    gpu_tex,
                    light: float,
                    depth: float,
                    surface_type: float,
                    *,
                    palette_index: int = 0,
                    extralight: int = 0,
                    fixed_colormap: int = -1,
                    use_fogmap: bool = False,
                ) -> tuple[int, int, int]:
                    z = -depth
                    verts = np.array(
                        [
                            [-1.0, -1.0, z, 0.0, 0.0, light, surface_type, 0.0],
                            [1.0, -1.0, z, 4.0, 0.0, light, surface_type, 0.0],
                            [1.0, 1.0, z, 4.0, 4.0, light, surface_type, 0.0],
                            [-1.0, -1.0, z, 0.0, 0.0, light, surface_type, 0.0],
                            [1.0, 1.0, z, 4.0, 4.0, light, surface_type, 0.0],
                            [-1.0, 1.0, z, 0.0, 4.0, light, surface_type, 0.0],
                        ],
                        dtype=np.float32,
                    )
                    vbo = rctx.ctx.buffer(verts.tobytes())
                    vao = shader.create_vao(vbo)
                    try:
                        rctx.clear(0.0, 0.0, 0.0)
                        shader.set_camera(view, proj)
                        shader.set_lighting(
                            palette_index=palette_index,
                            extralight=extralight,
                            fixed_colormap=fixed_colormap,
                            use_fogmap=use_fogmap,
                        )
                        shader.bind_surface_texture(mgr, gpu_tex)
                        vao.render()
                        fb = _read_pixels(rctx)
                        px = fb[16, 16]
                        return int(px[0]), int(px[1]), int(px[2])
                    finally:
                        vao.release()
                        vbo.release()

                # 1. Wall at various light levels and depths
                for light, depth in [(255.0, 32.0), (160.0, 64.0), (160.0, 512.0), (96.0, 256.0)]:
                    rgb = _render_quad(wall_tex, light, depth, SURFACE_WALL)
                    expected_row = compute_colormap_row(light, depth, SURFACE_WALL)
                    shaded_idx = int(mgr.colormap[expected_row, 40])
                    expected_rgb = tuple(int(c) for c in mgr.palettes[0, shaded_idx])
                    assert rgb == expected_rgb

                # 2. Flat at light=144, depth=80.0
                rgb_flat = _render_quad(flat_tex, 144.0, 80.0, SURFACE_FLAT)
                expected_flat_row = compute_colormap_row(144.0, 80.0, SURFACE_FLAT)
                shaded_flat_idx = int(mgr.colormap[expected_flat_row, 50])
                expected_flat_rgb = tuple(int(c) for c in mgr.palettes[0, shaded_flat_idx])
                assert rgb_flat == expected_flat_rgb

                # 3. Palette flash (palette_index=3) + fixed_colormap=5
                rgb_flash = _render_quad(
                    wall_tex,
                    128.0,
                    100.0,
                    SURFACE_WALL,
                    palette_index=3,
                    fixed_colormap=5,
                )
                shaded_fixed = int(mgr.colormap[5, 40])
                expected_flash_rgb = tuple(int(c) for c in mgr.palettes[3, shaded_fixed])
                assert rgb_flash == expected_flash_rgb

                # 4. Fogmap override
                rgb_fog = _render_quad(
                    wall_tex,
                    128.0,
                    128.0,
                    SURFACE_WALL,
                    use_fogmap=True,
                )
                fog_row = compute_colormap_row(128.0, 128.0, SURFACE_WALL)
                shaded_fog = int(mgr.fogmap[fog_row, 40])
                expected_fog_rgb = tuple(int(c) for c in mgr.palettes[0, shaded_fog])
                assert rgb_fog == expected_fog_rgb
            finally:
                shader.release()
                mgr.release()
    finally:
        wad.close()


def test_world_shader_mask_discard_and_masked_wall_clamp(tmp_path: Path) -> None:
    wad = _build_test_wad(tmp_path)
    try:
        with RenderContext(width=32, height=32, standalone=True) as rctx:
            assert rctx.ctx is not None
            mgr = GpuTextureManager(rctx.ctx, wad)
            shader = WorldShaderProgram(rctx.ctx)
            try:
                masked_tex = mgr.get_wall_texture("MASKWALL", translated=False)
                view = np.eye(4, dtype=np.float32)
                proj = np.eye(4, dtype=np.float32)
                proj[2, 2] = 0.0

                # Render quad spanning v = 0..16 on an 8x8 texture with SURFACE_MASKED_WALL (3.0)
                # Top half (v in 8..16) is out of vertical range -> discarded (background red)
                # Bottom half (v in 0..8): v in 0..4 opaque (120), v in 4..8 mask=0 -> discarded
                verts = np.array(
                    [
                        [-1.0, -1.0, -32.0, 0.0, 0.0, 255.0, SURFACE_MASKED_WALL, 0.0],
                        [1.0, -1.0, -32.0, 8.0, 0.0, 255.0, SURFACE_MASKED_WALL, 0.0],
                        [1.0, 1.0, -32.0, 8.0, 16.0, 255.0, SURFACE_MASKED_WALL, 0.0],
                        [-1.0, -1.0, -32.0, 0.0, 0.0, 255.0, SURFACE_MASKED_WALL, 0.0],
                        [1.0, 1.0, -32.0, 8.0, 16.0, 255.0, SURFACE_MASKED_WALL, 0.0],
                        [-1.0, 1.0, -32.0, 0.0, 16.0, 255.0, SURFACE_MASKED_WALL, 0.0],
                    ],
                    dtype=np.float32,
                )
                vbo = rctx.ctx.buffer(verts.tobytes())
                vao = shader.create_vao(vbo)
                try:
                    rctx.clear(1.0, 0.0, 0.0)
                    shader.set_camera(view, proj)
                    shader.set_lighting(fixed_colormap=0)
                    shader.bind_surface_texture(mgr, masked_tex)
                    vao.render()
                    fb = _read_pixels(rctx)
                    # Note: read_pixels flips vertically so row 0 is top of screen (v near 16)
                    # Top of screen (row 2): v > 8 -> discarded by SURFACE_MASKED_WALL -> red
                    assert tuple(int(c) for c in fb[2, 16, :3]) == (255, 0, 0)
                    # Middle-bottom (row 20, v in 4..8): mask == 0 -> discarded -> red (255, 0, 0)
                    assert tuple(int(c) for c in fb[20, 16, :3]) == (255, 0, 0)
                    # Bottom (row 30, v in 0..4): opaque index 120 -> PLAYPAL[0][COLORMAP[0][120]]
                    shaded_idx = int(mgr.colormap[0, 120])
                    expected_rgb = tuple(int(c) for c in mgr.palettes[0, shaded_idx])
                    assert tuple(int(c) for c in fb[30, 16, :3]) == expected_rgb
                finally:
                    vao.release()
                    vbo.release()
            finally:
                shader.release()
                mgr.release()
    finally:
        wad.close()


def test_sky_renderer_single_double_and_scroll(tmp_path: Path) -> None:
    wad = _build_test_wad(tmp_path)
    try:
        with RenderContext(width=32, height=32, standalone=True) as rctx:
            assert rctx.ctx is not None
            mgr = GpuTextureManager(rctx.ctx, wad)
            sky = SkyRenderer(rctx.ctx)
            try:
                # Create synthetic sky1 (left half index 0 = transparent in doublesky, right 60)
                # and sky2 (solid index 110)
                sky1_arr = np.zeros((200, 256), dtype=np.uint8)
                sky1_arr[:, :128] = 0
                sky1_arr[:, 128:] = 60
                sky2_arr = np.full((200, 256), 110, dtype=np.uint8)

                sky1_tex = mgr.upload_indexed("SKY1", sky1_arr)
                sky2_tex = mgr.upload_indexed("SKY2", sky2_arr)
                try:
                    # 1. Single sky: index 0 is rendered directly using PLAYPAL[0][0]
                    sky.set_skies(sky1_tex, sky2_tex, double_sky=False)
                    rctx.clear(0.0, 0.0, 0.0)
                    sky.render_fullscreen_quad(mgr, camera_yaw=0.0)
                    fb_single = _read_pixels(rctx)
                    pal0_0 = tuple(int(c) for c in mgr.palettes[0, 0])
                    pal0_60 = tuple(int(c) for c in mgr.palettes[0, 60])
                    pal0_110 = tuple(int(c) for c in mgr.palettes[0, 110])

                    # Center-right vs center-left columns sample both halves of sky1
                    colors_single = {tuple(int(c) for c in fb_single[16, x, :3]) for x in range(32)}
                    assert pal0_0 in colors_single or pal0_60 in colors_single
                    assert pal0_110 not in colors_single

                    # 2. Double sky: wherever sky1 has index 0, sky2 (index 110) shows through!
                    sky.set_skies(
                        sky1_tex,
                        sky2_tex,
                        sky1_scroll_delta=64 * FRACUNIT,
                        sky2_scroll_delta=32 * FRACUNIT,
                        double_sky=True,
                    )
                    rctx.clear(0.0, 0.0, 0.0)
                    sky.render_fullscreen_quad(mgr, camera_yaw=0.0)
                    fb_double = _read_pixels(rctx)
                    colors_double = {tuple(int(c) for c in fb_double[16, x, :3]) for x in range(32)}
                    assert pal0_0 not in colors_double
                    assert colors_double.issubset({pal0_60, pal0_110})

                    # 3. Scrolling ticks advance column offsets
                    sky.tick()
                    assert sky.sky1_offset_texels == pytest.approx(64.0)
                    assert sky.sky2_offset_texels == pytest.approx(32.0)
                finally:
                    sky1_tex.release()
                    sky2_tex.release()
            finally:
                sky.release()
                mgr.release()
    finally:
        wad.close()


def test_lightning_state_flash_and_sector_decay() -> None:
    rng = Random()
    state = LightningState()
    state.init(enabled=True, rng=rng, sky_flat_num=99)
    assert state.enabled is True
    assert state.next_flash > 0
    assert state.use_sky2 is False

    sectors = [
        _DummySector(lightlevel=120, special=0, ceilingpic=99),  # outdoor sky sector
        _DummySector(lightlevel=100, special=LIGHTNING_SPECIAL, ceilingpic=1),  # +64
        _DummySector(lightlevel=100, special=LIGHTNING_SPECIAL2, ceilingpic=1),  # +32
        _DummySector(lightlevel=80, special=0, ceilingpic=1),  # unaffected indoor sector
    ]

    # Force immediate lightning flash
    state.force_lightning(mode=0, rng=rng, sectors=sectors)
    assert state.use_sky2 is True
    assert 200 <= state.flash_light <= 231
    assert 8 <= state.flash_count <= 15
    assert sectors[0].lightlevel == state.flash_light
    assert sectors[1].lightlevel == 164
    assert sectors[2].lightlevel == 132
    assert sectors[3].lightlevel == 80

    # Tick once during flash: boosted sectors decay by 4
    initial_count = state.flash_count
    active = state.tick(rng=rng, sectors=sectors)
    assert active is True
    assert state.flash_count == initial_count - 1
    assert sectors[0].lightlevel == state.flash_light - 4
    assert sectors[1].lightlevel == 160
    assert sectors[2].lightlevel == 128
    assert sectors[3].lightlevel == 80

    # Tick until flash ends: original sector lights are restored and use_sky2 resets to False
    while state.flash_count > 0:
        state.tick(rng=rng, sectors=sectors)

    assert state.use_sky2 is False
    assert sectors[0].lightlevel == 120
    assert sectors[1].lightlevel == 100
    assert sectors[2].lightlevel == 100
    assert sectors[3].lightlevel == 80
