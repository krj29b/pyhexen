# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Adversarial stress tests for polyobject GPU mesh builder and renderer (T9.3).

Stress-tests multi-polyobject simultaneous motion, degenerate and empty geometry,
extreme coordinates, depth buffer occlusion, and fly mode camera angles.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fixtures.wadbuilder import MapBuilder, PatchPlacement, TextureDef, WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.mesh_walls import SURFACE_MASKED_WALL, SURFACE_OPAQUE_WALL
from pyhexen.render.polyobj import (
    PolyMesh,
    build_polyobj_mesh,
    compute_poly_model_matrix,
    interpolate_angle,
    interpolate_coord,
    poly_model_to_gl_matrix,
)
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.polyobj import DegenMobj, PolyDoorThinker, Polyobj, RotatePolyThinker
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Sector, Seg, Side, Vertex, load_level


def _create_adversarial_multi_poly_level() -> tuple[
    WadStack, WorldRenderer, RenderContext, GameSim
]:
    """Create a synthetic level with 4 distinct rooms containing 4 active polyobjects."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(1)
    wb.add_colormap(33, include_fogmap=True)

    tex_wall = np.full((64, 64), 140, dtype=np.uint8)
    wb.add_marker("P_START")
    wb.add_picture("WALL01", tex_wall)
    wb.add_marker("P_END")
    wb.add_textures(
        [TextureDef(name="WALL01", width=64, height=64, patches=[PatchPlacement(patch=0)])],
        pnames=["WALL01"],
    )
    wb.add_flat("FLOOR01", fill=30)
    wb.add_flat("CEIL01", fill=50)

    mb = MapBuilder("MAP01")
    # 4 distinct rooms side-by-side along the X-axis so auto-BSP
    # partitions them into distinct subsectors.
    mb.add_room(
        x=0,
        y=0,
        width=300,
        height=300,
        floor_pic="FLOOR01",
        ceiling_pic="CEIL01",
        wall_tex="WALL01",
        light=180,
    )
    mb.add_room(
        x=400,
        y=0,
        width=300,
        height=300,
        floor_pic="FLOOR01",
        ceiling_pic="CEIL01",
        wall_tex="WALL01",
        light=180,
    )
    mb.add_room(
        x=800,
        y=0,
        width=300,
        height=300,
        floor_pic="FLOOR01",
        ceiling_pic="CEIL01",
        wall_tex="WALL01",
        light=180,
    )
    mb.add_room(
        x=1200,
        y=0,
        width=300,
        height=300,
        floor_pic="FLOOR01",
        ceiling_pic="CEIL01",
        wall_tex="WALL01",
        light=180,
    )

    mb.add_polyobject_box(1, -2000, -2000, 150, 150, width=64, height=64)
    mb.add_polyobject_box(2, -1500, -2000, 550, 150, width=64, height=64)
    mb.add_polyobject_box(3, -1000, -2000, 950, 150, width=64, height=64)
    mb.add_polyobject_box(4, -500, -2000, 1350, 150, width=64, height=64)

    mb.add_thing(tid=1, x=50, y=50, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)

    ctx = RenderContext(standalone=True, width=320, height=200)
    renderer = WorldRenderer(ctx, wad=stack)
    renderer.load_map(stack, level)

    return stack, renderer, ctx, sim


def test_multi_polyobject_simultaneous_movement_and_rotation() -> None:
    """Stress-test 4 polyobjects simultaneously moving and rotating across multiple frames."""
    stack, renderer, ctx, sim = _create_adversarial_multi_poly_level()
    with stack:
        level = sim.level
        assert len(renderer.poly_meshes) == 4

        # PO 1: Slide North
        door1 = PolyDoorThinker(
            poly=level.polyobjs[1],
            door_type=0,
            speed=2 * FRACUNIT,
            dist=64 * FRACUNIT,
            total_dist=64 * FRACUNIT,
            direction=0,
            x_speed=0,
            y_speed=2 * FRACUNIT,
            wait_tics=35,
        )
        # PO 2: Slide South
        door2 = PolyDoorThinker(
            poly=level.polyobjs[2],
            door_type=0,
            speed=2 * FRACUNIT,
            dist=64 * FRACUNIT,
            total_dist=64 * FRACUNIT,
            direction=0,
            x_speed=0,
            y_speed=-2 * FRACUNIT,
            wait_tics=35,
        )
        # PO 3: Rotate perpetual
        rot3 = RotatePolyThinker(poly=level.polyobjs[3], speed=5, dist=0)
        # PO 4: Swing door
        swing4 = PolyDoorThinker(
            poly=level.polyobjs[4],
            door_type=1,
            speed=2,
            dist=64,
            total_dist=64,
            direction=1,
            x_speed=0,
            y_speed=0,
            wait_tics=35,
        )

        for t in (door1, door2, rot3, swing4):
            sim.thinkers.add(t)

        cam = Camera()
        cam.x = 150.0
        cam.y = 50.0
        cam.z = 100.0
        cam.yaw = math.pi / 2.0
        cam.pitch = 0.0

        for _ in range(5):
            sim.tick()
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                renderer.render_frame(cam, sim=sim, alpha=alpha)
                pixels = ctx.read_pixels()
                assert pixels.shape == (200, 320, 4)
                assert not np.isnan(pixels).any()

        renderer.release()


def test_polyobj_degenerate_empty_segs() -> None:
    """Verify that a polyobject with 0 segs produces an empty mesh without crashing."""
    po_empty = Polyobj(tag=99, index=0, numsegs=0, segs=[], start_spot=DegenMobj(0, 0, 0))
    mesh = build_polyobj_mesh(po_empty)

    assert isinstance(mesh, PolyMesh)
    assert mesh.tag == 99
    assert mesh.vertex_count == 0
    assert mesh.raw_vertices.shape == (0, 8)
    assert len(mesh.batches) == 0
    assert len(mesh.vertices) == 0
    assert len(mesh.uvs) == 0


def test_polyobj_degenerate_zero_length_seg() -> None:
    """Verify zero-length seg (v1 == v2) generates a valid mesh without zero-division or NaN."""
    v0 = Vertex(100, 100)
    seg_zero = Seg(
        index=0,
        v1=v0,
        v2=v0,
        offset=0,
        angle=0,
        linedef=None,
        sidedef=None,
        front_sector=None,
        back_sector=None,
    )
    po_zero = Polyobj(tag=100, index=1, numsegs=1, segs=[seg_zero], start_spot=DegenMobj(0, 0, 0))
    mesh_zero = build_polyobj_mesh(po_zero)

    assert mesh_zero.vertex_count == 6
    assert mesh_zero.raw_vertices.shape == (6, 8)
    assert not np.isnan(mesh_zero.raw_vertices).any()
    assert not np.isinf(mesh_zero.raw_vertices).any()


def test_polyobj_missing_sidedef_and_sector_fallback() -> None:
    """Verify graceful fallback when seg has no sidedef, linedef, or subsector."""
    v1 = Vertex(0, 0)
    v2 = Vertex(64, 0)
    seg = Seg(
        index=0,
        v1=v1,
        v2=v2,
        offset=0,
        angle=0,
        linedef=None,
        sidedef=None,
        front_sector=None,
        back_sector=None,
    )
    po = Polyobj(tag=101, index=0, numsegs=1, segs=[seg], start_spot=DegenMobj(0, 0, 0))
    mesh = build_polyobj_mesh(po)

    assert mesh.z_bottom == 0.0
    assert mesh.z_top == 128.0
    assert mesh.vertex_count == 6
    assert len(mesh.batches) == 1
    # Check default surface type
    assert mesh.batches[0].surface_type == SURFACE_OPAQUE_WALL


def test_polyobj_two_sided_masked_surface_type() -> None:
    """Verify that a 2-sided polyobject line receives SURFACE_MASKED_WALL."""
    sec_front = Sector(
        index=0,
        floor_height=0,
        ceiling_height=128 * FRACUNIT,
        floor_pic="F",
        ceiling_pic="C",
        light_level=160,
        special=0,
        tag=0,
    )
    sec_back = Sector(
        index=1,
        floor_height=0,
        ceiling_height=128 * FRACUNIT,
        floor_pic="F",
        ceiling_pic="C",
        light_level=160,
        special=0,
        tag=0,
    )

    side_front = Side(
        index=0,
        texture_offset=0,
        row_offset=0,
        top_texture="",
        bottom_texture="",
        mid_texture="BARS01",
        sector=sec_front,
    )
    v1 = Vertex(0, 0)
    v2 = Vertex(64, 0)
    seg = Seg(
        index=0,
        v1=v1,
        v2=v2,
        offset=0,
        angle=0,
        sidedef=side_front,
        linedef=None,
        front_sector=sec_front,
        back_sector=sec_back,
    )

    po = Polyobj(tag=102, index=0, numsegs=1, segs=[seg], start_spot=DegenMobj(0, 0, 0))
    mesh = build_polyobj_mesh(po)

    assert mesh.vertex_count == 6
    assert len(mesh.batches) == 1
    assert mesh.batches[0].surface_type == SURFACE_MASKED_WALL


def test_polyobj_extreme_model_matrix_transforms() -> None:
    """Verify matrix operations with extreme coordinate numbers and angle rotations."""
    m_huge = compute_poly_model_matrix(1e8, -1e8, 500.0 * math.pi)
    assert not np.isnan(m_huge).any()
    assert not np.isinf(m_huge).any()

    m_gl = poly_model_to_gl_matrix(m_huge)
    assert not np.isnan(m_gl).any()
    assert not np.isinf(m_gl).any()
    assert m_gl.shape == (4, 4)

    # Modular interpolation across 360 wrap
    lerped_ang = interpolate_angle(359.9, 0.1, 0.5)
    assert abs(lerped_ang - 0.0) < 0.2 or abs(lerped_ang - 360.0) < 0.2

    lerped_coord = interpolate_coord(-1e6, 1e6, 0.5)
    assert abs(lerped_coord - 0.0) < 1e-4


def test_polyobj_depth_occlusion_and_render_pass() -> None:
    """Verify polyobject correctly occludes background geometry in depth buffer."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(1)
    wb.add_colormap(33, include_fogmap=True)

    # Distinct solid textures for foreground polyobject and background wall
    tex_poly = np.full((64, 64), 100, dtype=np.uint8)
    tex_back = np.full((64, 64), 200, dtype=np.uint8)

    wb.add_marker("P_START")
    wb.add_picture("POLY_TEX", tex_poly)
    wb.add_picture("BACK_TEX", tex_back)
    wb.add_marker("P_END")
    wb.add_textures(
        [
            TextureDef(name="POLY_TEX", width=64, height=64, patches=[PatchPlacement(patch=0)]),
            TextureDef(name="BACK_TEX", width=64, height=64, patches=[PatchPlacement(patch=1)]),
        ],
        pnames=["POLY_TEX", "BACK_TEX"],
    )
    wb.add_flat("FLOOR01", fill=30)
    wb.add_flat("CEIL01", fill=50)

    mb = MapBuilder("MAP01")
    mb.add_room(
        0, 0, 512, 600, floor_pic="FLOOR01", ceiling_pic="CEIL01", wall_tex="BACK_TEX", light=180
    )
    mb.add_polyobject_box(
        tag=1,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=256,
        spawn_y=256,
        width=128,
        height=64,
        wall_tex="POLY_TEX",
    )
    mb.add_thing(tid=1, x=256, y=64, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)

    ctx = RenderContext(standalone=True, width=320, height=200)
    renderer = WorldRenderer(ctx, wad=stack)
    renderer.load_map(stack, level)

    with stack:
        # Camera 1: At (256, 100, 64) looking North (+Y) with Polyobject in direct line of sight
        cam1 = Camera()
        cam1.x = 256.0
        cam1.y = 100.0
        cam1.z = 64.0
        cam1.yaw = math.pi / 2.0
        cam1.pitch = 0.0

        renderer.render_frame(cam1, sim=sim)
        px_with_poly = ctx.read_pixels()[100, 160].copy()

        # Camera 2: At (256, 350, 64) behind the polyobject, looking at back wall (BACK_TEX)
        cam2 = Camera()
        cam2.x = 256.0
        cam2.y = 350.0
        cam2.z = 64.0
        cam2.yaw = math.pi / 2.0
        cam2.pitch = 0.0

        renderer.render_frame(cam2, sim=sim)
        px_back_wall = ctx.read_pixels()[100, 160].copy()

        # Center pixel with polyobject must be distinct from the back wall pixel
        assert not np.array_equal(px_with_poly, px_back_wall)

        renderer.release()


def test_polyobj_fly_mode_steep_camera_angles(tmp_path: Path) -> None:
    """Stress-test fly mode camera at high elevation, steep look-down, and steep look-up."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(1)
    wb.add_colormap(33, include_fogmap=True)
    tex_data = np.full((64, 64), 120, dtype=np.uint8)
    wb.add_marker("P_START")
    wb.add_picture("WALL01", tex_data)
    wb.add_marker("P_END")
    wb.add_textures(
        [TextureDef(name="WALL01", width=64, height=64, patches=[PatchPlacement(patch=0)])],
        pnames=["WALL01"],
    )
    wb.add_flat("FLOOR01", fill=30)
    wb.add_flat("CEIL01", fill=50)

    mb = MapBuilder("MAP01")
    mb.add_room(
        0, 0, 512, 512, floor_pic="FLOOR01", ceiling_pic="CEIL01", wall_tex="WALL01", light=180
    )
    mb.add_polyobject_box(
        tag=1,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=256,
        spawn_y=256,
        width=64,
        height=64,
        wall_tex="WALL01",
    )
    mb.add_thing(tid=1, x=256, y=64, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)

    ctx = RenderContext(standalone=True, width=320, height=200)
    renderer = WorldRenderer(ctx, wad=stack)
    renderer.load_map(stack, level)

    with stack:
        cam = Camera()

        # 1. High elevation, steep look-down (-85 degrees)
        cam.x = 256.0
        cam.y = 256.0
        cam.z = 800.0
        cam.yaw = 0.0
        cam.pitch = math.radians(-85.0)
        renderer.render_frame(cam, sim=sim)
        px_down = ctx.read_pixels()
        assert not np.isnan(px_down).any()

        # 2. Sub-floor elevation, steep look-up (+85 degrees)
        cam.x = 256.0
        cam.y = 256.0
        cam.z = -400.0
        cam.yaw = 0.0
        cam.pitch = math.radians(85.0)
        renderer.render_frame(cam, sim=sim)
        px_up = ctx.read_pixels()
        assert not np.isnan(px_up).any()

        # 3. High elevation angled view
        cam.x = 100.0
        cam.y = 100.0
        cam.z = 500.0
        cam.yaw = math.pi / 4.0
        cam.pitch = math.radians(-45.0)
        renderer.render_frame(cam, sim=sim)
        px_angled = ctx.read_pixels()
        assert not np.isnan(px_angled).any()

        # Save screenshot
        shot = renderer.save_screenshot(tmp_path / "flymode_stress.png")
        assert shot.is_file()
        assert shot.stat().st_size > 0

        renderer.release()
