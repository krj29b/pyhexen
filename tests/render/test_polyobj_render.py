# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Dedicated unit and integration tests for polyobject GPU rendering (T9.3).

Verifies local mesh construction, transformation matrices, sub-tic interpolation,
and full headless OpenGL rendering with real context (no mocks).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
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
from pyhexen.sim.polyobj import PolyDoorThinker, init_polyobjs
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _create_synthetic_poly_level(
    tag: int = 1,
    spawn: tuple[int, int] = (128, 128),
    anchor: tuple[int, int] = (-2000, -2000),
    width: int = 64,
    height: int = 64,
) -> tuple[WadStack, Level, GameSim]:
    """Create a synthetic WAD stack, Level, and GameSim containing a polyobject."""
    wb = WadBuilder("IWAD")
    wb.add_playpal(2)
    wb.add_colormap(33, include_fogmap=True)

    from fixtures.wadbuilder import PatchPlacement, TextureDef

    dummy_tex = np.full((64, 64), 160, dtype=np.uint8)
    wb.add_marker("P_START")
    wb.add_picture("WALL01", dummy_tex)
    wb.add_marker("P_END")
    wb.add_textures(
        [TextureDef(name="WALL01", width=64, height=64, patches=[PatchPlacement(patch=0)])],
        pnames=["WALL01"],
    )

    wb.add_flat("FLOOR01", fill=30)
    wb.add_flat("CEIL01", fill=50)

    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 512, 512, floor_pic="FLOOR01", ceiling_pic="CEIL01", light=180)
    mb.add_polyobject_box(
        tag=tag,
        anchor_x=anchor[0],
        anchor_y=anchor[1],
        spawn_x=spawn[0],
        spawn_y=spawn[1],
        width=width,
        height=height,
    )
    mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    init_polyobjs(level)
    sim = GameSim(level=level)
    return stack, level, sim


def test_poly_mesh_local_geometry() -> None:
    """Verify local quads, centering around start_spot, and UV coordinates."""
    stack, level, _ = _create_synthetic_poly_level(tag=10, spawn=(200, 200), width=64, height=64)
    with stack:
        po = level.polyobjs[10] if isinstance(level.polyobjs, dict) else level.polyobjs[0]
        mesh = build_polyobj_mesh(po)

        assert isinstance(mesh, PolyMesh)
        assert mesh.tag == 10
        assert len(mesh.vertices) == len(po.segs) * 4
        assert len(mesh.uvs) == len(po.segs) * 4
        assert len(mesh.texcoords) == len(mesh.uvs)

        # Confirm all vertices are in local coordinates centered on (0, 0)
        for vx, vy, _ in mesh.vertices:
            assert abs(vx) <= 32.5
            assert abs(vy) <= 32.5


def test_poly_model_matrices() -> None:
    """Verify 2D transformation matrix and OpenGL coordinate mapping."""
    # Identity transform
    m_id = compute_poly_model_matrix(0.0, 0.0, 0.0, 0.0, 0.0)
    assert m_id[0][0] == 1.0 and m_id[1][1] == 1.0 and m_id[2][2] == 1.0 and m_id[3][3] == 1.0

    # Translation
    m_tr = compute_poly_model_matrix(150.0, -75.0, 0.0, 0.0, 0.0)
    assert m_tr[3][0] == 150.0
    assert m_tr[3][1] == -75.0

    # 90 degrees CCW rotation
    m_rot = compute_poly_model_matrix(0.0, 0.0, math.pi / 2.0, 0.0, 0.0)
    assert abs(m_rot[0][0]) < 1e-4
    assert abs(m_rot[0][1] - 1.0) < 1e-4
    assert abs(m_rot[1][0] - (-1.0)) < 1e-4
    assert abs(m_rot[1][1]) < 1e-4

    # Convert to OpenGL matrix (X_gl=x, Y_gl=z, Z_gl=-y)
    m_gl = poly_model_to_gl_matrix(m_tr)
    assert m_gl.shape == (4, 4)
    # Translation in GL space: X is 150, Z is -(-75) = +75
    assert m_gl[0][3] == 150.0
    assert m_gl[2][3] == 75.0
    assert m_gl[1][3] == 0.0


def test_sub_tic_interpolation_logic() -> None:
    """Verify coordinate and angular sub-tic interpolation."""
    # Coordinate lerp
    assert interpolate_coord(10.0, 30.0, 0.0) == 10.0
    assert interpolate_coord(10.0, 30.0, 0.5) == 20.0
    assert interpolate_coord(10.0, 30.0, 1.0) == 30.0

    # Angular lerp: normal arc
    assert abs(interpolate_angle(45.0, 85.0, 0.5) - 65.0) < 1e-4

    # Angular lerp: wrapping across 0/360 boundary (350 deg to 10 deg)
    ang1 = interpolate_angle(350.0, 10.0, 0.5)
    assert abs(ang1 - 0.0) < 1e-3 or abs(ang1 - 360.0) < 1e-3

    # Reverse boundary (10 deg to 350 deg)
    ang2 = interpolate_angle(10.0, 350.0, 0.5)
    assert abs(ang2 - 0.0) < 1e-3 or abs(ang2 - 360.0) < 1e-3


def test_headless_polyobj_gpu_rendering(tmp_path: Path) -> None:
    """Full integration test: build mesh, upload GPU VBO/VAO, render frame, save screenshot."""
    stack, level, sim = _create_synthetic_poly_level(
        tag=5,
        spawn=(256, 256),
        width=64,
        height=64,
    )
    with stack:
        ctx = RenderContext(standalone=True, width=320, height=200)
        renderer = WorldRenderer(ctx, wad=stack)
        renderer.load_map(stack, level)

        # PolyMesh must be registered in renderer
        assert 5 in renderer.poly_meshes
        mesh = renderer.poly_meshes[5]
        assert mesh.vbo is not None
        assert mesh.vao is not None
        assert mesh.vertex_count > 0

        # Position camera looking toward the polyobject at (256, 256)
        camera = Camera()
        camera.x = 256.0
        camera.y = 100.0
        camera.z = 64.0
        camera.yaw = math.pi / 2.0  # Face North toward (256, 256)

        # Step simulation to initiate polyobject movement
        po = level.polyobjs[5] if isinstance(level.polyobjs, dict) else level.polyobjs[0]
        door = PolyDoorThinker(
            poly=po,
            door_type=0,  # Slide door
            speed=2 * FRACUNIT,
            dist=64 * FRACUNIT,
            total_dist=64 * FRACUNIT,
            direction=0,
            x_speed=2 * FRACUNIT,
            y_speed=0,
            wait_tics=35,
        )
        sim.thinkers.add(door)
        for _ in range(5):
            sim.tick()

        # Render frame with sub-tic interpolation
        renderer.render_frame(camera, sim=sim, alpha=0.5)

        # Save and verify screenshot image output
        shot = renderer.save_screenshot(tmp_path / "poly_render.png")
        assert shot.is_file()
        assert shot.stat().st_size > 0

        # Clean release
        renderer.release()
        assert mesh.vao is None
        assert mesh.vbo is None
