# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Real-IWAD empirical verification tests for polyobject GPU rendering (T9.3).

Verifies MAP01 and MAP02 polyobject GPU meshes, courtyard double door heights (-64 to 64),
textures, UVs, and real-time swinging door rendering animation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pyhexen.render.camera import Camera
from pyhexen.render.context import RenderContext
from pyhexen.render.polyobj import PolyMesh, build_polyobj_mesh
from pyhexen.render.renderer import WorldRenderer
from pyhexen.sim.game_sim import GameSim
from pyhexen.wad.textures import TextureRegistry
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import SPAC_USE, load_level


@pytest.mark.wad
def test_iwad_map01_polyobj_meshes(iwad_path: Path) -> None:
    """Validate 12 MAP01 polyobject local meshes have non-zero vertices and valid textures/UVs."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")
        tex_reg = TextureRegistry(wad)

        assert len(level.polyobjs) == 12

        for tag in range(4, 16):
            assert tag in level.polyobjs
            po = level.polyobjs[tag]
            mesh = build_polyobj_mesh(po, textures=tex_reg)

            assert isinstance(mesh, PolyMesh)
            assert mesh.tag == tag
            assert mesh.vertex_count > 0
            assert len(mesh.vertices) == len(po.segs) * 4
            assert len(mesh.uvs) == len(po.segs) * 4
            assert len(mesh.batches) > 0
            assert mesh.raw_vertices.shape == (mesh.vertex_count, 8)

            # Check for NaN or Inf in raw vertices
            assert not np.isnan(mesh.raw_vertices).any()
            assert not np.isinf(mesh.raw_vertices).any()

            # Check all batches have valid texture names
            for batch in mesh.batches:
                assert batch.tex_name != ""
                assert batch.vertex_count > 0


@pytest.mark.wad
def test_iwad_map01_courtyard_doors_height_and_textures(iwad_path: Path) -> None:
    """Verify MAP01 courtyard double doors (tags 4 and 5) match courtyard heights (-64 to 64)."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")
        tex_reg = TextureRegistry(wad)

        # Tags 4 and 5 are the courtyard double swinging doors
        for tag in (4, 5):
            po = level.polyobjs[tag]
            mesh = build_polyobj_mesh(po, textures=tex_reg)

            # Courtyard floor height is -64, ceiling height is 64
            assert mesh.z_bottom == -64.0
            assert mesh.z_top == 64.0

            batch_textures = {b.tex_name for b in mesh.batches}
            # In vanilla Hexen MAP01, courtyard doors use D_END2 and D_WINNOW
            assert "D_END2" in batch_textures
            assert "D_WINNOW" in batch_textures

            # All UV coordinates must be finite and within expected bounds
            for u, v in mesh.uvs:
                assert not math.isnan(u) and not math.isnan(v)
                assert not math.isinf(u) and not math.isinf(v)


@pytest.mark.wad
def test_iwad_map01_double_swinging_doors_render_animation(iwad_path: Path) -> None:
    """Empirically test rendering the opening double swinging doors in MAP01 with headless GL."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")
        sim = GameSim(level=level)

        ctx = RenderContext(standalone=True, width=320, height=200)
        renderer = WorldRenderer(ctx, wad=wad)
        renderer.load_map(wad, level)

        po4 = level.polyobjs[4]
        po5 = level.polyobjs[5]

        # Line 558 activates script 2 to open the courtyard doors
        line_558 = level.lines[558]
        activated = sim.on_line_activated(line_558, None, SPAC_USE)
        assert activated is True

        sim.tick()
        assert po4.busy is True
        assert po5.busy is True

        # Position camera in front of courtyard doors
        cam = Camera()
        cam.x = float(po4.start_spot.x) / 65536.0
        cam.y = float(po4.start_spot.y) / 65536.0 - 180.0
        cam.z = 0.0
        cam.yaw = math.pi / 2.0
        cam.pitch = 0.0

        # Step through 15 tics of opening motion with sub-tic interpolation
        for _ in range(15):
            sim.tick()
            for alpha in (0.0, 0.5, 1.0):
                renderer.render_frame(cam, sim=sim, alpha=alpha)
                pixels = ctx.read_pixels()
                assert pixels.shape == (200, 320, 4)
                assert not np.isnan(pixels).any()

        # Confirm doors rotated
        assert po4.angle != 0
        assert po5.angle != 0

        renderer.release()


@pytest.mark.wad
def test_iwad_map02_polyobjects_render_verification(iwad_path: Path) -> None:
    """Verify MAP02 polyobjects (including tags 1 and 2) generate valid GPU meshes."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP02")
        tex_reg = TextureRegistry(wad)

        # In MAP02, tags 1, 2, 3... exist
        assert 1 in level.polyobjs
        assert 2 in level.polyobjs

        for tag in (1, 2, 3):
            po = level.polyobjs[tag]
            mesh = build_polyobj_mesh(po, textures=tex_reg)
            assert mesh.tag == tag
            assert mesh.vertex_count > 0
            assert len(mesh.batches) > 0
            assert mesh.z_top > mesh.z_bottom
            assert not np.isnan(mesh.raw_vertices).any()
