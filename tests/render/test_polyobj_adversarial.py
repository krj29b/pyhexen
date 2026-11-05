# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Empirical adversarial stress testing for Milestone 3 (T9.3 Polyobject Dynamic GPU Rendering).

Covers:
1. Matrix & Math Stress:
   - Extreme values (zero translation, arbitrary angles, negative coords, large offsets)
   - Pivot transformation invariants
   - Isometry and perpendicularity preservation under GL +Y rotation
   - 3x3 proper rotation determinant (+1.0)
2. Interpolation Edge Cases:
   - Sub-tic coordinate interpolation with alpha bounds ([0, 1]) and extrapolation (< 0, > 1)
   - Circular angle boundary wrap: 359 -> 1, 1 -> 359, 180 -> -180, 179 -> 181
   - Smoothness, continuity, and monotonicity of interpolated trajectory
3. Shaders & Uniform State:
   - Restoration of u_model to 4x4 identity matrix after drawing polyobjects
   - Restoration of u_light_override to -1.0
   - Multi-polyobject isolation and uniform state hygiene
"""

from __future__ import annotations

import math
from typing import Any

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
from pyhexen.sim.polyobj import PolyDoorThinker, RotatePolyThinker, init_polyobjs
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _build_test_wad_and_level(
    poly_configs: list[dict[str, Any]] | None = None,
) -> tuple[WadStack, Level, GameSim]:
    """Helper to construct a synthetic level with one or more polyobjects."""
    if poly_configs is None:
        poly_configs = [{"tag": 1, "spawn": (128, 128), "anchor": (-2000, -2000), "w": 64, "h": 64}]

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

    for i, cfg in enumerate(poly_configs):
        rx = cfg.get("room_x", i * 1000)
        ry = cfg.get("room_y", 0)
        mb.add_room(
            rx,
            ry,
            cfg.get("room_w", 512),
            cfg.get("room_h", 512),
            floor_pic="FLOOR01",
            ceiling_pic="CEIL01",
            light=cfg.get("light", 180),
        )
        mb.add_polyobject_box(
            tag=cfg["tag"],
            anchor_x=cfg["anchor"][0],
            anchor_y=cfg["anchor"][1],
            spawn_x=cfg["spawn"][0],
            spawn_y=cfg["spawn"][1],
            width=cfg.get("w", 64),
            height=cfg.get("h", 64),
        )

    mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
    wb.add_map(mb)

    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    init_polyobjs(level)
    sim = GameSim(level=level)
    return stack, level, sim


# ===========================================================================
# 1. Matrix & Math Stress Tests
# ===========================================================================


class TestMatrixAndMathStress:
    """Stress-tests for 2D poly model matrix and 3D OpenGL world matrix."""

    def test_matrix_zero_translation(self) -> None:
        """Zero translation and zero angle must yield exact 4x4 identity."""
        m_2d = compute_poly_model_matrix(0.0, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(m_2d, np.eye(4, dtype=np.float32), atol=1e-6)

        m_gl = poly_model_to_gl_matrix(m_2d)
        np.testing.assert_allclose(m_gl, np.eye(4, dtype=np.float32), atol=1e-6)

    def test_matrix_negative_coordinates(self) -> None:
        """Negative world coordinates properly map to OpenGL (X_gl = x, Z_gl = -y)."""
        m_2d = compute_poly_model_matrix(-32768.0, -16384.0, 0.0, 0.0, 0.0)
        assert m_2d[3, 0] == -32768.0
        assert m_2d[3, 1] == -16384.0

        m_gl = poly_model_to_gl_matrix(m_2d)
        assert m_gl[0, 3] == -32768.0
        assert m_gl[2, 3] == 16384.0  # -(-16384) = +16384
        assert m_gl[1, 3] == 0.0
        assert m_gl[3, 3] == 1.0

    def test_matrix_large_offsets(self) -> None:
        """Large coordinates (10^7) maintain numerical validity without overflow."""
        large_x = 10_000_000.0
        large_y = -8_000_000.0
        m_2d = compute_poly_model_matrix(large_x, large_y, math.pi / 3.0)
        assert not np.isnan(m_2d).any()
        assert not np.isinf(m_2d).any()

        m_gl = poly_model_to_gl_matrix(m_2d)
        assert not np.isnan(m_gl).any()
        assert not np.isinf(m_gl).any()
        assert m_gl[0, 3] == large_x
        assert m_gl[2, 3] == -large_y

    def test_matrix_arbitrary_angles(self) -> None:
        """Test rotation matrices with extreme, negative, and multi-turn angles."""
        test_angles = [
            0.0,
            math.pi / 6,
            math.pi / 4,
            math.pi / 2,
            math.pi,
            3 * math.pi / 2,
            2 * math.pi,
            -math.pi / 3,
            -2 * math.pi,
            100 * math.pi,
            -73.45678,
            1e-8,
            -1e-8,
        ]
        for ang in test_angles:
            m_2d = compute_poly_model_matrix(50.0, -30.0, ang)
            cos_val = m_2d[0, 0]
            sin_val = m_2d[0, 1]
            # Trigonometric identity cos^2 + sin^2 == 1
            assert abs((cos_val * cos_val + sin_val * sin_val) - 1.0) < 1e-5

            m_gl = poly_model_to_gl_matrix(m_2d)
            # Check GL rotation 3x3 submatrix orthogonality: R * R^T = I
            r = m_gl[:3, :3]
            ident = np.dot(r, r.T)
            np.testing.assert_allclose(ident, np.eye(3, dtype=np.float32), atol=1e-5)

    def test_matrix_pivot_offset_invariant(self) -> None:
        """Rotating around pivot (px, py) must transform (px, py) exactly to (x, y)."""
        pivot_x = 42.5
        pivot_y = -18.75
        target_x = 250.0
        target_y = 350.0

        for ang_deg in [0, 30, 45, 90, 180, 270, 315, -60, 720]:
            ang_rad = math.radians(ang_deg)
            m_2d = compute_poly_model_matrix(
                target_x, target_y, ang_rad, pivot_x=pivot_x, pivot_y=pivot_y
            )

            # Transform pivot point: v_pivot = [px, py, 0, 1]
            v_pivot = np.array([pivot_x, pivot_y, 0.0, 1.0], dtype=np.float32)
            # In row-major convention: v * M
            v_transformed = np.dot(v_pivot, m_2d)

            assert abs(v_transformed[0] - target_x) < 1e-4
            assert abs(v_transformed[1] - target_y) < 1e-4

    def test_gl_matrix_isometry_distance_preservation(self) -> None:
        """Pure rotation around GL +Y preserves Euclidean distance between vertices."""
        rng = np.random.default_rng(seed=1337)
        # Generate 50 random vertex pairs
        v1_list = rng.uniform(-1000.0, 1000.0, size=(50, 3)).astype(np.float32)
        v2_list = rng.uniform(-1000.0, 1000.0, size=(50, 3)).astype(np.float32)

        angles = [0.1, math.pi / 4, math.pi / 2, 2.34, -1.8, math.pi]

        for ang in angles:
            m_2d = compute_poly_model_matrix(120.0, -80.0, ang)
            m_gl = poly_model_to_gl_matrix(m_2d)

            for v1, v2 in zip(v1_list, v2_list, strict=True):
                orig_dist = float(np.linalg.norm(v1 - v2))

                # Apply GL transform: M_gl * [vx, vy, vz, 1]^T
                p1 = np.dot(m_gl, np.array([v1[0], v1[1], v1[2], 1.0], dtype=np.float32))[:3]
                p2 = np.dot(m_gl, np.array([v2[0], v2[1], v2[2], 1.0], dtype=np.float32))[:3]

                new_dist = float(np.linalg.norm(p1 - p2))
                assert abs(orig_dist - new_dist) < 1e-3, (
                    f"Distance violated at ang={ang}: {orig_dist} vs {new_dist}"
                )

    def test_gl_matrix_perpendicularity_preservation(self) -> None:
        """Rotation preserves orthogonality of perpendicular vectors."""
        # Orthogonal unit vectors in local OpenGL space
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # East
        w = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # North in GL (Z_gl = -Y_map)
        assert np.dot(u, w) == 0.0

        for ang in [0.0, 0.5, 1.2, math.pi / 2, math.pi, -math.pi / 4]:
            m_2d = compute_poly_model_matrix(0.0, 0.0, ang)
            m_gl = poly_model_to_gl_matrix(m_2d)
            rot = m_gl[:3, :3]

            u_rot = np.dot(rot, u)
            w_rot = np.dot(rot, w)

            # Dot product must remain 0
            dot_val = float(np.dot(u_rot, w_rot))
            assert abs(dot_val) < 1e-6, f"Perpendicularity violated: dot={dot_val}"

            # Lengths must remain 1.0
            assert abs(np.linalg.norm(u_rot) - 1.0) < 1e-6
            assert abs(np.linalg.norm(w_rot) - 1.0) < 1e-6

    def test_gl_matrix_proper_rotation_determinant(self) -> None:
        """The 3x3 rotation part of M_gl must have determinant exactly +1.0
        (no reflection or inversion).
        """
        for ang in np.linspace(-4 * math.pi, 4 * math.pi, 30):
            m_2d = compute_poly_model_matrix(100.0, 200.0, ang)
            m_gl = poly_model_to_gl_matrix(m_2d)
            det = float(np.linalg.det(m_gl[:3, :3]))
            assert abs(det - 1.0) < 1e-5, f"Determinant {det} != 1.0 at angle {ang}"

    def test_gl_matrix_preserves_gl_y_axis(self) -> None:
        """Rotation around GL +Y axis must leave the Y coordinate untouched."""
        v = np.array([0.0, 150.0, 0.0, 1.0], dtype=np.float32)
        for ang in [0.3, 1.0, math.pi / 2, math.pi]:
            m_2d = compute_poly_model_matrix(50.0, 50.0, ang)
            m_gl = poly_model_to_gl_matrix(m_2d)
            v_out = np.dot(m_gl, v)
            assert abs(v_out[1] - 150.0) < 1e-5


# ===========================================================================
# 2. Interpolation Edge Cases
# ===========================================================================


class TestInterpolationEdgeCases:
    """Adversarial testing of coordinate and circular angle interpolation."""

    def test_interpolate_coord_boundary_and_extrapolation(self) -> None:
        """Test alpha at 0.0, 1.0, 0.5, negative, and > 1.0."""
        prev = 100.0
        curr = 200.0

        # Boundary values
        assert interpolate_coord(prev, curr, 0.0) == 100.0
        assert interpolate_coord(prev, curr, 1.0) == 200.0
        assert interpolate_coord(prev, curr, 0.5) == 150.0

        # Sub-zero extrapolation
        assert interpolate_coord(prev, curr, -0.5) == 50.0
        assert interpolate_coord(prev, curr, -1.0) == 0.0

        # Greater than 1.0 extrapolation
        assert interpolate_coord(prev, curr, 1.5) == 250.0
        assert interpolate_coord(prev, curr, 2.0) == 300.0

        # Identical endpoints
        assert interpolate_coord(75.0, 75.0, 0.5) == 75.0
        assert interpolate_coord(75.0, 75.0, -2.5) == 75.0
        assert interpolate_coord(75.0, 75.0, 3.5) == 75.0

    def test_interpolate_angle_circular_wrap_forward(self) -> None:
        """359 deg -> 1 deg must take +2 deg arc, not -358 deg backward arc."""
        prev = 359.0
        curr = 1.0

        # Start and end
        assert abs(interpolate_angle(prev, curr, 0.0) - 359.0) < 1e-5
        assert abs(interpolate_angle(prev, curr, 1.0) - 1.0) < 1e-5

        # Midpoint: 359 + 2 * 0.5 = 360 = 0 deg
        mid = interpolate_angle(prev, curr, 0.5)
        assert abs(mid - 0.0) < 1e-4 or abs(mid - 360.0) < 1e-4

        # Intermediate points
        q1 = interpolate_angle(prev, curr, 0.25)
        assert abs(q1 - 359.5) < 1e-4

        q3 = interpolate_angle(prev, curr, 0.75)
        assert abs(q3 - 0.5) < 1e-4

        # Verify it NEVER sweeps backwards through 180 degrees
        for alpha in np.linspace(0.0, 1.0, 50):
            val = interpolate_angle(prev, curr, float(alpha))
            # Must be either in [359, 360] or [0, 1]
            in_range = (359.0 <= val <= 360.0) or (0.0 <= val <= 1.0)
            assert in_range, f"Angle {val} leaked outside shortest arc at alpha={alpha}"

    def test_interpolate_angle_circular_wrap_backward(self) -> None:
        """1 deg -> 359 deg must take -2 deg arc."""
        prev = 1.0
        curr = 359.0

        assert abs(interpolate_angle(prev, curr, 0.0) - 1.0) < 1e-5
        assert abs(interpolate_angle(prev, curr, 1.0) - 359.0) < 1e-5

        mid = interpolate_angle(prev, curr, 0.5)
        assert abs(mid - 0.0) < 1e-4 or abs(mid - 360.0) < 1e-4

        q1 = interpolate_angle(prev, curr, 0.25)
        assert abs(q1 - 0.5) < 1e-4

        q3 = interpolate_angle(prev, curr, 0.75)
        assert abs(q3 - 359.5) < 1e-4

        for alpha in np.linspace(0.0, 1.0, 50):
            val = interpolate_angle(prev, curr, float(alpha))
            in_range = (0.0 <= val <= 1.0) or (359.0 <= val <= 360.0)
            assert in_range, f"Angle {val} leaked outside shortest arc at alpha={alpha}"

    def test_interpolate_angle_antipodal_and_congruent(self) -> None:
        """Test 180 -> -180 (congruent) and near-antipodal 179 -> 181."""
        # 180 and -180 are congruent mod 360; shortest arc difference is 0
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = interpolate_angle(180.0, -180.0, alpha)
            assert abs(val - 180.0) < 1e-4

        # 179 -> 181 (shortest arc +2 deg, passing through 180 at alpha = 0.5)
        assert abs(interpolate_angle(179.0, 181.0, 0.5) - 180.0) < 1e-4

        # 181 -> 179 (shortest arc -2 deg, passing through 180 at alpha = 0.5)
        assert abs(interpolate_angle(181.0, 179.0, 0.5) - 180.0) < 1e-4

    def test_sub_tic_interpolation_smooth_monotonic_trajectories(self) -> None:
        """Verify sub-tic angular interpolation produces continuous C0 trajectory without jumps."""
        pairs = [
            (359.0, 1.0),
            (1.0, 359.0),
            (350.0, 20.0),
            (45.0, 90.0),
            (179.0, 181.0),
            (270.0, 30.0),
        ]

        n_samples = 200
        alphas = np.linspace(0.0, 1.0, n_samples)

        for p_deg, c_deg in pairs:
            angles = [interpolate_angle(p_deg, c_deg, float(a)) for a in alphas]
            rads = [math.radians(ang) for ang in angles]
            cos_vals = [math.cos(r) for r in rads]
            sin_vals = [math.sin(r) for r in rads]

            # In Cartesian space (cos, sin), there must be no discontinuities
            for i in range(len(cos_vals) - 1):
                d_cos = abs(cos_vals[i + 1] - cos_vals[i])
                d_sin = abs(sin_vals[i + 1] - sin_vals[i])
                # Max step across 200 samples for shortest arc (<= 180 deg)
                # is <= 2 * pi / 200 ~ 0.032
                assert d_cos < 0.05, f"Discontinuity in cos: {d_cos} between steps {i} and {i + 1}"
                assert d_sin < 0.05, f"Discontinuity in sin: {d_sin} between steps {i} and {i + 1}"


# ===========================================================================
# 3. Shaders & Uniform State Hygiene
# ===========================================================================


class TestShaderUniformStateHygiene:
    """Verifies that u_model and u_light_override are strictly restored after polyobject
    rendering.
    """

    def test_uniforms_restored_after_render_frame(self) -> None:
        """WorldRenderer.render_frame restores u_model to 4x4 I and u_light_override to -1.0."""
        stack, level, sim = _build_test_wad_and_level()
        with stack:
            ctx = RenderContext(standalone=True, width=320, height=200)
            renderer = WorldRenderer(ctx, wad=stack)
            renderer.load_map(stack, level)

            camera = Camera()
            camera.x = 128.0
            camera.y = 64.0
            camera.z = 64.0
            camera.yaw = math.pi / 2.0

            # Render frame with alpha
            renderer.render_frame(camera, sim=sim, alpha=0.5)

            # 1. Check u_model in WorldShaderProgram is identity
            raw_model = renderer.world_shader.program["u_model"].read()
            m_gpu = np.frombuffer(raw_model, dtype=np.float32).reshape(4, 4)
            np.testing.assert_allclose(m_gpu, np.eye(4, dtype=np.float32), atol=1e-5)

            # 2. Check u_light_override is -1.0
            light_val = renderer.world_shader.program["u_light_override"].value
            assert light_val == -1.0, f"Expected -1.0, got {light_val}"

            renderer.release()

    def test_uniforms_restored_after_direct_render_polyobjs(self) -> None:
        """Calling render_polyobjs directly restores uniforms upon completion."""
        stack, level, _ = _build_test_wad_and_level()
        with stack:
            ctx = RenderContext(standalone=True, width=320, height=200)
            renderer = WorldRenderer(ctx, wad=stack)
            renderer.load_map(stack, level)

            # Call render_polyobjs
            renderer.render_polyobjs(alpha=0.33)

            # Check u_model restored to identity
            raw_model = renderer.world_shader.program["u_model"].read()
            m_gpu = np.frombuffer(raw_model, dtype=np.float32).reshape(4, 4)
            np.testing.assert_allclose(m_gpu, np.eye(4, dtype=np.float32), atol=1e-5)

            # Check u_light_override restored to -1.0
            assert renderer.world_shader.program["u_light_override"].value == -1.0

            renderer.release()

    def test_uniforms_safe_when_no_polyobjects(self) -> None:
        """In a scene without polyobjects, render_polyobjs leaves uniforms untouched."""
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33, include_fogmap=True)
        mb = MapBuilder("MAP01")
        mb.add_room(0, 0, 512, 512)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        stack = WadStack(wb.to_bytes())

        with stack:
            level = load_level(stack, "MAP01")
            ctx = RenderContext(standalone=True, width=320, height=200)
            renderer = WorldRenderer(ctx, wad=stack)
            renderer.load_map(stack, level)

            assert len(renderer.poly_meshes) == 0

            renderer.render_polyobjs()

            raw_model = renderer.world_shader.program["u_model"].read()
            m_gpu = np.frombuffer(raw_model, dtype=np.float32).reshape(4, 4)
            np.testing.assert_allclose(m_gpu, np.eye(4, dtype=np.float32), atol=1e-5)
            assert renderer.world_shader.program["u_light_override"].value == -1.0

            renderer.release()

    def test_multi_polyobject_isolation_and_restoration(self) -> None:
        """Multiple polyobjects with distinct positions and rotations all clean up."""
        configs = [
            {
                "tag": 10,
                "spawn": (100, 100),
                "anchor": (-1000, -1000),
                "w": 32,
                "h": 32,
                "room_x": 0,
                "room_y": 0,
            },
            {
                "tag": 20,
                "spawn": (1100, 100),
                "anchor": (-2000, -2000),
                "w": 48,
                "h": 48,
                "room_x": 1000,
                "room_y": 0,
            },
            {
                "tag": 30,
                "spawn": (2100, 100),
                "anchor": (-3000, -3000),
                "w": 64,
                "h": 64,
                "room_x": 2000,
                "room_y": 0,
            },
        ]
        stack, level, sim = _build_test_wad_and_level(configs)
        with stack:
            ctx = RenderContext(standalone=True, width=320, height=200)
            renderer = WorldRenderer(ctx, wad=stack)
            renderer.load_map(stack, level)

            assert len(renderer.poly_meshes) == 3

            # Add distinct motions
            po10 = level.polyobjs[10]
            po20 = level.polyobjs[20]
            sim.thinkers.add(RotatePolyThinker(poly=po10, speed=1000000, dist=10000000))
            sim.thinkers.add(
                PolyDoorThinker(
                    poly=po20,
                    door_type=0,
                    speed=2 * FRACUNIT,
                    dist=32 * FRACUNIT,
                    total_dist=32 * FRACUNIT,
                    direction=0,
                    x_speed=2 * FRACUNIT,
                    y_speed=0,
                    wait_tics=10,
                )
            )

            for _ in range(5):
                sim.tick()

            camera = Camera()
            camera.x = 256.0
            camera.y = 256.0
            camera.z = 64.0

            renderer.render_frame(camera, sim=sim, alpha=0.75)

            # Uniforms must be clean after rendering 3 different polyobjects
            raw_model = renderer.world_shader.program["u_model"].read()
            m_gpu = np.frombuffer(raw_model, dtype=np.float32).reshape(4, 4)
            np.testing.assert_allclose(m_gpu, np.eye(4, dtype=np.float32), atol=1e-5)
            assert renderer.world_shader.program["u_light_override"].value == -1.0

            renderer.release()


# ===========================================================================
# 4. Extreme Polyobject Mesh Corner Cases
# ===========================================================================


class TestExtremePolyMeshCornerCases:
    """Stress tests for degenerate geometry, zero length segs, and buffer management."""

    def test_poly_mesh_degenerate_empty_segs(self) -> None:
        """Polyobject with empty segs list builds clean zero-vertex PolyMesh."""
        stack, level, _ = _build_test_wad_and_level()
        with stack:
            po = level.polyobjs[1]
            po.segs = []
            mesh = build_polyobj_mesh(po)
            assert isinstance(mesh, PolyMesh)
            assert mesh.vertex_count == 0
            assert len(mesh.vertices) == 0
            assert len(mesh.batches) == 0
            mesh.release()

    def test_poly_mesh_zero_length_seg_handled(self) -> None:
        """Seg where v1 == v2 does not cause NaN or ZeroDivisionError in UV or mesh math."""
        stack, level, _ = _build_test_wad_and_level()
        with stack:
            po = level.polyobjs[1]
            # Force seg vertices to identical coordinates
            po.segs[0].v2 = po.segs[0].v1
            mesh = build_polyobj_mesh(po)
            assert not np.isnan(mesh.raw_vertices).any()
            mesh.release()


# ===========================================================================
# 5. Integration: Lighting, Chunk Exclusion, and Render Ordering
# ===========================================================================


class TestIntegrationAdversarialGuards:
    """Stress tests verifying dynamic lighting transitions, chunk exclusion, and render order."""

    def test_dynamic_sector_lighting_interpolation_across_boundary(self) -> None:
        """Polyobject interpolating across sector boundary samples the destination
        sector's light.
        """
        wb = WadBuilder("IWAD")
        wb.add_playpal(2).add_colormap(33, include_fogmap=True)
        mb = MapBuilder("MAP01")
        # Room 1: light 80; Room 2: light 240
        mb.add_two_sector_rooms(light1=80, light2=240)
        # Polyobject spawning at (128, 128) in sector 0
        mb.add_polyobject_box(tag=1, anchor_x=-2000, anchor_y=-2000, spawn_x=128, spawn_y=128)
        mb.add_thing(tid=1, x=64, y=64, z=0, angle=0, type=1)
        wb.add_map(mb)
        stack = WadStack(wb.to_bytes())

        with stack:
            level = load_level(stack, "MAP01")
            ctx = RenderContext(standalone=True, width=320, height=200)
            renderer = WorldRenderer(ctx, wad=stack)
            renderer.load_map(stack, level)

            po = level.polyobjs[1]
            po.prev_x = 128 * FRACUNIT
            po.prev_y = 128 * FRACUNIT
            # Simulate movement to sector 1 at (384, 128)
            po.start_spot.x = 384 * FRACUNIT
            po.start_spot.y = 128 * FRACUNIT

            observed_lights: list[float] = []
            orig_set_light = renderer.world_shader.set_light_override

            def spy_light(lvl: float = -1.0) -> None:
                if lvl >= 0.0:
                    observed_lights.append(lvl)
                orig_set_light(lvl)

            renderer.world_shader.set_light_override = spy_light  # type: ignore[assignment]

            # At alpha = 0.0 (in sector 0)
            renderer.render_polyobjs(alpha=0.0)
            assert len(observed_lights) == 1
            assert observed_lights[0] == 80.0

            observed_lights.clear()
            # At alpha = 1.0 (in sector 1)
            renderer.render_polyobjs(alpha=1.0)
            assert len(observed_lights) == 1
            assert observed_lights[0] == 240.0

            renderer.release()

    def test_static_chunk_manager_strictly_excludes_poly_linedefs(self) -> None:
        """Static sector chunks must contain zero geometry from polyobject linedefs."""
        stack, level, _ = _build_test_wad_and_level()
        with stack:
            ctx = RenderContext(standalone=True, width=320, height=200)
            renderer = WorldRenderer(ctx, wad=stack)
            renderer.load_map(stack, level)

            from pyhexen.render.mesh_walls import build_sector_walls

            # Find all poly linedef indices
            poly_line_indices = {
                i for i, ld in enumerate(level.lines) if getattr(ld, "is_polyobj", False)
            }
            assert len(poly_line_indices) > 0

            # Directly verify build_sector_walls excludes poly lines
            for sec_idx in range(len(level.sectors)):
                quads = build_sector_walls(level, sec_idx, textures=None)
                # None of the quads should originate from a poly line
                for q in quads:
                    assert q.line_index not in poly_line_indices

            renderer.release()

    def test_render_order_polyobjs_before_sprites(self) -> None:
        """WorldRenderer must execute render_polyobjs strictly before sprite rendering."""
        stack, level, sim = _build_test_wad_and_level()
        with stack:
            ctx = RenderContext(standalone=True, width=320, height=200)
            renderer = WorldRenderer(ctx, wad=stack)
            renderer.load_map(stack, level)

            call_order: list[str] = []

            orig_poly_render = renderer.render_polyobjs

            def spy_poly_render(*args: Any, **kwargs: Any) -> None:
                call_order.append("polyobjs")
                orig_poly_render(*args, **kwargs)

            renderer.render_polyobjs = spy_poly_render  # type: ignore[assignment]

            if renderer.sprite_renderer is not None:
                orig_sprite_opaque = renderer.sprite_renderer.render_opaque

                def spy_sprite_opaque(*args: Any, **kwargs: Any) -> None:
                    call_order.append("sprites_opaque")
                    orig_sprite_opaque(*args, **kwargs)

                renderer.sprite_renderer.render_opaque = spy_sprite_opaque  # type: ignore[assignment]

            camera = Camera()
            camera.x = 64.0
            camera.y = 64.0
            camera.z = 64.0

            renderer.render_frame(camera, sim=sim, alpha=1.0)

            assert "polyobjs" in call_order
            if "sprites_opaque" in call_order:
                poly_idx = call_order.index("polyobjs")
                sprite_idx = call_order.index("sprites_opaque")
                assert poly_idx < sprite_idx, f"Pass order inverted: {call_order}"

            renderer.release()
