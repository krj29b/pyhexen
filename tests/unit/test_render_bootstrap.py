# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T4.1: GL bootstrap, main loop accumulator, and 3D camera."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyhexen.app.main_loop import (
    TIC_DURATION,
    TICRATE,
    FixedTimestepAccumulator,
    FPSCounter,
    MainLoop,
)
from pyhexen.core.angles import ANG90, ANG180, ANG270
from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.camera import (
    AUTHENTIC_PIXEL_ASPECT,
    Camera,
    bam_to_rad,
    map_to_gl_coords,
    rad_to_bam,
)
from pyhexen.render.context import RenderContext


def test_fixed_timestep_accumulator_steady_35hz() -> None:
    acc = FixedTimestepAccumulator()
    assert TICRATE == 35
    assert math.isclose(TIC_DURATION, 1.0 / 35.0)

    # Simulate 10 seconds at 60 FPS (600 frames)
    dt = 1.0 / 60.0
    total_tics = 0
    for _ in range(600):
        tics = acc.advance(dt)
        total_tics += tics
        assert 0.0 <= acc.alpha < 1.0

    # 10.0 seconds * 35 Hz = 350 tics (or 349 with float rounding right at boundary)
    assert total_tics in (349, 350)
    assert acc.total_tics == total_tics


def test_fixed_timestep_accumulator_clamps_spiral_of_death() -> None:
    acc = FixedTimestepAccumulator(max_frame_dt=0.25)
    tics = acc.advance(5.0)
    # 0.25s * 35 tics/s = 8.75 -> 8 tics maximum in one frame
    assert tics == 8
    assert 0.0 <= acc.alpha < 1.0


def test_fps_counter() -> None:
    counter = FPSCounter(window_seconds=1.0)
    updated = False
    for _ in range(60):
        if counter.record(1.0 / 60.0, 0):
            updated = True
    # Record one more tiny step if float summation was 0.9999999999999999
    if not updated:
        updated = counter.record(1e-4, 35)
    assert updated
    assert counter.fps > 55.0
    text = counter.overlay_text("PyHexen")
    assert "FPS" in text and "tics/s" in text


def test_camera_coordinate_mapping_and_bam_conversion() -> None:
    assert map_to_gl_coords(10.0, 20.0, 30.0) == (10.0, 30.0, -20.0)

    assert math.isclose(bam_to_rad(0), 0.0)
    assert math.isclose(bam_to_rad(ANG90), math.pi * 0.5, rel_tol=1e-6)
    assert math.isclose(bam_to_rad(ANG180), math.pi, rel_tol=1e-6)
    assert math.isclose(bam_to_rad(ANG270), math.pi * 1.5, rel_tol=1e-6)

    for bam in (0, ANG90, ANG180, ANG270):
        assert rad_to_bam(bam_to_rad(bam)) == bam


def test_camera_interpolation_and_shortest_arc_yaw() -> None:
    cam = Camera()
    cam.set_from_fixed(0, 0, 0, 0, snap=True)
    cam.set_from_fixed(100 * FRACUNIT, 200 * FRACUNIT, 40 * FRACUNIT, ANG90, snap=False)

    ix, iy, iz, iyaw, _ = cam.interpolated_state(0.5)
    assert math.isclose(ix, 50.0)
    assert math.isclose(iy, 100.0)
    assert math.isclose(iz, 20.0)
    assert math.isclose(iyaw, math.pi * 0.25, rel_tol=1e-5)

    # Shortest-arc interpolation across 2*pi -> 0 boundary
    cam.prev_yaw = math.radians(350.0)
    cam.yaw = math.radians(10.0)
    _, _, _, wrap_yaw, _ = cam.interpolated_state(0.5)
    assert math.isclose(wrap_yaw, 0.0, abs_tol=1e-6) or math.isclose(
        wrap_yaw, 2.0 * math.pi, abs_tol=1e-6
    )


def test_camera_view_and_projection_matrices() -> None:
    # Camera at origin facing East (yaw = 0), no authentic aspect scaling for exact check
    cam = Camera(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, authentic_aspect=False)
    cam.save_prev_state()

    view = cam.view_matrix(1.0)
    assert view.dtype == np.float32
    assert view.shape == (4, 4)

    # A point 100 units East in map coords is (100, 0, 0) -> GL (100, 0, 0)
    # In eye space, directly ahead is (0, 0, -100, 1)
    pt_east_gl = np.array([100.0, 0.0, 0.0, 1.0], dtype=np.float32)
    pt_eye = view @ pt_east_gl
    assert np.allclose(pt_eye, [0.0, 0.0, -100.0, 1.0], atol=1e-5)

    # With authentic_aspect=True, vertical eye coordinate is scaled by 1.2
    cam.authentic_aspect = True
    view_auth = cam.view_matrix(1.0)
    pt_up_gl = np.array([100.0, 10.0, 0.0, 1.0], dtype=np.float32)
    pt_eye_auth = view_auth @ pt_up_gl
    assert math.isclose(float(pt_eye_auth[1]), 10.0 * AUTHENTIC_PIXEL_ASPECT, rel_tol=1e-5)

    # Y-shearing (default true_pitch=False) vs true pitch (true_pitch=True)
    cam.pitch = math.radians(20.0)
    cam.save_prev_state()
    proj_shear = cam.projection_matrix(aspect=1.6)
    assert proj_shear[1, 2] != 0.0

    cam.true_pitch = True
    proj_true = cam.projection_matrix(aspect=1.6)
    assert proj_true[1, 2] == 0.0


def test_camera_free_fly_movement() -> None:
    cam = Camera(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0)
    cam.move_free_fly(dt=1.0, forward=1.0, right=0.0, up=0.5, pitch_delta=10.0, speed=100.0)
    assert math.isclose(cam.x, 100.0)
    assert math.isclose(cam.y, 0.0, abs_tol=1e-6)
    assert math.isclose(cam.z, 50.0)
    assert math.isclose(cam.pitch, math.radians(89.0))


def test_standalone_gl_context_and_test_triangle() -> None:
    try:
        ctx = RenderContext(width=64, height=64, standalone=True)
    except Exception as exc:
        pytest.skip(f"Headless OpenGL 3.3 context unavailable: {exc}")

    with ctx:
        pixels = ctx.render_test_triangle(read_pixels=True)
        assert pixels is not None
        assert pixels.shape == (64, 64, 4)
        # Center of screen is inside the triangle (bright non-background color)
        center_rgb = pixels[32, 32, :3]
        corner_rgb = pixels[2, 2, :3]
        assert int(center_rgb.max()) > 80
        assert int(corner_rgb.max()) < 30

        # Test resize and MainLoop step_frame
        ctx.resize(128, 96)
        assert ctx.width == 128 and ctx.height == 96

        sim_tics = 0

        def on_tic() -> None:
            nonlocal sim_tics
            sim_tics += 1

        loop = MainLoop(ctx, sim_tick=on_tic)
        tics = loop.step_frame(1.0 / 35.0)
        assert tics == 1
        assert sim_tics == 1
