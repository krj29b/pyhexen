# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Fixed 35 Hz simulation timestep accumulator, FPS counter, and main loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import pygame

from pyhexen.render.context import RenderContext

TICRATE: int = 35
TIC_DURATION: float = 1.0 / float(TICRATE)
MAX_FRAME_DT: float = 0.25


@dataclass(slots=True)
class FixedTimestepAccumulator:
    """Accumulates frame delta time and yields fixed 35 Hz tics plus interpolation alpha."""

    tic_duration: float = TIC_DURATION
    max_frame_dt: float = MAX_FRAME_DT
    accumulator: float = 0.0
    total_tics: int = 0

    @property
    def alpha(self) -> float:
        """Interpolation factor in [0.0, 1.0) between previous and current tic."""
        if self.tic_duration <= 0.0:
            return 0.0
        return max(0.0, min(1.0 - 1e-9, self.accumulator / self.tic_duration))

    def advance(self, dt: float) -> int:
        """Add `dt` seconds to the accumulator and return the number of tics to execute."""
        clamped_dt = max(0.0, min(self.max_frame_dt, float(dt)))
        self.accumulator += clamped_dt
        tics = int(self.accumulator // self.tic_duration)
        if tics > 0:
            self.accumulator -= tics * self.tic_duration
            self.total_tics += tics
        return tics

    def reset(self) -> None:
        """Reset accumulated time and tic count."""
        self.accumulator = 0.0
        self.total_tics = 0


@dataclass(slots=True)
class FPSCounter:
    """Rolling-window FPS and TPS (tics per second) counter."""

    window_seconds: float = 1.0
    fps: float = 0.0
    tps: float = 0.0
    _frames: int = 0
    _tics: int = 0
    _elapsed: float = 0.0

    def record(self, dt: float, tics: int) -> bool:
        """Record a rendered frame and elapsed tics. Returns True when metrics update."""
        self._frames += 1
        self._tics += tics
        self._elapsed += max(0.0, dt)
        if self._elapsed >= self.window_seconds:
            self.fps = self._frames / self._elapsed
            self.tps = self._tics / self._elapsed
            self._frames = 0
            self._tics = 0
            self._elapsed = 0.0
            return True
        return False

    def overlay_text(self, base_title: str = "PyHexen") -> str:
        """Format FPS and TPS for display overlay / window caption."""
        return f"{base_title} | {self.fps:.1f} FPS | {self.tps:.1f} tics/s"


class MainLoop:
    """Coordinates the 35 Hz fixed-timestep simulation loop and variable-rate rendering."""

    def __init__(
        self,
        render_ctx: RenderContext,
        *,
        sim_tick: Callable[[], None] | None = None,
        render_frame: Callable[[float], None] | None = None,
        event_handler: Callable[[pygame.event.Event], None] | None = None,
    ) -> None:
        self.render_ctx = render_ctx
        self.sim_tick = sim_tick
        self.render_frame = render_frame
        self.event_handler = event_handler
        self.timestep = FixedTimestepAccumulator()
        self.fps_counter = FPSCounter()
        self.running = False

    def step_frame(self, dt: float) -> int:
        """Advance one frame by `dt` seconds, running 0+ sim tics and 1 render pass."""
        tics = self.timestep.advance(dt)
        if self.sim_tick is not None:
            for _ in range(tics):
                self.sim_tick()

        alpha = self.timestep.alpha
        if self.render_frame is not None:
            self.render_frame(alpha)
        else:
            self.render_ctx.render_test_triangle()

        self.render_ctx.swap_buffers()
        if self.fps_counter.record(dt, tics):
            self.render_ctx.set_caption(self.fps_counter.overlay_text(self.render_ctx.title))
        return tics

    def run(self, *, max_frames: int | None = None) -> None:
        """Run the main loop until quit or `max_frames` is reached."""
        self.running = True
        last_time = time.perf_counter()
        frames = 0

        while self.running:
            if not self.render_ctx.standalone:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.render_ctx.resize(event.w, event.h)
                    if self.event_handler is not None:
                        self.event_handler(event)

            if not self.running:
                break

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            self.step_frame(dt)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                self.running = False
