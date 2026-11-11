# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""OpenGL 3.3 Core window and ModernGL context management."""

from __future__ import annotations

from typing import Self

import moderngl
import numpy as np
import pygame

_TEST_TRIANGLE_VERT = """
#version 330 core
in vec2 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_color = in_color;
}
"""

_TEST_TRIANGLE_FRAG = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


class RenderContext:
    """Manages the pygame-ce OpenGL 3.3 window (or standalone headless context) and ModernGL."""

    def __init__(
        self,
        width: int = 960,
        height: int = 600,
        *,
        title: str = "PyHexen",
        vsync: bool = True,
        fullscreen: bool = False,
        standalone: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.vsync = vsync
        self.fullscreen = fullscreen
        self.standalone = standalone
        self._fbo: moderngl.Framebuffer | None = None

        if standalone:
            self.ctx = moderngl.create_standalone_context(require=330)
            color_tex = self.ctx.texture((width, height), 4)
            depth_rb = self.ctx.depth_renderbuffer((width, height))
            self._fbo = self.ctx.framebuffer(
                color_attachments=[color_tex],
                depth_attachment=depth_rb,
            )
            self._fbo.use()
        else:
            pygame.init()
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK,
                pygame.GL_CONTEXT_PROFILE_CORE,
            )
            # SDL's default depth buffer can be 16-bit, which is far too coarse for
            # the 1..16384 depth range: distant walls bled through near ones.
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            if fullscreen:
                flags |= pygame.FULLSCREEN
            else:
                flags |= pygame.RESIZABLE
            pygame.display.set_mode((width, height), flags, vsync=1 if vsync else 0)
            pygame.display.set_caption(title)
            self.ctx = moderngl.create_context(require=330)

        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable(moderngl.DEPTH_TEST)

    @property
    def aspect_ratio(self) -> float:
        """Return viewport width / height aspect ratio."""
        return self.width / max(self.height, 1)

    def resize(self, width: int, height: int) -> None:
        """Update viewport dimensions on window resize."""
        self.width = max(1, width)
        self.height = max(1, height)
        if self.standalone and self._fbo is not None:
            self._fbo.release()
            color_tex = self.ctx.texture((self.width, self.height), 4)
            depth_rb = self.ctx.depth_renderbuffer((self.width, self.height))
            self._fbo = self.ctx.framebuffer(
                color_attachments=[color_tex],
                depth_attachment=depth_rb,
            )
            self._fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)

    def set_vsync(self, enabled: bool) -> None:
        """Toggle vertical sync on the display window."""
        self.vsync = enabled
        if not self.standalone:
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            if self.fullscreen:
                flags |= pygame.FULLSCREEN
            else:
                flags |= pygame.RESIZABLE
            pygame.display.set_mode(
                (self.width, self.height),
                flags,
                vsync=1 if enabled else 0,
            )
            self.ctx.viewport = (0, 0, self.width, self.height)

    def set_fullscreen(self, enabled: bool) -> None:
        """Toggle fullscreen mode on the display window."""
        self.fullscreen = enabled
        if not self.standalone:
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            if enabled:
                flags |= pygame.FULLSCREEN
            else:
                flags |= pygame.RESIZABLE
            pygame.display.set_mode(
                (self.width, self.height),
                flags,
                vsync=1 if self.vsync else 0,
            )
            self.ctx.viewport = (0, 0, self.width, self.height)

    def set_resolution(self, width: int, height: int) -> None:
        """Update window and viewport resolution."""
        self.width = max(1, width)
        self.height = max(1, height)
        if not self.standalone:
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            if self.fullscreen:
                flags |= pygame.FULLSCREEN
            else:
                flags |= pygame.RESIZABLE
            pygame.display.set_mode(
                (self.width, self.height),
                flags,
                vsync=1 if self.vsync else 0,
            )
        self.resize(self.width, self.height)

    def set_caption(self, caption: str) -> None:
        """Update the window title bar text."""
        if not self.standalone:
            pygame.display.set_caption(caption)

    def clear(
        self,
        r: float = 0.0,
        g: float = 0.0,
        b: float = 0.0,
        a: float = 1.0,
    ) -> None:
        """Clear color and depth buffers."""
        if self._fbo is not None:
            self._fbo.use()
        self.ctx.clear(r, g, b, a, depth=1.0)

    def swap_buffers(self) -> None:
        """Swap display buffers if running in windowed mode."""
        if not self.standalone:
            pygame.display.flip()

    def read_pixels(self) -> np.ndarray:
        """Read the current framebuffer as a top-down (height, width, 4) uint8 RGBA array."""
        target = self._fbo if self._fbo is not None else self.ctx.screen
        raw = target.read(components=4)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 4))
        return np.flipud(arr).copy()

    def render_test_triangle(self, *, read_pixels: bool = False) -> np.ndarray | None:
        """Render a colored test triangle and optionally return the RGBA framebuffer array."""
        prog = self.ctx.program(
            vertex_shader=_TEST_TRIANGLE_VERT,
            fragment_shader=_TEST_TRIANGLE_FRAG,
        )
        vertices = np.array(
            [
                # x,    y,   r,   g,   b
                0.0,
                0.6,
                1.0,
                0.2,
                0.2,
                -0.6,
                -0.6,
                0.2,
                1.0,
                0.2,
                0.6,
                -0.6,
                0.2,
                0.4,
                1.0,
            ],
            dtype=np.float32,
        )
        vbo = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.vertex_array(prog, [(vbo, "2f 3f", "in_pos", "in_color")])
        try:
            self.clear(0.05, 0.05, 0.08, 1.0)
            vao.render(moderngl.TRIANGLES)
            if not read_pixels:
                return None
            return self.read_pixels()
        finally:
            vao.release()
            vbo.release()
            prog.release()

    def close(self) -> None:
        """Release GL resources and close the display window."""
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        self.ctx.release()
        if not self.standalone:
            pygame.display.quit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
