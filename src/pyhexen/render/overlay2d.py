# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""320x200 virtual 2D overlay canvas with aspect-correct 4:3 GPU presentation (T12.1).

Port of V_DrawPatch, V_DrawTLPatch, V_DrawRawScreen (v_video.c) and MN_DrText* (mn_menu.c).
"""

from __future__ import annotations

import moderngl
import numpy as np

from pyhexen.render.textures import GpuTextureManager
from pyhexen.ui.fonts import FontSet, blit_picture
from pyhexen.wad.palette import apply_palette
from pyhexen.wad.picture import Picture
from pyhexen.wad.wadfile import WadFile, WadStack

VIRTUAL_WIDTH: int = 320
VIRTUAL_HEIGHT: int = 200
DISPLAY_ASPECT: float = 4.0 / 3.0

_OVERLAY_VERT_SHADER = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_OVERLAY_FRAG_SHADER = """
#version 330 core
uniform sampler2D u_indices;
uniform sampler2D u_mask;
uniform sampler2D u_palette;
uniform int u_palette_index;
uniform float u_gamma;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    ivec2 tex_size = textureSize(u_indices, 0);
    ivec2 coord = clamp(ivec2(v_uv * vec2(tex_size)), ivec2(0), tex_size - ivec2(1));
    float mask_val = texelFetch(u_mask, coord, 0).r;
    if (mask_val < 0.5) {
        discard;
    }
    int color_idx = int(texelFetch(u_indices, coord, 0).r * 255.0 + 0.5);
    int num_pals = textureSize(u_palette, 0).y;
    int pal_row = clamp(u_palette_index, 0, max(0, num_pals - 1));
    vec3 rgb = texelFetch(u_palette, ivec2(color_idx, pal_row), 0).rgb;
    if (u_gamma > 0.0 && abs(u_gamma - 1.0) > 1e-4) {
        rgb = pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / u_gamma));
    }
    frag_color = vec4(rgb, 1.0);
}
"""


def compute_4_3_viewport(window_width: int, window_height: int) -> tuple[int, int, int, int]:
    """Compute a centred 4:3 `(x, y, width, height)` viewport box inside window bounds."""
    w = max(1, int(window_width))
    h = max(1, int(window_height))
    if w / h >= DISPLAY_ASPECT:
        vp_h = h
        vp_w = max(1, int(round(h * DISPLAY_ASPECT)))
        vp_x = (w - vp_w) // 2
        vp_y = 0
    else:
        vp_w = w
        vp_h = max(1, int(round(w / DISPLAY_ASPECT)))
        vp_x = 0
        vp_y = (h - vp_h) // 2
    return (vp_x, vp_y, vp_w, vp_h)


class Overlay2D:
    """320x200 indexed 2D canvas for status bar, messages, menus, and intermission screens."""

    def __init__(
        self,
        fonts: FontSet | None = None,
        *,
        wad: WadStack | WadFile | None = None,
    ) -> None:
        self.fonts: FontSet = fonts if fonts is not None else FontSet.from_wad(wad)
        self.indices: np.ndarray = np.zeros((VIRTUAL_HEIGHT, VIRTUAL_WIDTH), dtype=np.uint8)
        self.mask: np.ndarray = np.zeros((VIRTUAL_HEIGHT, VIRTUAL_WIDTH), dtype=bool)

        # Lazily initialized GPU resources tied to a ModernGL context
        self._ctx: moderngl.Context | None = None
        self._prog: moderngl.Program | None = None
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None
        self._index_tex: moderngl.Texture | None = None
        self._mask_tex: moderngl.Texture | None = None

    def clear(self) -> None:
        """Clear the 320x200 overlay canvas to fully transparent."""
        self.indices.fill(0)
        self.mask.fill(False)

    def fill_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color_index: int,
    ) -> None:
        """Fill an opaque rectangle on the 320x200 canvas with palette index `color_index`."""
        x0 = max(0, min(VIRTUAL_WIDTH, int(x)))
        y0 = max(0, min(VIRTUAL_HEIGHT, int(y)))
        x1 = max(0, min(VIRTUAL_WIDTH, int(x) + int(w)))
        y1 = max(0, min(VIRTUAL_HEIGHT, int(y) + int(h)))
        if x1 <= x0 or y1 <= y0:
            return
        self.indices[y0:y1, x0:x1] = color_index & 0xFF
        self.mask[y0:y1, x0:x1] = True

    def draw_raw_screen(self, raw_screen: bytes | bytearray | memoryview | np.ndarray) -> None:
        """Draw a full 320x200 raw screen lump (port of V_DrawRawScreen in v_video.c)."""
        if isinstance(raw_screen, np.ndarray):
            arr = np.asarray(raw_screen, dtype=np.uint8).reshape((VIRTUAL_HEIGHT, VIRTUAL_WIDTH))
        else:
            buf = memoryview(raw_screen)
            expected = VIRTUAL_WIDTH * VIRTUAL_HEIGHT
            if len(buf) < expected:
                raise ValueError(f"Raw screen requires {expected} bytes, got {len(buf)}")
            arr = np.frombuffer(buf, dtype=np.uint8, count=expected).reshape(
                (VIRTUAL_HEIGHT, VIRTUAL_WIDTH)
            )
        np.copyto(self.indices, arr)
        self.mask.fill(True)

    def draw_picture(
        self,
        pic: Picture,
        x: int,
        y: int,
        *,
        flip: bool = False,
        translucent: bool = False,
    ) -> None:
        """Draw a `Picture` patch/sprite onto the 320x200 canvas at `(x, y)`."""
        blit_picture(
            self.indices,
            self.mask,
            pic,
            x,
            y,
            flip=flip,
            translucent=translucent,
        )

    def draw_text_a(
        self,
        text: str,
        x: int,
        y: int,
        *,
        yellow: bool = False,
        center: bool = False,
    ) -> int:
        """Draw text using `FontA` (or `FontAYellow` when `yellow=True`).

        When `center=True`, each line is horizontally centred around `x`.
        """
        font = self.fonts.font_a_yellow if yellow else self.fonts.font_a
        if not center:
            return font.draw_text(self.indices, self.mask, text, x, y)

        cur_y = int(y)
        end_x = int(x)
        for line in text.split("\n"):
            line_w = font.text_width(line)
            start_x = int(x) - (line_w // 2)
            end_x = font.draw_text(self.indices, self.mask, line, start_x, cur_y)
            cur_y += font.line_height
        return end_x

    def draw_text_b(
        self,
        text: str,
        x: int,
        y: int,
        *,
        center: bool = False,
    ) -> int:
        """Draw large title/menu text using `FontB`.

        When `center=True`, each line is horizontally centred around `x`.
        """
        font = self.fonts.font_b
        if not center:
            return font.draw_text(self.indices, self.mask, text, x, y)

        cur_y = int(y)
        end_x = int(x)
        for line in text.split("\n"):
            line_w = font.text_width(line)
            start_x = int(x) - (line_w // 2)
            end_x = font.draw_text(self.indices, self.mask, line, start_x, cur_y)
            cur_y += font.line_height
        return end_x

    def to_rgba(self, palette: np.ndarray) -> np.ndarray:
        """Convert the current 320x200 canvas to a `(200, 320, 4)` uint8 RGBA array."""
        pal = np.asarray(palette, dtype=np.uint8)
        if pal.ndim == 3:
            pal = pal[0]
        return apply_palette(self.indices, pal, self.mask)

    def _ensure_gpu_resources(self, ctx: moderngl.Context) -> None:
        if self._ctx is ctx and self._prog is not None:
            return

        self.release_gpu()
        self._ctx = ctx
        self._prog = ctx.program(
            vertex_shader=_OVERLAY_VERT_SHADER,
            fragment_shader=_OVERLAY_FRAG_SHADER,
        )
        # Fullscreen NDC quad: (x, y, u, v) with v=0 at top (y=+1) and v=1 at bottom (y=-1)
        quad_vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )
        self._vbo = ctx.buffer(quad_vertices.tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 2f", "in_pos", "in_uv")],
        )

        self._index_tex = ctx.texture((VIRTUAL_WIDTH, VIRTUAL_HEIGHT), 1, alignment=1)
        self._index_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._index_tex.repeat_x = False
        self._index_tex.repeat_y = False

        self._mask_tex = ctx.texture((VIRTUAL_WIDTH, VIRTUAL_HEIGHT), 1, alignment=1)
        self._mask_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._mask_tex.repeat_x = False
        self._mask_tex.repeat_y = False

    def render(
        self,
        ctx: moderngl.Context,
        tex_mgr: GpuTextureManager,
        palette_index: int = 0,
        gamma: float = 1.0,
        *,
        preserve_aspect: bool = True,
    ) -> tuple[int, int, int, int]:
        """Upload the 320x200 canvas to GPU textures and composite over the active framebuffer.

        Returns the `(x, y, width, height)` viewport box used for presentation.
        """
        self._ensure_gpu_resources(ctx)
        assert self._prog is not None
        assert self._vao is not None
        assert self._index_tex is not None
        assert self._mask_tex is not None

        self._index_tex.write(np.ascontiguousarray(self.indices, dtype=np.uint8).tobytes())
        mask_u8 = np.where(self.mask, np.uint8(255), np.uint8(0))
        self._mask_tex.write(np.ascontiguousarray(mask_u8, dtype=np.uint8).tobytes())

        prev_viewport = ctx.viewport
        fbo_w, fbo_h = ctx.fbo.size
        target_vp = (
            compute_4_3_viewport(fbo_w, fbo_h)
            if preserve_aspect
            else (0, 0, max(1, fbo_w), max(1, fbo_h))
        )
        ctx.viewport = target_vp

        try:
            self._index_tex.use(location=0)
            self._mask_tex.use(location=1)
            tex_mgr.palette_tex.use(location=2)

            self._prog["u_indices"].value = 0
            self._prog["u_mask"].value = 1
            self._prog["u_palette"].value = 2
            self._prog["u_palette_index"].value = int(palette_index)
            self._prog["u_gamma"].value = float(gamma)

            ctx.disable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.CULL_FACE)
            self._vao.render(moderngl.TRIANGLES)
        finally:
            ctx.viewport = prev_viewport

        return target_vp

    def release_gpu(self) -> None:
        """Release any allocated ModernGL objects."""
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._vbo is not None:
            self._vbo.release()
            self._vbo = None
        if self._prog is not None:
            self._prog.release()
            self._prog = None
        if self._index_tex is not None:
            self._index_tex.release()
            self._index_tex = None
        if self._mask_tex is not None:
            self._mask_tex.release()
            self._mask_tex = None
        self._ctx = None
