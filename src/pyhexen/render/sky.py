# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Single/double-layer sky renderer (T4.6).

Port of R_DrawSkyPlane / R_DrawDoubleSkyPlane (r_plane.c). Sky scrolling and the
lightning sky swap are simulation state (P_AnimateSurfaces / P_LightningFlash in
`sim/specials/anim.py`); the renderer only reads them (`sync_from_animator`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import moderngl
import numpy as np

from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.textures import GpuTexture, GpuTextureManager
from pyhexen.wad.mapinfo import MapInfo

if TYPE_CHECKING:
    from pyhexen.sim.specials.anim import SurfaceAnimator

_SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


class SkyRenderer:
    """Renders Hexen single-layer and double-layer scrolling skies (T4.6).

    Supports:
    - Single sky (``sky1``, or ``sky2`` during a lightning flash)
    - Double sky (``doublesky`` in ``MAPINFO``): ``sky1`` rendered in front with palette
      index 0 transparent over ``sky2``
    - Horizontal scroll from the sim's 16.16 column offsets
    - Cylindrical yaw projection + vertical pitch/lookdir offset without perspective
      seam distortion
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        vert_src = (_SHADERS_DIR / "sky.vert").read_text(encoding="utf-8")
        frag_src = (_SHADERS_DIR / "sky.frag").read_text(encoding="utf-8")
        self.program = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        self.program["u_sky1_tex"].value = 0
        self.program["u_sky2_tex"].value = 1
        self.program["u_palette"].value = 2

        self._u_view = self.program["u_view"]
        self._u_proj = self.program["u_proj"]
        self._u_camera_yaw = self.program["u_camera_yaw"]
        self._u_fov_x = self.program["u_fov_x"]
        self._u_pitch_offset = self.program["u_pitch_offset"]
        self._u_sky1_offset = self.program["u_sky1_offset"]
        self._u_sky2_offset = self.program["u_sky2_offset"]
        self._u_double_sky = self.program["u_double_sky"]
        self._u_use_sky2 = self.program["u_use_sky2"]
        self._u_palette_index = self.program["u_palette_index"]
        self._u_gamma = self.program["u_gamma"]

        self.sky1_tex: GpuTexture | None = None
        self.sky2_tex: GpuTexture | None = None
        self.sky1_column_offset: int = 0
        self.sky2_column_offset: int = 0
        self.double_sky: bool = False
        # Sky1Texture swapped to the map's sky2 during a lightning flash (P_LightningFlash).
        self.use_sky2: bool = False

        self._quad_vbo: moderngl.Buffer | None = None
        self._quad_vao: moderngl.VertexArray | None = None

    @staticmethod
    def resolve_sky_texture(tex_mgr: GpuTextureManager, name: str) -> GpuTexture:
        """Resolve a sky texture name from TextureRegistry or fallback patch lump."""
        if tex_mgr.texture_registry is not None:
            idx = tex_mgr.texture_registry.check_num_for_name(name)
            if idx >= 0:
                return tex_mgr.get_wall_texture(idx, translated=False)
        return tex_mgr.get_patch_texture(name)

    def set_skies(
        self,
        sky1_tex: GpuTexture,
        sky2_tex: GpuTexture | None = None,
        *,
        double_sky: bool = False,
    ) -> None:
        """Set the map's sky textures and doublesky flag (texture part of R_InitSky)."""
        self.sky1_tex = sky1_tex
        self.sky2_tex = sky2_tex if sky2_tex is not None else sky1_tex
        self.sky1_column_offset = 0
        self.sky2_column_offset = 0
        self.double_sky = bool(double_sky)
        self.use_sky2 = False

    def configure_from_mapinfo(self, map_entry: MapInfo, tex_mgr: GpuTextureManager) -> None:
        """Set sky textures from a parsed MAPINFO entry (R_InitSky)."""
        sky1 = self.resolve_sky_texture(tex_mgr, map_entry.sky1_texture)
        sky2 = (
            self.resolve_sky_texture(tex_mgr, map_entry.sky2_texture)
            if map_entry.sky2_texture
            else sky1
        )
        self.set_skies(sky1, sky2, double_sky=map_entry.double_sky)

    def sync_from_animator(self, animator: SurfaceAnimator) -> None:
        """Take the sky column offsets and the lightning sky swap from the sim."""
        self.sky1_column_offset = animator.sky1_column_offset & 0xFFFFFFFF
        self.sky2_column_offset = animator.sky2_column_offset & 0xFFFFFFFF
        self.use_sky2 = animator.sky_flash

    @property
    def sky1_offset_texels(self) -> float:
        """Return current sky1 horizontal column offset in texels."""
        # Convert 32-bit fixed-point offset (or signed fixed delta) to texels
        signed = (
            self.sky1_column_offset - 0x100000000
            if self.sky1_column_offset & 0x80000000
            else self.sky1_column_offset
        )
        return float(signed) / float(FRACUNIT)

    @property
    def sky2_offset_texels(self) -> float:
        """Return current sky2 horizontal column offset in texels."""
        signed = (
            self.sky2_column_offset - 0x100000000
            if self.sky2_column_offset & 0x80000000
            else self.sky2_column_offset
        )
        return float(signed) / float(FRACUNIT)

    def create_vao(
        self,
        vbo: moderngl.Buffer,
        ibo: moderngl.Buffer | None = None,
        *,
        world_vertex_format: bool = True,
    ) -> moderngl.VertexArray:
        """Create a VAO for sky polygons.

        If ``world_vertex_format=True`` (the default), expects the shared 8-float
        world vertex layout ``(in_pos, in_uv, in_light, in_surface_type, in_tex_id)``
        and skips the trailing 5 floats via ``3f 20x``.
        """
        fmt = "3f 20x" if world_vertex_format else "3f"
        return self.ctx.vertex_array(
            self.program,
            [(vbo, fmt, "in_pos")],
            index_buffer=ibo,
        )

    def prepare_frame(
        self,
        tex_mgr: GpuTextureManager,
        *,
        view: np.ndarray,
        proj: np.ndarray,
        camera_yaw: float = 0.0,
        fov_x: float = math.pi / 2.0,
        pitch_offset: float = 0.0,
        palette_index: int = 0,
        gamma: float = 1.0,
        use_sky2_override: bool | None = None,
    ) -> bool:
        """Bind textures and set shader uniforms for sky rendering once per frame."""
        if self.sky1_tex is None:
            return False
        sky2 = self.sky2_tex if self.sky2_tex is not None else self.sky1_tex

        self.sky1_tex.index_tex.use(location=0)
        sky2.index_tex.use(location=1)
        tex_mgr.palette_tex.use(location=2)

        self._u_view.write(np.ascontiguousarray(view.T, dtype=np.float32).tobytes())
        self._u_proj.write(np.ascontiguousarray(proj.T, dtype=np.float32).tobytes())
        self._u_camera_yaw.value = float(camera_yaw)
        self._u_fov_x.value = float(fov_x)
        self._u_pitch_offset.value = float(pitch_offset)
        self._u_sky1_offset.value = self.sky1_offset_texels
        self._u_sky2_offset.value = self.sky2_offset_texels
        self._u_double_sky.value = 1 if self.double_sky else 0

        use_sky2 = self.use_sky2 if use_sky2_override is None else use_sky2_override
        self._u_use_sky2.value = 1 if use_sky2 else 0
        self._u_palette_index.value = int(palette_index)
        self._u_gamma.value = float(gamma)
        return True

    def render(
        self,
        vao: moderngl.VertexArray,
        tex_mgr: GpuTextureManager | None = None,
        *,
        view: np.ndarray | None = None,
        proj: np.ndarray | None = None,
        camera_yaw: float = 0.0,
        fov_x: float = math.pi / 2.0,
        pitch_offset: float = 0.0,
        palette_index: int = 0,
        gamma: float = 1.0,
        use_sky2_override: bool | None = None,
        vertices: int = -1,
        first: int = 0,
    ) -> None:
        """Render sky geometry using the configured single or double sky layers."""
        if view is not None and proj is not None and tex_mgr is not None:
            if not self.prepare_frame(
                tex_mgr,
                view=view,
                proj=proj,
                camera_yaw=camera_yaw,
                fov_x=fov_x,
                pitch_offset=pitch_offset,
                palette_index=palette_index,
                gamma=gamma,
                use_sky2_override=use_sky2_override,
            ):
                return
        vao.render(mode=moderngl.TRIANGLES, vertices=vertices, first=first)

    def render_fullscreen_quad(
        self,
        tex_mgr: GpuTextureManager,
        *,
        camera_yaw: float = 0.0,
        fov_x: float = math.pi / 2.0,
        pitch_offset: float = 0.0,
        palette_index: int = 0,
        gamma: float = 1.0,
        use_sky2_override: bool | None = None,
    ) -> None:
        """Render the sky onto a full-screen NDC quad (for testing or background sky pass)."""
        if self._quad_vao is None:
            quad = np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            self._quad_vbo = self.ctx.buffer(quad.tobytes())
            self._quad_vao = self.create_vao(self._quad_vbo, world_vertex_format=False)

        identity = np.eye(4, dtype=np.float32)
        self.render(
            self._quad_vao,
            tex_mgr,
            view=identity,
            proj=identity,
            camera_yaw=camera_yaw,
            fov_x=fov_x,
            pitch_offset=pitch_offset,
            palette_index=palette_index,
            gamma=gamma,
            use_sky2_override=use_sky2_override,
        )

    def release(self) -> None:
        """Release GPU resources owned by SkyRenderer."""
        if self._quad_vao is not None:
            self._quad_vao.release()
            self._quad_vao = None
        if self._quad_vbo is not None:
            self._quad_vbo.release()
            self._quad_vbo = None
        self.program.release()
