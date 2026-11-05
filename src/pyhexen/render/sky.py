# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Single/double-layer sky renderer and lightning flash state machine (T4.6).

Port of R_InitSky / R_DrawSkyPlane / R_DrawDoubleSkyPlane (r_plane.c) and
P_InitLightning / P_LightningFlash / P_ForceLightning / P_AnimateSurfaces (p_anim.c).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import moderngl
import numpy as np

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.render.textures import GpuTexture, GpuTextureManager
from pyhexen.wad.mapinfo import MapInfo

# Sector special constants for lightning from p_anim.c
LIGHTNING_SPECIAL = 198
LIGHTNING_SPECIAL2 = 199
SKYCHANGE_SPECIAL = 200

_SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


def _is_sky_sector(sector: Any, sky_flat_num: int) -> bool:
    """Return True if the sector has a sky ceiling or indoor lightning special."""
    ceiling_pic = getattr(sector, "ceilingpic", None)
    if ceiling_pic is not None and sky_flat_num >= 0 and ceiling_pic == sky_flat_num:
        return True
    ceiling_name = getattr(sector, "ceiling_flat", None)
    if isinstance(ceiling_name, str) and ceiling_name.upper() == "F_SKY":
        return True
    return False


@dataclass(slots=True)
class LightningState:
    """Tracks Hexen MAPINFO lightning flashes and sector light boosts (p_anim.c)."""

    enabled: bool = False
    next_flash: int = 0
    flash_count: int = 0
    flash_light: int = 0
    use_sky2: bool = False
    sky_flat_num: int = -1
    saved_sector_lights: list[int] = field(default_factory=list)
    active_sector_mask: list[bool] = field(default_factory=list)

    def init(self, enabled: bool, rng: Random | None = None, sky_flat_num: int = -1) -> None:
        """Port of P_InitLightning (p_anim.c)."""
        self.enabled = enabled
        self.sky_flat_num = sky_flat_num
        self.flash_count = 0
        self.flash_light = 0
        self.use_sky2 = False
        self.saved_sector_lights.clear()
        self.active_sector_mask.clear()

        if not enabled:
            self.next_flash = 0
            return

        r_val = rng.p_random() if rng is not None else 0
        # don't flash at level start: ((P_Random() & 15) + 1) * 35
        self.next_flash = ((r_val & 15) + 1) * 35

    def force_lightning(
        self,
        mode: int = 0,
        rng: Random | None = None,
        sectors: Sequence[Any] | None = None,
    ) -> None:
        """Port of P_ForceLightning (p_anim.c)."""
        if mode == 1:
            self.next_flash = 0
        else:
            self.trigger_flash(rng=rng, sectors=sectors)

    def trigger_flash(
        self,
        rng: Random | None = None,
        sectors: Sequence[Any] | None = None,
    ) -> bool:
        """Port of P_LightningFlash (p_anim.c).

        Swaps active sky to sky2 for the duration of the flash, boosts outdoor sky
        sectors and indoor lightning sectors (specials 198, 199), and schedules
        the next flash.
        """
        r1 = rng.p_random() if rng is not None else 0
        self.flash_light = 200 + (r1 & 31)
        self.use_sky2 = True

        self.saved_sector_lights.clear()
        self.active_sector_mask.clear()
        found_sector = False

        if sectors is not None:
            for sec in sectors:
                light = int(getattr(sec, "lightlevel", 0))
                special = int(getattr(sec, "special", 0))
                is_sky = _is_sky_sector(sec, self.sky_flat_num)
                self.saved_sector_lights.append(light)

                if is_sky or special in (
                    LIGHTNING_SPECIAL,
                    LIGHTNING_SPECIAL2,
                    SKYCHANGE_SPECIAL,
                ):
                    if special == LIGHTNING_SPECIAL:
                        new_light = min(self.flash_light, min(255, light + 64))
                    elif special == LIGHTNING_SPECIAL2:
                        new_light = min(self.flash_light, min(255, light + 32))
                    else:
                        new_light = self.flash_light

                    if new_light >= light:
                        sec.lightlevel = new_light
                        self.active_sector_mask.append(True)
                        found_sector = True
                    else:
                        self.active_sector_mask.append(False)
                else:
                    self.active_sector_mask.append(False)
        else:
            # If no sector array is bound, still allow sky flash state for rendering
            found_sector = True

        if found_sector:
            r2 = rng.p_random() if rng is not None else 0
            self.flash_count = (r2 & 7) + 8
        else:
            self.use_sky2 = False
            self.flash_count = 0

        # Schedule next flash per P_LightningFlash in p_anim.c
        r3 = rng.p_random() if rng is not None else 0
        r4 = rng.p_random() if rng is not None else 0
        if r3 < 50 and self.next_flash == 0:
            self.next_flash = ((r4 & 7) + 16) * 35
        else:
            self.next_flash = ((r4 & 15) + 5) * 35

        return found_sector

    def tick(
        self,
        rng: Random | None = None,
        sectors: Sequence[Any] | None = None,
    ) -> bool:
        """Advance lightning state by one simulation tic (P_AnimateSurfaces).

        Returns True if a lightning flash is currently active on this tic.
        """
        if not self.enabled and self.flash_count <= 0:
            return False

        if self.flash_count > 0:
            self.flash_count -= 1
            if self.flash_count > 0:
                if sectors is not None:
                    for i, sec in enumerate(sectors):
                        if i < len(self.active_sector_mask) and self.active_sector_mask[i]:
                            saved = self.saved_sector_lights[i]
                            cur = int(getattr(sec, "lightlevel", saved))
                            if cur > saved + 4:
                                sec.lightlevel = cur - 4
                return True

            # Flash ended: restore sector light levels and revert to sky1
            if sectors is not None:
                for i, sec in enumerate(sectors):
                    if i < len(self.active_sector_mask) and self.active_sector_mask[i]:
                        sec.lightlevel = self.saved_sector_lights[i]
            self.use_sky2 = False
            return False

        self.next_flash -= 1
        if self.next_flash <= 0:
            self.trigger_flash(rng=rng, sectors=sectors)
        return self.flash_count > 0


class SkyRenderer:
    """Renders Hexen single-layer and double-layer scrolling skies (T4.6).

    Supports:
    - Single sky (``sky1``, or ``sky2`` during a lightning flash / ``SKYCHANGE_SPECIAL``)
    - Double sky (``doublesky`` in ``MAPINFO``): ``sky1`` rendered in front with palette
      index 0 transparent over ``sky2``
    - Independent 16.16 fixed-point horizontal scroll deltas per tic
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

        self.sky1_tex: GpuTexture | None = None
        self.sky2_tex: GpuTexture | None = None
        self.sky1_scroll_delta: int = 0
        self.sky2_scroll_delta: int = 0
        self.sky1_column_offset: int = 0
        self.sky2_column_offset: int = 0
        self.double_sky: bool = False
        self.lightning = LightningState()

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
        sky1_scroll_delta: int = 0,
        sky2_scroll_delta: int = 0,
        double_sky: bool = False,
        lightning: bool = False,
        rng: Random | None = None,
        sky_flat_num: int = -1,
    ) -> None:
        """Configure sky textures, fixed-point scroll speeds, doublesky, and lightning."""
        self.sky1_tex = sky1_tex
        self.sky2_tex = sky2_tex if sky2_tex is not None else sky1_tex
        self.sky1_scroll_delta = int(sky1_scroll_delta)
        self.sky2_scroll_delta = int(sky2_scroll_delta)
        self.sky1_column_offset = 0
        self.sky2_column_offset = 0
        self.double_sky = bool(double_sky)
        self.lightning.init(enabled=lightning, rng=rng, sky_flat_num=sky_flat_num)

    def configure_from_mapinfo(
        self,
        map_entry: MapInfo,
        tex_mgr: GpuTextureManager,
        *,
        rng: Random | None = None,
        sky_flat_num: int = -1,
    ) -> None:
        """Configure sky parameters from a parsed MAPINFO MapInfo (R_InitSky)."""
        sky1 = self.resolve_sky_texture(tex_mgr, map_entry.sky1_texture)
        sky2 = (
            self.resolve_sky_texture(tex_mgr, map_entry.sky2_texture)
            if map_entry.sky2_texture
            else sky1
        )
        self.set_skies(
            sky1,
            sky2,
            sky1_scroll_delta=map_entry.sky1_scroll_delta,
            sky2_scroll_delta=map_entry.sky2_scroll_delta,
            double_sky=map_entry.double_sky,
            lightning=map_entry.lightning,
            rng=rng,
            sky_flat_num=sky_flat_num,
        )

    def tick(
        self,
        rng: Random | None = None,
        sectors: Sequence[Any] | None = None,
    ) -> None:
        """Advance sky scroll offsets and lightning flash timers by one tic (P_AnimateSurfaces)."""
        # In p_anim.c: Sky1ColumnOffset += Sky1ScrollDelta; Sky2ColumnOffset += Sky2ScrollDelta;
        self.sky1_column_offset = (self.sky1_column_offset + self.sky1_scroll_delta) & 0xFFFFFFFF
        self.sky2_column_offset = (self.sky2_column_offset + self.sky2_scroll_delta) & 0xFFFFFFFF
        self.lightning.tick(rng=rng, sectors=sectors)

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

    def render(
        self,
        vao: moderngl.VertexArray,
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
        vertices: int = -1,
    ) -> None:
        """Render sky geometry using the configured single or double sky layers."""
        if self.sky1_tex is None:
            return
        sky2 = self.sky2_tex if self.sky2_tex is not None else self.sky1_tex

        self.sky1_tex.index_tex.use(location=0)
        sky2.index_tex.use(location=1)
        tex_mgr.palette_tex.use(location=2)

        self.program["u_view"].write(np.ascontiguousarray(view, dtype=np.float32).tobytes())
        self.program["u_proj"].write(np.ascontiguousarray(proj, dtype=np.float32).tobytes())
        self.program["u_camera_yaw"].value = float(camera_yaw)
        self.program["u_fov_x"].value = float(fov_x)
        self.program["u_pitch_offset"].value = float(pitch_offset)
        self.program["u_sky1_offset"].value = self.sky1_offset_texels
        self.program["u_sky2_offset"].value = self.sky2_offset_texels
        self.program["u_double_sky"].value = 1 if self.double_sky else 0

        use_sky2 = self.lightning.use_sky2 if use_sky2_override is None else use_sky2_override
        self.program["u_use_sky2"].value = 1 if use_sky2 else 0
        self.program["u_palette_index"].value = int(palette_index)
        self.program["u_gamma"].value = float(gamma)

        vao.render(mode=moderngl.TRIANGLES, vertices=vertices)

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
