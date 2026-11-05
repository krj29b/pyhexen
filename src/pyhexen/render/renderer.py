# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Top-level 3D world renderer integrating chunks, shaders, sky, sprites, and screenshots (T4.8)."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import moderngl
import numpy as np

from pyhexen.core.events import EventBus
from pyhexen.core.random import Random
from pyhexen.render.camera import Camera
from pyhexen.render.chunks import ChunkManager, DrawBatch, SectorChunk
from pyhexen.render.context import RenderContext
from pyhexen.render.polyobj import (
    PolyMesh,
    build_polyobj_mesh,
    compute_poly_model_matrix,
    interpolate_angle,
    interpolate_coord,
    poly_model_to_gl_matrix,
)
from pyhexen.render.sky import LIGHTNING_SPECIAL, LIGHTNING_SPECIAL2, SkyRenderer
from pyhexen.render.sprites import SpriteRenderer
from pyhexen.render.textures import (
    SURFACE_FLAT,
    SURFACE_MASKED_WALL,
    SURFACE_SKY,
    SURFACE_WALL,
    GpuTexture,
    GpuTextureManager,
    WorldShaderProgram,
)
from pyhexen.wad.mapinfo import MapInfo, load_mapinfo
from pyhexen.wad.palette import save_png
from pyhexen.wad.sprites import SpriteRegistry
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import Level, Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


class WorldRenderer:
    """Orchestrates 3D rendering of a Hexen level: sky, walls, flats, and sprites (T4.8)."""

    def __init__(
        self,
        context: RenderContext,
        wad: WadStack | WadFile | None = None,
        *,
        fov_deg: float = 90.0,
    ) -> None:
        self.context = context
        self.ctx: moderngl.Context = context.ctx
        self.fov_deg: float = fov_deg

        self.wad: WadStack | WadFile | None = wad
        self.level: Level | None = None
        self.map_info: MapInfo | None = None
        self.use_fogmap: bool = False
        self.use_fullbright: bool = True

        self.world_shader = WorldShaderProgram(self.ctx)
        self.sky_renderer = SkyRenderer(self.ctx)

        self.tex_mgr: GpuTextureManager | None = None
        self.chunks: ChunkManager | None = None
        self.sprite_registry: SpriteRegistry | None = None
        self.sprite_renderer: SpriteRenderer | None = None

        self._chunk_world_vaos: dict[int, tuple[moderngl.Buffer, moderngl.VertexArray]] = {}
        self._chunk_sky_vaos: dict[int, tuple[moderngl.Buffer, moderngl.VertexArray]] = {}
        self.poly_meshes: dict[int, PolyMesh] = {}
        self._fallback_tex: GpuTexture | None = None

        if wad is not None:
            self.tex_mgr = GpuTextureManager(self.ctx, wad)

    def _get_fallback_texture(self) -> GpuTexture:
        if self._fallback_tex is not None:
            return self._fallback_tex
        assert self.tex_mgr is not None
        checker = np.zeros((64, 64), dtype=np.uint8)
        checker[::2, ::2] = 64
        checker[1::2, 1::2] = 64
        checker[::2, 1::2] = 128
        checker[1::2, ::2] = 128
        self._fallback_tex = self.tex_mgr.upload_indexed("_FALLBACK", checker, repeat=True)
        return self._fallback_tex

    def _resolve_sky_safe(self, name: str) -> GpuTexture:
        assert self.tex_mgr is not None
        try:
            return self.sky_renderer.resolve_sky_texture(self.tex_mgr, name)
        except Exception:
            if self.tex_mgr.texture_registry is not None and len(self.tex_mgr.texture_registry) > 1:
                return self.tex_mgr.get_wall_texture(1, translated=False)
            return self._get_fallback_texture()

    def load_map(
        self,
        wad: WadStack | WadFile,
        level: Level,
        map_info: MapInfo | None = None,
        *,
        event_bus: EventBus | None = None,
        rng: Random | None = None,
    ) -> None:
        """Initialize GPU textures, sector chunks, sky, and sprite registry for `level`."""
        self._release_chunk_vaos()
        if self.sprite_renderer is not None:
            self.sprite_renderer.release()
            self.sprite_renderer = None

        if self.tex_mgr is None or self.wad is not wad:
            if self._fallback_tex is not None:
                self._fallback_tex.release()
                self._fallback_tex = None
            if self.tex_mgr is not None:
                self.tex_mgr.release()
            self.wad = wad
            self.tex_mgr = GpuTextureManager(self.ctx, wad)

        self.level = level

        if map_info is None and "MAPINFO" in wad:
            try:
                info_table = load_mapinfo(wad)
                map_info = info_table.get(level.name)
            except Exception:
                map_info = None
        self.map_info = map_info

        if map_info is not None:
            self.use_fogmap = map_info.fadetable.upper() == "FOGMAP"
            self.use_fullbright = not self.use_fogmap
            sky1 = self._resolve_sky_safe(map_info.sky1_texture)
            sky2 = self._resolve_sky_safe(map_info.sky2_texture) if map_info.sky2_texture else sky1
            self.sky_renderer.set_skies(
                sky1,
                sky2,
                sky1_scroll_delta=map_info.sky1_scroll_delta,
                sky2_scroll_delta=map_info.sky2_scroll_delta,
                double_sky=map_info.double_sky,
                lightning=map_info.lightning,
                rng=rng,
            )
        else:
            self.use_fogmap = False
            self.use_fullbright = True
            sky1 = self._resolve_sky_safe("SKY1")
            self.sky_renderer.set_skies(sky1, sky1, double_sky=False, lightning=False, rng=rng)

        self.chunks = ChunkManager(
            level,
            textures=self.tex_mgr.texture_registry,
            flats=self.tex_mgr.flat_registry,
            event_bus=event_bus,
            ctx=self.ctx,
        )

        self.sprite_registry = SpriteRegistry(wad)
        self.sprite_renderer = SpriteRenderer(self.ctx, self.sprite_registry, self.tex_mgr)

        self._release_poly_vaos()
        if hasattr(level, "polyobjs") and level.polyobjs:
            poly_list = (
                level.polyobjs.values() if isinstance(level.polyobjs, dict) else level.polyobjs
            )
            tex_reg = self.tex_mgr.texture_registry if self.tex_mgr else None
            for po in poly_list:
                mesh = build_polyobj_mesh(po, textures=tex_reg, ctx=self.ctx)
                if mesh.vbo is not None:
                    mesh.vao = self.world_shader.create_vao(mesh.vbo)
                self.poly_meshes[po.tag] = mesh

    def _release_poly_vaos(self) -> None:
        for mesh in self.poly_meshes.values():
            mesh.release()
        self.poly_meshes.clear()

    def _release_chunk_vaos(self) -> None:
        for _, vao in self._chunk_world_vaos.values():
            vao.release()
        self._chunk_world_vaos.clear()
        for _, vao in self._chunk_sky_vaos.values():
            vao.release()
        self._chunk_sky_vaos.clear()

    def _get_chunk_world_vao(self, chunk: SectorChunk) -> moderngl.VertexArray | None:
        if chunk.vbo is None:
            return None
        cached = self._chunk_world_vaos.get(chunk.sector_index)
        if cached is not None and cached[0] is chunk.vbo:
            return cached[1]
        if cached is not None:
            cached[1].release()
        vao = self.world_shader.create_vao(chunk.vbo)
        self._chunk_world_vaos[chunk.sector_index] = (chunk.vbo, vao)
        return vao

    def _get_chunk_sky_vao(self, chunk: SectorChunk) -> moderngl.VertexArray | None:
        if chunk.vbo is None:
            return None
        cached = self._chunk_sky_vaos.get(chunk.sector_index)
        if cached is not None and cached[0] is chunk.vbo:
            return cached[1]
        if cached is not None:
            cached[1].release()
        vao = self.sky_renderer.create_vao(chunk.vbo, world_vertex_format=True)
        self._chunk_sky_vaos[chunk.sector_index] = (chunk.vbo, vao)
        return vao

    def _resolve_batch_texture(self, batch: DrawBatch) -> GpuTexture:
        assert self.tex_mgr is not None
        try:
            if batch.surface_type == SURFACE_FLAT:
                if batch.tex_id >= 0:
                    return self.tex_mgr.get_flat_texture(batch.tex_id, translated=True)
                return self.tex_mgr.get_flat_texture(batch.tex_name, translated=True)
            if batch.tex_id > 0:
                return self.tex_mgr.get_wall_texture(batch.tex_id, translated=True)
            if batch.tex_name and batch.tex_name != "-":
                return self.tex_mgr.get_wall_texture(batch.tex_name, translated=True)
        except Exception:
            pass
        return self._get_fallback_texture()

    def _sector_lightning_extralight(self, sector: Sector) -> int:
        """Compute temporary extralight boost (0..15) for a sector during a lightning flash."""
        flash = self.sky_renderer.lightning.light_flash
        if flash <= 0:
            return 0
        is_sky = sector.ceiling_pic.upper().startswith("F_SKY")
        if is_sky:
            delta = max(0, flash - sector.light_level)
            return delta >> 4
        if sector.special == LIGHTNING_SPECIAL:
            return 4  # +64 light units = +4 lightnum steps
        if sector.special == LIGHTNING_SPECIAL2:
            return 2  # +32 light units = +2 lightnum steps
        return 0

    def tick(self, level_time: int = 0, rng: Random | None = None) -> None:
        """Advance texture/flat animations and sky scrolling/lightning by one tic (35 Hz)."""
        if self.tex_mgr is not None:
            self.tex_mgr.update_animations(level_time, rng)
        sectors = self.level.sectors if self.level is not None else None
        self.sky_renderer.tick(rng=rng, sectors=sectors)

    def render_frame(
        self,
        camera: Camera,
        sim: GameSim | None = None,
        *,
        alpha: float = 1.0,
        palette_index: int = 0,
        extralight: int = 0,
        fixed_colormap: int = -1,
        gamma: float = 1.0,
    ) -> None:
        """Render a complete 3D frame:
        sky, opaque walls/flats, masked walls/sprites, translucent.
        """
        if self.chunks is None or self.tex_mgr is None or self.level is None:
            self.context.clear(0.0, 0.0, 0.0, 1.0)
            return

        # 1. Rebuild any dirty sector chunks
        rebuilt = self.chunks.rebuild_dirty(self.ctx)
        for sec_idx in rebuilt:
            if sec_idx in self._chunk_world_vaos:
                self._chunk_world_vaos.pop(sec_idx)[1].release()
            if sec_idx in self._chunk_sky_vaos:
                self._chunk_sky_vaos.pop(sec_idx)[1].release()

        # 2. Clear color and depth buffers
        self.context.clear(0.0, 0.0, 0.0, 1.0)

        view = camera.view_matrix(alpha)
        proj = camera.projection_matrix(
            self.context.aspect_ratio,
            fov_deg=self.fov_deg,
            alpha=alpha,
        )
        _, _, _, iyaw, ipitch = camera.interpolated_state(alpha)
        fov_x_rad = math.radians(self.fov_deg)
        pitch_offset = math.tan(ipitch) * 160.0

        # 3. Draw sky polygons (SURFACE_SKY = 2.0) writing depth so sky ceilings/caps occlude
        for chunk in self.chunks.chunks:
            if chunk.vbo is None or len(chunk.vertices) == 0:
                continue
            sky_batches = [b for b in chunk.batches if b.surface_type == SURFACE_SKY]
            if not sky_batches:
                continue
            sky_vao = self._get_chunk_sky_vao(chunk)
            if sky_vao is None:
                continue
            for batch in sky_batches:
                self.sky_renderer.render(
                    sky_vao,
                    self.tex_mgr,
                    view=view,
                    proj=proj,
                    camera_yaw=iyaw,
                    fov_x=fov_x_rad,
                    pitch_offset=pitch_offset,
                    palette_index=palette_index,
                    gamma=gamma,
                    vertices=batch.vertex_count,
                    first=batch.vertex_offset,
                )

        # 4. Draw opaque walls (SURFACE_WALL = 0.0) and flats (SURFACE_FLAT = 1.0)
        self.world_shader.set_camera(view, proj)
        self.world_shader.set_model(None)
        self.world_shader.set_light_override(-1.0)
        for chunk in self.chunks.chunks:
            if chunk.vbo is None or len(chunk.vertices) == 0:
                continue
            opaque_batches = [
                b for b in chunk.batches if b.surface_type in (SURFACE_WALL, SURFACE_FLAT)
            ]
            if not opaque_batches:
                continue
            world_vao = self._get_chunk_world_vao(chunk)
            if world_vao is None:
                continue
            sector = self.level.sectors[chunk.sector_index]
            sec_extra = extralight + self._sector_lightning_extralight(sector)
            self.world_shader.set_lighting(
                palette_index=palette_index,
                extralight=sec_extra,
                fixed_colormap=fixed_colormap,
                use_fogmap=self.use_fogmap,
                gamma=gamma,
                alpha=1.0,
            )
            for batch in opaque_batches:
                gpu_tex = self._resolve_batch_texture(batch)
                self.world_shader.bind_surface_texture(self.tex_mgr, gpu_tex)
                world_vao.render(
                    mode=moderngl.TRIANGLES,
                    vertices=batch.vertex_count,
                    first=batch.vertex_offset,
                )

        # 5. Draw masked mid-walls (SURFACE_MASKED_WALL = 3.0)
        for chunk in self.chunks.chunks:
            if chunk.vbo is None or len(chunk.vertices) == 0:
                continue
            masked_batches = [b for b in chunk.batches if b.surface_type == SURFACE_MASKED_WALL]
            if not masked_batches:
                continue
            world_vao = self._get_chunk_world_vao(chunk)
            if world_vao is None:
                continue
            sector = self.level.sectors[chunk.sector_index]
            sec_extra = extralight + self._sector_lightning_extralight(sector)
            self.world_shader.set_lighting(
                palette_index=palette_index,
                extralight=sec_extra,
                fixed_colormap=fixed_colormap,
                use_fogmap=self.use_fogmap,
                gamma=gamma,
                alpha=1.0,
            )
            for batch in masked_batches:
                gpu_tex = self._resolve_batch_texture(batch)
                self.world_shader.bind_surface_texture(self.tex_mgr, gpu_tex)
                world_vao.render(
                    mode=moderngl.TRIANGLES,
                    vertices=batch.vertex_count,
                    first=batch.vertex_offset,
                )

        # 5.5 Draw dynamic polyobjects (opaque and masked walls)
        self.render_polyobjs(
            alpha=alpha,
            extralight=extralight,
            palette_index=palette_index,
            fixed_colormap=fixed_colormap,
            gamma=gamma,
        )

        # 6. Collect and draw opaque + translucent sprites
        if self.sprite_renderer is not None:
            opaque_sprites, translucent_sprites = self.sprite_renderer.collect_batches(
                camera,
                sim=sim,
                level=self.level,
                alpha=alpha,
                use_fullbright=self.use_fullbright,
            )
            self.sprite_renderer.render_opaque(
                self.world_shader,
                opaque_sprites,
                palette_index=palette_index,
                extralight=extralight,
                fixed_colormap=fixed_colormap,
                use_fogmap=self.use_fogmap,
                gamma=gamma,
            )
            self.sprite_renderer.render_translucent(
                self.world_shader,
                translucent_sprites,
                palette_index=palette_index,
                extralight=extralight,
                fixed_colormap=fixed_colormap,
                use_fogmap=self.use_fogmap,
                gamma=gamma,
            )

    def render_polyobjs(
        self,
        alpha: float = 1.0,
        extralight: int = 0,
        palette_index: int = 0,
        fixed_colormap: int = -1,
        gamma: float = 1.0,
    ) -> None:
        """Render dynamic polyobject meshes with interpolated transform and sector lighting."""
        if not self.poly_meshes or self.level is None or not hasattr(self.level, "polyobjs"):
            return
        poly_list = (
            self.level.polyobjs.values()
            if isinstance(self.level.polyobjs, dict)
            else self.level.polyobjs
        )
        if not poly_list:
            return

        from pyhexen.core.fixed import FRACUNIT

        for po in poly_list:
            mesh = self.poly_meshes.get(po.tag)
            if mesh is None or mesh.vbo is None or mesh.vertex_count == 0:
                continue
            vao = mesh.vao
            if vao is None:
                vao = self.world_shader.create_vao(mesh.vbo)
                mesh.vao = vao
            if vao is None:
                continue

            curr_x = (
                po.start_spot.x if hasattr(po, "start_spot") else getattr(po, "start_spot_x", 0)
            ) / float(FRACUNIT)
            curr_y = (
                po.start_spot.y if hasattr(po, "start_spot") else getattr(po, "start_spot_y", 0)
            ) / float(FRACUNIT)
            prev_x = getattr(po, "prev_x", int(curr_x * FRACUNIT)) / float(FRACUNIT)
            prev_y = getattr(po, "prev_y", int(curr_y * FRACUNIT)) / float(FRACUNIT)

            curr_ang_bam = getattr(po, "angle", 0) & 0xFFFFFFFF
            prev_ang_bam = getattr(po, "prev_angle", curr_ang_bam) & 0xFFFFFFFF

            ix = interpolate_coord(prev_x, curr_x, alpha)
            iy = interpolate_coord(prev_y, curr_y, alpha)

            curr_ang_deg = (curr_ang_bam * 360.0) / float(1 << 32)
            prev_ang_deg = (prev_ang_bam * 360.0) / float(1 << 32)
            iangle_deg = interpolate_angle(prev_ang_deg, curr_ang_deg, alpha)
            iangle_rad = math.radians(iangle_deg)

            m_2d = compute_poly_model_matrix(ix, iy, iangle_rad)
            m_gl = poly_model_to_gl_matrix(m_2d)
            self.world_shader.set_model(m_gl)

            sub = self.level.point_in_subsector(
                round(ix * float(FRACUNIT)),
                round(iy * float(FRACUNIT)),
            )
            sector = (
                sub.sector
                if sub is not None
                else (po.subsector.sector if po.subsector is not None else None)
            )
            sec_light = float(sector.light_level) if sector is not None else 160.0
            sec_extra = extralight + (
                self._sector_lightning_extralight(sector) if sector is not None else 0
            )

            self.world_shader.set_light_override(sec_light)
            self.world_shader.set_lighting(
                palette_index=palette_index,
                extralight=sec_extra,
                fixed_colormap=fixed_colormap,
                use_fogmap=self.use_fogmap,
                gamma=gamma,
                alpha=1.0,
            )

            for batch in mesh.batches:
                gpu_tex = self._resolve_batch_texture(batch)
                self.world_shader.bind_surface_texture(self.tex_mgr, gpu_tex)
                vao.render(
                    mode=moderngl.TRIANGLES,
                    vertices=batch.vertex_count,
                    first=batch.vertex_offset,
                )

        # Reset model transform and light override
        self.world_shader.set_model(None)
        self.world_shader.set_light_override(-1.0)

    def save_screenshot(self, path: Path | str | None = None) -> Path:
        """Read the current framebuffer and save it as a top-down PNG image."""
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = Path("out") / f"screenshot_{timestamp}.png"
        else:
            out_path = Path(path)

        rgba = self.context.read_pixels()
        save_png(rgba, out_path)
        return out_path

    def release(self) -> None:
        """Release all GPU resources owned by WorldRenderer."""
        self._release_poly_vaos()
        self._release_chunk_vaos()
        if self.chunks is not None:
            for chunk in self.chunks.chunks:
                if chunk.vbo is not None and hasattr(chunk.vbo, "release"):
                    chunk.vbo.release()
                    chunk.vbo = None
            self.chunks = None
        if self.sprite_renderer is not None:
            self.sprite_renderer.release()
            self.sprite_renderer = None
        if self._fallback_tex is not None:
            self._fallback_tex.release()
            self._fallback_tex = None
        if self.tex_mgr is not None:
            self.tex_mgr.release()
            self.tex_mgr = None
        self.sky_renderer.release()
        self.world_shader.release()


def render_polyobjs(renderer_inst: WorldRenderer, *args: Any, **kwargs: Any) -> Any:
    """Helper module-level entry point for polyobject rendering."""
    return renderer_inst.render_polyobjs(*args, **kwargs)
