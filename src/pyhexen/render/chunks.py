# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Per-sector geometry chunks and dynamic dirty rebuilds (ADR-004).

Groups each sector's floor, ceiling, and wall pieces into a `SectorChunk`
with draw batches and an optional GPU VBO. Subscribes to `SectorChanged`
events to mark both the modified sector and its two-sided neighbours dirty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyhexen.core.events import EventBus, SectorChanged
from pyhexen.render.mesh_flats import (
    SURFACE_FLAT,
    SURFACE_SKY,
    Point2D,
    build_sector_flats_from_triangles,
    triangulate_sector,
)
from pyhexen.render.mesh_walls import build_sector_walls
from pyhexen.render.textures import SURFACE_MASKED_WALL, SURFACE_WALL
from pyhexen.wad.flats import FlatRegistry
from pyhexen.wad.textures import TextureRegistry
from pyhexen.wad.wadfile import normalize_lump_name
from pyhexen.world.level import Level


@dataclass(slots=True)
class DrawBatch:
    """A contiguous slice of vertices in a `SectorChunk` sharing surface type and texture."""

    surface_type: float
    tex_id: int
    tex_name: str
    vertex_offset: int
    vertex_count: int
    gpu_tex: Any = None
    is_animated: bool = False


@dataclass(slots=True)
class SectorChunk:
    """CPU/GPU mesh data and draw batches for a single sector."""

    sector_index: int
    neighbor_sectors: tuple[int, ...]
    vertices: np.ndarray  # float32[N, 8]
    batches: list[DrawBatch] = field(default_factory=list)
    used_flat_fallback: bool = False
    dirty: bool = False
    vbo: Any = None
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    sky_batches: tuple[DrawBatch, ...] = field(default_factory=tuple)
    opaque_batches: tuple[DrawBatch, ...] = field(default_factory=tuple)
    masked_batches: tuple[DrawBatch, ...] = field(default_factory=tuple)
    world_vao: Any = None
    sky_vao: Any = None


class ChunkManager:
    """Owns per-sector mesh chunks and rebuilds dirty sectors on demand (ADR-004)."""

    def __init__(
        self,
        level: Level,
        textures: TextureRegistry | None = None,
        flats: FlatRegistry | None = None,
        event_bus: EventBus | None = None,
        ctx: Any = None,
    ) -> None:
        self.level = level
        self.textures = textures
        self.flats = flats
        self.ctx = ctx

        num_sectors = len(level.sectors)
        self._neighbors: list[tuple[int, ...]] = self._compute_neighbors()
        # Cache 2D polygon triangulation per sector since 2D boundaries are static
        self._cached_tris: list[tuple[list[tuple[Point2D, Point2D, Point2D]], bool]] = [
            triangulate_sector(level, i) for i in range(num_sectors)
        ]

        self.chunks: list[SectorChunk] = [
            self._build_sector_chunk(i, existing_vbo=None) for i in range(num_sectors)
        ]

        if event_bus is not None:
            event_bus.subscribe(SectorChanged, self._on_sector_changed)

    def _compute_neighbors(self) -> list[tuple[int, ...]]:
        neighbors: list[set[int]] = [set() for _ in range(len(self.level.sectors))]
        for line in self.level.lines:
            if line.front_sector is not None and line.back_sector is not None:
                f_idx = line.front_sector.index
                b_idx = line.back_sector.index
                if f_idx != b_idx:
                    neighbors[f_idx].add(b_idx)
                    neighbors[b_idx].add(f_idx)
        return [tuple(sorted(s)) for s in neighbors]

    def _on_sector_changed(self, event: SectorChanged) -> None:
        self.mark_sector_dirty(event.sector_index)

    def mark_sector_dirty(self, sector_index: int) -> None:
        """Mark `sector_index` and all of its neighbouring sectors dirty."""
        if sector_index < 0 or sector_index >= len(self.chunks):
            return
        self.chunks[sector_index].dirty = True
        for nbr in self._neighbors[sector_index]:
            self.chunks[nbr].dirty = True

    def _build_sector_chunk(
        self,
        sector_index: int,
        *,
        existing_vbo: Any = None,
        ctx: Any = None,
    ) -> SectorChunk:
        sector = self.level.sectors[sector_index]
        tris, used_fallback = self._cached_tris[sector_index]
        flat_mesh = build_sector_flats_from_triangles(sector, tris, used_fallback, flats=self.flats)
        wall_quads = build_sector_walls(self.level, sector_index, textures=self.textures)

        batches: list[DrawBatch] = []
        arrays: list[np.ndarray] = []
        offset = 0

        if len(flat_mesh.floor_vertices) > 0:
            floor_surf = SURFACE_SKY if flat_mesh.floor_is_sky else SURFACE_FLAT
            count = len(flat_mesh.floor_vertices)
            batches.append(
                DrawBatch(
                    surface_type=floor_surf,
                    tex_id=flat_mesh.floor_tex_id,
                    tex_name=normalize_lump_name(sector.floor_pic),
                    vertex_offset=offset,
                    vertex_count=count,
                )
            )
            arrays.append(flat_mesh.floor_vertices)
            offset += count

        if len(flat_mesh.ceiling_vertices) > 0:
            ceil_surf = SURFACE_SKY if flat_mesh.ceil_is_sky else SURFACE_FLAT
            count = len(flat_mesh.ceiling_vertices)
            batches.append(
                DrawBatch(
                    surface_type=ceil_surf,
                    tex_id=flat_mesh.ceil_tex_id,
                    tex_name=normalize_lump_name(sector.ceiling_pic),
                    vertex_offset=offset,
                    vertex_count=count,
                )
            )
            arrays.append(flat_mesh.ceiling_vertices)
            offset += count

        # Group wall quads by (surface_type, tex_id, tex_name) to minimize draw calls
        grouped_walls: dict[tuple[float, int, str], list[np.ndarray]] = {}
        for wq in wall_quads:
            key = (wq.surface_type, wq.tex_id, wq.tex_name)
            grouped_walls.setdefault(key, []).append(wq.vertices)

        for (surf_type, tex_id, tex_name), quad_list in grouped_walls.items():
            count = 6 * len(quad_list)
            batches.append(
                DrawBatch(
                    surface_type=surf_type,
                    tex_id=tex_id,
                    tex_name=tex_name,
                    vertex_offset=offset,
                    vertex_count=count,
                )
            )
            arrays.extend(quad_list)
            offset += count

        if arrays:
            vertices = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
        else:
            vertices = np.zeros((0, 8), dtype=np.float32)

        active_ctx = ctx if ctx is not None else self.ctx
        vbo = existing_vbo
        if active_ctx is not None:
            raw_bytes = vertices.tobytes()
            if len(raw_bytes) > 0:
                if vbo is not None and getattr(vbo, "size", -1) == len(raw_bytes):
                    vbo.write(raw_bytes)
                else:
                    if vbo is not None and hasattr(vbo, "release"):
                        vbo.release()
                    vbo = active_ctx.buffer(raw_bytes)

        cx = 0.0
        cy = 0.0
        radius = 0.0
        if len(vertices) > 0:
            min_x = float(np.min(vertices[:, 0]))
            max_x = float(np.max(vertices[:, 0]))
            map_y = -vertices[:, 2]
            min_y = float(np.min(map_y))
            max_y = float(np.max(map_y))
            cx = (min_x + max_x) * 0.5
            cy = (min_y + max_y) * 0.5
            hx = (max_x - min_x) * 0.5
            hy = (max_y - min_y) * 0.5
            radius = math.sqrt(hx * hx + hy * hy)

        sky_batches = tuple(b for b in batches if b.surface_type == SURFACE_SKY)
        opaque_batches = tuple(b for b in batches if b.surface_type in (SURFACE_WALL, SURFACE_FLAT))
        masked_batches = tuple(b for b in batches if b.surface_type == SURFACE_MASKED_WALL)

        return SectorChunk(
            sector_index=sector_index,
            neighbor_sectors=self._neighbors[sector_index],
            vertices=vertices,
            batches=batches,
            used_flat_fallback=used_fallback,
            dirty=False,
            vbo=vbo,
            cx=cx,
            cy=cy,
            radius=radius,
            sky_batches=sky_batches,
            opaque_batches=opaque_batches,
            masked_batches=masked_batches,
        )

    def rebuild_dirty(self, ctx: Any = None) -> list[int]:
        """Rebuild all dirty sector chunks and return their sector indices."""
        rebuilt: list[int] = []
        for idx, chunk in enumerate(self.chunks):
            if chunk.dirty:
                old_vbo = chunk.vbo
                old_world_vao = chunk.world_vao
                old_sky_vao = chunk.sky_vao
                new_chunk = self._build_sector_chunk(idx, existing_vbo=chunk.vbo, ctx=ctx)
                if new_chunk.vbo is old_vbo:
                    new_chunk.world_vao = old_world_vao
                    new_chunk.sky_vao = old_sky_vao
                else:
                    if old_world_vao is not None and hasattr(old_world_vao, "release"):
                        old_world_vao.release()
                    if old_sky_vao is not None and hasattr(old_sky_vao, "release"):
                        old_sky_vao.release()
                self.chunks[idx] = new_chunk
                rebuilt.append(idx)
        return rebuilt

    @property
    def fallback_sectors(self) -> list[int]:
        """Return indices of sectors whose flats used the BSP-clip fallback."""
        return [c.sector_index for c in self.chunks if c.used_flat_fallback]
