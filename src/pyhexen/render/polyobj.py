# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Polyobject dynamic mesh generation, model transforms, and interpolation.

Implements local coordinate wall quad generation for moving/rotating polyobjects,
4x4 model matrix construction for OpenGL vertex transformation, and shortest-arc
frame interpolation matching vanilla Hexen mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pyhexen.core.fixed import FRACUNIT
from pyhexen.render.chunks import DrawBatch
from pyhexen.render.mesh_walls import (
    SURFACE_MASKED_WALL,
    SURFACE_OPAQUE_WALL,
    _resolve_texture,
)
from pyhexen.render.textures import WORLD_VERTEX_ATTRIBS, WORLD_VERTEX_FORMAT
from pyhexen.world.level import ML_DONTPEGBOTTOM

if TYPE_CHECKING:
    import moderngl

    from pyhexen.sim.polyobj import Polyobj
    from pyhexen.wad.textures import TextureRegistry


@dataclass(slots=True)
class PolyMesh:
    """Dynamic GPU mesh for a single polyobject in local coordinates."""

    tag: int
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    batches: list[DrawBatch] = field(default_factory=list)
    vbo: Any = None
    vao: Any = None
    vertex_count: int = 0
    raw_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    z_bottom: float = 0.0
    z_top: float = 128.0
    local_origin_x: float = 0.0
    local_origin_y: float = 0.0

    @property
    def texcoords(self) -> list[tuple[float, float]]:
        """Alias for uvs property."""
        return self.uvs

    def release(self) -> None:
        """Release GPU buffers and VAO."""
        if self.vao is not None and hasattr(self.vao, "release"):
            self.vao.release()
            self.vao = None
        if self.vbo is not None and hasattr(self.vbo, "release"):
            self.vbo.release()
            self.vbo = None


def compute_poly_model_matrix(
    x: float,
    y: float,
    angle_rad: float,
    pivot_x: float = 0.0,
    pivot_y: float = 0.0,
) -> np.ndarray:
    """Compute 4x4 2D model transform matrix M(theta, x, y) with optional pivot offset.

    Returns a 4x4 float32 matrix satisfying the standard 2D homogeneous transform:
      m[0][0] = cos(angle), m[0][1] = sin(angle)
      m[1][0] = -sin(angle), m[1][1] = cos(angle)
      m[3][0] = tx, m[3][1] = ty
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    tx = x - (pivot_x * cos_a - pivot_y * sin_a)
    ty = y - (pivot_x * sin_a + pivot_y * cos_a)
    return np.array(
        [
            [cos_a, sin_a, 0.0, 0.0],
            [-sin_a, cos_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [tx, ty, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def poly_model_to_gl_matrix(m_2d: np.ndarray) -> np.ndarray:
    """Map 2D map transform matrix to 3D OpenGL world matrix (X_gl=x, Y_gl=z, Z_gl=-y).

    Transforms local OpenGL point (lx, z, -ly, 1) into (x_world, z, -y_world, 1).
    """
    cos_a = float(m_2d[0, 0])
    sin_a = float(m_2d[0, 1])
    tx = float(m_2d[3, 0])
    ty = float(m_2d[3, 1])
    return np.array(
        [
            [cos_a, 0.0, sin_a, tx],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_a, 0.0, cos_a, -ty],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def interpolate_coord(prev: float, curr: float, alpha: float) -> float:
    """Linearly interpolate coordinate across sub-tic interval with factor alpha in [0, 1]."""
    return float(prev + alpha * (curr - prev))


def interpolate_angle(prev_deg: float, curr_deg: float, alpha: float) -> float:
    """Interpolate angle in degrees across shortest arc modular 0/360 boundary."""
    diff = (curr_deg - prev_deg + 180.0) % 360.0 - 180.0
    return float((prev_deg + diff * alpha) % 360.0)


def build_polyobj_mesh(
    po: Polyobj,
    textures: TextureRegistry | None = None,
    ctx: moderngl.Context | None = None,
    world_program: Any = None,
) -> PolyMesh:
    """Build local-coordinate wall quads and GPU batches for a polyobject.

    Wall geometry is generated relative to po.start_spot so that GPU transformation
    via u_model places and rotates the mesh at runtime.
    """
    mesh = PolyMesh(tag=po.tag)
    if not po.segs:
        return mesh

    # Determine vertical floor and ceiling span.
    # In vanilla Hexen (r_bsp.c:454), polyobjects inherit heights from po.subsector.sector.
    sector = None
    if po.subsector is not None:
        sector = po.subsector.sector
    elif po.segs and po.segs[0].sidedef is not None:
        sector = po.segs[0].sidedef.sector

    if sector is not None:
        z_bottom = float(sector.floor_height) / float(FRACUNIT)
        z_top = float(sector.ceiling_height) / float(FRACUNIT)
        light = float(sector.light_level)
    else:
        z_bottom = 0.0
        z_top = 128.0
        light = 160.0

    mesh.z_bottom = z_bottom
    mesh.z_top = z_top

    # Origin for local coordinates
    orig_x_fixed = po.start_spot.x if hasattr(po, "start_spot") else getattr(po, "start_spot_x", 0)
    orig_y_fixed = po.start_spot.y if hasattr(po, "start_spot") else getattr(po, "start_spot_y", 0)
    mesh.local_origin_x = orig_x_fixed / float(FRACUNIT)
    mesh.local_origin_y = orig_y_fixed / float(FRACUNIT)

    # Angle un-rotation if needed
    po_angle_bam = getattr(po, "angle", 0) & 0xFFFFFFFF
    cos_unrot = 1.0
    sin_unrot = 0.0
    if po_angle_bam != 0:
        ang_rad = po_angle_bam * (2.0 * math.pi) / float(1 << 32)
        cos_unrot = math.cos(-ang_rad)
        sin_unrot = math.sin(-ang_rad)

    num_segs = len(po.segs)
    has_orig_pts = len(po.original_pts) >= num_segs

    quad_vertex_arrays: list[np.ndarray] = []
    grouped_quads: dict[tuple[float, int, str], list[np.ndarray]] = {}

    for i, seg in enumerate(po.segs):
        if has_orig_pts:
            v1_lx = po.original_pts[i].x / float(FRACUNIT)
            v1_ly = po.original_pts[i].y / float(FRACUNIT)
            if num_segs > 1:
                next_i = (i + 1) % num_segs
                v2_lx = po.original_pts[next_i].x / float(FRACUNIT)
                v2_ly = po.original_pts[next_i].y / float(FRACUNIT)
            else:
                v2_lx = (seg.v2.x - orig_x_fixed) / float(FRACUNIT)
                v2_ly = (seg.v2.y - orig_y_fixed) / float(FRACUNIT)
        else:
            dx1 = (seg.v1.x - orig_x_fixed) / float(FRACUNIT)
            dy1 = (seg.v1.y - orig_y_fixed) / float(FRACUNIT)
            dx2 = (seg.v2.x - orig_x_fixed) / float(FRACUNIT)
            dy2 = (seg.v2.y - orig_y_fixed) / float(FRACUNIT)
            if po_angle_bam != 0:
                v1_lx = dx1 * cos_unrot - dy1 * sin_unrot
                v1_ly = dx1 * sin_unrot + dy1 * cos_unrot
                v2_lx = dx2 * cos_unrot - dy2 * sin_unrot
                v2_ly = dx2 * sin_unrot + dy2 * cos_unrot
            else:
                v1_lx, v1_ly = dx1, dy1
                v2_lx, v2_ly = dx2, dy2

        # 4 vertices per seg quad (for feature tests and collision/bounding checks)
        mesh.vertices.extend(
            [
                (v1_lx, v1_ly, z_bottom),
                (v1_lx, v1_ly, z_top),
                (v2_lx, v2_ly, z_top),
                (v2_lx, v2_ly, z_bottom),
            ]
        )

        wall_len = math.hypot(v2_lx - v1_lx, v2_ly - v1_ly)
        side = seg.sidedef
        line = seg.linedef

        tex_offset = (side.texture_offset / float(FRACUNIT)) if side else 0.0
        row_offset = (side.row_offset / float(FRACUNIT)) if side else 0.0
        u0 = tex_offset
        u1 = tex_offset + wall_len

        # Resolve wall texture
        tex_name_raw = ""
        if side is not None:
            tex_name_raw = side.mid_texture or side.top_texture or side.bottom_texture or ""
        tex_id, tex_name, tex_h = _resolve_texture(tex_name_raw, textures)

        span = z_top - z_bottom
        if line is not None and (line.flags & ML_DONTPEGBOTTOM):
            v1_uv = row_offset + tex_h
            v0_uv = v1_uv - span
        else:
            v0_uv = row_offset
            v1_uv = row_offset + span

        # 4 UV pairs corresponding to the 4 vertices
        mesh.uvs.extend(
            [
                (u0, v1_uv),
                (u0, v0_uv),
                (u1, v0_uv),
                (u1, v1_uv),
            ]
        )

        # Build 6 vertices (2 CCW triangles) for GPU rendering
        # Mapping: (lx, z, -ly) in local OpenGL space
        surface_type = (
            SURFACE_MASKED_WALL
            if (seg.back_sector is not None or (line and line.back_side is not None))
            else SURFACE_OPAQUE_WALL
        )

        tid = float(tex_id)
        tl = (v1_lx, z_top, -v1_ly, u0, v0_uv, light, surface_type, tid)
        bl = (v1_lx, z_bottom, -v1_ly, u0, v1_uv, light, surface_type, tid)
        br = (v2_lx, z_bottom, -v2_ly, u1, v1_uv, light, surface_type, tid)
        tr = (v2_lx, z_top, -v2_ly, u1, v0_uv, light, surface_type, tid)

        quad_verts = np.array([tl, bl, br, tl, br, tr], dtype=np.float32)
        quad_vertex_arrays.append(quad_verts)

        batch_key = (surface_type, tex_id, tex_name)
        grouped_quads.setdefault(batch_key, []).append(quad_verts)

    # Build draw batches
    offset = 0
    batch_arrays: list[np.ndarray] = []
    for (surf_type, tex_id, tex_name), quads in grouped_quads.items():
        count = 6 * len(quads)
        mesh.batches.append(
            DrawBatch(
                surface_type=surf_type,
                tex_id=tex_id,
                tex_name=tex_name,
                vertex_offset=offset,
                vertex_count=count,
            )
        )
        batch_arrays.extend(quads)
        offset += count

    if batch_arrays:
        mesh.raw_vertices = np.concatenate(batch_arrays, axis=0).astype(np.float32, copy=False)
        mesh.vertex_count = len(mesh.raw_vertices)
    else:
        mesh.raw_vertices = np.zeros((0, 8), dtype=np.float32)
        mesh.vertex_count = 0

    # Build GPU VBO and VAO if context provided
    if ctx is not None and len(mesh.raw_vertices) > 0:
        mesh.vbo = ctx.buffer(mesh.raw_vertices.tobytes())
        if world_program is not None:
            mesh.vao = ctx.vertex_array(
                world_program,
                [(mesh.vbo, WORLD_VERTEX_FORMAT, *WORLD_VERTEX_ATTRIBS)],
            )

    return mesh
