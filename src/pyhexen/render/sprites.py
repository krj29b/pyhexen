# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sprite rotation selection, billboard quad generation, and batch rendering (T5.5).

Port of R_ProjectSprite and R_DrawVisSprite (r_things.c).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import moderngl
import numpy as np

from pyhexen.core.angles import ANG45, point_to_angle2
from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div, to_uint32
from pyhexen.render.camera import Camera, rad_to_bam
from pyhexen.render.textures import (
    SURFACE_MASKED_WALL,
    GpuTexture,
    GpuTextureManager,
    WorldShaderProgram,
)
from pyhexen.sim.info_data import MOBJINFO, STATES, MobjFlag, MobjFlag2
from pyhexen.sim.mobj import (
    DOOMEDNUM_TO_MOBJTYPE,
    FLOAT_BOB_OFFSETS,
    MTF_FIGHTER,
    MTF_GSINGLE,
    MTF_NORMAL,
    Mobj,
)
from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.sprites import SpriteFrame, SpriteRegistry
from pyhexen.wad.wadfile import LumpInfo

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import Level

# Frame flags from r_things.c / info.h
FF_FULLBRIGHT: int = 0x8000
FF_FRAMEMASK: int = 0x7FFF

# Default alpha for shadow / translucent sprites (TINTTAB approximation in GL)
SHADOW_ALPHA: float = 0.45


def select_sprite_rotation(
    sprite_frame: SpriteFrame,
    mobj_angle: int,
    view_x: int,
    view_y: int,
    mobj_x: int,
    mobj_y: int,
) -> tuple[LumpInfo, bool]:
    """Select the (LumpInfo, flip) pair for a sprite frame viewed from (view_x, view_y).

    Port of rotation selection in R_ProjectSprite (r_things.c):
      ang = R_PointToAngle(thing->x, thing->y);
      rot = (ang - thing->angle + (unsigned)(ANG45 / 2) * 9) >> 29;
    Coordinates `(view_x, view_y)` and `(mobj_x, mobj_y)` are 16.16 fixed-point,
    and `mobj_angle` is a 32-bit unsigned BAM angle.
    """
    if sprite_frame.rotate == 0:
        return sprite_frame.lumps[0], sprite_frame.flip[0]

    ang = point_to_angle2(view_x, view_y, mobj_x, mobj_y)
    rot = ((ang - (mobj_angle & 0xFFFFFFFF) + (ANG45 // 2) * 9) & 0xFFFFFFFF) >> 29
    return sprite_frame.lumps[rot], sprite_frame.flip[rot]


def _extract_interpolated_pos(
    mobj_or_pos: Mobj | Sequence[float | int],
    alpha: float,
    level_time: int = 0,
) -> tuple[float, float, float]:
    """Extract interpolated map-space (x, y, z) in float map units."""
    a = max(0.0, min(1.0, float(alpha)))
    if isinstance(mobj_or_pos, Mobj):
        mo = mobj_or_pos
        x_prev = float(mo.prev_x)
        y_prev = float(mo.prev_y)
        z_prev = float(mo.prev_z)
        ix = (x_prev + (float(mo.x) - x_prev) * a) / 65536.0
        iy = (y_prev + (float(mo.y) - y_prev) * a) / 65536.0
        iz_fixed = z_prev + (float(mo.z) - z_prev) * a - float(mo.floorclip)
        if mo.flags2 & MobjFlag2.MF2_FLOATBOB:
            bob_idx = (level_time + mo.health) & 63
            iz_fixed += float(FLOAT_BOB_OFFSETS[bob_idx])
        return ix, iy, iz_fixed / 65536.0

    seq = mobj_or_pos
    return float(seq[0]), float(seq[1]), float(seq[2])


_SPRITE_QUAD_STRUCT: struct.Struct = struct.Struct("<48f")


def build_sprite_quad_bytes(
    mobj_or_pos: Mobj | Sequence[float | int],
    pic: Picture,
    camera: Camera,
    light_level: float,
    fullbright: bool,
    flip: bool,
    alpha: float = 1.0,
    shadow: bool = False,
    *,
    tex_id: float = 0.0,
    level_time: int = 0,
    cam_right: tuple[float, float] | None = None,
    pos: tuple[float, float, float] | None = None,
) -> bytes:
    """Build a camera-facing billboard quad (6 vertices x 8 floats) as raw bytes for a sprite."""
    _ = shadow
    if pos is not None:
        ix, iy, iz = pos
    else:
        ix, iy, iz = _extract_interpolated_pos(mobj_or_pos, alpha, level_time=level_time)

    if cam_right is not None:
        rx, ry = cam_right
    else:
        _, _, _, iyaw, _ = camera.interpolated_state(alpha)
        rx = math.sin(iyaw)
        ry = -math.cos(iyaw)

    w = float(pic.width)
    h = float(pic.height)
    left_ofs = float(pic.left)
    top_ofs = float(pic.top)

    if flip:
        ofs_l = -(w - left_ofs)
        ofs_r = left_ofs
        u0 = w
        u1 = 0.0
    else:
        ofs_l = -left_ofs
        ofs_r = w - left_ofs
        u0 = 0.0
        u1 = w

    x0 = ix + rx * ofs_l
    y0 = iy + ry * ofs_l
    x1 = ix + rx * ofs_r
    y1 = iy + ry * ofs_r

    z_top = iz + top_ofs
    z_bot = z_top - h

    v_top = 0.0
    v_bot = h

    eff_light = 256.0 if fullbright else float(light_level)
    surf = float(SURFACE_MASKED_WALL)
    tid = float(tex_id)

    ny0 = -y0
    ny1 = -y1

    return _SPRITE_QUAD_STRUCT.pack(
        x0,
        z_bot,
        ny0,
        u0,
        v_bot,
        eff_light,
        surf,
        tid,
        x1,
        z_bot,
        ny1,
        u1,
        v_bot,
        eff_light,
        surf,
        tid,
        x1,
        z_top,
        ny1,
        u1,
        v_top,
        eff_light,
        surf,
        tid,
        x0,
        z_bot,
        ny0,
        u0,
        v_bot,
        eff_light,
        surf,
        tid,
        x1,
        z_top,
        ny1,
        u1,
        v_top,
        eff_light,
        surf,
        tid,
        x0,
        z_top,
        ny0,
        u0,
        v_top,
        eff_light,
        surf,
        tid,
    )


def build_sprite_quad(
    mobj_or_pos: Mobj | Sequence[float | int],
    pic: Picture,
    camera: Camera,
    light_level: float,
    fullbright: bool,
    flip: bool,
    alpha: float = 1.0,
    shadow: bool = False,
    *,
    tex_id: float = 0.0,
    level_time: int = 0,
) -> np.ndarray:
    """Build a camera-facing billboard quad (6 vertices x 8 floats) for a sprite."""
    raw = build_sprite_quad_bytes(
        mobj_or_pos,
        pic,
        camera,
        light_level,
        fullbright,
        flip,
        alpha,
        shadow,
        tex_id=tex_id,
        level_time=level_time,
    )
    return np.frombuffer(raw, dtype=np.float32).reshape(6, 8)


@dataclass(slots=True)
class _TranslucentSpriteItem:
    dist_sq: float
    gpu_tex: GpuTexture
    quad: bytes | np.ndarray
    alpha_val: float


class SpriteRenderer:
    """Collects, sorts, batches, and renders world sprites (T5.5)."""

    def __init__(
        self,
        ctx: moderngl.Context,
        sprite_registry: SpriteRegistry,
        tex_mgr: GpuTextureManager,
    ) -> None:
        self.ctx = ctx
        self.sprite_registry = sprite_registry
        self.tex_mgr = tex_mgr
        self._pic_cache: dict[int, tuple[Picture, GpuTexture]] = {}
        self._dynamic_vbo: moderngl.Buffer | None = None
        self._dynamic_vao: moderngl.VertexArray | None = None

    def _get_sprite_picture_and_tex(self, lump: LumpInfo) -> tuple[Picture, GpuTexture]:
        lump_id = id(lump)
        cached = self._pic_cache.get(lump_id)
        if cached is not None:
            return cached

        if self.tex_mgr.wad is not None:
            raw = self.tex_mgr.wad.get_bytes(lump)
        elif lump._wad is not None:
            raw = lump._wad.get_bytes(lump)
        else:
            raise RuntimeError(f"Cannot read sprite lump {lump.name!r} without WAD")

        pic = decode_picture(raw)
        key = f"{lump.source}:{lump.index}:{lump.name}"
        gpu_tex = self.tex_mgr.upload_indexed(
            key,
            pic.indices,
            pic.mask,
            left=pic.left,
            top=pic.top,
            repeat=False,
        )
        self._pic_cache[lump_id] = (pic, gpu_tex)
        return pic, gpu_tex

    def _build_static_mobjs_from_level(self, level: Level) -> list[Mobj]:
        """Build lightweight Mobj proxies from `level.things` when running without GameSim."""
        mobjs: list[Mobj] = []
        for mthing in level.things:
            options = mthing.flags
            if not (options & MTF_GSINGLE):
                continue
            if not (options & MTF_FIGHTER) and not (options & (MTF_FIGHTER << 1)):
                pass
            if not (options & (MTF_NORMAL | 1 | 4)):
                continue
            mtype = DOOMEDNUM_TO_MOBJTYPE.get(mthing.type)
            if mtype is None:
                continue
            info = MOBJINFO[mtype]
            st = STATES[info.spawnstate]
            x = mthing.x << FRACBITS
            y = mthing.y << FRACBITS
            ss = level.point_in_subsector(x, y)
            sec = ss.sector
            if info.flags & MobjFlag.MF_SPAWNCEILING:
                z = sec.ceiling_height - info.height - (mthing.z << FRACBITS)
            else:
                z = sec.floor_height + (mthing.z << FRACBITS)
            angle = to_uint32(c_div(mthing.angle << 8, 360) << 24)
            mo = Mobj(
                x=x,
                y=y,
                z=z,
                prev_x=x,
                prev_y=y,
                prev_z=z,
                angle=angle,
                prev_angle=angle,
                sprite=st.sprite,
                frame=st.frame,
                radius=info.radius,
                height=info.height,
                type=mtype,
                info=info,
                tics=st.tics,
                state_num=info.spawnstate,
                state=st,
                flags=info.flags,
                flags2=info.flags2,
                subsector=ss,
                sector=sec,
                floorz=sec.floor_height,
                ceilingz=sec.ceiling_height,
            )
            mobjs.append(mo)
        return mobjs

    def collect_batches(
        self,
        camera: Camera,
        *,
        sim: GameSim | None = None,
        level: Level | None = None,
        mobjs: Sequence[Mobj] | None = None,
        alpha: float = 1.0,
        use_fullbright: bool = True,
    ) -> tuple[list[tuple[GpuTexture, np.ndarray]], list[_TranslucentSpriteItem]]:
        """Collect opaque texture batches and back-to-front sorted translucent sprites."""
        if mobjs is not None:
            source_mobjs = mobjs
            level_time = sim.level_time if sim is not None else 0
        elif sim is not None:
            source_mobjs = sim.thinkers
            level_time = sim.level_time
        elif level is not None:
            source_mobjs = self._build_static_mobjs_from_level(level)
            level_time = 0
        else:
            return [], []

        cam_x, cam_y, _, iyaw, _ = camera.interpolated_state(alpha)
        view_x_fixed = int(round(cam_x * FRACUNIT))
        view_y_fixed = int(round(cam_y * FRACUNIT))

        cam_fx = math.cos(iyaw)
        cam_fy = math.sin(iyaw)
        cam_rx = math.sin(iyaw)
        cam_ry = -math.cos(iyaw)
        fov_deg = camera.fov_x_deg if hasattr(camera, "fov_x_deg") else 45.0
        tan_fov = math.tan(math.radians(fov_deg * 0.5)) * 1.35
        inv_frac = 1.0 / 65536.0
        is_alpha_one = alpha >= 1.0 or alpha <= 0.0

        opaque_by_tex: dict[int, tuple[GpuTexture, list[bytes]]] = {}
        translucent_list: list[_TranslucentSpriteItem] = []

        for mo in source_mobjs:
            if not isinstance(mo, Mobj) or mo.removed:
                continue
            if mo.flags2 & MobjFlag2.MF2_DONTDRAW:
                continue

            # Frustum culling: fast 2D test against camera plane
            if is_alpha_one:
                mx = float(mo.x) * inv_frac
                my = float(mo.y) * inv_frac
            else:
                x_prev = float(mo.prev_x)
                y_prev = float(mo.prev_y)
                mx = (x_prev + (float(mo.x) - x_prev) * alpha) * inv_frac
                my = (y_prev + (float(mo.y) - y_prev) * alpha) * inv_frac

            dx = mx - cam_x
            dy = my - cam_y
            depth = dx * cam_fx + dy * cam_fy
            if depth < -128.0:
                continue
            if abs(dx * cam_rx + dy * cam_ry) > depth * tan_fov + 128.0:
                continue

            if mo.sprite < 0 or mo.sprite >= len(self.sprite_registry):
                continue
            sprdef = self.sprite_registry.get(mo.sprite)
            frame_idx = mo.frame & FF_FRAMEMASK
            if frame_idx < 0 or frame_idx >= len(sprdef.frames):
                continue
            sprframe = sprdef.frames[frame_idx]

            lump, flip = select_sprite_rotation(
                sprframe,
                mo.angle,
                view_x_fixed,
                view_y_fixed,
                mo.x,
                mo.y,
            )
            pic, gpu_tex = self._get_sprite_picture_and_tex(lump)
            if pic.width <= 0 or pic.height <= 0:
                continue

            sec = mo.sector
            if sec is None and mo.subsector is not None:
                sec = mo.subsector.sector
            light_level = float(sec.light_level) if sec is not None else 160.0
            fullbright = bool(use_fullbright and (mo.frame & FF_FULLBRIGHT))
            is_shadow = bool(mo.flags & (MobjFlag.MF_SHADOW | MobjFlag.MF_ALTSHADOW))

            if is_alpha_one:
                iz_fixed = float(mo.z - mo.floorclip)
            else:
                z_prev = float(mo.prev_z)
                iz_fixed = z_prev + (float(mo.z) - z_prev) * alpha - float(mo.floorclip)
            if mo.flags2 & MobjFlag2.MF2_FLOATBOB:
                bob_idx = (level_time + mo.health) & 63
                iz_fixed += float(FLOAT_BOB_OFFSETS[bob_idx])
            pos = (mx, my, iz_fixed * inv_frac)

            w = float(pic.width)
            h = float(pic.height)
            left_ofs = float(pic.left)
            top_ofs = float(pic.top)

            if flip:
                ofs_l = -(w - left_ofs)
                ofs_r = left_ofs
                u0 = w
                u1 = 0.0
            else:
                ofs_l = -left_ofs
                ofs_r = w - left_ofs
                u0 = 0.0
                u1 = w

            x0 = mx + cam_rx * ofs_l
            y0 = my + cam_ry * ofs_l
            x1 = mx + cam_rx * ofs_r
            y1 = my + cam_ry * ofs_r

            z_top = pos[2] + top_ofs
            z_bot = z_top - h

            eff_light = 256.0 if fullbright else light_level
            ny0 = -y0
            ny1 = -y1

            quad_bytes = _SPRITE_QUAD_STRUCT.pack(
                x0,
                z_bot,
                ny0,
                u0,
                h,
                eff_light,
                3.0,
                0.0,
                x1,
                z_bot,
                ny1,
                u1,
                h,
                eff_light,
                3.0,
                0.0,
                x1,
                z_top,
                ny1,
                u1,
                0.0,
                eff_light,
                3.0,
                0.0,
                x0,
                z_bot,
                ny0,
                u0,
                h,
                eff_light,
                3.0,
                0.0,
                x1,
                z_top,
                ny1,
                u1,
                0.0,
                eff_light,
                3.0,
                0.0,
                x0,
                z_top,
                ny0,
                u0,
                0.0,
                eff_light,
                3.0,
                0.0,
            )

            if is_shadow:
                dist_sq = (pos[0] - cam_x) ** 2 + (pos[1] - cam_y) ** 2
                translucent_list.append(
                    _TranslucentSpriteItem(
                        dist_sq=dist_sq,
                        gpu_tex=gpu_tex,
                        quad=quad_bytes,
                        alpha_val=SHADOW_ALPHA,
                    )
                )
            else:
                tex_key = id(gpu_tex)
                if tex_key not in opaque_by_tex:
                    opaque_by_tex[tex_key] = (gpu_tex, [quad_bytes])
                else:
                    opaque_by_tex[tex_key][1].append(quad_bytes)

        opaque_batches: list[tuple[GpuTexture, bytes, int]] = [
            (gtex, b"".join(quads), len(quads) * 6) for gtex, quads in opaque_by_tex.values()
        ]
        # Sort translucent sprites back-to-front (largest distance first)
        translucent_list.sort(key=lambda item: item.dist_sq, reverse=True)
        return opaque_batches, translucent_list

    def _get_dynamic_vao(
        self,
        shader: WorldShaderProgram,
        data_bytes: bytes,
    ) -> moderngl.VertexArray:
        size = len(data_bytes)
        if self._dynamic_vbo is None or self._dynamic_vbo.size < size:
            if self._dynamic_vao is not None:
                self._dynamic_vao.release()
            if self._dynamic_vbo is not None:
                self._dynamic_vbo.release()
            alloc_size = max(size, 65536)
            self._dynamic_vbo = self.ctx.buffer(reserve=alloc_size, dynamic=True)
            self._dynamic_vao = shader.create_vao(self._dynamic_vbo)
        self._dynamic_vbo.write(data_bytes)
        return self._dynamic_vao

    def render_opaque(
        self,
        shader: WorldShaderProgram,
        opaque_batches: Sequence[Any],
        *,
        palette_index: int = 0,
        extralight: int = 0,
        fixed_colormap: int = -1,
        use_fogmap: bool = False,
        gamma: float = 1.0,
    ) -> None:
        """Render batched opaque sprites using the world shader."""
        if not opaque_batches:
            return
        shader.set_lighting(
            palette_index=palette_index,
            extralight=extralight,
            fixed_colormap=fixed_colormap,
            use_fogmap=use_fogmap,
            gamma=gamma,
            alpha=1.0,
        )
        for item in opaque_batches:
            gpu_tex = item[0]
            data = item[1]
            vert_count = item[2] if len(item) > 2 else (len(data) // 32)
            raw_bytes = (
                data
                if isinstance(data, bytes)
                else np.ascontiguousarray(data, dtype=np.float32).tobytes()
            )
            vao = self._get_dynamic_vao(shader, raw_bytes)
            shader.bind_surface_texture(self.tex_mgr, gpu_tex)
            vao.render(mode=moderngl.TRIANGLES, vertices=vert_count)

    def render_translucent(
        self,
        shader: WorldShaderProgram,
        translucent_items: Sequence[_TranslucentSpriteItem],
        *,
        palette_index: int = 0,
        extralight: int = 0,
        fixed_colormap: int = -1,
        use_fogmap: bool = False,
        gamma: float = 1.0,
    ) -> None:
        """Render back-to-front sorted translucent/shadow sprites with alpha blending."""
        if not translucent_items:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.depth_mask = False
        try:
            for item in translucent_items:
                shader.set_lighting(
                    palette_index=palette_index,
                    extralight=extralight,
                    fixed_colormap=fixed_colormap,
                    use_fogmap=use_fogmap,
                    gamma=gamma,
                    alpha=item.alpha_val,
                )
                if isinstance(item.quad, bytes):
                    raw_bytes = item.quad
                    num_verts = 6
                else:
                    raw_bytes = np.ascontiguousarray(item.quad, dtype=np.float32).tobytes()
                    num_verts = len(item.quad)
                vao = self._get_dynamic_vao(shader, raw_bytes)
                shader.bind_surface_texture(self.tex_mgr, item.gpu_tex)
                vao.render(mode=moderngl.TRIANGLES, vertices=num_verts)
        finally:
            self.ctx.depth_mask = True
            self.ctx.disable(moderngl.BLEND)
            shader.set_lighting(
                palette_index=palette_index,
                extralight=extralight,
                fixed_colormap=fixed_colormap,
                use_fogmap=use_fogmap,
                gamma=gamma,
                alpha=1.0,
            )

    def release(self) -> None:
        """Release cached sprite GPU textures and dynamic buffers."""
        if self._dynamic_vao is not None:
            self._dynamic_vao.release()
            self._dynamic_vao = None
        if self._dynamic_vbo is not None:
            self._dynamic_vbo.release()
            self._dynamic_vbo = None
        for _, gpu_tex in self._pic_cache.values():
            gpu_tex.release()
        self._pic_cache.clear()


__all__ = [
    "FF_FRAMEMASK",
    "FF_FULLBRIGHT",
    "SHADOW_ALPHA",
    "SpriteRenderer",
    "build_sprite_quad",
    "rad_to_bam",
    "select_sprite_rotation",
]
