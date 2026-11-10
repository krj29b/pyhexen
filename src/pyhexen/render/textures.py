# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""GPU texture manager, animation translation tables, and world lighting shader (T4.2, T4.5).

Port of R_InitData / R_InitTextures / R_InitFlats / R_InitLightTables (r_data.c, r_main.c)
and P_InitFTAnims / P_AnimateSurfaces (p_anim.c).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from pyhexen.core.random import Random
from pyhexen.wad.animdefs import AnimDef, load_animdefs
from pyhexen.wad.flats import FlatRegistry
from pyhexen.wad.palette import load_colormap, load_playpal
from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.textures import TextureRegistry
from pyhexen.wad.wadfile import WadFile, WadStack

# Lighting constants from r_local.h / r_main.c
LIGHTLEVELS = 16
LIGHTSEGSHIFT = 4
MAXLIGHTSCALE = 48
LIGHTSCALESHIFT = 12
MAXLIGHTZ = 128
LIGHTZSHIFT = 20
NUMCOLORMAPS = 32
DISTMAP = 2

# Surface type constants matching world.vert / world.frag
SURFACE_WALL = 0.0
SURFACE_FLAT = 1.0
SURFACE_SKY = 2.0
SURFACE_MASKED_WALL = 3.0

WORLD_VERTEX_FORMAT = "3f 2f 1f 1f 1f"
WORLD_VERTEX_ATTRIBS = ("in_pos", "in_uv", "in_light", "in_surface_type", "in_tex_id")

_SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


def compute_colormap_row(
    light_level: int | float,
    depth: float,
    surface_type: int | float = SURFACE_WALL,
    extralight: int = 0,
    fixed_colormap: int = -1,
) -> int:
    """Compute the COLORMAP/FOGMAP row index (0..31) matching world.frag and r_main.c."""
    if fixed_colormap >= 0:
        return max(0, min(NUMCOLORMAPS - 1, int(fixed_colormap)))

    lightnum = max(0, min(LIGHTLEVELS - 1, (int(light_level) >> LIGHTSEGSHIFT) + extralight))
    startmap = (LIGHTLEVELS - 1 - lightnum) * 4

    if 0.5 < float(surface_type) < 1.5:
        z_index = max(0, min(MAXLIGHTZ - 1, int(max(depth, 1.0)) >> 4))
        flat_scale = 160 // (z_index + 1)
        diminish = flat_scale // DISTMAP
    else:
        wall_scale = max(0, min(MAXLIGHTSCALE - 1, int(2560.0 / max(depth, 1.0))))
        diminish = wall_scale // DISTMAP

    return max(0, min(NUMCOLORMAPS - 1, startmap - diminish))


@dataclass(slots=True)
class GpuTexture:
    """Uploaded 8-bit indexed texture + 8-bit opacity mask in GPU memory."""

    name: str
    width: int
    height: int
    index_tex: moderngl.Texture
    mask_tex: moderngl.Texture
    left: int = 0
    top: int = 0

    def bind(self, index_unit: int = 0, mask_unit: int = 1) -> None:
        """Bind index and mask textures to the given texture units."""
        self.index_tex.use(location=index_unit)
        self.mask_tex.use(location=mask_unit)

    def release(self) -> None:
        """Release underlying GPU texture objects."""
        self.index_tex.release()
        self.mask_tex.release()


@dataclass(slots=True)
class _ActiveAnim:
    """Runtime animation state for a single ANIMDEFS sequence (port of anim_t in p_anim.c)."""

    is_flat: bool
    base_index: int
    # Tuple of (target_index, tics, rand_min, rand_max)
    frames: tuple[tuple[int, int, int, int], ...]
    current_frame: int
    tics_remaining: int


class GpuTextureManager:
    """Manages GPU textures for palettes, colormaps, walls, flats, and sprites (T4.2).

    All indexed textures use ``R8`` with ``GL_NEAREST`` filtering and ``alignment=1``
    so the fragment shader can perform exact ``texelFetch`` lookups into ``PLAYPAL``
    and ``COLORMAP``/``FOGMAP``.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        wad: WadStack | WadFile | None = None,
        *,
        palettes: np.ndarray | None = None,
        colormap: np.ndarray | None = None,
        fogmap: np.ndarray | None = None,
        texture_registry: TextureRegistry | None = None,
        flat_registry: FlatRegistry | None = None,
        animdefs: list[AnimDef] | None = None,
    ) -> None:
        self.ctx = ctx
        self.wad = wad

        if palettes is None:
            if wad is None:
                raise ValueError("Either wad or palettes must be provided")
            palettes = load_playpal(wad)
        self.palettes: np.ndarray = np.ascontiguousarray(palettes, dtype=np.uint8)

        if colormap is None:
            if wad is None:
                raise ValueError("Either wad or colormap must be provided")
            colormap = load_colormap(wad, "COLORMAP")
        self.colormap: np.ndarray = np.ascontiguousarray(colormap, dtype=np.uint8)

        if fogmap is None:
            if wad is not None and wad.find("FOGMAP") is not None:
                fogmap = load_colormap(wad, "FOGMAP")
            else:
                fogmap = self.colormap
        self.fogmap: np.ndarray = np.ascontiguousarray(fogmap, dtype=np.uint8)

        if texture_registry is None and wad is not None:
            if wad.find("PNAMES") is not None and (
                wad.find("TEXTURE1") is not None or wad.find("TEXTURE2") is not None
            ):
                texture_registry = TextureRegistry(wad)
        self.texture_registry = texture_registry

        if flat_registry is None and wad is not None:
            if wad.find("F_START") is not None or wad.find("FF_START") is not None:
                flat_registry = FlatRegistry(wad)
        self.flat_registry = flat_registry

        # Upload PLAYPAL as (256, num_palettes) 3-component RGB texture
        num_pals = self.palettes.shape[0]
        self.palette_tex = self.ctx.texture(
            (256, num_pals), 3, data=self.palettes.tobytes(), alignment=1
        )
        self.palette_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.palette_tex.repeat_x = False
        self.palette_tex.repeat_y = False

        # Upload COLORMAP and FOGMAP as (256, rows) 1-component R8 textures
        self.colormap_tex = self.ctx.texture(
            (256, self.colormap.shape[0]), 1, data=self.colormap.tobytes(), alignment=1
        )
        self.colormap_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.colormap_tex.repeat_x = False
        self.colormap_tex.repeat_y = False

        self.fogmap_tex = self.ctx.texture(
            (256, self.fogmap.shape[0]), 1, data=self.fogmap.tobytes(), alignment=1
        )
        self.fogmap_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fogmap_tex.repeat_x = False
        self.fogmap_tex.repeat_y = False

        # Caches of uploaded GpuTextures
        self._wall_cache: dict[int, GpuTexture] = {}
        self._flat_cache: dict[int, GpuTexture] = {}
        self._picture_cache: dict[str, GpuTexture] = {}

        # Animation indirection tables (texturetranslation / flattranslation in r_data.c)
        num_walls = len(self.texture_registry) if self.texture_registry is not None else 0
        num_flats = len(self.flat_registry) if self.flat_registry is not None else 0
        self.wall_translation: list[int] = list(range(num_walls))
        self.flat_translation: list[int] = list(range(num_flats))

        # Parse and initialize ANIMDEFS sequences (P_InitFTAnims in p_anim.c)
        if animdefs is None and wad is not None and wad.find("ANIMDEFS") is not None:
            animdefs = load_animdefs(wad)
        self._active_anims: list[_ActiveAnim] = []
        self.animated_wall_indices: set[int] = set()
        self.animated_flat_indices: set[int] = set()
        if animdefs:
            self._init_anims(animdefs)

        self.smooth_filter: bool = False

    def set_filtering(self, smooth: bool) -> None:
        """Toggle linear filtering for texture mask/alpha sampling."""
        self.smooth_filter = smooth
        flt = (moderngl.LINEAR, moderngl.LINEAR) if smooth else (moderngl.NEAREST, moderngl.NEAREST)
        for cache in (self._wall_cache, self._flat_cache, self._picture_cache):
            for tex in cache.values():
                if hasattr(tex, "mask_tex") and tex.mask_tex is not None:
                    tex.mask_tex.filter = flt

    def _init_anims(self, animdefs: list[AnimDef]) -> None:
        """Port of P_InitFTAnims (p_anim.c): resolve ANIMDEFS into runtime state."""
        for anim in animdefs:
            if not anim.frames:
                continue
            if not anim.is_flat:
                if self.texture_registry is None:
                    continue
                base_idx = self.texture_registry.check_num_for_name(anim.name)
                if base_idx <= 0:
                    continue
                resolved: list[tuple[int, int, int, int]] = []
                valid = True
                for frame in anim.frames:
                    if isinstance(frame.index, int):
                        target = base_idx + frame.index - 1
                    else:
                        target = self.texture_registry.check_num_for_name(frame.index)
                    if target <= 0 or target >= len(self.texture_registry):
                        valid = False
                        break
                    resolved.append((target, frame.tics, frame.rand_min, frame.rand_max))
                if valid and resolved:
                    self._active_anims.append(
                        _ActiveAnim(
                            is_flat=False,
                            base_index=base_idx,
                            frames=tuple(resolved),
                            current_frame=len(resolved) - 1,
                            tics_remaining=1,
                        )
                    )
            else:
                if self.flat_registry is None:
                    continue
                base_idx = self.flat_registry.check_num_for_name(anim.name)
                if base_idx < 0:
                    continue
                resolved = []
                valid = True
                for frame in anim.frames:
                    if isinstance(frame.index, int):
                        target = base_idx + frame.index - 1
                    else:
                        target = self.flat_registry.check_num_for_name(frame.index)
                    if target < 0 or target >= len(self.flat_registry):
                        valid = False
                        break
                    resolved.append((target, frame.tics, frame.rand_min, frame.rand_max))
                if valid and resolved:
                    self._active_anims.append(
                        _ActiveAnim(
                            is_flat=True,
                            base_index=base_idx,
                            frames=tuple(resolved),
                            current_frame=len(resolved) - 1,
                            tics_remaining=1,
                        )
                    )
        self.animated_wall_indices = {a.base_index for a in self._active_anims if not a.is_flat}
        self.animated_flat_indices = {a.base_index for a in self._active_anims if a.is_flat}

    def update_animations(self, level_time: int = 0, rng: Random | None = None) -> None:
        """Advance texture/flat animation translation tables by one tic (P_AnimateSurfaces).

        Preserves exact ``P_Random`` consumption order from ``p_anim.c`` when random-duration
        frames transition.
        """
        _ = level_time
        for anim in self._active_anims:
            anim.tics_remaining -= 1
            if anim.tics_remaining <= 0:
                anim.current_frame = (anim.current_frame + 1) % len(anim.frames)
                target_idx, tics, rand_min, rand_max = anim.frames[anim.current_frame]
                if rand_max > 0:
                    r_val = rng.p_random() if rng is not None else 0
                    anim.tics_remaining = rand_min + (r_val % (rand_max - rand_min + 1))
                else:
                    anim.tics_remaining = max(1, tics)
                if anim.is_flat:
                    self.flat_translation[anim.base_index] = target_idx
                else:
                    self.wall_translation[anim.base_index] = target_idx

    def upload_indexed(
        self,
        name: str,
        indices: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        left: int = 0,
        top: int = 0,
        repeat: bool = True,
    ) -> GpuTexture:
        """Upload (H, W) uint8 palette indices and optional (H, W) bool/uint8 opacity mask."""
        idx_arr = np.ascontiguousarray(indices, dtype=np.uint8)
        height, width = idx_arr.shape
        if mask is None:
            mask_arr = np.full((height, width), 255, dtype=np.uint8)
        elif mask.dtype == np.bool_:
            mask_arr = np.where(mask, np.uint8(255), np.uint8(0))
            mask_arr = np.ascontiguousarray(mask_arr, dtype=np.uint8)
        else:
            mask_arr = np.ascontiguousarray(mask, dtype=np.uint8)

        index_tex = self.ctx.texture((width, height), 1, data=idx_arr.tobytes(), alignment=1)
        index_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        index_tex.repeat_x = repeat
        index_tex.repeat_y = repeat

        mask_tex = self.ctx.texture((width, height), 1, data=mask_arr.tobytes(), alignment=1)
        mask_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        mask_tex.repeat_x = repeat
        mask_tex.repeat_y = repeat

        return GpuTexture(
            name=name,
            width=width,
            height=height,
            index_tex=index_tex,
            mask_tex=mask_tex,
            left=left,
            top=top,
        )

    def get_wall_texture(self, index_or_name: int | str, *, translated: bool = True) -> GpuTexture:
        """Lazily upload and return a wall texture by index or name."""
        if self.texture_registry is None:
            raise RuntimeError("No TextureRegistry configured on GpuTextureManager")
        if isinstance(index_or_name, str):
            tex_idx = self.texture_registry.num_for_name(index_or_name)
        else:
            tex_idx = int(index_or_name)

        if translated and 0 <= tex_idx < len(self.wall_translation):
            tex_idx = self.wall_translation[tex_idx]

        cached = self._wall_cache.get(tex_idx)
        if cached is not None:
            return cached

        tex = self.texture_registry.get(tex_idx)
        if tex.width <= 0 or tex.height <= 0:
            # Dummy 1x1 transparent texture for slot 0 ("-")
            gpu_tex = self.upload_indexed(
                "-",
                np.zeros((1, 1), dtype=np.uint8),
                np.zeros((1, 1), dtype=np.uint8),
                repeat=True,
            )
        else:
            gpu_tex = self.upload_indexed(tex.name, tex.indices, tex.mask, repeat=True)
        self._wall_cache[tex_idx] = gpu_tex
        return gpu_tex

    def get_flat_texture(self, index_or_name: int | str, *, translated: bool = True) -> GpuTexture:
        """Lazily upload and return a 64x64 flat texture by index or name."""
        if self.flat_registry is None:
            raise RuntimeError("No FlatRegistry configured on GpuTextureManager")
        if isinstance(index_or_name, str):
            flat_idx = self.flat_registry.num_for_name(index_or_name)
        else:
            flat_idx = int(index_or_name)

        if translated and 0 <= flat_idx < len(self.flat_translation):
            flat_idx = self.flat_translation[flat_idx]

        cached = self._flat_cache.get(flat_idx)
        if cached is not None:
            return cached

        flat = self.flat_registry.get(flat_idx)
        indices = flat.indices[:64, :64]
        gpu_tex = self.upload_indexed(flat.name, indices, None, repeat=True)
        self._flat_cache[flat_idx] = gpu_tex
        return gpu_tex

    def upload_picture(self, name: str, pic: Picture) -> GpuTexture:
        """Upload a decoded Picture patch/sprite to GPU memory (cached by name)."""
        key = name.upper()
        cached = self._picture_cache.get(key)
        if cached is not None:
            return cached
        gpu_tex = self.upload_indexed(
            key,
            pic.indices,
            pic.mask,
            left=pic.left,
            top=pic.top,
            repeat=False,
        )
        self._picture_cache[key] = gpu_tex
        return gpu_tex

    def get_patch_texture(self, lump_or_name: int | str) -> GpuTexture:
        """Decode and upload a patch/sprite lump from the WAD."""
        if self.wad is None:
            raise RuntimeError("No WAD configured on GpuTextureManager")
        if isinstance(lump_or_name, str):
            key = lump_or_name.upper()
            cached = self._picture_cache.get(key)
            if cached is not None:
                return cached
            raw = self.wad.get_bytes(key)
        else:
            key = f"LUMP_{lump_or_name}"
            cached = self._picture_cache.get(key)
            if cached is not None:
                return cached
            raw = self.wad.get_bytes(lump_or_name)

        pic = decode_picture(raw)
        return self.upload_picture(key, pic)

    def bind_lookup_tables(
        self,
        palette_unit: int = 2,
        colormap_unit: int = 3,
        fogmap_unit: int = 4,
    ) -> None:
        """Bind PLAYPAL, COLORMAP, and FOGMAP textures to their shader texture units."""
        self.palette_tex.use(location=palette_unit)
        self.colormap_tex.use(location=colormap_unit)
        self.fogmap_tex.use(location=fogmap_unit)

    def release(self) -> None:
        """Release all uploaded GPU textures."""
        for tex in self._wall_cache.values():
            tex.release()
        self._wall_cache.clear()
        for tex in self._flat_cache.values():
            tex.release()
        self._flat_cache.clear()
        for tex in self._picture_cache.values():
            tex.release()
        self._picture_cache.clear()
        self.palette_tex.release()
        self.colormap_tex.release()
        self.fogmap_tex.release()


# 4x4 float32 identity matrix bytes (column-major in OpenGL)
_IDENTITY_4X4_BYTES: bytes = np.eye(4, dtype=np.float32).tobytes()


class WorldShaderProgram:
    """Wrapper around world.vert + world.frag for 8-bit palette & colormap lighting (T4.5)."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        vert_src = (_SHADERS_DIR / "world.vert").read_text(encoding="utf-8")
        frag_src = (_SHADERS_DIR / "world.frag").read_text(encoding="utf-8")
        self.program = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # Default uniform values
        self.program["u_index_tex"].value = 0
        self.program["u_mask_tex"].value = 1
        self.program["u_palette"].value = 2
        self.program["u_colormap"].value = 3
        self.program["u_fogmap"].value = 4
        if "u_light_override" in self.program:
            self.program["u_light_override"].value = -1.0

        # Cache uniform references for fast access
        self._u_palette_index = self.program.get("u_palette_index", None)
        self._u_extralight = self.program.get("u_extralight", None)
        self._u_fixed_colormap = self.program.get("u_fixed_colormap", None)
        self._u_use_fogmap = self.program.get("u_use_fogmap", None)
        self._u_gamma = self.program.get("u_gamma", None)
        self._u_alpha = self.program.get("u_alpha", None)
        self._u_tex_size = self.program.get("u_tex_size", None)
        self._u_model = self.program.get("u_model", None)
        self._u_view = self.program.get("u_view", None)
        self._u_proj = self.program.get("u_proj", None)
        self._u_light_override = self.program.get("u_light_override", None)
        self._model_is_identity: bool = False

        self._last_bound_tex: GpuTexture | None = None
        self._last_tex_w: int = -1
        self._last_tex_h: int = -1
        self._last_palette_index: int = -999999
        self._last_extralight: int = -999999
        self._last_fixed_colormap: int = -999999
        self._last_use_fogmap: int = -999999
        self._last_gamma: float = -999999.0
        self._last_alpha: float = -999999.0

        identity = np.eye(4, dtype=np.float32)
        self.set_model(identity)
        self.set_camera(identity, identity)
        self.set_lighting()

    def set_model(self, model: np.ndarray | bytes | None = None) -> None:
        """Update model transform matrix (converts 4x4 numpy or raw bytes to GL)."""
        if self._u_model is None:
            return
        if model is None:
            if not self._model_is_identity:
                self._u_model.write(_IDENTITY_4X4_BYTES)
                self._model_is_identity = True
            return
        self._model_is_identity = False
        if isinstance(model, bytes):
            self._u_model.write(model)
        else:
            self._u_model.write(np.ascontiguousarray(model.T, dtype=np.float32).tobytes())

    def set_light_override(self, light_level: float = -1.0) -> None:
        """Override vertex lighting if light_level >= 0.0, or use per-vertex light if -1.0."""
        if self._u_light_override is not None:
            self._u_light_override.value = float(light_level)

    def set_camera(self, view: np.ndarray, proj: np.ndarray) -> None:
        """Update view and projection matrices (converts row-major 4x4 numpy to GL column-major)."""
        if self._u_view is not None:
            self._u_view.write(np.ascontiguousarray(view.T, dtype=np.float32).tobytes())
        if self._u_proj is not None:
            self._u_proj.write(np.ascontiguousarray(proj.T, dtype=np.float32).tobytes())

    def reset_state_cache(self) -> None:
        """Reset cached uniforms and texture bindings."""
        self._last_bound_tex = None
        self._last_tex_w = -1
        self._last_tex_h = -1
        self._last_palette_index = -999999
        self._last_extralight = -999999
        self._last_fixed_colormap = -999999
        self._last_use_fogmap = -999999
        self._last_gamma = -999999.0
        self._last_alpha = -999999.0

    def set_lighting(
        self,
        *,
        palette_index: int = 0,
        extralight: int = 0,
        fixed_colormap: int = -1,
        use_fogmap: bool = False,
        gamma: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        """Configure palette flash index, extralight, fixedcolormap, fogmap, gamma, and alpha."""
        if palette_index != self._last_palette_index and self._u_palette_index is not None:
            self._u_palette_index.value = int(palette_index)
            self._last_palette_index = palette_index
        if extralight != self._last_extralight and self._u_extralight is not None:
            self._u_extralight.value = int(extralight)
            self._last_extralight = extralight
        if fixed_colormap != self._last_fixed_colormap and self._u_fixed_colormap is not None:
            self._u_fixed_colormap.value = int(fixed_colormap)
            self._last_fixed_colormap = fixed_colormap
        fog_int = 1 if use_fogmap else 0
        if fog_int != self._last_use_fogmap and self._u_use_fogmap is not None:
            self._u_use_fogmap.value = fog_int
            self._last_use_fogmap = fog_int
        if gamma != self._last_gamma and self._u_gamma is not None:
            self._u_gamma.value = float(gamma)
            self._last_gamma = gamma
        if self._u_alpha is not None and alpha != self._last_alpha:
            self._u_alpha.value = float(alpha)
            self._last_alpha = alpha

    def bind_surface_texture(
        self,
        tex_mgr: GpuTextureManager,
        gpu_tex: GpuTexture,
        *,
        bind_tables: bool = True,
    ) -> None:
        """Bind lookup tables and the given surface texture + its dimensions."""
        if bind_tables:
            tex_mgr.bind_lookup_tables(palette_unit=2, colormap_unit=3, fogmap_unit=4)
        if gpu_tex is not self._last_bound_tex:
            gpu_tex.bind(index_unit=0, mask_unit=1)
            w = gpu_tex.width
            h = gpu_tex.height
            if (w != self._last_tex_w or h != self._last_tex_h) and self._u_tex_size is not None:
                self._u_tex_size.value = (float(w), float(h))
                self._last_tex_w = w
                self._last_tex_h = h
            self._last_bound_tex = gpu_tex

    def create_vao(
        self,
        vbo: moderngl.Buffer,
        ibo: moderngl.Buffer | None = None,
    ) -> moderngl.VertexArray:
        """Create a VAO for the shared 8-float world vertex format (3f 2f 1f 1f 1f)."""
        return self.ctx.vertex_array(
            self.program,
            [(vbo, WORLD_VERTEX_FORMAT, *WORLD_VERTEX_ATTRIBS)],
            index_buffer=ibo,
        )

    def release(self) -> None:
        """Release the compiled shader program."""
        self.program.release()
