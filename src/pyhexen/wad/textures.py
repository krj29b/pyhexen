# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Composite wall texture loader and lazy compositor (PNAMES + TEXTURE1/TEXTURE2).

Port of R_InitTextures, R_GenerateLookup, and R_GenerateComposite (r_data.c).
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from pyhexen.wad.picture import Picture, decode_picture
from pyhexen.wad.wadfile import LumpInfo, WadError, WadFile, WadStack, normalize_lump_name

_MAPTEXTURE_HEADER = struct.Struct("<8si2hih")
_MAPPATCH_STRUCT = struct.Struct("<5h")


@dataclass(slots=True, frozen=True)
class TexturePatch:
    """Placement of a single patch within a composite wall texture.

    Port of mappatch_t / texpatch_t (xddefs.h, r_data.c).
    """

    origin_x: int
    origin_y: int
    patch_index: int
    stepdir: int = 1
    colormap: int = 0


@dataclass(slots=True)
class Texture:
    """Composed wall texture with 8-bit palette indices and transparency mask.

    Port of texture_t (r_data.c).
    """

    name: str
    width: int
    height: int
    indices: np.ndarray  # uint8[height, width]
    mask: np.ndarray  # bool[height, width]
    patches: tuple[TexturePatch, ...] = ()


@dataclass(slots=True, frozen=True)
class _TextureHeader:
    name: str
    masked: bool
    width: int
    height: int
    patches: tuple[TexturePatch, ...]


class TextureRegistry:
    """Parses PNAMES and TEXTURE1/TEXTURE2 and lazily composes wall textures.

    Index 0 is reserved for the "-" ("no texture") entry, matching
    R_CheckTextureNumForName / R_TextureNumForName in r_data.c.
    """

    def __init__(self, wad: WadStack | WadFile) -> None:
        self._wad = wad
        self.pnames: tuple[str, ...] = self._load_pnames()
        self._patch_lumps: list[LumpInfo | None] = [
            wad.find(pname) if pname else None for pname in self.pnames
        ]
        self._patch_cache: dict[int, Picture] = {}

        self._headers: list[_TextureHeader] = [
            _TextureHeader(name="-", masked=False, width=0, height=0, patches=())
        ]
        self._by_name: dict[str, int] = {"-": 0}

        self._load_texture_lump("TEXTURE1", required=True)
        if wad.find("TEXTURE2") is not None:
            self._load_texture_lump("TEXTURE2", required=False)

        self._cache: dict[int, Texture] = {
            0: Texture(
                name="-",
                width=0,
                height=0,
                indices=np.zeros((0, 0), dtype=np.uint8),
                mask=np.zeros((0, 0), dtype=bool),
                patches=(),
            )
        }

    def _load_pnames(self) -> tuple[str, ...]:
        raw = self._wad.get_bytes("PNAMES")
        if len(raw) < 4:
            raise ValueError(f"PNAMES lump too short ({len(raw)} bytes < 4)")
        (num_patches,) = struct.unpack_from("<i", raw, 0)
        if num_patches < 0 or 4 + num_patches * 8 > len(raw):
            raise ValueError(f"PNAMES count {num_patches} exceeds lump size {len(raw)}")
        names: list[str] = []
        for i in range(num_patches):
            raw_name = raw[4 + i * 8 : 4 + (i + 1) * 8]
            names.append(normalize_lump_name(raw_name))
        return tuple(names)

    def _load_texture_lump(self, lump_name: str, *, required: bool) -> None:
        lump = self._wad.find(lump_name)
        if lump is None:
            if required:
                self._wad.require(lump_name)
            return
        raw = self._wad.get_bytes(lump)
        max_off = len(raw)
        if max_off < 4:
            raise ValueError(f"{lump_name} lump too short ({max_off} bytes < 4)")

        (num_tex,) = struct.unpack_from("<i", raw, 0)
        if num_tex < 0 or 4 + num_tex * 4 > max_off:
            raise ValueError(f"{lump_name} texture count {num_tex} exceeds lump size {max_off}")
        offsets = struct.unpack_from(f"<{num_tex}i", raw, 4) if num_tex > 0 else ()

        for off in offsets:
            if off < 0 or off + _MAPTEXTURE_HEADER.size > max_off:
                raise ValueError(
                    f"R_InitTextures: bad texture directory offset {off} in {lump_name}"
                )
            raw_name, masked, width, height, _obsolete, patch_count = (
                _MAPTEXTURE_HEADER.unpack_from(raw, off)
            )
            if width <= 0 or height <= 0 or patch_count < 0:
                raise ValueError(
                    f"Invalid texture header in {lump_name}: width={width}, height={height}, "
                    f"patch_count={patch_count}"
                )
            patches_end = off + _MAPTEXTURE_HEADER.size + patch_count * _MAPPATCH_STRUCT.size
            if patches_end > max_off:
                raise ValueError(
                    f"Texture patches in {lump_name} at offset {off} exceed lump size {max_off}"
                )

            name = normalize_lump_name(raw_name)
            patches: list[TexturePatch] = []
            pos = off + _MAPTEXTURE_HEADER.size
            for _ in range(patch_count):
                ox, oy, patch_idx, stepdir, colormap = _MAPPATCH_STRUCT.unpack_from(raw, pos)
                pos += _MAPPATCH_STRUCT.size
                if patch_idx < 0 or patch_idx >= len(self.pnames):
                    raise ValueError(
                        f"R_InitTextures: patch index {patch_idx} out of range "
                        f"[0..{len(self.pnames)}) in texture {name!r}"
                    )
                patches.append(
                    TexturePatch(
                        origin_x=ox,
                        origin_y=oy,
                        patch_index=patch_idx,
                        stepdir=stepdir,
                        colormap=colormap,
                    )
                )

            if name == "-" and len(self._headers) == 1:
                # Replace dummy slot 0 if TEXTURE1 explicitly defines "-" first
                self._headers[0] = _TextureHeader(
                    name="-",
                    masked=bool(masked),
                    width=width,
                    height=height,
                    patches=tuple(patches),
                )
                continue

            idx = len(self._headers)
            self._headers.append(
                _TextureHeader(
                    name=name,
                    masked=bool(masked),
                    width=width,
                    height=height,
                    patches=tuple(patches),
                )
            )
            # First occurrence wins in R_CheckTextureNumForName
            self._by_name.setdefault(name, idx)

    def __len__(self) -> int:
        return len(self._headers)

    def __contains__(self, name: str) -> bool:
        return self.check_num_for_name(name) != -1

    def __iter__(self) -> Iterator[Texture]:
        for i in range(len(self._headers)):
            yield self.get(i)

    def __getitem__(self, key: int | str) -> Texture:
        return self.get(key)

    def check_num_for_name(self, name: str) -> int:
        """Return texture index for `name`, or -1 if not found.

        "-" (or any name starting with "-") resolves to 0.
        Port of R_CheckTextureNumForName (r_data.c).
        """
        norm = normalize_lump_name(name)
        if not norm or norm.startswith("-"):
            return 0
        return self._by_name.get(norm, -1)

    def num_for_name(self, name: str) -> int:
        """Return texture index for `name`, or raise KeyError if not found.

        Port of R_TextureNumForName (r_data.c).
        """
        idx = self.check_num_for_name(name)
        if idx == -1:
            raise KeyError(f"R_TextureNumForName: {normalize_lump_name(name)!r} not found")
        return idx

    def _get_patch_picture(self, patch_index: int) -> Picture:
        cached = self._patch_cache.get(patch_index)
        if cached is not None:
            return cached
        lump = self._patch_lumps[patch_index]
        if lump is None:
            pname = self.pnames[patch_index]
            raise WadError(f"R_InitTextures: Missing patch lump {pname!r} (#{patch_index})")
        pic = decode_picture(self._wad.get_bytes(lump))
        self._patch_cache[patch_index] = pic
        return pic

    def _compose(self, index: int) -> Texture:
        hdr = self._headers[index]
        indices = np.zeros((hdr.height, hdr.width), dtype=np.uint8)
        mask = np.zeros((hdr.height, hdr.width), dtype=bool)

        for tp in hdr.patches:
            pic = self._get_patch_picture(tp.patch_index)
            dst_x0 = max(0, tp.origin_x)
            dst_x1 = min(hdr.width, tp.origin_x + pic.width)
            dst_y0 = max(0, tp.origin_y)
            dst_y1 = min(hdr.height, tp.origin_y + pic.height)

            if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
                continue

            src_x0 = dst_x0 - tp.origin_x
            src_x1 = dst_x1 - tp.origin_x
            src_y0 = dst_y0 - tp.origin_y
            src_y1 = dst_y1 - tp.origin_y

            patch_mask = pic.mask[src_y0:src_y1, src_x0:src_x1]
            patch_pixels = pic.indices[src_y0:src_y1, src_x0:src_x1]

            dst_pixels = indices[dst_y0:dst_y1, dst_x0:dst_x1]
            dst_pixels[patch_mask] = patch_pixels[patch_mask]
            mask[dst_y0:dst_y1, dst_x0:dst_x1] |= patch_mask

        return Texture(
            name=hdr.name,
            width=hdr.width,
            height=hdr.height,
            indices=indices,
            mask=mask,
            patches=hdr.patches,
        )

    def get(self, index_or_name: int | str) -> Texture:
        """Return the composed `Texture` by integer index or texture name (cached)."""
        if isinstance(index_or_name, int):
            if index_or_name < 0 or index_or_name >= len(self._headers):
                raise IndexError(
                    f"Texture index {index_or_name} out of range [0..{len(self._headers)})"
                )
            idx = index_or_name
        elif isinstance(index_or_name, str):
            idx = self.num_for_name(index_or_name)
        else:
            raise TypeError(f"Unsupported texture selector: {type(index_or_name)!r}")

        tex = self._cache.get(idx)
        if tex is None:
            tex = self._compose(idx)
            self._cache[idx] = tex
        return tex


TextureManager = TextureRegistry
