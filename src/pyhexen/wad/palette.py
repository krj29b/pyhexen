# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""PLAYPAL, COLORMAP, and FOGMAP decoders and RGBA/PNG helpers.

Port of R_InitColormaps (r_data.c) and palette handling in v_video.c / i_video.c.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from pyhexen.wad.wadfile import WadFile, WadStack


def _extract_lump_bytes(wad_or_bytes: WadStack | WadFile | bytes, lump_name: str) -> bytes:
    if isinstance(wad_or_bytes, (WadStack, WadFile)):
        return wad_or_bytes.get_bytes(lump_name)
    if isinstance(wad_or_bytes, (bytes, bytearray, memoryview)):
        return bytes(wad_or_bytes)
    raise TypeError(f"Expected WadStack, WadFile, or bytes, got {type(wad_or_bytes)!r}")


def load_playpal(wad: WadStack | WadFile | bytes) -> np.ndarray:
    """Load PLAYPAL into an `(N, 256, 3)` uint8 numpy array (typically N=14 in Hexen)."""
    raw = _extract_lump_bytes(wad, "PLAYPAL")
    if len(raw) < 768 or len(raw) % 768 != 0:
        raise ValueError(
            f"PLAYPAL lump size must be a positive multiple of 768 bytes, got {len(raw)}"
        )
    num_palettes = len(raw) // 768
    return np.frombuffer(raw, dtype=np.uint8).reshape(num_palettes, 256, 3).copy()


def load_colormap(
    wad: WadStack | WadFile | bytes,
    lump_name: str = "COLORMAP",
) -> np.ndarray:
    """Load COLORMAP or FOGMAP into a `(rows, 256)` uint8 numpy array.

    Port of R_InitColormaps (r_data.c).
    """
    raw = _extract_lump_bytes(wad, lump_name)
    if len(raw) < 256 or len(raw) % 256 != 0:
        raise ValueError(
            f"{lump_name} lump size must be a positive multiple of 256 bytes, got {len(raw)}"
        )
    num_rows = len(raw) // 256
    return np.frombuffer(raw, dtype=np.uint8).reshape(num_rows, 256).copy()


def apply_palette(
    indices: np.ndarray,
    palette: np.ndarray,
    mask: np.ndarray | None = None,
    colormap_row: np.ndarray | None = None,
) -> np.ndarray:
    """Map an 8-bit index array `(H, W)` through an optional colormap row and RGB palette.

    Returns an `(H, W, 4)` RGBA uint8 array. Transparent pixels (`mask == False`)
    are set to `(0, 0, 0, 0)`.
    """
    idx = np.asarray(indices, dtype=np.uint8)
    pal = np.asarray(palette, dtype=np.uint8)
    if pal.ndim == 3:
        if pal.shape[0] < 1 or pal.shape[1:] != (256, 3):
            raise ValueError(f"Expected palette shape (N, 256, 3), got {pal.shape}")
        pal2d = pal[0]
    elif pal.ndim == 2:
        if pal.shape != (256, 3):
            raise ValueError(f"Expected palette shape (256, 3), got {pal.shape}")
        pal2d = pal
    else:
        raise ValueError(f"Unsupported palette dimensions: {pal.shape}")

    if colormap_row is not None:
        cmap = np.asarray(colormap_row, dtype=np.uint8)
        if cmap.shape != (256,):
            raise ValueError(f"Expected colormap_row shape (256,), got {cmap.shape}")
        idx = cmap[idx]

    rgb = pal2d[idx]
    rgba = np.empty((*idx.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    if mask is None:
        rgba[..., 3] = 255
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != idx.shape:
            raise ValueError(
                f"Mask shape {mask_arr.shape} does not match indices shape {idx.shape}"
            )
        rgba[..., 3] = np.where(mask_arr, 255, 0)
        rgba[~mask_arr, :3] = 0
    return rgba


def _write_png_rgba(rgba: np.ndarray, out_path: Path) -> None:
    """Write an `(H, W, 4)` uint8 RGBA array to a PNG file (via pygame or zlib fallback)."""
    height, width, channels = rgba.shape
    assert channels == 4
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pygame

        surface = pygame.image.frombuffer(rgba.tobytes(), (width, height), "RGBA")
        pygame.image.save(surface, str(out_path))
        return
    except Exception:
        pass

    # Pure-Python PNG writer fallback
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    # Prepend filter byte 0 to each scanline in vectorized numpy
    scanlines = np.empty((height, width * 4 + 1), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = rgba.reshape(height, width * 4)
    idat = zlib.compress(scanlines.tobytes())
    png_bytes = (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    )
    out_path.write_bytes(png_bytes)


def save_png(
    rgba_or_indices: np.ndarray,
    path: Path | str,
    palette: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> None:
    """Save an RGBA `(H, W, 4)`, RGB `(H, W, 3)`, or indexed `(H, W)` array to a PNG file."""
    arr = np.asarray(rgba_or_indices, dtype=np.uint8)
    if arr.ndim == 2:
        if palette is None:
            # Default grayscale palette if none provided
            gray = np.arange(256, dtype=np.uint8)
            palette = np.stack([gray, gray, gray], axis=-1)
        rgba = apply_palette(arr, palette, mask=mask)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgba = np.empty((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = arr
        if mask is None:
            rgba[..., 3] = 255
        else:
            mask_arr = np.asarray(mask, dtype=bool)
            rgba[..., 3] = np.where(mask_arr, 255, 0)
            rgba[~mask_arr, :3] = 0
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgba = np.ascontiguousarray(arr)
    else:
        raise ValueError(
            f"Expected (H, W) indices or (H, W, 3|4) image array, got shape {arr.shape}"
        )

    _write_png_rgba(rgba, Path(path))
