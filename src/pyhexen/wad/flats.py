# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""64x64 floor and ceiling flat loader.

Port of R_InitFlats and R_FlatNumForName (r_data.c).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from pyhexen.wad.namespaces import flat_lumps
from pyhexen.wad.wadfile import LumpInfo, WadFile, WadStack, normalize_lump_name

FLAT_WIDTH = 64
FLAT_HEIGHT = 64
FLAT_BYTE_SIZE = FLAT_WIDTH * FLAT_HEIGHT  # 4096


@dataclass(slots=True)
class Flat:
    """Decoded 64x64 floor/ceiling flat texture."""

    name: str
    indices: np.ndarray  # uint8[64, 64]


def decode_flat(data: bytes | bytearray | memoryview, name: str = "") -> Flat:
    """Decode a 64x64 flat from raw bytes (accepts >= 4096 bytes for scrolling flats)."""
    if len(data) < FLAT_BYTE_SIZE:
        raise ValueError(f"Flat {name!r} data too short ({len(data)} bytes < {FLAT_BYTE_SIZE})")
    arr = (
        np.frombuffer(data, dtype=np.uint8, count=FLAT_BYTE_SIZE)
        .reshape((FLAT_HEIGHT, FLAT_WIDTH))
        .copy()
    )
    return Flat(name=normalize_lump_name(name), indices=arr)


class FlatRegistry:
    """Loads 64x64 flats from the F_START..F_END / FF_START..FF_END namespace.

    Port of R_InitFlats and R_FlatNumForName (r_data.c).
    """

    def __init__(self, wad: WadStack | WadFile) -> None:
        self._wad = wad
        self._lumps: list[LumpInfo] = flat_lumps(wad, deduplicate=True)
        self._by_name: dict[str, int] = {}
        for idx, lump in enumerate(self._lumps):
            self._by_name[lump.name] = idx
        self._cache: dict[int, Flat] = {}

    @property
    def lumps(self) -> list[LumpInfo]:
        """Flat lumps in registry order."""
        return self._lumps

    def __len__(self) -> int:
        return len(self._lumps)

    def __contains__(self, name: str) -> bool:
        return self.check_num_for_name(name) != -1

    def __iter__(self) -> Iterator[Flat]:
        for i in range(len(self._lumps)):
            yield self.get(i)

    def __getitem__(self, key: int | str) -> Flat:
        return self.get(key)

    def check_num_for_name(self, name: str) -> int:
        """Return flat index for `name`, or -1 if not found."""
        norm = normalize_lump_name(name)
        return self._by_name.get(norm, -1)

    def num_for_name(self, name: str) -> int:
        """Return flat index for `name`, or raise KeyError if not found.

        Port of R_FlatNumForName (r_data.c).
        """
        idx = self.check_num_for_name(name)
        if idx == -1:
            raise KeyError(f"R_FlatNumForName: {normalize_lump_name(name)!r} not found")
        return idx

    def get(self, index_or_name: int | str) -> Flat:
        """Return the decoded `Flat` by index or name (cached)."""
        if isinstance(index_or_name, int):
            if index_or_name < 0 or index_or_name >= len(self._lumps):
                raise IndexError(f"Flat index {index_or_name} out of range [0..{len(self._lumps)})")
            idx = index_or_name
        elif isinstance(index_or_name, str):
            idx = self.num_for_name(index_or_name)
        else:
            raise TypeError(f"Unsupported flat selector: {type(index_or_name)!r}")

        flat = self._cache.get(idx)
        if flat is None:
            lump = self._lumps[idx]
            flat = decode_flat(self._wad.get_bytes(lump), name=lump.name)
            self._cache[idx] = flat
        return flat
