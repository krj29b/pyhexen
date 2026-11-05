# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""WAD container reader and PWAD merge stack.

Port of w_wad.c / w_wad.h from Chocolate Doom.
"""

from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Self

_HEADER_STRUCT = struct.Struct("<4sii")
_DIR_ENTRY_STRUCT = struct.Struct("<ii8s")


class WadError(ValueError):
    """Raised when a WAD file is malformed or an invalid operation is requested."""


class LumpNotFoundError(WadError, KeyError):
    """Raised when a required lump is not found in a WAD or WadStack."""


def normalize_lump_name(name: str | bytes) -> str:
    """Normalize a lump name: strip at first NUL, uppercase, max 8 ASCII chars."""
    if isinstance(name, bytes):
        raw = name.split(b"\x00", 1)[0]
        text = raw.decode("latin1")
    else:
        text = name.split("\x00", 1)[0]
    return text.upper()[:8]


@dataclass(slots=True)
class LumpInfo:
    """Metadata for a single lump in a WAD directory.

    Port of lumpinfo_t (w_wad.h).
    """

    name: str
    offset: int
    size: int
    source: str
    index: int
    _wad: WadFile | None = field(default=None, repr=False, compare=False)


def _normalize_marker_set(markers: str | Sequence[str]) -> frozenset[str]:
    if isinstance(markers, str):
        return frozenset({normalize_lump_name(markers)})
    return frozenset(normalize_lump_name(m) for m in markers)


class WadFile:
    """A single IWAD or PWAD file backed by mmap (for files) or bytes (in-memory).

    Port of W_AddFile / W_OpenFile / W_ReadLump (w_wad.c).
    """

    def __init__(
        self,
        source: Path | str | bytes | bytearray | memoryview,
        *,
        name: str | None = None,
    ) -> None:
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        self._buffer: bytes | bytearray | memoryview | mmap.mmap | None = None
        self.path: Path | None = None

        if isinstance(source, (bytes, bytearray, memoryview)):
            self._buffer = source
            self.source = name if name is not None else "<memory>"
        elif isinstance(source, (Path, str)):
            path = Path(source)
            self.path = path
            self.source = name if name is not None else path.name
            if not path.is_file():
                raise FileNotFoundError(f"WAD file not found: {path}")
            file_size = path.stat().st_size
            if file_size < _HEADER_STRUCT.size:
                raise WadError(f"WAD file {self.source!r} is too small ({file_size} bytes < 12)")
            f = path.open("rb")
            try:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                f.close()
                raise
            self._file = f
            self._mmap = mm
            self._buffer = mm
        else:
            raise TypeError(f"Unsupported WAD source type: {type(source)!r}")

        try:
            self.wad_type, self._lumps = self._parse_directory()
        except Exception:
            self.close()
            raise

        # Map normalized name -> list of lump indices in directory order
        self._by_name: dict[str, list[int]] = {}
        for idx, lump in enumerate(self._lumps):
            self._by_name.setdefault(lump.name, []).append(idx)

    @classmethod
    def from_bytes(cls, data: bytes, *, name: str = "<memory>") -> Self:
        """Open an in-memory WAD byte buffer."""
        return cls(data, name=name)

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        """Open a WAD file from disk using mmap."""
        return cls(path)

    def _parse_directory(self) -> tuple[str, list[LumpInfo]]:
        buf = self._buffer
        if buf is None:
            raise WadError("WAD file is closed")

        total_size = len(buf)
        if total_size < _HEADER_STRUCT.size:
            raise WadError(f"WAD {self.source!r} is too small for header ({total_size} bytes < 12)")

        ident_bytes, num_lumps, dir_offset = _HEADER_STRUCT.unpack_from(buf, 0)
        if ident_bytes not in (b"IWAD", b"PWAD"):
            raise WadError(
                f"WAD {self.source!r} has invalid magic {ident_bytes!r} "
                "(expected b'IWAD' or b'PWAD')"
            )
        wad_type = ident_bytes.decode("ascii")

        if num_lumps < 0:
            raise WadError(f"WAD {self.source!r} has negative lump count: {num_lumps}")
        if dir_offset < 0:
            raise WadError(f"WAD {self.source!r} has negative directory offset: {dir_offset}")

        dir_size = num_lumps * _DIR_ENTRY_STRUCT.size
        if num_lumps > 0 and (dir_offset + dir_size > total_size):
            raise WadError(
                f"WAD {self.source!r} directory [{dir_offset}..{dir_offset + dir_size}] "
                f"exceeds file size ({total_size})"
            )

        lumps: list[LumpInfo] = []
        for i in range(num_lumps):
            entry_pos = dir_offset + i * _DIR_ENTRY_STRUCT.size
            filepos, size, raw_name = _DIR_ENTRY_STRUCT.unpack_from(buf, entry_pos)
            if size < 0:
                raise WadError(f"WAD {self.source!r} lump #{i} has negative size {size}")
            if size > 0 and (filepos < 0 or filepos + size > total_size):
                raise WadError(
                    f"WAD {self.source!r} lump #{i} data [{filepos}..{filepos + size}] "
                    f"exceeds file bounds [0..{total_size}]"
                )
            if size == 0 and (filepos < 0 or filepos > total_size):
                raise WadError(
                    f"WAD {self.source!r} marker lump #{i} offset {filepos} "
                    f"out of bounds [0..{total_size}]"
                )
            name = normalize_lump_name(raw_name)
            lumps.append(
                LumpInfo(
                    name=name,
                    offset=filepos,
                    size=size,
                    source=self.source,
                    index=i,
                    _wad=self,
                )
            )

        return wad_type, lumps

    @property
    def lumps(self) -> list[LumpInfo]:
        """All lumps in directory order."""
        return self._lumps

    def __len__(self) -> int:
        return len(self._lumps)

    def __iter__(self) -> Iterator[LumpInfo]:
        return iter(self._lumps)

    def __contains__(self, name: str) -> bool:
        return normalize_lump_name(name) in self._by_name

    def __getitem__(self, key: int | str) -> LumpInfo:
        if isinstance(key, int):
            return self._lumps[key]
        return self.require(key)

    def find(self, name: str) -> LumpInfo | None:
        """Find the last lump matching `name`, or None if not found.

        Port of W_CheckNumForName (w_wad.c).
        """
        indices = self._by_name.get(normalize_lump_name(name))
        if not indices:
            return None
        return self._lumps[indices[-1]]

    def require(self, name: str) -> LumpInfo:
        """Find the last lump matching `name`, or raise LumpNotFoundError.

        Port of W_GetNumForName (w_wad.c).
        """
        lump = self.find(name)
        if lump is None:
            raise LumpNotFoundError(
                f"W_GetNumForName: {normalize_lump_name(name)!r} not found in {self.source}"
            )
        return lump

    def find_all(self, name: str) -> list[LumpInfo]:
        """Return all lumps matching `name` in directory order."""
        indices = self._by_name.get(normalize_lump_name(name))
        if not indices:
            return []
        return [self._lumps[i] for i in indices]

    def get_bytes(self, lump_or_name: LumpInfo | str | int) -> bytes:
        """Read the raw bytes of a lump by LumpInfo, lump name, or lump index.

        Port of W_ReadLump / W_CacheLumpNum / W_CacheLumpName (w_wad.c).
        """
        if self._buffer is None:
            raise WadError(f"WAD {self.source!r} is closed")

        if isinstance(lump_or_name, LumpInfo):
            lump = lump_or_name
        elif isinstance(lump_or_name, int):
            if lump_or_name < 0 or lump_or_name >= len(self._lumps):
                raise IndexError(f"W_ReadLump: {lump_or_name} out of range [0..{len(self._lumps)})")
            lump = self._lumps[lump_or_name]
        elif isinstance(lump_or_name, str):
            lump = self.require(lump_or_name)
        else:
            raise TypeError(f"Unsupported lump selector: {type(lump_or_name)!r}")

        if lump.size == 0:
            return b""
        return bytes(self._buffer[lump.offset : lump.offset + lump.size])

    def lumps_between(
        self,
        start_markers: str | Sequence[str],
        end_markers: str | Sequence[str],
    ) -> list[LumpInfo]:
        """Return all lumps enclosed between `start_markers` and `end_markers`."""
        return _collect_lumps_between(self._lumps, start_markers, end_markers)

    def close(self) -> None:
        """Close the underlying mmap and file handle, if any."""
        self._buffer = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def _collect_lumps_between(
    lumps: Sequence[LumpInfo],
    start_markers: str | Sequence[str],
    end_markers: str | Sequence[str],
) -> list[LumpInfo]:
    start_set = _normalize_marker_set(start_markers)
    end_set = _normalize_marker_set(end_markers)

    result: list[LumpInfo] = []
    current_block: list[LumpInfo] | None = None

    for lump in lumps:
        if current_block is None:
            if lump.name in start_set:
                current_block = []
        else:
            if lump.name in end_set:
                result.extend(current_block)
                current_block = None
            elif lump.name in start_set:
                # Restart block if a nested/duplicate start marker appears
                current_block = []
            else:
                current_block.append(lump)

    return result


class WadStack:
    """Ordered stack of an IWAD and zero or more PWADs with last-match override.

    Port of the global lump directory in w_wad.c.
    """

    def __init__(
        self,
        wads: Sequence[WadFile | Path | str | bytes] | WadFile | Path | str | bytes | None = None,
    ) -> None:
        self._wads: list[WadFile] = []
        self._owned_wads: list[WadFile] = []
        self._lumps: list[LumpInfo] = []
        self._by_name: dict[str, list[int]] = {}

        if wads is not None:
            if isinstance(wads, (WadFile, Path, str, bytes)):
                self.add_wad(wads)
            else:
                for w in wads:
                    self.add_wad(w)

    def add_wad(
        self,
        wad: WadFile | Path | str | bytes,
        *,
        source: str | None = None,
    ) -> WadFile:
        """Append a WAD to the stack. Later WADs override earlier ones in `find()`.

        Port of W_AddFile (w_wad.c).
        """
        if isinstance(wad, WadFile):
            wad_file = wad
        else:
            wad_file = WadFile(wad, name=source)
            self._owned_wads.append(wad_file)

        self._wads.append(wad_file)
        base_index = len(self._lumps)
        for i, lump in enumerate(wad_file.lumps):
            global_index = base_index + i
            stacked_lump = LumpInfo(
                name=lump.name,
                offset=lump.offset,
                size=lump.size,
                source=lump.source,
                index=global_index,
                _wad=wad_file,
            )
            self._lumps.append(stacked_lump)
            self._by_name.setdefault(stacked_lump.name, []).append(global_index)

        return wad_file

    @property
    def wads(self) -> list[WadFile]:
        """Loaded WAD files in stack order."""
        return self._wads

    @property
    def lumps(self) -> list[LumpInfo]:
        """All lumps across the stack in load order."""
        return self._lumps

    def __len__(self) -> int:
        return len(self._lumps)

    def __iter__(self) -> Iterator[LumpInfo]:
        return iter(self._lumps)

    def __contains__(self, name: str) -> bool:
        return normalize_lump_name(name) in self._by_name

    def __getitem__(self, key: int | str) -> LumpInfo:
        if isinstance(key, int):
            return self._lumps[key]
        return self.require(key)

    def find(self, name: str) -> LumpInfo | None:
        """Return the last lump matching `name` (PWAD override), or None.

        Port of W_CheckNumForName (w_wad.c).
        """
        indices = self._by_name.get(normalize_lump_name(name))
        if not indices:
            return None
        return self._lumps[indices[-1]]

    def require(self, name: str) -> LumpInfo:
        """Return the last lump matching `name`, or raise LumpNotFoundError.

        Port of W_GetNumForName (w_wad.c).
        """
        lump = self.find(name)
        if lump is None:
            raise LumpNotFoundError(f"W_GetNumForName: {normalize_lump_name(name)!r} not found!")
        return lump

    def find_all(self, name: str) -> list[LumpInfo]:
        """Return all lumps matching `name` in stack order (earliest to latest)."""
        indices = self._by_name.get(normalize_lump_name(name))
        if not indices:
            return []
        return [self._lumps[i] for i in indices]

    def get_bytes(self, lump_or_name: LumpInfo | str | int) -> bytes:
        """Read the raw bytes of a lump by LumpInfo, lump name, or stack lump index.

        Port of W_ReadLump / W_CacheLumpNum / W_CacheLumpName (w_wad.c).
        """
        if isinstance(lump_or_name, LumpInfo):
            lump = lump_or_name
            if lump._wad is not None:
                return lump._wad.get_bytes(lump)
            # Fallback if LumpInfo was constructed without _wad reference
            if 0 <= lump.index < len(self._lumps):
                target = self._lumps[lump.index]
                if target._wad is not None and target.name == lump.name:
                    return target._wad.get_bytes(lump)
            for w in reversed(self._wads):
                if w.source == lump.source:
                    return w.get_bytes(lump)
            raise WadError(f"Cannot resolve source WAD for lump {lump!r}")
        if isinstance(lump_or_name, int):
            if lump_or_name < 0 or lump_or_name >= len(self._lumps):
                raise IndexError(f"W_ReadLump: {lump_or_name} out of range [0..{len(self._lumps)})")
            target = self._lumps[lump_or_name]
            assert target._wad is not None
            return target._wad.get_bytes(target)
        if isinstance(lump_or_name, str):
            target = self.require(lump_or_name)
            assert target._wad is not None
            return target._wad.get_bytes(target)
        raise TypeError(f"Unsupported lump selector: {type(lump_or_name)!r}")

    def lumps_between(
        self,
        start_markers: str | Sequence[str],
        end_markers: str | Sequence[str],
    ) -> list[LumpInfo]:
        """Return all lumps between `start_markers` and `end_markers` across the stack."""
        return _collect_lumps_between(self._lumps, start_markers, end_markers)

    def close(self) -> None:
        """Close all WAD files owned by this stack."""
        for wad in self._owned_wads:
            wad.close()
        self._owned_wads.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
