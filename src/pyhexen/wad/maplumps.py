# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Hexen map lump parsers into numpy structured arrays.

Port of map lump layouts in xddefs.h and p_setup.c.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyhexen.wad.wadfile import WadError, WadFile, WadStack, normalize_lump_name

# Lump order in a Hexen map WAD (xddefs.h ML_LABEL .. ML_BEHAVIOR)
ML_LABEL = 0
ML_THINGS = 1
ML_LINEDEFS = 2
ML_SIDEDEFS = 3
ML_VERTEXES = 4
ML_SEGS = 5
ML_SSECTORS = 6
ML_NODES = 7
ML_SECTORS = 8
ML_REJECT = 9
ML_BLOCKMAP = 10
ML_BEHAVIOR = 11

HEXEN_MAP_LUMP_ORDER: tuple[tuple[int, str], ...] = (
    (ML_THINGS, "THINGS"),
    (ML_LINEDEFS, "LINEDEFS"),
    (ML_SIDEDEFS, "SIDEDEFS"),
    (ML_VERTEXES, "VERTEXES"),
    (ML_SEGS, "SEGS"),
    (ML_SSECTORS, "SSECTORS"),
    (ML_NODES, "NODES"),
    (ML_SECTORS, "SECTORS"),
    (ML_REJECT, "REJECT"),
    (ML_BLOCKMAP, "BLOCKMAP"),
    (ML_BEHAVIOR, "BEHAVIOR"),
)

# Structured numpy dtypes matching reference xddefs.h (little-endian)
THING_DTYPE = np.dtype(
    [
        ("tid", "<i2"),
        ("x", "<i2"),
        ("y", "<i2"),
        ("height", "<i2"),
        ("angle", "<i2"),
        ("type", "<i2"),
        ("options", "<i2"),
        ("special", "u1"),
        ("args", "u1", (5,)),
    ]
)

LINEDEF_DTYPE = np.dtype(
    [
        ("v1", "<u2"),
        ("v2", "<u2"),
        ("flags", "<i2"),
        ("special", "u1"),
        ("args", "u1", (5,)),
        ("sidenum", "<i2", (2,)),
    ]
)

SIDEDEF_DTYPE = np.dtype(
    [
        ("textureoffset", "<i2"),
        ("rowoffset", "<i2"),
        ("toptexture", "S8"),
        ("bottomtexture", "S8"),
        ("midtexture", "S8"),
        ("sector", "<i2"),
    ]
)

VERTEX_DTYPE = np.dtype(
    [
        ("x", "<i2"),
        ("y", "<i2"),
    ]
)

SEG_DTYPE = np.dtype(
    [
        ("v1", "<u2"),
        ("v2", "<u2"),
        ("angle", "<i2"),
        ("linedef", "<u2"),
        ("side", "<i2"),
        ("offset", "<i2"),
    ]
)

SSECTOR_DTYPE = np.dtype(
    [
        ("numsegs", "<i2"),
        ("firstseg", "<u2"),
    ]
)

NODE_DTYPE = np.dtype(
    [
        ("x", "<i2"),
        ("y", "<i2"),
        ("dx", "<i2"),
        ("dy", "<i2"),
        ("bbox", "<i2", (2, 4)),
        ("children", "<u2", (2,)),
    ]
)

SECTOR_DTYPE = np.dtype(
    [
        ("floorheight", "<i2"),
        ("ceilingheight", "<i2"),
        ("floorpic", "S8"),
        ("ceilingpic", "S8"),
        ("lightlevel", "<i2"),
        ("special", "<i2"),
        ("tag", "<i2"),
    ]
)


class MapFormatError(WadError, ValueError):
    """Raised when a map is missing required Hexen lumps (e.g. Doom format without BEHAVIOR)."""


@dataclass(slots=True)
class RawMapLumps:
    """Parsed raw numpy structured arrays and byte buffers for a single Hexen map."""

    name: str
    things: np.ndarray
    linedefs: np.ndarray
    sidedefs: np.ndarray
    vertexes: np.ndarray
    segs: np.ndarray
    ssectors: np.ndarray
    nodes: np.ndarray
    sectors: np.ndarray
    reject: bytes
    blockmap: bytes
    behavior: bytes


def _from_buffer_checked(data: bytes, dtype: np.dtype, lump_name: str, map_name: str) -> np.ndarray:
    itemsize = dtype.itemsize
    if len(data) % itemsize != 0:
        raise MapFormatError(
            f"Map {map_name}: lump {lump_name} size ({len(data)} bytes) is not a multiple of "
            f"{itemsize} bytes"
        )
    return np.frombuffer(data, dtype=dtype).copy()


def parse_map_lumps(wad: WadStack | WadFile, map_name: str) -> RawMapLumps:
    """Parse all 11 Hexen map lumps following the `map_name` marker into structured arrays.

    Raises `MapFormatError` (subclass of `ValueError` and `WadError`) if the map is in Doom
    format (missing `BEHAVIOR` after `BLOCKMAP`) or if any lump is missing or malformed.
    """
    norm_name = normalize_lump_name(map_name)
    marker = wad.require(norm_name)
    base_idx = marker.index
    total_lumps = len(wad.lumps)

    # Check BEHAVIOR specifically first to give a clear Doom-format rejection message
    behavior_idx = base_idx + ML_BEHAVIOR
    if behavior_idx >= total_lumps or wad.lumps[behavior_idx].name != "BEHAVIOR":
        found = wad.lumps[behavior_idx].name if behavior_idx < total_lumps else "<EOF>"
        raise MapFormatError(
            f"Map {norm_name} is not in Hexen format: expected 'BEHAVIOR' lump at offset "
            f"+{ML_BEHAVIOR}, found {found!r} (Doom-format maps without BEHAVIOR are not supported)"
        )

    for offset, expected_name in HEXEN_MAP_LUMP_ORDER:
        idx = base_idx + offset
        if idx >= total_lumps or wad.lumps[idx].name != expected_name:
            found = wad.lumps[idx].name if idx < total_lumps else "<EOF>"
            raise MapFormatError(
                f"Map {norm_name}: expected lump {expected_name!r} at offset +{offset}, "
                f"found {found!r}"
            )

    things_bytes = wad.get_bytes(base_idx + ML_THINGS)
    linedefs_bytes = wad.get_bytes(base_idx + ML_LINEDEFS)
    sidedefs_bytes = wad.get_bytes(base_idx + ML_SIDEDEFS)
    vertexes_bytes = wad.get_bytes(base_idx + ML_VERTEXES)
    segs_bytes = wad.get_bytes(base_idx + ML_SEGS)
    ssectors_bytes = wad.get_bytes(base_idx + ML_SSECTORS)
    nodes_bytes = wad.get_bytes(base_idx + ML_NODES)
    sectors_bytes = wad.get_bytes(base_idx + ML_SECTORS)
    reject_bytes = wad.get_bytes(base_idx + ML_REJECT)
    blockmap_bytes = wad.get_bytes(base_idx + ML_BLOCKMAP)
    behavior_bytes = wad.get_bytes(base_idx + ML_BEHAVIOR)

    return RawMapLumps(
        name=norm_name,
        things=_from_buffer_checked(things_bytes, THING_DTYPE, "THINGS", norm_name),
        linedefs=_from_buffer_checked(linedefs_bytes, LINEDEF_DTYPE, "LINEDEFS", norm_name),
        sidedefs=_from_buffer_checked(sidedefs_bytes, SIDEDEF_DTYPE, "SIDEDEFS", norm_name),
        vertexes=_from_buffer_checked(vertexes_bytes, VERTEX_DTYPE, "VERTEXES", norm_name),
        segs=_from_buffer_checked(segs_bytes, SEG_DTYPE, "SEGS", norm_name),
        ssectors=_from_buffer_checked(ssectors_bytes, SSECTOR_DTYPE, "SSECTORS", norm_name),
        nodes=_from_buffer_checked(nodes_bytes, NODE_DTYPE, "NODES", norm_name),
        sectors=_from_buffer_checked(sectors_bytes, SECTOR_DTYPE, "SECTORS", norm_name),
        reject=reject_bytes,
        blockmap=blockmap_bytes,
        behavior=behavior_bytes,
    )
