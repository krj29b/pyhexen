# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Synthetic WAD and Hexen-format map builder for CI-safe unit/integration tests.

Generates valid little-endian IWAD/PWAD files, synthetic palettes, colormaps,
Doom-format pictures (patches/sprites), flats, PNAMES/TEXTURE1 lumps, and
Hexen-format maps (including THINGS, LINEDEFS, SIDEDEFS, VERTEXES, SEGS,
SSECTORS, NODES, SECTORS, REJECT, BLOCKMAP, and BEHAVIOR).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np

from pyhexen.wad.wadfile import normalize_lump_name

NF_SUBSECTOR = 0x8000

ML_BLOCKING = 0x0001
ML_BLOCKMONSTERS = 0x0002
ML_TWOSIDED = 0x0004
ML_DONTPEGTOP = 0x0008
ML_DONTPEGBOTTOM = 0x0010


def _pack_name8(name: str) -> bytes:
    """Pack a lump/texture name into an 8-byte NUL-padded ASCII field."""
    norm = normalize_lump_name(name)
    encoded = norm.encode("ascii")[:8]
    return encoded.ljust(8, b"\x00")


def encode_picture(
    pixels: np.ndarray,
    mask: np.ndarray | None = None,
    left: int = 0,
    top: int = 0,
) -> bytes:
    """Encode a 2D uint8 array and optional bool mask into Doom picture format.

    Matches patch_t / column_t / post_t in reference v_patch.h and r_data.c.
    """
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"pixels must be 2D (height, width), got shape {arr.shape}")
    height, width = arr.shape

    if mask is None:
        mask_arr = np.ones((height, width), dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (height, width):
            raise ValueError(
                f"mask shape {mask_arr.shape} does not match pixels shape {(height, width)}"
            )

    header_size = 8 + 4 * width
    columns_data: list[bytes] = []
    column_offsets: list[int] = []
    current_offset = header_size

    for x in range(width):
        column_offsets.append(current_offset)
        col_bytes = bytearray()
        y = 0
        while y < height:
            if not mask_arr[y, x]:
                y += 1
                continue
            run_start = y
            while y < height and mask_arr[y, x] and (y - run_start) < 128:
                y += 1
            run_len = y - run_start
            if run_start >= 254:
                raise ValueError(
                    "encode_picture supports column post topdelta < 254 for synthetic fixtures"
                )
            col_bytes.append(run_start)
            col_bytes.append(run_len)
            col_bytes.append(0)  # pre-pad byte
            col_bytes.extend(arr[run_start : run_start + run_len, x].tobytes())
            col_bytes.append(0)  # post-pad byte

        col_bytes.append(0xFF)  # column terminator
        col_blob = bytes(col_bytes)
        columns_data.append(col_blob)
        current_offset += len(col_blob)

    header = struct.pack("<4h", width, height, left, top)
    ofs_table = struct.pack(f"<{width}I", *column_offsets) if width > 0 else b""
    return header + ofs_table + b"".join(columns_data)


@dataclass(slots=True)
class PatchPlacement:
    """A single patch placement inside a composite texture (mappatch_t)."""

    originx: int = 0
    originy: int = 0
    patch: int = 0
    stepdir: int = 1
    colormap: int = 0


@dataclass(slots=True)
class TextureDef:
    """Definition of a composite wall texture for TEXTURE1 (maptexture_t)."""

    name: str
    width: int
    height: int
    patches: list[PatchPlacement] = field(default_factory=list)
    masked: int = 0
    obsolete: int = 0


class WadBuilder:
    """Builds synthetic IWAD or PWAD files in memory or on disk."""

    def __init__(self, wad_type: str = "IWAD") -> None:
        wad_type_upper = wad_type.upper()
        if wad_type_upper not in ("IWAD", "PWAD"):
            raise ValueError(f"wad_type must be 'IWAD' or 'PWAD', got {wad_type!r}")
        self.wad_type = wad_type_upper
        self._lumps: list[tuple[str, bytes]] = []

    @staticmethod
    def encode_picture(
        pixels: np.ndarray,
        mask: np.ndarray | None = None,
        left: int = 0,
        top: int = 0,
    ) -> bytes:
        """Encode a 2D uint8 array + bool mask into Doom picture format."""
        return encode_picture(pixels, mask=mask, left=left, top=top)

    def add_lump(self, name: str, data: bytes | bytearray | memoryview = b"") -> Self:
        """Append a named lump with raw byte content."""
        norm = normalize_lump_name(name)
        if not norm:
            raise ValueError("Lump name must not be empty")
        self._lumps.append((norm, bytes(data)))
        return self

    def add_marker(self, name: str) -> Self:
        """Append a 0-byte marker lump (such as S_START, F_END, MAP01)."""
        return self.add_lump(name, b"")

    def add_playpal(self, num_palettes: int = 14) -> Self:
        """Add a synthetic PLAYPAL lump with `num_palettes` x 256 RGB triplets."""
        pal = np.zeros((num_palettes, 256, 3), dtype=np.uint8)
        idx = np.arange(256, dtype=np.uint8)
        for p in range(num_palettes):
            pal[p, :, 0] = idx
            pal[p, :, 1] = (idx + p * 17) & 0xFF
            pal[p, :, 2] = (255 - idx) ^ (p * 7)
        return self.add_lump("PLAYPAL", pal.tobytes())

    def add_colormap(self, num_rows: int = 33, *, include_fogmap: bool = True) -> Self:
        """Add synthetic COLORMAP (and optionally FOGMAP) lumps of `num_rows` x 256 bytes."""
        cmap = np.zeros((num_rows, 256), dtype=np.uint8)
        base = np.arange(256, dtype=np.int32)
        for r in range(num_rows):
            cmap[r] = np.clip(base - r, 0, 255).astype(np.uint8)
        self.add_lump("COLORMAP", cmap.tobytes())
        if include_fogmap:
            fmap = np.zeros((num_rows, 256), dtype=np.uint8)
            for r in range(num_rows):
                fmap[r] = ((base + r * 3) & 0xFF).astype(np.uint8)
            self.add_lump("FOGMAP", fmap.tobytes())
        return self

    def add_picture(
        self,
        name: str,
        pixels: np.ndarray,
        mask: np.ndarray | None = None,
        left: int = 0,
        top: int = 0,
    ) -> Self:
        """Encode and add a Doom picture lump."""
        return self.add_lump(name, encode_picture(pixels, mask=mask, left=left, top=top))

    def add_flat(
        self,
        name: str,
        data: bytes | np.ndarray | None = None,
        *,
        fill: int = 1,
    ) -> Self:
        """Add a 64x64 (4096-byte) flat lump."""
        if data is None:
            arr = np.full((64, 64), fill & 0xFF, dtype=np.uint8)
            raw = arr.tobytes()
        elif isinstance(data, np.ndarray):
            arr = np.asarray(data, dtype=np.uint8)
            if arr.shape != (64, 64):
                raise ValueError(f"Flat array must have shape (64, 64), got {arr.shape}")
            raw = arr.tobytes()
        else:
            raw = bytes(data)
            if len(raw) != 4096:
                raise ValueError(f"Flat data must be 4096 bytes, got {len(raw)}")
        return self.add_lump(name, raw)

    def add_textures(
        self,
        textures: Sequence[TextureDef | dict[str, Any]],
        pnames: Sequence[str] = (),
        *,
        lump_name: str = "TEXTURE1",
    ) -> Self:
        """Add synthetic PNAMES and TEXTURE1 lumps.

        Matches mappatch_t and maptexture_t in reference xddefs.h / r_data.c.
        """
        pnames_bytes = bytearray(struct.pack("<i", len(pnames)))
        for pname in pnames:
            pnames_bytes.extend(_pack_name8(pname))
        self.add_lump("PNAMES", bytes(pnames_bytes))

        tex_defs: list[TextureDef] = []
        for t in textures:
            if isinstance(t, TextureDef):
                tex_defs.append(t)
            elif isinstance(t, dict):
                raw_patches = t.get("patches", [])
                patches = [
                    p
                    if isinstance(p, PatchPlacement)
                    else PatchPlacement(
                        originx=int(p.get("originx", 0)),
                        originy=int(p.get("originy", 0)),
                        patch=int(p.get("patch", 0)),
                        stepdir=int(p.get("stepdir", 1)),
                        colormap=int(p.get("colormap", 0)),
                    )
                    for p in raw_patches
                ]
                tex_defs.append(
                    TextureDef(
                        name=str(t["name"]),
                        width=int(t["width"]),
                        height=int(t["height"]),
                        patches=patches,
                        masked=int(t.get("masked", 0)),
                        obsolete=int(t.get("obsolete", 0)),
                    )
                )
            else:
                raise TypeError(f"Unsupported texture spec: {type(t)!r}")

        num_tex = len(tex_defs)
        header_size = 4 + 4 * num_tex
        offsets: list[int] = []
        blobs: list[bytes] = []
        cur_ofs = header_size

        for td in tex_defs:
            offsets.append(cur_ofs)
            entry = bytearray(
                struct.pack(
                    "<8si2hih",
                    _pack_name8(td.name),
                    td.masked,
                    td.width,
                    td.height,
                    td.obsolete,
                    len(td.patches),
                )
            )
            for mp in td.patches:
                entry.extend(
                    struct.pack(
                        "<5h",
                        mp.originx,
                        mp.originy,
                        mp.patch,
                        mp.stepdir,
                        mp.colormap,
                    )
                )
            blob = bytes(entry)
            blobs.append(blob)
            cur_ofs += len(blob)

        tex_lump = (
            struct.pack("<i", num_tex)
            + (struct.pack(f"<{num_tex}i", *offsets) if num_tex > 0 else b"")
            + b"".join(blobs)
        )
        return self.add_lump(lump_name, tex_lump)

    def add_map(self, map_builder: MapBuilder) -> Self:
        """Append all lumps generated by `map_builder` to this WAD."""
        for lump_name, lump_data in map_builder.build_lumps():
            self.add_lump(lump_name, lump_data)
        return self

    def to_bytes(self) -> bytes:
        """Serialize the WAD header, lump payloads, and directory to bytes."""
        num_lumps = len(self._lumps)
        dir_offset = 12 + sum(len(data) for _, data in self._lumps)

        out = bytearray(struct.pack("<4sii", self.wad_type.encode("ascii"), num_lumps, dir_offset))
        dir_entries = bytearray()
        cur_pos = 12

        for name, data in self._lumps:
            size = len(data)
            filepos = cur_pos if size > 0 else cur_pos
            out.extend(data)
            cur_pos += size
            dir_entries.extend(struct.pack("<ii8s", filepos, size, _pack_name8(name)))

        out.extend(dir_entries)
        return bytes(out)

    def write(self, path: Path | str) -> Path:
        """Write the serialized WAD to `path` and return the Path."""
        out_path = Path(path)
        out_path.write_bytes(self.to_bytes())
        return out_path


class MapBuilder:
    """Builds a valid Hexen-format map lump sequence for synthetic testing.

    Lump sequence matches reference xddefs.h (ML_LABEL..ML_BEHAVIOR):
    MAPxx, THINGS, LINEDEFS, SIDEDEFS, VERTEXES, SEGS, SSECTORS, NODES,
    SECTORS, REJECT, BLOCKMAP, BEHAVIOR.

    Limits of automatic BSP generation:
    - Each room added via `add_room()` or `add_two_sector_rooms()` must be an
      axis-aligned convex rectangle, producing one convex subsector per room.
    - When 1 subsector is present, a single root node with both children set to
      `NF_SUBSECTOR | 0` is emitted so `numnodes >= 1` holds.
    - When >= 2 subsectors are present and no manual nodes are added, a binary
      tree of partition nodes splitting subsectors along vertical/horizontal
      lines is generated automatically.
    """

    def __init__(self, name: str = "MAP01") -> None:
        self.name = normalize_lump_name(name)
        self.things: list[bytes] = []
        self.linedefs: list[bytes] = []
        self._linedef_coords: list[tuple[int, int, int, int]] = []
        self.sidedefs: list[bytes] = []
        self.vertices: list[tuple[int, int]] = []
        self.segs: list[bytes] = []
        self.subsectors: list[tuple[int, int]] = []
        self._subsector_bboxes: list[tuple[int, int, int, int]] = []
        self.nodes: list[bytes] = []
        self.sectors: list[bytes] = []
        self._behavior: bytes | None = None

    def add_vertex(self, x: int, y: int) -> int:
        """Add a vertex in map units (int16) and return its index."""
        idx = len(self.vertices)
        self.vertices.append((int(x), int(y)))
        return idx

    def add_sector(
        self,
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_pic: str = "FLOOR01",
        ceiling_pic: str = "CEIL01",
        light: int = 160,
        special: int = 0,
        tag: int = 0,
    ) -> int:
        """Add a sector (26 bytes: <2h8s8s3h) and return its index."""
        idx = len(self.sectors)
        self.sectors.append(
            struct.pack(
                "<2h8s8s3h",
                floor_height,
                ceiling_height,
                _pack_name8(floor_pic),
                _pack_name8(ceiling_pic),
                light,
                special,
                tag,
            )
        )
        return idx

    def add_sidedef(
        self,
        sector: int,
        *,
        textureoffset: int = 0,
        rowoffset: int = 0,
        toptexture: str = "-",
        bottomtexture: str = "-",
        midtexture: str = "WALL01",
    ) -> int:
        """Add a sidedef (30 bytes: <2h8s8s8sh) and return its index."""
        idx = len(self.sidedefs)
        self.sidedefs.append(
            struct.pack(
                "<2h8s8s8sh",
                textureoffset,
                rowoffset,
                _pack_name8(toptexture),
                _pack_name8(bottomtexture),
                _pack_name8(midtexture),
                sector,
            )
        )
        return idx

    def add_linedef(
        self,
        v1: int,
        v2: int,
        front_side: int,
        back_side: int = 0xFFFF,
        *,
        flags: int = ML_BLOCKING,
        special: int = 0,
        args: Sequence[int] = (0, 0, 0, 0, 0),
    ) -> int:
        """Add a Hexen linedef (16 bytes: <3H6B2H) and return its index."""
        if len(args) != 5:
            raise ValueError(f"Hexen linedef requires 5 args, got {len(args)}")
        idx = len(self.linedefs)
        self.linedefs.append(
            struct.pack(
                "<3H6B2H",
                v1 & 0xFFFF,
                v2 & 0xFFFF,
                flags & 0xFFFF,
                special & 0xFF,
                args[0] & 0xFF,
                args[1] & 0xFF,
                args[2] & 0xFF,
                args[3] & 0xFF,
                args[4] & 0xFF,
                front_side & 0xFFFF,
                back_side & 0xFFFF,
            )
        )
        x1, y1 = self.vertices[v1]
        x2, y2 = self.vertices[v2]
        self._linedef_coords.append((x1, y1, x2, y2))
        return idx

    def add_seg(
        self,
        v1: int,
        v2: int,
        angle: int,
        linedef: int,
        side: int = 0,
        offset: int = 0,
    ) -> int:
        """Add a seg (12 bytes: <6h) and return its index."""
        idx = len(self.segs)
        # Convert unsigned 16-bit BAM angle to signed int16 if needed
        signed_angle = ((angle + 0x8000) & 0xFFFF) - 0x8000
        self.segs.append(struct.pack("<6h", v1, v2, signed_angle, linedef, side, offset))
        return idx

    def add_subsector(
        self,
        numsegs: int,
        firstseg: int,
        *,
        bbox: tuple[int, int, int, int] = (128, 0, 0, 128),
    ) -> int:
        """Add a subsector (4 bytes: <2h) and return its index.

        `bbox` is (top, bottom, left, right) in map units.
        """
        idx = len(self.subsectors)
        self.subsectors.append((numsegs, firstseg))
        self._subsector_bboxes.append(bbox)
        return idx

    def add_node(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        bbox_right: tuple[int, int, int, int],
        bbox_left: tuple[int, int, int, int],
        child_right: int,
        child_left: int,
    ) -> int:
        """Add a BSP node (28 bytes: <12h2H) and return its index.

        `bbox_right` and `bbox_left` are (top, bottom, left, right) matching
        BOXTOP, BOXBOTTOM, BOXLEFT, BOXRIGHT in m_bbox.h.
        """
        idx = len(self.nodes)
        self.nodes.append(
            struct.pack(
                "<12h2H",
                x,
                y,
                dx,
                dy,
                bbox_right[0],
                bbox_right[1],
                bbox_right[2],
                bbox_right[3],
                bbox_left[0],
                bbox_left[1],
                bbox_left[2],
                bbox_left[3],
                child_right & 0xFFFF,
                child_left & 0xFFFF,
            )
        )
        return idx

    def add_thing(
        self,
        x: int,
        y: int,
        type: int = 1,
        *,
        tid: int = 0,
        z: int = 0,
        angle: int = 0,
        flags: int = 0x07E7,
        special: int = 0,
        args: Sequence[int] = (0, 0, 0, 0, 0),
    ) -> int:
        """Add a Hexen mapthing_t (20 bytes: <7hB5B) and return its index."""
        if len(args) != 5:
            raise ValueError(f"Hexen thing requires 5 args, got {len(args)}")
        idx = len(self.things)
        self.things.append(
            struct.pack(
                "<7hB5B",
                tid,
                x,
                y,
                z,
                angle,
                type,
                flags,
                special & 0xFF,
                args[0] & 0xFF,
                args[1] & 0xFF,
                args[2] & 0xFF,
                args[3] & 0xFF,
                args[4] & 0xFF,
            )
        )
        return idx

    def set_behavior(self, data: bytes) -> Self:
        """Set custom BEHAVIOR lump bytes."""
        self._behavior = bytes(data)
        return self

    def add_room(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 256,
        height: int = 256,
        *,
        floor_height: int = 0,
        ceiling_height: int = 128,
        floor_pic: str = "FLOOR01",
        ceiling_pic: str = "CEIL01",
        wall_tex: str = "WALL01",
        light: int = 160,
        special: int = 0,
        tag: int = 0,
    ) -> int:
        """Add a closed rectangular room (1 sector, 4 vertices, 4 lines, 4 segs, 1 subsector).

        Vertices are wound clockwise so the front (right) side of every wall
        faces the room interior. Returns the created sector index.
        """
        sec_idx = self.add_sector(
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            floor_pic=floor_pic,
            ceiling_pic=ceiling_pic,
            light=light,
            special=special,
            tag=tag,
        )
        # Clockwise vertices: TL, TR, BR, BL
        v_tl = self.add_vertex(x, y + height)
        v_tr = self.add_vertex(x + width, y + height)
        v_br = self.add_vertex(x + width, y)
        v_bl = self.add_vertex(x, y)

        edges = [
            (v_tl, v_tr, 0x0000),  # Top edge: East (0 deg)
            (v_tr, v_br, 0xC000),  # Right edge: South (270 deg)
            (v_br, v_bl, 0x8000),  # Bottom edge: West (180 deg)
            (v_bl, v_tl, 0x4000),  # Left edge: North (90 deg)
        ]
        first_seg = len(self.segs)
        for v1, v2, bam_angle in edges:
            sd = self.add_sidedef(sec_idx, midtexture=wall_tex)
            ld = self.add_linedef(v1, v2, sd, flags=ML_BLOCKING)
            self.add_seg(v1, v2, bam_angle, ld, side=0, offset=0)

        bbox = (y + height, y, x, x + width)
        self.add_subsector(4, first_seg, bbox=bbox)
        return sec_idx

    def add_two_sector_rooms(
        self,
        x: int = 0,
        y: int = 0,
        width1: int = 256,
        width2: int = 256,
        height: int = 256,
        *,
        floor1: int = 0,
        ceil1: int = 128,
        floor2: int = 16,
        ceil2: int = 128,
        floor_pic1: str = "FLOOR01",
        ceil_pic1: str = "CEIL01",
        floor_pic2: str = "FLOOR01",
        ceil_pic2: str = "CEIL01",
        wall_tex: str = "WALL01",
        light1: int = 160,
        light2: int = 160,
        portal_flags: int = ML_TWOSIDED,
        portal_special: int = 0,
        portal_args: Sequence[int] = (0, 0, 0, 0, 0),
    ) -> tuple[int, int, int]:
        """Create two adjacent rectangular rooms connected by a two-sided portal linedef.

        Returns `(sector1_index, sector2_index, portal_linedef_index)`.
        """
        sec1 = self.add_sector(
            floor_height=floor1,
            ceiling_height=ceil1,
            floor_pic=floor_pic1,
            ceiling_pic=ceil_pic1,
            light=light1,
        )
        sec2 = self.add_sector(
            floor_height=floor2,
            ceiling_height=ceil2,
            floor_pic=floor_pic2,
            ceiling_pic=ceil_pic2,
            light=light2,
        )

        x_mid = x + width1
        x_right = x + width1 + width2
        y_top = y + height

        # Left room vertices: TL, TM (top-mid), BM (bottom-mid), BL
        v_tl = self.add_vertex(x, y_top)
        v_tm = self.add_vertex(x_mid, y_top)
        v_bm = self.add_vertex(x_mid, y)
        v_bl = self.add_vertex(x, y)
        # Right room outer vertices: TR, BR
        v_tr = self.add_vertex(x_right, y_top)
        v_br = self.add_vertex(x_right, y)

        # Left room outer walls
        sd_l_top = self.add_sidedef(sec1, midtexture=wall_tex)
        ld_l_top = self.add_linedef(v_tl, v_tm, sd_l_top, flags=ML_BLOCKING)

        sd_l_bot = self.add_sidedef(sec1, midtexture=wall_tex)
        ld_l_bot = self.add_linedef(v_bm, v_bl, sd_l_bot, flags=ML_BLOCKING)

        sd_l_left = self.add_sidedef(sec1, midtexture=wall_tex)
        ld_l_left = self.add_linedef(v_bl, v_tl, sd_l_left, flags=ML_BLOCKING)

        # Shared portal line (v_tm -> v_bm: front=sec1 on right/west, back=sec2 on left/east)
        sd_portal_front = self.add_sidedef(
            sec1, toptexture=wall_tex, bottomtexture=wall_tex, midtexture="-"
        )
        sd_portal_back = self.add_sidedef(
            sec2, toptexture=wall_tex, bottomtexture=wall_tex, midtexture="-"
        )
        ld_portal = self.add_linedef(
            v_tm,
            v_bm,
            sd_portal_front,
            sd_portal_back,
            flags=portal_flags,
            special=portal_special,
            args=portal_args,
        )

        # Right room outer walls
        sd_r_top = self.add_sidedef(sec2, midtexture=wall_tex)
        ld_r_top = self.add_linedef(v_tm, v_tr, sd_r_top, flags=ML_BLOCKING)

        sd_r_right = self.add_sidedef(sec2, midtexture=wall_tex)
        ld_r_right = self.add_linedef(v_tr, v_br, sd_r_right, flags=ML_BLOCKING)

        sd_r_bot = self.add_sidedef(sec2, midtexture=wall_tex)
        ld_r_bot = self.add_linedef(v_br, v_bm, sd_r_bot, flags=ML_BLOCKING)

        # Subsector 0 (left room)
        s0_first = len(self.segs)
        self.add_seg(v_tl, v_tm, 0x0000, ld_l_top, side=0)
        self.add_seg(v_tm, v_bm, 0xC000, ld_portal, side=0)
        self.add_seg(v_bm, v_bl, 0x8000, ld_l_bot, side=0)
        self.add_seg(v_bl, v_tl, 0x4000, ld_l_left, side=0)
        bbox1 = (y_top, y, x, x_mid)
        ss0 = self.add_subsector(4, s0_first, bbox=bbox1)

        # Subsector 1 (right room)
        s1_first = len(self.segs)
        self.add_seg(v_tm, v_tr, 0x0000, ld_r_top, side=0)
        self.add_seg(v_tr, v_br, 0xC000, ld_r_right, side=0)
        self.add_seg(v_br, v_bm, 0x8000, ld_r_bot, side=0)
        self.add_seg(v_bm, v_tm, 0x4000, ld_portal, side=1)
        bbox2 = (y_top, y, x_mid, x_right)
        ss1 = self.add_subsector(4, s1_first, bbox=bbox2)

        # BSP node splitting along x = x_mid, directed downward (dx=0, dy=-height):
        # For R_PointOnSide: x <= x_mid -> side 0 (right=ss0), x > x_mid -> side 1 (left=ss1)
        self.add_node(
            x=x_mid,
            y=y_top,
            dx=0,
            dy=-height,
            bbox_right=bbox1,
            bbox_left=bbox2,
            child_right=NF_SUBSECTOR | ss0,
            child_left=NF_SUBSECTOR | ss1,
        )
        return sec1, sec2, ld_portal

    def _build_nodes_bytes(self) -> bytes:
        if self.nodes:
            return b"".join(self.nodes)

        num_ss = len(self.subsectors)
        if num_ss == 0:
            return b""
        if num_ss == 1:
            bbox = self._subsector_bboxes[0]
            top, bottom, left, right = bbox
            return struct.pack(
                "<12h2H",
                right,
                top,
                0,
                -(top - bottom or 128),
                top,
                bottom,
                left,
                right,
                top,
                bottom,
                left,
                right,
                NF_SUBSECTOR | 0,
                NF_SUBSECTOR | 0,
            )

        # Build a simple balanced binary tree of nodes over subsectors 0..num_ss-1
        generated_nodes: list[bytes] = []

        def _union_bbox(
            b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]
        ) -> tuple[int, int, int, int]:
            return (
                max(b1[0], b2[0]),
                min(b1[1], b2[1]),
                min(b1[2], b2[2]),
                max(b1[3], b2[3]),
            )

        def _build_range(
            indices: list[int],
        ) -> tuple[int, tuple[int, int, int, int]]:
            if len(indices) == 1:
                ss_idx = indices[0]
                return NF_SUBSECTOR | ss_idx, self._subsector_bboxes[ss_idx]
            # Sort by horizontal center (left + right)
            sorted_idx = sorted(
                indices,
                key=lambda i: (
                    self._subsector_bboxes[i][2] + self._subsector_bboxes[i][3],
                    self._subsector_bboxes[i][1] + self._subsector_bboxes[i][0],
                ),
            )
            mid = len(sorted_idx) // 2
            left_group = sorted_idx[:mid]
            right_group = sorted_idx[mid:]

            child0, bbox0 = _build_range(left_group)
            child1, bbox1 = _build_range(right_group)
            split_x = bbox0[3]
            split_y = max(bbox0[0], bbox1[0])
            split_dy = -max(1, split_y - min(bbox0[1], bbox1[1]))

            node_idx = len(generated_nodes)
            generated_nodes.append(
                struct.pack(
                    "<12h2H",
                    split_x,
                    split_y,
                    0,
                    split_dy,
                    bbox0[0],
                    bbox0[1],
                    bbox0[2],
                    bbox0[3],
                    bbox1[0],
                    bbox1[1],
                    bbox1[2],
                    bbox1[3],
                    child0 & 0xFFFF,
                    child1 & 0xFFFF,
                )
            )
            return node_idx, _union_bbox(bbox0, bbox1)

        _build_range(list(range(num_ss)))
        return b"".join(generated_nodes)

    def _build_reject_bytes(self) -> bytes:
        num_sec = len(self.sectors)
        size = math.ceil((num_sec * num_sec) / 8)
        return b"\x00" * size

    def _build_blockmap_bytes(self) -> bytes:
        if not self.vertices:
            # Minimal 1x1 empty blockmap
            # Header (4 shorts): orgx=0, orgy=0, width=1, height=1
            # Offset table (1 uint16): 5 (words from start of lump)
            # Blocklist (2 shorts): 0, -1 (0xFFFF)
            return struct.pack("<4hHhh", 0, 0, 1, 1, 5, 0, -1)

        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        org_x = min(xs)
        org_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        bmap_width = max(1, ((max_x - org_x) >> 7) + 1)
        bmap_height = max(1, ((max_y - org_y) >> 7) + 1)
        num_blocks = bmap_width * bmap_height

        # Word offset where blocklists begin: 4 header shorts + num_blocks offset shorts
        cur_word_offset = 4 + num_blocks
        offsets: list[int] = []
        lists_words: list[int] = []

        for by in range(bmap_height):
            by_min = org_y + by * 128
            by_max = by_min + 128
            for bx in range(bmap_width):
                bx_min = org_x + bx * 128
                bx_max = bx_min + 128
                offsets.append(cur_word_offset)
                block_lines: list[int] = [0]
                for ld_idx, (x1, y1, x2, y2) in enumerate(self._linedef_coords):
                    lx_min, lx_max = (x1, x2) if x1 <= x2 else (x2, x1)
                    ly_min, ly_max = (y1, y2) if y1 <= y2 else (y2, y1)
                    if (
                        lx_max >= bx_min
                        and lx_min <= bx_max
                        and ly_max >= by_min
                        and ly_min <= by_max
                    ):
                        block_lines.append(ld_idx)
                block_lines.append(-1)
                lists_words.extend(block_lines)
                cur_word_offset += len(block_lines)

        header = struct.pack("<4h", org_x, org_y, bmap_width, bmap_height)
        ofs_table = struct.pack(f"<{num_blocks}H", *offsets)
        lists_blob = struct.pack(f"<{len(lists_words)}h", *lists_words)
        return header + ofs_table + lists_blob

    def _build_behavior_bytes(self) -> bytes:
        if self._behavior is not None:
            return self._behavior
        # Minimal valid Hexen ACS BEHAVIOR lump:
        # "ACS\0" (4B), infoOffset=8 (4B), scriptCount=0 (4B), stringCount=0 (4B)
        return b"ACS\x00" + struct.pack("<iii", 8, 0, 0)

    def build_lumps(self) -> list[tuple[str, bytes]]:
        """Return the 12 Hexen map lumps as `(name, bytes)` pairs."""
        vertex_bytes = b"".join(struct.pack("<2h", x, y) for x, y in self.vertices)
        ssector_bytes = b"".join(
            struct.pack("<2h", numsegs, firstseg) for numsegs, firstseg in self.subsectors
        )
        return [
            (self.name, b""),
            ("THINGS", b"".join(self.things)),
            ("LINEDEFS", b"".join(self.linedefs)),
            ("SIDEDEFS", b"".join(self.sidedefs)),
            ("VERTEXES", vertex_bytes),
            ("SEGS", b"".join(self.segs)),
            ("SSECTORS", ssector_bytes),
            ("NODES", self._build_nodes_bytes()),
            ("SECTORS", b"".join(self.sectors)),
            ("REJECT", self._build_reject_bytes()),
            ("BLOCKMAP", self._build_blockmap_bytes()),
            ("BEHAVIOR", self._build_behavior_bytes()),
        ]

    def write_to(self, wad_builder: WadBuilder) -> WadBuilder:
        """Append all lumps of this map into `wad_builder`."""
        return wad_builder.add_map(self)
