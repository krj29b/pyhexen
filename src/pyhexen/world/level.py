# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Level geometry and map structure construction.

Port of P_LoadVertexes, P_LoadSectors, P_LoadSideDefs, P_LoadLineDefs,
P_LoadSubsectors, P_LoadNodes, P_LoadSegs, P_GroupLines, and P_SetupLevel (p_setup.c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from pyhexen.core.fixed import FRACBITS, MAXINT, MININT, c_div, fixed_div, to_uint32
from pyhexen.wad.maplumps import RawMapLumps, parse_map_lumps
from pyhexen.wad.wadfile import WadFile, WadStack, normalize_lump_name
from pyhexen.world.blockmap import MAPBLOCKSHIFT, MAXRADIUS, Blockmap
from pyhexen.world.bsp import BSP, NF_SUBSECTOR
from pyhexen.world.reject import RejectMatrix

# Bounding box indices (m_bbox.h)
BOXTOP: int = 0
BOXBOTTOM: int = 1
BOXLEFT: int = 2
BOXRIGHT: int = 3

# Linedef flags (xddefs.h)
ML_BLOCKING: int = 0x0001
ML_BLOCKMONSTERS: int = 0x0002
ML_TWOSIDED: int = 0x0004
ML_DONTPEGTOP: int = 0x0008
ML_DONTPEGBOTTOM: int = 0x0010
ML_SECRET: int = 0x0020
ML_SOUNDBLOCK: int = 0x0040
ML_DONTDRAW: int = 0x0080
ML_MAPPED: int = 0x0100
ML_REPEAT_SPECIAL: int = 0x0200
ML_SPAC_SHIFT: int = 10
ML_SPAC_MASK: int = 0x1C00

# Special activation types (xddefs.h)
SPAC_CROSS: int = 0
SPAC_USE: int = 1
SPAC_MCROSS: int = 2
SPAC_IMPACT: int = 3
SPAC_PUSH: int = 4
SPAC_PCROSS: int = 5


def GET_SPAC(flags: int) -> int:  # noqa: N802
    """Port of GET_SPAC(flags) macro (xddefs.h)."""
    return (flags & ML_SPAC_MASK) >> ML_SPAC_SHIFT


class SlopeType(IntEnum):
    """Port of slopetype_t (r_local.h)."""

    ST_HORIZONTAL = 0
    ST_VERTICAL = 1
    ST_POSITIVE = 2
    ST_NEGATIVE = 3


@dataclass(slots=True)
class Vertex:
    """Port of vertex_t (r_local.h). Coordinates are 16.16 fixed-point."""

    x: int
    y: int


@dataclass(slots=True)
class Sector:
    """Port of sector_t (r_local.h)."""

    index: int
    floor_height: int
    ceiling_height: int
    floor_pic: str
    ceiling_pic: str
    light_level: int
    special: int
    tag: int
    lines: list[Line] = field(default_factory=list)
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    sound_org_x: int = 0
    sound_org_y: int = 0
    thing_list: Any = None
    valid_count: int = 0
    seq_type: int = 0
    prev_floor_height: int = 0
    prev_ceiling_height: int = 0
    special_data: Any = None
    sound_target: Any = None
    sound_traversed: int = 0


@dataclass(slots=True)
class Side:
    """Port of side_t (r_local.h)."""

    index: int
    texture_offset: int
    row_offset: int
    top_texture: str
    bottom_texture: str
    mid_texture: str
    sector: Sector


@dataclass(slots=True)
class Line:
    """Port of line_t (r_local.h)."""

    index: int
    v1: Vertex
    v2: Vertex
    dx: int
    dy: int
    flags: int
    special: int
    args: tuple[int, int, int, int, int]
    sidenum: tuple[int, int]
    front_side: Side | None
    back_side: Side | None
    front_sector: Sector | None
    back_sector: Sector | None
    slopetype: SlopeType
    bbox: tuple[int, int, int, int]
    valid_count: int = 0
    special_data: Any = None
    is_polyobj: bool = False


@dataclass(slots=True)
class Seg:
    """Port of seg_t (r_local.h)."""

    index: int
    v1: Vertex
    v2: Vertex
    offset: int
    angle: int
    sidedef: Side
    linedef: Line
    front_sector: Sector
    back_sector: Sector | None


@dataclass(slots=True)
class Subsector:
    """Port of subsector_t (r_local.h)."""

    index: int
    sector: Sector
    num_lines: int
    first_line: int
    poly: Any = None


@dataclass(slots=True)
class Node:
    """Port of node_t (r_local.h)."""

    index: int
    x: int
    y: int
    dx: int
    dy: int
    bbox: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
    children: tuple[int, int]


@dataclass(slots=True)
class MapThing:
    """Port of mapthing_t (xddefs.h)."""

    tid: int
    x: int
    y: int
    z: int
    angle: int
    type: int
    flags: int
    special: int
    args: tuple[int, int, int, int, int]


@dataclass(slots=True)
class Level:
    """Constructed Hexen level owning all geometry, BSP, blockmap, reject, and raw BEHAVIOR."""

    name: str
    vertexes: list[Vertex]
    sectors: list[Sector]
    sides: list[Side]
    lines: list[Line]
    segs: list[Seg]
    subsectors: list[Subsector]
    nodes: list[Node]
    things: list[MapThing]
    bsp: BSP
    blockmap: Blockmap
    reject: RejectMatrix
    behavior_bytes: bytes
    polyobjs: dict[int, Any] = field(default_factory=dict)
    polyobjs_by_index: list[Any] = field(default_factory=list)

    @property
    def polyobjs_by_tag(self) -> dict[int, Any]:
        """Convenience property matching tag lookup."""
        return self.polyobjs

    def point_in_subsector(self, x: int, y: int) -> Subsector:
        """Port of R_PointInSubsector (r_main.c)."""
        return self.bsp.point_in_subsector(x, y)


def _clear_box() -> list[int]:
    """Port of M_ClearBox (m_bbox.c)."""
    return [MININT, MAXINT, MAXINT, MININT]


def _add_to_box(box: list[int], x: int, y: int) -> None:
    """Port of M_AddToBox (m_bbox.c)."""
    if x < box[BOXLEFT]:
        box[BOXLEFT] = x
    if x > box[BOXRIGHT]:
        box[BOXRIGHT] = x
    if y < box[BOXBOTTOM]:
        box[BOXBOTTOM] = y
    if y > box[BOXTOP]:
        box[BOXTOP] = y


def build_level_from_raw(raw: RawMapLumps) -> Level:
    """Construct a `Level` from `RawMapLumps` following `P_SetupLevel` (p_setup.c)."""
    # 1. P_LoadVertexes
    vertexes: list[Vertex] = [
        Vertex(x=int(v["x"]) << FRACBITS, y=int(v["y"]) << FRACBITS) for v in raw.vertexes
    ]

    # 2. P_LoadSectors
    sectors: list[Sector] = []
    for i, s in enumerate(raw.sectors):
        fh = int(s["floorheight"]) << FRACBITS
        ch = int(s["ceilingheight"]) << FRACBITS
        sectors.append(
            Sector(
                index=i,
                floor_height=fh,
                ceiling_height=ch,
                floor_pic=normalize_lump_name(bytes(s["floorpic"])),
                ceiling_pic=normalize_lump_name(bytes(s["ceilingpic"])),
                light_level=int(s["lightlevel"]),
                special=int(s["special"]),
                tag=int(s["tag"]),
                prev_floor_height=fh,
                prev_ceiling_height=ch,
            )
        )

    # 3. P_LoadSideDefs
    sides: list[Side] = []
    for i, sd in enumerate(raw.sidedefs):
        sec_idx = int(sd["sector"])
        sides.append(
            Side(
                index=i,
                texture_offset=int(sd["textureoffset"]) << FRACBITS,
                row_offset=int(sd["rowoffset"]) << FRACBITS,
                top_texture=normalize_lump_name(bytes(sd["toptexture"])),
                bottom_texture=normalize_lump_name(bytes(sd["bottomtexture"])),
                mid_texture=normalize_lump_name(bytes(sd["midtexture"])),
                sector=sectors[sec_idx],
            )
        )

    # 4. P_LoadLineDefs
    lines: list[Line] = []
    for i, mld in enumerate(raw.linedefs):
        v1 = vertexes[int(mld["v1"])]
        v2 = vertexes[int(mld["v2"])]
        dx = v2.x - v1.x
        dy = v2.y - v1.y
        if dx == 0:
            slopetype = SlopeType.ST_VERTICAL
        elif dy == 0:
            slopetype = SlopeType.ST_HORIZONTAL
        else:
            if fixed_div(dy, dx) > 0:
                slopetype = SlopeType.ST_POSITIVE
            else:
                slopetype = SlopeType.ST_NEGATIVE

        if v1.x < v2.x:
            box_left, box_right = v1.x, v2.x
        else:
            box_left, box_right = v2.x, v1.x
        if v1.y < v2.y:
            box_bottom, box_top = v1.y, v2.y
        else:
            box_bottom, box_top = v2.y, v1.y

        s0 = int(mld["sidenum"][0])
        s1 = int(mld["sidenum"][1])
        front_side = sides[s0] if s0 != -1 else None
        back_side = sides[s1] if s1 != -1 else None
        front_sector = front_side.sector if front_side is not None else None
        back_sector = back_side.sector if back_side is not None else None

        raw_args = mld["args"]
        args = (
            int(raw_args[0]),
            int(raw_args[1]),
            int(raw_args[2]),
            int(raw_args[3]),
            int(raw_args[4]),
        )
        lines.append(
            Line(
                index=i,
                v1=v1,
                v2=v2,
                dx=dx,
                dy=dy,
                flags=int(mld["flags"]) & 0xFFFF,
                special=int(mld["special"]),
                args=args,
                sidenum=(s0, s1),
                front_side=front_side,
                back_side=back_side,
                front_sector=front_sector,
                back_sector=back_sector,
                slopetype=slopetype,
                bbox=(box_top, box_bottom, box_left, box_right),
            )
        )

    # 5. P_LoadBlockMap
    blockmap = Blockmap.from_bytes(raw.blockmap, lines)

    # 6. P_LoadSegs
    segs: list[Seg] = []
    for i, ml in enumerate(raw.segs):
        v1 = vertexes[int(ml["v1"])]
        v2 = vertexes[int(ml["v2"])]
        angle = to_uint32(int(ml["angle"]) << 16)
        offset = int(ml["offset"]) << 16
        ldef = lines[int(ml["linedef"])]
        side_num = int(ml["side"])
        sidedef = sides[ldef.sidenum[side_num]]
        front_sector = sidedef.sector
        if ldef.flags & ML_TWOSIDED:
            other_side_idx = ldef.sidenum[side_num ^ 1]
            back_sector = sides[other_side_idx].sector if other_side_idx != -1 else None
        else:
            back_sector = None
        segs.append(
            Seg(
                index=i,
                v1=v1,
                v2=v2,
                offset=offset,
                angle=angle,
                sidedef=sidedef,
                linedef=ldef,
                front_sector=front_sector,
                back_sector=back_sector,
            )
        )

    # 7. P_LoadSubsectors + P_GroupLines subsector->sector assignment
    subsectors: list[Subsector] = []
    for i, ms in enumerate(raw.ssectors):
        num_lines = int(ms["numsegs"])
        first_line = int(ms["firstseg"])
        sec = segs[first_line].sidedef.sector if segs else sectors[0]
        subsectors.append(
            Subsector(
                index=i,
                sector=sec,
                num_lines=num_lines,
                first_line=first_line,
            )
        )

    # 8. P_LoadNodes
    nodes: list[Node] = []
    for i, mn in enumerate(raw.nodes):
        raw_bbox = mn["bbox"]
        bbox0 = (
            int(raw_bbox[0][0]) << FRACBITS,
            int(raw_bbox[0][1]) << FRACBITS,
            int(raw_bbox[0][2]) << FRACBITS,
            int(raw_bbox[0][3]) << FRACBITS,
        )
        bbox1 = (
            int(raw_bbox[1][0]) << FRACBITS,
            int(raw_bbox[1][1]) << FRACBITS,
            int(raw_bbox[1][2]) << FRACBITS,
            int(raw_bbox[1][3]) << FRACBITS,
        )
        children = (int(mn["children"][0]) & 0xFFFF, int(mn["children"][1]) & 0xFFFF)
        nodes.append(
            Node(
                index=i,
                x=int(mn["x"]) << FRACBITS,
                y=int(mn["y"]) << FRACBITS,
                dx=int(mn["dx"]) << FRACBITS,
                dy=int(mn["dy"]) << FRACBITS,
                bbox=(bbox0, bbox1),
                children=children,
            )
        )

    # 9. P_GroupLines: build sector line lists, sound origins, and blockboxes
    for sector in sectors:
        bbox = _clear_box()
        for li in lines:
            if li.front_sector is sector or li.back_sector is sector:
                sector.lines.append(li)
                _add_to_box(bbox, li.v1.x, li.v1.y)
                _add_to_box(bbox, li.v2.x, li.v2.y)

        if sector.lines:
            sector.sound_org_x = c_div(bbox[BOXRIGHT] + bbox[BOXLEFT], 2)
            sector.sound_org_y = c_div(bbox[BOXTOP] + bbox[BOXBOTTOM], 2)

            b_top = (bbox[BOXTOP] - blockmap.org_y + MAXRADIUS) >> MAPBLOCKSHIFT
            if b_top >= blockmap.height:
                b_top = blockmap.height - 1
            b_bot = (bbox[BOXBOTTOM] - blockmap.org_y - MAXRADIUS) >> MAPBLOCKSHIFT
            if b_bot < 0:
                b_bot = 0
            b_right = (bbox[BOXRIGHT] - blockmap.org_x + MAXRADIUS) >> MAPBLOCKSHIFT
            if b_right >= blockmap.width:
                b_right = blockmap.width - 1
            b_left = (bbox[BOXLEFT] - blockmap.org_x - MAXRADIUS) >> MAPBLOCKSHIFT
            if b_left < 0:
                b_left = 0
            sector.bbox = [b_top, b_bot, b_left, b_right]

    # 10. Things list
    things: list[MapThing] = []
    for mt in raw.things:
        t_args = mt["args"]
        things.append(
            MapThing(
                tid=int(mt["tid"]),
                x=int(mt["x"]),
                y=int(mt["y"]),
                z=int(mt["height"]),
                angle=int(mt["angle"]),
                type=int(mt["type"]),
                flags=int(mt["options"]),
                special=int(mt["special"]),
                args=(
                    int(t_args[0]),
                    int(t_args[1]),
                    int(t_args[2]),
                    int(t_args[3]),
                    int(t_args[4]),
                ),
            )
        )

    bsp = BSP(nodes=nodes, subsectors=subsectors)
    reject = RejectMatrix(data=raw.reject, num_sectors=len(sectors))

    level = Level(
        name=raw.name,
        vertexes=vertexes,
        sectors=sectors,
        sides=sides,
        lines=lines,
        segs=segs,
        subsectors=subsectors,
        nodes=nodes,
        things=things,
        bsp=bsp,
        blockmap=blockmap,
        reject=reject,
        behavior_bytes=raw.behavior,
    )

    from pyhexen.sim.polyobj import init_polyobjs

    init_polyobjs(level)

    return level


def load_level(wad: WadStack | WadFile, map_name: str) -> Level:
    """Parse map lumps from `wad` and construct a `Level` instance."""
    raw = parse_map_lumps(wad, map_name)
    return build_level_from_raw(raw)


__all__ = [
    "BOXBOTTOM",
    "BOXLEFT",
    "BOXRIGHT",
    "BOXTOP",
    "GET_SPAC",
    "Level",
    "Line",
    "ML_BLOCKING",
    "ML_BLOCKMONSTERS",
    "ML_DONTDRAW",
    "ML_DONTPEGBOTTOM",
    "ML_DONTPEGTOP",
    "ML_MAPPED",
    "ML_REPEAT_SPECIAL",
    "ML_SECRET",
    "ML_SOUNDBLOCK",
    "ML_SPAC_MASK",
    "ML_SPAC_SHIFT",
    "ML_TWOSIDED",
    "MapThing",
    "NF_SUBSECTOR",
    "Node",
    "SPAC_CROSS",
    "SPAC_IMPACT",
    "SPAC_MCROSS",
    "SPAC_PCROSS",
    "SPAC_PUSH",
    "SPAC_USE",
    "Sector",
    "Seg",
    "Side",
    "SlopeType",
    "Subsector",
    "Vertex",
    "build_level_from_raw",
    "load_level",
]
