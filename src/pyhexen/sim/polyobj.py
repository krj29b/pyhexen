# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Polyobject simulation data structures, line collection, and spawning routines.

Port of po_man.c, r_local.h, and p_local.h.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import (
    ANG90,
    ANG180,
    ANG_MAX,
    ANGLETOFINESHIFT,
    FINEMASK,
    fine_cosine,
    fine_sine,
)
from pyhexen.core.events import SoundEvent
from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div, fixed_div, fixed_mul
from pyhexen.sim.info_data import MobjFlag
from pyhexen.sim.map import check_position
from pyhexen.sim.maputl import box_on_line_side
from pyhexen.sim.mobj import Mobj
from pyhexen.sim.thinker import Thinker
from pyhexen.world.blockmap import MAPBLOCKSHIFT, MAXRADIUS, Blockmap
from pyhexen.world.level import (
    BOXBOTTOM,
    BOXLEFT,
    BOXRIGHT,
    BOXTOP,
    Line,
    Seg,
    SlopeType,
    Subsector,
    Vertex,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import Level

# Sound sequence base ID (po_man.c, sndseq.h)
SEQ_DOOR_STONE: int = 10

# Thing marker types (p_local.h)
PO_ANCHOR_TYPE: int = 3000
PO_SPAWN_TYPE: int = 3001
PO_SPAWNCRUSH_TYPE: int = 3002

# Linedef specials for polyobject collection (po_man.c)
PO_LINE_START: int = 1
PO_LINE_EXPLICIT: int = 5
PO_MAXPOLYSEGS: int = 64

# Sound sequence types (po_man.c, sndseq.h)
SEQTYPE_STONE: int = 0
SEQTYPE_NUMSEQ: int = 10


@dataclass(slots=True)
class DegenMobj:
    """Port of degenmobj_t (h2def.h, r_local.h).

    Coordinate container representing world position and sound origin.
    """

    x: int = 0
    y: int = 0
    z: int = 0


@dataclass(slots=True)
class PolyBlock:
    """Port of polyblock_t (r_local.h).

    Doubly-linked list node connecting polyobjects to Blockmap cells.
    """

    polyobj: Polyobj | None = None
    prev: PolyBlock | None = None
    next: PolyBlock | None = None


@dataclass(slots=True)
class Polyobj:
    """Port of polyobj_t (r_local.h, po_man.c).

    Represents a dynamic 2D wall polygon structure in PyHexen simulation.
    """

    tag: int
    index: int
    numsegs: int
    segs: list[Seg]
    start_spot: DegenMobj
    original_pts: list[Vertex] = field(default_factory=list)
    prev_pts: list[Vertex] = field(default_factory=list)
    angle: int = 0
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    valid_count: int = 0
    crush: bool = False
    seq_type: int = 0
    size: int = 0
    specialdata: Any = None
    subsector: Subsector | None = None
    mirror_tag: int = 0
    prev_x: int = 0
    prev_y: int = 0
    prev_angle: int = 0

    def capture_prev_state(self) -> None:
        """Copy current position and angle into prev_* fields for sub-tic render interpolation."""
        self.prev_x = self.start_spot.x
        self.prev_y = self.start_spot.y
        self.prev_angle = self.angle

    def to_state(self) -> dict[str, Any]:
        """Port of ArchivePolyobjs (sv_save.c:3175-3188)."""
        return {
            "tag": self.tag,
            "angle": self.angle,
            "start_spot_x": self.start_spot.x,
            "start_spot_y": self.start_spot.y,
        }

    def from_state(self, state: dict[str, Any], sim: GameSim | None = None) -> None:
        """Port of UnarchivePolyobjs (sv_save.c:3196-3218)."""
        target_angle = state["angle"]
        target_x = state["start_spot_x"]
        target_y = state["start_spot_y"]
        if sim is not None:
            po_rotate_polyobj(sim, self, target_angle)
            delta_x = target_x - self.start_spot.x
            delta_y = target_y - self.start_spot.y
            po_move_polyobj(sim, self, delta_x, delta_y)
        else:
            self.angle = target_angle
            self.start_spot.x = target_x
            self.start_spot.y = target_y

    @property
    def busy(self) -> bool:
        """Port of PO_Busy (po_man.c:1491-1504)."""
        return self.specialdata is not None

    def is_busy(self) -> bool:
        """Alias for busy property."""
        return self.busy

    @property
    def mirror(self) -> int:
        return self.mirror_tag

    @mirror.setter
    def mirror(self, val: int) -> None:
        self.mirror_tag = val

    @property
    def start_spot_x(self) -> int:
        return self.start_spot.x

    @start_spot_x.setter
    def start_spot_x(self, val: int) -> None:
        self.start_spot.x = val

    @property
    def start_spot_y(self) -> int:
        return self.start_spot.y

    @start_spot_y.setter
    def start_spot_y(self, val: int) -> None:
        self.start_spot.y = val

    def clear_specialdata(self) -> None:
        self.specialdata = None


def iter_find_poly_segs(level: Level, start_seg: Seg, tag: int) -> list[Seg]:
    """Port of IterFindPolySegs (po_man.c:1172-1197).

    Traces segs starting from start_seg in perimeter order until returning to
    start_seg.v1 coordinates.
    """
    poly_start_x = start_seg.v1.x
    poly_start_y = start_seg.v1.y
    collected_segs: list[Seg] = [start_seg]
    max_segs = max(PO_MAXPOLYSEGS, len(level.segs))

    curr_x = start_seg.v2.x
    curr_y = start_seg.v2.y

    while True:
        if curr_x == poly_start_x and curr_y == poly_start_y:
            break
        if len(collected_segs) > max_segs:
            raise RuntimeError("IterFindPolySegs:  Non-closed Polyobj located.\n")

        found = False
        for s in level.segs:
            if s.v1.x == curr_x and s.v1.y == curr_y:
                collected_segs.append(s)
                curr_x = s.v2.x
                curr_y = s.v2.y
                found = True
                break

        if not found:
            raise RuntimeError("IterFindPolySegs:  Non-closed Polyobj located.\n")

    return collected_segs


def spawn_polyobj(
    level: Level,
    index: int,
    tag: int,
    crush: bool,
    start_spot_x: int,
    start_spot_y: int,
) -> Polyobj:
    """Port of SpawnPolyobj (po_man.c:1206-1325)."""
    # 1. Search for PO_LINE_START (special 1)
    matching_start_lines = {
        id(s.linedef): s
        for s in level.segs
        if s.linedef.special == PO_LINE_START and s.linedef.args[0] == tag
    }

    if len(matching_start_lines) > 1:
        raise RuntimeError(f"SpawnPolyobj:  Polyobj {tag} already spawned.\n")

    if len(matching_start_lines) == 1:
        start_seg = next(iter(matching_start_lines.values()))
        first_ld = start_seg.linedef
        mirror_tag = first_ld.args[1]
        seq_type = first_ld.args[2]
        if seq_type < 0 or seq_type >= SEQTYPE_NUMSEQ:
            seq_type = 0

        first_ld.special = 0
        first_ld.args = (0, first_ld.args[1], first_ld.args[2], first_ld.args[3], first_ld.args[4])
        first_ld.is_polyobj = True

        collected_segs = iter_find_poly_segs(level, start_seg, tag)
        for s in collected_segs:
            s.linedef.is_polyobj = True

        return Polyobj(
            tag=tag,
            index=index,
            numsegs=len(collected_segs),
            segs=collected_segs,
            start_spot=DegenMobj(start_spot_x, start_spot_y, 0),
            crush=crush,
            seq_type=seq_type,
            mirror_tag=mirror_tag,
        )

    # 2. Search for PO_LINE_EXPLICIT (special 5)
    poly_seg_list: list[Seg] = []

    for j in range(1, PO_MAXPOLYSEGS):
        ps_index_old = len(poly_seg_list)

        # Pass A: Find segs matching order j
        for seg in level.segs:
            ld = seg.linedef
            if ld.special == PO_LINE_EXPLICIT and ld.args[0] == tag:
                order = ld.args[1]
                if order == 0:
                    raise RuntimeError(
                        f"SpawnPolyobj:  Explicit line missing order number (probably {j}) "
                        f"in poly {tag}.\n"
                    )
                if order == j:
                    poly_seg_list.append(seg)
                    if len(poly_seg_list) > PO_MAXPOLYSEGS:
                        raise RuntimeError("SpawnPolyobj:  psIndex > PO_MAXPOLYSEGS\n")

        # Pass B: Clear out specials for matched segs
        for seg in level.segs:
            ld = seg.linedef
            if ld.special == PO_LINE_EXPLICIT and ld.args[0] == tag and ld.args[1] == j:
                ld.special = 0
                ld.args = (0, ld.args[1], ld.args[2], ld.args[3], ld.args[4])

        # Pass C: Check if an explicit line order has been skipped
        if len(poly_seg_list) == ps_index_old:
            for seg in level.segs:
                ld = seg.linedef
                if ld.special == PO_LINE_EXPLICIT and ld.args[0] == tag:
                    raise RuntimeError(f"SpawnPolyobj:  Missing explicit line {j} for poly {tag}\n")

    if poly_seg_list:
        first_ld = poly_seg_list[0].linedef
        seq_type = first_ld.args[3]
        if seq_type < 0 or seq_type >= SEQTYPE_NUMSEQ:
            seq_type = 0

        mirror_tag = first_ld.args[2]
        # Propagate mirror tag to arg2 on the first line
        first_ld.args = (
            first_ld.args[0],
            mirror_tag,
            first_ld.args[2],
            first_ld.args[3],
            first_ld.args[4],
        )

        for seg in poly_seg_list:
            seg.linedef.is_polyobj = True

        return Polyobj(
            tag=tag,
            index=index,
            numsegs=len(poly_seg_list),
            segs=poly_seg_list,
            start_spot=DegenMobj(start_spot_x, start_spot_y, 0),
            crush=crush,
            seq_type=seq_type,
            mirror_tag=mirror_tag,
        )

    return Polyobj(
        tag=tag,
        index=index,
        numsegs=0,
        segs=[],
        start_spot=DegenMobj(start_spot_x, start_spot_y, 0),
        crush=crush,
    )


def translate_to_start_spot(level: Level, tag: int, origin_x: int, origin_y: int) -> None:
    """Port of TranslateToStartSpot (po_man.c:1333-1414)."""
    po = level.polyobjs.get(tag)
    if po is None:
        raise RuntimeError(f"TranslateToStartSpot:  Unable to match polyobj tag: {tag}\n")
    if not po.segs:
        raise RuntimeError(
            f"TranslateToStartSpot:  Anchor point located without a StartSpot point: {tag}\n"
        )

    po.original_pts = []
    po.prev_pts = []
    delta_x = origin_x - po.start_spot.x
    delta_y = origin_y - po.start_spot.y

    seen_line_ids: set[int] = set()
    seen_v1_ids: set[int] = set()
    avg_map_x = 0
    avg_map_y = 0

    for seg in po.segs:
        ld = seg.linedef
        if id(ld) not in seen_line_ids:
            seen_line_ids.add(id(ld))
            top, bottom, left, right = ld.bbox
            ld.bbox = (top - delta_y, bottom - delta_y, left - delta_x, right - delta_x)
            ld.is_polyobj = True

        v1 = seg.v1
        if id(v1) not in seen_v1_ids:
            seen_v1_ids.add(id(v1))
            v1.x -= delta_x
            v1.y -= delta_y

        avg_map_x += v1.x >> FRACBITS
        avg_map_y += v1.y >> FRACBITS

        po.original_pts.append(Vertex(x=v1.x - po.start_spot.x, y=v1.y - po.start_spot.y))
        po.prev_pts.append(Vertex(x=v1.x, y=v1.y))

    # Also translate any v2 not shared with any v1
    for seg in po.segs:
        v2 = seg.v2
        if id(v2) not in seen_v1_ids:
            seen_v1_ids.add(id(v2))
            v2.x -= delta_x
            v2.y -= delta_y

    avg_x = c_div(avg_map_x, po.numsegs) << FRACBITS
    avg_y = c_div(avg_map_y, po.numsegs) << FRACBITS

    sub = level.point_in_subsector(avg_x, avg_y)
    if sub.poly is not None:
        raise RuntimeError(
            f"PO_TranslateToStartSpot:  Multiple polyobjs in a single subsector: "
            f"subsector={sub.index}, existing_poly={sub.poly.tag}, new_poly={po.tag}\n"
        )
    sub.poly = po
    po.subsector = sub
    po.prev_x = po.start_spot.x
    po.prev_y = po.start_spot.y
    po.prev_angle = po.angle


def link_polyobj(po: Polyobj, bm: Blockmap) -> None:
    """Port of LinkPolyobj (po_man.c:960-1040)."""
    if not po.segs:
        return

    left_x = min(seg.v1.x for seg in po.segs)
    right_x = max(seg.v1.x for seg in po.segs)
    bottom_y = min(seg.v1.y for seg in po.segs)
    top_y = max(seg.v1.y for seg in po.segs)

    po.bbox = [0, 0, 0, 0]
    po.bbox[BOXRIGHT] = (right_x - bm.org_x) >> MAPBLOCKSHIFT
    po.bbox[BOXLEFT] = (left_x - bm.org_x) >> MAPBLOCKSHIFT
    po.bbox[BOXTOP] = (top_y - bm.org_y) >> MAPBLOCKSHIFT
    po.bbox[BOXBOTTOM] = (bottom_y - bm.org_y) >> MAPBLOCKSHIFT

    for j in range(po.bbox[BOXBOTTOM], po.bbox[BOXTOP] + 1):
        if j < 0 or j >= bm.height:
            continue
        row_offset = j * bm.width
        for i in range(po.bbox[BOXLEFT], po.bbox[BOXRIGHT] + 1):
            if i < 0 or i >= bm.width:
                continue
            block_idx = row_offset + i
            link = bm.poly_block_map[block_idx]
            if link is None:
                bm.poly_block_map[block_idx] = PolyBlock(polyobj=po)
                continue
            curr = link
            while curr.next is not None and curr.polyobj is not None:
                curr = curr.next
            if curr.polyobj is None:
                curr.polyobj = po
            else:
                new_node = PolyBlock(polyobj=po, prev=curr)
                curr.next = new_node


def unlink_polyobj(po: Polyobj, bm: Blockmap) -> None:
    """Port of UnLinkPolyobj (po_man.c:924-951)."""
    for j in range(po.bbox[BOXBOTTOM], po.bbox[BOXTOP] + 1):
        if j < 0 or j >= bm.height:
            continue
        row_offset = j * bm.width
        for i in range(po.bbox[BOXLEFT], po.bbox[BOXRIGHT] + 1):
            if i < 0 or i >= bm.width:
                continue
            block_idx = row_offset + i
            curr = bm.poly_block_map[block_idx]
            while curr is not None and curr.polyobj is not po:
                curr = curr.next
            if curr is not None:
                curr.polyobj = None


def init_polyobjs(level: Level) -> list[Polyobj]:
    """Port of PO_Init (po_man.c:1422-1483).

    4-pass initialization routine:
    Pass 1: Parse spawn spots (3001, 3002) and collect constituent lines.
    Pass 2: Parse anchor spots (3000) and translate coordinates.
    Pass 3: Validate that every spawn spot has an anchor point.
    Pass 4: Spatially link polyobjects into Blockmap.poly_block_map.
    """
    if level.polyobjs:
        return level.polyobjs_by_index

    poly_index = 0

    spawn_things = [t for t in level.things if t.type in (PO_SPAWN_TYPE, PO_SPAWNCRUSH_TYPE)]
    anchor_things = [t for t in level.things if t.type == PO_ANCHOR_TYPE]
    if not spawn_things and not anchor_things:
        return []

    # Pass 1: Find spawn spots and collect polyobject lines
    for t in spawn_things:
        tag = t.angle
        if tag in level.polyobjs:
            raise RuntimeError(f"SpawnPolyobj:  Polyobj {tag} already spawned.\n")

        is_crush = t.type == PO_SPAWNCRUSH_TYPE
        po = spawn_polyobj(
            level=level,
            index=poly_index,
            tag=tag,
            crush=is_crush,
            start_spot_x=t.x << FRACBITS,
            start_spot_y=t.y << FRACBITS,
        )
        level.polyobjs[tag] = po
        level.polyobjs_by_index.append(po)
        poly_index += 1

    # Pass 2: Find anchor points and translate to spawn spots
    for t in level.things:
        if t.type == PO_ANCHOR_TYPE:
            translate_to_start_spot(
                level=level,
                tag=t.angle,
                origin_x=t.x << FRACBITS,
                origin_y=t.y << FRACBITS,
            )

    # Pass 3: Validate that every spawned polyobject received anchor points
    for po in level.polyobjs_by_index:
        if not po.original_pts:
            raise RuntimeError(f"PO_Init:  StartSpot located without an Anchor point: {po.tag}\n")

    # Pass 4: Link into dynamic blockmap
    for po in level.polyobjs_by_index:
        link_polyobj(po, level.blockmap)

    return level.polyobjs_by_index


def get_polyobj(level: Level, tag: int) -> Polyobj | None:
    """Port of GetPolyobj (po_man.c:559-571)."""
    return level.polyobjs.get(tag)


def get_polyobj_mirror(level: Level, tag: int) -> int:
    """Port of GetPolyobjMirror (po_man.c:579-591)."""
    po = level.polyobjs.get(tag)
    if po is not None:
        if po.mirror_tag:
            return po.mirror_tag
        if po.segs and len(po.segs[0].linedef.args) > 1:
            return po.segs[0].linedef.args[1]
    return 0


def start_poly_sound(sim: GameSim, poly: Polyobj) -> None:
    """Emit event to start polyobject door sound sequence.

    Port of SN_StartSequence (po_man.c).
    """
    seq_id = SEQ_DOOR_STONE + (poly.seq_type if 0 <= poly.seq_type < SEQTYPE_NUMSEQ else 0)
    sim.events.emit(
        SoundEvent(
            sound_id=seq_id,
            origin_id=id(poly),
            x=poly.start_spot.x,
            y=poly.start_spot.y,
            z=poly.start_spot.z,
            stop=False,
        )
    )


def stop_poly_sound(sim: GameSim, poly: Polyobj) -> None:
    """Emit event to stop polyobject door sound sequence.

    Port of SN_StopSequence (po_man.c).
    """
    seq_id = SEQ_DOOR_STONE + (poly.seq_type if 0 <= poly.seq_type < SEQTYPE_NUMSEQ else 0)
    sim.events.emit(
        SoundEvent(
            sound_id=seq_id,
            origin_id=id(poly),
            x=poly.start_spot.x,
            y=poly.start_spot.y,
            z=poly.start_spot.z,
            stop=True,
        )
    )


def rotate_pt(
    an: int,
    pt_or_x: Any,
    origin_x_or_y: int,
    origin_y_or_start_x: int = 0,
    start_y: int = 0,
) -> tuple[int, int] | None:
    """Rotate point `pt` by fine angle `an` around origin.

    Port of RotatePt (po_man.c:818-834).
    """
    if isinstance(pt_or_x, Vertex):
        trx = pt_or_x.x
        try_ = pt_or_x.y
        origin_x = origin_x_or_y
        origin_y = origin_y_or_start_x
        gxt = fixed_mul(trx, fine_cosine(an))
        gyt = fixed_mul(try_, fine_sine(an))
        pt_or_x.x = (gxt - gyt) + origin_x

        gxt = fixed_mul(trx, fine_sine(an))
        gyt = fixed_mul(try_, fine_cosine(an))
        pt_or_x.y = (gyt + gxt) + origin_y
        return None
    trx = int(pt_or_x)
    try_ = int(origin_x_or_y)
    origin_x = int(origin_y_or_start_x)
    origin_y = int(start_y)
    gxt = fixed_mul(trx, fine_cosine(an))
    gyt = fixed_mul(try_, fine_sine(an))
    res_x = (gxt - gyt) + origin_x
    gxt = fixed_mul(trx, fine_sine(an))
    gyt = fixed_mul(try_, fine_cosine(an))
    res_y = (gyt + gxt) + origin_y
    return res_x, res_y


def update_seg_bbox(seg: Seg) -> None:
    """Update linedef bbox, dx, dy, and slopetype from seg vertices.

    Port of UpdateSegBBox (po_man.c:658-707).
    """
    line = seg.linedef
    if seg.v1.x < seg.v2.x:
        box_left = seg.v1.x
        box_right = seg.v2.x
    else:
        box_left = seg.v2.x
        box_right = seg.v1.x

    if seg.v1.y < seg.v2.y:
        box_bottom = seg.v1.y
        box_top = seg.v2.y
    else:
        box_bottom = seg.v2.y
        box_top = seg.v1.y

    line.bbox = (box_top, box_bottom, box_left, box_right)

    line.dx = line.v2.x - line.v1.x
    line.dy = line.v2.y - line.v1.y
    if line.dx == 0:
        line.slopetype = SlopeType.ST_VERTICAL
    elif line.dy == 0:
        line.slopetype = SlopeType.ST_HORIZONTAL
    else:
        if fixed_div(line.dy, line.dx) > 0:
            line.slopetype = SlopeType.ST_POSITIVE
        else:
            line.slopetype = SlopeType.ST_NEGATIVE


def damage_mobj(
    target: Mobj,
    inflictor: Mobj | None = None,
    source: Mobj | None = None,
    damage: int = 0,
) -> None:
    """Apply damage to `target` mobj.

    Port of P_DamageMobj (p_inter.c:1735).
    """
    _ = (inflictor, source)
    if not (target.flags & MobjFlag.MF_SHOOTABLE) and target.player is None:
        return
    if target.health <= 0:
        return
    target.health -= damage
    if target.player is not None:
        target.player.health = target.health
        target.player.damagecount += damage


def thrust_mobj(sim: GameSim, mobj: Mobj, seg: Seg, po: Polyobj) -> None:
    """Impart perpendicular thrust momentum to `mobj` and deal crush damage if pinned.

    Port of ThrustMobj (po_man.c:599-650).
    """
    if not (mobj.flags & MobjFlag.MF_SHOOTABLE) and mobj.player is None:
        return
    if not (mobj.flags & 2) and mobj.player is None:
        return

    thrust_angle = ((seg.angle - ANG90) & 0xFFFFFFFF) >> ANGLETOFINESHIFT

    pe = po.specialdata
    if pe is not None:
        speed = abs(getattr(pe, "speed", 0))
        is_rotate = type(pe).__name__ == "RotatePolyThinker" or (
            type(pe).__name__ == "PolyDoorThinker"
            and getattr(pe, "door_type", None) == PolyDoorType.PODOOR_SWING
        )
        if is_rotate:
            force = speed >> 8
        else:
            force = speed >> 3
        if force < FRACUNIT:
            force = FRACUNIT
        elif force > 4 * FRACUNIT:
            force = 4 * FRACUNIT
    else:
        force = FRACUNIT

    thrust_x = fixed_mul(force, fine_cosine(thrust_angle))
    thrust_y = fixed_mul(force, fine_sine(thrust_angle))
    mobj.momx += thrust_x
    mobj.momy += thrust_y

    if po.crush:
        if not check_position(sim, mobj, mobj.x + thrust_x, mobj.y + thrust_y):
            damage_mobj(mobj, None, None, 3)


def check_mobj_blocking(sim: GameSim, seg: Seg, po: Polyobj) -> bool:
    """Test if solid mobjs or players intersect `seg`, thrusting them and returning blocking status.

    Port of CheckMobjBlocking (po_man.c:1048-1106).
    """
    level = sim.level
    if level is None:
        return False
    bm = level.blockmap
    ld = seg.linedef

    top_box, bottom_box, left_box, right_box = ld.bbox

    top = (top_box - bm.org_y + MAXRADIUS) >> MAPBLOCKSHIFT
    bottom = (bottom_box - bm.org_y - MAXRADIUS) >> MAPBLOCKSHIFT
    left = (left_box - bm.org_x - MAXRADIUS) >> MAPBLOCKSHIFT
    right = (right_box - bm.org_x + MAXRADIUS) >> MAPBLOCKSHIFT

    bottom = max(0, min(bottom, bm.height - 1))
    top = max(0, min(top, bm.height - 1))
    left = max(0, min(left, bm.width - 1))
    right = max(0, min(right, bm.width - 1))

    blocked = False
    seen_mobjs: set[int] = set()

    po_left = min(min(s.v1.x, s.v2.x) for s in po.segs)
    po_right = max(max(s.v1.x, s.v2.x) for s in po.segs)
    po_bottom = min(min(s.v1.y, s.v2.y) for s in po.segs)
    po_top = max(max(s.v1.y, s.v2.y) for s in po.segs)

    for by in range(bottom, top + 1):
        row_offset = by * bm.width
        for bx in range(left, right + 1):
            mobj = bm.block_links[row_offset + bx]
            while mobj is not None:
                nxt = getattr(mobj, "bnext", None)
                if id(mobj) not in seen_mobjs:
                    seen_mobjs.add(id(mobj))
                    if (mobj.flags & MobjFlag.MF_SOLID) or mobj.player is not None:
                        tmbox = (
                            mobj.y + mobj.radius,
                            mobj.y - mobj.radius,
                            mobj.x - mobj.radius,
                            mobj.x + mobj.radius,
                        )
                        is_intersecting_line = (
                            not (
                                tmbox[BOXRIGHT] <= left_box
                                or tmbox[BOXLEFT] >= right_box
                                or tmbox[BOXTOP] <= bottom_box
                                or tmbox[BOXBOTTOM] >= top_box
                            )
                            and box_on_line_side(tmbox, ld) == -1
                        )
                        is_inside_poly = (
                            po_left <= mobj.x <= po_right and po_bottom <= mobj.y <= po_top
                        )
                        if is_intersecting_line or is_inside_poly:
                            thrust_mobj(sim, mobj, seg, po)
                            blocked = True
                mobj = nxt

    # Also check any players in sim.players if not already encountered in blockmap
    for p in sim.players:
        mobj = p.mo
        if mobj is not None and id(mobj) not in seen_mobjs:
            seen_mobjs.add(id(mobj))
            if (mobj.flags & MobjFlag.MF_SOLID) or mobj.player is not None:
                tmbox = (
                    mobj.y + mobj.radius,
                    mobj.y - mobj.radius,
                    mobj.x - mobj.radius,
                    mobj.x + mobj.radius,
                )
                is_intersecting_line = (
                    not (
                        tmbox[BOXRIGHT] <= left_box
                        or tmbox[BOXLEFT] >= right_box
                        or tmbox[BOXTOP] <= bottom_box
                        or tmbox[BOXBOTTOM] >= top_box
                    )
                    and box_on_line_side(tmbox, ld) == -1
                )
                is_inside_poly = po_left <= mobj.x <= po_right and po_bottom <= mobj.y <= po_top
                if is_intersecting_line or is_inside_poly:
                    thrust_mobj(sim, mobj, seg, po)
                    blocked = True

    return blocked


def po_move_polyobj(sim: GameSim, tag_or_po: int | Polyobj, x: int, y: int) -> bool:
    """Move polyobject by relative fixed-point offset `(x, y)`.

    Port of PO_MovePolyobj (po_man.c:715-810).
    """
    level = sim.level
    if level is None:
        return False
    po = tag_or_po if isinstance(tag_or_po, Polyobj) else get_polyobj(level, tag_or_po)
    if po is None:
        raise RuntimeError(f"PO_MovePolyobj: Invalid polyobj number: {tag_or_po}\n")

    unlink_polyobj(po, level.blockmap)

    blocked = False
    seen_lines: set[int] = set()
    seen_vertices: set[int] = set()

    for k, seg in enumerate(po.segs):
        ld = seg.linedef
        if id(ld) not in seen_lines:
            seen_lines.add(id(ld))
            top, bottom, left, right = ld.bbox
            ld.bbox = (top + y, bottom + y, left + x, right + x)

        v1 = seg.v1
        if id(v1) not in seen_vertices:
            seen_vertices.add(id(v1))
            v1.x += x
            v1.y += y

        if k < len(po.prev_pts):
            po.prev_pts[k].x += x
            po.prev_pts[k].y += y

    for seg in po.segs:
        v2 = seg.v2
        if id(v2) not in seen_vertices:
            seen_vertices.add(id(v2))
            v2.x += x
            v2.y += y

    for seg in po.segs:
        if check_mobj_blocking(sim, seg, po):
            blocked = True

    if blocked:
        seen_lines.clear()
        seen_vertices.clear()
        for k, seg in enumerate(po.segs):
            ld = seg.linedef
            if id(ld) not in seen_lines:
                seen_lines.add(id(ld))
                top, bottom, left, right = ld.bbox
                ld.bbox = (top - y, bottom - y, left - x, right - x)

            v1 = seg.v1
            if id(v1) not in seen_vertices:
                seen_vertices.add(id(v1))
                v1.x -= x
                v1.y -= y

            if k < len(po.prev_pts):
                po.prev_pts[k].x -= x
                po.prev_pts[k].y -= y

        for seg in po.segs:
            v2 = seg.v2
            if id(v2) not in seen_vertices:
                seen_vertices.add(id(v2))
                v2.x -= x
                v2.y -= y

        link_polyobj(po, level.blockmap)
        return False

    po.start_spot.x += x
    po.start_spot.y += y
    link_polyobj(po, level.blockmap)
    return True


def po_rotate_polyobj(sim: GameSim, tag_or_po: int | Polyobj, angle: int) -> bool:
    """Rotate polyobject by BAM angle increment `angle`.

    Port of PO_RotatePolyobj (po_man.c:842-916).
    """
    level = sim.level
    if level is None:
        return False
    po = tag_or_po if isinstance(tag_or_po, Polyobj) else get_polyobj(level, tag_or_po)
    if po is None:
        raise RuntimeError(f"PO_RotatePolyobj: Invalid polyobj number: {tag_or_po}\n")

    an = ((po.angle + angle) & 0xFFFFFFFF) >> ANGLETOFINESHIFT

    unlink_polyobj(po, level.blockmap)

    for k, seg in enumerate(po.segs):
        if k < len(po.prev_pts):
            po.prev_pts[k].x = seg.v1.x
            po.prev_pts[k].y = seg.v1.y
        if k < len(po.original_pts):
            seg.v1.x = po.original_pts[k].x
            seg.v1.y = po.original_pts[k].y
        rotate_pt(an, seg.v1, po.start_spot.x, po.start_spot.y)

    blocked = False
    seen_lines: set[int] = set()

    for seg in po.segs:
        if check_mobj_blocking(sim, seg, po):
            blocked = True
        if id(seg.linedef) not in seen_lines:
            seen_lines.add(id(seg.linedef))
            update_seg_bbox(seg)
        seg.angle = (seg.angle + angle) & 0xFFFFFFFF

    if blocked:
        for k, seg in enumerate(po.segs):
            if k < len(po.prev_pts):
                seg.v1.x = po.prev_pts[k].x
                seg.v1.y = po.prev_pts[k].y

        seen_lines.clear()
        for seg in po.segs:
            if id(seg.linedef) not in seen_lines:
                seen_lines.add(id(seg.linedef))
                update_seg_bbox(seg)
            seg.angle = (seg.angle - angle) & 0xFFFFFFFF

        link_polyobj(po, level.blockmap)
        return False

    po.angle = (po.angle + angle) & 0xFFFFFFFF
    link_polyobj(po, level.blockmap)
    return True


# PascalCase aliases matching Chocolate Doom C API names
PO_MovePolyobj = po_move_polyobj
PO_RotatePolyobj = po_rotate_polyobj
CheckMobjBlocking = check_mobj_blocking
ThrustMobj = thrust_mobj
GetPolyobj = get_polyobj
GetPolyobjMirror = get_polyobj_mirror


class RotatePolyThinker(Thinker):
    """Port of T_RotatePoly and polyevent_t (po_man.c:75-106)."""

    __slots__ = ("poly", "polyobj", "speed", "dist")

    def __init__(self, poly: Polyobj | int, speed: int, dist: int) -> None:
        super().__init__()
        if isinstance(poly, Polyobj):
            self.poly: Polyobj = poly
            self.polyobj: int = poly.tag
        else:
            self.poly = None  # type: ignore[assignment]
            self.polyobj = poly
        self.speed: int = speed
        self.dist: int = dist

    def tick(self, sim: GameSim) -> None:
        if self.poly is None:
            if sim.level is not None:
                self.poly = sim.level.polyobjs.get(self.polyobj)
        if self.poly is None:
            self.removed = True
            return

        poly = self.poly
        if po_rotate_polyobj(sim, poly, self.speed):
            abs_speed = abs(self.speed)
            if self.dist == -1:
                return
            self.dist -= abs_speed
            if self.dist <= 0:
                if poly.specialdata is self:
                    poly.specialdata = None
                stop_poly_sound(sim, poly)
                sim.poly_finished(poly.tag)
                self.removed = True
                return
            if self.dist < abs_speed:
                self.speed = self.dist * (-1 if self.speed < 0 else 1)

    def state_tuple(self) -> tuple[Any, ...]:
        return ("RotatePolyThinker", self.polyobj, self.speed, self.dist)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "RotatePolyThinker",
            "polyobj": self.polyobj,
            "speed": self.speed,
            "dist": self.dist,
        }


def _cardinal_cos_sin(fine_an: int) -> tuple[int, int]:
    """Return exact (cos, sin) in 16.16 fixed point, using exact 1.0 for cardinal angles."""
    an = fine_an & FINEMASK
    if an == 0:
        return FRACUNIT, 0
    if an == 2048:
        return 0, FRACUNIT
    if an == 4096:
        return -FRACUNIT, 0
    if an == 6144:
        return 0, -FRACUNIT
    return fine_cosine(an), fine_sine(an)


class MovePolyThinker(Thinker):
    """Port of T_MovePoly and polyevent_t (po_man.c:210-238)."""

    __slots__ = ("poly", "polyobj", "speed", "dist", "angle", "x_speed", "y_speed")

    def __init__(
        self,
        poly: Polyobj | int,
        speed: int,
        dist: int,
        angle: int,
        x_speed: int,
        y_speed: int,
    ) -> None:
        super().__init__()
        if isinstance(poly, Polyobj):
            self.poly: Polyobj = poly
            self.polyobj: int = poly.tag
        else:
            self.poly = None  # type: ignore[assignment]
            self.polyobj = poly
        self.speed: int = speed
        self.dist: int = dist
        self.angle: int = angle
        self.x_speed: int = x_speed
        self.y_speed: int = y_speed

    def tick(self, sim: GameSim) -> None:
        if self.poly is None:
            if sim.level is not None:
                self.poly = sim.level.polyobjs.get(self.polyobj)
        if self.poly is None:
            self.removed = True
            return

        poly = self.poly
        abs_speed = abs(self.speed)
        if self.dist != -1 and self.dist < abs_speed:
            self.speed = self.dist * (-1 if self.speed < 0 else 1)
            cos_val, sin_val = _cardinal_cos_sin(self.angle)
            self.x_speed = fixed_mul(self.speed, cos_val)
            self.y_speed = fixed_mul(self.speed, sin_val)

        if po_move_polyobj(sim, poly, self.x_speed, self.y_speed):
            abs_speed = abs(self.speed)
            self.dist -= abs_speed
            if self.dist <= 0:
                if poly.specialdata is self:
                    poly.specialdata = None
                stop_poly_sound(sim, poly)
                sim.poly_finished(poly.tag)
                self.removed = True
                return
            if self.dist < abs_speed:
                self.speed = self.dist * (-1 if self.speed < 0 else 1)
                cos_val, sin_val = _cardinal_cos_sin(self.angle)
                self.x_speed = fixed_mul(self.speed, cos_val)
                self.y_speed = fixed_mul(self.speed, sin_val)

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "MovePolyThinker",
            self.polyobj,
            self.speed,
            self.dist,
            self.angle,
            self.x_speed,
            self.y_speed,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "MovePolyThinker",
            "polyobj": self.polyobj,
            "speed": self.speed,
            "dist": self.dist,
            "angle": self.angle,
            "x_speed": self.x_speed,
            "y_speed": self.y_speed,
        }


class PolyDoorType(IntEnum):
    """Port of podoortype_t (p_local.h:365-369)."""

    PODOOR_NONE = 0
    PODOOR_SLIDE = 1
    PODOOR_SWING = 2


class PolyDoorThinker(Thinker):
    """Port of T_PolyDoor and polydoor_t (po_man.c:329-448)."""

    __slots__ = (
        "poly",
        "polyobj",
        "door_type",
        "speed",
        "dist",
        "total_dist",
        "direction",
        "x_speed",
        "y_speed",
        "tics",
        "wait_tics",
        "close",
    )

    def __init__(
        self,
        poly: Polyobj | int,
        door_type: int | PolyDoorType,
        speed: int,
        dist: int,
        total_dist: int,
        direction: int,
        x_speed: int = 0,
        y_speed: int = 0,
        tics: int = 0,
        wait_tics: int = 0,
        close: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(poly, Polyobj):
            self.poly: Polyobj = poly
            self.polyobj: int = poly.tag
        else:
            self.poly = None  # type: ignore[assignment]
            self.polyobj = poly
        self.door_type: int = int(door_type)
        self.speed: int = speed
        self.dist: int = dist
        self.total_dist: int = total_dist
        self.direction: int = direction
        self.x_speed: int = x_speed
        self.y_speed: int = y_speed
        self.tics: int = tics
        self.wait_tics: int = wait_tics
        self.close: bool = close

    @property
    def type(self) -> int:
        return self.door_type

    def tick(self, sim: GameSim) -> None:
        if self.poly is None:
            if sim.level is not None:
                self.poly = sim.level.polyobjs.get(self.polyobj)
        if self.poly is None:
            self.removed = True
            return

        poly = self.poly
        if self.tics:
            self.tics -= 1
            if not self.tics:
                start_poly_sound(sim, poly)
            return

        if self.door_type == PolyDoorType.PODOOR_SLIDE:
            if po_move_polyobj(sim, poly, self.x_speed, self.y_speed):
                abs_speed = abs(self.speed)
                self.dist -= abs_speed
                if self.dist <= 0:
                    stop_poly_sound(sim, poly)
                    if not self.close:
                        self.dist = self.total_dist
                        self.close = True
                        self.tics = self.wait_tics
                        self.direction = ((ANG_MAX >> ANGLETOFINESHIFT) - self.direction) & FINEMASK
                        self.x_speed = -self.x_speed
                        self.y_speed = -self.y_speed
                    else:
                        if poly.specialdata is self:
                            poly.specialdata = None
                        sim.poly_finished(poly.tag)
                        self.removed = True
            else:
                if poly.crush or not self.close:
                    return
                # Obstacle reversal for non-crusher while closing
                self.dist = self.total_dist - self.dist
                self.direction = ((ANG_MAX >> ANGLETOFINESHIFT) - self.direction) & FINEMASK
                self.x_speed = -self.x_speed
                self.y_speed = -self.y_speed
                self.close = False
                start_poly_sound(sim, poly)
        elif self.door_type == PolyDoorType.PODOOR_SWING:
            if po_rotate_polyobj(sim, poly, self.speed):
                abs_speed = abs(self.speed)
                if self.dist == -1:
                    return
                self.dist -= abs_speed
                if self.dist <= 0:
                    stop_poly_sound(sim, poly)
                    if not self.close:
                        self.dist = self.total_dist
                        self.close = True
                        self.tics = self.wait_tics
                        self.speed = -self.speed
                    else:
                        if poly.specialdata is self:
                            poly.specialdata = None
                        sim.poly_finished(poly.tag)
                        self.removed = True
            else:
                if poly.crush or not self.close:
                    return
                # Obstacle reversal for non-crusher while closing
                self.dist = self.total_dist - self.dist
                self.speed = -self.speed
                self.close = False
                start_poly_sound(sim, poly)

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "PolyDoorThinker",
            self.polyobj,
            self.door_type,
            self.speed,
            self.dist,
            self.total_dist,
            self.direction,
            self.x_speed,
            self.y_speed,
            self.tics,
            self.wait_tics,
            self.close,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "PolyDoorThinker",
            "polyobj": self.polyobj,
            "door_type": int(self.door_type),
            "speed": self.speed,
            "dist": self.dist,
            "total_dist": self.total_dist,
            "direction": self.direction,
            "x_speed": self.x_speed,
            "y_speed": self.y_speed,
            "tics": self.tics,
            "wait_tics": self.wait_tics,
            "close": self.close,
        }


def ev_rotate_poly(
    sim: GameSim,
    line: Line | None,
    args: Sequence[int],
    direction: int,
    override: bool,
) -> bool:
    """Port of EV_RotatePoly (po_man.c:114-202)."""
    if sim.level is None:
        return False
    poly_num = args[0]
    poly = get_polyobj(sim.level, poly_num)
    if poly is None:
        return False
    if poly.busy and not override:
        return False

    if poly.specialdata is not None and override:
        old_th = poly.specialdata
        if hasattr(old_th, "removed"):
            old_th.removed = True
        poly.specialdata = None

    ang_arg = args[2]
    if ang_arg != 0:
        if ang_arg == 255:
            dist = -1
        else:
            dist = ang_arg * (ANG90 // 64)
    else:
        dist = ANG_MAX - 1

    speed = (args[1] * direction * (ANG90 // 64)) >> 3

    th = RotatePolyThinker(
        poly=poly,
        speed=speed,
        dist=dist,
    )
    poly.specialdata = th
    sim.thinkers.add(th)

    start_poly_sound(sim, poly)

    curr_poly_num = poly_num
    curr_direction = direction
    visited = {poly_num}

    while True:
        mirror_tag = get_polyobj_mirror(sim.level, curr_poly_num)
        if mirror_tag == 0 or mirror_tag in visited:
            break
        visited.add(mirror_tag)

        mirror_poly = get_polyobj(sim.level, mirror_tag)
        if mirror_poly is None:
            break
        if mirror_poly.busy and not override:
            break

        if mirror_poly.specialdata is not None and override:
            old_mth = mirror_poly.specialdata
            if hasattr(old_mth, "removed"):
                old_mth.removed = True
            mirror_poly.specialdata = None

        curr_direction = -curr_direction
        mirror_speed = (args[1] * curr_direction * (ANG90 // 64)) >> 3

        mirror_th = RotatePolyThinker(
            poly=mirror_poly,
            speed=mirror_speed,
            dist=dist,
        )
        mirror_poly.specialdata = mirror_th
        sim.thinkers.add(mirror_th)

        start_poly_sound(sim, mirror_poly)
        curr_poly_num = mirror_tag

    return True


def ev_move_poly(
    sim: GameSim,
    line: Line | None,
    args: Sequence[int],
    times_eight: bool,
    override: bool,
) -> bool:
    """Port of EV_MovePoly (po_man.c:246-321)."""
    if sim.level is None:
        return False
    poly_num = args[0]
    poly = get_polyobj(sim.level, poly_num)
    if poly is None:
        return False
    if poly.busy and not override:
        return False

    if poly.specialdata is not None and override:
        old_th = poly.specialdata
        if hasattr(old_th, "removed"):
            old_th.removed = True
        poly.specialdata = None

    if times_eight:
        dist = args[3] * 8 * FRACUNIT
    else:
        dist = args[3] * FRACUNIT

    speed = args[1] * (FRACUNIT // 8)
    if dist != -1 and dist < abs(speed):
        speed = dist * (-1 if speed < 0 else 1)
    an = (args[2] * (ANG90 // 64)) & 0xFFFFFFFF
    fine_an = (an >> ANGLETOFINESHIFT) & FINEMASK
    cos_val, sin_val = _cardinal_cos_sin(fine_an)
    x_speed = fixed_mul(speed, cos_val)
    y_speed = fixed_mul(speed, sin_val)

    th = MovePolyThinker(
        poly=poly,
        speed=speed,
        dist=dist,
        angle=fine_an,
        x_speed=x_speed,
        y_speed=y_speed,
    )
    poly.specialdata = th
    sim.thinkers.add(th)

    start_poly_sound(sim, poly)

    curr_poly_num = poly_num
    curr_an = an
    visited = {poly_num}

    while True:
        mirror_tag = get_polyobj_mirror(sim.level, curr_poly_num)
        if mirror_tag == 0 or mirror_tag in visited:
            break
        visited.add(mirror_tag)

        mirror_poly = get_polyobj(sim.level, mirror_tag)
        if mirror_poly is None:
            break
        if mirror_poly.busy and not override:
            break

        if mirror_poly.specialdata is not None and override:
            old_mth = mirror_poly.specialdata
            if hasattr(old_mth, "removed"):
                old_mth.removed = True
            mirror_poly.specialdata = None

        curr_an = (curr_an + ANG180) & 0xFFFFFFFF
        curr_fine_an = (curr_an >> ANGLETOFINESHIFT) & FINEMASK
        cos_val, sin_val = _cardinal_cos_sin(curr_fine_an)
        mirror_x_speed = fixed_mul(speed, cos_val)
        mirror_y_speed = fixed_mul(speed, sin_val)

        mirror_th = MovePolyThinker(
            poly=mirror_poly,
            speed=speed,
            dist=dist,
            angle=curr_fine_an,
            x_speed=mirror_x_speed,
            y_speed=mirror_y_speed,
        )
        mirror_poly.specialdata = mirror_th
        sim.thinkers.add(mirror_th)

        start_poly_sound(sim, mirror_poly)
        curr_poly_num = mirror_tag

    return True


def ev_open_poly_door(
    sim: GameSim,
    line: Line | None,
    args: Sequence[int],
    door_type: PolyDoorType,
) -> bool:
    """Port of EV_OpenPolyDoor (po_man.c:456-549)."""
    if sim.level is None:
        return False
    poly_num = args[0]
    poly = get_polyobj(sim.level, poly_num)
    if poly is None or poly.busy:
        return False

    if door_type == PolyDoorType.PODOOR_SLIDE:
        wait_tics = args[4]
        speed = args[1] * (FRACUNIT // 8)
        total_dist = args[3] * FRACUNIT
        if total_dist != -1 and total_dist < abs(speed):
            speed = total_dist * (-1 if speed < 0 else 1)
        dist = total_dist
        an = (args[2] * (ANG90 // 64)) & 0xFFFFFFFF
        direction = (an >> ANGLETOFINESHIFT) & FINEMASK
        cos_val, sin_val = _cardinal_cos_sin(direction)
        x_speed = fixed_mul(speed, cos_val)
        y_speed = fixed_mul(speed, sin_val)

        th = PolyDoorThinker(
            poly=poly,
            door_type=door_type,
            speed=speed,
            dist=dist,
            total_dist=total_dist,
            wait_tics=wait_tics,
            direction=direction,
            x_speed=x_speed,
            y_speed=y_speed,
        )
    elif door_type == PolyDoorType.PODOOR_SWING:
        wait_tics = args[3]
        direction = 1
        speed = (args[1] * direction * (ANG90 // 64)) >> 3
        total_dist = args[2] * (ANG90 // 64)
        dist = total_dist

        th = PolyDoorThinker(
            poly=poly,
            door_type=door_type,
            speed=speed,
            dist=dist,
            total_dist=total_dist,
            wait_tics=wait_tics,
            direction=direction,
        )
    else:
        return False

    poly.specialdata = th
    sim.thinkers.add(th)

    start_poly_sound(sim, poly)

    curr_poly_num = poly_num
    visited = {poly_num}
    curr_an = an if door_type == PolyDoorType.PODOOR_SLIDE else 0
    curr_direction = 1

    while True:
        mirror_tag = get_polyobj_mirror(sim.level, curr_poly_num)
        if mirror_tag == 0 or mirror_tag in visited:
            break
        visited.add(mirror_tag)

        mirror_poly = get_polyobj(sim.level, mirror_tag)
        if mirror_poly is None or mirror_poly.busy:
            break

        if door_type == PolyDoorType.PODOOR_SLIDE:
            curr_an = (curr_an + ANG180) & 0xFFFFFFFF
            mirror_dir = (curr_an >> ANGLETOFINESHIFT) & FINEMASK
            cos_val, sin_val = _cardinal_cos_sin(mirror_dir)
            mirror_x_speed = fixed_mul(speed, cos_val)
            mirror_y_speed = fixed_mul(speed, sin_val)

            mirror_th = PolyDoorThinker(
                poly=mirror_poly,
                door_type=door_type,
                speed=speed,
                dist=dist,
                total_dist=total_dist,
                wait_tics=wait_tics,
                direction=mirror_dir,
                x_speed=mirror_x_speed,
                y_speed=mirror_y_speed,
            )
        elif door_type == PolyDoorType.PODOOR_SWING:
            curr_direction = -curr_direction
            mirror_speed = (args[1] * curr_direction * (ANG90 // 64)) >> 3

            mirror_th = PolyDoorThinker(
                poly=mirror_poly,
                door_type=door_type,
                speed=mirror_speed,
                dist=dist,
                total_dist=total_dist,
                wait_tics=wait_tics,
                direction=curr_direction,
            )

        mirror_poly.specialdata = mirror_th
        sim.thinkers.add(mirror_th)

        start_poly_sound(sim, mirror_poly)
        curr_poly_num = mirror_tag

    return True


__all__ = [
    "CheckMobjBlocking",
    "DegenMobj",
    "GetPolyobj",
    "GetPolyobjMirror",
    "MAXRADIUS",
    "MovePolyThinker",
    "PO_ANCHOR_TYPE",
    "PO_LINE_EXPLICIT",
    "PO_LINE_START",
    "PO_MAXPOLYSEGS",
    "PO_MovePolyobj",
    "PO_RotatePolyobj",
    "PO_SPAWNCRUSH_TYPE",
    "PO_SPAWN_TYPE",
    "PolyBlock",
    "PolyDoorThinker",
    "PolyDoorType",
    "Polyobj",
    "RotatePolyThinker",
    "SEQTYPE_NUMSEQ",
    "SEQTYPE_STONE",
    "SEQ_DOOR_STONE",
    "ThrustMobj",
    "check_mobj_blocking",
    "check_position",
    "damage_mobj",
    "ev_move_poly",
    "ev_open_poly_door",
    "ev_rotate_poly",
    "get_polyobj",
    "get_polyobj_mirror",
    "init_polyobjs",
    "iter_find_poly_segs",
    "link_polyobj",
    "po_move_polyobj",
    "po_rotate_polyobj",
    "rotate_pt",
    "spawn_polyobj",
    "start_poly_sound",
    "stop_poly_sound",
    "thrust_mobj",
    "translate_to_start_spot",
    "unlink_polyobj",
    "update_seg_bbox",
]
