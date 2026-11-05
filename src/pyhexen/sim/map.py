# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Map collision detection, position checks, movement stepping, and wall sliding.

Port of P_CheckPosition, PIT_CheckLine, PIT_CheckThing, P_TryMove, P_TeleportMove,
P_SlideMove, P_HitSlideLine, PTR_SlideTraverse, and P_CheckOnmobj (p_map.c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyhexen.core.angles import (
    ANG180,
    ANGLETOFINESHIFT,
    approx_distance,
    fine_cosine,
    fine_sine,
    point_to_angle2,
)
from pyhexen.core.fixed import FRACUNIT, abs32, fixed_mul
from pyhexen.sim.info_data import MobjFlag, MobjFlag2, MobjType
from pyhexen.sim.maputl import (
    PT_ADDLINES,
    Intercept,
    box_on_line_side,
    line_opening,
    path_traverse,
    point_on_line_side,
)
from pyhexen.sim.mobj import Mobj, set_thing_position, unset_thing_position
from pyhexen.world.blockmap import MAPBLOCKSHIFT
from pyhexen.world.level import (
    BOXBOTTOM,
    BOXLEFT,
    BOXRIGHT,
    BOXTOP,
    GET_SPAC,
    ML_BLOCKING,
    ML_BLOCKMONSTERS,
    ML_TWOSIDED,
    SPAC_CROSS,
    SPAC_IMPACT,
    SPAC_MCROSS,
    SPAC_PCROSS,
    SPAC_PUSH,
    Line,
    SlopeType,
)

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

MAXRADIUS: int = 32 * FRACUNIT
STEPHEIGHT: int = 24 * FRACUNIT
MAXSPECIALCROSS: int = 8


@dataclass(slots=True)
class MapCheckState:
    """Mutable collision state matching module globals in p_map.c."""

    tmthing: Mobj | None = None
    tmflags: int = 0
    tmx: int = 0
    tmy: int = 0
    tmbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    tmfloorz: int = 0
    tmceilingz: int = 0
    tmdropoffz: int = 0
    tmfloorpic: int = 0
    ceilingline: Line | None = None
    blocking_mobj: Mobj | None = None
    floatok: bool = False
    spechit: list[Line] = field(default_factory=list)
    last_spechit: list[Line] = field(default_factory=list)
    crossed_lines: list[Line] = field(default_factory=list)
    # Slide move state
    slide_mo: Mobj | None = None
    best_slide_frac: int = 0
    second_slide_frac: int = 0
    best_slide_line: Line | None = None
    second_slide_line: Line | None = None
    tmxmove: int = 0
    tmymove: int = 0


def check_for_push_special(sim: GameSim, line: Line, side: int, mobj: Mobj) -> None:
    """Activate push or impact line specials when `mobj` is blocked by `line`.

    Port of P_CheckForPushSpecial (p_map.c).
    """
    _ = side
    if not line.special:
        return
    if mobj.flags2 & MobjFlag2.MF2_PUSHWALL:
        sim.on_line_activated(line, mobj, SPAC_PUSH)
    elif mobj.flags2 & MobjFlag2.MF2_IMPACT:
        sim.on_line_activated(line, mobj, SPAC_IMPACT)


def pit_check_line(sim: GameSim, ld: Line) -> bool:
    """Adjust `tmfloorz` and `tmceilingz` as lines are contacted; return False if blocked.

    Port of PIT_CheckLine (p_map.c).
    """
    st = sim.map_state
    tmbox = st.tmbox
    tmthing = st.tmthing
    assert tmthing is not None

    if (
        tmbox[BOXRIGHT] <= ld.bbox[BOXLEFT]
        or tmbox[BOXLEFT] >= ld.bbox[BOXRIGHT]
        or tmbox[BOXTOP] <= ld.bbox[BOXBOTTOM]
        or tmbox[BOXBOTTOM] >= ld.bbox[BOXTOP]
    ):
        return True

    if box_on_line_side(tmbox, ld) != -1:
        return True

    # A line has been hit
    if ld.back_sector is None:
        check_for_push_special(sim, ld, 0, tmthing)
        return False

    if not (tmthing.flags & MobjFlag.MF_MISSILE):
        if ld.flags & ML_BLOCKING:
            check_for_push_special(sim, ld, 0, tmthing)
            return False
        if (
            tmthing.player is None
            and tmthing.type
            not in (
                MobjType.MT_PLAYER_FIGHTER,
                MobjType.MT_PLAYER_CLERIC,
                MobjType.MT_PLAYER_MAGE,
                MobjType.MT_ETTIN_MACE,
            )
            and (ld.flags & ML_BLOCKMONSTERS)
        ):
            return False

    opening = line_opening(ld)
    if opening.opentop < st.tmceilingz:
        st.tmceilingz = opening.opentop
        st.ceilingline = ld

    if opening.openbottom > st.tmfloorz:
        st.tmfloorz = opening.openbottom

    if opening.lowfloor < st.tmdropoffz:
        st.tmdropoffz = opening.lowfloor

    if ld.special:
        st.spechit.append(ld)

    return True


def pit_check_thing(sim: GameSim, thing: Mobj) -> bool:
    """Check collision between `st.tmthing` and `thing`; return False if blocked.

    Port of PIT_CheckThing (p_map.c).
    """
    st = sim.map_state
    tmthing = st.tmthing
    assert tmthing is not None

    if not (thing.flags & (MobjFlag.MF_SOLID | MobjFlag.MF_SPECIAL | MobjFlag.MF_SHOOTABLE)):
        return True

    blockdist = thing.radius + tmthing.radius
    if abs32(thing.x - st.tmx) >= blockdist or abs32(thing.y - st.tmy) >= blockdist:
        return True

    if thing is tmthing:
        return True

    st.blocking_mobj = thing

    if tmthing.flags2 & MobjFlag2.MF2_PASSMOBJ:
        if not (thing.flags & (MobjFlag.MF_MISSILE | MobjFlag.MF_SKULLFLY)):
            if tmthing.z + tmthing.height <= thing.z:
                return True  # underneath
            if tmthing.z >= thing.z + thing.height:
                return True  # overhead

    # Check for skulls slamming into things
    if tmthing.flags & MobjFlag.MF_SKULLFLY:
        tmthing.flags &= ~MobjFlag.MF_SKULLFLY
        tmthing.momx = 0
        tmthing.momy = 0
        tmthing.momz = 0
        return False

    # Missiles can hit other things
    if tmthing.flags & MobjFlag.MF_MISSILE:
        if tmthing.z > thing.z + thing.height:
            return True  # overhead
        if tmthing.z + tmthing.height < thing.z:
            return True  # underneath
        if tmthing.target is not None and tmthing.target.type == thing.type:
            if thing is tmthing.target:
                return True
            if thing.player is None:
                return False
        if not (thing.flags & MobjFlag.MF_SHOOTABLE):
            return not bool(thing.flags & MobjFlag.MF_SOLID)
        return False

    if thing.flags & MobjFlag.MF_SPECIAL:
        return not bool(thing.flags & MobjFlag.MF_SOLID)

    return not bool(thing.flags & MobjFlag.MF_SOLID)


def check_position(sim: GameSim, thing: Mobj, x: int, y: int) -> bool:
    """Test whether `thing` can occupy `(x, y)` without hitting solid things or blocking lines.

    Port of P_CheckPosition (p_map.c). Populates `sim.map_state` (`tmfloorz`, `tmceilingz`,
    `tmdropoffz`, `spechit`, `blocking_mobj`).
    """
    level = sim.level
    if level is None:
        return True

    st = sim.map_state
    st.tmthing = thing
    st.tmflags = thing.flags
    st.tmx = x
    st.tmy = y

    st.tmbox[BOXTOP] = y + thing.radius
    st.tmbox[BOXBOTTOM] = y - thing.radius
    st.tmbox[BOXRIGHT] = x + thing.radius
    st.tmbox[BOXLEFT] = x - thing.radius

    newsubsec = level.point_in_subsector(x, y)
    st.ceilingline = None

    st.tmfloorz = newsubsec.sector.floor_height
    st.tmdropoffz = st.tmfloorz
    st.tmceilingz = newsubsec.sector.ceiling_height

    bm = level.blockmap
    valid_count = bm.next_valid_count()
    st.spechit.clear()
    st.blocking_mobj = None

    if st.tmflags & MobjFlag.MF_NOCLIP:
        return True

    # Check things first, possibly picking up items
    xl = (st.tmbox[BOXLEFT] - bm.org_x - MAXRADIUS) >> MAPBLOCKSHIFT
    xh = (st.tmbox[BOXRIGHT] - bm.org_x + MAXRADIUS) >> MAPBLOCKSHIFT
    yl = (st.tmbox[BOXBOTTOM] - bm.org_y - MAXRADIUS) >> MAPBLOCKSHIFT
    yh = (st.tmbox[BOXTOP] - bm.org_y + MAXRADIUS) >> MAPBLOCKSHIFT

    for bx in range(xl, xh + 1):
        for by in range(yl, yh + 1):
            if not bm.block_things_iterator(bx, by, lambda it: pit_check_thing(sim, it)):
                return False

    # Check lines
    xl = (st.tmbox[BOXLEFT] - bm.org_x) >> MAPBLOCKSHIFT
    xh = (st.tmbox[BOXRIGHT] - bm.org_x) >> MAPBLOCKSHIFT
    yl = (st.tmbox[BOXBOTTOM] - bm.org_y) >> MAPBLOCKSHIFT
    yh = (st.tmbox[BOXTOP] - bm.org_y) >> MAPBLOCKSHIFT

    for bx in range(xl, xh + 1):
        for by in range(yl, yh + 1):
            if not bm.block_lines_iterator(
                bx, by, lambda ld: pit_check_line(sim, ld), valid_count=valid_count
            ):
                return False

    st.last_spechit = list(st.spechit)
    return True


def _check_spechit_push(sim: GameSim, thing: Mobj, oldx: int, oldy: int) -> None:
    """Trigger push/impact specials on lines in `spechit` when movement fails."""
    st = sim.map_state
    while st.spechit:
        ld = st.spechit.pop()
        side = point_on_line_side(oldx, oldy, ld)
        check_for_push_special(sim, ld, side, thing)


def try_move(sim: GameSim, thing: Mobj, x: int, y: int) -> bool:
    """Attempt to move `thing` to `(x, y)` if unblocked and height clearances allow.

    Port of P_TryMove (p_map.c).
    """
    level = sim.level
    st = sim.map_state
    st.floatok = False
    st.crossed_lines.clear()
    oldx = thing.x
    oldy = thing.y

    if not check_position(sim, thing, x, y):
        if st.blocking_mobj is None:
            _check_spechit_push(sim, thing, oldx, oldy)
        return False

    if not (thing.flags & MobjFlag.MF_NOCLIP):
        if st.tmceilingz - st.tmfloorz < thing.height:
            # Doesn't fit
            _check_spechit_push(sim, thing, oldx, oldy)
            return False

        st.floatok = True

        if (
            not (thing.flags & MobjFlag.MF_TELEPORT)
            and (st.tmceilingz - thing.z < thing.height)
            and not (thing.flags2 & MobjFlag2.MF2_FLY)
        ):
            # Mobj must lower itself to fit
            _check_spechit_push(sim, thing, oldx, oldy)
            return False

        if not (thing.flags & MobjFlag.MF_TELEPORT):
            if st.tmfloorz - thing.z > STEPHEIGHT:
                # Too big a step up
                _check_spechit_push(sim, thing, oldx, oldy)
                return False

        if (thing.flags & MobjFlag.MF_MISSILE) and st.tmfloorz > thing.z:
            _check_spechit_push(sim, thing, oldx, oldy)
            return False

        if (
            not (thing.flags & (MobjFlag.MF_DROPOFF | MobjFlag.MF_FLOAT))
            and (st.tmfloorz - st.tmdropoffz > STEPHEIGHT)
            and not (thing.flags2 & MobjFlag2.MF2_BLASTED)
        ):
            # Don't stand over a dropoff
            return False

    # Move is valid; relink thing into new position
    if level is not None:
        unset_thing_position(level, thing)
    thing.floorz = st.tmfloorz
    thing.ceilingz = st.tmceilingz
    thing.x = x
    thing.y = y
    if level is not None:
        set_thing_position(level, thing)

    # Check if any special lines were crossed
    if not (thing.flags & (MobjFlag.MF_TELEPORT | MobjFlag.MF_NOCLIP)):
        while st.spechit:
            ld = st.spechit.pop()
            side = point_on_line_side(thing.x, thing.y, ld)
            oldside = point_on_line_side(oldx, oldy, ld)
            if side != oldside and ld.special:
                st.crossed_lines.append(ld)
                if thing.player is not None or thing.type in (
                    MobjType.MT_PLAYER_FIGHTER,
                    MobjType.MT_PLAYER_CLERIC,
                    MobjType.MT_PLAYER_MAGE,
                ):
                    sim.on_line_activated(ld, thing, SPAC_CROSS)
                elif thing.flags2 & MobjFlag2.MF2_MCROSS:
                    sim.on_line_activated(ld, thing, SPAC_MCROSS)
                elif thing.flags2 & MobjFlag2.MF2_PCROSS:
                    sim.on_line_activated(ld, thing, SPAC_PCROSS)
                else:
                    sim.on_line_activated(ld, thing, GET_SPAC(ld.flags))

    return True


def teleport_move(sim: GameSim, thing: Mobj, x: int, y: int, boss: bool = False) -> bool:
    """Teleport `thing` to `(x, y)` if not blocked.

    Port of P_TeleportMove (p_map.c).
    """
    _ = boss
    if not check_position(sim, thing, x, y):
        return False

    st = sim.map_state
    if sim.level is not None:
        unset_thing_position(sim.level, thing)

    thing.floorz = st.tmfloorz
    thing.ceilingz = st.tmceilingz
    thing.x = x
    thing.y = y

    if sim.level is not None:
        set_thing_position(sim.level, thing)

    return True


def check_on_mobj(sim: GameSim, thing: Mobj) -> Mobj | None:
    """Return the solid `Mobj` directly supporting `thing` from below, or `None`.

    Port of P_CheckOnmobj (p_map.c).
    """
    level = sim.level
    if level is None:
        return None

    st = sim.map_state
    x = thing.x
    y = thing.y
    st.tmthing = thing
    st.tmflags = thing.flags
    oldz = thing.z
    thing.z += thing.momz
    if thing.z > thing.floorz:
        thing.z -= FRACUNIT

    st.tmx = x
    st.tmy = y
    st.tmbox[BOXTOP] = y + thing.radius
    st.tmbox[BOXBOTTOM] = y - thing.radius
    st.tmbox[BOXRIGHT] = x + thing.radius
    st.tmbox[BOXLEFT] = x - thing.radius

    bm = level.blockmap
    xl = (st.tmbox[BOXLEFT] - bm.org_x - MAXRADIUS) >> MAPBLOCKSHIFT
    xh = (st.tmbox[BOXRIGHT] - bm.org_x + MAXRADIUS) >> MAPBLOCKSHIFT
    yl = (st.tmbox[BOXBOTTOM] - bm.org_y - MAXRADIUS) >> MAPBLOCKSHIFT
    yh = (st.tmbox[BOXTOP] - bm.org_y + MAXRADIUS) >> MAPBLOCKSHIFT

    found: Mobj | None = None

    def _pit_check_onmobj(other: Mobj) -> bool:
        nonlocal found
        if not (other.flags & (MobjFlag.MF_SOLID | MobjFlag.MF_SPECIAL | MobjFlag.MF_SHOOTABLE)):
            return True
        blockdist = other.radius + thing.radius
        if abs32(other.x - x) >= blockdist or abs32(other.y - y) >= blockdist:
            return True
        if other is thing:
            return True
        if thing.z > other.z + other.height:
            return True
        if thing.z + thing.height < other.z:
            return True
        if other.flags & MobjFlag.MF_SOLID:
            found = other
            return False
        return True

    for bx in range(xl, xh + 1):
        for by in range(yl, yh + 1):
            if not bm.block_things_iterator(bx, by, _pit_check_onmobj):
                thing.z = oldz
                return found

    thing.z = oldz
    return None


def hit_slide_line(sim: GameSim, ld: Line) -> None:
    """Project `sim.map_state.tmxmove, tmymove` parallel to `ld`.

    Port of P_HitSlideLine (p_map.c).
    """
    st = sim.map_state
    slide_mo = st.slide_mo
    assert slide_mo is not None

    if ld.slopetype == SlopeType.ST_HORIZONTAL:
        st.tmymove = 0
        return

    if ld.slopetype == SlopeType.ST_VERTICAL:
        st.tmxmove = 0
        return

    side = point_on_line_side(slide_mo.x, slide_mo.y, ld)
    lineangle = point_to_angle2(0, 0, ld.dx, ld.dy)
    if side == 1:
        lineangle = (lineangle + ANG180) & 0xFFFFFFFF

    moveangle = point_to_angle2(0, 0, st.tmxmove, st.tmymove)
    deltaangle = (moveangle - lineangle) & 0xFFFFFFFF
    if deltaangle > ANG180:
        deltaangle = (deltaangle + ANG180) & 0xFFFFFFFF

    lineangle >>= ANGLETOFINESHIFT
    deltaangle >>= ANGLETOFINESHIFT

    movelen = approx_distance(st.tmxmove, st.tmymove)
    newlen = fixed_mul(movelen, fine_cosine(deltaangle))

    st.tmxmove = fixed_mul(newlen, fine_cosine(lineangle))
    st.tmymove = fixed_mul(newlen, fine_sine(lineangle))


def _ptr_slide_traverse(sim: GameSim, in_: Intercept) -> bool:
    """Intercept callback for `slide_move`.

    Port of PTR_SlideTraverse (p_map.c).
    """
    st = sim.map_state
    slide_mo = st.slide_mo
    assert slide_mo is not None
    li = in_.line
    assert li is not None

    if not (li.flags & ML_TWOSIDED) or li.back_sector is None:
        if point_on_line_side(slide_mo.x, slide_mo.y, li) != 0:
            return True  # Don't hit the back side
    else:
        opening = line_opening(li)
        if (
            opening.openrange >= slide_mo.height
            and opening.opentop - slide_mo.z >= slide_mo.height
            and opening.openbottom - slide_mo.z <= STEPHEIGHT
        ):
            return True  # Doesn't block movement

    # The line does block movement; see if it is closer than best so far
    if in_.frac < st.best_slide_frac:
        st.second_slide_frac = st.best_slide_frac
        st.second_slide_line = st.best_slide_line
        st.best_slide_frac = in_.frac
        st.best_slide_line = li

    return False  # Stop searching


def slide_move(sim: GameSim, mo: Mobj) -> None:
    """Slide `mo` along blocking walls by tracing its 3 leading corners.

    Port of P_SlideMove (p_map.c).
    """
    level = sim.level
    if level is None:
        return

    st = sim.map_state
    st.slide_mo = mo

    for _hitcount in range(3):
        if mo.momx > 0:
            leadx = mo.x + mo.radius
            trailx = mo.x - mo.radius
        else:
            leadx = mo.x - mo.radius
            trailx = mo.x + mo.radius

        if mo.momy > 0:
            leady = mo.y + mo.radius
            traily = mo.y - mo.radius
        else:
            leady = mo.y - mo.radius
            traily = mo.y + mo.radius

        st.best_slide_frac = FRACUNIT + 1
        st.best_slide_line = None

        path_traverse(
            level,
            leadx,
            leady,
            leadx + mo.momx,
            leady + mo.momy,
            PT_ADDLINES,
            lambda in_: _ptr_slide_traverse(sim, in_),
        )
        path_traverse(
            level,
            trailx,
            leady,
            trailx + mo.momx,
            leady + mo.momy,
            PT_ADDLINES,
            lambda in_: _ptr_slide_traverse(sim, in_),
        )
        path_traverse(
            level,
            leadx,
            traily,
            leadx + mo.momx,
            traily + mo.momy,
            PT_ADDLINES,
            lambda in_: _ptr_slide_traverse(sim, in_),
        )

        # No blocking lines intercepted along the 3 corner rays: stairstep
        if st.best_slide_frac == FRACUNIT + 1 or st.best_slide_line is None:
            if not try_move(sim, mo, mo.x, mo.y + mo.momy):
                try_move(sim, mo, mo.x + mo.momx, mo.y)
            return

        # Fudge bit (0x800 = 1/32 of FRACUNIT) to stay just short of wall
        st.best_slide_frac -= 0x800
        if st.best_slide_frac > 0:
            newx = fixed_mul(mo.momx, st.best_slide_frac)
            newy = fixed_mul(mo.momy, st.best_slide_frac)
            if not try_move(sim, mo, mo.x + newx, mo.y + newy):
                if not try_move(sim, mo, mo.x, mo.y + mo.momy):
                    try_move(sim, mo, mo.x + mo.momx, mo.y)
                return

        # Calculate remainder of move and project along blocking line
        st.best_slide_frac = FRACUNIT - (st.best_slide_frac + 0x800)
        if st.best_slide_frac > FRACUNIT:
            st.best_slide_frac = FRACUNIT
        if st.best_slide_frac <= 0:
            return

        st.tmxmove = fixed_mul(mo.momx, st.best_slide_frac)
        st.tmymove = fixed_mul(mo.momy, st.best_slide_frac)

        hit_slide_line(sim, st.best_slide_line)

        mo.momx = st.tmxmove
        mo.momy = st.tmymove

        if try_move(sim, mo, mo.x + st.tmxmove, mo.y + st.tmymove):
            return

    # Exhausted 3 slide iterations; stairstep fallback
    if not try_move(sim, mo, mo.x, mo.y + mo.momy):
        try_move(sim, mo, mo.x + mo.momx, mo.y)
