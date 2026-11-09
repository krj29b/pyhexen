# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Map objects (mobj_t), state transitions, world linking, and map thing spawning.

Port of P_SpawnMobj, P_RemoveMobj, P_SetMobjState, P_SetMobjStateNF, P_SpawnMapThing,
P_CreateTIDList, P_InsertMobjIntoTIDList, P_RemoveMobjFromTIDList, P_FindMobjFromTID
(p_mobj.c) and P_SetThingPosition, P_UnsetThingPosition (p_maputl.c).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import ANG45
from pyhexen.core.fixed import FRACBITS, FRACUNIT, MAXINT, MININT, c_div, to_uint32
from pyhexen.sim.actions import ActionCallable, get_action
from pyhexen.sim.info_data import (
    MOBJINFO,
    STATES,
    MobjFlag,
    MobjFlag2,
    MobjInfo,
    MobjType,
    State,
    StateNum,
)
from pyhexen.sim.thinker import Thinker
from pyhexen.world.blockmap import MAPBLOCKSHIFT

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import Level, MapThing, Sector, Subsector

ONFLOORZ: int = MININT
ONCEILINGZ: int = MAXINT
FLOATRANDZ: int = MAXINT - 1
FROMCEILINGZ128: int = MAXINT - 2

MAXPLAYERS: int = 8
SK_NIGHTMARE: int = 4

# MapThing option flags (h2def.h)
MTF_EASY: int = 1
MTF_NORMAL: int = 2
MTF_HARD: int = 4
MTF_AMBUSH: int = 8
MTF_DORMANT: int = 16
MTF_FIGHTER: int = 32
MTF_CLERIC: int = 64
MTF_MAGE: int = 128
MTF_GSINGLE: int = 256
MTF_GCOOP: int = 512
MTF_GDEATHMATCH: int = 1024

# Polyobj doomednums (p_local.h)
PO_ANCHOR_TYPE: int = 3000
PO_SPAWN_TYPE: int = 3001
PO_SPAWNCRUSH_TYPE: int = 3002

# FloatBobOffsets[64] from p_mobj.c
FLOAT_BOB_OFFSETS: tuple[int, ...] = (
    0,
    51389,
    102283,
    152192,
    200636,
    247147,
    291278,
    332604,
    370727,
    405280,
    435929,
    462380,
    484378,
    501712,
    514213,
    521763,
    524287,
    521763,
    514213,
    501712,
    484378,
    462380,
    435929,
    405280,
    370727,
    332604,
    291278,
    247147,
    200636,
    152192,
    102283,
    51389,
    -1,
    -51390,
    -102284,
    -152193,
    -200637,
    -247148,
    -291279,
    -332605,
    -370728,
    -405281,
    -435930,
    -462381,
    -484380,
    -501713,
    -514215,
    -521764,
    -524288,
    -521764,
    -514214,
    -501713,
    -484379,
    -462381,
    -435930,
    -405280,
    -370728,
    -332605,
    -291279,
    -247148,
    -200637,
    -152193,
    -102284,
    -51389,
)

# Precomputed map from doomednum to MobjType index
DOOMEDNUM_TO_MOBJTYPE: dict[int, int] = {
    info.doomednum: idx for idx, info in enumerate(MOBJINFO) if info.doomednum != -1
}


@dataclass(slots=True)
class Mobj(Thinker):
    """Map object actor thinker.

    Port of mobj_t (h2def.h).
    """

    x: int = 0
    y: int = 0
    z: int = 0
    prev_x: int = 0
    prev_y: int = 0
    prev_z: int = 0
    angle: int = 0
    prev_angle: int = 0
    sprite: int = 0
    frame: int = 0
    floorpic: int = 0
    radius: int = 0
    height: int = 0
    momx: int = 0
    momy: int = 0
    momz: int = 0
    valid_count: int = 0
    type: int = 0
    info: MobjInfo = field(default_factory=lambda: MOBJINFO[0])
    tics: int = 0
    state_num: int = 0
    state: State = field(default_factory=lambda: STATES[0])
    damage: int = 0
    flags: int = 0
    flags2: int = 0
    special1: Any = 0
    special2: Any = 0
    thinker_func: Any = None
    health: int = 0
    movedir: int = 0
    movecount: int = 0
    target: Mobj | None = None
    tracer: Mobj | None = None
    reactiontime: int = 0
    threshold: int = 0
    player: Any = None
    lastlook: int = 0
    floorclip: int = 0
    archive_num: int = 0
    tid: int = 0
    special: int = 0
    args: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    subsector: Subsector | None = None
    sector: Sector | None = None
    floorz: int = 0
    ceilingz: int = 0
    snext: Mobj | None = None
    sprev: Mobj | None = None
    bnext: Mobj | None = None
    bprev: Mobj | None = None

    def __post_init__(self) -> None:
        Thinker.__init__(self)

    def capture_prev_state(self) -> None:
        """Copy current transform into prev_* fields for render interpolation."""
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        self.prev_angle = self.angle

    def tick(self, sim: GameSim) -> None:
        """Port of P_MobjThinker (p_mobj.c)."""
        if self.thinker_func is not None:
            self.thinker_func(sim, self)
            return
        from pyhexen.sim.movement import p_mobj_thinker

        p_mobj_thinker(sim, self)

    def state_tuple(self) -> tuple[Any, ...]:
        """Return deterministic state tuple for `GameSim.state_hash()`."""
        return (
            int(self.type),
            self.x,
            self.y,
            self.z,
            self.angle,
            self.health,
            int(self.state_num),
        )


def set_thing_position(level: Level, mobj: Mobj) -> None:
    """Link `mobj` into its subsector/sector thing list and blockmap cell.

    Port of P_SetThingPosition (p_maputl.c).
    """
    ss = level.point_in_subsector(mobj.x, mobj.y)
    mobj.subsector = ss
    mobj.sector = ss.sector

    if not (mobj.flags & MobjFlag.MF_NOSECTOR):
        sec = ss.sector
        mobj.sprev = None
        mobj.snext = sec.thing_list
        if sec.thing_list is not None:
            sec.thing_list.sprev = mobj
        sec.thing_list = mobj

    if not (mobj.flags & MobjFlag.MF_NOBLOCKMAP):
        bm = level.blockmap
        blockx = (mobj.x - bm.org_x) >> MAPBLOCKSHIFT
        blocky = (mobj.y - bm.org_y) >> MAPBLOCKSHIFT
        if 0 <= blockx < bm.width and 0 <= blocky < bm.height:
            idx = blocky * bm.width + blockx
            link = bm.block_links[idx]
            mobj.bprev = None
            mobj.bnext = link
            if link is not None:
                link.bprev = mobj
            bm.block_links[idx] = mobj
        else:
            mobj.bnext = None
            mobj.bprev = None


def unset_thing_position(level: Level, mobj: Mobj) -> None:
    """Unlink `mobj` from its sector thing list and blockmap cell.

    Port of P_UnsetThingPosition (p_maputl.c).
    """
    if not (mobj.flags & MobjFlag.MF_NOSECTOR):
        if mobj.snext is not None:
            mobj.snext.sprev = mobj.sprev
        if mobj.sprev is not None:
            mobj.sprev.snext = mobj.snext
        elif mobj.subsector is not None:
            mobj.subsector.sector.thing_list = mobj.snext
        mobj.snext = None
        mobj.sprev = None

    if not (mobj.flags & MobjFlag.MF_NOBLOCKMAP):
        if mobj.bnext is not None:
            mobj.bnext.bprev = mobj.bprev
        if mobj.bprev is not None:
            mobj.bprev.bnext = mobj.bnext
        else:
            bm = level.blockmap
            blockx = (mobj.x - bm.org_x) >> MAPBLOCKSHIFT
            blocky = (mobj.y - bm.org_y) >> MAPBLOCKSHIFT
            if 0 <= blockx < bm.width and 0 <= blocky < bm.height:
                bm.block_links[blocky * bm.width + blockx] = mobj.bnext
        mobj.bnext = None
        mobj.bprev = None


def insert_mobj_into_tid_list(sim: GameSim, mobj: Mobj, tid: int) -> None:
    """Insert `mobj` into `sim.tid_map` under `tid`.

    Port of P_InsertMobjIntoTIDList (p_mobj.c).
    """
    if tid == 0:
        mobj.tid = 0
        return
    mobj.tid = tid
    bucket = sim.tid_map.setdefault(tid, [])
    if mobj not in bucket:
        bucket.append(mobj)


def remove_mobj_from_tid_list(sim: GameSim, mobj: Mobj) -> None:
    """Remove `mobj` from `sim.tid_map`.

    Port of P_RemoveMobjFromTIDList (p_mobj.c).
    """
    tid = mobj.tid
    if tid != 0 and tid in sim.tid_map:
        bucket = sim.tid_map[tid]
        try:
            bucket.remove(mobj)
        except ValueError:
            pass
    mobj.tid = 0


def find_mobj_from_tid(sim: GameSim, tid: int) -> list[Mobj]:
    """Return all active mobjs matching `tid`.

    Port of P_FindMobjFromTID (p_mobj.c).
    """
    return sim.find_mobjs_by_tid(tid)


def _invoke_action(act: ActionCallable, sim: GameSim, mobj: Mobj) -> None:
    """Invoke a registered state action callable, adapting to its parameter signature."""
    if getattr(act, "_is_stub", False):
        act(sim, mobj)
        return
    try:
        sig = inspect.signature(act)
    except (ValueError, TypeError):
        act(sim, mobj)
        return

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        act(sim, mobj)
        return

    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(pos_params) == 0:
        act()
    elif len(pos_params) == 1:
        if pos_params[0].name in ("sim", "game_sim"):
            act(sim)
        else:
            act(mobj)
    elif len(pos_params) == 2:
        if pos_params[0].name in ("mobj", "mo", "actor", "thing"):
            act(mobj, sim)
        else:
            act(sim, mobj)
    else:
        act(sim, mobj, None)


def set_mobj_state(sim: GameSim, mobj: Mobj, state_num: int | StateNum) -> bool:
    """Transition `mobj` to `state_num`, executing 0-tic state chains and actions.

    Port of P_SetMobjState (p_mobj.c). Returns False if the mobj entered S_NULL and was removed.
    """
    st_idx = int(state_num)
    while True:
        if st_idx == StateNum.S_NULL:
            mobj.state_num = int(StateNum.S_NULL)
            remove_mobj(sim, mobj)
            return False

        st = STATES[st_idx]
        mobj.state_num = st_idx
        mobj.state = st
        mobj.tics = st.tics
        mobj.sprite = st.sprite
        mobj.frame = st.frame

        act = get_action(st.action_name)
        if act is not None:
            _invoke_action(act, sim, mobj)
            if mobj.removed:
                return False

        st_idx = st.next_state
        if mobj.tics != 0:
            break

    return True


def set_mobj_state_nf(sim: GameSim, mobj: Mobj, state_num: int | StateNum) -> bool:
    """Transition `mobj` to `state_num` without calling the state action function.

    Port of P_SetMobjStateNF (p_mobj.c).
    """
    st_idx = int(state_num)
    if st_idx == StateNum.S_NULL:
        mobj.state_num = int(StateNum.S_NULL)
        remove_mobj(sim, mobj)
        return False

    st = STATES[st_idx]
    mobj.state_num = st_idx
    mobj.state = st
    mobj.tics = st.tics
    mobj.sprite = st.sprite
    mobj.frame = st.frame
    return True


def remove_mobj(sim: GameSim, mobj: Mobj) -> None:
    """Unlink `mobj` from world (sector, blockmap, TID list) and mark its thinker removed.

    Port of P_RemoveMobj (p_mobj.c).
    """
    if mobj.tid != 0:
        remove_mobj_from_tid_list(sim, mobj)

    if sim.level is not None:
        unset_thing_position(sim.level, mobj)

    sim.thinkers.remove(mobj)


def spawn_mobj(
    sim: GameSim,
    x: int,
    y: int,
    z: int,
    mobj_type: int | MobjType,
) -> Mobj:
    """Spawn a map object at `(x, y, z)` of `mobj_type`, link into level and thinkers.

    Port of P_SpawnMobj (p_mobj.c).
    """
    mtype = int(mobj_type)
    info = MOBJINFO[mtype]
    st_idx = info.spawnstate
    st = STATES[st_idx]

    reactiontime = 0 if sim.skill == SK_NIGHTMARE else info.reactiontime
    lastlook = sim.rng.p_random() % MAXPLAYERS

    mobj = Mobj(
        x=x,
        y=y,
        z=0,
        prev_x=x,
        prev_y=y,
        prev_z=0,
        angle=0,
        prev_angle=0,
        sprite=st.sprite,
        frame=st.frame,
        radius=info.radius,
        height=info.height,
        type=mtype,
        info=info,
        tics=st.tics,
        state_num=st_idx,
        state=st,
        damage=info.damage,
        flags=info.flags,
        flags2=info.flags2,
        health=info.spawnhealth,
        reactiontime=reactiontime,
        lastlook=lastlook,
    )

    if sim.level is not None:
        set_thing_position(sim.level, mobj)
        assert mobj.subsector is not None
        sec = mobj.subsector.sector
        mobj.floorz = sec.floor_height
        mobj.ceilingz = sec.ceiling_height

    if z == ONFLOORZ:
        mobj.z = mobj.floorz
    elif z == ONCEILINGZ:
        mobj.z = mobj.ceilingz - mobj.info.height
    elif z == FLOATRANDZ:
        space = (mobj.ceilingz - mobj.info.height) - mobj.floorz
        if space > 48 * FRACUNIT:
            space -= 40 * FRACUNIT
            mobj.z = ((space * sim.rng.p_random()) >> 8) + mobj.floorz + 40 * FRACUNIT
        else:
            mobj.z = mobj.floorz
    elif mobj.flags2 & MobjFlag2.MF2_FLOATBOB:
        mobj.z = mobj.floorz + z
    else:
        mobj.z = z

    mobj.capture_prev_state()
    sim.thinkers.add(mobj)
    return mobj


def spawn_map_thing(sim: GameSim, mthing: MapThing) -> Mobj | None:
    """Spawn a map thing according to game mode, class, skill, and flags.

    Port of P_SpawnMapThing (p_mobj.c).
    """
    # Polyobj anchor and spawn spots are handled by PolyObj initialization
    if mthing.type in (PO_ANCHOR_TYPE, PO_SPAWN_TYPE, PO_SPAWNCRUSH_TYPE):
        return None

    # Player 1..4 (1..4) and Player 5..8 (9100..9103) start spots
    if mthing.type in (1, 2, 3, 4) or 9100 <= mthing.type <= 9103:
        if mthing.type in (1, 2, 3, 4):
            player_num = mthing.type - 1
        else:
            player_num = 4 + (mthing.type - 9100)
        spot_group = mthing.args[0]
        sim.player_starts[(player_num, spot_group)] = mthing
        return None

    # Deathmatch start spots
    if mthing.type == 11:
        sim.deathmatch_starts.append(mthing)
        return None

    # Ambient sound sequence overrides (1400..1409)
    if 1400 <= mthing.type < 1410:
        if sim.level is not None:
            ss = sim.level.point_in_subsector(mthing.x << FRACBITS, mthing.y << FRACBITS)
            ss.sector.seq_type = mthing.type - 1400
        return None

    options = getattr(mthing, "options", mthing.flags)

    # Filter by game mode (single-player, cooperative, deathmatch)
    if not sim.netgame:
        if not (options & MTF_GSINGLE):
            return None
    elif not sim.deathmatch:
        if not (options & MTF_GCOOP):
            return None
    else:
        if not (options & MTF_GDEATHMATCH):
            return None

    # Filter by player class in single-player (0=Fighter, 1=Cleric, 2=Mage)
    if not sim.netgame:
        class_mask = MTF_FIGHTER << sim.player_class
        if not (options & class_mask):
            return None

    # Filter by skill level (0=baby, 1=easy, 2=medium, 3=hard, 4=nightmare)
    if sim.skill in (0, 1):
        skill_bit = MTF_EASY
    elif sim.skill in (3, 4):
        skill_bit = MTF_HARD
    else:
        skill_bit = MTF_NORMAL
    if not (options & skill_bit):
        return None

    # Resolve doomednum to MobjType
    mtype = DOOMEDNUM_TO_MOBJTYPE.get(mthing.type)
    if mtype is None:
        return None

    info = MOBJINFO[mtype]

    # Check -nomonsters
    if sim.nomonsters and (info.flags & MobjFlag.MF_COUNTKILL):
        return None

    x = mthing.x << FRACBITS
    y = mthing.y << FRACBITS
    height_fixed = mthing.z << FRACBITS

    # Determine initial Z placement
    if info.flags & MobjFlag.MF_SPAWNCEILING:
        z = ONCEILINGZ
    elif info.flags2 & MobjFlag2.MF2_SPAWNFLOAT:
        z = FLOATRANDZ
    elif info.flags2 & MobjFlag2.MF2_FLOATBOB:
        z = height_fixed
    else:
        z = ONFLOORZ

    mobj = spawn_mobj(sim, x, y, z, mtype)
    if z == ONFLOORZ:
        mobj.z += height_fixed
    elif z == ONCEILINGZ:
        mobj.z -= height_fixed

    if mthing.tid != 0:
        insert_mobj_into_tid_list(sim, mobj, mthing.tid)

    mobj.special = mthing.special
    mobj.args = mthing.args

    if mobj.flags2 & MobjFlag2.MF2_FLOATBOB:
        mobj.health = sim.rng.p_random()
        mobj.special1 = height_fixed

    if mobj.tics > 0:
        mobj.tics = 1 + (sim.rng.p_random() % mobj.tics)

    if mobj.flags & MobjFlag.MF_COUNTKILL:
        mobj.angle = to_uint32(ANG45 * c_div(mthing.angle, 45))
    else:
        mobj.angle = to_uint32(c_div(mthing.angle << 8, 360) << 24)

    if options & MTF_AMBUSH:
        mobj.flags |= MobjFlag.MF_AMBUSH

    if options & MTF_DORMANT:
        mobj.flags2 |= MobjFlag2.MF2_DORMANT
        if mobj.type == MobjType.MT_ICEGUY:
            set_mobj_state(sim, mobj, StateNum.S_ICEGUY_DORMANT)
        mobj.tics = -1

    mobj.capture_prev_state()
    return mobj


def spawn_map_things(sim: GameSim) -> list[Mobj]:
    """Spawn all map things in `sim.level.things` in lump order."""
    if sim.level is None:
        return []
    spawned: list[Mobj] = []
    for mthing in sim.level.things:
        mo = spawn_map_thing(sim, mthing)
        if mo is not None:
            spawned.append(mo)
    return spawned
