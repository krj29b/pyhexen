# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Simulation serialization framework (T13.2).

Port of SV_SaveMap, SV_LoadMap, ArchiveWorld, ArchivePolyobjs, ArchiveMobjs,
ArchiveThinkers, ArchiveScripts, ArchivePlayers in sv_save.c.
"""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING, Any

from pyhexen.sim.acs.loader import ACScriptState
from pyhexen.sim.acs.vm import RunningScript
from pyhexen.sim.mobj import Mobj, set_thing_position
from pyhexen.sim.player import Player
from pyhexen.sim.polyobj import (
    MovePolyThinker,
    PolyDoorThinker,
    PolyDoorType,
    Polyobj,
    RotatePolyThinker,
)
from pyhexen.sim.specials.ceilings import CeilingMove
from pyhexen.sim.specials.doors import DoorType, VerticalDoor
from pyhexen.sim.specials.floors import FloorMove, FloorWaggle
from pyhexen.sim.specials.lights import LightFade, LightGlow
from pyhexen.sim.specials.pillars import Pillar
from pyhexen.sim.specials.plats import Plat
from pyhexen.sim.specials.scroll import ScrollThinker
from pyhexen.sim.specials.sector_specials import PhasedLightThinker
from pyhexen.sim.specials.switches import ButtonThinker, SwitchWhere

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


def serialize_sim(sim: GameSim, save_players: bool = True) -> dict[str, Any]:
    """Serialize the complete simulation state into a JSON-compatible dictionary.

    Port of SV_SaveMap (sv_save.c:2029).
    """
    # 1. Header & Meta
    state: dict[str, Any] = {
        "version": 1,
        "level_time": sim.level_time,
        "map_num": getattr(sim, "map_num", 1),
        "cluster": getattr(sim, "cluster", 1),
        "skill": sim.skill,
        "player_class": sim.player_class,
        "netgame": sim.netgame,
        "deathmatch": sim.deathmatch,
        "nomonsters": sim.nomonsters,
        "timer_game": sim.timer_game,
        "paused": sim.paused,
        "rng": list(sim.rng.get_state()),
    }

    # 2. Level World Geometry (Sectors, Lines, Sides)
    world_state: dict[str, Any] = {"sectors": [], "lines": [], "sides": []}
    if sim.level is not None:
        if hasattr(sim.level, "sectors"):
            world_state["sectors"] = [s.to_state() for s in sim.level.sectors]
        if hasattr(sim.level, "lines"):
            world_state["lines"] = [line.to_state() for line in sim.level.lines]
        if hasattr(sim.level, "sides"):
            world_state["sides"] = [si.to_state() for si in sim.level.sides]
    state["world"] = world_state

    # 3. Polyobjects
    poly_list: list[dict[str, Any]] = []
    if sim.level is not None and hasattr(sim.level, "polyobjs"):
        polys = (
            sim.level.polyobjs.values()
            if isinstance(sim.level.polyobjs, dict)
            else sim.level.polyobjs
        )
        for po in polys:
            if hasattr(po, "to_state"):
                poly_list.append(po.to_state())
    state["polyobjs"] = poly_list

    # 4. Mobjs & Thinkers mapping
    # Collect all mobjs and assign index mapping to serialize cross-object pointers
    mobj_list: list[Mobj] = []
    mobj_index_map: dict[int, int] = {}
    for thinker in sim.thinkers:
        if isinstance(thinker, Mobj) and not thinker.removed:
            if not save_players and thinker.player is not None:
                continue
            mobj_index_map[id(thinker)] = len(mobj_list)
            mobj_list.append(thinker)

    # Serialize thinkers in exact execution order
    thinker_entries: list[dict[str, Any]] = []
    for thinker in sim.thinkers:
        if thinker.removed:
            continue
        if isinstance(thinker, Mobj):
            if not save_players and thinker.player is not None:
                continue
            mo_dict = thinker.to_state(mobj_index_map)
            mo_dict["_is_mobj"] = True
            mo_dict["_mobj_idx"] = mobj_index_map[id(thinker)]
            thinker_entries.append(mo_dict)
        elif hasattr(thinker, "to_state"):
            t_dict = thinker.to_state()
            t_dict["_is_mobj"] = False
            thinker_entries.append(t_dict)
        else:
            # Fallback for generic thinkers
            thinker_entries.append(
                {
                    "_is_mobj": False,
                    "type": type(thinker).__name__,
                }
            )
    state["thinkers"] = thinker_entries

    # 5. Players
    players_list: list[dict[str, Any]] = []
    if save_players:
        for player in sim.players:
            players_list.append(player.to_state(mobj_index_map))
    state["players"] = players_list

    # 6. ACS VM & Store
    if sim.acs_vm is not None:
        state["acs_vm"] = sim.acs_vm.to_state()
    else:
        state["acs_vm"] = {"map_vars": [], "world_vars": []}

    state["acs_store"] = sim.acs_store.to_state()

    return state


def deserialize_sim(sim: GameSim, state: dict[str, Any]) -> None:
    """Restore simulation state from a dictionary produced by `serialize_sim`.

    Port of SV_LoadMap (sv_save.c:2069).
    """
    # 1. Header & Meta
    sim.level_time = state.get("level_time", 0)
    sim.map_num = state.get("map_num", getattr(sim, "map_num", 1))
    sim.cluster = state.get("cluster", getattr(sim, "cluster", 1))
    sim.skill = state.get("skill", sim.skill)
    sim.player_class = state.get("player_class", sim.player_class)
    sim.netgame = state.get("netgame", sim.netgame)
    sim.deathmatch = state.get("deathmatch", sim.deathmatch)
    sim.nomonsters = state.get("nomonsters", sim.nomonsters)
    sim.timer_game = state.get("timer_game", 0)
    sim.paused = state.get("paused", False)

    # 2. RNG
    if "rng" in state:
        sim.rng.set_state(tuple(state["rng"]))

    # 3. World (Sectors, Lines, Sides)
    world_data = state.get("world", {})
    if sim.level is not None:
        if "sectors" in world_data and hasattr(sim.level, "sectors"):
            for sec_data in world_data["sectors"]:
                idx = sec_data["index"]
                if 0 <= idx < len(sim.level.sectors):
                    sim.level.sectors[idx].from_state(sec_data)

        if "lines" in world_data and hasattr(sim.level, "lines"):
            for line_data in world_data["lines"]:
                idx = line_data["index"]
                if 0 <= idx < len(sim.level.lines):
                    sim.level.lines[idx].from_state(line_data)

        if "sides" in world_data and hasattr(sim.level, "sides"):
            for side_data in world_data["sides"]:
                idx = side_data["index"]
                if 0 <= idx < len(sim.level.sides):
                    sim.level.sides[idx].from_state(side_data)

    # 4. Polyobjects
    if sim.level is not None and hasattr(sim.level, "polyobjs"):
        poly_map: dict[int, Polyobj] = (
            sim.level.polyobjs
            if isinstance(sim.level.polyobjs, dict)
            else {po.tag: po for po in sim.level.polyobjs}
        )
        for poly_data in state.get("polyobjs", []):
            tag = poly_data.get("tag", 0)
            po = poly_map.get(tag)
            if po is not None:
                po.from_state(poly_data, sim=sim)

    # 5. Clear thinkers, spatial links, and player list
    sim.thinkers.clear()
    sim.tid_map.clear()
    if state.get("players"):
        sim.players.clear()

    if sim.level is not None and hasattr(sim.level, "sectors"):
        for sec in sim.level.sectors:
            sec.thing_list = None
            sec.special_data = None

    # 6. Reconstruct Thinkers and Mobjs in exact saved order
    restored_mobjs: dict[int, Mobj] = {}
    pending_mobj_links: list[tuple[Mobj, dict[str, Any]]] = []

    for entry in state.get("thinkers", []):
        if entry.get("_is_mobj", False):
            mobj_idx = entry.get("_mobj_idx", len(restored_mobjs))
            mo = Mobj()
            mo.from_state(entry)
            restored_mobjs[mobj_idx] = mo
            pending_mobj_links.append((mo, entry))

            if sim.level is not None:
                set_thing_position(sim.level, mo)

            sim.thinkers.add(mo)
            if mo.tid != 0:
                sim.tid_map.setdefault(mo.tid, []).append(mo)
        else:
            # Restore special mover thinker
            t_type = entry.get("type", "")
            thinker_obj = _restore_special_thinker(sim, t_type, entry)
            if thinker_obj is not None:
                sim.thinkers.add(thinker_obj)

    # Resolve target and tracer references for mobjs
    for mo, entry in pending_mobj_links:
        target_id = entry.get("target_id")
        if target_id == "PLAYER":
            if sim.players and sim.players[0].mo is not None:
                mo.target = sim.players[0].mo
        elif target_id is not None and target_id in restored_mobjs:
            mo.target = restored_mobjs[target_id]
        tracer_id = entry.get("tracer_id")
        if tracer_id == "PLAYER":
            if sim.players and sim.players[0].mo is not None:
                mo.tracer = sim.players[0].mo
        elif tracer_id is not None and tracer_id in restored_mobjs:
            mo.tracer = restored_mobjs[tracer_id]

    # 7. Players
    for p_data in state.get("players", []):
        player = Player()
        player.from_state(p_data)
        mo_id = p_data.get("mo_id")
        if mo_id is not None and mo_id in restored_mobjs:
            mo = restored_mobjs[mo_id]
            player.mo = mo
            mo.player = player
        sim.players.append(player)

    # 8. ACS VM & Store
    if sim.acs_vm is not None and "acs_vm" in state:
        sim.acs_vm.from_state(state["acs_vm"])

    if "acs_store" in state:
        sim.acs_store.from_state(state["acs_store"])


def _restore_special_thinker(
    sim: GameSim,
    t_type: str,
    entry: dict[str, Any],
) -> Any | None:
    """Instantiate a non-mobj thinker from its serialized dictionary."""
    if sim.level is None:
        return None

    if t_type == "VerticalDoor":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            door = VerticalDoor(
                sec,
                DoorType(entry.get("door_type", 0)),
                entry.get("speed", 0),
                entry.get("topwait", 0),
            )
            door.topheight = entry.get("topheight", 0)
            door.topcountdown = entry.get("topcountdown", 0)
            door.direction = entry.get("direction", 1)
            sec.special_data = door
            return door

    elif t_type == "FloorMove":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            fm = FloorMove(
                sec,
                entry.get("speed", 0),
                entry.get("dest", 0),
                entry.get("direction", 1),
                entry.get("crush", 0),
            )
            sec.special_data = fm
            return fm

    elif t_type == "FloorWaggle":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            fw = FloorWaggle(
                sec,
                entry.get("original_height", 0),
                entry.get("amplitude", 0),
                entry.get("frequency", 0),
            )
            fw.accumulator = entry.get("accumulator", 0)
            sec.special_data = fw
            return fw

    elif t_type == "CeilingMove":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            cm = CeilingMove(
                sec,
                entry.get("speed", 0),
                entry.get("topheight", 0),
                entry.get("bottomheight", 0),
                entry.get("direction", 1),
                entry.get("crush", 0),
                entry.get("repeat", False),
            )
            sec.special_data = cm
            return cm

    elif t_type == "Plat":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            plat = Plat(
                sec,
                entry.get("speed", 0),
                entry.get("low", 0),
                entry.get("high", 0),
                entry.get("direction", -1),
                entry.get("wait", 0),
            )
            plat.count = entry.get("count", 0)
            sec.special_data = plat
            return plat

    elif t_type == "Pillar":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            pil = Pillar(
                sec,
                entry.get("speed", 0),
                entry.get("floordest", 0),
                entry.get("ceilingdest", 0),
                entry.get("crush", 0),
            )
            pil.floor_done = entry.get("floor_done", False)
            pil.ceiling_done = entry.get("ceiling_done", False)
            sec.special_data = pil
            return pil

    elif t_type == "LightFade":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            lf = LightFade(sec, entry.get("target", 0), entry.get("tics", 1))
            lf.step = entry.get("step", 0.0)
            lf.current = entry.get("current", 0.0)
            return lf

    elif t_type == "LightGlow":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            lg = LightGlow(
                sec,
                entry.get("min_light", 0),
                entry.get("max_light", 0),
                entry.get("speed", 2),
            )
            lg.direction = entry.get("direction", 1)
            return lg

    elif t_type == "PhasedLightThinker":
        sec_idx = entry.get("sector_index", -1)
        if 0 <= sec_idx < len(sim.level.sectors):
            sec = sim.level.sectors[sec_idx]
            return PhasedLightThinker(sec, entry.get("base", 0), entry.get("index", 0))

    elif t_type == "ButtonThinker":
        line_idx = entry.get("line_index", -1)
        if 0 <= line_idx < len(sim.level.lines):
            line = sim.level.lines[line_idx]
            return ButtonThinker(
                line,
                SwitchWhere(entry.get("where", 0)),
                entry.get("btexture", ""),
                entry.get("btimer", 0),
                entry.get("sound_id", 0),
            )

    elif t_type == "ScrollThinker":
        line_idx = entry.get("line_index", -1)
        if 0 <= line_idx < len(sim.level.lines):
            line = sim.level.lines[line_idx]
            side = line.front_side
            if side is not None:
                st = ScrollThinker(line, side, entry.get("dx", 0), entry.get("dy", 0))
                side.texture_offset = entry.get("texture_offset", side.texture_offset)
                side.row_offset = entry.get("row_offset", side.row_offset)
                return st

    elif t_type == "RotatePolyThinker":
        tag = entry.get("polyobj", 0)
        po = sim.level.polyobjs.get(tag) if hasattr(sim.level, "polyobjs") else None
        rpt = RotatePolyThinker(po if po else tag, entry.get("speed", 0), entry.get("dist", 0))
        if po is not None:
            po.specialdata = rpt
        return rpt

    elif t_type == "MovePolyThinker":
        tag = entry.get("polyobj", 0)
        po = sim.level.polyobjs.get(tag) if hasattr(sim.level, "polyobjs") else None
        mpt = MovePolyThinker(
            po if po else tag,
            entry.get("speed", 0),
            entry.get("dist", 0),
            entry.get("angle", 0),
            entry.get("x_speed", 0),
            entry.get("y_speed", 0),
        )
        if po is not None:
            po.specialdata = mpt
        return mpt

    elif t_type == "PolyDoorThinker":
        tag = entry.get("polyobj", 0)
        po = sim.level.polyobjs.get(tag) if hasattr(sim.level, "polyobjs") else None
        pdt = PolyDoorThinker(
            po if po else tag,
            PolyDoorType(entry.get("door_type", 1)),
            entry.get("speed", 0),
            entry.get("dist", 0),
            entry.get("tics", 0),
            entry.get("close", False),
        )
        pdt.total_dist = entry.get("total_dist", 0)
        pdt.direction = entry.get("direction", 1)
        pdt.x_speed = entry.get("x_speed", 0)
        pdt.y_speed = entry.get("y_speed", 0)
        pdt.wait_tics = entry.get("wait_tics", 0)
        if po is not None:
            po.specialdata = pdt
        return pdt

    elif t_type == "RunningScript":
        if sim.acs_vm is not None and sim.acs_vm.behavior is not None:
            snum = entry.get("number", 0)
            sdef = sim.acs_vm.behavior.get_script(snum)
            if sdef is not None:
                line_idx = entry.get("line_index")
                line_obj = (
                    sim.level.lines[line_idx]
                    if line_idx is not None and 0 <= line_idx < len(sim.level.lines)
                    else None
                )
                rs = RunningScript(
                    number=snum,
                    def_=sdef,
                    ip=entry.get("ip", 0),
                    state=ACScriptState(entry.get("state", 0)),
                    vars=list(entry.get("vars", [])),
                    stack=list(entry.get("stack", [])),
                    delay_count=entry.get("delay_count", 0),
                    wait_value=entry.get("wait_value", 0),
                    line=line_obj,
                    side=entry.get("side", 0),
                    print_buffer=list(entry.get("print_buffer", [])),
                )
                sim.acs_vm.scripts[snum] = rs
                return rs

    return None


def save_sim_to_bytes(sim: GameSim, save_players: bool = True) -> bytes:
    """Serialize simulation state to compressed gzip JSON bytes."""
    state_dict = serialize_sim(sim, save_players=save_players)
    data = json.dumps(state_dict, separators=(",", ":")).encode("utf-8")
    return gzip.compress(data)


def load_sim_from_bytes(sim: GameSim, data: bytes) -> None:
    """Restore simulation state from compressed gzip JSON bytes."""
    decompressed = gzip.decompress(data)
    state_dict = json.loads(decompressed.decode("utf-8"))
    deserialize_sim(sim, state_dict)
