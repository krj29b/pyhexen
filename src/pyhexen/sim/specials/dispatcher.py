# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Line special table, registry, dispatcher, and sector/line tag lookup helpers.

Port of P_ExecuteLineSpecial, P_FindSectorFromTag, P_FindLine, getNextSector,
P_FindLowestFloorSurrounding, P_FindHighestFloorSurrounding, P_FindNextHighestFloor,
P_FindLowestCeilingSurrounding, and P_FindHighestCeilingSurrounding (p_spec.c).
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pyhexen.core.angles import ANG90
from pyhexen.core.events import PrintMessage, SectorChanged, SoundEvent
from pyhexen.core.fixed import FRACBITS, FRACUNIT, MAXINT
from pyhexen.sim.info_data import MobjFlag2
from pyhexen.sim.mobj import Mobj, remove_mobj
from pyhexen.sim.movement import thrust_mobj
from pyhexen.world.level import ML_TWOSIDED, Level, Line, Sector

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

SpecialHandler = Callable[..., bool]

# Complete table of Hexen line specials (p_spec.c)
SPECIAL_NAMES: dict[int, str] = {
    1: "Polyobj_StartLine",
    2: "Polyobj_RotateLeft",
    3: "Polyobj_RotateRight",
    4: "Polyobj_Move",
    5: "Polyobj_ExplicitLine",
    6: "Polyobj_MoveTimes8",
    7: "Polyobj_DoorSwing",
    8: "Polyobj_DoorSlide",
    10: "Door_Close",
    11: "Door_Open",
    12: "Door_Raise",
    13: "Door_LockedRaise",
    20: "Floor_LowerByValue",
    21: "Floor_LowerToLowest",
    22: "Floor_LowerToNearest",
    23: "Floor_RaiseByValue",
    24: "Floor_RaiseToHighest",
    25: "Floor_RaiseToNearest",
    26: "Stairs_BuildDownNormal",
    27: "Stairs_BuildUpNormal",
    28: "Floor_RaiseAndCrush",
    29: "Pillar_Build",
    30: "Pillar_Open",
    31: "Stairs_BuildDownSync",
    32: "Stairs_BuildUpSync",
    35: "Floor_RaiseByValueTimes8",
    36: "Floor_LowerByValueTimes8",
    40: "Ceiling_LowerByValue",
    41: "Ceiling_RaiseByValue",
    42: "Ceiling_CrushAndRaise",
    43: "Ceiling_LowerAndCrush",
    44: "Ceiling_CrushStop",
    45: "Ceiling_CrushRaiseAndStay",
    46: "Floor_CrushStop",
    60: "Plat_PerpetualRaise",
    61: "Plat_Stop",
    62: "Plat_DownWaitUpStay",
    63: "Plat_DownByValueWaitUpStay",
    64: "Plat_UpWaitDownStay",
    65: "Plat_UpByValueWaitDownStay",
    66: "Floor_LowerInstant",
    67: "Floor_RaiseInstant",
    68: "Floor_MoveToValueTimes8",
    69: "Ceiling_MoveToValueTimes8",
    70: "Teleport",
    71: "Teleport_NoFog",
    72: "ThrustThing",
    73: "DamageThing",
    74: "Teleport_NewMap",
    75: "Teleport_EndGame",
    80: "ACS_Execute",
    81: "ACS_Suspend",
    82: "ACS_Terminate",
    83: "ACS_LockedExecute",
    90: "Polyobj_OR_RotateLeft",
    91: "Polyobj_OR_RotateRight",
    92: "Polyobj_OR_Move",
    93: "Polyobj_OR_MoveTimes8",
    94: "Pillar_BuildAndCrush",
    95: "FloorAndCeiling_LowerByValue",
    96: "FloorAndCeiling_RaiseByValue",
    100: "Scroll_Texture_Left",
    101: "Scroll_Texture_Right",
    102: "Scroll_Texture_Up",
    103: "Scroll_Texture_Down",
    109: "Light_ForceLightning",
    110: "Light_RaiseByValue",
    111: "Light_LowerByValue",
    112: "Light_ChangeToValue",
    113: "Light_Fade",
    114: "Light_Glow",
    115: "Light_Flicker",
    116: "Light_Strobe",
    120: "Radius_Quake",
    121: "Line_SetIdentification",
    129: "UsePuzzleItem",
    130: "Thing_Activate",
    131: "Thing_Deactivate",
    132: "Thing_Remove",
    133: "Thing_Destroy",
    134: "Thing_Projectile",
    135: "Thing_Spawn",
    136: "Thing_ProjectileGravity",
    137: "Thing_SpawnNoFog",
    138: "Floor_Waggle",
    140: "Sector_ChangeSound",
    243: "Exit_Normal",
    244: "Exit_Secret",
}

_HANDLERS: dict[int, SpecialHandler] = {}

_SIBLING_MODULES: tuple[str, ...] = (
    "doors",
    "floors",
    "ceilings",
    "plats",
    "pillars",
    "lights",
    "teleport",
    "switches",
    "scroll",
)


def register_special(
    special_id: int,
    handler: SpecialHandler,
    name: str | None = None,
) -> None:
    """Register a line special handler function for `special_id`."""
    _HANDLERS[special_id] = handler
    if name is not None and special_id not in SPECIAL_NAMES:
        SPECIAL_NAMES[special_id] = name


def line_special(
    special_id: int,
    name: str | None = None,
) -> Callable[[SpecialHandler], SpecialHandler]:
    """Decorator registering a line special handler for `special_id`."""

    def _decorator(func: SpecialHandler) -> SpecialHandler:
        register_special(special_id, func, name=name)
        return func

    return _decorator


def _sync_sibling_handlers() -> None:
    """Dynamically import and merge `SPECIAL_HANDLERS` from sibling modules in `sim.specials`."""
    for mod_name in _SIBLING_MODULES:
        try:
            mod = importlib.import_module(f"pyhexen.sim.specials.{mod_name}")
        except ImportError:
            continue
        handlers = getattr(mod, "SPECIAL_HANDLERS", None)
        if isinstance(handlers, dict):
            for spec_id, func in handlers.items():
                if spec_id not in _HANDLERS:
                    _HANDLERS[spec_id] = func


def notify_sector_changed(sim: GameSim, sector: Sector) -> None:
    """Emit a `SectorChanged` event on `sim.events` for `sector` (ADR-004)."""
    sim.events.emit(SectorChanged(sector_index=sector.index))


def find_sectors_from_tag(level: Level, tag: int) -> list[Sector]:
    """Return all sectors in `level` whose `tag == tag`.

    Port of P_FindSectorFromTag (p_spec.c).
    """
    if tag == 0:
        return []
    return [sec for sec in level.sectors if sec.tag == tag]


def find_lines_from_id(level: Level, line_id: int, sim: GameSim | None = None) -> list[Line]:
    """Return all lines identified by `line_id` (via `Line_SetIdentification` special 121 or arg1).

    Port of P_FindLine (p_spec.c).
    """
    if line_id == 0:
        return []
    if sim is not None and line_id in sim.tagged_lines:
        return list(sim.tagged_lines[line_id])
    return [ld for ld in level.lines if ld.special == 121 and ld.args[0] == line_id]


def get_next_sector(line: Line, sec: Sector) -> Sector | None:
    """Return the sector on the other side of two-sided `line` from `sec`, or `None`.

    Port of getNextSector (p_spec.c).
    """
    if not (line.flags & ML_TWOSIDED):
        return None
    if line.front_sector is sec:
        return line.back_sector
    return line.front_sector


def find_lowest_floor_surrounding(sec: Sector) -> int:
    """Return the lowest floor height among adjacent sectors (including `sec.floor_height`).

    Port of P_FindLowestFloorSurrounding (p_spec.c).
    """
    floor = sec.floor_height
    for check in sec.lines:
        other = get_next_sector(check, sec)
        if other is not None and other.floor_height < floor:
            floor = other.floor_height
    return floor


def find_highest_floor_surrounding(sec: Sector) -> int:
    """Return the highest floor height among adjacent sectors.

    Port of P_FindHighestFloorSurrounding (p_spec.c).
    """
    floor = -500 * FRACUNIT
    for check in sec.lines:
        other = get_next_sector(check, sec)
        if other is not None and other.floor_height > floor:
            floor = other.floor_height
    return floor


def find_next_highest_floor(sec: Sector, current_height: int) -> int:
    """Return the lowest adjacent floor height strictly above `current_height`.

    Port of P_FindNextHighestFloor (p_spec.c).
    """
    higher = [
        other.floor_height
        for check in sec.lines
        if (other := get_next_sector(check, sec)) is not None
        and other.floor_height > current_height
    ]
    if not higher:
        return 0
    return min(higher)


def find_lowest_ceiling_surrounding(sec: Sector) -> int:
    """Return the lowest ceiling height among adjacent sectors.

    Port of P_FindLowestCeilingSurrounding (p_spec.c).
    """
    height = MAXINT
    for check in sec.lines:
        other = get_next_sector(check, sec)
        if other is not None and other.ceiling_height < height:
            height = other.ceiling_height
    return height


def find_highest_ceiling_surrounding(sec: Sector) -> int:
    """Return the highest ceiling height among adjacent sectors.

    Port of P_FindHighestCeilingSurrounding (p_spec.c).
    """
    height = 0
    for check in sec.lines:
        other = get_next_sector(check, sec)
        if other is not None and other.ceiling_height > height:
            height = other.ceiling_height
    return height


def _invoke_handler(
    handler: SpecialHandler,
    sim: GameSim,
    special: int,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int,
) -> bool:
    """Invoke `handler`, adapting to its positional parameter names/count."""
    try:
        sig = inspect.signature(handler)
    except (ValueError, TypeError):
        return bool(handler(sim, args, line, mo))

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return bool(handler(sim, args, line, mo))

    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    name_map: dict[str, Any] = {
        "sim": sim,
        "game_sim": sim,
        "special": special,
        "special_id": special,
        "args": args,
        "arg": args,
        "line": line,
        "ld": line,
        "mo": mo,
        "mobj": mo,
        "thing": mo,
        "activator": mo,
        "side": side,
    }
    if all(p.name in name_map for p in pos_params):
        call_args = [name_map[p.name] for p in pos_params]
        return bool(handler(*call_args))

    default_order = [sim, args, line, mo, side]
    return bool(handler(*default_order[: len(pos_params)]))


def execute_line_special(
    sim: GameSim,
    special: int,
    args: Sequence[int],
    line: Line | None = None,
    mo: Mobj | None = None,
    *,
    side: int = 0,
) -> bool:
    """Execute Hexen line special `special` with 5-byte `args`.

    Port of P_ExecuteLineSpecial (p_spec.c).
    """
    _sync_sibling_handlers()
    padded = tuple(int(args[i]) if i < len(args) else 0 for i in range(5))
    args5: tuple[int, int, int, int, int] = (
        padded[0],
        padded[1],
        padded[2],
        padded[3],
        padded[4],
    )

    handler = _HANDLERS.get(special)
    if handler is not None:
        return _invoke_handler(handler, sim, special, args5, line, mo, side)

    return False


# Polyobject line specials (po_man.c, p_spec.c)


@line_special(1, "Polyobj_StartLine")
def _ev_poly_start_line(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 1 (p_spec.c:513). Initialization special; no-op at runtime."""
    _ = (sim, args, line)
    return False


@line_special(2, "Polyobj_RotateLeft")
def _ev_poly_rotate_left(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 2 (p_spec.c:516)."""
    from pyhexen.sim.polyobj import ev_rotate_poly

    return ev_rotate_poly(sim, line, args, direction=1, override=False)


@line_special(3, "Polyobj_RotateRight")
def _ev_poly_rotate_right(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 3 (p_spec.c:519)."""
    from pyhexen.sim.polyobj import ev_rotate_poly

    return ev_rotate_poly(sim, line, args, direction=-1, override=False)


@line_special(4, "Polyobj_Move")
def _ev_poly_move(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 4 (p_spec.c:522)."""
    from pyhexen.sim.polyobj import ev_move_poly

    return ev_move_poly(sim, line, args, times_eight=False, override=False)


@line_special(5, "Polyobj_ExplicitLine")
def _ev_poly_explicit_line(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 5 (p_spec.c:524). Initialization special; no-op at runtime."""
    _ = (sim, args, line)
    return False


@line_special(6, "Polyobj_MoveTimes8")
def _ev_poly_move_times8(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 6 (p_spec.c:527)."""
    from pyhexen.sim.polyobj import ev_move_poly

    return ev_move_poly(sim, line, args, times_eight=True, override=False)


@line_special(7, "Polyobj_DoorSwing")
def _ev_poly_door_swing(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 7 (p_spec.c:530)."""
    from pyhexen.sim.polyobj import PolyDoorType, ev_open_poly_door

    return ev_open_poly_door(sim, line, args, PolyDoorType.PODOOR_SWING)


@line_special(8, "Polyobj_DoorSlide")
def _ev_poly_door_slide(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 8 (p_spec.c:533)."""
    from pyhexen.sim.polyobj import PolyDoorType, ev_open_poly_door

    return ev_open_poly_door(sim, line, args, PolyDoorType.PODOOR_SLIDE)


@line_special(90, "Polyobj_OR_RotateLeft")
def _ev_poly_or_rotate_left(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 90 (p_spec.c:741)."""
    from pyhexen.sim.polyobj import ev_rotate_poly

    return ev_rotate_poly(sim, line, args, direction=1, override=True)


@line_special(91, "Polyobj_OR_RotateRight")
def _ev_poly_or_rotate_right(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 91 (p_spec.c:744)."""
    from pyhexen.sim.polyobj import ev_rotate_poly

    return ev_rotate_poly(sim, line, args, direction=-1, override=True)


@line_special(92, "Polyobj_OR_Move")
def _ev_poly_or_move(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 92 (p_spec.c:747)."""
    from pyhexen.sim.polyobj import ev_move_poly

    return ev_move_poly(sim, line, args, times_eight=False, override=True)


@line_special(93, "Polyobj_OR_MoveTimes8")
def _ev_poly_or_move_times8(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None = None,
) -> bool:
    """Port of Special 93 (p_spec.c:750)."""
    from pyhexen.sim.polyobj import ev_move_poly

    return ev_move_poly(sim, line, args, times_eight=True, override=True)


# Built-in miscellaneous line specials in p_spec.c


@line_special(72, "ThrustThing")
def _ev_thrust_thing(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    _ = (sim, line)
    if side != 0 or mo is None:
        return False
    angle = args[0] * (ANG90 // 64)
    thrust_mobj(mo, angle, args[1] << FRACBITS)
    return True


@line_special(73, "DamageThing")
def _ev_damage_thing(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
) -> bool:
    _ = (sim, line)
    if mo is None:
        return False
    dmg = args[0] if args[0] > 0 else 10000
    mo.health -= dmg
    if mo.player is not None:
        mo.player.health = mo.health
        mo.player.damagecount += dmg
    return True


@line_special(70, "Teleport")
def _ev_teleport(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    _ = line
    from pyhexen.sim.specials.teleport import special_teleport

    return special_teleport(sim, 70, args, line, mo, side=side)


@line_special(71, "Teleport_NoFog")
def _ev_teleport_no_fog(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    _ = line
    from pyhexen.sim.specials.teleport import special_teleport

    return special_teleport(sim, 71, args, line, mo, side=side)


@line_special(74, "Teleport_NewMap")
def _ev_teleport_new_map(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    _ = line
    if side != 0:
        return False
    if mo is not None and mo.player is not None and getattr(mo.player, "player_state", 0) == 1:
        return False
    from pyhexen.sim.transitions import teleport_new_map

    return teleport_new_map(sim, args[0], args[1])


@line_special(75, "Teleport_EndGame")
def _ev_teleport_end_game(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    _ = (args, line)
    if side != 0:
        return False
    if mo is not None and mo.player is not None and getattr(mo.player, "player_state", 0) == 1:
        return False
    from pyhexen.sim.transitions import teleport_end_game

    return teleport_end_game(sim)


@line_special(243, "Exit_Normal")
def _ev_exit_normal(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    _ = line
    if side != 0:
        return False
    if mo is not None and mo.player is not None and getattr(mo.player, "player_state", 0) == 1:
        return False
    from pyhexen.sim.transitions import exit_normal

    return exit_normal(sim, position=args[0])


@line_special(244, "Exit_Secret")
def _ev_exit_secret(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    _ = line
    if side != 0:
        return False
    if mo is not None and mo.player is not None and getattr(mo.player, "player_state", 0) == 1:
        return False
    from pyhexen.sim.transitions import exit_secret

    return exit_secret(sim, position=args[0])


@line_special(80, "ACS_Execute")
def _ev_acs_execute(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    script_num, target_map, arg1, arg2, arg3 = args
    if target_map != 0 and target_map != getattr(sim, "map_num", 1):
        return sim.acs_store.queue_script(target_map, script_num, (arg1, arg2, arg3))
    if sim.acs_vm is None:
        return False
    res = sim.acs_vm.start_script(
        script_num,
        args=(arg1, arg2, arg3),
        activator=mo,
        line=line,
        side=side,
        sim=sim,
    )
    return res is not None


@line_special(81, "ACS_Suspend")
def _ev_acs_suspend(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
) -> bool:
    script_num, target_map = args[0], args[1]
    if target_map != 0 and target_map != getattr(sim, "map_num", 1):
        return False
    if sim.acs_vm is None:
        return False
    return sim.acs_vm.suspend_script(script_num)


@line_special(82, "ACS_Terminate")
def _ev_acs_terminate(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
) -> bool:
    script_num, target_map = args[0], args[1]
    if target_map != 0 and target_map != getattr(sim, "map_num", 1):
        return False
    if sim.acs_vm is None:
        return False
    return sim.acs_vm.terminate_script(script_num, sim=sim)


@line_special(83, "ACS_LockedExecute")
def _ev_acs_locked_execute(
    sim: GameSim,
    args: tuple[int, int, int, int, int],
    line: Line | None,
    mo: Mobj | None,
    side: int = 0,
) -> bool:
    from pyhexen.sim.specials.doors import KEY_NAMES

    script_num, target_map, arg1, arg2, lock = args
    if mo is None or mo.player is None:
        return False
    if lock != 0:
        key_bit = 1 << (lock - 1)
        if not (mo.player.keys & key_bit):
            key_name = KEY_NAMES[lock - 1] if 0 <= lock - 1 < len(KEY_NAMES) else f"KEY {lock}"
            sim.events.emit(PrintMessage(text=f"YOU NEED THE {key_name}\n", bold=True))
            sim.events.emit(SoundEvent(sound_id="DoorLocked", volume=127))
            return False
    if target_map != 0 and target_map != getattr(sim, "map_num", 1):
        return sim.acs_store.queue_script(target_map, script_num, (arg1, arg2, 0))
    if sim.acs_vm is None:
        return False
    res = sim.acs_vm.start_script(
        script_num,
        args=(arg1, arg2, 0),
        activator=mo,
        line=line,
        side=side,
        sim=sim,
    )
    return res is not None


@line_special(109, "Light_ForceLightning")
def _ev_force_lightning(sim: GameSim) -> bool:
    if sim.surface_animator is not None:
        sim.surface_animator.force_lightning()
    return True


@line_special(130, "Thing_Activate")
def _ev_thing_activate(sim: GameSim, args: tuple[int, int, int, int, int]) -> bool:
    mobjs = sim.find_mobjs_by_tid(args[0])
    if not mobjs:
        return False
    for m in mobjs:
        m.flags2 &= ~MobjFlag2.MF2_DORMANT
        if m.tics == -1:
            m.tics = 1
    return True


@line_special(131, "Thing_Deactivate")
def _ev_thing_deactivate(sim: GameSim, args: tuple[int, int, int, int, int]) -> bool:
    mobjs = sim.find_mobjs_by_tid(args[0])
    if not mobjs:
        return False
    for m in mobjs:
        m.flags2 |= MobjFlag2.MF2_DORMANT
        m.tics = -1
    return True


@line_special(132, "Thing_Remove")
def _ev_thing_remove(sim: GameSim, args: tuple[int, int, int, int, int]) -> bool:
    mobjs = sim.find_mobjs_by_tid(args[0])
    if not mobjs:
        return False
    for m in mobjs:
        remove_mobj(sim, m)
    return True


@line_special(140, "Sector_ChangeSound")
def _ev_sector_change_sound(sim: GameSim, args: tuple[int, int, int, int, int]) -> bool:
    if sim.level is None or args[0] == 0:
        return False
    secs = find_sectors_from_tag(sim.level, args[0])
    for s in secs:
        s.seq_type = args[1]
    return bool(secs)


def get_specials_report() -> dict[int, tuple[str, bool]]:
    """Return a sorted mapping of `special_id -> (name, is_implemented)`."""
    _sync_sibling_handlers()
    return {
        spec_id: (name, spec_id in _HANDLERS) for spec_id, name in sorted(SPECIAL_NAMES.items())
    }
