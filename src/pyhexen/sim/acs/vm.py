# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""ACS Virtual Machine core interpreter, game pcodes, and script lifecycle (T8.2, T8.3, T8.4).

Port of p_acs.c.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pyhexen.core.events import PrintMessage, SoundEvent
from pyhexen.core.fixed import c_div, c_mod
from pyhexen.sim.acs.loader import (
    ACS_STACK_DEPTH,
    MAX_ACS_MAP_VARS,
    MAX_ACS_SCRIPT_VARS,
    MAX_ACS_WORLD_VARS,
    ACScriptDef,
    ACScriptState,
    BehaviorData,
)
from pyhexen.sim.acs.pcodes import PCode
from pyhexen.sim.info_data import MobjFlag, MobjType
from pyhexen.sim.specials.dispatcher import (
    execute_line_special,
    find_lines_from_id,
    find_sectors_from_tag,
    notify_sector_changed,
)
from pyhexen.sim.thinker import Thinker
from pyhexen.world.level import ML_BLOCKING, Line

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.sim.mobj import Mobj

MAX_INSTRUCTIONS_PER_TIC: int = 50000


def _to_int32(val: int) -> int:
    """Clamp integer to signed 32-bit two's complement range."""
    val = val & 0xFFFFFFFF
    return val if val < 0x80000000 else val - 0x100000000


# Translation of ACS thing types (T_*) to internal MobjType (MT_*).
# Port of TranslateThingType array in p_things.c.
TRANSLATE_THING_TYPE: tuple[int, ...] = (
    MobjType.MT_MAPSPOT,  # 0: T_NONE
    MobjType.MT_CENTAUR,  # 1: T_CENTAUR
    MobjType.MT_CENTAURLEADER,  # 2: T_CENTAURLEADER
    MobjType.MT_DEMON,  # 3: T_DEMON
    MobjType.MT_ETTIN,  # 4: T_ETTIN
    MobjType.MT_FIREDEMON,  # 5: T_FIREGARGOYLE
    MobjType.MT_SERPENT,  # 6: T_WATERLURKER
    MobjType.MT_SERPENTLEADER,  # 7: T_WATERLURKERLEADER
    MobjType.MT_WRAITH,  # 8: T_WRAITH
    MobjType.MT_WRAITHB,  # 9: T_WRAITHBURIED
    MobjType.MT_FIREBALL1,  # 10: T_FIREBALL1
    MobjType.MT_MANA1,  # 11: T_MANA1
    MobjType.MT_MANA2,  # 12: T_MANA2
    MobjType.MT_SPEEDBOOTS,  # 13: T_ITEMBOOTS
    MobjType.MT_ARTIEGG,  # 14: T_ITEMEGG
    MobjType.MT_ARTIFLY,  # 15: T_ITEMFLIGHT
    MobjType.MT_SUMMONMAULATOR,  # 16: T_ITEMSUMMON
    MobjType.MT_TELEPORTOTHER,  # 17: T_ITEMTPORTOTHER
    MobjType.MT_ARTITELEPORT,  # 18: T_ITEMTELEPORT
    MobjType.MT_BISHOP,  # 19: T_BISHOP
    MobjType.MT_ICEGUY,  # 20: T_ICEGOLEM
    MobjType.MT_BRIDGE,  # 21: T_BRIDGE
    MobjType.MT_BOOSTARMOR,  # 22: T_DRAGONSKINBRACERS
    MobjType.MT_HEALINGBOTTLE,  # 23: T_ITEMHEALTHPOTION
    MobjType.MT_HEALTHFLASK,  # 24: T_ITEMHEALTHFLASK
    MobjType.MT_ARTISUPERHEAL,  # 25: T_ITEMHEALTHFULL
    MobjType.MT_BOOSTMANA,  # 26: T_ITEMBOOSTMANA
    MobjType.MT_FW_AXE,  # 27: T_FIGHTERAXE
    MobjType.MT_FW_HAMMER,  # 28: T_FIGHTERHAMMER
    MobjType.MT_FW_SWORD1,  # 29: T_FIGHTERSWORD1
    MobjType.MT_FW_SWORD2,  # 30: T_FIGHTERSWORD2
    MobjType.MT_FW_SWORD3,  # 31: T_FIGHTERSWORD3
    MobjType.MT_CW_SERPSTAFF,  # 32: T_CLERICSTAFF
    MobjType.MT_CW_HOLY1,  # 33: T_CLERICHOLY1
    MobjType.MT_CW_HOLY2,  # 34: T_CLERICHOLY2
    MobjType.MT_CW_HOLY3,  # 35: T_CLERICHOLY3
    MobjType.MT_MW_CONE,  # 36: T_MAGESHARDS
    MobjType.MT_MW_STAFF1,  # 37: T_MAGESTAFF1
    MobjType.MT_MW_STAFF2,  # 38: T_MAGESTAFF2
    MobjType.MT_MW_STAFF3,  # 39: T_MAGESTAFF3
    MobjType.MT_EGGFX,  # 40: T_MORPHBLAST
    MobjType.MT_ROCK1,  # 41: T_ROCK1
    MobjType.MT_ROCK2,  # 42: T_ROCK2
    MobjType.MT_ROCK3,  # 43: T_ROCK3
    MobjType.MT_DIRT1,  # 44: T_DIRT1
    MobjType.MT_DIRT2,  # 45: T_DIRT2
    MobjType.MT_DIRT3,  # 46: T_DIRT3
    MobjType.MT_DIRT4,  # 47: T_DIRT4
    MobjType.MT_DIRT5,  # 48: T_DIRT5
    MobjType.MT_DIRT6,  # 49: T_DIRT6
    MobjType.MT_ARROW,  # 50: T_ARROW
    MobjType.MT_DART,  # 51: T_DART
    MobjType.MT_POISONDART,  # 52: T_POISONDART
    MobjType.MT_RIPPERBALL,  # 53: T_RIPPERBALL
    MobjType.MT_SGSHARD1,  # 54: T_STAINEDGLASS1
    MobjType.MT_SGSHARD2,  # 55: T_STAINEDGLASS2
    MobjType.MT_SGSHARD3,  # 56: T_STAINEDGLASS3
    MobjType.MT_SGSHARD4,  # 57: T_STAINEDGLASS4
    MobjType.MT_SGSHARD5,  # 58: T_STAINEDGLASS5
    MobjType.MT_SGSHARD6,  # 59: T_STAINEDGLASS6
    MobjType.MT_SGSHARD7,  # 60: T_STAINEDGLASS7
    MobjType.MT_SGSHARD8,  # 61: T_STAINEDGLASS8
    MobjType.MT_SGSHARD9,  # 62: T_STAINEDGLASS9
    MobjType.MT_SGSHARD0,  # 63: T_STAINEDGLASS0
    MobjType.MT_PROJECTILE_BLADE,  # 64: T_BLADE
    MobjType.MT_ICESHARD,  # 65: T_ICESHARD
    MobjType.MT_FLAME_SMALL,  # 66: T_FLAME_SMALL
    MobjType.MT_FLAME_LARGE,  # 67: T_FLAME_LARGE
    MobjType.MT_ARMOR_1,  # 68: T_MESHARMOR
    MobjType.MT_ARMOR_2,  # 69: T_FALCONSHIELD
    MobjType.MT_ARMOR_3,  # 70: T_PLATINUMHELM
    MobjType.MT_ARMOR_4,  # 71: T_AMULETOFWARDING
    MobjType.MT_ARTIPOISONBAG,  # 72: T_ITEMFLECHETTE
    MobjType.MT_ARTITORCH,  # 73: T_ITEMTORCH
    MobjType.MT_BLASTRADIUS,  # 74: T_ITEMREPULSION
    MobjType.MT_MANA3,  # 75: T_MANA3
    MobjType.MT_ARTIPUZZSKULL,  # 76: T_PUZZSKULL
    MobjType.MT_ARTIPUZZGEMBIG,  # 77: T_PUZZGEMBIG
    MobjType.MT_ARTIPUZZGEMRED,  # 78: T_PUZZGEMRED
    MobjType.MT_ARTIPUZZGEMGREEN1,  # 79: T_PUZZGEMGREEN1
    MobjType.MT_ARTIPUZZGEMGREEN2,  # 80: T_PUZZGEMGREEN2
    MobjType.MT_ARTIPUZZGEMBLUE1,  # 81: T_PUZZGEMBLUE1
    MobjType.MT_ARTIPUZZGEMBLUE2,  # 82: T_PUZZGEMBLUE2
    MobjType.MT_ARTIPUZZBOOK1,  # 83: T_PUZZBOOK1
    MobjType.MT_ARTIPUZZBOOK2,  # 84: T_PUZZBOOK2
    MobjType.MT_KEY1,  # 85: T_METALKEY
    MobjType.MT_KEY2,  # 86: T_SMALLMETALKEY
    MobjType.MT_KEY3,  # 87: T_AXEKEY
    MobjType.MT_KEY4,  # 88: T_FIREKEY
    MobjType.MT_KEY5,  # 89: T_GREENKEY
    MobjType.MT_KEY6,  # 90: T_MACEKEY
    MobjType.MT_KEY7,  # 91: T_SILVERKEY
    MobjType.MT_KEY8,  # 92: T_RUSTYKEY
    MobjType.MT_KEY9,  # 93: T_HORNKEY
    MobjType.MT_KEYA,  # 94: T_SERPENTKEY
    MobjType.MT_WATER_DRIP,  # 95: T_WATERDRIP
    MobjType.MT_FLAME_SMALL_TEMP,  # 96: T_TEMPSMALLFLAME
    MobjType.MT_FLAME_SMALL,  # 97: T_PERMSMALLFLAME
    MobjType.MT_FLAME_LARGE_TEMP,  # 98: T_TEMPLARGEFLAME
    MobjType.MT_FLAME_LARGE,  # 99: T_PERMLARGEFLAME
    MobjType.MT_DEMON_MASH,  # 100: T_DEMON_MASH
    MobjType.MT_DEMON2_MASH,  # 101: T_DEMON2_MASH
    MobjType.MT_ETTIN_MASH,  # 102: T_ETTIN_MASH
    MobjType.MT_CENTAUR_MASH,  # 103: T_CENTAUR_MASH
    MobjType.MT_THRUSTFLOOR_UP,  # 104: T_THRUSTSPIKEUP
    MobjType.MT_THRUSTFLOOR_DOWN,  # 105: T_THRUSTSPIKEDOWN
    MobjType.MT_WRAITHFX4,  # 106: T_FLESH_DRIP1
    MobjType.MT_WRAITHFX5,  # 107: T_FLESH_DRIP2
    MobjType.MT_WRAITHFX2,  # 108: T_SPARK_DRIP
)


def thing_count(sim: GameSim, type_: int, tid: int) -> int:
    """Port of ThingCount (p_acs.c)."""
    if type_ == 0 and tid == 0:
        return 0
    mo_type = TRANSLATE_THING_TYPE[type_] if 0 <= type_ < len(TRANSLATE_THING_TYPE) else 0
    count = 0
    if tid != 0:
        for mobj in sim.find_mobjs_by_tid(tid):
            if type_ == 0:
                count += 1
            elif mobj.type == mo_type:
                if (mobj.flags & MobjFlag.MF_COUNTKILL) and mobj.health <= 0:
                    continue
                count += 1
    else:
        from pyhexen.sim.mobj import Mobj

        for thinker in sim.thinkers:
            if isinstance(thinker, Mobj) and not thinker.removed:
                if thinker.type == mo_type:
                    if (thinker.flags & MobjFlag.MF_COUNTKILL) and thinker.health <= 0:
                        continue
                    count += 1
    return count


@dataclass(slots=True)
class RunningScript(Thinker):
    """Port of acs_t (p_acs.c). State of an active ACS script thread."""

    number: int
    def_: ACScriptDef
    ip: int
    state: ACScriptState = ACScriptState.ASTE_RUNNING
    vars: list[int] = field(default_factory=lambda: [0] * MAX_ACS_SCRIPT_VARS)
    stack: list[int] = field(default_factory=list)
    delay_count: int = 0
    wait_value: int = 0
    activator: Mobj | None = None
    line: Line | None = None
    side: int = 0
    print_buffer: list[str] = field(default_factory=list)

    def tick(self, sim: GameSim) -> None:
        if sim.acs_vm is not None:
            sim.acs_vm.run_script(self, sim)

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "ACS",
            self.number,
            self.ip,
            int(self.state),
            tuple(self.vars),
            tuple(self.stack),
            self.delay_count,
            self.wait_value,
        )


class ACSVM:
    """Port of ACS engine in p_acs.c."""

    def __init__(self, behavior: BehaviorData | None = None) -> None:
        self.behavior: BehaviorData | None = behavior
        self.map_vars: list[int] = [0] * MAX_ACS_MAP_VARS
        self.world_vars: list[int] = [0] * MAX_ACS_WORLD_VARS
        self.scripts: dict[int, RunningScript] = {}

    def set_behavior(self, behavior: BehaviorData) -> None:
        self.behavior = behavior
        self.map_vars = [0] * MAX_ACS_MAP_VARS
        self.scripts.clear()

    def start_open_scripts(self, sim: GameSim) -> None:
        """Port of P_StartOpenACS (p_acs.c)."""
        if self.behavior is None:
            return
        for sdef in self.behavior.open_scripts:
            # Open scripts run after 1-second initialization delay (35 tics)
            self.start_script(
                sdef.number,
                args=(0, 0, 0),
                sim=sim,
                delay=35,
            )

    def start_script(
        self,
        number: int,
        args: Sequence[int] = (0, 0, 0),
        activator: Mobj | None = None,
        line: Line | None = None,
        side: int = 0,
        sim: GameSim | None = None,
        delay: int = 0,
    ) -> RunningScript | None:
        """Port of P_StartACS (p_acs.c)."""
        if self.behavior is None:
            return None
        sdef = self.behavior.get_script(number)
        if sdef is None:
            return None

        # Check if already running or suspended
        existing = self.scripts.get(number)
        if existing is not None and not existing.removed:
            if existing.state == ACScriptState.ASTE_SUSPENDED:
                existing.state = ACScriptState.ASTE_RUNNING
                return existing
            return None

        script = RunningScript(
            number=number,
            def_=sdef,
            ip=sdef.offset,
            state=ACScriptState.ASTE_RUNNING,
            delay_count=delay,
            activator=activator,
            line=line,
            side=side,
        )
        for i, val in enumerate(args[:MAX_ACS_SCRIPT_VARS]):
            script.vars[i] = _to_int32(val)

        self.scripts[number] = script
        if sim is not None:
            sim.thinkers.add(script)
        return script

    def suspend_script(self, number: int) -> bool:
        """Port of P_SuspendACS (p_acs.c)."""
        script = self.scripts.get(number)
        if script is not None and not script.removed and script.state == ACScriptState.ASTE_RUNNING:
            script.state = ACScriptState.ASTE_SUSPENDED
            return True
        return False

    def terminate_script(self, number: int, sim: GameSim | None = None) -> bool:
        """Port of P_TerminateACS (p_acs.c)."""
        script = self.scripts.get(number)
        if script is not None and not script.removed:
            script.state = ACScriptState.ASTE_TERMINATING
            script.removed = True
            if sim is not None:
                self.script_finished(number, sim)
            return True
        return False

    def is_tag_busy(self, tag: int, sim: GameSim) -> bool:
        """Port of TagBusy (p_acs.c)."""
        if sim.level is None or tag == 0:
            return False
        return any(s.special_data is not None for s in find_sectors_from_tag(sim.level, tag))

    def tag_finished(self, tag: int, sim: GameSim | None = None) -> None:
        """Port of P_TagFinished (p_acs.c)."""
        if sim is not None and self.is_tag_busy(tag, sim):
            return
        for s in self.scripts.values():
            if (
                not s.removed
                and s.state == ACScriptState.ASTE_WAITINGFORTAG
                and s.wait_value == tag
            ):
                s.state = ACScriptState.ASTE_RUNNING

    @staticmethod
    def is_poly_busy(po_tag: int, sim: GameSim) -> bool:
        """Check if polyobject with `po_tag` has active thinkers (po_man.c PO_Busy)."""
        if sim.level is None or not hasattr(sim.level, "polyobjs"):
            return False
        if isinstance(sim.level.polyobjs, dict):
            po = sim.level.polyobjs.get(po_tag)
        else:
            po = next((p for p in sim.level.polyobjs if getattr(p, "tag", None) == po_tag), None)
        if po is None:
            return False
        return bool(po.busy)

    def poly_finished(self, po: int, sim: GameSim | None = None) -> None:
        """Port of P_PolyobjFinished (p_acs.c:851-867)."""
        if sim is not None and self.is_poly_busy(po, sim):
            return
        for s in list(self.scripts.values()):
            if (
                not s.removed
                and s.state == ACScriptState.ASTE_WAITINGFORPOLY
                and s.wait_value == po
            ):
                s.state = ACScriptState.ASTE_RUNNING

    def script_finished(self, number: int, sim: GameSim) -> None:
        """Port of ScriptFinished (p_acs.c)."""
        for s in list(self.scripts.values()):
            if (
                not s.removed
                and s.state == ACScriptState.ASTE_WAITINGFORSCRIPT
                and s.wait_value == number
            ):
                s.state = ACScriptState.ASTE_RUNNING

    def run_script(self, script: RunningScript, sim: GameSim) -> None:
        """Port of P_RunACS / T_InterpretACS (p_acs.c)."""
        if script.removed:
            return

        if script.state != ACScriptState.ASTE_RUNNING:
            return

        if script.delay_count > 0:
            script.delay_count -= 1
            return

        assert self.behavior is not None
        code = self.behavior.code
        code_len = len(code)

        def read_int() -> int:
            nonlocal script
            if script.ip + 4 > code_len:
                return 0
            (val,) = struct.unpack_from("<i", code, script.ip)
            script.ip += 4
            return val

        def push(val: int) -> None:
            if len(script.stack) < ACS_STACK_DEPTH:
                script.stack.append(_to_int32(val))

        def pop() -> int:
            return script.stack.pop() if script.stack else 0

        instructions_count = 0

        while instructions_count < MAX_INSTRUCTIONS_PER_TIC:
            instructions_count += 1
            if script.ip >= code_len:
                script.removed = True
                self.script_finished(script.number, sim)
                return

            cmd = read_int()

            if cmd == PCode.PCD_NOP:
                pass
            elif cmd == PCode.PCD_TERMINATE:
                script.removed = True
                script.state = ACScriptState.ASTE_INACTIVE
                self.script_finished(script.number, sim)
                return
            elif cmd == PCode.PCD_SUSPEND:
                script.state = ACScriptState.ASTE_SUSPENDED
                return
            elif cmd == PCode.PCD_PUSHNUMBER:
                push(read_int())
            elif PCode.PCD_LSPEC1 <= cmd <= PCode.PCD_LSPEC5:
                count = cmd - PCode.PCD_LSPEC1 + 1
                special = read_int()
                args = [0, 0, 0, 0, 0]
                for i in range(count - 1, -1, -1):
                    args[i] = pop()
                execute_line_special(
                    sim,
                    special,
                    tuple(args),
                    line=script.line,
                    mo=script.activator,
                    side=script.side,
                )
            elif PCode.PCD_LSPEC1DIRECT <= cmd <= PCode.PCD_LSPEC5DIRECT:
                count = cmd - PCode.PCD_LSPEC1DIRECT + 1
                special = read_int()
                args = [read_int() for _ in range(count)]
                while len(args) < 5:
                    args.append(0)
                execute_line_special(
                    sim,
                    special,
                    tuple(args),
                    line=script.line,
                    mo=script.activator,
                    side=script.side,
                )
            elif cmd == PCode.PCD_ADD:
                push(pop() + pop())
            elif cmd == PCode.PCD_SUBTRACT:
                b = pop()
                a = pop()
                push(a - b)
            elif cmd == PCode.PCD_MULTIPLY:
                push(pop() * pop())
            elif cmd == PCode.PCD_DIVIDE:
                b = pop()
                a = pop()
                push(c_div(a, b) if b != 0 else 0)
            elif cmd == PCode.PCD_MODULUS:
                b = pop()
                a = pop()
                push(c_mod(a, b) if b != 0 else 0)
            elif cmd == PCode.PCD_EQ:
                push(1 if pop() == pop() else 0)
            elif cmd == PCode.PCD_NE:
                push(1 if pop() != pop() else 0)
            elif cmd == PCode.PCD_LT:
                b = pop()
                a = pop()
                push(1 if a < b else 0)
            elif cmd == PCode.PCD_GT:
                b = pop()
                a = pop()
                push(1 if a > b else 0)
            elif cmd == PCode.PCD_LE:
                b = pop()
                a = pop()
                push(1 if a <= b else 0)
            elif cmd == PCode.PCD_GE:
                b = pop()
                a = pop()
                push(1 if a >= b else 0)
            elif cmd == PCode.PCD_ASSIGNSCRIPTVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_SCRIPT_VARS:
                    script.vars[var] = _to_int32(val)
            elif cmd == PCode.PCD_ASSIGNMAPVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_MAP_VARS:
                    self.map_vars[var] = _to_int32(val)
            elif cmd == PCode.PCD_ASSIGNWORLDVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_WORLD_VARS:
                    self.world_vars[var] = _to_int32(val)
            elif cmd == PCode.PCD_PUSHSCRIPTVAR:
                var = read_int()
                push(script.vars[var] if 0 <= var < MAX_ACS_SCRIPT_VARS else 0)
            elif cmd == PCode.PCD_PUSHMAPVAR:
                var = read_int()
                push(self.map_vars[var] if 0 <= var < MAX_ACS_MAP_VARS else 0)
            elif cmd == PCode.PCD_PUSHWORLDVAR:
                var = read_int()
                push(self.world_vars[var] if 0 <= var < MAX_ACS_WORLD_VARS else 0)
            elif cmd == PCode.PCD_ADDSCRIPTVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_SCRIPT_VARS:
                    script.vars[var] = _to_int32(script.vars[var] + val)
            elif cmd == PCode.PCD_ADDMAPVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_MAP_VARS:
                    self.map_vars[var] = _to_int32(self.map_vars[var] + val)
            elif cmd == PCode.PCD_ADDWORLDVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_WORLD_VARS:
                    self.world_vars[var] = _to_int32(self.world_vars[var] + val)
            elif cmd == PCode.PCD_SUBSCRIPTVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_SCRIPT_VARS:
                    script.vars[var] = _to_int32(script.vars[var] - val)
            elif cmd == PCode.PCD_SUBMAPVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_MAP_VARS:
                    self.map_vars[var] = _to_int32(self.map_vars[var] - val)
            elif cmd == PCode.PCD_SUBWORLDVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_WORLD_VARS:
                    self.world_vars[var] = _to_int32(self.world_vars[var] - val)
            elif cmd == PCode.PCD_MULSCRIPTVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_SCRIPT_VARS:
                    script.vars[var] = _to_int32(script.vars[var] * val)
            elif cmd == PCode.PCD_MULMAPVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_MAP_VARS:
                    self.map_vars[var] = _to_int32(self.map_vars[var] * val)
            elif cmd == PCode.PCD_MULWORLDVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_WORLD_VARS:
                    self.world_vars[var] = _to_int32(self.world_vars[var] * val)
            elif cmd == PCode.PCD_DIVSCRIPTVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_SCRIPT_VARS and val != 0:
                    script.vars[var] = _to_int32(c_div(script.vars[var], val))
            elif cmd == PCode.PCD_DIVMAPVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_MAP_VARS and val != 0:
                    self.map_vars[var] = _to_int32(c_div(self.map_vars[var], val))
            elif cmd == PCode.PCD_DIVWORLDVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_WORLD_VARS and val != 0:
                    self.world_vars[var] = _to_int32(c_div(self.world_vars[var], val))
            elif cmd == PCode.PCD_MODSCRIPTVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_SCRIPT_VARS and val != 0:
                    script.vars[var] = _to_int32(c_mod(script.vars[var], val))
            elif cmd == PCode.PCD_MODMAPVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_MAP_VARS and val != 0:
                    self.map_vars[var] = _to_int32(c_mod(self.map_vars[var], val))
            elif cmd == PCode.PCD_MODWORLDVAR:
                var = read_int()
                val = pop()
                if 0 <= var < MAX_ACS_WORLD_VARS and val != 0:
                    self.world_vars[var] = _to_int32(c_mod(self.world_vars[var], val))
            elif cmd == PCode.PCD_INCSCRIPTVAR:
                var = read_int()
                if 0 <= var < MAX_ACS_SCRIPT_VARS:
                    script.vars[var] = _to_int32(script.vars[var] + 1)
            elif cmd == PCode.PCD_INCMAPVAR:
                var = read_int()
                if 0 <= var < MAX_ACS_MAP_VARS:
                    self.map_vars[var] = _to_int32(self.map_vars[var] + 1)
            elif cmd == PCode.PCD_INCWORLDVAR:
                var = read_int()
                if 0 <= var < MAX_ACS_WORLD_VARS:
                    self.world_vars[var] = _to_int32(self.world_vars[var] + 1)
            elif cmd == PCode.PCD_DECSCRIPTVAR:
                var = read_int()
                if 0 <= var < MAX_ACS_SCRIPT_VARS:
                    script.vars[var] = _to_int32(script.vars[var] - 1)
            elif cmd == PCode.PCD_DECMAPVAR:
                var = read_int()
                if 0 <= var < MAX_ACS_MAP_VARS:
                    self.map_vars[var] = _to_int32(self.map_vars[var] - 1)
            elif cmd == PCode.PCD_DECWORLDVAR:
                var = read_int()
                if 0 <= var < MAX_ACS_WORLD_VARS:
                    self.world_vars[var] = _to_int32(self.world_vars[var] - 1)
            elif cmd == PCode.PCD_GOTO:
                script.ip = read_int()
            elif cmd == PCode.PCD_IFGOTO:
                dest = read_int()
                if pop() != 0:
                    script.ip = dest
            elif cmd == PCode.PCD_IFNOTGOTO:
                dest = read_int()
                if pop() == 0:
                    script.ip = dest
            elif cmd == PCode.PCD_DROP:
                pop()
            elif cmd == PCode.PCD_DELAY:
                script.delay_count = pop()
                return
            elif cmd == PCode.PCD_DELAYDIRECT:
                script.delay_count = read_int()
                return
            elif cmd == PCode.PCD_RANDOM:
                high = pop()
                low = pop()
                span = high - low + 1
                r = sim.rng.p_random()
                push(low + (r % span) if span > 0 else low)
            elif cmd == PCode.PCD_RANDOMDIRECT:
                low = read_int()
                high = read_int()
                span = high - low + 1
                r = sim.rng.p_random()
                push(low + (r % span) if span > 0 else low)
            elif cmd == PCode.PCD_THINGCOUNT:
                tid = pop()
                type_ = pop()
                push(thing_count(sim, type_, tid))
            elif cmd == PCode.PCD_THINGCOUNTDIRECT:
                type_ = read_int()
                tid = read_int()
                push(thing_count(sim, type_, tid))
            elif cmd == PCode.PCD_TAGWAIT:
                tag = pop()
                if self.is_tag_busy(tag, sim):
                    script.wait_value = tag
                    script.state = ACScriptState.ASTE_WAITINGFORTAG
                    return
            elif cmd == PCode.PCD_TAGWAITDIRECT:
                tag = read_int()
                if self.is_tag_busy(tag, sim):
                    script.wait_value = tag
                    script.state = ACScriptState.ASTE_WAITINGFORTAG
                    return
            elif cmd == PCode.PCD_POLYWAIT:
                tag = pop()
                if self.is_poly_busy(tag, sim):
                    script.wait_value = tag
                    script.state = ACScriptState.ASTE_WAITINGFORPOLY
                    return
            elif cmd == PCode.PCD_POLYWAITDIRECT:
                tag = read_int()
                if self.is_poly_busy(tag, sim):
                    script.wait_value = tag
                    script.state = ACScriptState.ASTE_WAITINGFORPOLY
                    return
            elif cmd == PCode.PCD_CHANGEFLOOR:
                f_idx = pop()
                tag = pop()
                f_name = (
                    self.behavior.strings[f_idx]
                    if 0 <= f_idx < len(self.behavior.strings)
                    else "FLOOR01"
                )
                if sim.level is not None:
                    for sec in find_sectors_from_tag(sim.level, tag):
                        sec.floor_pic = f_name
                        notify_sector_changed(sim, sec)
            elif cmd == PCode.PCD_CHANGEFLOORDIRECT:
                tag = read_int()
                f_idx = read_int()
                f_name = (
                    self.behavior.strings[f_idx]
                    if 0 <= f_idx < len(self.behavior.strings)
                    else "FLOOR01"
                )
                if sim.level is not None:
                    for sec in find_sectors_from_tag(sim.level, tag):
                        sec.floor_pic = f_name
                        notify_sector_changed(sim, sec)
            elif cmd == PCode.PCD_CHANGECEILING:
                c_idx = pop()
                tag = pop()
                c_name = (
                    self.behavior.strings[c_idx]
                    if 0 <= c_idx < len(self.behavior.strings)
                    else "CEIL01"
                )
                if sim.level is not None:
                    for sec in find_sectors_from_tag(sim.level, tag):
                        sec.ceiling_pic = c_name
                        notify_sector_changed(sim, sec)
            elif cmd == PCode.PCD_CHANGECEILINGDIRECT:
                tag = read_int()
                c_idx = read_int()
                c_name = (
                    self.behavior.strings[c_idx]
                    if 0 <= c_idx < len(self.behavior.strings)
                    else "CEIL01"
                )
                if sim.level is not None:
                    for sec in find_sectors_from_tag(sim.level, tag):
                        sec.ceiling_pic = c_name
                        notify_sector_changed(sim, sec)
            elif cmd == PCode.PCD_RESTART:
                script.ip = script.def_.offset
            elif cmd == PCode.PCD_ANDLOGICAL:
                b = pop()
                a = pop()
                push(1 if (a != 0 and b != 0) else 0)
            elif cmd == PCode.PCD_ORLOGICAL:
                b = pop()
                a = pop()
                push(1 if (a != 0 or b != 0) else 0)
            elif cmd == PCode.PCD_ANDBITWISE:
                b = pop()
                a = pop()
                push(a & b)
            elif cmd == PCode.PCD_ORBITWISE:
                b = pop()
                a = pop()
                push(a | b)
            elif cmd == PCode.PCD_EORBITWISE:
                b = pop()
                a = pop()
                push(a ^ b)
            elif cmd == PCode.PCD_NEGATELOGICAL:
                push(1 if pop() == 0 else 0)
            elif cmd == PCode.PCD_LSHIFT:
                b = pop()
                a = pop()
                push(a << (b & 31))
            elif cmd == PCode.PCD_RSHIFT:
                b = pop()
                a = pop()
                push(a >> (b & 31))
            elif cmd == PCode.PCD_UNARYMINUS:
                push(-pop())
            elif cmd == PCode.PCD_LINESIDE:
                push(script.side)
            elif cmd == PCode.PCD_SCRIPTWAIT:
                target_script = pop()
                target = self.scripts.get(target_script)
                if (
                    target is not None
                    and not target.removed
                    and target.state != ACScriptState.ASTE_INACTIVE
                ):
                    script.wait_value = target_script
                    script.state = ACScriptState.ASTE_WAITINGFORSCRIPT
                    return
            elif cmd == PCode.PCD_SCRIPTWAITDIRECT:
                target_script = read_int()
                target = self.scripts.get(target_script)
                if (
                    target is not None
                    and not target.removed
                    and target.state != ACScriptState.ASTE_INACTIVE
                ):
                    script.wait_value = target_script
                    script.state = ACScriptState.ASTE_WAITINGFORSCRIPT
                    return
            elif cmd == PCode.PCD_CLEARLINESPECIAL:
                if script.line is not None:
                    script.line.special = 0
            elif cmd == PCode.PCD_CASEGOTO:
                val = read_int()
                dest = read_int()
                if script.stack and script.stack[-1] == val:
                    script.stack.pop()
                    script.ip = dest
            elif cmd == PCode.PCD_BEGINPRINT:
                script.print_buffer.clear()
            elif cmd == PCode.PCD_PRINTSTRING:
                idx = pop()
                if 0 <= idx < len(self.behavior.strings):
                    script.print_buffer.append(self.behavior.strings[idx])
            elif cmd == PCode.PCD_PRINTNUMBER:
                script.print_buffer.append(str(pop()))
            elif cmd == PCode.PCD_PRINTCHARACTER:
                script.print_buffer.append(chr(pop() & 0xFF))
            elif cmd == PCode.PCD_ENDPRINT:
                msg = "".join(script.print_buffer)
                sim.events.emit(PrintMessage(text=msg, bold=False))
                script.print_buffer.clear()
            elif cmd == PCode.PCD_ENDPRINTBOLD:
                msg = "".join(script.print_buffer)
                sim.events.emit(PrintMessage(text=msg, bold=True))
                script.print_buffer.clear()
            elif cmd == PCode.PCD_PLAYERCOUNT:
                alive_players = len([p for p in sim.players if p.mo and p.mo.health > 0])
                push(alive_players if alive_players > 0 else 1)
            elif cmd == PCode.PCD_GAMETYPE:
                # 0 = single player, 1 = deathmatch, 2 = coop
                push(1 if sim.deathmatch else (2 if sim.netgame else 0))
            elif cmd == PCode.PCD_GAMESKILL:
                push(sim.skill)
            elif cmd == PCode.PCD_TIMER:
                push(sim.level_time)
            elif cmd == PCode.PCD_SECTORSOUND:
                vol = pop()
                s_idx = pop()
                s_name = (
                    self.behavior.strings[s_idx]
                    if 0 <= s_idx < len(self.behavior.strings)
                    else "DEFAULT"
                )
                x = 0
                y = 0
                if script.line is not None and script.line.front_sector is not None:
                    snd_org = script.line.front_sector.sound_origin
                    x, y = snd_org[0], snd_org[1]
                sim.events.emit(SoundEvent(sound_id=s_name, volume=vol, x=x, y=y))
            elif cmd == PCode.PCD_AMBIENTSOUND:
                vol = pop()
                s_idx = pop()
                s_name = (
                    self.behavior.strings[s_idx]
                    if 0 <= s_idx < len(self.behavior.strings)
                    else "DEFAULT"
                )
                sim.events.emit(SoundEvent(sound_id=s_name, volume=vol))
            elif cmd == PCode.PCD_SOUNDSEQUENCE:
                s_idx = pop()
                seq_name = (
                    self.behavior.strings[s_idx]
                    if 0 <= s_idx < len(self.behavior.strings)
                    else "DEFAULT"
                )
                x = 0
                y = 0
                if script.line is not None and script.line.front_sector is not None:
                    snd_org = script.line.front_sector.sound_origin
                    x, y = snd_org[0], snd_org[1]
                sim.events.emit(SoundEvent(sound_id=seq_name, volume=127, x=x, y=y))
            elif cmd == PCode.PCD_SETLINETEXTURE:
                tex_idx = pop()
                pos = pop()
                side = pop()
                tag = pop()
                t_name = (
                    self.behavior.strings[tex_idx]
                    if 0 <= tex_idx < len(self.behavior.strings)
                    else "-"
                )
                if sim.level is not None:
                    lines = find_lines_from_id(sim.level, tag, sim)
                    for ld in lines:
                        sd = ld.front_side if side == 0 else ld.back_side
                        if sd is not None:
                            if pos == 0:
                                sd.top_texture = t_name
                            elif pos == 1:
                                sd.mid_texture = t_name
                            else:
                                sd.bottom_texture = t_name
            elif cmd == PCode.PCD_SETLINEBLOCKING:
                blocking = pop()
                tag = pop()
                if sim.level is not None:
                    lines = find_lines_from_id(sim.level, tag, sim)
                    for ld in lines:
                        if blocking:
                            ld.flags |= ML_BLOCKING
                        else:
                            ld.flags &= ~ML_BLOCKING
            elif cmd == PCode.PCD_SETLINESPECIAL:
                args = [pop() for _ in range(5)]
                special = pop()
                tag = pop()
                if sim.level is not None:
                    lines = find_lines_from_id(sim.level, tag, sim)
                    for ld in lines:
                        ld.special = special
                        ld.args = tuple(args)  # type: ignore[assignment]
            elif cmd == PCode.PCD_THINGSOUND:
                vol = pop()
                s_idx = pop()
                tid = pop()
                s_name = (
                    self.behavior.strings[s_idx]
                    if 0 <= s_idx < len(self.behavior.strings)
                    else "DEFAULT"
                )
                mobjs = sim.find_mobjs_by_tid(tid)
                for m in mobjs:
                    sim.events.emit(SoundEvent(sound_id=s_name, volume=vol, x=m.x, y=m.y, z=m.z))
            else:
                pass
