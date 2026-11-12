# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for ACS VM core, game pcodes, script lifecycle, and ACS store (T8.2, T8.3, T8.4)."""

import struct
from types import SimpleNamespace

import pytest

from pyhexen.core.events import PrintMessage, SequenceEvent, SoundEvent
from pyhexen.sim.acs.loader import (
    ACScriptDef,
    ACScriptState,
    BehaviorData,
)
from pyhexen.sim.acs.pcodes import PCode
from pyhexen.sim.acs.vm import ACSVM
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag, MobjType
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.player import Player
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.world.level import Sector


def build_behavior(
    scripts_code: dict[int, list[int | PCode]],
    strings: list[str] | None = None,
    open_script_nums: list[int] | None = None,
) -> BehaviorData:
    """Helper to assemble a synthetic BehaviorData for ACS VM tests."""
    code_buf = bytearray()
    script_defs: list[ACScriptDef] = []
    scripts_by_number: dict[int, ACScriptDef] = {}

    for num, code_list in sorted(scripts_code.items()):
        offset = len(code_buf)
        for item in code_list:
            code_buf.extend(struct.pack("<i", int(item)))
        is_open = bool(open_script_nums and num in open_script_nums)
        sdef = ACScriptDef(
            number=num,
            offset=offset,
            arg_count=3,
            is_open=is_open,
        )
        script_defs.append(sdef)
        scripts_by_number[num] = sdef

    return BehaviorData(
        code=bytes(code_buf),
        info_offset=len(code_buf),
        scripts=tuple(script_defs),
        strings=tuple(strings or []),
        scripts_by_number=scripts_by_number,
    )


# =========================================================================
# T8.2 VM Core Tests
# =========================================================================


class TestACSVMCore:
    """Acceptance tests for T8.2: Stack machine, arithmetic, logic, vars, flow, and delay."""

    def test_arithmetic_and_stack(self) -> None:
        """Test PUSHNUMBER, ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS, UNARYMINUS, DROP."""
        # 10 + 5 = 15; 15 - 3 = 12; 12 * 2 = 24; 24 / 4 = 6; 6 % 4 = 2; -2; drop; terminate
        code = [
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_PUSHNUMBER,
            5,
            PCode.PCD_ADD,
            PCode.PCD_PUSHNUMBER,
            3,
            PCode.PCD_SUBTRACT,
            PCode.PCD_PUSHNUMBER,
            2,
            PCode.PCD_MULTIPLY,
            PCode.PCD_PUSHNUMBER,
            4,
            PCode.PCD_DIVIDE,
            PCode.PCD_PUSHNUMBER,
            4,
            PCode.PCD_MODULUS,
            PCode.PCD_UNARYMINUS,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()

        # Result: 10+5=15; 15-3=12; 12*2=24; 24/4=6; 6%4=2; -(2) = -2
        assert script.vars[0] == -2
        assert script.removed is True

    def test_divide_and_mod_by_zero(self) -> None:
        """Divide or modulo by 0 should produce 0 without raising exception."""
        code = [
            PCode.PCD_PUSHNUMBER,
            42,
            PCode.PCD_PUSHNUMBER,
            0,
            PCode.PCD_DIVIDE,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_PUSHNUMBER,
            42,
            PCode.PCD_PUSHNUMBER,
            0,
            PCode.PCD_MODULUS,
            PCode.PCD_ASSIGNSCRIPTVAR,
            1,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()
        assert script.vars[0] == 0
        assert script.vars[1] == 0

    def test_bitwise_and_shifts(self) -> None:
        """Test ANDBITWISE, ORBITWISE, EORBITWISE, LSHIFT, RSHIFT."""
        code = [
            # 0xF0 & 0x33 = 0x30
            PCode.PCD_PUSHNUMBER,
            0xF0,
            PCode.PCD_PUSHNUMBER,
            0x33,
            PCode.PCD_ANDBITWISE,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            # 0xF0 | 0x0F = 0xFF
            PCode.PCD_PUSHNUMBER,
            0xF0,
            PCode.PCD_PUSHNUMBER,
            0x0F,
            PCode.PCD_ORBITWISE,
            PCode.PCD_ASSIGNSCRIPTVAR,
            1,
            # 0xFF ^ 0x0F = 0xF0
            PCode.PCD_PUSHNUMBER,
            0xFF,
            PCode.PCD_PUSHNUMBER,
            0x0F,
            PCode.PCD_EORBITWISE,
            PCode.PCD_ASSIGNSCRIPTVAR,
            2,
            # 1 << 4 = 16
            PCode.PCD_PUSHNUMBER,
            1,
            PCode.PCD_PUSHNUMBER,
            4,
            PCode.PCD_LSHIFT,
            PCode.PCD_ASSIGNSCRIPTVAR,
            3,
            # 32 >> 2 = 8
            PCode.PCD_PUSHNUMBER,
            32,
            PCode.PCD_PUSHNUMBER,
            2,
            PCode.PCD_RSHIFT,
            PCode.PCD_ASSIGNSCRIPTVAR,
            4,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()
        assert script.vars[0] == 0x30
        assert script.vars[1] == 0xFF
        assert script.vars[2] == 0xF0
        assert script.vars[3] == 16
        assert script.vars[4] == 8

    def test_comparisons_and_logic(self) -> None:
        """Test EQ, NE, LT, GT, LE, GE, ANDLOGICAL, ORLOGICAL, NEGATELOGICAL."""
        code = [
            # 5 == 5 -> 1
            PCode.PCD_PUSHNUMBER,
            5,
            PCode.PCD_PUSHNUMBER,
            5,
            PCode.PCD_EQ,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            # 5 < 10 -> 1
            PCode.PCD_PUSHNUMBER,
            5,
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_LT,
            PCode.PCD_ASSIGNSCRIPTVAR,
            1,
            # 10 <= 10 -> 1
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_LE,
            PCode.PCD_ASSIGNSCRIPTVAR,
            2,
            # 10 > 5 -> 1
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_PUSHNUMBER,
            5,
            PCode.PCD_GT,
            PCode.PCD_ASSIGNSCRIPTVAR,
            3,
            # 1 && 0 -> 0
            PCode.PCD_PUSHNUMBER,
            1,
            PCode.PCD_PUSHNUMBER,
            0,
            PCode.PCD_ANDLOGICAL,
            PCode.PCD_ASSIGNSCRIPTVAR,
            4,
            # 1 || 0 -> 1
            PCode.PCD_PUSHNUMBER,
            1,
            PCode.PCD_PUSHNUMBER,
            0,
            PCode.PCD_ORLOGICAL,
            PCode.PCD_ASSIGNSCRIPTVAR,
            5,
            # !0 -> 1
            PCode.PCD_PUSHNUMBER,
            0,
            PCode.PCD_NEGATELOGICAL,
            PCode.PCD_ASSIGNSCRIPTVAR,
            6,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()
        assert script.vars[0] == 1
        assert script.vars[1] == 1
        assert script.vars[2] == 1
        assert script.vars[3] == 1
        assert script.vars[4] == 0
        assert script.vars[5] == 1
        assert script.vars[6] == 1

    def test_variables_scope_and_ops(self) -> None:
        """Test script, map, and world variable assignment, push, add, sub, mul, inc, dec."""
        code = [
            # Script var 0 = 10, then inc, dec, add 5
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_INCSCRIPTVAR,
            0,  # 11
            PCode.PCD_PUSHNUMBER,
            4,
            PCode.PCD_ADDSCRIPTVAR,
            0,  # 15
            PCode.PCD_DECSCRIPTVAR,
            0,  # 14
            # Map var 2 = 100, add 50
            PCode.PCD_PUSHNUMBER,
            100,
            PCode.PCD_ASSIGNMAPVAR,
            2,
            PCode.PCD_PUSHNUMBER,
            50,
            PCode.PCD_ADDMAPVAR,
            2,
            # World var 5 = 20, mul 3
            PCode.PCD_PUSHNUMBER,
            20,
            PCode.PCD_ASSIGNWORLDVAR,
            5,
            PCode.PCD_PUSHNUMBER,
            3,
            PCode.PCD_MULWORLDVAR,
            5,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()
        assert script.vars[0] == 14
        assert vm.map_vars[2] == 150
        assert vm.world_vars[5] == 60

    def test_control_flow_and_branching(self) -> None:
        """Test GOTO, IFGOTO, IFNOTGOTO, CASEGOTO."""
        # Byte layout:
        # 0: PUSHNUMBER (cmd=3)
        # 4: 1
        # 8: IFNOTGOTO (cmd=79)
        # 12: offset 32 (skip to var=99)
        # 16: PUSHNUMBER (cmd=3)
        # 20: 42
        # 24: ASSIGNSCRIPTVAR (cmd=25)
        # 28: 0
        # 32: PUSHNUMBER (cmd=3)
        # 36: 10
        # 40: CASEGOTO (cmd=84)
        # 44: val=10
        # 48: offset 64 (jump to terminate)
        # 52: PUSHNUMBER 999
        # 56: ASSIGNSCRIPTVAR 1
        # 60: TERMINATE
        # 64: TERMINATE
        code = [
            PCode.PCD_PUSHNUMBER,
            1,
            PCode.PCD_IFNOTGOTO,
            32,
            PCode.PCD_PUSHNUMBER,
            42,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_CASEGOTO,
            10,
            64,
            PCode.PCD_PUSHNUMBER,
            999,
            PCode.PCD_ASSIGNSCRIPTVAR,
            1,
            PCode.PCD_TERMINATE,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()
        assert script.vars[0] == 42
        assert script.vars[1] == 0  # Was skipped by CASEGOTO
        assert script.removed is True

    def test_delay_suspends_execution(self) -> None:
        """Test DELAY and DELAYDIRECT."""
        code = [
            PCode.PCD_PUSHNUMBER,
            10,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_DELAYDIRECT,
            3,  # Delay 3 tics
            PCode.PCD_PUSHNUMBER,
            20,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None

        # Tic 1: executes until delay, delay_count becomes 3
        sim.tick()
        assert script.vars[0] == 10
        assert script.delay_count == 3
        assert not script.removed

        # Tic 2: delay decrements to 2
        sim.tick()
        assert script.vars[0] == 10
        assert script.delay_count == 2

        # Tic 3: delay decrements to 1
        sim.tick()
        assert script.vars[0] == 10
        assert script.delay_count == 1

        # Tic 4: delay decrements to 0
        sim.tick()
        assert script.vars[0] == 10
        assert script.delay_count == 0

        # Tic 5: resumes and completes
        sim.tick()
        assert script.vars[0] == 20
        assert script.removed is True

    def test_budget_guard_infinite_loop(self) -> None:
        """Test that infinite loop does not hang simulation and yields."""
        # 0: GOTO 0
        code = [
            PCode.PCD_GOTO,
            0,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None

        # Ticking should hit instruction limit (50000) and yield safely
        sim.tick()
        assert script.state == ACScriptState.ASTE_RUNNING
        assert not script.removed


# =========================================================================
# T8.3 Game PCodes Tests
# =========================================================================


class TestACSGamePCodes:
    """Acceptance tests for T8.3: Line specials, waits, prints, sounds, textures, queries."""

    def test_line_specials_direct_and_stack(self) -> None:
        """Test PCD_LSPEC1DIRECT..5DIRECT and stack PCD_LSPEC1..5."""
        # Special 73: DamageThing (dmg)
        # Stack: push 50; LSPEC1 73
        # Direct: LSPEC1DIRECT 73, 25
        code = [
            PCode.PCD_PUSHNUMBER,
            50,
            PCode.PCD_LSPEC1,
            73,
            PCode.PCD_LSPEC1DIRECT,
            73,
            25,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        # DamageThing goes through P_DamageMobj, which only hurts shootable things.
        target_mo = Mobj(
            type=MobjType.MT_PLAYER_FIGHTER,
            x=0,
            y=0,
            z=0,
            health=200,
            flags=MobjFlag.MF_SHOOTABLE,
        )
        script = vm.start_script(1, activator=target_mo, sim=sim)
        assert script is not None

        sim.tick()
        # 200 - 50 - 25 = 125
        assert target_mo.health == 125
        assert script.removed is True

    def test_print_and_printbold(self) -> None:
        """Test BEGINPRINT, PRINTSTRING, PRINTNUMBER, PRINTCHARACTER, ENDPRINT, ENDPRINTBOLD."""
        strings = ["HELLO ", " WORLD"]
        code = [
            PCode.PCD_BEGINPRINT,
            PCode.PCD_PUSHNUMBER,
            0,
            PCode.PCD_PRINTSTRING,
            PCode.PCD_PUSHNUMBER,
            42,
            PCode.PCD_PRINTNUMBER,
            PCode.PCD_PUSHNUMBER,
            33,  # '!'
            PCode.PCD_PRINTCHARACTER,
            PCode.PCD_ENDPRINT,
            PCode.PCD_BEGINPRINT,
            PCode.PCD_PUSHNUMBER,
            1,
            PCode.PCD_PRINTSTRING,
            PCode.PCD_ENDPRINTBOLD,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code}, strings=strings)
        sim = GameSim()
        player = Player()
        sim.players.append(player)  # players[consoleplayer] always exists in vanilla
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        messages: list[PrintMessage] = []
        sim.events.subscribe(PrintMessage, messages.append)

        vm.start_script(1, sim=sim)
        sim.tick()

        assert len(messages) == 2
        assert messages[0].text == "HELLO 42!"
        assert messages[0].bold is False
        assert messages[1].text == " WORLD"
        assert messages[1].bold is True
        # CmdEndPrintBold uses P_SetYellowMessage with ultmsg=true; it replaced the print.
        assert player.message == " WORLD"
        assert player.yellow_message is True
        assert player.ultimate_message is True

    def test_sound_events(self) -> None:
        """Test PCD_AMBIENTSOUND, PCD_SECTORSOUND, PCD_THINGSOUND, PCD_SOUNDSEQUENCE."""
        strings = ["Thunder", "DoorOpen", "Seq1"]
        code = [
            # Ambient sound: vol=127, str_idx=0
            PCode.PCD_PUSHNUMBER,
            0,
            PCode.PCD_PUSHNUMBER,
            127,
            PCode.PCD_AMBIENTSOUND,
            # Sound sequence: str_idx=2
            PCode.PCD_PUSHNUMBER,
            2,
            PCode.PCD_SOUNDSEQUENCE,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code}, strings=strings)
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        sounds: list[SoundEvent] = []
        sequences: list[SequenceEvent] = []
        sim.events.subscribe(SoundEvent, sounds.append)
        sim.events.subscribe(SequenceEvent, sequences.append)

        vm.start_script(1, sim=sim)
        sim.tick()

        assert len(sounds) == 1
        assert sounds[0].sound_id == "Thunder"
        assert sounds[0].volume == 127
        assert sounds[0].origin is None  # ambient: at the listener
        # No activating line: the sequence plays at the listener (origin None).
        assert [(e.sequence, e.origin, e.stop) for e in sequences] == [("Seq1", None, False)]

    def test_tagwait_and_tag_finished(self) -> None:
        """Test PCD_TAGWAIT and tag_finished wake-up."""
        code = [
            PCode.PCD_TAGWAITDIRECT,
            5,
            PCode.PCD_PUSHNUMBER,
            99,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sec = Sector(0, 0, 128 << 16, "FLOOR01", "CEIL01", 128, 0, tag=5)
        level = SimpleNamespace(sectors=[sec])
        sim = GameSim(level=level)  # type: ignore[arg-type]
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        # Make sector busy
        sec.special_data = "BUSY_THINKER"

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()

        assert script.state == ACScriptState.ASTE_WAITINGFORTAG
        assert script.wait_value == 5
        assert script.vars[0] == 0

        # Free sector and notify tag finished
        sec.special_data = None
        sim.tag_finished(5)

        assert script.state == ACScriptState.ASTE_RUNNING
        sim.tick()
        assert script.vars[0] == 99
        assert script.removed is True

    def test_scriptwait_and_script_finished(self) -> None:
        """Test PCD_SCRIPTWAIT and wake-up when target terminates."""
        # Script 1 waits for script 2
        code1 = [
            PCode.PCD_SCRIPTWAITDIRECT,
            2,
            PCode.PCD_PUSHNUMBER,
            111,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_TERMINATE,
        ]
        # Script 2 delays 2 tics then finishes
        code2 = [
            PCode.PCD_DELAYDIRECT,
            2,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code1, 2: code2})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        s1 = vm.start_script(1, sim=sim)
        s2 = vm.start_script(2, sim=sim)
        assert s1 is not None and s2 is not None

        # Tic 1: s1 enters WAITINGFORSCRIPT, s2 delays (delay_count=2)
        sim.tick()
        assert s1.state == ACScriptState.ASTE_WAITINGFORSCRIPT
        assert s1.vars[0] == 0

        # Tic 2: s2 delay decrements to 1
        sim.tick()
        assert s1.state == ACScriptState.ASTE_WAITINGFORSCRIPT

        # Tic 3: s2 delay decrements to 0
        sim.tick()
        assert s1.state == ACScriptState.ASTE_WAITINGFORSCRIPT

        # Tic 4: s2 runs from IP after delay, hits TERMINATE, wakes up s1
        sim.tick()
        assert s2.removed is True
        assert s1.state == ACScriptState.ASTE_RUNNING

        # Tic 5: s1 runs to completion
        sim.tick()
        assert s1.vars[0] == 111
        assert s1.removed is True

    def test_game_queries(self) -> None:
        """Test PCD_GAMETYPE, PCD_GAMESKILL, PCD_TIMER, PCD_PLAYERCOUNT."""
        code = [
            PCode.PCD_GAMETYPE,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_GAMESKILL,
            PCode.PCD_ASSIGNSCRIPTVAR,
            1,
            PCode.PCD_TIMER,
            PCode.PCD_ASSIGNSCRIPTVAR,
            2,
            PCode.PCD_PLAYERCOUNT,
            PCode.PCD_ASSIGNSCRIPTVAR,
            3,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim(skill=3, deathmatch=True)
        sim.level_time = 120
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()

        assert script.vars[0] == 1  # Deathmatch
        assert script.vars[1] == 3  # Skill
        assert script.vars[2] == 120  # Level time
        assert script.vars[3] == 1  # Player count

    def test_thing_count_pcode(self) -> None:
        """Test PCD_THINGCOUNT and PCD_THINGCOUNTDIRECT."""
        code = [
            # Count by TID 50 (any type)
            PCode.PCD_THINGCOUNTDIRECT,
            0,
            50,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            # Count Ettin (type=4) with TID 50
            PCode.PCD_THINGCOUNTDIRECT,
            4,
            50,
            PCode.PCD_ASSIGNSCRIPTVAR,
            1,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        m1 = spawn_mobj(sim, 0, 0, 0, MobjType.MT_ETTIN)
        m1.tid = 50
        sim.tid_map[50] = [m1]

        m2 = spawn_mobj(sim, 0, 0, 0, MobjType.MT_CENTAUR)
        m2.tid = 50
        sim.tid_map[50].append(m2)

        script = vm.start_script(1, sim=sim)
        assert script is not None
        sim.tick()

        assert script.vars[0] == 2  # Total mobjs with TID 50
        assert script.vars[1] == 1  # Only Ettins with TID 50


# =========================================================================
# T8.4 Lifecycle, Specials 80-83, Open Scripts, and ACSStore Tests
# =========================================================================


class TestACSLifecycleAndStore:
    """Acceptance tests for T8.4: ACS_Execute, Suspend, Terminate, LockedExecute, ACSStore."""

    def test_line_special_acs_execute_local(self) -> None:
        """Special 80 starts local script."""
        code = [
            PCode.PCD_PUSHNUMBER,
            777,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({10: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        res = execute_line_special(sim, 80, (10, 0, 0, 0, 0))
        assert res is True
        assert 10 in vm.scripts

        sim.tick()
        assert vm.scripts[10].vars[0] == 777
        assert vm.scripts[10].removed is True

    def test_line_special_acs_execute_remote_map(self) -> None:
        """Special 80 with remote map queues script into ACSStore."""
        behavior = build_behavior({10: [PCode.PCD_TERMINATE]})
        sim = GameSim()
        sim.map_num = 1
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        # Target map 2 (remote)
        res = execute_line_special(sim, 80, (10, 2, 1, 2, 3))
        assert res is True
        assert len(sim.acs_store.entries) == 1
        entry = sim.acs_store.entries[0]
        assert entry.map_num == 2
        assert entry.script_num == 10
        assert entry.args == (1, 2, 3)

        # Run pending for map 2 on map 2 sim
        sim2 = GameSim()
        sim2.map_num = 2
        sim2.acs_vm = ACSVM(behavior)
        sim.acs_store.run_pending_for_map(2, sim2)

        assert len(sim.acs_store.entries) == 0
        assert 10 in sim2.acs_vm.scripts

    def test_line_special_acs_suspend_and_terminate(self) -> None:
        """Specials 81 and 82 suspend and terminate running scripts."""
        code = [
            PCode.PCD_DELAYDIRECT,
            10,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({5: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        vm.start_script(5, sim=sim)
        sim.tick()
        assert vm.scripts[5].state == ACScriptState.ASTE_RUNNING

        # Suspend (81)
        res = execute_line_special(sim, 81, (5, 0, 0, 0, 0))
        assert res is True
        assert vm.scripts[5].state == ACScriptState.ASTE_SUSPENDED

        # Terminate (82)
        res = execute_line_special(sim, 82, (5, 0, 0, 0, 0))
        assert res is True
        assert vm.scripts[5].removed is True

    def test_line_special_locked_execute(self) -> None:
        """Special 83 requires key lock."""
        code = [
            PCode.PCD_PUSHNUMBER,
            1,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code})
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        player = Player()
        mo = Mobj(type=MobjType.MT_PLAYER_FIGHTER, x=0, y=0, z=0)
        mo.player = player
        player.mo = mo

        # Lock 1 = STEEL KEY (bit 1 << 0)
        # 1. Try without key -> fails with message
        messages: list[PrintMessage] = []
        sim.events.subscribe(PrintMessage, messages.append)

        res = execute_line_special(sim, 83, (1, 0, 0, 0, 1), mo=mo)
        assert res is False
        assert 1 not in vm.scripts
        assert len(messages) == 1
        assert "STEEL KEY" in messages[0].text

        # 2. Give key -> succeeds
        player.keys |= 1
        res = execute_line_special(sim, 83, (1, 0, 0, 0, 1), mo=mo)
        assert res is True
        assert 1 in vm.scripts

    def test_open_scripts_start_with_delay(self) -> None:
        """Open scripts automatically start on map load with a 35-tic delay."""
        code = [
            PCode.PCD_PUSHNUMBER,
            1234,
            PCode.PCD_ASSIGNSCRIPTVAR,
            0,
            PCode.PCD_TERMINATE,
        ]
        behavior = build_behavior({1: code}, open_script_nums=[1])
        sim = GameSim()
        vm = ACSVM(behavior)
        sim.acs_vm = vm

        vm.start_open_scripts(sim)
        assert 1 in vm.scripts
        script = vm.scripts[1]
        assert script.delay_count == 35

        # Tick 35 times -> script delays
        for _ in range(35):
            sim.tick()
        assert script.vars[0] == 0

        # Tick 36 -> runs to completion
        sim.tick()
        assert script.vars[0] == 1234
        assert script.removed is True


@pytest.mark.wad
def test_real_iwad_map01_behavior(iwad_path: str) -> None:
    """Test loading and running real MAP01 ACS behavior from Hexen IWAD."""
    from pyhexen.wad.wadfile import WadStack
    from pyhexen.world.level import load_level

    stack = WadStack([iwad_path])
    level = load_level(stack, "MAP01")
    assert level.behavior_bytes, "MAP01 must have BEHAVIOR lump"

    sim = GameSim(level=level)
    assert sim.acs_vm is not None
    assert len(sim.acs_vm.behavior.scripts) > 0

    # Spawning specials runs start_open_scripts
    sim.setup_level_specials()

    open_scripts = [s for s in sim.acs_vm.scripts.values() if s.def_.is_open]
    assert len(open_scripts) > 0, "MAP01 must have open scripts"

    # Tick 40 frames past the 35-tic initial delay
    for _ in range(40):
        sim.tick()
