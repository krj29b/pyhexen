# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T5.1: info_data tables, action registry, and report_actions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pyhexen.sim.actions import (
    _ACTIONS,
    action,
    get_action,
    get_action_report,
    is_action_implemented,
)
from pyhexen.sim.info_data import (
    MOBJINFO,
    SPRNAMES,
    STATES,
    MobjFlag,
    MobjFlag2,
    MobjType,
    SfxEnum,
    SpriteNum,
    StateNum,
)

FRACUNIT = 1 << 16


def test_info_table_counts() -> None:
    assert len(SPRNAMES) == SpriteNum.NUMSPRITES
    assert len(STATES) == StateNum.NUMSTATES
    assert len(MOBJINFO) == MobjType.NUMMOBJTYPES
    assert SpriteNum.NUMSPRITES == 300
    assert StateNum.NUMSTATES == 2846
    assert MobjType.NUMMOBJTYPES == 398


def test_sprnames_samples() -> None:
    assert SPRNAMES[SpriteNum.SPR_MAN1] == "MAN1"
    assert SPRNAMES[SpriteNum.SPR_PLAY] == "PLAY"
    assert SPRNAMES[SpriteNum.SPR_KORX] == "KORX"
    assert SPRNAMES[SpriteNum.SPR_ABAT] == "ABAT"
    assert all(len(name) == 4 for name in SPRNAMES)


def test_states_samples() -> None:
    s_null = STATES[StateNum.S_NULL]
    assert s_null.sprite == SpriteNum.SPR_MAN1
    assert s_null.frame == 0
    assert s_null.tics == -1
    assert s_null.action_name is None
    assert s_null.next_state == StateNum.S_NULL

    s_free = STATES[StateNum.S_FREETARGMOBJ]
    assert s_free.sprite == SpriteNum.SPR_ACLO
    assert s_free.frame == 4
    assert s_free.tics == 1050
    assert s_free.action_name == "A_FreeTargMobj"
    assert s_free.next_state == StateNum.S_NULL

    s_korax_look = STATES[StateNum.S_KORAX_LOOK1]
    assert s_korax_look.sprite == SpriteNum.SPR_KORX
    assert s_korax_look.tics == 5
    assert s_korax_look.action_name == "A_Look"


def test_mobjinfo_samples() -> None:
    mapspot = MOBJINFO[MobjType.MT_MAPSPOT]
    assert mapspot.doomednum == 9001
    assert mapspot.spawnstate == StateNum.S_MAPSPOT
    assert mapspot.spawnhealth == 1000
    assert mapspot.radius == 20 * FRACUNIT
    assert mapspot.height == 16 * FRACUNIT
    assert mapspot.flags == (MobjFlag.MF_NOBLOCKMAP | MobjFlag.MF_NOSECTOR | MobjFlag.MF_NOGRAVITY)
    assert mapspot.flags2 == 0

    fireball = MOBJINFO[MobjType.MT_FIREBALL1]
    assert fireball.doomednum == -1
    assert fireball.spawnstate == StateNum.S_FIREBALL1_1
    assert fireball.deathstate == StateNum.S_FIREBALL1_X1
    assert fireball.deathsound == SfxEnum.SFX_FIREBALL
    assert fireball.speed == 2 * FRACUNIT
    assert fireball.radius == 8 * FRACUNIT
    assert fireball.height == 8 * FRACUNIT
    assert fireball.damage == 4
    assert fireball.flags2 == (MobjFlag2.MF2_NOTELEPORT | MobjFlag2.MF2_FIREDAMAGE)


def test_action_registry_and_stubs() -> None:
    assert get_action(None) is None
    assert get_action("NULL") is None

    stub = get_action("A_UnimplementedTestAction")
    assert stub is not None
    assert getattr(stub, "_is_stub", False) is True
    assert not is_action_implemented("A_UnimplementedTestAction")
    stub()  # Calling stub should not raise

    called_with: list[int] = []

    @action("A_TestCustomAction")
    def _custom_action(val: int) -> None:
        called_with.append(val)

    try:
        resolved = get_action("A_TestCustomAction")
        assert resolved is _custom_action
        assert getattr(resolved, "_is_stub", False) is False
        assert is_action_implemented("A_TestCustomAction")
        resolved(42)
        assert called_with == [42]
    finally:
        _ACTIONS.pop("A_TestCustomAction", None)


def test_action_report() -> None:
    report = get_action_report()
    assert len(report) == 226
    assert "A_Look" in report
    assert "A_Chase" in report
    assert "A_FreeTargMobj" in report

    script = Path(__file__).resolve().parents[2] / "tools" / "report_actions.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Total unique actions: {len(report)}" in proc.stdout
    assert "A_Look" in proc.stdout
    assert "STUB" in proc.stdout
