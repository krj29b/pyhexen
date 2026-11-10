# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T14.1 demo recording, playback, and determinism harness."""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.angles import ANG45
from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.ticcmd import AFLAG_JUMP, BT_ATTACK, BT_USE, TicCmd
from pyhexen.sim.demo import (
    DEMO_MAGIC,
    Demo,
    DemoError,
    DemoPlayer,
    DemoRecorder,
    DemoVersionError,
    pack_ticcmd,
    play_demo_to_completion,
    unpack_ticcmd,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import Mobj
from pyhexen.sim.player import spawn_player
from pyhexen.sim.specials.doors import DoorType, VerticalDoor
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _create_synthetic_demo_level() -> tuple[WadStack, Level]:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)

    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms()
    mb.add_thing(tid=1, x=128, y=128, z=0, angle=0, type=1)
    wb.add_map(mb)

    raw_wad = wb.to_bytes()
    stack = WadStack(raw_wad)
    level = load_level(stack, "MAP01")
    return stack, level


def test_ticcmd_packing_and_unpacking() -> None:
    """TicCmd bitfields, deltas, and buttons pack and unpack losslessly."""
    cmd = TicCmd(
        forwardmove=25,
        sidemove=-30,
        angleturn=1234,
        consistancy=5678,
        chatchar=65,
        buttons=BT_ATTACK | BT_USE,
        lookfly=0x12,
        arti=AFLAG_JUMP | 5,
    )

    packed = pack_ticcmd(cmd)
    assert len(packed) == 10

    unpacked = unpack_ticcmd(packed)
    assert unpacked.forwardmove == 25
    assert unpacked.sidemove == -30
    assert unpacked.angleturn == 1234
    assert unpacked.consistancy == 5678
    assert unpacked.chatchar == 65
    assert unpacked.buttons == BT_ATTACK | BT_USE
    assert unpacked.lookfly == 0x12
    assert unpacked.arti == AFLAG_JUMP | 5
    assert unpacked.is_jump
    assert unpacked.artifact_type == 5


def test_demo_recording_and_playback_determinism(tmp_path: Path) -> None:
    """A recorded demo replays deterministically with identical final state hash."""
    stack, _ = _create_synthetic_demo_level()

    def make_sim() -> GameSim:
        lvl = load_level(stack, "MAP01")
        sim = GameSim(level=lvl, skill=2)
        sim.map_num = 1
        spawn_player(sim, player_num=0)

        # Add a door in sector 1
        sec = lvl.sectors[1]
        sec.ceiling_height = sec.floor_height
        door = VerticalDoor(sec, DoorType.DREV_NORMAL, speed=2 * FRACUNIT, topwait=150)
        door.direction = 1
        sec.special_data = door
        sim.thinkers.add(door)

        # Add an active monster
        m = Mobj()
        m.x = 256 * FRACUNIT
        m.y = 256 * FRACUNIT
        m.health = 50
        sim.thinkers.add(m)
        return sim

    # 1. Record demo
    sim1 = make_sim()
    demo_file = tmp_path / "test_demo.pxd"
    recorder = DemoRecorder(sim1, file_path=demo_file)

    # Generate 40 tics of player actions
    for tic_i in range(40):
        cmd = TicCmd(
            forwardmove=20 if tic_i < 20 else 0,
            sidemove=-10 if 10 <= tic_i < 30 else 0,
            angleturn=((ANG45 >> 16) // 10) if tic_i < 10 else 0,
            buttons=BT_ATTACK if tic_i in (5, 15, 25) else 0,
            arti=AFLAG_JUMP if tic_i == 12 else 0,
        )
        recorded_cmds = recorder.record_tic([cmd])
        sim1.tick(recorded_cmds)

    recorded_demo = recorder.finish()
    assert demo_file.is_file()
    assert recorded_demo.header.total_tics == 40
    assert recorded_demo.header.final_state_hash == sim1.state_hash()

    # 2. Replay demo in fresh simulation from scratch
    sim2 = make_sim()
    # Reset PRNG state to match demo recording start
    sim2.rng.set_state(recorded_demo.header.rng_state)

    player = DemoPlayer.from_file(demo_file)
    assert player.total_tics == 40
    assert not player.is_finished

    success = play_demo_to_completion(sim2, player)
    assert success
    assert player.is_finished
    assert sim2.state_hash() == recorded_demo.header.final_state_hash
    assert player.is_finished
    assert sim2.state_hash() == recorded_demo.header.final_state_hash


def test_demo_determinism_divergence_detection() -> None:
    """Altered input during demo playback correctly causes determinism check to fail."""
    stack, _ = _create_synthetic_demo_level()

    lvl = load_level(stack, "MAP01")
    sim = GameSim(level=lvl)
    spawn_player(sim, player_num=0)

    recorder = DemoRecorder(sim)
    for _ in range(15):
        cmd = TicCmd(forwardmove=15, angleturn=100)
        recorder.record_tic([cmd])
        sim.tick([cmd])

    demo = recorder.finish()

    # Replay with altered command on tic 5
    lvl2 = load_level(stack, "MAP01")
    sim2 = GameSim(level=lvl2)
    spawn_player(sim2, player_num=0)
    sim2.rng.set_state(demo.header.rng_state)

    player = DemoPlayer(demo)
    for tic_i in range(demo.header.total_tics):
        cmds = player.get_next_cmds()
        if tic_i == 5:
            # Diverge by providing different turn delta
            altered_cmd = TicCmd(forwardmove=cmds[0].forwardmove, angleturn=cmds[0].angleturn + 500)
            sim2.tick([altered_cmd])
        else:
            sim2.tick(cmds)

    assert not player.check_determinism(sim2)


def test_demo_file_version_and_corruption_error(tmp_path: Path) -> None:
    """Incompatible demo version or corrupted magic triggers proper DemoError."""
    stack, _ = _create_synthetic_demo_level()
    lvl = load_level(stack, "MAP01")
    sim = GameSim(level=lvl)
    spawn_player(sim, player_num=0)

    demo_file = tmp_path / "corrupt.pxd"
    recorder = DemoRecorder(sim, file_path=demo_file)
    recorder.record_tic([TicCmd(forwardmove=10)])
    recorder.finish()

    # Corrupt version
    with gzip.open(demo_file, "rb") as f:
        raw = f.read()
    magic, hdr_len = struct.unpack("<4sI", raw[:8])
    hdr_dict = json.loads(raw[8 : 8 + hdr_len].decode("utf-8"))
    hdr_dict["version"] = 999
    new_hdr_bytes = json.dumps(hdr_dict).encode("utf-8")
    new_raw = (
        struct.pack("<4sI", DEMO_MAGIC, len(new_hdr_bytes)) + new_hdr_bytes + raw[8 + hdr_len :]
    )
    with gzip.open(demo_file, "wb") as f:
        f.write(new_raw)

    with pytest.raises(DemoVersionError, match="Incompatible demo version"):
        Demo.load_file(demo_file)

    # Corrupt magic
    new_raw_bad_magic = (
        struct.pack("<4sI", b"BADM", len(new_hdr_bytes)) + new_hdr_bytes + raw[8 + hdr_len :]
    )
    with gzip.open(demo_file, "wb") as f:
        f.write(new_raw_bad_magic)

    with pytest.raises(DemoError, match="Invalid demo magic"):
        Demo.load_file(demo_file)
