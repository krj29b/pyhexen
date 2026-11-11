# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Empirical verification suite for Milestone 2 (T9.2 Polyobject Movement & Physical Interaction).

Empirically tests polyobject movement, specials activation, physical interactions,
and ACS PCD_POLYWAIT/PCD_POLYWAITDIRECT against real commercial Hexen IWAD maps
(MAP01, MAP02, MAP05, MAP08, MAP27).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhexen.core.angles import ANG90
from pyhexen.core.events import SequenceEvent
from pyhexen.sim.acs.vm import ACScriptState
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjFlag, MobjType
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import SPAC_USE, load_level


@pytest.mark.wad
def test_iwad_map01_opening_double_swinging_doors(iwad_path: Path) -> None:
    """Verify MAP01 opening double swinging doors with mirror linkage (tags 4 & 5)."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")

    sim = GameSim(level=level)
    po4 = level.polyobjs[4]
    po5 = level.polyobjs[5]

    # Verify initial geometry and linkage
    assert po4.mirror_tag == 5
    assert po4.angle == 0
    assert po5.mirror_tag == 0
    assert po5.angle == 0
    assert po4.busy is False
    assert po5.busy is False

    # Collect initial vertex positions for polyobj 4 and 5
    po4_v_init = [(v.x, v.y) for seg in po4.segs for v in (seg.v1, seg.v2)]
    po5_v_init = [(v.x, v.y) for seg in po5.segs for v in (seg.v1, seg.v2)]

    # Listen for sound events on event bus
    sound_events: list[SequenceEvent] = []
    sim.events.subscribe(SequenceEvent, sound_events.append)

    # Line 558 has special 80 (ACS_Execute Script 2) with SPAC_USE
    line_558 = level.lines[558]
    assert line_558.special == 80
    assert line_558.args[0] == 2

    # Activate the line via player use
    activated = sim.on_line_activated(line_558, None, SPAC_USE)
    assert activated is True

    # Advance 1 tic: thinkers should be spawned and running
    sim.tick()
    assert po4.busy is True
    assert po5.busy is True

    # Check sound sequence started
    assert any(ev.stop is False for ev in sound_events)

    # Advance simulation until doors finish opening (takes exactly 64 tics)
    for _ in range(2, 70):
        sim.tick()
        if not po4.busy and not po5.busy:
            break

    assert po4.busy is False
    assert po5.busy is False

    # Poly 4 rotates clockwise (right) 90 degrees: -ANG90
    expected_po4_ang = (-ANG90) & 0xFFFFFFFF
    # Poly 5 mirrors and rotates counter-clockwise (left) 90 degrees: +ANG90
    expected_po5_ang = ANG90 & 0xFFFFFFFF

    assert po4.angle == expected_po4_ang
    assert po5.angle == expected_po5_ang

    # Vertex positions must have rotated and changed
    po4_v_final = [(v.x, v.y) for seg in po4.segs for v in (seg.v1, seg.v2)]
    po5_v_final = [(v.x, v.y) for seg in po5.segs for v in (seg.v1, seg.v2)]
    assert po4_v_final != po4_v_init
    assert po5_v_final != po5_v_init

    # Stop sound events must have been published
    assert any(ev.stop is True for ev in sound_events)


@pytest.mark.wad
def test_iwad_map01_sliding_pillars(iwad_path: Path) -> None:
    """Verify MAP01 puzzle room sliding pillars (tags 8..15) driven by Script 4."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")

    sim = GameSim(level=level)
    pillars = [level.polyobjs[t] for t in (8, 9, 10, 11, 12, 13, 14, 15)]
    init_spots = [(p.start_spot.x, p.start_spot.y) for p in pillars]

    assert sim.acs_vm is not None
    sim.acs_vm.start_script(4, sim=sim)

    # Run for 150 tics so all staggered sliding thinkers complete
    for _ in range(150):
        sim.tick()

    for p in pillars:
        assert p.busy is False, f"Pillar {p.tag} still busy"

    # Primary pillars 8, 10, 12, 14 moved South (dy = -64 units)
    # Mirrored pillars 9, 11, 13, 15 moved North (dy = +64 units)
    for p, (orig_x, orig_y) in zip(pillars, init_spots, strict=False):
        dx = (p.start_spot.x - orig_x) >> 16
        dy = (p.start_spot.y - orig_y) >> 16
        assert dx == 0
        if p.tag in (8, 10, 12, 14):
            assert dy == -64, f"Pillar {p.tag} dy={dy} expected -64"
        else:
            assert dy == 64, f"Pillar {p.tag} dy={dy} expected +64"


@pytest.mark.wad
def test_iwad_map02_rotating_pillars_stagger(iwad_path: Path) -> None:
    """Verify MAP02 rotating crushing pillars (tags 26, 27, 28) with 25-tic stagger in Script 2."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP02")

    sim = GameSim(level=level)
    p26 = level.polyobjs[26]
    p27 = level.polyobjs[27]
    p28 = level.polyobjs[28]

    assert sim.acs_vm is not None
    sim.acs_vm.start_script(2, sim=sim)

    # Step through simulation: each pillar takes 128 tics to rotate 90 degrees
    # p26: tics 0..128
    # p27: tics 25..153
    # p28: tics 50..178
    for _ in range(190):
        sim.tick()

    assert p26.busy is False
    assert p27.busy is False
    assert p28.busy is False

    expected_ang = (-ANG90) & 0xFFFFFFFF
    assert p26.angle == expected_ang
    assert p27.angle == expected_ang
    assert p28.angle == expected_ang


@pytest.mark.wad
def test_iwad_map05_sliding_door_cycle(iwad_path: Path) -> None:
    """Verify MAP05 sliding door (Line 1510, special 8) complete open-wait-close cycle."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP05")

    sim = GameSim(level=level)
    po19 = level.polyobjs[19]
    init_x = po19.start_spot.x
    init_y = po19.start_spot.y

    line_1510 = level.lines[1510]
    assert line_1510.special == 8
    # args: (19, speed=32, angle=192, dist=64, wait=70)

    sim.on_line_activated(line_1510, None, SPAC_USE)
    assert po19.busy is True

    # Advance 16 tics: door should be fully open at dy = -64
    for _ in range(16):
        sim.tick()

    assert ((po19.start_spot.y - init_y) >> 16) == -64

    # Advance 70 wait tics + 16 close tics + buffer
    for _ in range(95):
        sim.tick()

    assert po19.busy is False
    # Door returned to exact starting coordinates
    assert po19.start_spot.x == init_x
    assert po19.start_spot.y == init_y


@pytest.mark.wad
def test_iwad_acs_pcd_polywaitdirect_map08(iwad_path: Path) -> None:
    """Verify PCD_POLYWAITDIRECT pausing and resuming script execution in MAP08 Script 1."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP08")

    sim = GameSim(level=level)
    po1 = level.polyobjs[1]

    assert sim.acs_vm is not None
    sc = sim.acs_vm.start_script(1, sim=sim)
    assert sc is not None

    # Tic 1: Script executes LSPEC3DIRECT (rotate po1) and hits POLYWAITDIRECT tag=1
    sim.tick()
    assert po1.busy is True
    assert sc.state == ACScriptState.ASTE_WAITINGFORPOLY
    assert sc.wait_value == 1

    # Advance tics until polyobj 1 completes its 180-degree rotation (256 tics)
    for _ in range(2, 260):
        sim.tick()
        if not po1.busy:
            break

    assert po1.busy is False
    assert po1.angle == 0x80000000

    # Advance 1 tic: VM wakes script, executes TERMINATE, script becomes inactive/removed
    sim.tick()
    assert sc.state == ACScriptState.ASTE_INACTIVE
    assert sc.removed is True


@pytest.mark.wad
def test_iwad_acs_pcd_polywait_stack_map27(iwad_path: Path) -> None:
    """Verify PCD_POLYWAIT (tag popped from stack) in real MAP27 Script 6."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP27")

    sim = GameSim(level=level)
    po8 = level.polyobjs[8]

    assert sim.acs_vm is not None
    sc = sim.acs_vm.start_script(6, sim=sim)
    assert sc is not None

    # Step through initial 105-tic delay
    for _ in range(106):
        sim.tick()

    # Next tic: script rotates polyobjs 8..11 and hits PCD_POLYWAIT for tag 8
    sim.tick()
    assert po8.busy is True
    assert sc.state == ACScriptState.ASTE_WAITINGFORPOLY
    assert sc.wait_value == 8

    # Advance until po8 completes its rotation (256 tics)
    for _ in range(260):
        sim.tick()
        if not po8.busy:
            break

    assert po8.busy is False
    assert po8.angle == 0x80000000


@pytest.mark.wad
def test_iwad_crush_damage_against_mobj(iwad_path: Path) -> None:
    """Verify pinned mobjs take 3 crush damage per tic from crushing polyobjects (MAP01 tag 4)."""
    with WadStack(iwad_path) as wad:
        level = load_level(wad, "MAP01")

    sim = GameSim(level=level)
    po4 = level.polyobjs[4]
    assert po4.crush is True

    # Place an actor right inside polyobj 4's sweep path
    from pyhexen.sim.mobj import spawn_mobj

    mobj = spawn_mobj(sim, po4.start_spot.x, po4.start_spot.y, 0, MobjType.MT_CENTAUR)
    mobj.flags |= MobjFlag.MF_SOLID | MobjFlag.MF_SHOOTABLE
    mobj.health = 100

    # Start opening door (script 2)
    assert sim.acs_vm is not None
    sim.acs_vm.start_script(2, sim=sim)

    # Advance a few tics: mobj should take 3 damage per collision step
    initial_health = mobj.health
    for _ in range(5):
        sim.tick()

    assert mobj.health < initial_health, "Mobj did not take crushing damage from crushing polyobj"
