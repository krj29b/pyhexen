# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T13.2 simulation serialization framework."""

from __future__ import annotations

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.sim.acs.loader import ACScriptDef, BehaviorData
from pyhexen.sim.acs.vm import ACSVM, RunningScript
from pyhexen.sim.artifacts import ArtiType, give_artifact
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.mobj import Mobj, spawn_map_things
from pyhexen.sim.player import PlayerClass, spawn_player
from pyhexen.sim.polyobj import ev_rotate_poly, get_polyobj
from pyhexen.sim.specials.ceilings import CeilingMove, CeilingType, add_active_ceiling
from pyhexen.sim.specials.doors import DoorType, VerticalDoor
from pyhexen.sim.specials.floors import FloorMove, FloorType, FloorWaggle
from pyhexen.sim.specials.lights import Light, LightType
from pyhexen.sim.specials.pillars import Pillar
from pyhexen.sim.specials.plats import Plat, PlatStatus, PlatType, add_active_plat
from pyhexen.sim.specials.scroll import ScrollThinker
from pyhexen.sim.specials.sector_specials import PhasedLightThinker
from pyhexen.sim.specials.switches import ButtonThinker, SwitchWhere
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level


def _build_test_sim_with_poly() -> tuple[Level, GameSim, WadStack]:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_polyobject_box(
        tag=1,
        anchor_x=-2000,
        anchor_y=-2000,
        spawn_x=128,
        spawn_y=128,
        width=64,
        height=64,
    )
    # Player start
    mb.add_thing(tid=1, x=256, y=256, z=0, angle=0, type=1)
    # Monster / target
    mb.add_thing(tid=2, x=384, y=384, z=0, angle=0, type=10030)  # Ettin
    wb.add_map(mb)
    raw_wad = wb.to_bytes()
    stack = WadStack(raw_wad)
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    spawn_map_things(sim)
    return level, sim, stack


def test_level_geometry_serialization() -> None:
    level, sim, _ = _build_test_sim_with_poly()
    sec = level.sectors[0]
    sec.floor_height = 42 * FRACUNIT
    sec.ceiling_height = 180 * FRACUNIT
    sec.light_level = 200
    sec.special = 15
    sec.tag = 99
    sec.floor_pic = "FLAT5"
    sec.ceiling_pic = "CEIL3"

    line = level.lines[0]
    line.flags = 0x1234
    line.special = 45
    line.args = (10, 20, 30, 0, 0)

    side = level.sides[0]
    side.texture_offset = 12 * FRACUNIT
    side.row_offset = 34 * FRACUNIT
    side.top_texture = "TOP1"
    side.mid_texture = "MID1"
    side.bottom_texture = "BOT1"

    data = sim.save_to_bytes()

    # Reset level to defaults
    sec.floor_height = 0
    sec.ceiling_height = 0
    sec.light_level = 0
    line.flags = 0
    line.special = 0
    side.texture_offset = 0
    side.row_offset = 0

    sim.load_from_bytes(data)

    assert sec.floor_height == 42 * FRACUNIT
    assert sec.ceiling_height == 180 * FRACUNIT
    assert sec.light_level == 200
    assert sec.special == 15
    assert sec.tag == 99
    assert sec.floor_pic == "FLAT5"
    assert sec.ceiling_pic == "CEIL3"

    assert line.flags == 0x1234
    assert line.special == 45
    assert line.args == (10, 20, 30, 0, 0)

    assert side.texture_offset == 12 * FRACUNIT
    assert side.row_offset == 34 * FRACUNIT
    assert side.top_texture == "TOP1"
    assert side.mid_texture == "MID1"
    assert side.bottom_texture == "BOT1"


def test_polyobj_serialization() -> None:
    level, sim, _ = _build_test_sim_with_poly()
    po = get_polyobj(level, 1)
    assert po is not None

    # Rotate polyobject
    ev_rotate_poly(sim, line=None, args=(1, 64, 64, 0, 0), direction=1, override=False)
    for _ in range(10):
        sim.tick()

    saved_angle = po.angle
    saved_v0 = (po.segs[0].v1.x, po.segs[0].v1.y)

    data = sim.save_to_bytes()

    # Create fresh level & sim
    level2, sim2, _ = _build_test_sim_with_poly()
    po2 = get_polyobj(level2, 1)
    assert po2 is not None
    assert po2.angle != saved_angle

    sim2.load_from_bytes(data)

    assert po2.angle == saved_angle
    assert (po2.segs[0].v1.x, po2.segs[0].v1.y) == saved_v0


def test_mobjs_and_players_serialization() -> None:
    level, sim, _ = _build_test_sim_with_poly()
    player = spawn_player(sim, player_num=0)
    player.player_class = PlayerClass.PCLASS_CLERIC
    player.health = 85
    player.armor_points = [10, 20, 30, 40]
    player.mana = [50, 75]
    player.keys = 0b0101
    for _ in range(3):
        give_artifact(player, ArtiType.ARTI_INVULNERABILITY)

    # Set up target / tracer cross-references
    m1 = Mobj()
    m1.x = 100 * FRACUNIT
    m1.y = 100 * FRACUNIT
    m1.tid = 50
    m2 = Mobj()
    m2.x = 200 * FRACUNIT
    m2.y = 200 * FRACUNIT
    m2.tid = 51
    m1.target = m2
    m2.tracer = m1

    sim.thinkers.add(m1)
    sim.thinkers.add(m2)
    sim.tid_map[50] = [m1]
    sim.tid_map[51] = [m2]

    data = sim.save_to_bytes()

    # Fresh sim
    level2, sim2, _ = _build_test_sim_with_poly()
    sim2.load_from_bytes(data)

    assert len(sim2.players) == 1
    p2 = sim2.players[0]
    assert p2.player_class == PlayerClass.PCLASS_CLERIC
    assert p2.health == 85
    assert p2.armor_points == [10, 20, 30, 40]
    assert p2.mana == [50, 75]
    assert p2.keys == 0b0101
    assert p2.inventory[0].type == ArtiType.ARTI_INVULNERABILITY
    assert p2.inventory[0].count == 3
    assert p2.mo is not None
    assert p2.mo.player is p2

    # Check restored target/tracer pointers
    restored_m1 = next(t for t in sim2.thinkers if isinstance(t, Mobj) and t.tid == 50)
    restored_m2 = next(t for t in sim2.thinkers if isinstance(t, Mobj) and t.tid == 51)
    assert restored_m1.target is restored_m2
    assert restored_m2.tracer is restored_m1


def test_special_movers_serialization() -> None:
    level, sim, _ = _build_test_sim_with_poly()
    sec = level.sectors[0]

    # Add door
    door = VerticalDoor(sec, DoorType.DREV_NORMAL, speed=2 * FRACUNIT, topwait=150)
    door.direction = 1
    door.topcountdown = 75
    sec.special_data = door
    sim.thinkers.add(door)

    # Add floor move
    fm = FloorMove(
        sec,
        speed=FRACUNIT,
        dest=64 * FRACUNIT,
        direction=1,
        crush=10,
        floor_type=FloorType.FLEV_RAISEBUILDSTEP,
    )
    fm.reset_height, fm.reset_delay, fm.reset_delay_count = 8 * FRACUNIT, 30, 12
    fm.delay_count, fm.delay_total = 3, 7
    fm.stairs_delay_height, fm.stairs_delay_height_delta = 16 * FRACUNIT, 8 * FRACUNIT
    sim.thinkers.add(fm)

    # Add ceiling
    cm = CeilingMove(sec, CeilingType.CLEV_CRUSHANDRAISE, speed=2 * FRACUNIT)
    cm.topheight, cm.bottomheight, cm.direction, cm.crush = 128 * FRACUNIT, 8 * FRACUNIT, -1, 5
    add_active_ceiling(sim, cm)
    sim.thinkers.add(cm)

    # Add floor waggle
    fw = FloorWaggle(sec, original_height=0, acc_delta=8 << 10, target_scale=16 << 10)
    fw.accumulator = 1234
    fw.state = 2
    sim.thinkers.add(fw)

    # Add plat
    plat = Plat(sec, PlatType.PLAT_PERPETUALRAISE, speed=3 * FRACUNIT, tag=7)
    plat.low, plat.high, plat.wait = 0, 128 * FRACUNIT, 105
    plat.count = 42
    plat.status = PlatStatus.PLAT_WAITING
    add_active_plat(sim, plat)
    sim.thinkers.add(plat)

    # Add pillar
    pil = Pillar(
        sec,
        floor_speed=2 * FRACUNIT,
        ceiling_speed=FRACUNIT,
        floordest=32 * FRACUNIT,
        ceilingdest=64 * FRACUNIT,
        direction=-1,
        crush=3,
    )
    sim.thinkers.add(pil)

    # Add lights
    lf = Light(sec, LightType.LITE_FADE, value1=120, value2=-74898, tics2=-1)
    sim.thinkers.add(lf)

    lg = Light(sec, LightType.LITE_STROBE, value1=220, value2=32, tics1=4, tics2=8, count=3)
    sim.thinkers.add(lg)

    plt = PhasedLightThinker(sec, base=64, index=3)
    sim.thinkers.add(plt)

    # Add button and scroll
    line = level.lines[0]
    btn = ButtonThinker(line, SwitchWhere.SWTCH_TOP, btexture="SW1ON", btimer=70, sound_id=12)
    sim.thinkers.add(btn)

    scr = ScrollThinker(line, line.front_side, dx=2 * FRACUNIT, dy=0)
    sim.thinkers.add(scr)

    data = sim.save_to_bytes()

    level2, sim2, _ = _build_test_sim_with_poly()
    sim2.load_from_bytes(data)

    # Check thinkers restored in order
    types = [type(t).__name__ for t in sim2.thinkers]
    assert "VerticalDoor" in types
    assert "FloorMove" in types
    assert "FloorWaggle" in types
    assert "Plat" in types
    assert "Pillar" in types
    lights = [t.state_tuple() for t in sim2.thinkers if isinstance(t, Light)]
    assert lights == [lf.state_tuple(), lg.state_tuple()]
    assert "PhasedLightThinker" in types
    assert "ButtonThinker" in types
    assert "ScrollThinker" in types

    restored_door = next(t for t in sim2.thinkers if isinstance(t, VerticalDoor))
    assert restored_door.topcountdown == 75
    assert restored_door.direction == 1
    assert restored_door.sector is sim2.level.sectors[0]

    restored_fw = next(t for t in sim2.thinkers if isinstance(t, FloorWaggle))
    assert restored_fw.accumulator == 1234
    assert (restored_fw.state, restored_fw.target_scale) == (2, 16 << 10)

    restored_plat = next(t for t in sim2.thinkers if isinstance(t, Plat))
    assert restored_plat.count == 42

    # Every mover field survives, and plats and ceilings are back in the active lists.
    for original in (fm, cm, plat, pil):
        restored = next(t for t in sim2.thinkers if type(t) is type(original))
        assert restored.state_tuple() == original.state_tuple()
    assert sim2.active_plats[0] is restored_plat and sim2.active_plats[1] is None
    assert isinstance(sim2.active_ceilings[0], CeilingMove) and sim2.active_ceilings[1] is None

    restored_btn = next(t for t in sim2.thinkers if isinstance(t, ButtonThinker))
    assert restored_btn.btexture == "SW1ON"
    assert restored_btn.btimer == 70
    assert restored_btn.line is sim2.level.lines[0]


def test_acs_and_rng_serialization() -> None:
    level, sim, _ = _build_test_sim_with_poly()
    sdef = ACScriptDef(number=1, offset=0, arg_count=0)
    beh = BehaviorData(
        code=b"\x00" * 32,
        info_offset=0,
        scripts=(sdef,),
        strings=(),
        scripts_by_number={1: sdef},
    )
    sim.acs_vm = ACSVM(beh)
    sim.acs_vm.map_vars[5] = 999
    sim.acs_vm.world_vars[2] = 42

    # Queued script
    sim.acs_store.queue_script(2, 10, (1, 2, 3))

    # Running script
    rs = RunningScript(
        number=1,
        def_=sdef,
        ip=24,
        stack=[100, 200],
        vars=[5, 6, 7],
        delay_count=18,
    )
    sim.acs_vm.scripts[1] = rs
    sim.thinkers.add(rs)

    # Step RNG
    for _ in range(13):
        sim.rng.p_random()
    saved_rng = sim.rng.get_state()
    sim.level_time = 140

    data = sim.save_to_bytes()

    level2, sim2, _ = _build_test_sim_with_poly()
    sim2.acs_vm = ACSVM(beh)
    sim2.load_from_bytes(data)

    assert sim2.level_time == 140
    assert sim2.rng.get_state() == saved_rng
    assert sim2.acs_vm.map_vars[5] == 999
    assert sim2.acs_vm.world_vars[2] == 42
    assert len(sim2.acs_store.entries) == 1
    assert sim2.acs_store.entries[0].map_num == 2
    assert sim2.acs_store.entries[0].script_num == 10
    assert sim2.acs_store.entries[0].args == (1, 2, 3)

    restored_rs = sim2.acs_vm.scripts.get(1)
    assert restored_rs is not None
    assert restored_rs.ip == 24
    assert restored_rs.stack == [100, 200]
    assert restored_rs.vars == [5, 6, 7]
    assert restored_rs.delay_count == 18


def test_roundtrip_state_hash_and_determinism() -> None:
    level, sim1, _ = _build_test_sim_with_poly()
    spawn_player(sim1, player_num=0)

    # Tick for 20 tics
    for _ in range(20):
        sim1.tick()

    hash_before = sim1.state_hash()
    data = sim1.save_to_bytes()

    # Create fresh sim with same initial level
    level2, sim2, _ = _build_test_sim_with_poly()
    sim2.load_from_bytes(data)

    # State hash must match immediately upon restoration
    assert sim2.state_hash() == hash_before

    # Now tick both sims in parallel for 35 tics; state hashes must stay identical
    for _ in range(35):
        sim1.tick()
        sim2.tick()
        assert sim1.state_hash() == sim2.state_hash()


def test_movers_from_saves_before_t15_8_still_load() -> None:
    """Older saves have fewer mover fields; the movers continue with sensible defaults."""
    from pyhexen.sim.serialization import _restore_special_thinker

    _, sim, _ = _build_test_sim_with_poly()
    old_entries = [
        {"type": "FloorMove", "sector_index": 0, "speed": FRACUNIT, "dest": 64 * FRACUNIT,
         "direction": 1, "crush": 0},
        {"type": "CeilingMove", "sector_index": 0, "speed": FRACUNIT, "topheight": 128 * FRACUNIT,
         "bottomheight": 8 * FRACUNIT, "direction": -1, "crush": 1, "repeat": True},
        {"type": "Plat", "sector_index": 0, "speed": FRACUNIT, "low": 0, "high": 64 * FRACUNIT,
         "direction": 0, "wait": 105, "count": 42},
        {"type": "Pillar", "sector_index": 0, "speed": 2 * FRACUNIT, "floordest": 32 * FRACUNIT,
         "ceilingdest": 32 * FRACUNIT, "crush": 0, "floor_done": False, "ceiling_done": False},
    ]  # fmt: skip
    floor, ceiling, plat, pillar = (
        _restore_special_thinker(sim, entry["type"], entry) for entry in old_entries
    )
    assert (floor.speed, floor.floordestheight, floor.direction) == (FRACUNIT, 64 * FRACUNIT, 1)
    assert floor.reset_delay == 0 and floor.delay_count == 0
    assert ceiling.type == CeilingType.CLEV_CRUSHANDRAISE  # was `repeat`
    assert (ceiling.direction, ceiling.bottomheight) == (-1, 8 * FRACUNIT)
    assert sim.active_ceilings[0] is ceiling
    assert (plat.status, plat.count, plat.high) == (PlatStatus.PLAT_WAITING, 42, 64 * FRACUNIT)
    assert sim.active_plats[0] is plat
    assert (pillar.floor_speed, pillar.ceiling_speed, pillar.direction) == (
        2 * FRACUNIT,
        2 * FRACUNIT,
        1,
    )
