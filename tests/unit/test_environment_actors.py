# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for environment actors and actions (T10.11)."""

from __future__ import annotations

from fixtures.wadbuilder import MapBuilder, WadBuilder
from pyhexen.core.events import EventBus
from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.random import Random
from pyhexen.sim.actions.environment import (
    _PLAYER_BODY_QUEUE_SIZE,
    a_addplayercorpse,
    a_batmove,
    a_batspawn,
    a_batspawninit,
    a_bellreset1,
    a_bellreset2,
    a_bridgeinit,
    a_bridgeorbit,
    a_bridgeremove,
    a_corpseblooddrip,
    a_corpseexplode,
    a_esound,
    a_flamecheck,
    a_fogmove,
    a_fogspawn,
    a_leafcheck,
    a_leafspawn,
    a_leafthrust,
    a_nogravity,
    a_potterycheck,
    a_potterychoosebit,
    a_potteryexplode,
    a_quake,
    a_thrustimpale,
    a_thrustinitdn,
    a_thrustinitup,
    a_treedeath,
    p_init_player_body_queue,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import (
    MobjFlag,
    MobjFlag2,
    MobjType,
    StateNum,
)
from pyhexen.sim.mobj import Mobj, spawn_map_things, spawn_mobj
from pyhexen.sim.player import Player, PlayerClass
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _make_test_sim() -> GameSim:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_room(0, 0, 1024, 1024)
    mb.add_thing(tid=0, x=256, y=256, z=0, angle=0, type=1)
    wb.add_map(mb)
    stack = WadStack(wb.to_bytes())
    level = load_level(stack, "MAP01")
    sim = GameSim(level=level)
    sim.rng = Random(0x12345678)
    sim.events = EventBus()
    spawn_map_things(sim)
    return sim


def test_player_corpse_queue() -> None:
    p_init_player_body_queue()
    sim = _make_test_sim()

    corpses = []
    for i in range(_PLAYER_BODY_QUEUE_SIZE + 5):
        mo = spawn_mobj(sim, (i + 1) * 10 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
        a_addplayercorpse(mo, sim)
        corpses.append(mo)

    # First 5 corpses should have been removed
    for c in corpses[:5]:
        assert c not in sim.thinkers
    # The last 32 should still be in the thinker list
    for c in corpses[5:]:
        assert c in sim.thinkers


def test_pottery_explode_and_bits() -> None:
    sim = _make_test_sim()
    pot = spawn_mobj(sim, 256 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_POTTERY1)
    pot.args = (0, 0, 0, 0, 0)

    a_potteryexplode(pot, sim)
    assert pot not in sim.thinkers

    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    bits = [m for m in mobjs if m.type == MobjType.MT_POTTERYBIT1]
    assert len(bits) >= 3

    bit = bits[0]
    a_potterychoosebit(bit, sim)
    assert bit.tics >= 256


def test_pottery_check_sight() -> None:
    sim = _make_test_sim()
    pmo = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    pmo.angle = 0  # Facing east
    player = Player(mo=pmo, player_class=int(PlayerClass.PCLASS_FIGHTER))
    pmo.player = player
    sim.players = [player]

    # Shard placed directly in front of player (east)
    bit = spawn_mobj(sim, 200 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_POTTERYBIT1)
    bit.state_num = int(StateNum.S_POTTERYBIT_EX1_2)

    a_potterycheck(bit, sim)
    assert bit.state_num == int(StateNum.S_POTTERYBIT_EX1)


def test_corpse_blood_drip_and_explode() -> None:
    sim = _make_test_sim()
    corpse = spawn_mobj(sim, 200 * FRACUNIT, 200 * FRACUNIT, 50 * FRACUNIT, MobjType.MT_MISC77)

    # Repeated calls should spawn at least one blood drop
    for _ in range(5):
        a_corpseblooddrip(corpse, sim)
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    drips = [m for m in mobjs if m.type == MobjType.MT_CORPSEBLOODDRIP]
    assert len(drips) > 0

    # Explode corpse into bits
    a_corpseexplode(corpse, sim)
    assert corpse not in sim.thinkers
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    cbits = [m for m in mobjs if m.type == MobjType.MT_CORPSEBIT]
    assert len(cbits) >= 4  # 3+ bits plus skull


def test_leaves() -> None:
    sim = _make_test_sim()
    spawner = spawn_mobj(sim, 300 * FRACUNIT, 300 * FRACUNIT, 100 * FRACUNIT, MobjType.MT_MISC79)

    a_leafspawn(spawner, sim)
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    leaves = [m for m in mobjs if m.type in (MobjType.MT_LEAF1, MobjType.MT_LEAF2)]
    assert len(leaves) >= 1

    leaf = leaves[0]
    initial_momz = leaf.momz
    for _ in range(10):
        a_leafthrust(leaf, sim)
    # Momz should have been boosted by upward flutter
    assert leaf.momz >= initial_momz

    # Test leaf expiration check
    leaf.special1 = 19
    a_leafcheck(leaf, sim)
    assert leaf.state_num == int(StateNum.S_NULL)


def test_bridge_triad() -> None:
    sim = _make_test_sim()
    bridge = spawn_mobj(sim, 400 * FRACUNIT, 400 * FRACUNIT, 0, MobjType.MT_BRIDGE)

    a_bridgeinit(bridge, sim)
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    balls = [m for m in mobjs if m.type == MobjType.MT_BRIDGEBALL and m.target is bridge]
    assert len(balls) == 3

    # Orbit should advance angles and update coordinates
    prev_ang = balls[0].args[0]
    a_bridgeorbit(balls[0], sim)
    assert balls[0].args[0] == (prev_ang + 3) & 255

    # Bridge remove removes balls
    a_bridgeremove(bridge, sim)
    assert not (bridge.flags & MobjFlag.MF_SOLID)
    a_bridgeorbit(balls[0], sim)
    assert balls[0].state_num == int(StateNum.S_NULL)


def test_ambient_sound() -> None:
    sim = _make_test_sim()
    wind = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_SOUNDWIND)
    events_received = []
    from pyhexen.core.events import SoundEvent

    sim.events.subscribe(SoundEvent, events_received.append)
    a_esound(wind, sim)
    assert len(events_received) == 1


def test_fog_spawner_and_move() -> None:
    sim = _make_test_sim()
    spawner = spawn_mobj(sim, 200 * FRACUNIT, 200 * FRACUNIT, 0, MobjType.MT_FOGSPAWNER)
    spawner.args = (5, 10, 0, 10, 0)
    spawner.special1 = 0

    a_fogspawn(spawner, sim)
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    fog_patches = [
        m
        for m in mobjs
        if m.type in (MobjType.MT_FOGPATCHS, MobjType.MT_FOGPATCHM, MobjType.MT_FOGPATCHL)
    ]
    assert len(fog_patches) == 1

    patch = fog_patches[0]
    assert patch.args[4] == 1  # Moving
    a_fogmove(patch, sim)
    assert patch.momx != 0 or patch.momy != 0


def test_quake() -> None:
    sim = _make_test_sim()
    pmo = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)
    player = Player(mo=pmo, health=100, player_class=int(PlayerClass.PCLASS_FIGHTER))
    pmo.player = player
    sim.players = [player]

    quake_mo = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_QUAKE_FOCUS)
    quake_mo.args = (5, 10, 20, 20, 0)  # Richter 5, duration 10, radius 20

    a_quake(quake_mo, sim)
    assert getattr(player, "quake_intensity", 0) == 5

    # Expire quake
    quake_mo.args = (5, 1, 20, 20, 0)
    a_quake(quake_mo, sim)
    assert getattr(player, "quake_intensity", 0) == 0
    assert quake_mo.state_num == int(StateNum.S_NULL)


def test_thrust_spikes() -> None:
    sim = _make_test_sim()
    spike = spawn_mobj(sim, 300 * FRACUNIT, 300 * FRACUNIT, 0, MobjType.MT_THRUSTFLOOR_UP)
    victim = spawn_mobj(sim, 300 * FRACUNIT, 300 * FRACUNIT, 0, MobjType.MT_ETTIN)
    victim.flags |= MobjFlag.MF_SHOOTABLE

    a_thrustinitdn(spike, sim)
    assert spike.floorclip > 0
    assert not (spike.flags & MobjFlag.MF_SOLID)

    a_thrustinitup(spike, sim)
    assert spike.floorclip == 0
    assert bool(spike.flags & MobjFlag.MF_SOLID)

    # Impale
    a_thrustimpale(spike, sim)
    assert victim.health <= 0
    assert spike.args[1] == 1  # Marked bloody


def test_bells_flames_trees() -> None:
    sim = _make_test_sim()

    # Bell
    bell = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ZBELL)
    h0 = bell.height
    a_bellreset1(bell, sim)
    assert bell.height == h0 << 2
    assert bool(bell.flags & MobjFlag.MF_NOGRAVITY)
    a_bellreset2(bell, sim)
    assert bell.health == 5
    assert bool(bell.flags & MobjFlag.MF_SHOOTABLE)

    # Flame check
    flame = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_FIREBOMB)
    flame.args = (1, 0, 0, 0, 0)
    a_flamecheck(flame, sim)
    assert flame.args[0] == 0
    a_flamecheck(flame, sim)
    assert flame.state_num == int(StateNum.S_NULL)

    # Tree
    tree = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ZSHRUB1)
    tree.flags2 &= ~MobjFlag2.MF2_FIREDAMAGE
    a_treedeath(tree, sim)
    assert tree.health == 35

    tree_fire = spawn_mobj(sim, 100 * FRACUNIT, 100 * FRACUNIT, 0, MobjType.MT_ZSHRUB1)
    tree_fire.flags2 |= MobjFlag2.MF2_FIREDAMAGE
    a_treedeath(tree_fire, sim)
    assert tree_fire.state_num == int(tree_fire.info.meleestate)


def test_bats_and_nogravity() -> None:
    sim = _make_test_sim()

    spawner = spawn_mobj(sim, 200 * FRACUNIT, 200 * FRACUNIT, 0, MobjType.MT_BAT_SPAWNER)
    spawner.args = (1, 10, 0, 5, 10)
    a_batspawninit(spawner, sim)
    assert spawner.special1 == 0

    a_batspawn(spawner, sim)
    mobjs = [th for th in sim.thinkers if isinstance(th, Mobj)]
    bats = [m for m in mobjs if m.type == MobjType.MT_BAT]
    assert len(bats) == 1

    bat = bats[0]
    initial_special2 = bat.special2
    a_batmove(bat, sim)
    assert bat.special2 == initial_special2 - 2

    # A_NoGravity
    item = spawn_mobj(sim, 0, 0, 0, MobjType.MT_DIRT1)
    a_nogravity(item, sim)
    assert bool(item.flags & MobjFlag.MF_NOGRAVITY)
