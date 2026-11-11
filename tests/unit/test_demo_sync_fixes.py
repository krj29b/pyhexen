# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sim fixes found by comparing demo traces with Chocolate Hexen (T15.6).

Each test pins one place where PyHexen differed from the reference source.
"""

from __future__ import annotations

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import FRACUNIT
from pyhexen.core.ticcmd import TicCmd
from pyhexen.sim.actions.environment import a_fogmove
from pyhexen.sim.enemy import DI_EAST, move
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MF_TRANSSHIFT, MOBJINFO, MobjType
from pyhexen.sim.map import try_move
from pyhexen.sim.mobj import Mobj, spawn_mobj
from pyhexen.sim.specials.anim import load_surface_animator
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level

TRANSLATION_MASK = 3 << MF_TRANSSHIFT


def _level(mb: MapBuilder, wb: WadBuilder | None = None) -> Level:
    wb = wb or WadBuilder("IWAD")
    return load_level(WadStack(wb.add_map(mb).to_bytes()), mb.name)


def _room(**kwargs: object) -> MapBuilder:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512, **kwargs)  # type: ignore[arg-type]
    return mb


def _mobjs(sim: GameSim) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and not t.removed]


def test_player_spawns_at_its_start_in_thing_order() -> None:
    """P_SpawnMapThing spawns the player when P_LoadThings reaches its start."""
    mb = _room()
    mb.add_thing(x=100, y=100, type=10030)  # ettin, before the start
    mb.add_thing(x=256, y=256, type=1)  # player 1 start
    mb.add_thing(x=400, y=400, type=10030)  # ettin, after the start
    sim = GameSim(level=_level(mb))
    sim.setup_level(player_nums=(0,))
    order = [m.type for m in _mobjs(sim)]
    assert order == [MobjType.MT_ETTIN, MobjType.MT_PLAYER_FIGHTER, MobjType.MT_ETTIN]


def test_player_translation_bits() -> None:
    """P_SpawnPlayer: the Fighter at start 1 is drawn blue (translation 2)."""
    mb = _room()
    mb.add_thing(x=256, y=256, type=1)
    fighter = GameSim(level=_level(mb), player_class=0)
    fighter.setup_level(player_nums=(0,))
    assert fighter.players[0].mo is not None
    assert fighter.players[0].mo.flags & TRANSLATION_MASK == 2 << MF_TRANSSHIFT

    cleric = GameSim(level=_level(mb), player_class=1)
    cleric.setup_level(player_nums=(0,))
    assert cleric.players[0].mo is not None
    assert cleric.players[0].mo.flags & TRANSLATION_MASK == 0


def test_float_bob_counter_is_not_masked() -> None:
    """P_MobjThinker: FloatBobOffsets[(mobj->health++) & 63] keeps the whole counter."""
    mb = _room()
    mb.add_thing(x=256, y=256, type=81)  # MT_HEALINGBOTTLE, MF2_FLOATBOB
    sim = GameSim(level=_level(mb))
    sim.setup_level()
    bottle = next(m for m in _mobjs(sim) if m.type == MobjType.MT_HEALINGBOTTLE)
    bottle.health = 63
    sim.tick([])
    sim.tick([])
    assert bottle.health == 65


def test_lynched_corpse_gets_a_blood_pool_first() -> None:
    mb = _room()
    mb.add_thing(x=256, y=256, type=109)  # MT_ZLYNCHED_NOHEART
    sim = GameSim(level=_level(mb))
    sim.setup_level()
    assert [m.type for m in _mobjs(sim)] == [MobjType.MT_BLOODPOOL, MobjType.MT_ZLYNCHED_NOHEART]


def test_spawn_in_liquid_sets_floorclip() -> None:
    """P_SpawnMobj clips MF2_FLOORCLIP things standing on a liquid floor."""
    sim = GameSim(level=_level(_room(floor_pic="X_005")))  # water
    ettin = spawn_mobj(sim, 256 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    assert ettin.floorclip == 10 * FRACUNIT
    dry = GameSim(level=_level(_room()))
    assert spawn_mobj(dry, 256 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN).floorclip == 0


def test_fog_move_tests_lifetime_before_decrementing() -> None:
    """A_FogMove: `if (actor->args[3]-- <= 0)`, then the death state without its action."""
    sim = GameSim(level=_level(_room()))
    fog = spawn_mobj(sim, 256 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_FOGPATCHM)
    fog.args = (1, 0, 0, 1, 1)
    a_fogmove(sim, fog)
    assert fog.args[3] == 0 and fog.state_num != MOBJINFO[fog.type].deathstate
    assert fog.momx == 65535  # FixedMul(1 << FRACBITS, finecosine[0] == 65535)
    a_fogmove(sim, fog)
    assert fog.state_num == MOBJINFO[fog.type].deathstate
    assert fog.args[3] == 255  # byte wrap


def test_blocked_move_without_specials_keeps_movedir() -> None:
    """P_Move only clears movedir when it goes on to try special lines."""
    sim = GameSim(level=_level(_room()))
    ettin = spawn_mobj(sim, 500 * FRACUNIT, 256 * FRACUNIT, 0, MobjType.MT_ETTIN)
    ettin.movedir = DI_EAST  # straight into the east wall
    assert not move(sim, ettin)
    assert ettin.movedir == DI_EAST


def test_cant_leave_floorpic_also_refuses_a_height_change() -> None:
    """P_TryMove: MF2_CANTLEAVEFLOORPIC things keep the same flat and floor height."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(floor1=0, floor2=16, floor_pic1="X_005", floor_pic2="X_005")
    sim = GameSim(level=_level(mb))
    serpent = spawn_mobj(sim, 230 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_SERPENT)
    assert not try_move(sim, serpent, 290 * FRACUNIT, 128 * FRACUNIT)  # same flat, 16 higher
    assert try_move(sim, serpent, 200 * FRACUNIT, 128 * FRACUNIT)  # same sector
    assert serpent.floorpic == "X_005"


def test_surface_animator_draws_random_tics_from_the_sim_rng() -> None:
    """P_InitFTAnims + P_AnimateSurfaces: every anim advances on the first tic."""
    wb = WadBuilder("IWAD")
    wb.add_marker("F_START").add_flat("FA1").add_flat("FA2").add_marker("F_END")
    wb.add_lump(
        "ANIMDEFS",
        b"flat FA1\n pic 1 tics 8\n pic 2 rand 2 6\nflat MISSING\n pic 1 tics 8\n pic 2 tics 8\n",
    )
    stack = WadStack(wb.add_map(_room()).to_bytes())
    animator = load_surface_animator(stack)
    assert [seq.base_name for seq in animator.sequences] == ["FA1"]  # missing flat skipped

    sim = GameSim(level=load_level(stack, "MAP01"))
    sim.surface_animator = animator
    sim.setup_level()
    before = sim.rng.prndindex
    sim.tick([TicCmd()])
    seq = animator.sequences[0]
    assert seq.current_frame == 0 and seq.tics == 8  # wrapped from the last frame
    for _ in range(8):
        sim.tick([])
    # Frame 2 ("rand 2 6") starts now and draws its length from P_Random.
    assert seq.current_frame == 1 and 2 <= seq.tics <= 6
    assert sim.rng.prndindex != before


def test_lightning_setup_draws_one_random_number() -> None:
    """P_InitLightning: NextLightningFlash = ((P_Random() & 15) + 5) * 35."""
    plain = GameSim(level=_level(_room(ceiling_pic="F_SKY")))
    plain.setup_level()
    lit = GameSim(level=_level(_room(ceiling_pic="F_SKY")))
    lit.setup_level(lightning=True)
    assert lit.rng.prndindex == plain.rng.prndindex + 1
    assert lit.surface_animator.level_has_lightning
    assert lit.surface_animator.next_lightning_flash % 35 == 0
