# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""S2a: multi-player sim rules (coop and deathmatch) with scripted ticcmd streams."""

from __future__ import annotations

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.events import MapExit
from pyhexen.core.fixed import FRACBITS
from pyhexen.core.ticcmd import BT_ATTACK, BT_USE, TicCmd
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MF_TRANSSHIFT, MOBJINFO, MobjType
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.message import console_player
from pyhexen.sim.mobj import (
    MTF_CLERIC,
    MTF_EASY,
    MTF_FIGHTER,
    MTF_GCOOP,
    MTF_GDEATHMATCH,
    MTF_GSINGLE,
    MTF_HARD,
    MTF_MAGE,
    MTF_NORMAL,
    Mobj,
)
from pyhexen.sim.player import PlayerState
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Level, load_level

ALL_SKILLS = MTF_EASY | MTF_NORMAL | MTF_HARD
ALL_MODES = MTF_GSINGLE | MTF_GCOOP | MTF_GDEATHMATCH
ALL_CLASSES = MTF_FIGHTER | MTF_CLERIC | MTF_MAGE
COOP_STARTS = [(100, 100), (200, 100), (300, 100), (400, 100)]
DM_STARTS = [(100, 400), (200, 400), (300, 400), (400, 400)]
MAGE_ONLY_ITEM = MOBJINFO[MobjType.MT_MW_CONE].doomednum  # Frost Shards


def _level(dm_starts: int = 4) -> Level:
    wb = WadBuilder("IWAD")
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    for i, (x, y) in enumerate(COOP_STARTS):
        mb.add_thing(x=x, y=y, type=i + 1, angle=90)
    for x, y in DM_STARTS[:dm_starts]:
        mb.add_thing(x=x, y=y, type=11, angle=270)
    mb.add_thing(x=256, y=256, type=MAGE_ONLY_ITEM, flags=ALL_SKILLS | ALL_MODES | MTF_MAGE)
    return load_level(WadStack(wb.add_map(mb).to_bytes()), "MAP01")


def _sim(
    classes: list[int], *, deathmatch: bool = False, dm_starts: int = 4, timer: int = 0
) -> GameSim:
    sim = GameSim(
        level=_level(dm_starts),
        netgame=True,
        deathmatch=deathmatch,
        player_classes=classes,
    )
    sim.setup_level(player_nums=range(len(classes)), timer_minutes=timer)
    return sim


def _mobjs_of(sim: GameSim, mtype: MobjType) -> list[Mobj]:
    return [t for t in sim.thinkers if isinstance(t, Mobj) and not t.removed and t.type == mtype]


def _pos(mo: Mobj) -> tuple[int, int]:
    return mo.x >> FRACBITS, mo.y >> FRACBITS


def test_coop_players_spawn_at_their_starts_with_their_class_and_colour() -> None:
    sim = _sim([0, 1, 2])
    assert [p.player_class for p in sim.players] == [0, 1, 2]
    types = [p.mo.type for p in sim.players if p.mo is not None]
    assert types == [MobjType.MT_PLAYER_FIGHTER, MobjType.MT_PLAYER_CLERIC, MobjType.MT_PLAYER_MAGE]
    assert [_pos(p.mo) for p in sim.players if p.mo] == COOP_STARTS[:3]
    translations = [(p.mo.flags >> MF_TRANSSHIFT) & 7 for p in sim.players if p.mo]
    assert translations == [2, 1, 2]  # Fighter in slot 1 is blue; others use slot - 1
    assert all(p.keys == 0 for p in sim.players)


def test_coop_spawns_things_for_any_class_in_the_game() -> None:
    assert len(_mobjs_of(_sim([0, 1]), MobjType.MT_MW_CONE)) == 0  # no Mage playing
    assert len(_mobjs_of(_sim([0, 2]), MobjType.MT_MW_CONE)) == 1
    assert len(_mobjs_of(_sim([0, 1], deathmatch=True), MobjType.MT_MW_CONE)) == 1


def test_deathmatch_players_spawn_at_random_deathmatch_starts_with_all_keys() -> None:
    sim = _sim([0, 0, 1, 2], deathmatch=True)
    positions = [_pos(p.mo) for p in sim.players if p.mo is not None]
    assert len(positions) == 4
    assert all(pos in DM_STARTS for pos in positions)
    assert len(set(positions)) == 4  # G_CheckSpot refuses an occupied start
    assert all(p.keys == 2047 for p in sim.players)
    # the temporary Fighters used for G_CheckSpot are gone
    assert (
        len(
            [
                t
                for t in sim.thinkers
                if isinstance(t, Mobj)
                and t.player is None
                and t.type == MobjType.MT_PLAYER_FIGHTER
                and not t.removed
            ]
        )
        == 0
    )


def test_deathmatch_needs_a_start_per_player() -> None:
    with pytest.raises(RuntimeError, match="exceeds deathmatch spots"):
        _sim([0, 1, 2], deathmatch=True, dm_starts=2)


def test_frags_count_kills_by_victim_number() -> None:
    sim = _sim([0, 1, 2], deathmatch=True)
    p0, p1, p2 = sim.players
    damage_mobj(sim, p0.mo, p1.mo, p1.mo, 10000)  # player 1 kills player 0
    assert p1.frags[0] == 1
    damage_mobj(sim, p2.mo, p2.mo, p2.mo, 10000)  # player 2 kills itself
    assert p2.frags[2] == -1
    damage_mobj(sim, p1.mo, None, None, 10000)  # no source: counts as a self-frag
    assert p1.frags[1] == -1
    assert p0.frags == [0] * len(p0.frags)


def _die_and_press_use(sim: GameSim, num: int) -> Mobj:
    player = sim.players[num]
    corpse = player.mo
    assert corpse is not None
    damage_mobj(sim, corpse, None, None, 10000)
    assert player.player_state == PlayerState.PST_DEAD
    cmds = [TicCmd() for _ in sim.players]
    for _ in range(40):
        sim.tick(cmds)
    cmds[num] = TicCmd(buttons=BT_USE)
    sim.tick(cmds)  # P_DeathThink: PST_REBORN
    sim.tick([TicCmd() for _ in sim.players])  # G_Ticker reborn before players think
    return corpse


def test_coop_reborn_keeps_keys_and_weapons_and_respawns_at_the_start() -> None:
    sim = _sim([0, 1])
    player = sim.players[1]
    player.weaponowned = [True, True, False, True]
    player.keys = 5
    player.mana = [150, 150]
    player.frags[0] = 3
    corpse = _die_and_press_use(sim, 1)
    assert player.player_state == PlayerState.PST_LIVE and player.health == 100
    assert player.mo is not corpse and corpse.player is None  # corpse disassociated
    assert _pos(player.mo) == COOP_STARTS[1]
    assert player.weaponowned == [True, True, False, True]
    assert player.keys == 5 and player.mana == [25, 25]
    assert player.pendingweapon == 3  # best weapon raised
    assert player.frags[0] == 3  # G_PlayerReborn keeps frags


def test_deathmatch_reborn_resets_the_player_at_a_deathmatch_start() -> None:
    sim = _sim([0, 1], deathmatch=True)
    player = sim.players[0]
    player.weaponowned = [True, True, True, False]
    _die_and_press_use(sim, 0)
    assert player.player_state == PlayerState.PST_LIVE
    assert player.weaponowned == [True, False, False, False]
    assert player.keys == 2047
    assert _pos(player.mo) in DM_STARTS


def test_deathmatch_timer_exits_the_map() -> None:
    sim = _sim([0, 1], deathmatch=True, timer=1)
    assert sim.timer_game == 35 * 60
    exits = []
    for _ in range(35 * 60):
        sim.tick([TicCmd(), TicCmd()])
        exits.extend(e for e in sim.events.drain() if isinstance(e, MapExit))
    assert len(exits) == 1


def test_same_ticcmd_streams_give_the_same_state() -> None:
    def run() -> str:
        sim = _sim([0, 1, 2], deathmatch=True)
        for tic in range(300):
            cmds = [
                TicCmd(
                    forwardmove=25 if (tic // 30 + n) % 2 else -25,
                    angleturn=(n + 1) * 300 if tic % 50 < 10 else 0,
                    buttons=BT_ATTACK if (tic + n * 7) % 20 < 5 else 0,
                )
                for n in range(3)
            ]
            sim.tick(cmds)
        return sim.state_hash()

    assert run() == run()


def test_console_player_is_per_node() -> None:
    sim = _sim([0, 1, 2])
    assert console_player(sim) is sim.players[0]
    sim.consoleplayer = 2
    assert console_player(sim) is sim.players[2]


def test_a_player_who_leaves_stops_thinking_and_is_not_targeted() -> None:
    from pyhexen.sim.enemy import look_for_players
    from pyhexen.sim.netplay import player_quit

    sim = _sim([0, 1])
    sim.consoleplayer = 0
    gone = sim.players[1]
    body = gone.mo
    player_quit(sim, 1)
    assert not gone.in_game
    assert "PLAYER 2 LEFT THE GAME" in sim.players[0].message
    x = body.x
    for _ in range(10):
        sim.tick([TicCmd(), TicCmd(forwardmove=50)])
    assert body.x == x  # no P_PlayerThink for a player not in the game
    ettin = spawn_ettin(sim, body.x + (64 << FRACBITS), body.y)
    sim.players[0].mo.x += 1 << 30  # move player 0 far out of sight
    for start in range(8):
        ettin.lastlook = start
        ettin.target = None
        look_for_players(sim, ettin, True)
        assert ettin.target is not body


def spawn_ettin(sim: GameSim, x: int, y: int) -> Mobj:
    from pyhexen.sim.mobj import spawn_mobj

    return spawn_mobj(sim, x, y, 0, MobjType.MT_ETTIN)


def test_banishment_sends_monsters_to_deathmatch_starts_and_players_to_player_starts() -> None:
    from pyhexen.sim.artifacts import teleport_other

    sim = _sim([0, 1])
    ettin = spawn_ettin(sim, 256 << FRACBITS, 300 << FRACBITS)
    teleport_other(sim, ettin)
    assert _pos(ettin) in DM_STARTS
    player = sim.players[1]
    teleport_other(sim, player.mo)
    assert _pos(player.mo) in COOP_STARTS[:2]
