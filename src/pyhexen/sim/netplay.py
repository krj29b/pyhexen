# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Multi-player spawning and reborn rules (coop and deathmatch).

Ports G_CheckSpot, G_DeathMatchSpawnPlayer and the netgame branch of G_DoReborn
(g_game.c), plus the deathmatch part of P_SetupLevel (p_setup.c). None of this knows
about the network: a netgame sim just has several players and `netgame` set (S2a).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyhexen.core.angles import ANG45, ANGLETOFINESHIFT, fine_cosine, fine_sine
from pyhexen.core.fixed import FRACBITS, FRACUNIT, c_div
from pyhexen.sim.info_data import MobjFlag2, MobjType, SfxEnum
from pyhexen.sim.map import check_position
from pyhexen.sim.mobj import Mobj, remove_mobj, spawn_mobj
from pyhexen.sim.sound import start_sound

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import MapThing

TELEFOGHEIGHT: int = 32 * FRACUNIT
NUMWEAPONS: int = 4


def check_spot(sim: GameSim, player_num: int, mthing: MapThing) -> bool:
    """Port of G_CheckSpot (g_game.c): is the start free? Spawns teleport fog if so."""
    mo = sim.players[player_num].mo
    assert mo is not None and sim.level is not None
    x = mthing.x << FRACBITS
    y = mthing.y << FRACBITS
    mo.flags2 &= ~MobjFlag2.MF2_PASSMOBJ
    if not check_position(sim, mo, x, y):
        mo.flags2 |= MobjFlag2.MF2_PASSMOBJ
        return False
    mo.flags2 |= MobjFlag2.MF2_PASSMOBJ

    # spawn a teleport fog
    ss = sim.level.point_in_subsector(x, y)
    an = ((ANG45 * c_div(mthing.angle, 45)) & 0xFFFFFFFF) >> ANGLETOFINESHIFT
    fog = spawn_mobj(
        sim,
        x + 20 * fine_cosine(an),
        y + 20 * fine_sine(an),
        ss.sector.floor_height + TELEFOGHEIGHT,
        MobjType.MT_TFOG,
    )
    start_sound(sim, fog, int(SfxEnum.SFX_TELEPORT))
    return True


def deathmatch_spawn_player(sim: GameSim, player_num: int) -> None:
    """Port of G_DeathMatchSpawnPlayer (g_game.c): up to 20 tries at random starts."""
    from pyhexen.sim.player import spawn_player

    starts = sim.deathmatch_starts
    selections = len(starts)
    if selections:
        for _ in range(20):
            spot = starts[sim.rng.p_random() % selections]
            if check_spot(sim, player_num, spot):
                spawn_player(
                    sim,
                    player_num,
                    x=spot.x << FRACBITS,
                    y=spot.y << FRACBITS,
                    angle=ANG45 * c_div(spot.angle, 45),
                )
                return
    # no good spot, so the player will probably get stuck
    spawn_player(sim, player_num, 0)


def spawn_deathmatch_players(sim: GameSim) -> None:
    """Deathmatch part of P_SetupLevel: every player gets a random deathmatch start.

    G_CheckSpot needs a player mobj, so vanilla spawns a temporary Fighter at the
    player's coop start and removes it afterwards.
    """
    for num in range(len(sim.players)):
        start = sim.player_starts.get((num, 0))
        x = (start.x << FRACBITS) if start is not None else 0
        y = (start.y << FRACBITS) if start is not None else 0
        temp = spawn_mobj(sim, x, y, 0, MobjType.MT_PLAYER_FIGHTER)
        sim.players[num].mo = temp
        deathmatch_spawn_player(sim, num)
        remove_mobj(sim, temp)


def check_deathmatch_starts(sim: GameSim) -> None:
    """Port of the P_LoadThings check: enough deathmatch starts for every player."""
    if sim.deathmatch and len(sim.deathmatch_starts) < len(sim.players):
        raise RuntimeError(
            f"P_LoadThings: Player count ({len(sim.players)}) exceeds deathmatch "
            f"spots ({len(sim.deathmatch_starts)})"
        )


def do_reborn(sim: GameSim, player_num: int) -> None:
    """Port of the netgame branch of G_DoReborn (g_game.c).

    Deathmatch respawns at a random deathmatch start. Coop respawns at the player's
    start for the map entry position (or another free player start), keeping keys,
    weapon pieces and weapons, with 25 of each mana and the best weapon raised.
    """
    from pyhexen.sim.player import spawn_player

    player = sim.players[player_num]
    if player.mo is not None:
        player.mo.player = None  # Dissassociate the corpse
    if sim.deathmatch:
        deathmatch_spawn_player(sim, player_num)
        return

    # Cooperative net-play, retain keys and weapons
    old_keys = player.keys
    old_pieces = player.pieces
    old_weapons = list(player.weaponowned)
    position = sim.reborn_position

    found = False
    own = sim.player_starts.get((player_num, position))
    if own is not None and check_spot(sim, player_num, own):
        spawn_player(sim, player_num, position)
        found = True
    else:
        for i in range(len(sim.players)):
            other = sim.player_starts.get((i, position))
            if other is not None and check_spot(sim, player_num, other):
                # Fake as other player: the start's place, this player's colours
                spawn_player(
                    sim,
                    player_num,
                    x=other.x << FRACBITS,
                    y=other.y << FRACBITS,
                    angle=ANG45 * c_div(other.angle, 45),
                )
                found = True
                break
    if not found:  # Player's going to be inside something
        spawn_player(sim, player_num, position)

    # Restore keys and weapons
    player.keys = old_keys
    player.pieces = old_pieces
    best = 0
    for i in range(NUMWEAPONS):
        if old_weapons[i]:
            best = i
            player.weaponowned[i] = True
    player.mana[0] = 25
    player.mana[1] = 25
    if best:  # Bring up the best weapon
        player.pendingweapon = best


def player_mobjs(sim: GameSim) -> list[Mobj]:
    """Live player mobjs, in player order (for tests and the UI)."""
    return [p.mo for p in sim.players if p.mo is not None]


def player_quit(sim: GameSim, player_num: int) -> None:
    """Port of PlayerQuitGame (d_net.c): the player leaves; its body stays where it is.

    The message goes to the console player only, so it is the one piece of state that
    differs between nodes; messages never feed back into the game.
    """
    from pyhexen.sim.message import console_player, set_message

    sim.players[player_num].in_game = False
    viewer = console_player(sim)
    if viewer is not None and viewer.in_game:
        set_message(sim, viewer, f"PLAYER {player_num + 1} LEFT THE GAME", True)
    start_sound(sim, None, int(SfxEnum.SFX_CHAT))
