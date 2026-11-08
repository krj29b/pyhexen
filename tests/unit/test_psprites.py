# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T10.6: Weapon framework & psprites (sim/psprites.py, render/psprites.py)."""

from __future__ import annotations

import numpy as np
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder, encode_picture

from pyhexen.core.fixed import FRACBITS, FRACUNIT
from pyhexen.core.ticcmd import BT_ATTACK, BT_CHANGE, BT_WEAPONSHIFT
from pyhexen.render.overlay2d import Overlay2D
from pyhexen.render.psprites import (
    PSPRITE_SY,
    PspriteRenderer,
    compute_psprite_draw_pos,
    is_psprite_fullbright,
    is_psprite_translucent,
)
from pyhexen.sim.actions import is_action_implemented
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import SPRNAMES, STATES, MobjFlag, MobjType, StateNum
from pyhexen.sim.inter import damage_mobj
from pyhexen.sim.mobj import ONFLOORZ, set_mobj_state, spawn_mobj
from pyhexen.sim.player import PSTATE_ATTACK, PSTATE_NORMAL, PlayerClass, spawn_player
from pyhexen.sim.psprites import (
    LOWERSPEED,
    MANA_1,
    MANA_2,
    MANA_BOTH,
    MANA_NONE,
    PS_FLASH,
    PS_WEAPON,
    RAISESPEED,
    WEAPON_INFO,
    WEAPON_MANA_USE,
    WEAPONBOTTOM,
    WEAPONTOP,
    WP_FIRST,
    WP_FOURTH,
    WP_NOCHANGE,
    WP_SECOND,
    WP_THIRD,
    WPIECE1,
    WPIECE2,
    WPIECE3,
    a_light0,
    a_refire,
    deduct_weapon_mana,
    give_mana,
    give_weapon,
    give_weapon_piece,
    p_activate_morph_weapon,
    p_bring_up_weapon,
    p_check_mana,
    p_fire_weapon,
    p_move_psprites,
    p_post_morph_weapon,
    p_set_psprite,
)
from pyhexen.wad.wadfile import WadFile
from pyhexen.world.level import load_level


def _make_box_sim(player_class: int = int(PlayerClass.PCLASS_FIGHTER)) -> GameSim:
    wb = WadBuilder()
    mb = MapBuilder("MAP01")
    mb.add_room(-512, -512, 1024, 1024, floor_height=0, ceiling_height=256)
    wb.add_map(mb)
    wad = WadFile.from_bytes(wb.to_bytes())
    level = load_level(wad, "MAP01")
    return GameSim(level=level, player_class=player_class)


def test_psprite_actions_registered() -> None:
    """Verify all T10.6 psprite framework actions are registered in the action table."""
    for name in (
        "A_WeaponReady",
        "A_ReFire",
        "A_Lower",
        "A_Raise",
        "A_SnoutAttack",
        "A_Light0",
    ):
        assert is_action_implemented(name), f"{name} should be implemented"


def test_weapon_info_and_mana_tables() -> None:
    """Verify WeaponInfo and WeaponManaUse tables match p_pspr.c."""
    assert len(WEAPON_INFO) == 4
    for wp in range(4):
        assert len(WEAPON_INFO[wp]) == 4
    # First weapon uses no mana for any class
    for pclass in range(4):
        assert WEAPON_INFO[WP_FIRST][pclass].mana == MANA_NONE
        assert WEAPON_MANA_USE[pclass][WP_FIRST] == 0
    # Fighter WP_SECOND (Timon's Axe) and WP_THIRD (Hammer of Retribution) have MANA_NONE
    # in WeaponInfo (can swing melee without mana), while Cleric/Mage require MANA_1 / MANA_2
    assert WEAPON_INFO[WP_SECOND][0].mana == MANA_NONE
    assert WEAPON_INFO[WP_SECOND][1].mana == MANA_1
    assert WEAPON_INFO[WP_SECOND][2].mana == MANA_1
    assert WEAPON_INFO[WP_THIRD][0].mana == MANA_NONE
    assert WEAPON_INFO[WP_THIRD][1].mana == MANA_2
    assert WEAPON_INFO[WP_THIRD][2].mana == MANA_2
    # Fourth weapon uses MANA_BOTH for all 3 classes
    for pclass in range(3):
        assert WEAPON_INFO[WP_FOURTH][pclass].mana == MANA_BOTH


def test_weapon_raise_to_ready_and_bobbing() -> None:
    """Test p_setup_psprites -> A_Raise -> A_WeaponReady state machine and bobbing."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_FIGHTER))
    player = spawn_player(sim, x=0, y=0)
    psp = player.psprites[PS_WEAPON]

    # Initially raising WP_FIRST (S_PUNCHUP) from WEAPONBOTTOM
    assert psp.state_num == int(StateNum.S_PUNCHUP)
    assert psp.sy == WEAPONBOTTOM - RAISESPEED  # First tic of S_PUNCHUP runs A_Raise immediately

    # Advance until weapon reaches WEAPONTOP and transitions to S_PUNCHREADY
    for _ in range(30):
        p_move_psprites(sim, player)
        if psp.state_num == int(StateNum.S_PUNCHREADY):
            break

    assert psp.state_num == int(StateNum.S_PUNCHREADY)
    assert psp.sy == WEAPONTOP

    # Verify A_WeaponReady resets player mobj from PSTATE_ATTACK to PSTATE_NORMAL
    assert player.mo is not None
    set_mobj_state(sim, player.mo, PSTATE_ATTACK[player.player_class])
    player.bob = 8 * FRACUNIT
    sim.level_time = 5
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_PUNCHREADY))
    assert player.mo.state_num == PSTATE_NORMAL[player.player_class]
    # Bobbing shifts sx and sy based on player.bob and level_time
    assert psp.sx != FRACUNIT or psp.sy != WEAPONTOP


def test_weapon_switch_lower_and_raise() -> None:
    """Test switching weapons via BT_CHANGE -> A_Lower -> p_bring_up_weapon -> A_Raise."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_CLERIC))
    player = spawn_player(sim, x=0, y=0)
    player.weaponowned[WP_SECOND] = True
    player.mana[MANA_1] = 50
    psp = player.psprites[PS_WEAPON]

    # Put weapon in ready state
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_CMACEREADY))

    # Issue BT_CHANGE to WP_SECOND via player thinker
    player.cmd.buttons = BT_CHANGE | (WP_SECOND << BT_WEAPONSHIFT)
    sim.tick()
    assert player.pendingweapon in (WP_SECOND, WP_NOCHANGE)

    # Tick until WP_SECOND is raised to S_CSTAFFREADY
    player.cmd.buttons = 0
    for _ in range(60):
        sim.tick()
        if player.readyweapon == WP_SECOND and psp.state_num in (
            int(StateNum.S_CSTAFFREADY),
            int(StateNum.S_CSTAFFREADY1),
            int(StateNum.S_CSTAFFREADY2),
        ):
            break

    assert player.readyweapon == WP_SECOND
    assert player.pendingweapon == WP_NOCHANGE


def test_fighter_glowing_axe_states_and_mana_transitions() -> None:
    """Test Fighter's Timon's Axe glowing state transitions with and without Blue Mana."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_FIGHTER))
    player = spawn_player(sim, x=0, y=0)
    player.weaponowned[WP_SECOND] = True
    psp = player.psprites[PS_WEAPON]

    # Raise WP_SECOND without mana -> S_FAXEUP -> S_FAXEREADY
    player.mana[MANA_1] = 0
    player.readyweapon = WP_SECOND
    player.pendingweapon = WP_SECOND
    p_bring_up_weapon(sim, player)
    assert psp.state_num == int(StateNum.S_FAXEUP)
    for _ in range(30):
        p_move_psprites(sim, player)
        if psp.state_num == int(StateNum.S_FAXEREADY):
            break
    assert psp.state_num == int(StateNum.S_FAXEREADY)

    # Firing without mana uses non-glowing S_FAXEATK_1
    p_fire_weapon(sim, player)
    assert psp.state_num == int(StateNum.S_FAXEATK_1)

    # Giving Blue Mana while in S_FAXEREADY transitions to S_FAXEREADY_G
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_FAXEREADY))
    assert give_mana(sim, player, MANA_1, 25) is True
    assert player.mana[MANA_1] == 25
    assert psp.state_num == int(StateNum.S_FAXEREADY_G)

    # Firing with mana uses glowing S_FAXEATK_G1
    p_fire_weapon(sim, player)
    assert psp.state_num == int(StateNum.S_FAXEATK_G1)


def test_mana_check_and_auto_switch_when_out_of_mana() -> None:
    """Test p_check_mana auto-switching to best available weapon when out of mana."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_MAGE))
    player = spawn_player(sim, x=0, y=0)
    psp = player.psprites[PS_WEAPON]
    player.weaponowned = [True, True, True, True]
    player.readyweapon = WP_FOURTH
    psp.sy = WEAPONTOP
    player.mana = [5, 5]  # Mage WP_FOURTH costs 15 of both mana

    # Not enough for WP_FOURTH (15) or WP_THIRD (5 is enough! Mage WP_THIRD costs 5 MANA_2)
    assert p_check_mana(sim, player) is False
    assert player.pendingweapon == WP_THIRD
    assert psp.sy == WEAPONTOP + LOWERSPEED

    # Now set MANA_2 to 0 and MANA_1 to 5 (Mage WP_SECOND costs 3 MANA_1)
    player.readyweapon = WP_THIRD
    player.pendingweapon = WP_NOCHANGE
    psp.sy = WEAPONTOP
    player.mana = [5, 0]
    assert p_check_mana(sim, player) is False
    assert player.pendingweapon == WP_SECOND

    # Now set both mana to 0 -> falls back to WP_FIRST
    player.readyweapon = WP_SECOND
    player.pendingweapon = WP_NOCHANGE
    psp.sy = WEAPONTOP
    player.mana = [0, 0]
    assert p_check_mana(sim, player) is False
    assert player.pendingweapon == WP_FIRST

    # Deduct weapon mana helper
    player.readyweapon = WP_FOURTH
    player.mana = [50, 50]
    deduct_weapon_mana(player)
    assert player.mana == [35, 35]


def test_refire_and_light0() -> None:
    """Test A_ReFire continuous attack and A_Light0 resetting extralight."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_FIGHTER))
    player = spawn_player(sim, x=0, y=0)
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_PUNCHREADY))

    # Hold attack button and call a_refire
    player.cmd.buttons = BT_ATTACK
    player.pendingweapon = WP_NOCHANGE
    a_refire(sim, player, psp)
    assert player.refire == 1
    assert psp.state_num == int(StateNum.S_PUNCHATK1_1)

    # Release attack button -> refire resets to 0
    player.cmd.buttons = 0
    a_refire(sim, player, psp)
    assert player.refire == 0

    player.extralight = 2
    a_light0(sim, player, psp)
    assert player.extralight == 0


def test_fourth_weapon_assembly_from_three_pieces() -> None:
    """Test assembling the 4th weapon (WP_FOURTH) from WPIECE1, WPIECE2, and WPIECE3."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_FIGHTER))
    player = spawn_player(sim, x=0, y=0)

    assert player.weaponowned[WP_FOURTH] is False
    assert player.pieces == 0

    # Piece 1
    assert give_weapon_piece(sim, player, int(PlayerClass.PCLASS_FIGHTER), WPIECE1) == (
        True,
        False,
    )
    assert player.pieces == WPIECE1
    assert player.weaponowned[WP_FOURTH] is False
    assert player.mana == [20, 20]

    # Wrong class piece gives mana only without adding to player.pieces
    assert give_weapon_piece(sim, player, int(PlayerClass.PCLASS_MAGE), WPIECE2) == (
        True,
        False,
    )
    assert player.pieces == WPIECE1
    assert player.mana == [40, 40]

    # Piece 2
    assert give_weapon_piece(sim, player, int(PlayerClass.PCLASS_FIGHTER), WPIECE2) == (
        True,
        False,
    )
    assert player.pieces == (WPIECE1 | WPIECE2)
    assert player.weaponowned[WP_FOURTH] is False

    # Piece 3 completes the 4th weapon (Quietus)!
    assert give_weapon_piece(sim, player, int(PlayerClass.PCLASS_FIGHTER), WPIECE3) == (
        True,
        True,
    )
    assert player.pieces == (WPIECE1 | WPIECE2 | WPIECE3)
    assert player.weaponowned[WP_FOURTH] is True
    assert player.pendingweapon == WP_FOURTH
    assert player.message == "QUIETUS ASSEMBLED"


def test_give_weapon_and_drop_weapon_on_death() -> None:
    """Test give_weapon pickup logic and p_drop_weapon when player dies."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_CLERIC))
    player = spawn_player(sim, x=0, y=0)
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_CMACEREADY))

    assert give_weapon(sim, player, int(PlayerClass.PCLASS_CLERIC), WP_THIRD) is True
    assert player.weaponowned[WP_THIRD] is True
    assert player.pendingweapon == WP_THIRD
    assert player.mana[MANA_2] == 25

    # Kill the player -> drop_weapon puts ps_weapon into downstate (S_CMACEDOWN)
    assert player.mo is not None
    damage_mobj(sim, player.mo, None, None, 10000)
    assert psp.state_num == int(StateNum.S_CMACEDOWN)


def test_pig_morph_weapon_and_snout_attack() -> None:
    """Test P_ActivateMorphWeapon, A_SnoutAttack melee hit, and P_PostMorphWeapon."""
    sim = _make_box_sim(int(PlayerClass.PCLASS_FIGHTER))
    player = spawn_player(sim, x=0, y=0, angle=0)
    assert player.mo is not None

    # Spawn a target directly in front of the player (east, +X)
    target = spawn_mobj(sim, 40 * FRACUNIT, 0, ONFLOORZ, MobjType.MT_ETTIN)
    initial_health = target.health

    player.player_class = int(PlayerClass.PCLASS_PIG)
    p_activate_morph_weapon(sim, player)
    psp = player.psprites[PS_WEAPON]
    assert psp.state_num == int(StateNum.S_SNOUTREADY)

    # Fire snout attack: S_SNOUTATK1 (4 tics) -> S_SNOUTATK2 (calls A_SnoutAttack)
    p_fire_weapon(sim, player)
    assert psp.state_num == int(StateNum.S_SNOUTATK1)
    for _ in range(5):
        p_move_psprites(sim, player)
    assert target.health < initial_health

    # Restore post-morph weapon
    player.player_class = int(PlayerClass.PCLASS_FIGHTER)
    p_post_morph_weapon(sim, player, WP_FIRST)
    assert psp.state_num == int(StateNum.S_PUNCHUP)
    assert psp.sy == WEAPONBOTTOM - RAISESPEED


def test_psprite_renderer_and_overlay_coordinates() -> None:
    """Test PspriteRenderer drawing ps_weapon onto a 320x200 Overlay2D."""
    wb = WadBuilder()
    mb = MapBuilder("MAP01")
    mb.add_room(-512, -512, 1024, 1024, floor_height=0, ceiling_height=256)
    wb.add_map(mb)

    # Build a synthetic sprite lump for Fighter punch ready
    spr_prefix = SPRNAMES[STATES[int(StateNum.S_PUNCHREADY)].sprite]
    idx = np.full((16, 20), 42, dtype=np.uint8)
    msk = np.ones((16, 20), dtype=bool)
    wb.add_lump("S_START", b"")
    wb.add_lump(f"{spr_prefix}A0", encode_picture(idx, msk, left=-150, top=-100))
    wb.add_lump("S_END", b"")

    wad = WadFile.from_bytes(wb.to_bytes())
    level = load_level(wad, "MAP01")
    sim = GameSim(level=level, player_class=int(PlayerClass.PCLASS_FIGHTER))
    player = spawn_player(sim, x=0, y=0)
    psp = player.psprites[PS_WEAPON]
    p_set_psprite(sim, player, PS_WEAPON, int(StateNum.S_PUNCHREADY))
    psp.sx = FRACUNIT
    psp.sy = WEAPONTOP
    psp.prev_sx = FRACUNIT
    psp.prev_sy = WEAPONTOP

    # Check fullscreen vs status-bar coordinate calculation
    player.readyweapon = WP_SECOND
    x_fs, y_fs = compute_psprite_draw_pos(psp, player, fullscreen=True)
    x_sb, y_sb = compute_psprite_draw_pos(psp, player, fullscreen=False)
    assert x_fs == 1
    assert y_fs == (WEAPONTOP + PSPRITE_SY[0][WP_SECOND]) >> FRACBITS
    assert x_sb == 1
    assert y_sb == (WEAPONTOP >> FRACBITS) - 20

    # Check fullbright and translucent helpers
    assert is_psprite_fullbright(psp) is False
    assert is_psprite_translucent(player) is False
    assert player.mo is not None
    player.mo.flags |= MobjFlag.MF_SHADOW
    assert is_psprite_translucent(player) is True

    # Render onto Overlay2D
    overlay = Overlay2D(wad=wad)
    renderer = PspriteRenderer(wad)
    drawn = renderer.draw(overlay, player, fullscreen=True)
    assert drawn == 1
    assert np.any(overlay.mask)
    assert np.any(overlay.indices == 42)
    assert player.psprites[PS_FLASH].state is None
