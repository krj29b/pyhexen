# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for T6.1: TicCmd bitfields and InputHandler.build_ticcmd."""

from __future__ import annotations

import pygame
import pytest

from pyhexen.app.input import (
    ANGLE_TURN,
    FORWARD_MOVE,
    SIDE_MOVE,
    SLOW_TURN_TICS,
    InputHandler,
    KeyBindings,
)
from pyhexen.core.ticcmd import (
    AFLAG_JUMP,
    AFLAG_SUICIDE,
    BT_ATTACK,
    BT_CHANGE,
    BT_USE,
    TOCENTER,
    pack_lookfly,
    unpack_lookfly,
)


def test_lookfly_pack_unpack_roundtrip() -> None:
    for look in range(-8, 8):
        for fly in range(-8, 8):
            packed = pack_lookfly(look, fly)
            assert 0 <= packed <= 255
            assert unpack_lookfly(packed) == (look, fly)

    packed_center = pack_lookfly(TOCENTER, TOCENTER)
    assert unpack_lookfly(packed_center) == (TOCENTER, TOCENTER)

    with pytest.raises(ValueError):
        pack_lookfly(-9, 0)
    with pytest.raises(ValueError):
        pack_lookfly(0, 8)


def test_walk_and_run_movement() -> None:
    handler = InputHandler()
    b = handler.bindings

    # Walk forward & strafe right
    cmd_walk = handler.build_ticcmd(pressed_keys={b.forward, b.strafe_right})
    assert cmd_walk.forwardmove == FORWARD_MOVE[0]  # 0x19
    assert cmd_walk.sidemove == SIDE_MOVE[0]  # 0x18

    # Run forward & strafe left
    cmd_run = handler.build_ticcmd(pressed_keys={b.forward, b.strafe_left, b.speed})
    assert cmd_run.forwardmove == FORWARD_MOVE[1]  # 0x32
    assert cmd_run.sidemove == -SIDE_MOVE[1]  # -0x28

    # Opposite keys cancel out
    cmd_cancel = handler.build_ticcmd(
        pressed_keys={b.forward, b.backward, b.strafe_left, b.strafe_right}
    )
    assert cmd_cancel.forwardmove == 0
    assert cmd_cancel.sidemove == 0


def test_two_stage_accelerative_turning() -> None:
    handler = InputHandler()
    b = handler.bindings

    # First SLOW_TURN_TICS - 1 tics (1..5) use slow turn speed (ANGLE_TURN[2] == 320)
    for _ in range(SLOW_TURN_TICS - 1):
        cmd = handler.build_ticcmd(pressed_keys={b.turn_left})
        assert cmd.angleturn == ANGLE_TURN[2]  # 320

    # On the 6th tic, it switches to normal turn speed (640) or fast turn speed (1280)
    cmd_walk_turn = handler.build_ticcmd(pressed_keys={b.turn_left})
    assert cmd_walk_turn.angleturn == ANGLE_TURN[0]  # 640

    cmd_run_turn = handler.build_ticcmd(pressed_keys={b.turn_right, b.speed})
    assert cmd_run_turn.angleturn == -ANGLE_TURN[1]  # -1280

    # Releasing resets turnheld
    handler.build_ticcmd(pressed_keys=set())
    cmd_reset = handler.build_ticcmd(pressed_keys={b.turn_left})
    assert cmd_reset.angleturn == ANGLE_TURN[2]


def test_strafe_modifier_and_mouse_input() -> None:
    handler = InputHandler()
    b = handler.bindings

    # Strafe modifier turns turn_right into sidemove
    cmd = handler.build_ticcmd(pressed_keys={b.strafe_mod, b.turn_right})
    assert cmd.angleturn == 0
    assert cmd.sidemove == SIDE_MOVE[0]

    # Mouse X turns angleturn by -dx * 8
    cmd_mouse = handler.build_ticcmd(pressed_keys=set(), mouse_dx=10, mouse_dy=5)
    assert cmd_mouse.angleturn == -80
    assert cmd_mouse.forwardmove == 5


def test_buttons_weapons_jump_and_artifacts() -> None:
    handler = InputHandler()
    b = handler.bindings

    cmd = handler.build_ticcmd(
        pressed_keys={b.fire, b.use, b.jump, b.weapon_3, b.use_artifact},
        ready_artifact=5,
        suicide=True,
    )
    assert cmd.buttons & BT_ATTACK
    assert cmd.buttons & BT_USE
    assert cmd.buttons & BT_CHANGE
    assert cmd.weapon_change == 2  # 0-based index for weapon_3
    assert cmd.is_jump
    assert cmd.arti & AFLAG_JUMP
    assert cmd.arti & AFLAG_SUICIDE
    assert cmd.artifact_type == 5

    # Holding use_artifact on the next tic without releasing does not re-fire the artifact
    cmd_next = handler.build_ticcmd(
        pressed_keys={b.use_artifact},
        ready_artifact=5,
    )
    assert cmd_next.artifact_type == 0


def test_look_and_fly_keys() -> None:
    handler = InputHandler()
    b = handler.bindings

    cmd_up = handler.build_ticcmd(pressed_keys={b.look_up, b.fly_up})
    assert cmd_up.look_delta == 1
    assert cmd_up.fly_delta == 5

    cmd_down = handler.build_ticcmd(pressed_keys={b.look_down, b.fly_down})
    assert cmd_down.look_delta == -1
    assert cmd_down.fly_delta == -5

    cmd_center = handler.build_ticcmd(pressed_keys={b.fly_center})
    assert cmd_center.look_delta == TOCENTER
    assert cmd_center.fly_delta == TOCENTER


def test_pygame_event_processing() -> None:
    bindings = KeyBindings()
    handler = InputHandler(bindings=bindings)

    events = [
        pygame.event.Event(pygame.KEYDOWN, key=bindings.forward),
        pygame.event.Event(pygame.KEYDOWN, key=bindings.inv_right),
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1),
        pygame.event.Event(pygame.MOUSEMOTION, rel=(4, 0)),
    ]
    handler.process_events(events)

    assert handler.consume_inventory_moves() == (0, 1)
    assert handler.consume_inventory_moves() == (0, 0)

    cmd = handler.build_ticcmd()
    assert cmd.forwardmove == FORWARD_MOVE[0]
    assert cmd.buttons & BT_ATTACK
    assert cmd.angleturn == -32
