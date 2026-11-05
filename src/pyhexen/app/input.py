# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Input handling and TicCmd builder.

Port of G_BuildTiccmd (g_game.c).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass, field

import pygame

from pyhexen.core.fixed import to_int16
from pyhexen.core.ticcmd import (
    AFLAG_JUMP,
    AFLAG_MASK,
    AFLAG_SUICIDE,
    BT_ATTACK,
    BT_CHANGE,
    BT_USE,
    BT_WEAPONSHIFT,
    TOCENTER,
    TicCmd,
    pack_lookfly,
)

# Per-class movement tables (g_game.c): Fighter=0, Cleric=1, Mage=2, Pig=3
MAX_PLAYER_MOVE: tuple[int, int, int, int] = (0x3C, 0x32, 0x2D, 0x31)
FORWARD_MOVE_BY_CLASS: tuple[tuple[int, int], ...] = (
    (0x1D, 0x3C),
    (0x19, 0x32),
    (0x16, 0x2E),
    (0x18, 0x31),
)
SIDE_MOVE_BY_CLASS: tuple[tuple[int, int], ...] = (
    (0x1B, 0x3B),
    (0x18, 0x28),
    (0x15, 0x25),
    (0x17, 0x27),
)

# Default movement tables (Cleric / standard speeds)
FORWARD_MOVE: tuple[int, int] = (0x19, 0x32)
SIDE_MOVE: tuple[int, int] = (0x18, 0x28)
ANGLE_TURN: tuple[int, int, int] = (640, 1280, 320)
SLOW_TURN_TICS: int = 6


@dataclass(slots=True)
class KeyBindings:
    """Configurable pygame key bindings for player controls."""

    forward: int = pygame.K_w
    backward: int = pygame.K_s
    strafe_left: int = pygame.K_a
    strafe_right: int = pygame.K_d
    turn_left: int = pygame.K_LEFT
    turn_right: int = pygame.K_RIGHT
    speed: int = pygame.K_LSHIFT
    strafe_mod: int = pygame.K_LALT
    fire: int = pygame.K_LCTRL
    use: int = pygame.K_e
    jump: int = pygame.K_SPACE
    look_up: int = pygame.K_PAGEUP
    look_down: int = pygame.K_PAGEDOWN
    look_center: int = pygame.K_HOME
    fly_up: int = pygame.K_INSERT
    fly_down: int = pygame.K_DELETE
    fly_center: int = pygame.K_END
    weapon_1: int = pygame.K_1
    weapon_2: int = pygame.K_2
    weapon_3: int = pygame.K_3
    weapon_4: int = pygame.K_4
    inv_left: int = pygame.K_LEFTBRACKET
    inv_right: int = pygame.K_RIGHTBRACKET
    use_artifact: int = pygame.K_RETURN


@dataclass(slots=True)
class InputHandler:
    """Samples keyboard and mouse state and builds deterministic TicCmd objects.

    Port of G_BuildTiccmd (g_game.c).
    """

    bindings: KeyBindings = field(default_factory=KeyBindings)
    mouse_sensitivity: float = 1.0
    always_run: bool = False

    _keys_down: set[int] = field(default_factory=set)
    _mouse_buttons_down: set[int] = field(default_factory=set)
    _mouse_dx: int = 0
    _mouse_dy: int = 0
    _turn_held: int = 0
    _look_held: int = 0
    _use_arti_armed: bool = True
    _inv_move_left: int = 0
    _inv_move_right: int = 0

    def reset(self) -> None:
        """Clear all held input states and accumulators."""
        self._keys_down.clear()
        self._mouse_buttons_down.clear()
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._turn_held = 0
        self._look_held = 0
        self._use_arti_armed = True
        self._inv_move_left = 0
        self._inv_move_right = 0

    def set_key(self, key: int, pressed: bool) -> None:
        """Set whether a key is currently held down."""
        if pressed:
            self._keys_down.add(key)
            if key == self.bindings.inv_left:
                self._inv_move_left += 1
            elif key == self.bindings.inv_right:
                self._inv_move_right += 1
        else:
            self._keys_down.discard(key)
            if key == self.bindings.use_artifact:
                self._use_arti_armed = True

    def set_mouse_button(self, button: int, pressed: bool) -> None:
        """Set whether a mouse button (1=left, 2=middle, 3=right) is held down."""
        if pressed:
            self._mouse_buttons_down.add(button)
        else:
            self._mouse_buttons_down.discard(button)

    def add_mouse_delta(self, dx: int, dy: int = 0) -> None:
        """Accumulate relative mouse motion for the next tic."""
        self._mouse_dx += dx
        self._mouse_dy += dy

    def process_event(self, event: pygame.event.Event) -> None:
        """Process a single pygame input event."""
        if event.type == pygame.KEYDOWN:
            self.set_key(event.key, True)
        elif event.type == pygame.KEYUP:
            self.set_key(event.key, False)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.set_mouse_button(event.button, True)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.set_mouse_button(event.button, False)
        elif event.type == pygame.MOUSEMOTION:
            rel_x, rel_y = event.rel
            self.add_mouse_delta(rel_x, rel_y)

    def process_events(self, events: Iterable[pygame.event.Event]) -> None:
        """Process a sequence of pygame input events."""
        for ev in events:
            self.process_event(ev)

    def consume_inventory_moves(self) -> tuple[int, int]:
        """Return and clear accumulated (left, right) inventory cursor presses."""
        left, right = self._inv_move_left, self._inv_move_right
        self._inv_move_left = 0
        self._inv_move_right = 0
        return left, right

    def build_ticcmd(
        self,
        *,
        pressed_keys: Set[int] | Mapping[int, bool] | None = None,
        mouse_dx: int | None = None,
        mouse_dy: int | None = None,
        player_class: int = 1,
        has_speed_power: bool = False,
        is_morphed: bool = False,
        ready_artifact: int = 0,
        suicide: bool = False,
    ) -> TicCmd:
        """Build a deterministic TicCmd for one simulation tic.

        Port of G_BuildTiccmd (g_game.c).
        """
        b = self.bindings

        def is_down(k: int) -> bool:
            if pressed_keys is not None:
                if isinstance(pressed_keys, Mapping):
                    return bool(pressed_keys.get(k, False))
                return k in pressed_keys
            return k in self._keys_down

        mdx = self._mouse_dx if mouse_dx is None else mouse_dx
        mdy = self._mouse_dy if mouse_dy is None else mouse_dy
        self._mouse_dx = 0
        self._mouse_dy = 0

        fmove_table = FORWARD_MOVE_BY_CLASS[player_class]
        smove_table = SIDE_MOVE_BY_CLASS[player_class]
        max_move = MAX_PLAYER_MOVE[player_class]

        strafe = is_down(b.strafe_mod)
        speed = 1 if (self.always_run or is_down(b.speed)) else 0

        forward = 0
        side = 0
        angleturn = 0
        look = 0
        flyheight = 0
        arti = 0
        buttons = 0

        # Two-stage accelerative turning
        if is_down(b.turn_left) or is_down(b.turn_right):
            self._turn_held += 1
        else:
            self._turn_held = 0
        tspeed = 2 if self._turn_held < SLOW_TURN_TICS else speed

        # Two-stage look speed
        if is_down(b.look_up) or is_down(b.look_down):
            self._look_held += 1
        else:
            self._look_held = 0
        lspeed = 1 if self._look_held < SLOW_TURN_TICS else 2

        if strafe:
            if is_down(b.turn_right):
                side += smove_table[speed]
            if is_down(b.turn_left):
                side -= smove_table[speed]
        else:
            if is_down(b.turn_right):
                angleturn -= ANGLE_TURN[tspeed]
            if is_down(b.turn_left):
                angleturn += ANGLE_TURN[tspeed]

        if is_down(b.forward):
            forward += fmove_table[speed]
        if is_down(b.backward):
            forward -= fmove_table[speed]

        if is_down(b.strafe_right):
            side += smove_table[speed]
        if is_down(b.strafe_left):
            side -= smove_table[speed]

        # Look up/down/center
        if is_down(b.look_up):
            look = lspeed
        if is_down(b.look_down):
            look = -lspeed
        if is_down(b.look_center):
            look = TOCENTER

        # Fly up/down/center
        if is_down(b.fly_up):
            flyheight = 5
        if is_down(b.fly_down):
            flyheight = -5
        if is_down(b.fly_center):
            flyheight = TOCENTER
            look = TOCENTER

        # Use artifact
        if is_down(b.use_artifact):
            if self._use_arti_armed and ready_artifact > 0:
                arti |= ready_artifact & AFLAG_MASK
                self._use_arti_armed = False
        else:
            self._use_arti_armed = True

        # Jump & suicide flags in arti byte
        if is_down(b.jump):
            arti |= AFLAG_JUMP
        if suicide:
            arti |= AFLAG_SUICIDE

        # Buttons: fire, use, weapon change
        if is_down(b.fire) or (1 in self._mouse_buttons_down):
            buttons |= BT_ATTACK
        if is_down(b.use):
            buttons |= BT_USE

        weapon_keys = (b.weapon_1, b.weapon_2, b.weapon_3, b.weapon_4)
        for i, wk in enumerate(weapon_keys):
            if is_down(wk):
                buttons |= BT_CHANGE
                buttons |= i << BT_WEAPONSHIFT
                break

        # Mouse movement
        scaled_mx = round(mdx * self.mouse_sensitivity)
        scaled_my = round(mdy * self.mouse_sensitivity)
        if strafe:
            side += scaled_mx * 2
        else:
            angleturn -= scaled_mx * 8
        forward += scaled_my

        # Clamp forward and side movement
        if forward > max_move:
            forward = max_move
        elif forward < -max_move:
            forward = -max_move

        if side > max_move:
            side = max_move
        elif side < -max_move:
            side = -max_move

        if has_speed_power and not is_morphed:
            forward = (3 * forward) >> 1
            side = (3 * side) >> 1

        return TicCmd(
            forwardmove=forward,
            sidemove=side,
            angleturn=to_int16(angleturn),
            buttons=buttons,
            lookfly=pack_lookfly(look, flyheight),
            arti=arti,
        )
