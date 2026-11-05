# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Action function registry and stub resolution for state table execution."""

from __future__ import annotations

import logging
from collections.abc import Callable

from pyhexen.sim.info_data import STATES

logger = logging.getLogger(__name__)

ActionCallable = Callable[..., None]

_ACTIONS: dict[str, ActionCallable] = {}
_STUBS: dict[str, ActionCallable] = {}


def action(name: str) -> Callable[[ActionCallable], ActionCallable]:
    """Decorator registering an action function under its original C name."""

    def _decorator(func: ActionCallable) -> ActionCallable:
        _ACTIONS[name] = func
        return func

    return _decorator


def _make_stub(name: str) -> ActionCallable:
    """Create a logging stub function for an unimplemented action."""

    def _stub(*args: object, **kwargs: object) -> None:
        logger.debug("Unimplemented action stub called: %s", name)

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub._is_stub = True  # type: ignore[attr-defined]
    return _stub


def get_action(name: str | None) -> ActionCallable | None:
    """Return the registered action callable for `name`, a stub if unimplemented, or None."""
    if name is None or name == "NULL":
        return None
    if name in _ACTIONS:
        return _ACTIONS[name]
    if name not in _STUBS:
        _STUBS[name] = _make_stub(name)
    return _STUBS[name]


def is_action_implemented(name: str) -> bool:
    """Return True if `name` has a registered non-stub implementation."""
    return name in _ACTIONS


def get_action_report() -> dict[str, bool]:
    """Return a sorted mapping of every unique action_name in STATES to its implemented status."""
    unique_names = sorted({s.action_name for s in STATES if s.action_name is not None})
    return {name: is_action_implemented(name) for name in unique_names}
