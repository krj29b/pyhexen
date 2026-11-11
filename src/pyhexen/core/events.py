# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Event bus and core simulation events for decoupling sim from render, audio, and UI."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class SoundEvent:
    """Emitted when the simulation starts or stops a sound effect.

    `origin` is the emitting object (Mobj, sector, polyobject spot or fixed point)
    so the audio layer can follow it as it moves; `None` means "at the listener".
    Build these with `pyhexen.sim.sound`.
    """

    sound_id: int | str
    origin_id: int | None = None
    x: int = 0
    y: int = 0
    z: int = 0
    volume: int = 127
    stop: bool = False
    origin: Any = field(default=None, compare=False)


@dataclass(slots=True, frozen=True)
class SequenceEvent:
    """Emitted when a sound sequence (SNDSEQ) starts or stops on an origin.

    `sequence` is a `SEQ_*` number or a sequence name. Port of the
    SN_StartSequence / SN_StopSequence calls in the sim.
    """

    origin: Any = field(compare=False)
    sequence: int | str
    stop: bool = False


@dataclass(slots=True, frozen=True)
class SectorChanged:
    """Emitted when a sector's floor/ceiling height, light, or pic changes (ADR-004)."""

    sector_index: int


@dataclass(slots=True, frozen=True)
class PrintMessage:
    """Emitted when a player message or ACS print/printbold occurs."""

    text: str
    player_index: int | None = None
    bold: bool = False


@dataclass(slots=True, frozen=True)
class MapExit:
    """Emitted when the level exits or transitions to another map in a hub."""

    next_map: int
    position: int = 0
    secret: bool = False


class EventBus:
    """Simple synchronous publish/subscribe and queue-draining event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[type[Any], list[Callable[[Any], None]]] = defaultdict(list)
        self._queue: list[Any] = []

    def subscribe(self, event_type: type[T], callback: Callable[[T], None]) -> None:
        """Register a callback to be invoked immediately when `event_type` is emitted."""
        self._subscribers[event_type].append(callback)

    def emit(self, event: Any) -> None:
        """Queue `event` and notify any subscribers registered for its exact type."""
        self._queue.append(event)
        for callback in self._subscribers.get(type(event), ()):
            callback(event)

    def drain(self) -> list[Any]:
        """Return all queued events in emission order and clear the internal queue."""
        events = list(self._queue)
        self._queue.clear()
        return events

    def clear(self) -> None:
        """Discard all queued events without returning them."""
        self._queue.clear()
