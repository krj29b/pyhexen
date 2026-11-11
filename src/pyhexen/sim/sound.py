# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sound requests from the simulation (T15.2).

The sim never plays audio. It emits `SoundEvent` and `SequenceEvent` on its event
bus, and the audio layer turns them into mixer calls. This module is the single
place that builds those events, mirroring the vanilla entry points:
S_StartSound / S_StartSoundAtVolume / S_StopSound (s_sound.c) and
SN_StartSequence / SN_StartSequenceName / SN_StopSequence (sn_sonix.c).

An `origin` is whatever vanilla passes as `mobj_t *`: a `Mobj`, a sector (its
sound origin), a polyobject start spot, or a fixed `SoundPoint`. `None` means
"at the listener, full volume", like `S_StartSound(NULL, ...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyhexen.core.events import SequenceEvent, SoundEvent

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim


@dataclass(slots=True, frozen=True)
class SoundPoint:
    """A fixed sound position (16.16 map coordinates) for sounds without a mobj."""

    x: int
    y: int
    z: int = 0


def origin_xyz(origin: Any) -> tuple[int, int, int]:
    """Return the 16.16 position of a sound origin (Mobj, Sector, spot or point)."""
    if origin is None:
        return (0, 0, 0)
    if hasattr(origin, "sound_org_x"):
        return (int(origin.sound_org_x), int(origin.sound_org_y), 0)
    return (int(origin.x), int(origin.y), int(getattr(origin, "z", 0)))


def start_sound(sim: Any, origin: Any, sound_id: int | str, volume: int = 127) -> None:
    """Port of S_StartSound / S_StartSoundAtVolume (s_sound.c).

    `sound_id` is an `SfxEnum` value or a SNDINFO sound name.
    """
    if sim is None or not hasattr(sim, "events"):
        return
    if isinstance(sound_id, int) and sound_id <= 0:
        return
    if not sound_id:
        return
    x, y, z = origin_xyz(origin)
    sim.events.emit(
        SoundEvent(
            sound_id=sound_id,
            origin_id=id(origin) if origin is not None else None,
            x=x,
            y=y,
            z=z,
            volume=max(0, min(127, int(volume))),
            origin=origin,
        )
    )


def stop_sound(sim: Any, origin: Any) -> None:
    """Port of S_StopSound (s_sound.c): stop whatever `origin` is playing."""
    if sim is None or origin is None or not hasattr(sim, "events"):
        return
    sim.events.emit(SoundEvent(sound_id=0, origin_id=id(origin), stop=True, origin=origin))


def start_sequence(sim: GameSim, origin: Any, sequence: int | str) -> None:
    """Port of SN_StartSequence / SN_StartSequenceName (sn_sonix.c).

    `sequence` is a `SEQ_*` number or a SNDSEQ sequence name. A `None` origin
    plays at the listener, as vanilla does for ACS SoundSequence without a line.
    """
    sim.events.emit(SequenceEvent(origin=origin, sequence=sequence))


def stop_sequence(sim: GameSim, origin: Any) -> None:
    """Port of SN_StopSequence (sn_sonix.c)."""
    sim.events.emit(SequenceEvent(origin=origin, sequence=-1, stop=True))


# Sequence bases (h2def.h): a sector's sequence is base + sector.seq_type.
SEQ_PLATFORM: int = 0
SEQ_DOOR_STONE: int = 10
SEQTYPE_NUMSEQ: int = 10


def _seq_type(sector: Any) -> int:
    seq_type = int(getattr(sector, "seq_type", 0))
    return seq_type if 0 <= seq_type < SEQTYPE_NUMSEQ else 0


def start_door_sequence(sim: GameSim, sector: Any) -> None:
    """`SN_StartSequence(&sector->soundorg, SEQ_DOOR_STONE + sector->seqType)`."""
    start_sequence(sim, sector, SEQ_DOOR_STONE + _seq_type(sector))


def start_platform_sequence(sim: GameSim, sector: Any) -> None:
    """`SN_StartSequence(&sector->soundorg, SEQ_PLATFORM + sector->seqType)`."""
    start_sequence(sim, sector, SEQ_PLATFORM + _seq_type(sector))
