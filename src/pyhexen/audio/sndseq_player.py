# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Sound sequence runtime interpreter for doors, platforms, polyobjects, and environment sounds.

Port of SN_StartSequence, SN_StopSequence, SN_UpdateActiveSequences, and
SN_StopAllSequences (sn_sonix.c).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from pyhexen.core.events import EventBus, SoundEvent
from pyhexen.core.fixed import FRACBITS
from pyhexen.core.random import Random
from pyhexen.wad.sndseq import SEQUENCE_TRANSLATE, SeqOpcode, SoundSequence, load_sndseq
from pyhexen.wad.wadfile import WadFile, WadStack
from pyhexen.world.level import Level, MapThing, Sector

# MapThing doomednums for sequence overrides and environmental wind sound
THING_SEQ_OVERRIDE_MIN = 1400
THING_SEQ_OVERRIDE_MAX = 1409
THING_SOUND_WIND = 1410


class SeqId(IntEnum):
    """Standard sound sequence indices.

    Port of SEQ_PLATFORM..SEQ_NUMSEQ (h2def.h).
    """

    SEQ_PLATFORM = 0
    SEQ_PLATFORM_HEAVY = 1
    SEQ_PLATFORM_METAL = 2
    SEQ_PLATFORM_CREAK = 3
    SEQ_PLATFORM_SILENCE = 4
    SEQ_PLATFORM_LAVA = 5
    SEQ_PLATFORM_WATER = 6
    SEQ_PLATFORM_ICE = 7
    SEQ_PLATFORM_EARTH = 8
    SEQ_PLATFORM_METAL2 = 9
    SEQ_DOOR_STONE = 10
    SEQ_DOOR_HEAVY = 11
    SEQ_DOOR_METAL = 12
    SEQ_DOOR_CREAK = 13
    SEQ_DOOR_SILENCE = 14
    SEQ_DOOR_LAVA = 15
    SEQ_DOOR_WATER = 16
    SEQ_DOOR_ICE = 17
    SEQ_DOOR_EARTH = 18
    SEQ_DOOR_METAL2 = 19
    SEQ_ESOUND_WIND = 20


class SeqType(IntEnum):
    """Sector sound sequence types set by thing types 1400..1409.

    Port of seqtype_t (h2def.h).
    """

    SEQTYPE_STONE = 0
    SEQTYPE_HEAVY = 1
    SEQTYPE_METAL = 2
    SEQTYPE_CREAK = 3
    SEQTYPE_SILENCE = 4
    SEQTYPE_LAVA = 5
    SEQTYPE_WATER = 6
    SEQTYPE_ICE = 7
    SEQTYPE_EARTH = 8
    SEQTYPE_METAL2 = 9


def _extract_origin_coords(origin: Any) -> tuple[int | None, int, int, int]:
    """Extract `(origin_id, x, y, z)` in fixed-point from a Sector, Mobj, or MapThing."""
    if origin is None:
        return (None, 0, 0, 0)
    if isinstance(origin, Sector):
        return (id(origin), origin.sound_org_x, origin.sound_org_y, 0)
    if isinstance(origin, MapThing):
        return (id(origin), origin.x << FRACBITS, origin.y << FRACBITS, origin.z << FRACBITS)
    x = getattr(origin, "x", getattr(origin, "sound_org_x", 0))
    y = getattr(origin, "y", getattr(origin, "sound_org_y", 0))
    z = getattr(origin, "z", 0)
    return (id(origin), int(x), int(y), int(z))


@dataclass(slots=True)
class ActiveSequence:
    """An active sound sequence node executing bytecode per tic.

    Port of seqnode_t (h2def.h / sn_sonix.c).
    """

    sequence: SoundSequence
    ip: int = 0
    delay_tics: int = 0
    volume: int = 127
    origin: Any = None
    current_sound: str | None = None
    stop_sound: str | None = None


class SoundSequencePlayer:
    """Runtime interpreter for Hexen SNDSEQ scripts.

    Port of sn_sonix.c.
    """

    def __init__(self, sequences: dict[str, SoundSequence] | None = None) -> None:
        self.sequences: dict[str, SoundSequence] = dict(sequences) if sequences else {}
        self._by_lower: dict[str, SoundSequence] = {k.lower(): v for k, v in self.sequences.items()}
        self.active_sequences: list[ActiveSequence] = []
        self._playing_sounds: set[tuple[int | None, str]] = set()

    @classmethod
    def from_wad(cls, wad: WadStack | WadFile) -> SoundSequencePlayer:
        """Create a SoundSequencePlayer from the SNDSEQ lump in `wad`."""
        return cls(load_sndseq(wad))

    def resolve_sequence(self, seq_name_or_id: str | int) -> SoundSequence | None:
        """Resolve a sequence name or `SeqId` (0..20) to a `SoundSequence`."""
        if isinstance(seq_name_or_id, int):
            if 0 <= seq_name_or_id < len(SEQUENCE_TRANSLATE):
                name = SEQUENCE_TRANSLATE[seq_name_or_id]
            else:
                return None
        else:
            name = seq_name_or_id

        if name in self.sequences:
            return self.sequences[name]
        return self._by_lower.get(name.lower())

    def start_sequence(
        self,
        origin: Any,
        seq_name_or_id: str | int,
        *,
        event_bus: EventBus | None = None,
    ) -> ActiveSequence | None:
        """Start a sound sequence on `origin`, stopping any previous sequence on that origin.

        Port of SN_StartSequence and SN_StartSequenceName (sn_sonix.c).
        """
        self.stop_sequence(origin, event_bus=event_bus)
        seq = self.resolve_sequence(seq_name_or_id)
        if seq is None:
            return None

        node = ActiveSequence(
            sequence=seq,
            ip=0,
            delay_tics=0,
            volume=127,
            origin=origin,
            current_sound=None,
            stop_sound=seq.stop_sound,
        )
        # Prepend at head matching SequenceListHead in sn_sonix.c
        self.active_sequences.insert(0, node)
        return node

    def stop_sequence(
        self,
        origin: Any,
        *,
        event_bus: EventBus | None = None,
        play_stop_sound: bool = True,
    ) -> list[SoundEvent]:
        """Stop any active sequence on `origin` and optionally emit its `stop_sound`.

        Port of SN_StopSequence (sn_sonix.c).
        """
        emitted: list[SoundEvent] = []
        remaining: list[ActiveSequence] = []

        for node in self.active_sequences:
            if node.origin == origin or node.origin is origin:
                origin_id, x, y, z = _extract_origin_coords(node.origin)
                if node.current_sound is not None:
                    self._playing_sounds.discard((origin_id, node.current_sound))
                    stop_ev = SoundEvent(
                        sound_id=node.current_sound,
                        origin_id=origin_id,
                        x=x,
                        y=y,
                        z=z,
                        volume=0,
                        stop=True,
                    )
                    emitted.append(stop_ev)
                    if event_bus is not None:
                        event_bus.emit(stop_ev)

                if play_stop_sound and node.stop_sound:
                    ev = SoundEvent(
                        sound_id=node.stop_sound,
                        origin_id=origin_id,
                        x=x,
                        y=y,
                        z=z,
                        volume=node.volume,
                        stop=False,
                    )
                    emitted.append(ev)
                    if event_bus is not None:
                        event_bus.emit(ev)
            else:
                remaining.append(node)

        self.active_sequences = remaining
        return emitted

    def stop_all_sequences(self, *, event_bus: EventBus | None = None) -> None:
        """Stop all active sequences without playing stop sounds.

        Port of SN_StopAllSequences (sn_sonix.c).
        """
        for node in list(self.active_sequences):
            node.stop_sound = None
            self.stop_sequence(node.origin, event_bus=event_bus, play_stop_sound=False)

    def mark_sound_finished(self, origin: Any, sound_name: str) -> None:
        """Notify the sequence player that `sound_name` on `origin` has finished playing."""
        origin_id, _, _, _ = _extract_origin_coords(origin)
        self._playing_sounds.discard((origin_id, sound_name))

    def is_sound_playing(self, node: ActiveSequence) -> bool:
        """Return True if `node.current_sound` is marked as still playing on `node.origin`."""
        if node.current_sound is None:
            return False
        origin_id, _, _, _ = _extract_origin_coords(node.origin)
        return (origin_id, node.current_sound) in self._playing_sounds

    def tick(
        self,
        rng: Random,
        event_bus: EventBus | None = None,
    ) -> list[SoundEvent]:
        """Advance all active sound sequences by one tic (35 Hz).

        Port of SN_UpdateActiveSequences (sn_sonix.c).
        """
        if not self.active_sequences:
            return []

        emitted: list[SoundEvent] = []

        for node in list(self.active_sequences):
            if node not in self.active_sequences:
                continue

            if node.delay_tics > 0:
                node.delay_tics -= 1
                continue

            instructions = node.sequence.instructions
            if node.ip < 0 or node.ip >= len(instructions):
                self.stop_sequence(node.origin, event_bus=event_bus, play_stop_sound=True)
                continue

            instr = instructions[node.ip]
            opcode = instr[0]
            snd_playing = self.is_sound_playing(node)
            origin_id, x, y, z = _extract_origin_coords(node.origin)

            if opcode == SeqOpcode.SS_CMD_PLAY:
                sound_name = str(instr[1])
                if not snd_playing:
                    node.current_sound = sound_name
                    self._playing_sounds.add((origin_id, sound_name))
                    ev = SoundEvent(
                        sound_id=sound_name,
                        origin_id=origin_id,
                        x=x,
                        y=y,
                        z=z,
                        volume=node.volume,
                        stop=False,
                    )
                    emitted.append(ev)
                    if event_bus is not None:
                        event_bus.emit(ev)
                node.ip += 1

            elif opcode == SeqOpcode.SS_CMD_WAIT:
                if not snd_playing:
                    node.ip += 1
                    node.current_sound = None

            elif opcode == SeqOpcode.SS_CMD_PLAYTIME:
                sound_name = str(instr[1])
                tics = int(instr[2])
                if not snd_playing:
                    node.current_sound = sound_name
                    self._playing_sounds.add((origin_id, sound_name))
                    ev = SoundEvent(
                        sound_id=sound_name,
                        origin_id=origin_id,
                        x=x,
                        y=y,
                        z=z,
                        volume=node.volume,
                        stop=False,
                    )
                    emitted.append(ev)
                    if event_bus is not None:
                        event_bus.emit(ev)
                node.delay_tics = tics
                node.ip += 1

            elif opcode == SeqOpcode.SS_CMD_PLAYREPEAT:
                sound_name = str(instr[1])
                if not snd_playing:
                    node.current_sound = sound_name
                    self._playing_sounds.add((origin_id, sound_name))
                    ev = SoundEvent(
                        sound_id=sound_name,
                        origin_id=origin_id,
                        x=x,
                        y=y,
                        z=z,
                        volume=node.volume,
                        stop=False,
                    )
                    emitted.append(ev)
                    if event_bus is not None:
                        event_bus.emit(ev)

            elif opcode == SeqOpcode.SS_CMD_PLAYLOOP:
                sound_name = str(instr[1])
                tics = int(instr[2])
                node.current_sound = sound_name
                self._playing_sounds.add((origin_id, sound_name))
                ev = SoundEvent(
                    sound_id=sound_name,
                    origin_id=origin_id,
                    x=x,
                    y=y,
                    z=z,
                    volume=node.volume,
                    stop=False,
                )
                emitted.append(ev)
                if event_bus is not None:
                    event_bus.emit(ev)
                node.delay_tics = tics

            elif opcode == SeqOpcode.SS_CMD_DELAY:
                node.delay_tics = int(instr[1])
                node.ip += 1
                node.current_sound = None

            elif opcode == SeqOpcode.SS_CMD_DELAYRAND:
                min_tics = int(instr[1])
                max_tics = int(instr[2])
                span = max_tics - min_tics
                rand_offset = (rng.m_random() % span) if span > 0 else 0
                node.delay_tics = min_tics + rand_offset
                node.ip += 1
                node.current_sound = None

            elif opcode == SeqOpcode.SS_CMD_VOLUME:
                vol_pct = int(instr[1])
                node.volume = (127 * vol_pct) // 100
                node.ip += 1

            elif opcode == SeqOpcode.SS_CMD_STOPSOUND:
                # If STOPSOUND is not the final instruction before END, advance past it;
                # otherwise wait until SN_StopSequence stops the sequence.
                if (
                    node.ip + 1 < len(instructions)
                    and instructions[node.ip + 1][0] != SeqOpcode.SS_CMD_END
                ):
                    node.ip += 1

            elif opcode == SeqOpcode.SS_CMD_END:
                stop_events = self.stop_sequence(
                    node.origin,
                    event_bus=event_bus,
                    play_stop_sound=True,
                )
                emitted.extend(stop_events)

        return emitted

    def apply_level_things(
        self,
        level: Level,
        *,
        event_bus: EventBus | None = None,
    ) -> list[MapThing]:
        """Process sector sequence override things (1400..1409) and ambient wind things (1410).

        Port of sequence thing handling in P_SpawnMapThing (p_mobj.c).
        Returns the list of ambient sound things started.
        """
        ambient_things: list[MapThing] = []
        for thing in level.things:
            if THING_SEQ_OVERRIDE_MIN <= thing.type <= THING_SEQ_OVERRIDE_MAX:
                if level.subsectors:
                    sub = level.point_in_subsector(thing.x << FRACBITS, thing.y << FRACBITS)
                    sub.sector.seq_type = thing.type - THING_SEQ_OVERRIDE_MIN
            elif thing.type == THING_SOUND_WIND:
                ambient_things.append(thing)
                self.start_sequence(thing, SeqId.SEQ_ESOUND_WIND, event_bus=event_bus)
        return ambient_things
