# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Game audio: sim sound events, sound sequences and music (T15.2).

Port of the per-tic sound work in Hexen: S_UpdateSounds (with
SN_UpdateActiveSequences), S_StartSong on level start, S_StopAllSound and
SN_StopAllSequences between levels, and S_StartSongName for the title,
intermission and finale.

The simulation only emits events; this class turns them into mixer calls. It
uses its own RNG for sequence delays so audio never changes the game (vanilla
shares M_Random, see ADR-009).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pyhexen.audio.channels import Mixer, NullMixer, PygameMixer, SoundSystem
from pyhexen.audio.music import MusicBackend, MusicManager, NullMusicBackend
from pyhexen.audio.sfx import SfxCache
from pyhexen.audio.sndseq_player import SoundSequencePlayer
from pyhexen.core.events import SequenceEvent, SoundEvent
from pyhexen.core.random import Random
from pyhexen.wad.sndinfo import load_sndinfo

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pyhexen.wad.mapinfo import MapInfoTable
    from pyhexen.wad.wadfile import WadFile, WadStack

log = logging.getLogger(__name__)

MIXER_RATE: int = 44100
MIXER_CHANNELS: int = 16


def init_pygame_mixer() -> bool:
    """Open the audio device. Returns False (and logs) when there is none."""
    try:
        import pygame.mixer

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=MIXER_RATE, size=-16, channels=2, buffer=1024)
        pygame.mixer.set_num_channels(MIXER_CHANNELS)
        return True
    except Exception as exc:  # no audio device, driver error, ...
        log.warning("Audio disabled: %s", exc)
        return False


class GameAudio:
    """Plays the game's sound effects, sound sequences and music."""

    def __init__(
        self,
        wad: WadStack | WadFile,
        mapinfo: MapInfoTable | None = None,
        *,
        mixer: Mixer | None = None,
        music_backend: MusicBackend | None = None,
        sfx_volume: float = 0.8,
        music_volume: float = 0.7,
    ) -> None:
        sndinfo = load_sndinfo(wad) if "SNDINFO" in wad else None
        curve = wad.get_bytes("SNDCURVE") if "SNDCURVE" in wad else None

        if mixer is None or music_backend is None:
            device = init_pygame_mixer()
            if mixer is None:
                mixer = (
                    PygameMixer(SfxCache(wad, sndinfo, target_rate=MIXER_RATE))
                    if device
                    else NullMixer()
                )
            if music_backend is None and not device:
                music_backend = NullMusicBackend()

        self.sounds = SoundSystem(sndinfo, curve=curve, mixer=mixer)
        self.sequences = (
            SoundSequencePlayer.from_wad(wad) if "SNDSEQ" in wad else SoundSequencePlayer()
        )
        self.sequences.sound_query = self.sounds.is_playing
        self.music = MusicManager(
            wad, backend=music_backend, mapinfo=mapinfo, sndinfo=sndinfo, volume=music_volume
        )
        self._rng = Random()
        self.set_volumes(sfx_volume, music_volume)

    def set_volumes(self, sfx: float, music: float) -> None:
        """Apply config volumes (0..1): sfx maps to snd_MaxVolume 0..15."""
        self.sounds.max_volume = max(0, min(15, round(sfx * 15)))
        if abs(self.music.volume - music) > 1e-6:
            self.music.set_volume(music)

    # -- events --------------------------------------------------------

    def handle_events(self, events: Iterable[Any]) -> None:
        """Apply the `SoundEvent`s and `SequenceEvent`s from one sim tic."""
        for event in events:
            if isinstance(event, SoundEvent):
                self.sounds.handle_event(event)
            elif isinstance(event, SequenceEvent):
                if event.stop:
                    self._play(self.sequences.stop_sequence(event.origin))
                else:
                    self.sequences.start_sequence(event.origin, event.sequence)

    def _play(self, events: Iterable[SoundEvent]) -> None:
        for event in events:
            self.sounds.handle_event(event)

    def tick(self, listener: Any = None) -> None:
        """One tic of S_UpdateSounds: run sequences, then follow moving sounds.

        `listener` is the console player's mobj, or None outside a level.
        """
        self.sounds.set_listener(listener)
        self._play(self.sequences.tick(self._rng))
        self.sounds.update()

    # -- levels and music ---------------------------------------------

    def start_level(self, map_num: int) -> None:
        """Level load: stop everything, then S_StartSong(map)."""
        self.stop_all()
        self.music.play_map_music(map_num)

    def stop_all(self) -> None:
        """S_StopAllSound + SN_StopAllSequences (no stop sounds)."""
        self.sequences.stop_all_sequences()
        self.sounds.stop_all()

    def play_music(self, lump: str) -> None:
        """S_StartSongName for the title, intermission and finale."""
        if self.music.current_lump != lump.upper():
            self.music.play_lump(lump)

    def ui_sound(self, name: str) -> None:
        """Menu and screen sounds, played at the listener."""
        self.sounds.start_sound(None, name)
