# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Music playback backends (pygame MIDI, optional FluidSynth, and null) and MusicManager.

Port of S_StartSong, S_StartSongName, S_PauseSound, S_ResumeSound, and S_SetMusicVolume (s_sound.c).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

from pyhexen.audio.mus2mid import mus_to_midi
from pyhexen.wad.mapinfo import MapInfo, load_mapinfo
from pyhexen.wad.sndinfo import SndInfo, load_sndinfo
from pyhexen.wad.wadfile import WadFile, WadStack, normalize_lump_name


class MusicBackend(Protocol):
    """Protocol for pluggable MIDI music playback backends."""

    def play_midi(self, midi_data: bytes, *, loop: bool = True) -> bool: ...

    def stop(self) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def set_volume(self, volume: float) -> None: ...


class NullMusicBackend:
    """In-memory no-op music backend for headless execution and unit tests."""

    def __init__(self) -> None:
        self.current_midi: bytes | None = None
        self.loop: bool = False
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.volume: float = 0.7
        self.play_count: int = 0

    def play_midi(self, midi_data: bytes, *, loop: bool = True) -> bool:
        self.current_midi = midi_data
        self.loop = loop
        self.is_playing = True
        self.is_paused = False
        self.play_count += 1
        return True

    def stop(self) -> None:
        self.is_playing = False
        self.is_paused = False

    def pause(self) -> None:
        if self.is_playing:
            self.is_paused = True

    def resume(self) -> None:
        if self.is_playing:
            self.is_paused = False

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))


class PygameMidiBackend:
    """MIDI music backend powered by `pygame.mixer.music`."""

    def __init__(self, *, volume: float = 0.7) -> None:
        self.volume: float = max(0.0, min(1.0, float(volume)))
        self.is_playing: bool = False
        self.is_paused: bool = False

    def play_midi(self, midi_data: bytes, *, loop: bool = True) -> bool:
        try:
            import pygame.mixer

            if not pygame.mixer.get_init():
                return False
            pygame.mixer.music.load(io.BytesIO(midi_data))
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play(-1 if loop else 0)
            self.is_playing = True
            self.is_paused = False
            return True
        except Exception:
            self.is_playing = False
            return False

    def stop(self) -> None:
        self.is_playing = False
        self.is_paused = False
        try:
            import pygame.mixer

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def pause(self) -> None:
        if self.is_playing:
            self.is_paused = True
            try:
                import pygame.mixer

                if pygame.mixer.get_init():
                    pygame.mixer.music.pause()
            except Exception:
                pass

    def resume(self) -> None:
        if self.is_playing and self.is_paused:
            self.is_paused = False
            try:
                import pygame.mixer

                if pygame.mixer.get_init():
                    pygame.mixer.music.unpause()
            except Exception:
                pass

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))
        try:
            import pygame.mixer

            if pygame.mixer.get_init():
                pygame.mixer.music.set_volume(self.volume)
        except Exception:
            pass


class FluidSynthBackend:
    """Optional FluidSynth + SoundFont (.sf2) MIDI backend with graceful fallback."""

    def __init__(self, soundfont: str | Path = "", *, volume: float = 0.7) -> None:
        self.soundfont = Path(soundfont) if soundfont else None
        self.volume: float = max(0.0, min(1.0, float(volume)))
        self.is_playing: bool = False
        self.is_paused: bool = False
        self._synth: object | None = None
        self._fallback = PygameMidiBackend(volume=volume)

        if self.soundfont is not None and self.soundfont.is_file():
            try:
                import fluidsynth  # type: ignore[import-not-found]

                synth = fluidsynth.Synth()
                synth.start()
                sfid = synth.sfload(str(self.soundfont))
                synth.program_select(0, sfid, 0, 0)
                self._synth = synth
            except Exception:
                self._synth = None

    def play_midi(self, midi_data: bytes, *, loop: bool = True) -> bool:
        self.is_playing = True
        self.is_paused = False
        return self._fallback.play_midi(midi_data, loop=loop)

    def stop(self) -> None:
        self.is_playing = False
        self.is_paused = False
        self._fallback.stop()

    def pause(self) -> None:
        self.is_paused = True
        self._fallback.pause()

    def resume(self) -> None:
        self.is_paused = False
        self._fallback.resume()

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))
        self._fallback.set_volume(self.volume)


class MusicManager:
    """Manages map music resolution, MUS->MIDI conversion caching, and backend playback.

    Port of S_StartSong and S_StartSongName (s_sound.c).
    """

    def __init__(
        self,
        wad: WadStack | WadFile,
        *,
        backend: MusicBackend | None = None,
        mapinfo: dict[int, MapInfo] | None = None,
        sndinfo: SndInfo | None = None,
        volume: float = 0.7,
    ) -> None:
        self._wad = wad
        self.backend: MusicBackend = (
            backend if backend is not None else PygameMidiBackend(volume=volume)
        )
        self._sndinfo: SndInfo | None = (
            sndinfo if sndinfo is not None else (load_sndinfo(wad) if "SNDINFO" in wad else None)
        )
        self._mapinfo: dict[int, MapInfo] = (
            dict(mapinfo)
            if mapinfo is not None
            else (dict(load_mapinfo(wad)) if "MAPINFO" in wad else {})
        )
        self._midi_cache: dict[str, bytes] = {}
        self.current_map: int | None = None
        self.current_lump: str | None = None
        self.volume: float = max(0.0, min(1.0, float(volume)))
        self.backend.set_volume(self.volume)

    def resolve_map_music(self, map_num: int) -> str | None:
        """Resolve the music lump name for `map_num` from MAPINFO and SNDINFO.

        Port of P_GetMapSongLump (p_setup.c).
        """
        info = self._mapinfo.get(map_num)
        if info is not None and info.music_lump and info.music_lump.upper() != "DEFSONG":
            return normalize_lump_name(info.music_lump)
        if self._sndinfo is not None:
            song = self._sndinfo.map_music.get(map_num)
            if song and song.upper() != "DEFSONG":
                return normalize_lump_name(song)
        return None

    def get_midi_bytes(self, lump_name: str) -> bytes:
        """Convert (and cache) a MUS or MIDI lump from the WAD into Format 0 MIDI bytes."""
        norm = normalize_lump_name(lump_name)
        cached = self._midi_cache.get(norm)
        if cached is not None:
            return cached

        raw = self._wad.get_bytes(norm)
        if raw.startswith(b"MThd"):
            midi_bytes = raw
        else:
            midi_bytes = mus_to_midi(raw)

        self._midi_cache[norm] = midi_bytes
        return midi_bytes

    def play_map_music(self, map_num: int, *, loop: bool = True) -> bool:
        """Start playing the music track assigned to `map_num`.

        Port of S_StartSong (s_sound.c). Does not restart if `map_num` is already playing.
        """
        if map_num == self.current_map and self.current_lump is not None:
            return True

        song_lump = self.resolve_map_music(map_num)
        if not song_lump or song_lump not in self._wad:
            return False

        midi_bytes = self.get_midi_bytes(song_lump)
        self.backend.stop()
        ok = self.backend.play_midi(midi_bytes, loop=loop)
        self.current_map = map_num
        self.current_lump = song_lump
        return ok

    def play_lump(self, lump_name: str, *, loop: bool = True) -> bool:
        """Start playing a specific music lump by name (e.g. 'HEXEN', 'HUB', 'CHESS').

        Port of S_StartSongName (s_sound.c).
        """
        norm = normalize_lump_name(lump_name)
        if norm not in self._wad:
            return False

        midi_bytes = self.get_midi_bytes(norm)
        self.backend.stop()
        ok = self.backend.play_midi(midi_bytes, loop=loop)
        self.current_map = None
        self.current_lump = norm
        return ok

    def stop(self) -> None:
        """Stop current music playback."""
        self.backend.stop()
        self.current_map = None
        self.current_lump = None

    def pause(self) -> None:
        """Pause current music playback. Port of S_PauseSound (s_sound.c)."""
        self.backend.pause()

    def resume(self) -> None:
        """Resume paused music playback. Port of S_ResumeSound (s_sound.c)."""
        self.backend.resume()

    def set_volume(self, vol: float) -> None:
        """Set music volume (`0.0..1.0`). Port of S_SetMusicVolume (s_sound.c)."""
        self.volume = max(0.0, min(1.0, float(vol)))
        self.backend.set_volume(self.volume)
        if self.volume == 0.0:
            self.backend.pause()
        else:
            self.backend.resume()
