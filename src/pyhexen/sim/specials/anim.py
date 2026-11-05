# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Surface animations (ANIMDEFS flats and wall textures), sky scrolling, and lightning flashes.

Port of P_AnimateSurfaces, P_InitFTAnims, P_InitLightning, P_LightningFlash,
and P_ForceLightning (p_anim.c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyhexen.core.events import SectorChanged, SoundEvent
from pyhexen.sim.info_data import SfxEnum

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim

LIGHTNING_SPECIAL: int = 198
LIGHTNING_SPECIAL2: int = 199
SKYCHANGE_SPECIAL: int = 200


@dataclass(slots=True)
class AnimFrameDef:
    """Single animation frame definition `(frame_name, tics, rand_min, rand_max)`."""

    frame_name: str
    tics: int
    rand_min: int = 0
    rand_max: int = 0


@dataclass(slots=True)
class AnimSequenceState:
    """Runtime state of an animated flat or wall texture."""

    base_name: str
    is_flat: bool
    frames: list[AnimFrameDef]
    current_frame: int = 0
    tics: int = 1


@dataclass(slots=True)
class SurfaceAnimator:
    """Manages animated flats/textures, sky scrolling, and outdoor lightning flashes.

    Port of P_AnimateSurfaces and P_LightningFlash (p_anim.c).
    """

    sequences: list[AnimSequenceState] = field(default_factory=list)
    flat_translation: dict[str, str] = field(default_factory=dict)
    texture_translation: dict[str, str] = field(default_factory=dict)
    sky1_column_offset: int = 0
    sky2_column_offset: int = 0
    sky1_scroll_delta: int = 0
    sky2_scroll_delta: int = 0
    level_has_lightning: bool = False
    next_lightning_flash: int = 0
    lightning_flash: int = 0
    saved_light_levels: dict[int, int] = field(default_factory=dict)

    def add_sequence(
        self,
        base_name: str,
        frames: list[AnimFrameDef],
        *,
        is_flat: bool = True,
    ) -> None:
        """Register an animation sequence for `base_name`."""
        if not frames:
            return
        seq = AnimSequenceState(
            base_name=base_name.upper(),
            is_flat=is_flat,
            frames=frames,
            current_frame=len(frames) - 1,
            tics=1,
        )
        self.sequences.append(seq)
        if is_flat:
            self.flat_translation[seq.base_name] = frames[0].frame_name.upper()
        else:
            self.texture_translation[seq.base_name] = frames[0].frame_name.upper()

    def init_lightning(self, sim: GameSim, enabled: bool = False) -> None:
        """Port of P_InitLightning (p_anim.c)."""
        self.lightning_flash = 0
        self.saved_light_levels.clear()
        if not enabled or sim.level is None:
            self.level_has_lightning = False
            return

        has_sky_or_special = any(
            sec.ceiling_pic.startswith("F_SKY")
            or sec.special in (LIGHTNING_SPECIAL, LIGHTNING_SPECIAL2)
            for sec in sim.level.sectors
        )
        if not has_sky_or_special:
            self.level_has_lightning = False
            return

        self.level_has_lightning = True
        self.next_lightning_flash = ((sim.rng.p_random() & 15) + 5) * 35

    def force_lightning(self) -> None:
        """Port of P_ForceLightning (p_anim.c)."""
        self.next_lightning_flash = 0

    def _lightning_flash_step(self, sim: GameSim) -> None:
        """Port of P_LightningFlash (p_anim.c)."""
        if sim.level is None:
            return

        if self.lightning_flash > 0:
            self.lightning_flash -= 1
            if self.lightning_flash > 0:
                for sec in sim.level.sectors:
                    if sec.index in self.saved_light_levels:
                        orig = self.saved_light_levels[sec.index]
                        if orig < sec.light_level - 4:
                            sec.light_level -= 4
                            sim.events.emit(SectorChanged(sector_index=sec.index))
            else:
                for sec in sim.level.sectors:
                    if sec.index in self.saved_light_levels:
                        sec.light_level = self.saved_light_levels[sec.index]
                        sim.events.emit(SectorChanged(sector_index=sec.index))
                self.saved_light_levels.clear()
            return

        self.lightning_flash = (sim.rng.p_random() & 7) + 8
        flash_light = 200 + (sim.rng.p_random() & 31)
        found_sec = False

        for sec in sim.level.sectors:
            is_sky = sec.ceiling_pic.startswith("F_SKY")
            if is_sky or sec.special in (LIGHTNING_SPECIAL, LIGHTNING_SPECIAL2):
                orig = sec.light_level
                self.saved_light_levels[sec.index] = orig
                if sec.special == LIGHTNING_SPECIAL:
                    sec.light_level = min(orig + 64, flash_light)
                elif sec.special == LIGHTNING_SPECIAL2:
                    sec.light_level = min(orig + 32, flash_light)
                else:
                    sec.light_level = flash_light
                if sec.light_level < orig:
                    sec.light_level = orig
                sim.events.emit(SectorChanged(sector_index=sec.index))
                found_sec = True

        if found_sec:
            sim.events.emit(SoundEvent(sound_id=int(SfxEnum.SFX_THUNDER_CRASH)))

        if self.next_lightning_flash == 0:
            if sim.rng.p_random() < 50:
                self.next_lightning_flash = (sim.rng.p_random() & 15) + 16
            elif sim.rng.p_random() < 128 and not (sim.level_time & 32):
                self.next_lightning_flash = ((sim.rng.p_random() & 7) + 2) * 35
            else:
                self.next_lightning_flash = ((sim.rng.p_random() & 15) + 5) * 35

    def tick(self, sim: GameSim) -> None:
        """Port of P_AnimateSurfaces (p_anim.c)."""
        for seq in self.sequences:
            seq.tics -= 1
            if seq.tics <= 0:
                seq.current_frame = (seq.current_frame + 1) % len(seq.frames)
                frame = seq.frames[seq.current_frame]
                if frame.rand_max > frame.rand_min > 0:
                    mod = frame.rand_max - frame.rand_min + 1
                    seq.tics = frame.rand_min + (sim.rng.p_random() % mod)
                else:
                    seq.tics = max(1, frame.tics)

                target = frame.frame_name.upper()
                if seq.is_flat:
                    self.flat_translation[seq.base_name] = target
                else:
                    self.texture_translation[seq.base_name] = target

        self.sky1_column_offset += self.sky1_scroll_delta
        self.sky2_column_offset += self.sky2_scroll_delta

        if self.level_has_lightning:
            if self.next_lightning_flash == 0 or self.lightning_flash > 0:
                self._lightning_flash_step(sim)
            else:
                self.next_lightning_flash -= 1


def p_animate_surfaces(sim: GameSim) -> None:
    """Port of P_AnimateSurfaces (p_anim.c)."""
    if sim.surface_animator is not None:
        sim.surface_animator.tick(sim)
