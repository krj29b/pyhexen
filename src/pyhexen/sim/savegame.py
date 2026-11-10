# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Save/load file management, serialization slots, and IWAD checksum validation (T13.4).

Port of SV_SaveGame, SV_LoadGame, SV_InitBaseSlot, CopySaveSlot in sv_save.c,
and save menu interaction in mn_menu.c.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyhexen.sim.hub import HubManager

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.ui.menu import MenuManager
    from pyhexen.wad.wadfile import WadStack
    from pyhexen.world.level import Level

SAVE_MAGIC: str = "PYHEXEN_SAVE"
SAVE_VERSION: int = 1
NUM_SAVE_SLOTS: int = 6
MAX_DESC_LEN: int = 32


class SaveError(Exception):
    """Base exception for save/load operations."""


class SaveVersionError(SaveError):
    """Raised when save file version is incompatible."""


class SaveChecksumError(SaveError):
    """Raised when IWAD checksum validation fails."""


def compute_wad_checksum(target: Path | bytes | WadStack) -> str:
    """Compute deterministic SHA-256 hex digest for IWAD verification.

    Accepts a Path, raw bytes, or a WadStack instance.
    """
    hasher = hashlib.sha256()

    if isinstance(target, Path):
        with open(target, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(target, (bytes, bytearray)):
        hasher.update(target)
    else:
        # WadStack instance
        if target.files and target.files[0].path is not None:
            with open(target.files[0].path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        else:
            # Hash lumps directory names and offsets
            summary = ",".join(f"{lump.name}:{lump.size}" for lump in target.lumps)
            hasher.update(summary.encode("utf-8"))

    return hasher.hexdigest()


@dataclass(slots=True)
class SaveHeader:
    """Metadata header for a PyHexen save game file."""

    magic: str
    version: int
    description: str
    timestamp: str
    iwad_checksum: str
    map_num: int
    cluster: int
    skill: int
    player_class: int
    level_time: int


@dataclass(slots=True)
class SaveSlotInfo:
    """Status and overview of a save slot (0..5)."""

    slot: int
    occupied: bool
    description: str = ""
    timestamp: str = ""
    map_num: int = 1
    path: Path | None = None


class SaveGameManager:
    """Manages save slots, file compression, versioning, and IWAD checksum validation."""

    def __init__(
        self,
        save_dir: Path | str = "saves",
        iwad_checksum: str = "",
    ) -> None:
        self.save_dir: Path = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.iwad_checksum: str = iwad_checksum

    def get_slot_path(self, slot: int) -> Path:
        """Return the filesystem path for a given slot index (0..5)."""
        return self.save_dir / f"save{slot}.hxs"

    def read_slot_header(self, slot: int) -> SaveHeader | None:
        """Inspect and parse header of a save slot without loading the full game."""
        path = self.get_slot_path(slot)
        if not path.is_file():
            return None

        try:
            with gzip.open(path, "rb") as f:
                decompressed = f.read()
            raw = json.loads(decompressed.decode("utf-8"))
            hdr = raw.get("header", {})
            return SaveHeader(
                magic=hdr.get("magic", ""),
                version=hdr.get("version", 0),
                description=hdr.get("description", ""),
                timestamp=hdr.get("timestamp", ""),
                iwad_checksum=hdr.get("iwad_checksum", ""),
                map_num=hdr.get("map_num", 1),
                cluster=hdr.get("cluster", 1),
                skill=hdr.get("skill", 2),
                player_class=hdr.get("player_class", 0),
                level_time=hdr.get("level_time", 0),
            )
        except Exception:
            return None

    def list_slots(self) -> list[SaveSlotInfo]:
        """Return status for all 6 save slots."""
        slots: list[SaveSlotInfo] = []
        for i in range(NUM_SAVE_SLOTS):
            hdr = self.read_slot_header(i)
            if hdr is not None and hdr.magic == SAVE_MAGIC:
                slots.append(
                    SaveSlotInfo(
                        slot=i,
                        occupied=True,
                        description=hdr.description,
                        timestamp=hdr.timestamp,
                        map_num=hdr.map_num,
                        path=self.get_slot_path(i),
                    )
                )
            else:
                slots.append(
                    SaveSlotInfo(
                        slot=i,
                        occupied=False,
                        description="",
                        timestamp="",
                        map_num=1,
                        path=self.get_slot_path(i),
                    )
                )
        return slots

    def save_game(
        self,
        slot: int,
        description: str,
        sim: GameSim,
        hub: HubManager | None = None,
        iwad_checksum: str | None = None,
    ) -> Path:
        """Save complete game state (active map + hub snapshots + players) into a slot.

        Port of SV_SaveGame (sv_save.c:1969).
        """
        if not (0 <= slot < NUM_SAVE_SLOTS):
            raise ValueError(f"Invalid slot {slot}; must be 0..{NUM_SAVE_SLOTS - 1}")

        clean_desc = description.strip()[:MAX_DESC_LEN]
        checksum = iwad_checksum if iwad_checksum is not None else self.iwad_checksum

        header = {
            "magic": SAVE_MAGIC,
            "version": SAVE_VERSION,
            "description": clean_desc,
            "timestamp": datetime.now(UTC).isoformat(),
            "iwad_checksum": checksum,
            "map_num": getattr(sim, "map_num", 1),
            "cluster": getattr(sim, "cluster", 1),
            "skill": sim.skill,
            "player_class": sim.player_class,
            "level_time": sim.level_time,
        }

        # Full simulation state (including players)
        sim_state = sim.to_state(save_players=True)

        # Hub persistence state
        hub_state: dict[str, Any] = {
            "current_cluster": hub.current_cluster if hub else getattr(sim, "cluster", 1),
            "world_vars": hub.world_vars if hub else (sim.acs_vm.world_vars if sim.acs_vm else []),
            "snapshots": {str(k): v.hex() for k, v in (hub.snapshots.items() if hub else [])},
        }

        payload = {
            "header": header,
            "sim": sim_state,
            "hub": hub_state,
        }

        raw_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(raw_json)

        dest_path = self.get_slot_path(slot)
        temp_path = dest_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            f.write(compressed)
        temp_path.replace(dest_path)

        return dest_path

    def load_game(
        self,
        slot: int,
        sim: GameSim,
        hub: HubManager | None = None,
        expected_checksum: str | None = None,
        load_level_cb: Callable[[int], Level] | None = None,
        validate_checksum: bool = True,
    ) -> tuple[GameSim, HubManager]:
        """Load and restore complete game state from a save slot.

        Port of SV_LoadGame (sv_save.c:2069).
        """
        path = self.get_slot_path(slot)
        if not path.is_file():
            raise FileNotFoundError(f"Save slot file {path} does not exist")

        with gzip.open(path, "rb") as f:
            decompressed = f.read()

        payload = json.loads(decompressed.decode("utf-8"))
        header = payload.get("header", {})

        # 1. Magic check
        if header.get("magic") != SAVE_MAGIC:
            raise SaveError("File is not a valid PyHexen save game")

        # 2. Version check
        if header.get("version") != SAVE_VERSION:
            raise SaveVersionError(
                f"Incompatible save version {header.get('version')}; expected {SAVE_VERSION}"
            )

        # 3. IWAD checksum validation
        chk = expected_checksum if expected_checksum is not None else self.iwad_checksum
        saved_chk = header.get("iwad_checksum", "")
        if validate_checksum and chk and saved_chk and chk != saved_chk:
            raise SaveChecksumError(
                f"IWAD checksum mismatch: saved with {saved_chk[:8]}..., current is {chk[:8]}..."
            )

        # 4. Level loading if needed
        target_map = header.get("map_num", 1)
        if load_level_cb is not None and (
            sim.level is None or getattr(sim, "map_num", 0) != target_map
        ):
            sim.level = load_level_cb(target_map)

        # 5. Restore simulation
        sim.from_state(payload["sim"])
        sim.map_num = target_map
        sim.cluster = header.get("cluster", 1)

        # 6. Restore HubManager
        active_hub = hub if hub is not None else HubManager()
        hub_data = payload.get("hub", {})
        active_hub.current_cluster = hub_data.get("current_cluster", sim.cluster)
        active_hub.world_vars = list(hub_data.get("world_vars", []))
        active_hub.snapshots = {
            int(k): bytes.fromhex(v) for k, v in hub_data.get("snapshots", {}).items()
        }

        # Apply world vars to ACS VM if present
        if sim.acs_vm is not None and active_hub.world_vars:
            sim.acs_vm.world_vars = list(active_hub.world_vars)

        return sim, active_hub

    def delete_slot(self, slot: int) -> bool:
        """Delete save file for a given slot."""
        path = self.get_slot_path(slot)
        if path.is_file():
            path.unlink()
            return True
        return False

    def sync_menu(
        self,
        menu: MenuManager,
        sim_getter: Callable[[], GameSim],
        hub_getter: Callable[[], HubManager],
        on_loaded_cb: Callable[[GameSim, HubManager], None] | None = None,
        load_level_cb: Callable[[int], Level] | None = None,
    ) -> None:
        """Synchronize MenuManager slot names and wire save/load callbacks."""
        slot_infos = self.list_slots()
        for info in slot_infos:
            menu.slot_texts[info.slot] = info.description
            menu.slot_status[info.slot] = info.occupied

        def _on_save(slot: int, description: str) -> None:
            sim = sim_getter()
            hub = hub_getter()
            self.save_game(slot, description, sim, hub)
            menu.slot_texts[slot] = description
            menu.slot_status[slot] = True

        def _on_load(slot: int) -> None:
            sim = sim_getter()
            hub = hub_getter()
            restored_sim, restored_hub = self.load_game(slot, sim, hub, load_level_cb=load_level_cb)
            if on_loaded_cb is not None:
                on_loaded_cb(restored_sim, restored_hub)

        menu.on_save_game = _on_save
        menu.on_load_game = _on_load
