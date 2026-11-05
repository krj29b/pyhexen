# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""MAPINFO lump parser.

Port of InitMapInfo and map info query functions in p_setup.c.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyhexen.wad.scanner import Scanner
from pyhexen.wad.sndinfo import parse_sndinfo
from pyhexen.wad.wadfile import WadFile, WadStack, normalize_lump_name

MAPINFO_SCRIPT_NAME = "MAPINFO"

_MAP_CMD_NAMES: tuple[str, ...] = (
    "SKY1",
    "SKY2",
    "DOUBLESKY",
    "LIGHTNING",
    "FADETABLE",
    "CLUSTER",
    "WARPTRANS",
    "NEXT",
    "SECRETNEXT",
    "SECRET",
    "CDTRACK",
    "MUSIC",
    "SONG",
    "CD_START_TRACK",
    "CD_END1_TRACK",
    "CD_END2_TRACK",
    "CD_END3_TRACK",
    "CD_INTERMISSION_TRACK",
    "CD_TITLE_TRACK",
)


@dataclass(slots=True)
class MapInfo:
    """Metadata for a single Hexen map from MAPINFO.

    Port of mapInfo_t (p_setup.c).
    """

    map_num: int
    name: str
    lump_name: str
    cluster: int = 1
    warp_trans: int = 0
    next_map: int = 1
    secret_map: int = 1
    sky1_texture: str = "SKY1"
    sky1_scroll_delta: int = 0
    sky2_texture: str = "SKY1"
    sky2_scroll_delta: int = 0
    double_sky: bool = False
    lightning: bool = False
    fadetable: str = "COLORMAP"
    cd_track: int = 0
    music_lump: str = ""


class MapInfoTable(dict[int, MapInfo]):
    """Dictionary mapping map number (int) to MapInfo, also supporting lookup by 'MAPxx' name."""

    def _resolve_key(self, key: int | str) -> int | None:
        if isinstance(key, int):
            return key
        if isinstance(key, str):
            norm = normalize_lump_name(key)
            if norm.startswith("MAP") and norm[3:].isdigit():
                return int(norm[3:])
            for num, info in self.items():
                if info.lump_name == norm:
                    return num
        return None

    def __contains__(self, key: object) -> bool:
        if isinstance(key, (int, str)):
            resolved = self._resolve_key(key)
            if resolved is not None:
                return super().__contains__(resolved)
        return False

    def __getitem__(self, key: int | str) -> MapInfo:
        resolved = self._resolve_key(key)
        if resolved is None:
            raise KeyError(key)
        return super().__getitem__(resolved)

    def get(self, key: int | str, default: MapInfo | None = None) -> MapInfo | None:  # type: ignore[override]
        resolved = self._resolve_key(key)
        if resolved is None:
            return default
        return super().get(resolved, default)

    def translate_warp(self, warp_num: int) -> int:
        """Return the actual map number for a given warp translation number, or -1 if not found.

        Port of P_TranslateMap (p_setup.c).
        """
        for map_num in sorted(self):
            if self[map_num].warp_trans == warp_num:
                return map_num
        return -1


def parse_mapinfo(text_or_bytes: str | bytes) -> list[MapInfo]:
    """Parse a MAPINFO script into a list of MapInfo objects.

    Port of InitMapInfo (p_setup.c).
    """
    sc = Scanner(text_or_bytes, name=MAPINFO_SCRIPT_NAME)
    by_num: dict[int, MapInfo] = {}
    order: list[int] = []

    while sc.get_string():
        if not sc.compare("MAP"):
            sc.script_error(None)

        map_num = sc.must_get_number()
        if map_num < 1 or map_num > 99:
            sc.script_error(None)

        map_name = sc.must_get_string()
        lump_name = f"MAP{map_num:02d}"

        info = MapInfo(
            map_num=map_num,
            name=map_name,
            lump_name=lump_name,
            cluster=0,
            warp_trans=map_num,
            next_map=1,
            secret_map=1,
            sky1_texture="SKY1",
            sky1_scroll_delta=0,
            sky2_texture="SKY1",
            sky2_scroll_delta=0,
            double_sky=False,
            lightning=False,
            fadetable="COLORMAP",
            cd_track=1,
            music_lump="",
        )

        while sc.get_string():
            if sc.compare("MAP"):
                sc.unget()
                break

            cmd_idx = sc.must_match_string(_MAP_CMD_NAMES)
            cmd = _MAP_CMD_NAMES[cmd_idx]

            if cmd == "CLUSTER":
                info.cluster = sc.must_get_number()
            elif cmd == "WARPTRANS":
                info.warp_trans = sc.must_get_number()
            elif cmd == "NEXT":
                info.next_map = sc.must_get_number()
            elif cmd in ("SECRETNEXT", "SECRET"):
                info.secret_map = sc.must_get_number()
            elif cmd == "CDTRACK":
                info.cd_track = sc.must_get_number()
            elif cmd == "SKY1":
                info.sky1_texture = normalize_lump_name(sc.must_get_string())
                info.sky1_scroll_delta = sc.must_get_number() << 8
            elif cmd == "SKY2":
                info.sky2_texture = normalize_lump_name(sc.must_get_string())
                info.sky2_scroll_delta = sc.must_get_number() << 8
            elif cmd == "DOUBLESKY":
                info.double_sky = True
            elif cmd == "LIGHTNING":
                info.lightning = True
            elif cmd == "FADETABLE":
                info.fadetable = normalize_lump_name(sc.must_get_string())
            elif cmd in ("MUSIC", "SONG"):
                info.music_lump = normalize_lump_name(sc.must_get_string())
            else:
                # CD_START_TRACK, CD_END1_TRACK, CD_END2_TRACK, CD_END3_TRACK,
                # CD_INTERMISSION_TRACK, CD_TITLE_TRACK
                sc.must_get_number()

        if map_num not in by_num:
            order.append(map_num)
        by_num[map_num] = info

    return [by_num[num] for num in order]


def load_mapinfo(wad: WadStack | WadFile) -> MapInfoTable:
    """Load and parse MAPINFO from a WAD or WadStack, merging map music from SNDINFO if present."""
    raw = wad.get_bytes(MAPINFO_SCRIPT_NAME)
    infos = parse_mapinfo(raw)

    table = MapInfoTable()
    for info in infos:
        table[info.map_num] = info

    if "SNDINFO" in wad:
        sndinfo = parse_sndinfo(wad.get_bytes("SNDINFO"))
        for map_num, song_lump in sndinfo.map_music.items():
            if map_num in table and not table[map_num].music_lump:
                table[map_num].music_lump = song_lump

    return table
