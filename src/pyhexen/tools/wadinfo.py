# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""CLI utility for inspecting WAD directories, MAPINFO tables, and exporting lumps/PNGs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyhexen.app.config import resolve_iwad
from pyhexen.wad.flats import FlatRegistry
from pyhexen.wad.mapinfo import load_mapinfo
from pyhexen.wad.namespaces import flat_lumps, patch_lumps, sprite_lumps
from pyhexen.wad.palette import apply_palette, load_colormap, load_playpal, save_png
from pyhexen.wad.picture import decode_picture
from pyhexen.wad.textures import TextureRegistry
from pyhexen.wad.wadfile import WadStack, normalize_lump_name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyhexen.tools.wadinfo",
        description="Inspect Hexen WAD files, list maps, extract lumps, and export PNG assets.",
    )
    parser.add_argument(
        "--iwad",
        type=str,
        default=None,
        help="Path to HEXEN.WAD (defaults to PYHEXEN_IWAD or config.toml)",
    )
    parser.add_argument(
        "--file",
        "-f",
        nargs="*",
        default=[],
        help="Optional PWAD file(s) to merge on top of the IWAD",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List all lumps and summary asset counts")

    # extract
    extract_p = subparsers.add_parser("extract", help="Extract raw bytes of a lump")
    extract_p.add_argument("lump", type=str, help="Lump name to extract")
    extract_p.add_argument(
        "--out",
        type=str,
        default="out",
        help="Output directory or file path (default: out/<LUMP>.lmp)",
    )

    # export-png
    export_p = subparsers.add_parser(
        "export-png",
        help="Export texture, flat, sprite, patch, palette, or colormap as PNG",
    )
    export_p.add_argument(
        "kind",
        choices=("texture", "flat", "sprite", "patch", "palette", "colormap"),
        help="Asset category to export",
    )
    export_p.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Specific asset name to export (omit or pass --all to export all of that kind)",
    )
    export_p.add_argument(
        "--all",
        action="store_true",
        help="Export all assets of the given kind",
    )
    export_p.add_argument(
        "--out",
        type=str,
        default="out",
        help="Output directory for PNG files (default: out/)",
    )

    # maps
    subparsers.add_parser("maps", help="Print the MAPINFO table")

    return parser


def _cmd_list(wad: WadStack) -> int:
    print(f"{'IDX':>5}  {'NAME':<8}  {'SIZE':>10}  {'OFFSET':>10}  {'SOURCE'}")
    print("-" * 52)
    for lump in wad.lumps:
        print(f"{lump.index:>5}  {lump.name:<8}  {lump.size:>10}  {lump.offset:>10}  {lump.source}")

    num_sprites = len(sprite_lumps(wad, deduplicate=True))
    num_flats = len(flat_lumps(wad, deduplicate=True))
    num_patches = len(patch_lumps(wad, deduplicate=True))
    if "PNAMES" in wad and "TEXTURE1" in wad:
        num_textures = max(0, len(TextureRegistry(wad)) - 1)
    else:
        num_textures = 0

    print("-" * 52)
    print(
        f"Total lumps: {len(wad)} | Sprites: {num_sprites} | Flats: {num_flats} | "
        f"Patches: {num_patches} | Textures: {num_textures}"
    )
    return 0


def _cmd_extract(wad: WadStack, lump_name: str, out_arg: str) -> int:
    norm = normalize_lump_name(lump_name)
    data = wad.get_bytes(norm)
    out_path = Path(out_arg)

    if out_path.is_dir() or out_arg.endswith(("/", "\\")) or out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
        target_file = out_path / f"{norm}.lmp"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        target_file = out_path

    target_file.write_bytes(data)
    print(f"Extracted {norm} ({len(data)} bytes) -> {target_file}")
    return 0


def _render_palette_grid(playpal: np.ndarray) -> np.ndarray:
    """Render all palettes in PLAYPAL `(N, 256, 3)` as a grid image `(N * 16, 16, 3)`."""
    num_pals = playpal.shape[0]
    return playpal.reshape(num_pals * 16, 16, 3)


def _cmd_export_png(
    wad: WadStack,
    kind: str,
    name: str | None,
    export_all: bool,
    out_arg: str,
) -> int:
    out_dir = Path(out_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = load_playpal(wad)

    if kind == "palette":
        grid = _render_palette_grid(palette)
        target_name = normalize_lump_name(name) if name else "PLAYPAL"
        out_file = out_dir / f"{target_name}.png"
        save_png(grid, out_file)
        print(f"Exported palette ({palette.shape[0]} palettes) -> {out_file}")
        return 0

    if kind == "colormap":
        lump_name = normalize_lump_name(name) if name else "COLORMAP"
        cmap = load_colormap(wad, lump_name)
        rgba = apply_palette(cmap, palette)
        out_file = out_dir / f"{lump_name}.png"
        save_png(rgba, out_file)
        print(f"Exported colormap {lump_name} ({cmap.shape[0]} rows) -> {out_file}")
        return 0

    if kind == "texture":
        tex_reg = TextureRegistry(wad)
        if name and not export_all:
            tex = tex_reg.get(name)
            out_file = out_dir / f"{tex.name}.png"
            save_png(tex.indices, out_file, palette=palette, mask=tex.mask)
            print(f"Exported texture {tex.name} ({tex.width}x{tex.height}) -> {out_file}")
        else:
            count = 0
            for idx in range(1, len(tex_reg)):
                tex = tex_reg.get(idx)
                out_file = out_dir / f"{tex.name}.png"
                save_png(tex.indices, out_file, palette=palette, mask=tex.mask)
                count += 1
            print(f"Exported {count} textures -> {out_dir}")
        return 0

    if kind == "flat":
        flat_reg = FlatRegistry(wad)
        if name and not export_all:
            flat = flat_reg.get(name)
            out_file = out_dir / f"{flat.name}.png"
            save_png(flat.indices, out_file, palette=palette)
            print(f"Exported flat {flat.name} -> {out_file}")
        else:
            count = 0
            for flat in flat_reg:
                out_file = out_dir / f"{flat.name}.png"
                save_png(flat.indices, out_file, palette=palette)
                count += 1
            print(f"Exported {count} flats -> {out_dir}")
        return 0

    if kind in ("sprite", "patch"):
        if name and not export_all:
            norm = normalize_lump_name(name)
            pic = decode_picture(wad.get_bytes(norm))
            out_file = out_dir / f"{norm}.png"
            save_png(pic.indices, out_file, palette=palette, mask=pic.mask)
            print(f"Exported {kind} {norm} ({pic.width}x{pic.height}) -> {out_file}")
        else:
            lumps = (
                sprite_lumps(wad, deduplicate=True)
                if kind == "sprite"
                else patch_lumps(wad, deduplicate=True)
            )
            count = 0
            for lump in lumps:
                pic = decode_picture(wad.get_bytes(lump))
                out_file = out_dir / f"{lump.name}.png"
                save_png(pic.indices, out_file, palette=palette, mask=pic.mask)
                count += 1
            print(f"Exported {count} {kind}s -> {out_dir}")
        return 0

    raise ValueError(f"Unknown export-png kind: {kind!r}")


def _cmd_maps(wad: WadStack) -> int:
    table = load_mapinfo(wad)
    header = (
        f"{'MAP':>3}  {'LUMP':<6}  {'TITLE':<28}  {'CL':>2}  {'WARP':>4}  "
        f"{'NEXT':>4}  {'SECR':>4}  {'SKY1':<8}  {'SKY2':<8}  {'LGHT':<4}  {'MUSIC':<8}"
    )
    print(header)
    print("-" * len(header))
    for map_num in sorted(table):
        info = table[map_num]
        lght = "yes" if info.lightning else "no"
        print(
            f"{info.map_num:>3}  {info.lump_name:<6}  {info.name:<28.28}  "
            f"{info.cluster:>2}  {info.warp_trans:>4}  {info.next_map:>4}  "
            f"{info.secret_map:>4}  {info.sky1_texture:<8}  {info.sky2_texture:<8}  "
            f"{lght:<4}  {info.music_lump:<8}"
        )
    print("-" * len(header))
    print(f"Total maps: {len(table)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    iwad_path = resolve_iwad(args.iwad, must_exist=True)
    if iwad_path is None:
        print(
            "Error: No HEXEN.WAD found. Specify --iwad, PYHEXEN_IWAD, or config.toml.",
            file=sys.stderr,
        )
        return 1

    with WadStack(iwad_path) as wad:
        for pwad in args.file:
            wad.add_wad(pwad)

        if args.command == "list":
            return _cmd_list(wad)
        if args.command == "extract":
            return _cmd_extract(wad, args.lump, args.out)
        if args.command == "export-png":
            return _cmd_export_png(wad, args.kind, args.name, args.all, args.out)
        if args.command == "maps":
            return _cmd_maps(wad)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
