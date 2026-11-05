# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Generator for sim/info_data.py from Chocolate Doom's src/hexen/info.h and info.c."""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
REF_HEXEN = ROOT_DIR / "reference" / "src" / "hexen"
OUT_FILE = ROOT_DIR / "src" / "pyhexen" / "sim" / "info_data.py"

FRACUNIT = 1 << 16

MOBJINFO_FIELDS_24 = (
    "doomednum",
    "spawnstate",
    "spawnhealth",
    "seestate",
    "seesound",
    "reactiontime",
    "attacksound",
    "painstate",
    "painchance",
    "painsound",
    "meleestate",
    "missilestate",
    "crashstate",
    "deathstate",
    "xdeathstate",
    "deathsound",
    "speed",
    "radius",
    "height",
    "mass",
    "damage",
    "activesound",
    "flags",
    "flags2",
)

MOBJINFO_FIELDS_26 = (
    "doomednum",
    "spawnstate",
    "spawnhealth",
    "seestate",
    "seesound",
    "reactiontime",
    "attacksound",
    "painstate",
    "painchance",
    "painsound",
    "meleestate",
    "missilestate",
    "crashstate",
    "deathstate",
    "xdeathstate",
    "burnstate",
    "iceshatterstate",
    "deathsound",
    "speed",
    "radius",
    "height",
    "mass",
    "damage",
    "activesound",
    "flags",
    "flags2",
)


def strip_c_comments(text: str) -> str:
    """Remove /* ... */ and // ... comments from C source."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def parse_enum(text: str, typedef_name: str) -> list[tuple[str, int]]:
    """Extract (name, value) pairs from `typedef enum { ... } <typedef_name>;`."""
    pattern = rf"typedef\s+enum\s*\{{([^}}]*)\}}\s*{re.escape(typedef_name)}\s*;"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find enum {typedef_name}")
    body = match.group(1)
    items: list[tuple[str, int]] = []
    values_by_name: dict[str, int] = {}
    next_val = 0
    for raw_item in body.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            name_part, val_part = item.split("=", 1)
            name = name_part.strip()
            val_str = val_part.strip()
            if val_str in values_by_name:
                val = values_by_name[val_str]
            else:
                val = int(val_str, 0)
            next_val = val + 1
        else:
            name = item
            val = next_val
            next_val += 1
        items.append((name, val))
        values_by_name[name] = val
    return items


def parse_defines(text: str, prefix: str) -> list[tuple[str, int]]:
    """Extract `#define <PREFIX>_* <value>` constants."""
    results: list[tuple[str, int]] = []
    env: dict[str, int] = {
        "FRACUNIT": FRACUNIT,
        "INT_MAX": 0x7FFFFFFF,
        "MAXINT": 0x7FFFFFFF,
    }
    for line in text.splitlines():
        line = line.strip()
        m = re.match(rf"^#define\s+({re.escape(prefix)}\w+)\s+(.+)$", line)
        if m:
            name, expr = m.group(1), m.group(2).strip()
            val = eval_c_expr(expr, env)
            env[name] = val
            results.append((name, val))
    return results


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}


def eval_c_expr(expr: str, env: dict[str, int]) -> int:
    """Safely evaluate a C integer constant expression using AST."""
    tree = ast.parse(expr.strip(), mode="eval")

    def _eval(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise KeyError(f"Unknown identifier in C expression: {node.id!r}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported AST node in {expr!r}: {ast.dump(node)}")

    return _eval(tree)


def split_brace_blocks(body: str) -> list[list[str]]:
    """Extract top-level `{ ... }` initializer lists and split their comma-separated fields."""
    blocks: list[list[str]] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "{":
            if depth == 1:
                current = []
            depth += 1
            if depth > 2:
                current.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 1:
                block_str = "".join(current)
                fields = [f.strip() for f in block_str.split(",") if f.strip()]
                blocks.append(fields)
            elif depth > 1:
                current.append(ch)
        else:
            if depth >= 2:
                current.append(ch)
    return blocks


def parse_sprnames(info_c_clean: str) -> list[str]:
    """Parse `const char *sprnames[] = { ... };`."""
    m = re.search(
        r"const\s+char\s*\*\s*sprnames\s*\[\s*\]\s*=\s*\{(.*?)\}\s*;",
        info_c_clean,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("Could not find sprnames[] in info.c")
    body = m.group(1)
    names: list[str] = []
    for raw in body.split(","):
        token = raw.strip()
        if not token or token == "NULL":
            continue
        if token.startswith('"') and token.endswith('"'):
            names.append(token[1:-1])
        else:
            raise ValueError(f"Unexpected token in sprnames: {token!r}")
    return names


def parse_states(
    info_c_clean: str, env: dict[str, int]
) -> list[tuple[int, int, int, str | None, int, int, int]]:
    """Parse `state_t states[NUMSTATES] = { ... };`."""
    m = re.search(
        r"state_t\s+states\s*\[\s*NUMSTATES\s*\]\s*=\s*(\{.*?\})\s*;",
        info_c_clean,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("Could not find states[NUMSTATES] in info.c")
    raw_blocks = split_brace_blocks(m.group(1))
    states: list[tuple[int, int, int, str | None, int, int, int]] = []
    for idx, fields in enumerate(raw_blocks):
        if len(fields) != 7:
            raise ValueError(f"State {idx} has {len(fields)} fields, expected 7: {fields}")
        sprite = eval_c_expr(fields[0], env)
        frame = eval_c_expr(fields[1], env)
        tics = eval_c_expr(fields[2], env)
        action_raw = fields[3].strip()
        action_name = None if action_raw in ("NULL", "0") else action_raw
        next_state = eval_c_expr(fields[4], env)
        misc1 = eval_c_expr(fields[5], env)
        misc2 = eval_c_expr(fields[6], env)
        states.append((sprite, frame, tics, action_name, next_state, misc1, misc2))
    return states


def parse_mobjinfo(info_c_clean: str, env: dict[str, int]) -> list[dict[str, int]]:
    """Parse `mobjinfo_t mobjinfo[NUMMOBJTYPES] = { ... };`."""
    m = re.search(
        r"mobjinfo_t\s+mobjinfo\s*\[\s*NUMMOBJTYPES\s*\]\s*=\s*(\{.*?\})\s*;",
        info_c_clean,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("Could not find mobjinfo[NUMMOBJTYPES] in info.c")
    raw_blocks = split_brace_blocks(m.group(1))
    entries: list[dict[str, int]] = []
    for idx, fields in enumerate(raw_blocks):
        if len(fields) == 24:
            vals = [eval_c_expr(f, env) for f in fields]
            entry = dict(zip(MOBJINFO_FIELDS_24, vals, strict=True))
            entry["burnstate"] = env["S_NULL"]
            entry["iceshatterstate"] = env["S_NULL"]
        elif len(fields) == 26:
            vals = [eval_c_expr(f, env) for f in fields]
            entry = dict(zip(MOBJINFO_FIELDS_26, vals, strict=True))
        else:
            raise ValueError(
                f"mobjinfo[{idx}] has {len(fields)} fields, expected 24 or 26: {fields}"
            )
        entries.append(entry)
    return entries


def generate_info_data() -> str:
    """Parse reference headers/sources and build the Python source for info_data.py."""
    info_h = strip_c_comments((REF_HEXEN / "info.h").read_text(encoding="utf-8"))
    info_c = strip_c_comments((REF_HEXEN / "info.c").read_text(encoding="utf-8"))
    h2def_h = strip_c_comments((REF_HEXEN / "h2def.h").read_text(encoding="utf-8"))
    sounds_h = strip_c_comments((REF_HEXEN / "sounds.h").read_text(encoding="utf-8"))

    sprites = parse_enum(info_h, "spritenum_t")
    state_enums = parse_enum(info_h, "statenum_t")
    mobj_types = parse_enum(info_h, "mobjtype_t")
    sfx_enums = parse_enum(sounds_h, "sfxenum_t")

    mf_flags = [
        (name, val) for name, val in parse_defines(h2def_h, "MF_") if name != "MF_TRANSSHIFT"
    ]
    mf2_flags = parse_defines(h2def_h, "MF2_")

    env: dict[str, int] = {
        "FRACUNIT": FRACUNIT,
        "INT_MAX": 0x7FFFFFFF,
        "MAXINT": 0x7FFFFFFF,
    }
    for items in (sprites, state_enums, mobj_types, sfx_enums, mf_flags, mf2_flags):
        for name, val in items:
            env[name] = val

    sprnames = parse_sprnames(info_c)
    states = parse_states(info_c, env)
    mobjinfo = parse_mobjinfo(info_c, env)

    num_sprites = dict(sprites)["NUMSPRITES"]
    num_states = dict(state_enums)["NUMSTATES"]
    num_mobjtypes = dict(mobj_types)["NUMMOBJTYPES"]

    if len(sprnames) != num_sprites:
        raise ValueError(f"len(sprnames)={len(sprnames)} != NUMSPRITES={num_sprites}")
    if len(states) != num_states:
        raise ValueError(f"len(states)={len(states)} != NUMSTATES={num_states}")
    if len(mobjinfo) != num_mobjtypes:
        raise ValueError(f"len(mobjinfo)={len(mobjinfo)} != NUMMOBJTYPES={num_mobjtypes}")

    lines: list[str] = [
        "# PyHexen - Python reimplementation of the Hexen engine.",
        "# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,",
        "# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.",
        "",
        '"""Auto-generated actor info tables from Chocolate Doom src/hexen/info.h and info.c.',
        "",
        "Generated by tools/gen_info.py. Do not edit manually.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from enum import IntEnum, IntFlag",
        "",
        "",
        "class SpriteNum(IntEnum):",
        '    """Port of spritenum_t (info.h)."""',
        "",
    ]
    for name, val in sprites:
        lines.append(f"    {name} = {val}")

    lines.extend(
        [
            "",
            "",
            "class StateNum(IntEnum):",
            '    """Port of statenum_t (info.h)."""',
            "",
        ]
    )
    for name, val in state_enums:
        lines.append(f"    {name} = {val}")

    lines.extend(
        [
            "",
            "",
            "class MobjType(IntEnum):",
            '    """Port of mobjtype_t (info.h)."""',
            "",
        ]
    )
    for name, val in mobj_types:
        lines.append(f"    {name} = {val}")

    lines.extend(
        [
            "",
            "",
            "class SfxEnum(IntEnum):",
            '    """Port of sfxenum_t (sounds.h)."""',
            "",
        ]
    )
    for name, val in sfx_enums:
        lines.append(f"    {name} = {val}")

    lines.extend(
        [
            "",
            "",
            "class MobjFlag(IntFlag):",
            '    """Port of MF_* flags (h2def.h)."""',
            "",
        ]
    )
    for name, val in mf_flags:
        lines.append(f"    {name} = 0x{val:08X}")

    lines.append("")
    lines.append("MF_TRANSSHIFT: int = 26")

    lines.extend(
        [
            "",
            "",
            "class MobjFlag2(IntFlag):",
            '    """Port of MF2_* flags (h2def.h)."""',
            "",
        ]
    )
    for name, val in mf2_flags:
        lines.append(f"    {name} = 0x{val:08X}")

    lines.extend(
        [
            "",
            "",
            "@dataclass(slots=True, frozen=True)",
            "class State:",
            '    """Port of state_t (info.h)."""',
            "",
            "    sprite: int",
            "    frame: int",
            "    tics: int",
            "    action_name: str | None",
            "    next_state: int",
            "    misc1: int",
            "    misc2: int",
            "",
            "",
            "@dataclass(slots=True, frozen=True)",
            "class MobjInfo:",
            '    """Port of mobjinfo_t (info.h)."""',
            "",
            "    doomednum: int",
            "    spawnstate: int",
            "    spawnhealth: int",
            "    seestate: int",
            "    seesound: int",
            "    reactiontime: int",
            "    attacksound: int",
            "    painstate: int",
            "    painchance: int",
            "    painsound: int",
            "    meleestate: int",
            "    missilestate: int",
            "    crashstate: int",
            "    deathstate: int",
            "    xdeathstate: int",
            "    deathsound: int",
            "    speed: int",
            "    radius: int",
            "    height: int",
            "    mass: int",
            "    damage: int",
            "    activesound: int",
            "    flags: int",
            "    flags2: int",
            "    burnstate: int = 0",
            "    iceshatterstate: int = 0",
            "",
            "",
            "SPRNAMES: tuple[str, ...] = (",
        ]
    )
    for name in sprnames:
        lines.append(f'    "{name}",')
    lines.append(")")

    lines.extend(
        [
            "",
            "STATES: tuple[State, ...] = (",
        ]
    )
    for sprite, frame, tics, action_name, next_state, misc1, misc2 in states:
        act_repr = f'"{action_name}"' if action_name is not None else "None"
        lines.append(
            f"    State({sprite}, {frame}, {tics}, {act_repr}, {next_state}, {misc1}, {misc2}),"
        )
    lines.append(")")

    lines.extend(
        [
            "",
            "MOBJINFO: tuple[MobjInfo, ...] = (",
        ]
    )
    for entry in mobjinfo:
        args_24 = ", ".join(str(entry[k]) for k in MOBJINFO_FIELDS_24)
        lines.append(f"    MobjInfo({args_24}),")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = generate_info_data()
    OUT_FILE.write_text(content, encoding="utf-8")
    print(f"Generated {OUT_FILE}")


if __name__ == "__main__":
    main()
