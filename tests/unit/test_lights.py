# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.4: light specials 110-116 against values worked out from p_lights.c.

The expected levels below come from vanilla's arithmetic by hand:
step = FixedDiv((dest - start) << 16, tics << 16), then every tic
light = ((light << 16) + step) >> 16 (the fraction is dropped each tic).
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

import pyhexen.sim
from pyhexen.core.random import Random
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.specials.dispatcher import execute_line_special
from pyhexen.sim.specials.lights import Light, LightType
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import Sector, load_level

TAG = 7


def _sim(light: int) -> tuple[GameSim, Sector]:
    wb = WadBuilder("IWAD")
    wb.add_playpal(2).add_colormap(33)
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms()
    wb.add_map(mb)
    level = load_level(WadStack(wb.to_bytes()), "MAP01")
    sec = level.sectors[1]
    sec.tag = TAG
    sec.light_level = light
    return GameSim(level=level), sec


def _lights(sim: GameSim) -> list[Light]:
    return [t for t in sim.thinkers if isinstance(t, Light)]


def _run(sim: GameSim, sec: Sector, tics: int) -> list[int]:
    levels = []
    for _ in range(tics):
        sim.tick()
        levels.append(sec.light_level)
    return levels


def test_fade_down_drops_the_fraction_every_tic() -> None:
    # 160 -> 120 over 35 tics: step = -40/35 = -1.142..., FixedDiv = -74898.
    # floor(L - 1.14) = L - 2 every tic, so the fade ends after 20 tics, not 35.
    sim, sec = _sim(160)
    assert execute_line_special(sim, 113, (TAG, 120, 35, 0, 0))
    light = _lights(sim)[0]
    assert (light.value1, light.value2, light.tics2) == (120, -74898, -1)

    levels = _run(sim, sec, 21)
    assert levels[:3] == [158, 156, 154]
    assert levels[18] == 122
    assert levels[19] == 120  # tic 20: reached, thinker removed
    assert levels[20] == 120
    assert _lights(sim) == []


def test_fade_up_with_step_below_one_never_brightens() -> None:
    # 100 -> 110 over 20 tics: step = 0.5 (32768); floor(L + 0.5) = L, so it stalls forever.
    sim, sec = _sim(100)
    execute_line_special(sim, 113, (TAG, 110, 20, 0, 0))
    assert _lights(sim)[0].value2 == 32768
    assert set(_run(sim, sec, 100)) == {100}
    assert len(_lights(sim)) == 1


def test_fade_up_with_whole_step() -> None:
    # 100 -> 160 over 20 tics: step = 3.0 exactly.
    sim, sec = _sim(100)
    execute_line_special(sim, 113, (TAG, 160, 20, 0, 0))
    levels = _run(sim, sec, 21)
    assert levels[0] == 103
    assert levels[19] == 160
    assert _lights(sim) == []


def test_fade_with_zero_tics_finishes_in_one_tic() -> None:
    # FixedDiv(x, 0) clamps to INT_MIN; the sum wraps far below the target.
    sim, sec = _sim(160)
    execute_line_special(sim, 113, (TAG, 50, 0, 0, 0))
    assert _run(sim, sec, 1) == [50]
    assert _lights(sim) == []


def test_glow_bounces_between_upper_and_lower() -> None:
    # Light_Glow(tag, upper=192, lower=64, tics=16) from 128: step = 64/16 = 4.0.
    sim, sec = _sim(128)
    execute_line_special(sim, 114, (TAG, 192, 64, 16, 0))
    levels = _run(sim, sec, 50)
    assert levels[0] == 132
    assert levels[15] == 192  # tic 16: top, reverse
    assert levels[16] == 188
    assert levels[47] == 64  # tic 48: bottom, reverse
    assert levels[48] == 68
    light = _lights(sim)[0]
    assert (light.tics1, light.tics2) == (4 << 16, 1)


def test_strobe_timing() -> None:
    # Light_Strobe(tag, upper=200, lower=50, upper tics=3, lower tics=5).
    sim, sec = _sim(128)
    execute_line_special(sim, 116, (TAG, 200, 50, 3, 5))
    assert sec.light_level == 200
    levels = _run(sim, sec, 13)
    # count=3 holds 3 tics, then lower for 1 + 5 tics, then upper again.
    assert levels == [200, 200, 200, 50, 50, 50, 50, 50, 50, 200, 200, 200, 200]


def test_flicker_uses_p_random_like_vanilla() -> None:
    sim, sec = _sim(128)
    ref = Random()
    ref.set_state(sim.rng.get_state())
    execute_line_special(sim, 115, (TAG, 180, 40, 0, 0))
    assert sec.light_level == 180

    # EV_SpawnLight: count = (P_Random() & 64) + 1, then T_Light alternates.
    expected: list[int] = []
    level, count = 180, (ref.p_random() & 64) + 1
    for _ in range(200):
        if count:
            count -= 1
        elif level == 180:
            level, count = 40, (ref.p_random() & 7) + 1
        else:
            level, count = 180, (ref.p_random() & 31) + 1
        expected.append(level)
    assert _run(sim, sec, 200) == expected
    assert sim.rng.get_state() == ref.get_state()


def test_raise_lower_change_clamp_without_thinkers() -> None:
    sim, sec = _sim(200)
    assert execute_line_special(sim, 110, (TAG, 100, 0, 0, 0))
    assert sec.light_level == 255
    assert execute_line_special(sim, 111, (TAG, 30, 0, 0, 0))
    assert sec.light_level == 225
    execute_line_special(sim, 111, (TAG, 255, 0, 0, 0))
    assert sec.light_level == 0
    execute_line_special(sim, 112, (TAG, 77, 0, 0, 0))
    assert sec.light_level == 77
    assert _lights(sim) == []
    assert not execute_line_special(sim, 112, (99, 77, 0, 0, 0))  # no sector with tag 99


def test_light_state_round_trip() -> None:
    sim, sec = _sim(128)
    light = Light(sec, LightType.LITE_GLOW, value1=1, value2=2, tics1=-3, tics2=-1, count=4)
    assert Light.from_state(sec, light.to_state()).state_tuple() == light.state_tuple()


def test_no_floats_in_the_simulation() -> None:
    """AGENTS.md determinism rule: no float literals or float()/round() calls under sim/."""
    root = Path(pyhexen.sim.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                offenders.append(f"{path.name}:{node.lineno} float literal")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("float", "round")
            ):
                offenders.append(f"{path.name}:{node.lineno} {node.func.id}()")
    assert offenders == []
    lights_src = (root / "specials" / "lights.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(lights_src)):
        assert not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div))
