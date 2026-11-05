# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Scrolling wall textures and scrolling floor sector specials.

Port of scrolling lines (100..103) in P_SpawnSpecials (p_spec.c) and P_AnimateSurfaces (p_anim.c).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pyhexen.sim.thinker import Thinker

if TYPE_CHECKING:
    from pyhexen.sim.game_sim import GameSim
    from pyhexen.world.level import Line, Side

SCROLL_LEFT: int = 100
SCROLL_RIGHT: int = 101
SCROLL_UP: int = 102
SCROLL_DOWN: int = 103


class ScrollThinker(Thinker):
    """Per-tic wall texture offset scroller for line specials 100..103.

    Port of scrolling line updates in P_AnimateSurfaces (p_anim.c).
    """

    __slots__ = ("line", "side", "dx", "dy")

    def __init__(self, line: Line, side: Side, dx: int, dy: int) -> None:
        super().__init__()
        self.line: Line = line
        self.side: Side = side
        self.dx: int = dx
        self.dy: int = dy

    def tick(self, sim: GameSim) -> None:
        _ = sim
        self.side.texture_offset += self.dx
        self.side.row_offset += self.dy

    def state_tuple(self) -> tuple[Any, ...]:
        return (
            "ScrollThinker",
            self.line.index,
            self.side.texture_offset,
            self.side.row_offset,
        )


def spawn_line_scroller(sim: GameSim, line: Line) -> ScrollThinker | None:
    """Create and register a `ScrollThinker` if `line.special` is 100..103."""
    side = line.front_side
    if side is None:
        return None

    speed = line.args[0] << 10
    if line.special == SCROLL_LEFT:
        dx, dy = speed, 0
    elif line.special == SCROLL_RIGHT:
        dx, dy = -speed, 0
    elif line.special == SCROLL_UP:
        dx, dy = 0, speed
    elif line.special == SCROLL_DOWN:
        dx, dy = 0, -speed
    else:
        return None

    thinker = ScrollThinker(line=line, side=side, dx=dx, dy=dy)
    sim.thinkers.add(thinker)
    return thinker


def spawn_line_scrollers(sim: GameSim) -> list[ScrollThinker]:
    """Scan all lines in `sim.level` and spawn `ScrollThinker`s for specials 100..103."""
    if sim.level is None:
        return []
    scrollers: list[ScrollThinker] = []
    for line in sim.level.lines:
        if line.special in (SCROLL_LEFT, SCROLL_RIGHT, SCROLL_UP, SCROLL_DOWN):
            st = spawn_line_scroller(sim, line)
            if st is not None:
                scrollers.append(st)
    return scrollers


SPECIAL_HANDLERS: dict[int, Callable[..., bool]] = {}
