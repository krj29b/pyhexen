# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""T15.6: demo trace format, mobj hash, divergence finder and check_determinism."""

from __future__ import annotations

from dataclasses import replace

from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.ticcmd import TicCmd
from pyhexen.sim.demo import Demo, DemoHeader, DemoPlayer, play_demo_to_completion
from pyhexen.sim.demo_trace import (
    MobjDump,
    TraceTic,
    dump_mobjs,
    first_divergence,
    mobj_hash,
    parse_trace,
    trace_tic,
)
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjType
from pyhexen.sim.mobj import Mobj
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level


def _sim() -> GameSim:
    mb = MapBuilder("MAP01")
    mb.add_room(x=0, y=0, width=512, height=512)
    mb.add_thing(x=256, y=256, type=1)
    mb.add_thing(x=100, y=100, type=10030)  # MT_ETTIN
    level = load_level(WadStack(WadBuilder("IWAD").add_map(mb).to_bytes()), "MAP01")
    sim = GameSim(level=level)
    sim.setup_level(player_nums=(0,))
    return sim


def _tic(level_time: int, **changes: int) -> TraceTic:
    base = TraceTic(
        level_time, 10, 1 << 16, -(2 << 16), 0, 0x80000000, 5, -5, 0, 100, 3, 0xDEADBEEF
    )
    return replace(base, **changes)


def test_trace_round_trips_through_text() -> None:
    tics = [_tic(1), _tic(2, x=-7, mobj_hash=0x0000ABCD)]
    dump = [MobjDump(1, 2, 3, 4, -5, 6, 7, 0xFFFFFFFF, 100, 42, -1, 0x10)]
    text = tics[0].format() + "\n" + dump[0].format() + "\n" + tics[1].format() + "\n"
    assert "0000abcd" in text  # hash is 8 hex digits, like %08x
    trace = parse_trace(text)
    assert trace.tics == tics
    assert trace.dumps == {2: dump}  # dumps belong to the T line that follows them


def test_parse_rejects_garbage() -> None:
    try:
        parse_trace("T 1 2 3\n")
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_mobj_hash_wraps_like_unsigned_c() -> None:
    a = Mobj(x=-(100 << 16), y=200 << 16, z=0, type=5, health=-3, state_num=7)
    b = Mobj(x=1, y=2, z=3, type=4, health=5, state_num=6)
    count, h = mobj_hash([a, b])
    mask = 0xFFFFFFFF
    term_a = (a.x + 3 * a.y + 7 * a.z + 11 * 5 + 13 * -3 + 17 * 7) & mask
    term_b = (1 + 6 + 21 + 44 + 65 + 102) & mask
    expected = ((((0 * 1000003) & mask) ^ term_a) * 1000003 & mask) ^ term_b
    assert (count, h) == (2, expected)
    assert 0 <= h <= mask


def test_trace_tic_reads_player_and_skips_removed_mobjs() -> None:
    sim = _sim()
    sim.tick([TicCmd()])
    tic = trace_tic(sim)
    player = sim.players[0]
    assert player.mo is not None
    assert (tic.level_time, tic.x, tic.y, tic.health) == (1, player.mo.x, player.mo.y, 100)
    assert tic.prndindex == sim.rng.prndindex
    count = tic.mobj_count
    assert count == len(dump_mobjs(sim))

    ettin = next(m for m in dump_mobjs(sim) if m.type == MobjType.MT_ETTIN)
    assert ettin.health > 0
    victim = next(t for t in sim.thinkers if isinstance(t, Mobj) and t.type == MobjType.MT_ETTIN)
    victim.removed = True  # vanilla: function == -1, not a P_MobjThinker any more
    assert trace_tic(sim).mobj_count == count - 1


def test_first_divergence_reports_tic_and_fields() -> None:
    ref = [_tic(1), _tic(2), _tic(3)]
    assert first_divergence(ref, list(ref)) is None

    ours = [_tic(1), _tic(2, prndindex=11, momx=0), _tic(3)]
    div = first_divergence(ref, ours)
    assert div is not None
    assert div.index == 1 and div.fields == ("prndindex", "momx")
    assert div.describe().startswith("tic 2: prndindex 10 != 11")

    short = first_divergence(ref, ref[:2])
    assert short is not None and short.index == 2 and short.ours is None
    assert "our trace ended first" in short.describe()


def test_empty_state_hash_is_not_a_pass() -> None:
    """Audit §4: vanilla demos carry no hash; that used to count as "determinism passed"."""
    sim = _sim()
    demo = Demo(header=DemoHeader(final_state_hash=""), tics=[(TicCmd(),)] * 3)
    player = DemoPlayer(demo)
    assert not player.has_state_hash
    assert play_demo_to_completion(sim, player) is False
    assert player.is_finished

    hashed = DemoPlayer(Demo(header=DemoHeader(final_state_hash=sim.state_hash()), tics=[]))
    assert hashed.has_state_hash and hashed.check_determinism(sim)
