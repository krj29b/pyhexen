# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for deterministic RNG (m_random.c)."""

from __future__ import annotations

from pyhexen.core.random import RNDTABLE, Random


def test_rndtable_properties() -> None:
    assert len(RNDTABLE) == 256
    # Hexen's flat distribution table contains every byte 0..255 exactly once
    assert sorted(RNDTABLE) == list(range(256))
    assert RNDTABLE[0] == 201
    assert RNDTABLE[1] == 1
    assert RNDTABLE[255] == 31


def test_first_512_values_match_reference() -> None:
    rng = Random()
    rng.clear()

    # Reference C logic:
    # prndindex = (prndindex + 1) & 0xff; return rndtable[prndindex];
    expected = [RNDTABLE[(i + 1) & 0xFF] for i in range(512)]
    actual_p = [rng.p_random() for _ in range(512)]
    actual_m = [rng.m_random() for _ in range(512)]

    assert actual_p == expected
    assert actual_m == expected
    assert actual_p[0] == 1
    assert actual_p[254] == 31
    assert actual_p[255] == 201
    assert actual_p[256] == 1


def test_separate_indices_and_subrandom() -> None:
    rng = Random()
    assert rng.p_random() == RNDTABLE[1]
    assert rng.prndindex == 1
    assert rng.rndindex == 0

    assert rng.m_random() == RNDTABLE[1]
    assert rng.prndindex == 1
    assert rng.rndindex == 1

    # P_SubRandom consumes two P_Random values: RNDTABLE[2] - RNDTABLE[3]
    sub = rng.p_subrandom()
    assert sub == RNDTABLE[2] - RNDTABLE[3]
    assert rng.prndindex == 3
    assert rng.rndindex == 1


def test_state_get_set() -> None:
    rng = Random()
    for _ in range(17):
        rng.p_random()
    for _ in range(5):
        rng.m_random()

    saved = rng.get_state()
    expected_next = [rng.p_random() for _ in range(10)]

    rng2 = Random()
    rng2.set_state(saved)
    assert [rng2.p_random() for _ in range(10)] == expected_next
