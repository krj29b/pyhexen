# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for pyhexen.core.fixed (T1.1)."""

from __future__ import annotations

import ctypes
import math

import pytest

from pyhexen.core.fixed import (
    FRACBITS,
    FRACUNIT,
    MAXINT,
    MAXSHORT,
    MININT,
    MINSHORT,
    abs32,
    c_div,
    c_mod,
    fixed_div,
    fixed_div2,
    fixed_mul,
    to_int16,
    to_int32,
    to_uint16,
    to_uint32,
)


def _ref_fixed_mul(a: int, b: int) -> int:
    a32 = ctypes.c_int32(a).value
    b32 = ctypes.c_int32(b).value
    prod64 = ctypes.c_int64(a32 * b32).value
    return ctypes.c_int32(prod64 >> 16).value


def _ref_fixed_div(a: int, b: int) -> int:
    a32 = ctypes.c_int32(a).value
    b32 = ctypes.c_int32(b).value
    if (abs(a32) >> 14) >= abs(b32):
        return MININT if (a32 ^ b32) < 0 else MAXINT
    num64 = ctypes.c_int64(a32 << 16).value
    q = int(math.trunc(num64 / b32)) if abs(num64) < (1 << 53) else c_div(num64, b32)
    return ctypes.c_int32(q).value


def test_constants() -> None:
    assert FRACBITS == 16
    assert FRACUNIT == 65536
    assert MININT == -0x80000000
    assert MAXINT == 0x7FFFFFFF
    assert MINSHORT == -0x8000
    assert MAXSHORT == 0x7FFF


def test_int_wrapping() -> None:
    assert to_int32(0) == 0
    assert to_int32(MAXINT) == MAXINT
    assert to_int32(MININT) == MININT
    assert to_int32(MAXINT + 1) == MININT
    assert to_int32(MININT - 1) == MAXINT
    assert to_int32(0xFFFFFFFF) == -1

    assert to_uint32(0) == 0
    assert to_uint32(-1) == 0xFFFFFFFF
    assert to_uint32(MININT) == 0x80000000
    assert to_uint32(0x1_0000_0005) == 5

    assert to_int16(0) == 0
    assert to_int16(MAXSHORT) == MAXSHORT
    assert to_int16(MINSHORT) == MINSHORT
    assert to_int16(MAXSHORT + 1) == MINSHORT
    assert to_int16(MINSHORT - 1) == MAXSHORT
    assert to_int16(0xFFFF) == -1

    assert to_uint16(0) == 0
    assert to_uint16(-1) == 0xFFFF
    assert to_uint16(MINSHORT) == 0x8000
    assert to_uint16(0x1_0007) == 7


def test_abs32() -> None:
    assert abs32(0) == 0
    assert abs32(42) == 42
    assert abs32(-42) == 42
    assert abs32(MAXINT) == MAXINT
    assert abs32(MININT) == 0x80000000
    assert abs32(0xFFFFFFFF) == 1


def test_c_div_and_c_mod() -> None:
    # Truncation toward zero vs Python floor division
    assert c_div(7, 2) == 3
    assert c_div(-7, 2) == -3
    assert c_div(7, -2) == -3
    assert c_div(-7, -2) == 3
    assert c_div(0, 5) == 0
    assert c_div(0, -5) == 0

    # Remainder sign matches dividend
    assert c_mod(7, 2) == 1
    assert c_mod(-7, 2) == -1
    assert c_mod(7, -2) == 1
    assert c_mod(-7, -2) == -1
    assert c_mod(0, 5) == 0
    assert c_mod(0, -5) == 0

    for a in range(-50, 51, 3):
        for b in (-13, -7, -2, -1, 1, 2, 7, 13):
            q = c_div(a, b)
            r = c_mod(a, b)
            assert q * b + r == a
            assert abs(r) < abs(b)
            if r != 0:
                assert (r < 0) == (a < 0)

    with pytest.raises(ZeroDivisionError):
        c_div(1, 0)
    with pytest.raises(ZeroDivisionError):
        c_mod(1, 0)


def test_fixed_mul_basic_and_edges() -> None:
    assert fixed_mul(0, FRACUNIT) == 0
    assert fixed_mul(FRACUNIT, FRACUNIT) == FRACUNIT
    assert fixed_mul(2 * FRACUNIT, 3 * FRACUNIT) == 6 * FRACUNIT
    assert fixed_mul(-2 * FRACUNIT, 3 * FRACUNIT) == -6 * FRACUNIT
    assert fixed_mul(-2 * FRACUNIT, -3 * FRACUNIT) == 6 * FRACUNIT
    assert fixed_mul(FRACUNIT // 2, FRACUNIT // 2) == FRACUNIT // 4
    # Negative odd product arithmetic right shift check
    assert fixed_mul(-1, 1) == -1
    # 32-bit wrap-around on huge fixed_mul
    assert fixed_mul(0x40000000, 2 * FRACUNIT) == MININT


def test_fixed_div_basic_and_overflow_clamps() -> None:
    assert fixed_div(6 * FRACUNIT, 3 * FRACUNIT) == 2 * FRACUNIT
    assert fixed_div(-6 * FRACUNIT, 3 * FRACUNIT) == -2 * FRACUNIT
    assert fixed_div(6 * FRACUNIT, -3 * FRACUNIT) == -2 * FRACUNIT
    assert fixed_div(-6 * FRACUNIT, -3 * FRACUNIT) == 2 * FRACUNIT
    assert fixed_div(FRACUNIT, 2 * FRACUNIT) == FRACUNIT // 2

    # Division by zero clamps via overflow check
    assert fixed_div(FRACUNIT, 0) == MAXINT
    assert fixed_div(-FRACUNIT, 0) == MININT
    assert fixed_div(0, 0) == MAXINT

    # Overflow boundary: (abs(a) >> 14) >= abs(b)
    assert fixed_div(1 << 14, 1) == MAXINT
    assert fixed_div(-(1 << 14), 1) == MININT
    assert fixed_div(1 << 14, -1) == MININT
    assert fixed_div(-(1 << 14), -1) == MAXINT
    # Just below overflow boundary: (abs(a) >> 14) < abs(b)
    assert fixed_div((1 << 14) - 1, 1) == ((1 << 14) - 1) << 16
    assert fixed_div2((1 << 14) - 1, 1) == ((1 << 14) - 1) << 16


def test_fixed_property_against_c_reference() -> None:
    # Deterministic LCG to avoid importing stdlib random
    state = 0x13579BDF

    def next_int32() -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return to_int32(state)

    boundary_vals = [
        0,
        1,
        -1,
        2,
        -2,
        (1 << 14) - 1,
        1 << 14,
        (1 << 14) + 1,
        -((1 << 14) - 1),
        -(1 << 14),
        -((1 << 14) + 1),
        FRACUNIT - 1,
        FRACUNIT,
        FRACUNIT + 1,
        -FRACUNIT,
        MAXSHORT,
        MINSHORT,
        MAXINT,
        MININT,
    ]

    for a in boundary_vals:
        for b in boundary_vals:
            assert fixed_mul(a, b) == _ref_fixed_mul(a, b)
            assert fixed_div(a, b) == _ref_fixed_div(a, b)

    for _ in range(5000):
        a = next_int32()
        b = next_int32()
        assert fixed_mul(a, b) == _ref_fixed_mul(a, b)
        assert fixed_div(a, b) == _ref_fixed_div(a, b)
