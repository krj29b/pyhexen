# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Unit tests for pyhexen.core.angles and generated trig tables (T1.2)."""

from __future__ import annotations

from pyhexen.core.angles import (
    ANG1,
    ANG1_X,
    ANG45,
    ANG60,
    ANG90,
    ANG180,
    ANG270,
    ANG_MAX,
    ANGLETOFINESHIFT,
    DBITS,
    FINEANGLES,
    FINECOSINE,
    FINEMASK,
    FINESINE,
    FINETANGENT,
    SLOPEBITS,
    SLOPERANGE,
    TANTOANGLE,
    approx_distance,
    fine_cosine,
    fine_sine,
    point_to_angle2,
    slope_div,
)
from pyhexen.core.fixed import FRACUNIT, to_int32


def _c_ref_slope_div(num: int, den: int) -> int:
    if den < 512:
        return SLOPERANGE
    ans = (num << 3) // (den >> 8)
    return ans if ans <= SLOPERANGE else SLOPERANGE


def _c_ref_point_to_angle2(x1: int, y1: int, x2: int, y2: int) -> int:
    x = x2 - x1
    y = y2 - y1
    if not x and not y:
        return 0
    if x >= 0:
        if y >= 0:
            if x > y:
                return TANTOANGLE[_c_ref_slope_div(y, x)] & 0xFFFFFFFF
            return (ANG90 - 1 - TANTOANGLE[_c_ref_slope_div(x, y)]) & 0xFFFFFFFF
        y = -y
        if x > y:
            return (-TANTOANGLE[_c_ref_slope_div(y, x)]) & 0xFFFFFFFF
        return (ANG270 + TANTOANGLE[_c_ref_slope_div(x, y)]) & 0xFFFFFFFF
    x = -x
    if y >= 0:
        if x > y:
            return (ANG180 - 1 - TANTOANGLE[_c_ref_slope_div(y, x)]) & 0xFFFFFFFF
        return (ANG90 + TANTOANGLE[_c_ref_slope_div(x, y)]) & 0xFFFFFFFF
    y = -y
    if x > y:
        return (ANG180 + TANTOANGLE[_c_ref_slope_div(y, x)]) & 0xFFFFFFFF
    return (ANG270 - 1 - TANTOANGLE[_c_ref_slope_div(x, y)]) & 0xFFFFFFFF


def test_table_constants_and_lengths() -> None:
    assert FINEANGLES == 8192
    assert FINEMASK == 8191
    assert ANGLETOFINESHIFT == 19
    assert SLOPERANGE == 2048
    assert SLOPEBITS == 11
    assert DBITS == 5

    assert ANG45 == 0x20000000
    assert ANG90 == 0x40000000
    assert ANG180 == 0x80000000
    assert ANG270 == 0xC0000000
    assert ANG_MAX == 0xFFFFFFFF
    assert ANG1 == ANG45 // 45
    assert ANG60 == ANG180 // 3
    assert ANG1_X == 0x01000000

    assert len(FINETANGENT) == 4096
    assert len(FINESINE) == 10240
    assert len(FINECOSINE) == 8192
    assert len(TANTOANGLE) == 2049
    assert FINECOSINE == FINESINE[2048:10240]


def test_table_sample_values_against_reference() -> None:
    # Sample values verified directly from reference/src/tables.c
    assert FINETANGENT[0] == -170910304
    assert FINETANGENT[1] == -56965752
    assert FINETANGENT[2047] == -25
    assert FINETANGENT[2048] == 25
    assert FINETANGENT[4095] == 170910304

    assert FINESINE[0] == 25
    assert FINESINE[1] == 75
    assert FINESINE[2047] == 65535
    assert FINESINE[2048] == 65535
    assert FINESINE[4096] == -25
    assert FINESINE[6144] == -65535
    assert FINESINE[10239] == 65535

    assert FINECOSINE[0] == 65535
    assert FINECOSINE[2048] == -25
    assert FINECOSINE[4096] == -65535
    assert FINECOSINE[6144] == 25
    assert FINECOSINE[8191] == 65535

    assert TANTOANGLE[0] == 0
    assert TANTOANGLE[1] == 333772
    assert TANTOANGLE[1024] == 316933408
    assert TANTOANGLE[2048] == 536870912  # ANG45


def test_fine_sine_and_cosine_helpers() -> None:
    assert fine_sine(0) == FINESINE[0]
    assert fine_sine(2048) == FINESINE[2048]
    assert fine_sine(ANG90) == FINESINE[2048]
    assert fine_sine(ANG180) == FINESINE[4096]
    assert fine_sine(ANG270) == FINESINE[6144]

    assert fine_cosine(0) == FINECOSINE[0]
    assert fine_cosine(2048) == FINECOSINE[2048]
    assert fine_cosine(ANG90) == FINECOSINE[2048]
    assert fine_cosine(ANG180) == FINECOSINE[4096]
    assert fine_cosine(ANG270) == FINECOSINE[6144]


def test_slope_div() -> None:
    assert slope_div(100, 0) == SLOPERANGE
    assert slope_div(100, 511) == SLOPERANGE
    assert slope_div(0, 512) == 0
    assert slope_div(512, 512) == SLOPERANGE
    assert slope_div(1024, 512) == SLOPERANGE
    assert slope_div(FRACUNIT, 2 * FRACUNIT) == 1024


def test_point_to_angle2_cardinal_and_diagonals() -> None:
    # Zero vector
    assert point_to_angle2(0, 0, 0, 0) == 0
    assert point_to_angle2(12345, -6789, 12345, -6789) == 0

    d = 100 * FRACUNIT
    # Cardinal directions
    assert point_to_angle2(0, 0, d, 0) == 0  # East
    assert point_to_angle2(0, 0, 0, d) == ANG90 - 1  # North
    assert point_to_angle2(0, 0, -d, 0) == ANG180 - 1  # West
    assert point_to_angle2(0, 0, 0, -d) == ANG270  # South

    # Diagonals (45, 135, 225, 315 degrees)
    assert point_to_angle2(0, 0, d, d) == ANG45 - 1
    assert point_to_angle2(0, 0, -d, d) == ANG90 + ANG45
    assert point_to_angle2(0, 0, -d, -d) == ANG270 - 1 - ANG45
    assert point_to_angle2(0, 0, d, -d) == ANG270 + ANG45


def test_point_to_angle2_10k_random_cases() -> None:
    state = 0xCAFEBABE

    def next_coord() -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        # Map coordinates within [-8192, 8192) map units in 16.16 fixed point
        return to_int32(state) >> 2

    for _ in range(10_000):
        x1 = next_coord()
        y1 = next_coord()
        x2 = next_coord()
        y2 = next_coord()
        got = point_to_angle2(x1, y1, x2, y2)
        expected = _c_ref_point_to_angle2(x1, y1, x2, y2)
        assert got == expected
        assert 0 <= got <= ANG_MAX


def test_approx_distance() -> None:
    assert approx_distance(0, 0) == 0
    assert approx_distance(100 * FRACUNIT, 0) == 100 * FRACUNIT
    assert approx_distance(0, -100 * FRACUNIT) == 100 * FRACUNIT
    assert approx_distance(100 * FRACUNIT, 50 * FRACUNIT) == 125 * FRACUNIT
    assert approx_distance(-50 * FRACUNIT, -100 * FRACUNIT) == 125 * FRACUNIT
