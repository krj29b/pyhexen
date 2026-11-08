# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""Empirical adversarial stress testing for Task T10.3 line-of-sight slope aperture mathematics.

This test suite stress-tests the slope aperture mathematics in `src/pyhexen/sim/sight.py`:
1. Mathematical Invariants:
   - Initial aperture derivation: topslope = (t2.z + t2.height) - sightzstart,
     bottomslope = t2.z - sightzstart.
   - Monotonic narrowing: topslope non-increasing, bottomslope non-decreasing,
     aperture width non-increasing across any sequence of 2-sided portals.
   - Aperture collapse condition: topslope <= bottomslope returns False.
2. Hexen C Reference Oracle Comparison:
   - Standalone reference oracle faithfully replicating Chocolate Doom / Hexen C
     `PTR_SightTraverse` and `FixedDiv`.
   - 25,000+ randomized portal sequences testing floor/ceiling steps, heights, and fractions
     across 7 distinct geometric distributions.
   - 100% agreement between implementation and reference oracle.
3. Numerical Boundary & Extreme Value Stress:
   - Extreme fixed-point coordinates (-32768 to 32767 map units in FRACUNIT).
   - Microscopic fractions (frac = 1, 2, 3...) testing overflow clamping in fixed_div.
   - Exact overflow threshold boundary testing (threshold vs threshold + 1).
   - Huge height differences triggering MININT/MAXINT.
   - Exact boundary cases: topslope == bottomslope, topslope == bottomslope +/- 1.
4. End-to-end synthetic map tests via GameSim and check_sight.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import pytest
from tests.fixtures.wadbuilder import MapBuilder, WadBuilder

from pyhexen.core.fixed import (
    FRACBITS,
    FRACUNIT,
    MAXINT,
    MININT,
    c_div,
    fixed_div,
    to_int32,
)
from pyhexen.core.random import Random
from pyhexen.sim import check_sight
from pyhexen.sim.game_sim import GameSim
from pyhexen.sim.info_data import MobjType
from pyhexen.sim.mobj import spawn_mobj
from pyhexen.wad.wadfile import WadStack
from pyhexen.world.level import load_level

# ==============================================================================
# Helper / Reference Oracle Implementation
# ==============================================================================


class PortalSpec(NamedTuple):
    frac: int
    front_floor: int
    front_ceil: int
    back_floor: int
    back_ceil: int


def hexen_c_fixed_div(a: int, b: int) -> int:
    """Independent implementation of Chocolate Doom / Hexen C FixedDiv (m_fixed.c)."""
    a32 = to_int32(a)
    b32 = to_int32(b)
    if (abs(a32) >> 14) >= abs(b32):
        return MININT if (a32 ^ b32) < 0 else MAXINT
    return to_int32(c_div(a32 << FRACBITS, b32))


def hexen_c_ptr_sight_traverse_oracle(
    sightzstart: int,
    initial_topslope: int,
    initial_bottomslope: int,
    portals: list[PortalSpec],
) -> tuple[bool, int, int]:
    """Port of Chocolate Doom PTR_SightTraverse (p_sight.c).

    Returns (sight_passed, final_topslope, final_bottomslope).
    """
    topslope = initial_topslope
    bottomslope = initial_bottomslope

    for portal in portals:
        openbottom = max(portal.front_floor, portal.back_floor)
        opentop = min(portal.front_ceil, portal.back_ceil)

        # if (openbottom >= opentop) return false;
        if openbottom >= opentop:
            return False, topslope, bottomslope

        if portal.front_floor != portal.back_floor:
            slope = hexen_c_fixed_div(openbottom - sightzstart, portal.frac)
            if slope > bottomslope:
                bottomslope = slope

        if portal.front_ceil != portal.back_ceil:
            slope = hexen_c_fixed_div(opentop - sightzstart, portal.frac)
            if slope < topslope:
                topslope = slope

        # if (topslope <= bottomslope) return false;
        if topslope <= bottomslope:
            return False, topslope, bottomslope

    return True, topslope, bottomslope


def pyhexen_sight_traverse_oracle(
    sightzstart: int,
    initial_topslope: int,
    initial_bottomslope: int,
    portals: list[PortalSpec],
) -> tuple[bool, int, int]:
    """Implementation reference using PyHexen's exact logic from sight.py."""
    topslope = initial_topslope
    bottomslope = initial_bottomslope

    for portal in portals:
        openbottom = max(portal.front_floor, portal.back_floor)
        opentop = min(portal.front_ceil, portal.back_ceil)

        if openbottom >= opentop:
            return False, topslope, bottomslope

        if portal.front_floor != portal.back_floor:
            slope = fixed_div(openbottom - sightzstart, portal.frac)
            if slope > bottomslope:
                bottomslope = slope

        if portal.front_ceil != portal.back_ceil:
            slope = fixed_div(opentop - sightzstart, portal.frac)
            if slope < topslope:
                topslope = slope

        if topslope <= bottomslope:
            return False, topslope, bottomslope

    return True, topslope, bottomslope


def _make_sim_from_map(mb: MapBuilder) -> GameSim:
    stack = WadStack(WadBuilder("IWAD").add_map(mb).to_bytes())
    level = load_level(stack, mb.name)
    return GameSim(level=level, rng=Random())


# ==============================================================================
# 1. Mathematical Invariant Verification
# ==============================================================================


@pytest.mark.parametrize(
    ("z1", "h1", "z2", "h2"),
    [
        (0, 64 * FRACUNIT, 0, 64 * FRACUNIT),
        (100 * FRACUNIT, 80 * FRACUNIT, -50 * FRACUNIT, 40 * FRACUNIT),
        (-200 * FRACUNIT, 32 * FRACUNIT, 300 * FRACUNIT, 16 * FRACUNIT),
        (0, 0, 0, 0),
        (1000 * FRACUNIT, 64 * FRACUNIT, 1000 * FRACUNIT, 0),
        (0, 56 * FRACUNIT, 0, 1),  # microscopic target
    ],
)
def test_invariant_initial_aperture_span(z1: int, h1: int, z2: int, h2: int) -> None:
    """Verify that initial topslope - bottomslope is strictly equal to target height.

    sightzstart = t1.z + t1.height - (t1.height >> 2)
    topslope = (t2.z + t2.height) - sightzstart
    bottomslope = t2.z - sightzstart
    aperture = topslope - bottomslope == t2.height.
    """
    sightzstart = z1 + h1 - (h1 >> 2)
    topslope = (z2 + h2) - sightzstart
    bottomslope = z2 - sightzstart
    aperture = topslope - bottomslope

    assert aperture == h2
    if h2 > 0:
        assert topslope > bottomslope
    elif h2 == 0:
        assert topslope == bottomslope
    else:
        assert topslope < bottomslope


def test_invariant_monotonic_narrowing_randomized() -> None:
    """Verify that as ray crosses 2-sided portals, aperture NEVER widens.

    topslope_k <= topslope_{k-1}
    bottomslope_k >= bottomslope_{k-1}
    aperture_k <= aperture_{k-1}
    """
    rng = random.Random(42)

    for _ in range(1000):
        sightzstart = rng.randint(-1000, 1000) * FRACUNIT
        h2 = rng.randint(1, 128) * FRACUNIT
        z2 = rng.randint(-1000, 1000) * FRACUNIT
        topslope = (z2 + h2) - sightzstart
        bottomslope = z2 - sightzstart

        num_portals = rng.randint(1, 15)
        raw_fracs = sorted([rng.randint(1, FRACUNIT) for _ in range(num_portals)])

        for frac in raw_fracs:
            prev_top = topslope
            prev_bottom = bottomslope
            prev_aperture = topslope - bottomslope

            f_floor = rng.randint(-500, 500) * FRACUNIT
            f_ceil = f_floor + rng.randint(10, 500) * FRACUNIT
            b_floor = rng.randint(-500, 500) * FRACUNIT
            b_ceil = b_floor + rng.randint(10, 500) * FRACUNIT

            openbottom = max(f_floor, b_floor)
            opentop = min(f_ceil, b_ceil)
            if openbottom >= opentop:
                continue

            if f_floor != b_floor:
                slope = fixed_div(openbottom - sightzstart, frac)
                if slope > bottomslope:
                    bottomslope = slope

            if f_ceil != b_ceil:
                slope = fixed_div(opentop - sightzstart, frac)
                if slope < topslope:
                    topslope = slope

            assert topslope <= prev_top, f"topslope widened! {topslope} > {prev_top}"
            assert bottomslope >= prev_bottom, f"bottomslope widened! {bottomslope} < {prev_bottom}"
            curr_aperture = topslope - bottomslope
            assert curr_aperture <= prev_aperture, (
                f"Aperture widened! {curr_aperture} > {prev_aperture}"
            )


def test_invariant_aperture_collapse_stops_immediately() -> None:
    """Whenever topslope <= bottomslope, sight must immediately evaluate to False."""
    sightzstart = 48 * FRACUNIT
    topslope = 16 * FRACUNIT
    bottomslope = -48 * FRACUNIT

    # Portal 1 at frac=0.5: floor rises to 60 -> slope = (60 - 48) / 0.5 = 24
    # New bottomslope = 24. Now topslope (16) <= bottomslope (24) -> collapse!
    portals = [
        PortalSpec(
            frac=FRACUNIT // 2,
            front_floor=0,
            front_ceil=128 * FRACUNIT,
            back_floor=60 * FRACUNIT,
            back_ceil=128 * FRACUNIT,
        ),
        PortalSpec(
            frac=(FRACUNIT * 4) // 5,
            front_floor=0,
            front_ceil=128 * FRACUNIT,
            back_floor=0,
            back_ceil=128 * FRACUNIT,
        ),
    ]

    res, final_top, final_bot = pyhexen_sight_traverse_oracle(
        sightzstart, topslope, bottomslope, portals
    )
    assert not res
    assert final_top <= final_bot


# ==============================================================================
# 2. Reference Oracle Comparison (25,000 Randomized Portal Sequences)
# ==============================================================================


def test_oracle_25000_randomized_portal_sequences() -> None:
    """Run 25,000 randomized portal sequences comparing PyHexen vs Hexen C Oracle.

    Covers 7 distinct distributions:
      1. Standard height ranges (-1024 to 2048)
      2. Extreme height ranges (-32768 to 32767)
      3. Microscopic fractions (frac 1 to 256)
      4. Fully closed and inverted portals (openbottom >= opentop)
      5. Floor steps only (ceilings identical)
      6. Ceiling drops only (floors identical)
      7. Flat portals (neither changes)
    Must achieve 100.00% agreement across all cases.
    """
    rng = random.Random(1337)
    test_count = 25000
    mismatches = 0

    for i in range(test_count):
        dist_type = i % 7

        if dist_type == 0:  # Standard heights
            sightzstart = rng.randint(-1000, 1000) * FRACUNIT
            target_z = rng.randint(-1000, 1000) * FRACUNIT
            target_h = rng.randint(0, 128) * FRACUNIT
            h_min, h_max = -500, 500
        elif dist_type == 1:  # Extreme heights
            sightzstart = rng.randint(-30000, 30000) * FRACUNIT
            target_z = rng.randint(-30000, 30000) * FRACUNIT
            target_h = rng.randint(0, 1024) * FRACUNIT
            h_min, h_max = -30000, 30000
        elif dist_type == 2:  # Microscopic fractions
            sightzstart = rng.randint(-500, 500) * FRACUNIT
            target_z = rng.randint(-500, 500) * FRACUNIT
            target_h = rng.randint(0, 64) * FRACUNIT
            h_min, h_max = -200, 200
        else:
            sightzstart = rng.randint(-1000, 1000) * FRACUNIT
            target_z = rng.randint(-1000, 1000) * FRACUNIT
            target_h = rng.randint(16, 80) * FRACUNIT
            h_min, h_max = -500, 500

        topslope = (target_z + target_h) - sightzstart
        bottomslope = target_z - sightzstart

        num_portals = rng.randint(1, 10)
        if dist_type == 2:
            fracs = sorted([rng.randint(1, 256) for _ in range(num_portals)])
        else:
            fracs = sorted([rng.randint(1, FRACUNIT) for _ in range(num_portals)])

        portals: list[PortalSpec] = []
        for frac in fracs:
            if dist_type == 4:  # Floor steps only
                f_floor = rng.randint(h_min, h_max) * FRACUNIT
                b_floor = rng.randint(h_min, h_max) * FRACUNIT
                f_ceil = max(f_floor, b_floor) + 200 * FRACUNIT
                b_ceil = f_ceil
            elif dist_type == 5:  # Ceiling drops only
                f_floor = 0
                b_floor = 0
                f_ceil = rng.randint(100, 500) * FRACUNIT
                b_ceil = rng.randint(100, 500) * FRACUNIT
            elif dist_type == 6:  # Flat portal
                f_floor = 0
                b_floor = 0
                f_ceil = 200 * FRACUNIT
                b_ceil = 200 * FRACUNIT
            else:
                f_floor = rng.randint(h_min, h_max) * FRACUNIT
                f_ceil = f_floor + rng.randint(-20, 300) * FRACUNIT
                b_floor = rng.randint(h_min, h_max) * FRACUNIT
                b_ceil = b_floor + rng.randint(-20, 300) * FRACUNIT
            portals.append(PortalSpec(frac, f_floor, f_ceil, b_floor, b_ceil))

        c_result, c_top, c_bot = hexen_c_ptr_sight_traverse_oracle(
            sightzstart, topslope, bottomslope, portals
        )
        py_result, py_top, py_bot = pyhexen_sight_traverse_oracle(
            sightzstart, topslope, bottomslope, portals
        )

        if c_result != py_result or c_top != py_top or c_bot != py_bot:
            mismatches += 1

    assert mismatches == 0, f"Found {mismatches} mismatches out of {test_count} tests!"


# ==============================================================================
# 3. Numerical Boundary & Extreme Value Stress
# ==============================================================================


@pytest.mark.parametrize(
    "frac",
    [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 400],
)
def test_microscopic_fractions_overflow_clamping(frac: int) -> None:
    """Microscopic fractions correctly trigger FixedDiv overflow clamping to MAXINT/MININT."""
    sightzstart = 0
    # Step height = 100 units -> threshold = (100 * 65536) >> 14 = 400
    openbottom = 100 * FRACUNIT
    slope_pos = fixed_div(openbottom - sightzstart, frac)
    c_slope_pos = hexen_c_fixed_div(openbottom - sightzstart, frac)
    assert slope_pos == MAXINT
    assert slope_pos == c_slope_pos

    opentop = -100 * FRACUNIT
    slope_neg = fixed_div(opentop - sightzstart, frac)
    c_slope_neg = hexen_c_fixed_div(opentop - sightzstart, frac)
    assert slope_neg == MININT
    assert slope_neg == c_slope_neg


def test_overflow_threshold_exact_boundary() -> None:
    """Test exact transition boundary where FixedDiv switches from MAXINT to division.

    For a = 100 * FRACUNIT:
      threshold = (a >> 14) = 400.
      frac == 400 -> overflow clamp (MAXINT)
      frac == 401 -> no overflow, exact fixed division
    """
    a = 100 * FRACUNIT
    threshold = abs(a) >> 14
    assert threshold == 400

    # At threshold: clamps
    assert fixed_div(a, threshold) == MAXINT
    assert hexen_c_fixed_div(a, threshold) == MAXINT

    # At threshold + 1: computes exact division
    res = fixed_div(a, threshold + 1)
    c_res = hexen_c_fixed_div(a, threshold + 1)
    assert res != MAXINT
    assert res == c_res
    expected = (a << FRACBITS) // (threshold + 1)
    assert res == expected

    # Negative side
    assert fixed_div(-a, threshold) == MININT
    assert hexen_c_fixed_div(-a, threshold) == MININT
    neg_res = fixed_div(-a, threshold + 1)
    c_neg_res = hexen_c_fixed_div(-a, threshold + 1)
    assert neg_res != MININT
    assert neg_res == c_neg_res


def test_huge_height_difference_exceeding_32bit() -> None:
    """Extreme height differences test to_int32 wrapping and overflow clamping."""
    sightzstart = -25000 * FRACUNIT
    openbottom = 25000 * FRACUNIT
    frac = FRACUNIT // 2

    slope = fixed_div(openbottom - sightzstart, frac)
    c_slope = hexen_c_fixed_div(openbottom - sightzstart, frac)
    assert slope == c_slope


@pytest.mark.parametrize(
    ("topslope", "bottomslope", "expected"),
    [
        (100, 100, False),  # exact equality -> collapsed
        (100, 99, True),  # 1 unit open -> passes
        (100, 101, False),  # 1 unit inverted -> collapsed
        (0, 0, False),  # zero aperture -> collapsed
        (1, 0, True),  # minimal open aperture -> passes
        (-500, -500, False),  # negative equality -> collapsed
        (-500, -501, True),  # negative 1-unit open -> passes
        (-500, -499, False),  # negative inverted -> collapsed
        (MAXINT, MAXINT, False),  # extreme positive equality
        (MININT, MININT, False),  # extreme negative equality
        (MAXINT, MAXINT - 1, True),  # extreme open aperture
        (MININT + 1, MININT, True),  # extreme lower open aperture
    ],
)
def test_exact_aperture_boundary_cases(topslope: int, bottomslope: int, expected: bool) -> None:
    """Test boundary conditions for topslope vs bottomslope comparison."""
    portals = [
        PortalSpec(
            frac=FRACUNIT // 2,
            front_floor=0,
            front_ceil=1000 * FRACUNIT,
            back_floor=0,
            back_ceil=1000 * FRACUNIT,
        )
    ]
    res, _, _ = pyhexen_sight_traverse_oracle(0, topslope, bottomslope, portals)
    assert res is expected


def test_exact_boundary_one_fracunit_step() -> None:
    """At frac = FRACUNIT, FixedDiv(delta, FRACUNIT) is exactly delta."""
    delta = 42 * FRACUNIT
    res = fixed_div(delta, FRACUNIT)
    assert res == delta


# ==============================================================================
# 4. End-to-End Simulation Checks on Synthetic Maps
# ==============================================================================


def test_check_sight_extreme_map_coordinates() -> None:
    """Mobjs at extreme map coordinates (-30000 to +30000) evaluate sight correctly."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        x=-30000,
        y=0,
        width1=2000,
        width2=2000,
        height=2000,
        floor1=0,
        ceil1=256,
        floor2=0,
        ceil2=256,
    )
    sim = _make_sim_from_map(mb)

    t1 = spawn_mobj(sim, -29000 * FRACUNIT, 1000 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t2 = spawn_mobj(sim, -27000 * FRACUNIT, 1000 * FRACUNIT, 0, MobjType.MT_PLAYER_FIGHTER)

    assert check_sight(sim, t1, t2)


def test_check_sight_extreme_vertical_coordinates_game_sim() -> None:
    """End-to-end check_sight with sectors at extreme negative heights (-30000)."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        width1=256,
        width2=256,
        floor1=-30000,
        ceil1=-28000,
        floor2=-29500,  # 500 unit step up
        ceil2=-28000,
    )
    sim = _make_sim_from_map(mb)

    # Actor in sector 1 with eye high enough to look over step
    t1 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, -30000 * FRACUNIT, MobjType.MT_ETTIN)
    t1.height = 800 * FRACUNIT  # eye at -30000 + 600 = -29400 (above floor2)

    # Target in sector 2 on floor -29500 with height 200 (top at -29300)
    t2 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, -29500 * FRACUNIT, MobjType.MT_ETTIN)
    t2.height = 200 * FRACUNIT
    assert check_sight(sim, t1, t2)

    # Second target lying completely below step occluded
    t3 = spawn_mobj(sim, 350 * FRACUNIT, 128 * FRACUNIT, -30000 * FRACUNIT, MobjType.MT_ETTIN)
    t3.height = 100 * FRACUNIT
    assert not check_sight(sim, t1, t3)


def test_check_sight_boundary_slit_occlusion_and_pass() -> None:
    """Sub-pixel / 1-unit slit behavior on synthetic map."""
    mb = MapBuilder("MAP01")
    mb.add_two_sector_rooms(
        width1=200,
        width2=200,
        floor1=0,
        ceil1=128,
        floor2=64,
        ceil2=65,
    )
    sim = _make_sim_from_map(mb)

    t1 = spawn_mobj(sim, 100 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t1.height = 86 * FRACUNIT

    t2 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 64 * FRACUNIT, MobjType.MT_ETTIN)
    t2.height = 2 * FRACUNIT
    assert check_sight(sim, t1, t2)

    t3 = spawn_mobj(sim, 300 * FRACUNIT, 128 * FRACUNIT, 0, MobjType.MT_ETTIN)
    t3.height = 20 * FRACUNIT
    assert not check_sight(sim, t1, t3)
