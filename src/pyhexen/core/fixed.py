# PyHexen - Python reimplementation of the Hexen engine.
# Portions derived from Hexen (c) Raven Software / id Software and Chocolate Doom,
# released under the GNU GPL v2. This file is licensed GPL-2.0-or-later.

"""16.16 fixed-point arithmetic and C integer semantics.

Port of m_fixed.h and m_fixed.c.
"""

from __future__ import annotations

FRACBITS: int = 16
FRACUNIT: int = 1 << FRACBITS

MININT: int = -0x80000000
MAXINT: int = 0x7FFFFFFF
MINSHORT: int = -0x8000
MAXSHORT: int = 0x7FFF


def to_int32(val: int) -> int:
    """Wrap an integer to a signed 32-bit two's-complement value."""
    return ((val + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def to_uint32(val: int) -> int:
    """Wrap an integer to an unsigned 32-bit value."""
    return val & 0xFFFFFFFF


def to_int16(val: int) -> int:
    """Wrap an integer to a signed 16-bit two's-complement value."""
    return ((val + 0x8000) & 0xFFFF) - 0x8000


def to_uint16(val: int) -> int:
    """Wrap an integer to an unsigned 16-bit value."""
    return val & 0xFFFF


def abs32(val: int) -> int:
    """Return the non-negative magnitude of a 32-bit signed integer."""
    return abs(to_int32(val))


def c_div(a: int, b: int) -> int:
    """Integer division truncating toward zero, matching C's `/` operator."""
    if b == 0:
        raise ZeroDivisionError("integer division by zero")
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def c_mod(a: int, b: int) -> int:
    """Integer modulo with remainder sign matching dividend, matching C's `%` operator."""
    if b == 0:
        raise ZeroDivisionError("integer modulo by zero")
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def fixed_mul(a: int, b: int) -> int:
    """Multiply two 16.16 fixed-point numbers.

    Port of FixedMul (m_fixed.c).
    """
    a32 = to_int32(a)
    b32 = to_int32(b)
    return to_int32((a32 * b32) >> FRACBITS)


def fixed_div2(a: int, b: int) -> int:
    """Unchecked 64-bit intermediate fixed-point division.

    Port of FixedDiv2 (m_fixed.c).
    """
    a32 = to_int32(a)
    b32 = to_int32(b)
    return to_int32(c_div(a32 << FRACBITS, b32))


def fixed_div(a: int, b: int) -> int:
    """Divide two 16.16 fixed-point numbers with overflow clamping.

    Port of FixedDiv (m_fixed.c).
    """
    a32 = to_int32(a)
    b32 = to_int32(b)
    if (abs(a32) >> 14) >= abs(b32):
        return MININT if (a32 ^ b32) < 0 else MAXINT
    return fixed_div2(a32, b32)
