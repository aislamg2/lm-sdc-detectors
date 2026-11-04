"""IEEE-754 FP32 single-bit flips and bit-group sampling.

FP32 layout: bit 31 = sign, bits 30..23 = exponent, bits 22..0 = mantissa.
"""

import numpy as np

from sdc.records.schema import BitGroup

BIT_GROUP_POSITIONS = {
    BitGroup.SIGN: [31],
    BitGroup.EXPONENT: list(range(23, 31)),
    BitGroup.MANTISSA: list(range(0, 23)),
}


def bit_group_of(bit_position: int) -> BitGroup:
    """Return the group a bit position belongs to."""
    for group, positions in BIT_GROUP_POSITIONS.items():
        if bit_position in positions:
            return group
    raise ValueError(f"bit_position must be in 0..31, got {bit_position}")


def sample_bit(group: BitGroup, rng: np.random.Generator) -> int:
    """Sample a bit position uniformly within `group`."""
    return int(rng.choice(BIT_GROUP_POSITIONS[group]))


def flip_bit_fp32(value: np.floating, bit_position: int) -> np.float32:
    """Return `value` with one bit flipped (view as uint32, XOR, view back).

    Pass a np.float32 taken directly from the tensor: a Python float is 64-bit,
    and converting through it can alter the bits of a NaN.
    """
    if not 0 <= bit_position <= 31:
        raise ValueError(f"bit_position must be in 0..31, got {bit_position}")
    bits = np.array(value, dtype=np.float32).view(np.uint32)
    return (bits ^ np.uint32(1 << bit_position)).view(np.float32)[()]
