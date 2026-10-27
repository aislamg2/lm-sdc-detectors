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
    """Return the group a bit position belongs to.

    TODO: implement.
    """
    raise NotImplementedError


def sample_bit(group: BitGroup, rng: np.random.Generator) -> int:
    """Sample a bit position uniformly within `group`.

    TODO: implement.
    """
    raise NotImplementedError


def flip_bit_fp32(value: float, bit_position: int) -> float:
    """Return `value` with one bit flipped (view as uint32, XOR, view back).

    TODO: implement.
    """
    raise NotImplementedError
