import numpy as np
import pytest

from sdc.injection.bitflip import (
    BIT_GROUP_POSITIONS,
    bit_group_of,
    flip_bit_fp32,
    sample_bit,
)
from sdc.records.schema import BitGroup


def raw_bits(value) -> int:
    return int(np.array(value, dtype=np.float32).view(np.uint32))


def from_bits(bits: int) -> np.float32:
    return np.array(bits, dtype=np.uint32).view(np.float32)[()]


@pytest.mark.parametrize("bit, expected_bits", [
    (31, 0xBF800000),   # sign: -1.0
    (30, 0x7F800000),   # exponent: +inf
    (22, 0x3FC00000),   # mantissa: 1.5
    (0, 0x3F800001),    # mantissa: smallest step above 1.0
])
def test_flips_of_one_point_zero(bit, expected_bits):
    assert raw_bits(flip_bit_fp32(np.float32(1.0), bit)) == expected_bits


def test_decoded_values():
    assert flip_bit_fp32(np.float32(1.0), 31) == -1.0
    assert np.isposinf(flip_bit_fp32(np.float32(1.0), 30))
    assert flip_bit_fp32(np.float32(1.0), 22) == 1.5


def test_exponent_flip_can_produce_nan():
    # +inf with a mantissa bit set is NaN.
    assert np.isnan(flip_bit_fp32(from_bits(0x7F800000), 0))


@pytest.mark.parametrize("start", [
    0x3F800000,   # 1.0
    0x00000000,   # 0.0
    0x80000000,   # -0.0
    0x00000001,   # smallest subnormal
    0x7F800000,   # +inf
    0x7FC00001,   # quiet NaN with a payload
    0x7F800001,   # signaling NaN
])
@pytest.mark.parametrize("bit", range(32))
def test_flipping_twice_restores_exact_bits(start, bit):
    once = flip_bit_fp32(from_bits(start), bit)
    assert raw_bits(once) == start ^ (1 << bit)
    assert raw_bits(flip_bit_fp32(once, bit)) == start


@pytest.mark.parametrize("bit", [-1, 32])
def test_rejects_out_of_range_bit(bit):
    with pytest.raises(ValueError):
        flip_bit_fp32(np.float32(1.0), bit)
    with pytest.raises(ValueError):
        bit_group_of(bit)


def test_groups_cover_every_bit_once():
    positions = sorted(p for ps in BIT_GROUP_POSITIONS.values() for p in ps)
    assert positions == list(range(32))


def test_bit_group_boundaries():
    assert bit_group_of(31) == BitGroup.SIGN
    assert bit_group_of(30) == BitGroup.EXPONENT
    assert bit_group_of(23) == BitGroup.EXPONENT
    assert bit_group_of(22) == BitGroup.MANTISSA
    assert bit_group_of(0) == BitGroup.MANTISSA


@pytest.mark.parametrize("group", list(BitGroup))
def test_sample_bit_stays_in_group_and_is_deterministic(group):
    first = [sample_bit(group, np.random.default_rng(7)) for _ in range(3)]
    assert len(set(first)) == 1

    rng_a, rng_b = np.random.default_rng(42), np.random.default_rng(42)
    seq_a = [sample_bit(group, rng_a) for _ in range(200)]
    seq_b = [sample_bit(group, rng_b) for _ in range(200)]
    assert seq_a == seq_b
    assert set(seq_a) == set(BIT_GROUP_POSITIONS[group])
    assert all(isinstance(b, int) for b in seq_a)
