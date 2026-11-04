import math

import pytest
import torch

from sdc.observations.stats import blackbox_stats, layer_stats, whitebox_stats
from sdc.records.schema import LayerStats


def test_uniform_logits_have_max_entropy():
    stats = blackbox_stats(torch.zeros(10))
    assert stats["top1_prob"] == pytest.approx(0.1)
    assert stats["entropy"] == pytest.approx(math.log(10))


def test_known_two_class_distribution():
    # log-odds of log(3) means probabilities 0.75 and 0.25.
    stats = blackbox_stats(torch.tensor([math.log(3.0), 0.0]))
    assert stats["top1_prob"] == pytest.approx(0.75)
    assert stats["entropy"] == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))


def test_very_confident_output_is_finite_not_nan():
    logits = torch.zeros(10)
    logits[3] = 1e30
    stats = blackbox_stats(logits)
    assert stats["top1_prob"] == 1.0
    assert stats["entropy"] == pytest.approx(0.0)


def test_batch_dimension_gives_same_result():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert blackbox_stats(logits.unsqueeze(0)) == blackbox_stats(logits)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_give_nan(bad):
    logits = torch.zeros(10)
    logits[0] = bad
    stats = blackbox_stats(logits)
    assert math.isnan(stats["top1_prob"])
    assert math.isnan(stats["entropy"])


def test_layer_stats_known_values():
    stats = layer_stats(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert (stats.min, stats.max, stats.nonfinite) == (1.0, 4.0, 0)
    assert stats.mean == pytest.approx(2.5)
    assert stats.std == pytest.approx(math.sqrt(1.25))   # population, not sample


def test_layer_stats_handles_multidimensional_activations():
    stats = layer_stats(torch.arange(24.0).reshape(1, 2, 3, 4))
    assert (stats.min, stats.max) == (0.0, 23.0)


def test_non_finite_values_are_counted_and_excluded():
    stats = layer_stats(torch.tensor([1.0, float("nan"), 3.0, float("inf"), float("-inf")]))
    assert stats.nonfinite == 3
    assert (stats.min, stats.max, stats.mean) == (1.0, 3.0, 2.0)


def test_all_non_finite_gives_nan_stats_and_full_count():
    stats = layer_stats(torch.tensor([float("nan"), float("inf")]))
    assert stats.nonfinite == 2
    assert all(math.isnan(v) for v in (stats.min, stats.max, stats.mean, stats.std))


def test_several_huge_finite_values_do_not_overflow():
    activation = torch.tensor([3e38, 3e38, 3e38, 0.0], dtype=torch.float32)
    # Why we use float64: in float32 the sum overflows, so mean and std become inf.
    assert math.isinf(activation.mean().item())
    assert math.isinf(activation.std(unbiased=False).item())
    stats = layer_stats(activation)
    assert stats.nonfinite == 0
    assert stats.mean == pytest.approx(2.25e38, rel=1e-6)
    assert stats.std == pytest.approx(1.299e38, rel=1e-3)


def test_single_value_has_zero_std():
    assert layer_stats(torch.tensor([2.0])).std == 0.0


def test_inputs_are_not_modified():
    activation = torch.tensor([1.0, float("nan"), 3.0])
    before = activation.clone()
    layer_stats(activation)
    assert torch.equal(activation.isnan(), before.isnan())
    assert activation.dtype == torch.float32


def test_works_on_tensors_requiring_grad():
    x = torch.ones(3, requires_grad=True)
    assert layer_stats(x * 2).mean == 2.0
    assert blackbox_stats(x)["top1_prob"] == pytest.approx(1 / 3)


def test_whitebox_applies_to_every_point():
    stats = whitebox_stats({"stem": torch.tensor([1.0]), "fc": torch.tensor([2.0])})
    assert list(stats) == ["stem", "fc"]
    assert all(isinstance(s, LayerStats) for s in stats.values())
    assert stats["fc"].max == 2.0


def test_stats_are_plain_python_numbers():
    stats = layer_stats(torch.tensor([1.0, 2.0]))
    assert all(type(v) is float for v in (stats.min, stats.max, stats.mean, stats.std))
    assert type(stats.nonfinite) is int
    assert all(type(v) is float for v in blackbox_stats(torch.zeros(3)).values())
