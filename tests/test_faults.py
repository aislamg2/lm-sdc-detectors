import math
from collections import Counter

import numpy as np
import pytest
import torch

from sdc.injection.bitflip import BIT_GROUP_POSITIONS, flip_bit_fp32
from sdc.injection.faults import (
    ActivationInjector,
    FaultSpec,
    FaultValues,
    activation_shapes,
    injectable_layers,
    sample_fault,
    weight_fault,
    weight_shapes,
)
from sdc.model.monitor import ActivationMonitor
from sdc.model.resnet18 import build_resnet18
from sdc.records.schema import BitGroup, FaultSite

CONV_WEIGHT = (0, 0, 0, 0)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_resnet18().eval()


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(1)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture(scope="module")
def clean_logits(model, image):
    with torch.no_grad():
        return model(image).clone()


@pytest.fixture(scope="module")
def injector(model):
    return ActivationInjector(model)


def output_of(model, layer: str, image) -> torch.Tensor:
    """Run the model and return a copy of one layer's output."""
    seen = {}

    def hook(module, inputs, output):
        seen["x"] = output.clone()   # must return None: a returned value replaces the output

    handle = model.get_submodule(layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(image)
    finally:
        handle.remove()
    return seen["x"]


def bits_of(model) -> dict[str, bytes]:
    return {k: v.numpy().tobytes() for k, v in model.state_dict().items()}


def weight_spec(layer="layer3.1.conv2", element=CONV_WEIGHT, bit=30, group=BitGroup.EXPONENT):
    return FaultSpec(FaultSite.WEIGHT, layer, element, bit, group)


def activation_spec(layer="layer3.1.conv2", element=(0, 0, 0), bit=30, group=BitGroup.EXPONENT):
    return FaultSpec(FaultSite.ACTIVATION, layer, element, bit, group)


def test_injectable_layers_are_the_21_conv_and_linear_layers(model):
    layers = injectable_layers(model)
    assert len(layers) == 21
    assert list(layers)[0] == "conv1"
    assert list(layers)[-1] == "fc"
    assert "layer2.0.downsample.0" in layers
    assert not any("bn" in name for name in layers)


def test_shapes_match_the_real_layers(model):
    weights = weight_shapes(model)
    assert weights["conv1"] == (64, 3, 7, 7)
    assert weights["fc"] == (10, 512)
    outputs = activation_shapes(model)
    assert outputs["conv1"] == (64, 112, 112)
    assert outputs["layer4.1.conv2"] == (512, 7, 7)
    assert outputs["fc"] == (10,)
    assert list(outputs) == list(weights)


def test_activation_shapes_leave_no_hooks(model):
    activation_shapes(model)
    assert sum(len(m._forward_hooks) for m in model.modules()) == 0


def test_our_layer_order_matches_pytorchfi(model, injector):
    # A layer number from our names must point at the same layer in PyTorchFI.
    outputs = activation_shapes(model)
    assert len(injector._pfi.output_size) == 21
    for name, shape in zip(outputs, injector._pfi.output_size):
        assert tuple(shape[1:]) == outputs[name]


@pytest.mark.parametrize("site, shapes_of", [
    (FaultSite.WEIGHT, weight_shapes), (FaultSite.ACTIVATION, activation_shapes),
])
def test_sampled_faults_are_in_bounds(model, site, shapes_of):
    shapes = shapes_of(model)
    rng = np.random.default_rng(0)
    for group in BitGroup:
        for _ in range(100):
            fault = sample_fault(shapes, site, group, rng)
            assert fault.site == site and fault.bit_group == group
            assert fault.bit_position in BIT_GROUP_POSITIONS[group]
            assert len(fault.element) == len(shapes[fault.layer])
            assert all(0 <= i < n for i, n in zip(fault.element, shapes[fault.layer]))
            assert all(type(i) is int for i in fault.element)
            assert type(fault.layer) is str


def test_sampling_is_uniform_over_layers_not_weights(model):
    shapes = weight_shapes(model)
    rng = np.random.default_rng(0)
    counts = Counter(
        sample_fault(shapes, FaultSite.WEIGHT, BitGroup.SIGN, rng).layer for _ in range(4200)
    )
    assert set(counts) == set(shapes)
    # Expected 200 each, even though fc has far fewer weights than layer4 convs.
    assert all(130 < n < 270 for n in counts.values())


def test_sampling_is_deterministic(model):
    shapes = weight_shapes(model)
    def sample(seed):
        rng = np.random.default_rng(seed)
        return [sample_fault(shapes, FaultSite.WEIGHT, BitGroup.MANTISSA, rng) for _ in range(20)]
    assert sample(5) == sample(5)
    assert sample(5) != sample(6)


def test_weight_fault_changes_exactly_one_bit_during_the_block(model):
    fault = weight_spec()
    weight = model.get_submodule(fault.layer).weight
    before = weight.data.numpy().copy()
    with weight_fault(model, fault):
        during = weight.data.numpy().copy()
    assert (before.tobytes() != during.tobytes())
    changed = np.argwhere(before.view(np.uint32) != during.view(np.uint32))
    assert changed.tolist() == [list(CONV_WEIGHT)]
    xor = before.view(np.uint32)[CONV_WEIGHT] ^ during.view(np.uint32)[CONV_WEIGHT]
    assert int(xor) == 1 << fault.bit_position


def test_weight_fault_reports_the_values(model):
    fault = weight_spec()
    original = model.get_submodule(fault.layer).weight.data[CONV_WEIGHT].numpy().copy()
    with weight_fault(model, fault) as values:
        assert values == FaultValues(float(original), float(flip_bit_fp32(original, 30)))


def test_weight_fault_changes_the_output_when_its_input_is_nonzero(model, image, clean_logits):
    # fc.weight[3, k] multiplies feature k, so pick the largest feature.
    k = int(output_of(model, "avgpool", image).argmax())
    with torch.no_grad(), weight_fault(model, weight_spec("fc", (3, k), 30, BitGroup.EXPONENT)):
        assert not torch.equal(model(image), clean_logits)


def test_huge_weight_can_still_be_masked(model, image, clean_logits):
    # Documents why many faults are masked: this filter's input is all zeros after
    # ReLU, so even a weight near 1e37 changes nothing. Not a failed injection.
    with torch.no_grad(), weight_fault(model, weight_spec()) as values:
        assert abs(values.flipped) > 1e30
        assert torch.equal(model(image), clean_logits)


@pytest.mark.parametrize("layer, element", [("layer3.1.conv2", CONV_WEIGHT), ("fc", (3, 7))])
@pytest.mark.parametrize("bit, group", [
    (31, BitGroup.SIGN), (30, BitGroup.EXPONENT), (23, BitGroup.EXPONENT),
    (22, BitGroup.MANTISSA), (0, BitGroup.MANTISSA),
])
def test_weights_restored_bit_for_bit(model, image, clean_logits, layer, element, bit, group):
    before = bits_of(model)
    with torch.no_grad(), weight_fault(model, weight_spec(layer, element, bit, group)):
        model(image)
    assert bits_of(model) == before
    with torch.no_grad():
        assert torch.equal(model(image), clean_logits)


def test_weights_restored_and_error_propagates_after_crash(model):
    before = bits_of(model)
    with pytest.raises(RuntimeError, match="simulated crash"):
        with weight_fault(model, weight_spec()):
            raise RuntimeError("simulated crash")
    assert bits_of(model) == before


def test_flip_can_produce_inf_and_nan_and_still_restores(model):
    fault = weight_spec(bit=30)
    weight = model.get_submodule(fault.layer).weight
    keep = weight.data[CONV_WEIGHT].clone()
    # 1.0 flips to +inf; a value with all exponent bits but one set flips to NaN.
    weight.data[CONV_WEIGHT] = 1.0
    with weight_fault(model, fault) as values:
        assert values.original == 1.0 and math.isinf(values.flipped)
    assert weight.data[CONV_WEIGHT].item() == 1.0
    weight.data[CONV_WEIGHT] = float(np.array(0x3FC00000, dtype=np.uint32).view(np.float32))   # 1.5
    with weight_fault(model, fault) as values:
        assert math.isnan(values.flipped)
        assert torch.isnan(weight.data[CONV_WEIGHT])
    assert weight.data[CONV_WEIGHT].item() == 1.5
    weight.data[CONV_WEIGHT] = keep


@pytest.mark.parametrize("bad", [activation_spec(), weight_spec(group=BitGroup.SIGN)])
def test_weight_fault_rejects_wrong_site_or_group(model, bad):
    with pytest.raises(ValueError):
        with weight_fault(model, bad):
            pass


def test_non_float32_weights_are_rejected_and_restored():
    model = build_resnet18().double().eval()
    with pytest.raises(ValueError, match="float32"):
        with weight_fault(model, weight_spec()):
            pass


def test_activation_fault_changes_output_and_reports_values(injector, image, clean_logits):
    corrupted, values = injector.inject(activation_spec("fc", (2,), 30))
    assert values == []
    with torch.no_grad():
        logits = corrupted(image)
    assert not torch.equal(logits, clean_logits)
    assert len(values) == 1


def test_activation_flip_is_exactly_one_bit_of_the_right_element(model, injector, image):
    fault = activation_spec(layer="layer2.0.conv1", element=(5, 3, 2), bit=22, group=BitGroup.MANTISSA)
    clean_out = output_of(model, fault.layer, image)
    corrupted, values = injector.inject(fault)
    faulty_out = output_of(corrupted, fault.layer, image)

    a = clean_out.numpy().view(np.uint32)
    b = faulty_out.numpy().view(np.uint32)
    assert np.argwhere(a != b).tolist() == [[0, 5, 3, 2]]
    assert int(a[0, 5, 3, 2] ^ b[0, 5, 3, 2]) == 1 << 22
    assert values[0].original == clean_out[0, 5, 3, 2].item()
    assert values[0].flipped == faulty_out[0, 5, 3, 2].item()


def test_activation_fault_leaves_original_model_alone(model, injector, image, clean_logits):
    before = bits_of(model)
    corrupted, _ = injector.inject(activation_spec())
    with torch.no_grad():
        corrupted(image)
        assert torch.equal(model(image), clean_logits)
    assert bits_of(model) == before
    assert sum(len(m._forward_hooks) for m in model.modules()) == 0


def test_fc_activation_fault_works_and_monitor_sees_it(injector, image, clean_logits):
    # fc is both a fault target and a monitored point: the fault must land first.
    fault = activation_spec(layer="fc", element=(2,), bit=31, group=BitGroup.SIGN)
    corrupted, values = injector.inject(fault)
    with torch.no_grad(), ActivationMonitor(corrupted) as monitor:
        logits = corrupted(image)
    assert logits[0, 2].item() == -clean_logits[0, 2].item()
    assert torch.equal(monitor.activations["fc"], logits)
    assert values[0].flipped == logits[0, 2].item()


def test_monitor_sees_a_fault_in_an_earlier_layer(model, injector, image):
    with torch.no_grad(), ActivationMonitor(model) as clean:
        model(image)
    # Use the largest output so the flip stays positive and ReLU doesn't erase it.
    output = output_of(model, "layer1.0.conv1", image)[0]
    element = tuple(int(i) for i in np.unravel_index(int(output.argmax()), output.shape))
    corrupted, _ = injector.inject(activation_spec(layer="layer1.0.conv1", element=element, bit=30))
    with torch.no_grad(), ActivationMonitor(corrupted) as faulty:
        corrupted(image)
    assert torch.equal(faulty.activations["stem"], clean.activations["stem"])
    assert not torch.equal(faulty.activations["layer1"], clean.activations["layer1"])


def test_faults_do_not_leak_between_injections(injector, image, clean_logits):
    first, first_values = injector.inject(activation_spec(bit=30))
    second, second_values = injector.inject(activation_spec(bit=22, group=BitGroup.MANTISSA))
    with torch.no_grad():
        second(image)
    assert len(second_values) == 1 and first_values == []
    third, third_values = injector.inject(activation_spec(element=(1, 1, 1), bit=22, group=BitGroup.MANTISSA))
    with torch.no_grad():
        third(image)
    assert len(third_values) == 1


def test_activation_fault_rejects_wrong_site_and_out_of_bounds(injector):
    with pytest.raises(ValueError):
        injector.inject(weight_spec())
    with pytest.raises(AssertionError):
        injector.inject(activation_spec(element=(9999, 0, 0)))
