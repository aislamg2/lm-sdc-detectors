import pytest
import torch

from sdc.model.monitor import MONITORED_POINTS, POINT_TO_MODULE, ActivationMonitor
from sdc.model.resnet18 import build_resnet18


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_resnet18().eval()


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(1)
    return torch.randn(1, 3, 224, 224)


def hook_count(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


def manual_pass(model, x) -> dict[str, torch.Tensor]:
    """Run ResNet-18 layer by layer, copying each monitored output."""
    out = {}
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    out["stem"] = x.clone()
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        x = getattr(model, name)(x)
        out[name] = x.clone()
    x = torch.flatten(model.avgpool(x), 1)
    out["fc"] = model.fc(x).clone()
    return out


def test_captures_all_six_points_with_expected_shapes(model, image):
    with torch.no_grad(), ActivationMonitor(model) as mon:
        model(image)
    assert {p: tuple(a.shape) for p, a in mon.activations.items()} == {
        "stem": (1, 64, 56, 56),
        "layer1": (1, 64, 56, 56),
        "layer2": (1, 128, 28, 28),
        "layer3": (1, 256, 14, 14),
        "layer4": (1, 512, 7, 7),
        "fc": (1, 10),
    }


def test_captured_values_match_a_manual_pass(model, image):
    # Guards the choice to detach without copying: nothing may change a captured
    # output later in the forward pass, e.g. the in-place ReLU.
    with torch.no_grad():
        expected = manual_pass(model, image)
        with ActivationMonitor(model) as mon:
            model(image)
    for point in MONITORED_POINTS:
        assert torch.equal(mon.activations[point], expected[point]), point


def test_fc_capture_is_the_model_output(model, image):
    with torch.no_grad(), ActivationMonitor(model) as mon:
        logits = model(image)
    assert torch.equal(mon.activations["fc"], logits)


def test_monitoring_does_not_change_the_output(model, image):
    with torch.no_grad():
        plain = model(image)
        with ActivationMonitor(model):
            monitored = model(image)
    assert torch.equal(plain, monitored)


def test_hooks_removed_after_normal_exit(model, image):
    before = hook_count(model)
    with torch.no_grad(), ActivationMonitor(model):
        assert hook_count(model) == before + len(MONITORED_POINTS)
        model(image)
    assert hook_count(model) == before


def test_hooks_removed_and_error_propagates_after_crash(model):
    before = hook_count(model)
    with pytest.raises(RuntimeError, match="simulated crash"):
        with ActivationMonitor(model):
            raise RuntimeError("simulated crash")
    assert hook_count(model) == before


def test_no_capture_outside_the_with_block(model, image):
    with torch.no_grad():
        with ActivationMonitor(model) as mon:
            pass
        model(image)
    assert mon.activations == {}


def test_reusing_a_monitor_clears_the_previous_run(model, image):
    mon = ActivationMonitor(model, points=["stem"])
    with torch.no_grad(), mon:
        model(image)
    first = mon.activations
    assert list(first) == ["stem"]
    with mon:
        assert mon.activations == {}
    assert "stem" in first   # the earlier dict is left intact


def test_partial_run_keeps_only_points_reached(model, image):
    # A crash mid-inference still leaves the points before it available.
    with torch.no_grad(), ActivationMonitor(model) as mon:
        model.maxpool(model.relu(model.bn1(model.conv1(image))))
    assert list(mon.activations) == ["stem"]


def test_subset_of_points(model, image):
    with torch.no_grad(), ActivationMonitor(model, points=["layer2", "fc"]) as mon:
        model(image)
    assert set(mon.activations) == {"layer2", "fc"}


def test_unknown_point_is_rejected(model):
    with pytest.raises(ValueError, match="layer9"):
        ActivationMonitor(model, points=["stem", "layer9"])


def test_missing_module_leaves_no_hooks(model):
    class NoFc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = torch.nn.MaxPool2d(2)

    no_fc = NoFc()
    with pytest.raises(AttributeError):
        with ActivationMonitor(no_fc, points=["stem", "fc"]):
            pass
    assert hook_count(no_fc) == 0


def test_captures_are_detached_when_gradients_are_on(model, image):
    with ActivationMonitor(model, points=["fc"]) as mon:
        logits = model(image)
    assert logits.requires_grad
    assert not mon.activations["fc"].requires_grad


def test_every_point_has_a_module_mapping():
    assert set(MONITORED_POINTS) == set(POINT_TO_MODULE)
