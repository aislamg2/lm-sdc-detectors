import pickle

import pytest
import torch

from sdc.model.resnet18 import build_resnet18, load_checkpoint


@pytest.fixture
def saved(tmp_path):
    """A randomly initialised model saved the way the checkpoint should be."""
    torch.manual_seed(0)
    model = build_resnet18().eval()
    path = tmp_path / "resnet18.pt"
    torch.save(model.state_dict(), path)
    return model, str(path)


def test_loaded_weights_match_the_saved_ones(saved):
    original, path = saved
    loaded = load_checkpoint(path)
    original_state = original.state_dict()
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, original_state[name]), name


def test_loaded_model_gives_identical_output(saved):
    original, path = saved
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(load_checkpoint(path)(x), original(x))


def test_loaded_model_is_in_eval_mode(saved):
    _, path = saved
    model = load_checkpoint(path)
    assert not model.training
    assert not any(m.training for m in model.modules())


def test_inference_does_not_change_batch_norm_statistics(saved):
    _, path = saved
    model = load_checkpoint(path)
    before = model.bn1.running_mean.clone()
    with torch.no_grad():
        model(torch.randn(2, 3, 224, 224))
    assert torch.equal(model.bn1.running_mean, before)


def test_weights_are_float32(saved):
    # Bit flips assume FP32.
    _, path = saved
    assert all(p.dtype == torch.float32 for p in load_checkpoint(path).parameters())


def test_missing_weight_fails(saved, tmp_path):
    original, _ = saved
    state = original.state_dict()
    del state["fc.weight"]
    path = tmp_path / "missing.pt"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="fc.weight"):
        load_checkpoint(str(path))


def test_wrong_number_of_classes_fails(tmp_path):
    path = tmp_path / "wrong_classes.pt"
    torch.save(build_resnet18(num_classes=5).state_dict(), path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(str(path))


def test_wrapped_checkpoint_fails(saved, tmp_path):
    original, _ = saved
    path = tmp_path / "wrapped.pt"
    torch.save({"model": original.state_dict(), "epoch": 5}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(str(path))


def test_whole_pickled_model_is_refused(saved, tmp_path):
    original, _ = saved
    path = tmp_path / "whole_model.pt"
    torch.save(original, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(str(path))


def test_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "nope.pt"))
