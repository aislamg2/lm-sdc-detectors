"""Uses fake images: solid colors whose red channel encodes the class."""

import json
import runpy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets

import sdc.data.eurosat as eurosat
import sdc.model.resnet18 as resnet18
from sdc.config import REPO_ROOT
from sdc.data.eurosat import (
    CLASS_NAMES,
    EVAL_TRANSFORM,
    MEAN,
    STD,
    balanced_sample,
    build_splits,
    fit_eval_split,
    load_test_split,
    select_correct_images,
)

IMAGES_PER_CLASS = 6
WRONG_GREEN = 250   # fake model misclassifies images with this much green
SCRIPT = REPO_ROOT / "scripts" / "01_make_splits.py"


class FakeModel(torch.nn.Module):
    """Reads the class from the red channel; gets it wrong when green is high."""

    def forward(self, x):
        red = x[:, 0].mean(dim=(1, 2)) * STD[0] + MEAN[0]
        green = x[:, 1].mean(dim=(1, 2)) * STD[1] + MEAN[1]
        classes = ((red * 255 - 20) / 20).round().long()
        wrong = green * 255 > WRONG_GREEN - 5
        classes = torch.where(wrong, (classes + 1) % 10, classes)
        return torch.nn.functional.one_hot(classes, 10).float()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Six images per class; the last image of class 3 is one the model gets wrong."""
    root = tmp_path_factory.mktemp("eurosat")
    for cls, name in enumerate(CLASS_NAMES):
        (root / name).mkdir()
        for i in range(IMAGES_PER_CLASS):
            wrong = cls == 3 and i == IMAGES_PER_CLASS - 1
            color = (20 + 20 * cls, WRONG_GREEN if wrong else 10 * i, 0)
            Image.new("RGB", (8, 8), color).save(root / name / f"{name}_{i}.png")
    return root


@pytest.fixture(scope="module")
def indices_file(data_dir, tmp_path_factory) -> Path:
    """Treats every fake image as part of the test split."""
    path = tmp_path_factory.mktemp("indices") / "test_indices.json"
    path.write_text(json.dumps(list(range(IMAGES_PER_CLASS * 10))))
    return path


@pytest.fixture(scope="module")
def test_ds(data_dir, indices_file):
    return load_test_split(str(data_dir), indices_path=str(indices_file))


def test_reproduces_the_notebook_split(data_dir):
    full = datasets.ImageFolder(root=str(data_dir))
    n = len(full)
    train, val = int(0.8 * n), int(0.1 * n)
    _, _, expected = random_split(
        full, [train, val, n - train - val], generator=torch.Generator().manual_seed(42),
    )
    assert load_test_split(str(data_dir)).indices == expected.indices


def test_test_split_uses_eval_transform(data_dir):
    assert load_test_split(str(data_dir)).dataset.transform is EVAL_TRANSFORM


def test_indices_file_selects_exactly_those_images(data_dir, tmp_path):
    path = tmp_path / "idx.json"
    path.write_text("[5, 2, 9]")
    assert load_test_split(str(data_dir), indices_path=str(path)).indices == [5, 2, 9]


def test_indices_file_must_be_a_list(data_dir, tmp_path):
    path = tmp_path / "idx.json"
    path.write_text('{"test": [1]}')
    with pytest.raises(ValueError, match="list"):
        load_test_split(str(data_dir), indices_path=str(path))


def test_wrong_class_folders_are_rejected(tmp_path):
    (tmp_path / "Forest").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "Forest" / "a.png")
    with pytest.raises(ValueError, match="expected classes"):
        load_test_split(str(tmp_path))


def test_select_correct_finds_every_correct_image(test_ds):
    correct = select_correct_images(FakeModel(), test_ds, batch_size=7)
    wrong_id = 3 * IMAGES_PER_CLASS + IMAGES_PER_CLASS - 1
    assert correct == [i for i in range(60) if i != wrong_id]


def test_select_correct_does_not_depend_on_batch_size(test_ds):
    assert (select_correct_images(FakeModel(), test_ds, batch_size=1)
            == select_correct_images(FakeModel(), test_ds, batch_size=32))


def labeled(per_class_counts: list[int]) -> tuple[list[int], list[int]]:
    labels = [cls for cls, n in enumerate(per_class_counts) for _ in range(n)]
    return list(range(len(labels))), labels


def test_balanced_sample_takes_same_number_per_class():
    indices, labels = labeled([10] * 10)
    sampled = balanced_sample(indices, labels, 4, seed=0)
    assert Counter(labels[i] for i in sampled) == {cls: 4 for cls in range(10)}
    assert len(set(sampled)) == 40


def test_balanced_sample_defaults_to_smallest_class():
    indices, labels = labeled([10, 3] + [10] * 8)
    assert len(balanced_sample(indices, labels, None, seed=0)) == 30


def test_balanced_sample_is_deterministic_and_seed_dependent():
    indices, labels = labeled([50] * 10)
    assert balanced_sample(indices, labels, 5, seed=1) == balanced_sample(indices, labels, 5, seed=1)
    assert balanced_sample(indices, labels, 5, seed=1) != balanced_sample(indices, labels, 5, seed=2)


@pytest.mark.parametrize("per_class", [11, 0])
def test_balanced_sample_rejects_impossible_counts(per_class):
    indices, labels = labeled([10] * 10)
    with pytest.raises(ValueError, match="available"):
        balanced_sample(indices, labels, per_class, seed=0)


def test_balanced_sample_rejects_class_with_no_images():
    indices, labels = labeled([10] * 9 + [0])
    with pytest.raises(ValueError, match="available"):
        balanced_sample(indices, labels, None, seed=0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="labels"):
        balanced_sample([1, 2], [0], 1, seed=0)


def test_fit_eval_is_disjoint_complete_and_stratified():
    indices, labels = labeled([10] * 10)
    parts = fit_eval_split(indices, labels, 0.5, seed=0)
    assert not set(parts["fit"]) & set(parts["eval"])
    assert sorted(parts["fit"] + parts["eval"]) == indices
    assert Counter(labels[i] for i in parts["fit"]) == {cls: 5 for cls in range(10)}
    assert Counter(labels[i] for i in parts["eval"]) == {cls: 5 for cls in range(10)}


def test_fit_eval_is_deterministic():
    indices, labels = labeled([10] * 10)
    assert fit_eval_split(indices, labels, 0.5, 3) == fit_eval_split(indices, labels, 0.5, 3)
    assert fit_eval_split(indices, labels, 0.5, 3) != fit_eval_split(indices, labels, 0.5, 4)


def test_fit_eval_rejects_a_class_too_small_to_split():
    indices, labels = labeled([1] * 10)
    with pytest.raises(ValueError, match="too few"):
        fit_eval_split(indices, labels, 0.5, seed=0)


def test_build_splits_filters_balances_and_describes(test_ds):
    splits = build_splits(FakeModel(), test_ds, None, 0.5, seed=42)
    assert splits["num_test_images"] == 60
    assert splits["num_correct"] == 59
    assert splits["clean_accuracy"] == pytest.approx(59 / 60)
    assert splits["per_class"] == 5   # class 3 lost its one wrong image
    assert len(splits["fit"]) + len(splits["eval"]) == 50

    wrong_path = str(Path(CLASS_NAMES[3]) / f"{CLASS_NAMES[3]}_{IMAGES_PER_CLASS - 1}.png")
    everything = splits["fit"] + splits["eval"]
    assert wrong_path not in {e["image_path"] for e in everything}
    for entry in everything:
        assert CLASS_NAMES[entry["true_class"]] == Path(entry["image_path"]).parts[0]
        assert not Path(entry["image_path"]).is_absolute()
    assert not {e["image_id"] for e in splits["fit"]} & {e["image_id"] for e in splits["eval"]}
    json.dumps(splits)   # everything must be JSON-ready


def test_image_id_is_position_in_the_test_split(data_dir, tmp_path):
    # Dataset indices in reverse order, so position and dataset index differ.
    path = tmp_path / "idx.json"
    path.write_text(json.dumps(list(range(59, -1, -1))))
    test_ds = load_test_split(str(data_dir), indices_path=str(path))
    splits = build_splits(FakeModel(), test_ds, None, 0.5, seed=42)
    for entry in splits["fit"] + splits["eval"]:
        sample_path = test_ds.dataset.samples[test_ds.indices[entry["image_id"]]][0]
        assert sample_path.endswith(entry["image_path"])


def run_script(monkeypatch, overlay: Path | None):
    argv = [str(SCRIPT)] + (["--config", str(overlay)] if overlay else [])
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.fixture
def overlay(tmp_path, data_dir, indices_file, monkeypatch) -> Path:
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"pretend weights")
    monkeypatch.setattr(resnet18, "load_checkpoint", lambda path, device: FakeModel())
    path = tmp_path / "overlay.yaml"
    path.write_text(
        "paths:\n"
        f"  eurosat_dir: {data_dir}\n"
        f"  checkpoint: {checkpoint}\n"
        f"  test_indices: {indices_file}\n"
        f"  splits: {tmp_path / 'out' / 'splits.json'}\n"
    )
    return path


def test_script_writes_splits_with_provenance(monkeypatch, overlay, tmp_path, capsys):
    run_script(monkeypatch, overlay)
    splits = json.loads((tmp_path / "out" / "splits.json").read_text())
    assert splits["split_source"] == "test_indices"
    assert splits["checkpoint_sha256"] == resnet18.checkpoint_sha256(str(tmp_path / "fake.pt"))
    assert splits["seed"] == 42
    # 5 per class at 50/50 rounds down: 2 fit and 3 eval per class.
    assert len(splits["fit"]) == 20 and len(splits["eval"]) == 30
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["splits.json"]
    assert "59/60" in capsys.readouterr().out


def test_script_is_reproducible(monkeypatch, overlay, tmp_path):
    out = tmp_path / "out" / "splits.json"
    run_script(monkeypatch, overlay)
    first = out.read_text()
    out.unlink()
    run_script(monkeypatch, overlay)
    assert out.read_text() == first


def test_script_refuses_to_replace_existing_splits(monkeypatch, overlay, tmp_path):
    run_script(monkeypatch, overlay)
    out = tmp_path / "out" / "splits.json"
    before = out.read_text()
    with pytest.raises(SystemExit, match="already exists"):
        run_script(monkeypatch, overlay)
    assert out.read_text() == before


def test_script_falls_back_to_seed_split(monkeypatch, overlay, tmp_path, test_ds):
    overlay.write_text(overlay.read_text().replace("test_indices.json", "missing.json"))
    calls = []

    def spy(data_dir, fractions, seed, indices_path=None):
        calls.append((list(fractions), seed, indices_path))
        return test_ds   # the fake seed split is too small to cover all 10 classes

    monkeypatch.setattr(eurosat, "load_test_split", spy)
    run_script(monkeypatch, overlay)
    assert calls == [([0.8, 0.1, 0.1], 42, None)]
    splits = json.loads((tmp_path / "out" / "splits.json").read_text())
    assert splits["split_source"] == "reproduced_seed_split"


def test_script_fails_without_writing_if_a_class_runs_short(monkeypatch, overlay, tmp_path):
    overlay.write_text(overlay.read_text() + "data:\n  images_per_class: 6\n")
    with pytest.raises(ValueError, match="available"):
        run_script(monkeypatch, overlay)   # class 3 only has 5 correct images
    assert not (tmp_path / "out" / "splits.json").exists()


def test_checkpoint_hash_matches_known_value(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert resnet18.checkpoint_sha256(str(path)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
