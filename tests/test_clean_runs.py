"""Uses fake images (red channel encodes the class) and a fake model with the monitored modules."""

import json
import math
import runpy
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

import sdc.model.resnet18 as resnet18
from sdc.config import REPO_ROOT
from sdc.data.eurosat import CLASS_NAMES, MEAN, STD, build_splits, load_test_split, verify_splits
from sdc.model.monitor import MONITORED_POINTS
from sdc.model.resnet18 import build_resnet18, checkpoint_sha256
from sdc.observations.clean import (
    CLEAN_ARROW_SCHEMA,
    clean_run_row,
    load_clean_runs,
    run_clean,
)

IMAGES_PER_CLASS = 4
SCRIPT = REPO_ROOT / "scripts" / "02_clean_runs.py"


class ToLogits(torch.nn.Module):
    def forward(self, x):
        red = x[:, 0].mean(dim=(1, 2)) * STD[0] + MEAN[0]
        classes = ((red * 255 - 20) / 20).round().long()
        return torch.nn.functional.one_hot(classes, 10).float() * 5


class FakeResNet(torch.nn.Module):
    """Has every monitored module, and always predicts the true class of a fake image."""

    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.Identity()
        self.layer1 = torch.nn.Identity()
        self.layer2 = torch.nn.Identity()
        self.layer3 = torch.nn.Identity()
        self.layer4 = torch.nn.Identity()
        self.fc = ToLogits()

    def forward(self, x):
        return self.fc(self.layer4(self.layer3(self.layer2(self.layer1(self.maxpool(x))))))


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("eurosat")
    for cls, name in enumerate(CLASS_NAMES):
        (root / name).mkdir()
        for i in range(IMAGES_PER_CLASS):
            Image.new("RGB", (8, 8), (20 + 20 * cls, 10 * i, 0)).save(root / name / f"{name}_{i}.png")
    return root


@pytest.fixture(scope="module")
def indices_file(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("indices") / "test_indices.json"
    path.write_text(json.dumps(list(range(IMAGES_PER_CLASS * 10))))
    return path


@pytest.fixture(scope="module")
def test_ds(data_dir, indices_file):
    return load_test_split(str(data_dir), indices_path=str(indices_file))


@pytest.fixture(scope="module")
def splits(test_ds) -> dict:
    return build_splits(FakeResNet(), test_ds, None, 0.5, seed=42)


def test_row_holds_prediction_logits_and_stats(test_ds, splits):
    entry = splits["fit"][0]
    row = clean_run_row(FakeResNet(), test_ds[entry["image_id"]][0], entry)
    assert row["clean_pred"] == entry["true_class"]
    assert len(row["clean_logits"]) == 10
    assert row["image_path"] == entry["image_path"]
    assert set(row) == set(CLEAN_ARROW_SCHEMA.names)
    assert all(row[f"{p}_nonfinite"] == 0 for p in MONITORED_POINTS)


def test_fc_stats_describe_the_logits(test_ds, splits):
    entry = splits["fit"][0]
    row = clean_run_row(FakeResNet(), test_ds[entry["image_id"]][0], entry)
    assert row["fc_max"] == max(row["clean_logits"]) == 5.0
    assert row["fc_min"] == min(row["clean_logits"]) == 0.0
    assert 0.0 < row["top1_prob"] <= 1.0
    assert math.isfinite(row["entropy"])


def test_wrong_clean_prediction_stops_the_run(test_ds, splits):
    entry = dict(splits["fit"][0])
    entry["true_class"] = (entry["true_class"] + 1) % 10
    with pytest.raises(ValueError, match="clean prediction"):
        clean_run_row(FakeResNet(), test_ds[entry["image_id"]][0], entry)


def test_works_on_a_real_resnet():
    torch.manual_seed(0)
    model = build_resnet18().eval()
    image = torch.randn(3, 224, 224)
    pred = int(model(image.unsqueeze(0)).argmax())
    entry = {"image_id": 0, "image_path": "x.png", "true_class": pred}
    row = clean_run_row(model, image, entry)
    assert row["clean_pred"] == pred
    assert all(math.isfinite(row[f"{p}_std"]) for p in MONITORED_POINTS)
    assert sum(len(m._forward_hooks) for m in model.modules()) == 0


def test_single_image_run_is_deterministic(test_ds, splits):
    entry = splits["eval"][0]
    image = test_ds[entry["image_id"]][0]
    model = FakeResNet()
    assert clean_run_row(model, image, entry) == clean_run_row(model, image, entry)


def test_table_has_one_row_per_image_in_order(test_ds, splits):
    table = run_clean(FakeResNet(), test_ds, splits["fit"])
    assert table.schema.equals(CLEAN_ARROW_SCHEMA)
    assert table["image_id"].to_pylist() == [e["image_id"] for e in splits["fit"]]


def test_verify_accepts_matching_inputs(test_ds, splits):
    verify_splits({**splits, "checkpoint_sha256": "abc"}, test_ds, "abc")


def test_verify_rejects_different_checkpoint(test_ds, splits):
    with pytest.raises(ValueError, match="checkpoint differs"):
        verify_splits({**splits, "checkpoint_sha256": "abc"}, test_ds, "xyz")


def test_verify_rejects_different_image_path(test_ds, splits):
    entry = {**splits["fit"][0], "image_path": "Forest/other.png"}
    changed = {**splits, "checkpoint_sha256": "abc", "fit": [entry]}
    with pytest.raises(ValueError, match="dataset differs"):
        verify_splits(changed, test_ds, "abc")


def test_verify_rejects_different_class(test_ds, splits):
    entry = dict(splits["fit"][0])
    entry["true_class"] = (entry["true_class"] + 1) % 10
    changed = {**splits, "checkpoint_sha256": "abc", "fit": [entry]}
    with pytest.raises(ValueError, match="dataset differs"):
        verify_splits(changed, test_ds, "abc")


def test_verify_rejects_out_of_range_image_id(test_ds, splits):
    entry = {**splits["fit"][0], "image_id": 10_000}
    changed = {**splits, "checkpoint_sha256": "abc", "fit": [entry]}
    with pytest.raises(ValueError, match="outside"):
        verify_splits(changed, test_ds, "abc")


@pytest.fixture
def workspace(tmp_path, data_dir, indices_file, test_ds, splits, monkeypatch) -> Path:
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"pretend weights")
    monkeypatch.setattr(resnet18, "load_checkpoint", lambda path, device: FakeResNet())
    splits_file = tmp_path / "splits.json"
    splits_file.write_text(json.dumps({
        **splits,
        "split_source": "test_indices",
        "checkpoint_sha256": checkpoint_sha256(str(checkpoint)),
    }))
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "paths:\n"
        f"  eurosat_dir: {data_dir}\n"
        f"  checkpoint: {checkpoint}\n"
        f"  test_indices: {indices_file}\n"
        f"  splits: {splits_file}\n"
        f"  clean_runs: {tmp_path / 'clean'}\n"
    )
    return tmp_path


def run_script(monkeypatch, workspace: Path):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(workspace / "overlay.yaml")])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_script_writes_both_sets(monkeypatch, workspace, splits):
    run_script(monkeypatch, workspace)
    for split in ["fit", "eval"]:
        runs = load_clean_runs(str(workspace / "clean" / f"{split}.parquet"))
        assert runs["image_id"].tolist() == [e["image_id"] for e in splits[split]]
    assert sorted(p.name for p in (workspace / "clean").iterdir()) == ["eval.parquet", "fit.parquet"]


def test_script_output_is_reproducible(monkeypatch, workspace):
    run_script(monkeypatch, workspace)
    first = pq.read_table(workspace / "clean" / "fit.parquet")
    (workspace / "clean" / "fit.parquet").unlink()
    (workspace / "clean" / "eval.parquet").unlink()
    run_script(monkeypatch, workspace)
    assert pq.read_table(workspace / "clean" / "fit.parquet").equals(first)


def test_script_refuses_to_replace_existing_files(monkeypatch, workspace):
    run_script(monkeypatch, workspace)
    with pytest.raises(SystemExit, match="already exists"):
        run_script(monkeypatch, workspace)


def test_script_stops_on_different_checkpoint_and_writes_nothing(monkeypatch, workspace):
    (workspace / "fake.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checkpoint differs"):
        run_script(monkeypatch, workspace)
    assert not (workspace / "clean").exists()


def test_script_writes_nothing_if_the_second_set_fails(monkeypatch, workspace):
    splits = json.loads((workspace / "splits.json").read_text())
    # Change the model rather than the splits so verification still passes.
    class WrongOnEval(FakeResNet):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            logits = super().forward(x)
            return logits.roll(1, dims=1) if self.calls > len(splits["fit"]) else logits

    monkeypatch.setattr(resnet18, "load_checkpoint", lambda path, device: WrongOnEval())
    with pytest.raises(ValueError, match="clean prediction"):
        run_script(monkeypatch, workspace)
    assert not (workspace / "clean").exists()
