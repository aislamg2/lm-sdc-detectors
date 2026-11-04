"""Uses fake images and a real ResNet-18 with random weights, so labels mean nothing here."""

import json
import math
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from sdc.config import REPO_ROOT, load_config
from sdc.data.eurosat import CLASS_NAMES, load_test_split
from sdc.injection import campaign
from sdc.injection.campaign import (
    PlannedRun,
    RunContext,
    current_git_commit,
    plan_faults,
    plan_hash,
    run_campaign,
    run_one,
    time_limit,
    weights_fingerprint,
)
from sdc.injection.faults import ActivationInjector, FaultSpec, activation_shapes, weight_shapes
from sdc.model.monitor import MONITORED_POINTS
from sdc.model.resnet18 import build_resnet18, checkpoint_sha256
from sdc.observations.clean import CLEAN_ARROW_SCHEMA, clean_run_row
from sdc.records.io import ARROW_SCHEMA, completed_shards, load_runs, record_to_row
from sdc.records.schema import BitGroup, CrashReason, FaultSite, Label

SITES = [FaultSite.WEIGHT, FaultSite.ACTIVATION]
GROUPS = list(BitGroup)
SCRIPT = REPO_ROOT / "scripts" / "05_inject.py"
VOLATILE = ["timestamp", "inference_time_s", "git_commit"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_resnet18().eval()


@pytest.fixture(scope="module")
def shapes(model):
    return {FaultSite.WEIGHT: weight_shapes(model), FaultSite.ACTIVATION: activation_shapes(model)}


@pytest.fixture(scope="module")
def injector(model):
    return ActivationInjector(model)


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(1)
    return torch.randn(3, 224, 224)


@pytest.fixture(scope="module")
def clean_pred(model, image):
    with torch.no_grad():
        return int(model(image.unsqueeze(0)).argmax())


@pytest.fixture
def context():
    return RunContext(42, "fit", "abc", "def", "cpu", timeout_s=30)


ENTRY = {"image_path": "Forest/1.png", "true_class": 1}


def weight_run(layer="layer3.1.conv2", element=(0, 0, 0, 0), bit=22, group=BitGroup.MANTISSA):
    return PlannedRun(5, 7, FaultSpec(FaultSite.WEIGHT, layer, element, bit, group))


def activation_run(layer="fc", element=(2,), bit=31, group=BitGroup.SIGN):
    return PlannedRun(6, 7, FaultSpec(FaultSite.ACTIVATION, layer, element, bit, group))


def plan_of(shapes, image_ids=(0, 1), injections=2, seed=42, split="fit"):
    return plan_faults(list(image_ids), injections, SITES, GROUPS, shapes, seed, split)


# Planning

def test_plan_has_every_combination(shapes):
    plan = plan_of(shapes)
    assert len(plan) == 2 * 2 * 2 * 3
    assert [run.plan_idx for run in plan] == list(range(24))
    for image_id in (0, 1):
        runs = [run for run in plan if run.image_id == image_id]
        combos = {(run.fault.site, run.fault.bit_group) for run in runs}
        assert len(combos) == 6


def test_plan_is_deterministic_and_seed_dependent(shapes):
    assert plan_of(shapes) == plan_of(shapes)
    assert plan_hash(plan_of(shapes)) == plan_hash(plan_of(shapes))
    assert plan_hash(plan_of(shapes)) != plan_hash(plan_of(shapes, seed=1))


def test_plan_differs_between_splits(shapes):
    assert plan_hash(plan_of(shapes)) != plan_hash(plan_of(shapes, split="eval"))


def test_a_fault_does_not_depend_on_the_other_images(shapes):
    alone = plan_of(shapes, image_ids=[1])
    together = plan_of(shapes, image_ids=[0, 1])
    assert [r.fault for r in alone] == [r.fault for r in together if r.image_id == 1]


def test_plan_stays_in_bounds_and_varies(shapes):
    plan = plan_of(shapes, image_ids=range(20))
    for run in plan:
        limits = shapes[run.fault.site][run.fault.layer]
        assert all(0 <= i < n for i, n in zip(run.fault.element, limits))
    assert len({run.fault.layer for run in plan}) > 10


# Helpers

def test_time_limit_raises_and_clears_itself():
    with pytest.raises(TimeoutError):
        with time_limit(0.05):
            time.sleep(5)
    time.sleep(0.1)   # no alarm left over


def test_time_limit_does_nothing_for_fast_code():
    with time_limit(5):
        pass


def test_fingerprint_detects_a_single_bit(model):
    before = weights_fingerprint(model)
    weight = model.fc.weight.data
    keep = weight[0, 0].clone()
    weight[0, 0] = -keep
    assert weights_fingerprint(model) != before
    weight[0, 0] = keep
    assert weights_fingerprint(model) == before


def test_git_commit_is_a_string():
    assert isinstance(current_git_commit(), str) and current_git_commit()


# One run

def test_weight_run_produces_a_complete_record(model, injector, image, clean_pred, context):
    record = run_one(weight_run(), image, ENTRY, clean_pred, model, injector, context)
    assert record.run_id == "fit-00000005"
    assert record.fault_site == FaultSite.WEIGHT
    assert record.original_value != record.flipped_value
    assert record.label in (Label.MASKED, Label.SDC)
    assert record.crash_reason is None
    assert len(record.faulty_logits) == 10
    assert record.faulty_pred == int(np.argmax(record.faulty_logits))
    assert set(record.layer_stats) == set(MONITORED_POINTS)
    assert record.inference_time_s > 0
    assert set(record_to_row(record)) == set(ARROW_SCHEMA.names)


def test_weight_run_leaves_weights_unchanged(model, injector, image, clean_pred, context):
    before = weights_fingerprint(model)
    run_one(weight_run(bit=30, group=BitGroup.EXPONENT), image, ENTRY, clean_pred, model, injector, context)
    assert weights_fingerprint(model) == before


def test_activation_run_records_the_flip(model, injector, image, clean_pred, context):
    record = run_one(activation_run(), image, ENTRY, clean_pred, model, injector, context)
    assert record.fault_site == FaultSite.ACTIVATION
    assert record.flipped_value == -record.original_value
    assert record.faulty_logits[2] == pytest.approx(record.flipped_value)


def test_sdc_is_labeled_when_the_prediction_changes(model, injector, image, clean_pred, context):
    # Flipping the sign of the winning logit, if positive, normally changes the prediction.
    with torch.no_grad():
        logits = model(image.unsqueeze(0))[0]
    assert logits[clean_pred] > 0
    run = activation_run(element=(clean_pred,))
    record = run_one(run, image, ENTRY, clean_pred, model, injector, context)
    assert record.label == Label.SDC
    assert record.faulty_pred != clean_pred


def test_non_finite_logits_become_a_nonfinite_crash(model, injector, image, clean_pred, context):
    def poison(module, inputs, output):
        output[0, 0] = math.nan

    handle = model.fc.register_forward_hook(poison)
    try:
        record = run_one(weight_run(), image, ENTRY, clean_pred, model, injector, context)
    finally:
        handle.remove()
    assert record.label == Label.CRASH
    assert record.crash_reason == CrashReason.NONFINITE
    assert record.faulty_pred is None
    assert math.isnan(record.faulty_logits[0])
    assert math.isnan(record.top1_prob)
    assert record.layer_stats["fc"].nonfinite == 1


def test_exception_becomes_a_crash_and_restores_weights(model, injector, image, clean_pred, context):
    def explode(module, inputs, output):
        raise RuntimeError("simulated fault-induced error")

    before = weights_fingerprint(model)
    handle = model.layer2.register_forward_hook(explode)
    try:
        record = run_one(weight_run(), image, ENTRY, clean_pred, model, injector, context)
    finally:
        handle.remove()
    assert record.label == Label.CRASH
    assert record.crash_reason == CrashReason.EXCEPTION
    assert record.faulty_logits is None and record.top1_prob is None
    assert set(record.layer_stats) == {"stem", "layer1"}   # points reached before the error
    assert record.original_value is not None
    assert weights_fingerprint(model) == before
    assert sum(len(m._forward_hooks) for m in model.modules()) == 0   # the monitor cleaned up


def test_timeout_becomes_a_crash_and_restores_weights(model, injector, image, clean_pred):
    def stall(module, inputs, output):
        time.sleep(5)

    context = RunContext(42, "fit", "abc", "def", "cpu", timeout_s=0.1)
    before = weights_fingerprint(model)
    handle = model.layer1.register_forward_hook(stall)
    try:
        start = time.perf_counter()
        record = run_one(weight_run(bit=30, group=BitGroup.EXPONENT), image, ENTRY, clean_pred,
                         model, injector, context)
    finally:
        handle.remove()
    assert time.perf_counter() - start < 2
    assert record.crash_reason == CrashReason.TIMEOUT
    assert weights_fingerprint(model) == before


def test_bad_fault_setup_stops_instead_of_becoming_a_crash(model, injector, image, clean_pred, context):
    bad = PlannedRun(0, 0, FaultSpec(FaultSite.ACTIVATION, "fc", (9999,), 31, BitGroup.SIGN))
    with pytest.raises(AssertionError):
        run_one(bad, image, ENTRY, clean_pred, model, injector, context)


# Whole campaign

@pytest.fixture(scope="module")
def workspace(tmp_path_factory, model) -> Path:
    """Fake dataset, real checkpoint, and the splits and clean runs a campaign needs."""
    root = tmp_path_factory.mktemp("campaign")
    data_dir = root / "data"
    rng = np.random.default_rng(0)
    for name in CLASS_NAMES:
        (data_dir / name).mkdir(parents=True)
        pixels = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(data_dir / name / f"{name}_0.png")

    (root / "test_indices.json").write_text(json.dumps(list(range(10))))
    checkpoint = root / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    test_ds = load_test_split(str(data_dir), indices_path=str(root / "test_indices.json"))

    def entry(image_id):
        path, label = test_ds.dataset.samples[test_ds.indices[image_id]]
        return {
            "image_id": image_id,
            "image_path": str(Path(path).relative_to(data_dir)),
            "true_class": label,
        }

    splits = {"fit": [entry(i) for i in range(3)], "eval": [entry(i) for i in range(3, 5)]}
    (root / "splits.json").write_text(json.dumps({
        **splits, "split_source": "test_indices", "checkpoint_sha256": checkpoint_sha256(str(checkpoint)),
    }))

    (root / "clean").mkdir()
    for split, entries in splits.items():
        rows = []
        for e in entries:
            # Random weights rarely predict the folder's class, so take whatever they predict.
            pred = int(model(test_ds[e["image_id"]][0].unsqueeze(0)).argmax())
            rows.append(clean_run_row(model, test_ds[e["image_id"]][0], {**e, "true_class": pred}, "cpu"))
            rows[-1]["true_class"] = e["true_class"]
        pq.write_table(pa.Table.from_pylist(rows, schema=CLEAN_ARROW_SCHEMA), root / "clean" / f"{split}.parquet")
    return root


def write_overlay(workspace: Path, out: Path, injections=1, seed=42, shard_size=5) -> Path:
    path = out.parent / f"{out.name}_overlay.yaml"
    path.write_text(
        f"seed: {seed}\n"
        "paths:\n"
        f"  eurosat_dir: {workspace / 'data'}\n"
        f"  checkpoint: {workspace / 'model.pt'}\n"
        f"  test_indices: {workspace / 'test_indices.json'}\n"
        f"  splits: {workspace / 'splits.json'}\n"
        f"  clean_runs: {workspace / 'clean'}\n"
        f"  fault_runs: {out}\n"
        "injection:\n"
        f"  injections_per_image: {injections}\n"
        f"  shard_size: {shard_size}\n"
    )
    return path


def config_for(workspace, out, **kwargs) -> dict:
    return load_config(write_overlay(workspace, out, **kwargs))


def stable(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.drop(columns=VOLATILE).astype(str)


def test_campaign_writes_every_run(workspace, tmp_path):
    out = tmp_path / "out"
    status = run_campaign(config_for(workspace, out), "fit", out / "fit")
    assert status == {"total_shards": 4, "written": 4, "skipped": 0, "remaining": 0}
    runs = load_runs(out / "fit")
    assert len(runs) == 3 * 6
    assert runs["plan_idx"].tolist() == list(range(18))
    assert set(runs["split"]) == {"fit"}
    assert set(runs["image_id"]) == {0, 1, 2}
    assert runs["faulty_logits"].notna().all()


def test_manifest_records_the_campaign(workspace, tmp_path):
    out = tmp_path / "out"
    run_campaign(config_for(workspace, out), "fit", out / "fit")
    manifest = json.loads((out / "fit" / "campaign.json").read_text())
    assert manifest["num_runs"] == 18
    assert manifest["checkpoint_sha256"] == checkpoint_sha256(str(workspace / "model.pt"))
    assert manifest["schema_version"] == 1


def test_stopped_and_resumed_matches_uninterrupted(workspace, tmp_path):
    whole = tmp_path / "whole"
    run_campaign(config_for(workspace, whole), "fit", whole / "fit")

    parts = tmp_path / "parts"
    config = config_for(workspace, parts)
    assert run_campaign(config, "fit", parts / "fit", max_shards=1)["remaining"] == 3
    assert completed_shards(parts / "fit") == {0}
    status = run_campaign(config, "fit", parts / "fit", max_shards=2)
    assert (status["skipped"], status["written"], status["remaining"]) == (1, 2, 1)
    run_campaign(config, "fit", parts / "fit")
    pd.testing.assert_frame_equal(stable(load_runs(whole / "fit")), stable(load_runs(parts / "fit")))


def test_interrupt_mid_shard_loses_only_that_shard(workspace, tmp_path, monkeypatch):
    whole = tmp_path / "whole"
    run_campaign(config_for(workspace, whole), "fit", whole / "fit")

    out = tmp_path / "out"
    config = config_for(workspace, out)
    calls = {"n": 0}
    real_run_one = campaign.run_one

    def interrupted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 8:   # partway through the second shard
            raise KeyboardInterrupt
        return real_run_one(*args, **kwargs)

    monkeypatch.setattr(campaign, "run_one", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_campaign(config, "fit", out / "fit")
    assert completed_shards(out / "fit") == {0}
    monkeypatch.undo()

    status = run_campaign(config, "fit", out / "fit")
    assert status["skipped"] == 1
    pd.testing.assert_frame_equal(stable(load_runs(whole / "fit")), stable(load_runs(out / "fit")))


def test_finished_campaign_does_nothing_when_rerun(workspace, tmp_path):
    out = tmp_path / "out"
    config = config_for(workspace, out)
    run_campaign(config, "fit", out / "fit")
    before = {p.name: p.stat().st_mtime_ns for p in (out / "fit").iterdir()}
    status = run_campaign(config, "fit", out / "fit")
    assert status == {"total_shards": 4, "written": 0, "skipped": 4, "remaining": 0}
    assert {p.name: p.stat().st_mtime_ns for p in (out / "fit").iterdir()} == before


@pytest.mark.parametrize("changed, name", [
    ({"seed": 7}, "seed"),
    ({"injections": 2}, "injections_per_image"),
    ({"shard_size": 4}, "shard_size"),
])
def test_resuming_with_different_settings_stops(workspace, tmp_path, changed, name):
    out = tmp_path / "out"
    run_campaign(config_for(workspace, out), "fit", out / "fit", max_shards=1)
    with pytest.raises(ValueError, match=name):
        run_campaign(config_for(workspace, out, **changed), "fit", out / "fit")
    assert completed_shards(out / "fit") == {0}


def test_shards_without_a_manifest_stop(workspace, tmp_path):
    out = tmp_path / "out"
    config = config_for(workspace, out)
    run_campaign(config, "fit", out / "fit", max_shards=1)
    (out / "fit" / "campaign.json").unlink()
    with pytest.raises(ValueError, match="no campaign.json"):
        run_campaign(config, "fit", out / "fit")


def test_injections_per_image_must_be_set(workspace, tmp_path):
    out = tmp_path / "out"
    config = config_for(workspace, out)
    config["injection"]["injections_per_image"] = None
    with pytest.raises(ValueError, match="pilot"):
        run_campaign(config, "fit", out / "fit")


def test_different_checkpoint_stops(workspace, tmp_path):
    out = tmp_path / "out"
    config = config_for(workspace, out)
    other = tmp_path / "other.pt"
    torch.save(build_resnet18().state_dict(), other)
    config["paths"]["checkpoint"] = str(other)
    with pytest.raises(ValueError, match="checkpoint differs"):
        run_campaign(config, "fit", out / "fit")
    assert not (out / "fit").exists()


def test_clean_runs_for_other_images_stop(workspace, tmp_path):
    out = tmp_path / "out"
    config = config_for(workspace, out)
    (tmp_path / "clean").mkdir()
    table = pq.read_table(workspace / "clean" / "eval.parquet")
    pq.write_table(table, tmp_path / "clean" / "fit.parquet")
    config["paths"]["clean_runs"] = str(tmp_path / "clean")
    with pytest.raises(ValueError, match="doesn't match"):
        run_campaign(config, "fit", out / "fit")


def test_wrong_saved_clean_prediction_stops(workspace, tmp_path):
    out = tmp_path / "out"
    config = config_for(workspace, out)
    (tmp_path / "clean").mkdir()
    rows = pq.read_table(workspace / "clean" / "fit.parquet").to_pylist()
    rows[0]["clean_pred"] = (rows[0]["clean_pred"] + 1) % 10
    pq.write_table(pa.Table.from_pylist(rows, schema=CLEAN_ARROW_SCHEMA), tmp_path / "clean" / "fit.parquet")
    config["paths"]["clean_runs"] = str(tmp_path / "clean")
    with pytest.raises(ValueError, match="clean run saved"):
        run_campaign(config, "fit", out / "fit")


def test_unrestored_fault_stops_before_saving_the_shard(workspace, tmp_path, monkeypatch):
    out = tmp_path / "out"
    real_run_one = campaign.run_one

    def corrupting(run, image, entry, clean_pred, model, injector, context):
        record = real_run_one(run, image, entry, clean_pred, model, injector, context)
        model.fc.weight.data[0, 0] += 1.0
        return record

    monkeypatch.setattr(campaign, "run_one", corrupting)
    with pytest.raises(RuntimeError, match="not restored"):
        run_campaign(config_for(workspace, out), "fit", out / "fit")
    assert completed_shards(out / "fit") == set()


def test_shard_of_only_exceptions_stops_as_a_likely_bug(workspace, tmp_path, monkeypatch):
    out = tmp_path / "out"

    def always_fails(model, batch, timeout_s, outcome):
        outcome.crash_reason = CrashReason.EXCEPTION
        outcome.activations = {}

    monkeypatch.setattr(campaign, "_infer", always_fails)
    with pytest.raises(RuntimeError, match="looks like a bug"):
        run_campaign(config_for(workspace, out), "fit", out / "fit")
    assert completed_shards(out / "fit") == set()


def run_script(monkeypatch, overlay: Path, *args):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(overlay), *args])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_script_runs_both_splits_and_prints_a_summary(workspace, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    run_script(monkeypatch, write_overlay(workspace, out))
    assert len(load_runs(out / "fit")) == 18
    assert len(load_runs(out / "eval")) == 12
    printed = capsys.readouterr().out
    assert "fit: 18 runs saved" in printed and "eval: 12 runs saved" in printed


def test_script_can_run_one_split_and_stop_early(workspace, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    run_script(monkeypatch, write_overlay(workspace, out), "--split", "eval", "--max-shards", "1")
    assert completed_shards(out / "eval") == {0}
    assert not (out / "fit").exists()
    assert "remaining" in capsys.readouterr().out


def test_script_reports_an_interrupt_cleanly(workspace, tmp_path, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(campaign, "run_campaign", interrupt)
    with pytest.raises(SystemExit, match="resume"):
        run_script(monkeypatch, write_overlay(workspace, tmp_path / "out"))
