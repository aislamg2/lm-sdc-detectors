import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sdc.config import REPO_ROOT
from sdc.model.monitor import MONITORED_POINTS
from sdc.model.resnet18 import checkpoint_sha256
from sdc.observations.clean import CLEAN_ARROW_SCHEMA, load_clean_runs
from sdc.observations.stats import reference_ranges

SCRIPT = REPO_ROOT / "scripts" / "03_reference_ranges.py"


def make_row(image_id: int, low: float, high: float, nonfinite: int = 0) -> dict:
    row = {
        "image_id": image_id, "image_path": f"Forest/{image_id}.jpg", "true_class": 1,
        "clean_pred": 1, "clean_logits": [0.0] * 10, "top1_prob": 0.9, "entropy": 0.1,
    }
    for point in MONITORED_POINTS:
        row.update({
            f"{point}_min": low, f"{point}_max": high, f"{point}_mean": 0.0,
            f"{point}_std": 1.0, f"{point}_nonfinite": nonfinite,
        })
    return row


def write_runs(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CLEAN_ARROW_SCHEMA), path)
    return path


def load_rows(tmp_path, rows):
    return load_clean_runs(str(write_runs(tmp_path / "runs.parquet", rows)))


def test_min_and_max_are_the_extremes_across_runs(tmp_path):
    runs = load_rows(tmp_path, [make_row(0, -1.0, 2.0), make_row(1, -3.0, 1.0), make_row(2, 0.0, 5.0)])
    ranges = reference_ranges(runs)
    assert set(ranges) == set(MONITORED_POINTS)
    assert ranges["layer3"]["min"] == -3.0
    assert ranges["layer3"]["max"] == 5.0


def test_points_are_computed_separately(tmp_path):
    rows = [make_row(0, -1.0, 2.0)]
    rows[0]["fc_max"] = 99.0
    ranges = reference_ranges(load_rows(tmp_path, rows))
    assert ranges["fc"]["max"] == 99.0
    assert ranges["stem"]["max"] == 2.0


def test_quantile_bounds_ignore_a_single_extreme_run(tmp_path):
    rows = [make_row(i, -1.0 - i / 1000, 1.0 + i / 1000) for i in range(2000)]
    rows.append(make_row(2000, -1000.0, 1000.0))
    bounds = reference_ranges(load_rows(tmp_path, rows))["layer1"]
    assert bounds["min"] == -1000.0 and bounds["max"] == 1000.0
    assert -3.0 < bounds["quantile_low"] < -2.0
    assert 2.0 < bounds["quantile_high"] < 3.0


def test_quantiles_match_numpy(tmp_path):
    mins = [float(i) for i in range(11)]
    rows = [make_row(i, low, low + 100) for i, low in enumerate(mins)]
    bounds = reference_ranges(load_rows(tmp_path, rows))["stem"]
    assert bounds["quantile_low"] == pytest.approx(np.quantile(mins, 0.001))
    assert bounds["quantile_high"] == pytest.approx(np.quantile([m + 100 for m in mins], 0.999))


def test_quantile_bounds_stay_inside_min_and_max(tmp_path):
    rng = np.random.default_rng(0)
    rows = [make_row(i, float(-rng.random()), float(rng.random())) for i in range(50)]
    for bounds in reference_ranges(load_rows(tmp_path, rows)).values():
        assert bounds["min"] <= bounds["quantile_low"]
        assert bounds["quantile_high"] <= bounds["max"]


def test_results_are_plain_floats_ready_for_json(tmp_path):
    ranges = reference_ranges(load_rows(tmp_path, [make_row(0, 0.0, 1.0)]))
    assert all(type(v) is float for bounds in ranges.values() for v in bounds.values())
    json.dumps(ranges)


def test_non_finite_clean_run_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        reference_ranges(load_rows(tmp_path, [make_row(0, 0.0, 1.0, nonfinite=1)]))


def test_no_runs_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no clean runs"):
        reference_ranges(load_rows(tmp_path, []))


@pytest.fixture
def workspace(tmp_path) -> Path:
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"pretend weights")
    (tmp_path / "splits.json").write_text(json.dumps({
        "checkpoint_sha256": checkpoint_sha256(str(checkpoint)),
        "fit": [{"image_id": 0}, {"image_id": 1}],
        "eval": [{"image_id": 2}],
    }))
    write_runs(tmp_path / "clean" / "fit.parquet", [make_row(0, -1.0, 2.0), make_row(1, -3.0, 1.0)])
    (tmp_path / "overlay.yaml").write_text(
        "paths:\n"
        f"  checkpoint: {checkpoint}\n"
        f"  splits: {tmp_path / 'splits.json'}\n"
        f"  clean_runs: {tmp_path / 'clean'}\n"
        f"  reference_ranges: {tmp_path / 'out' / 'ranges.json'}\n"
    )
    return tmp_path


def run_script(monkeypatch, workspace: Path):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(workspace / "overlay.yaml")])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_script_writes_ranges_and_provenance(monkeypatch, workspace):
    run_script(monkeypatch, workspace)
    result = json.loads((workspace / "out" / "ranges.json").read_text())
    assert result["num_fit_images"] == 2
    assert result["checkpoint_sha256"] == checkpoint_sha256(str(workspace / "fake.pt"))
    assert result["points"]["layer2"]["min"] == -3.0
    assert result["points"]["layer2"]["max"] == 2.0
    assert (result["quantile_low"], result["quantile_high"]) == (0.001, 0.999)
    assert [p.name for p in (workspace / "out").iterdir()] == ["ranges.json"]


def test_script_never_reads_eval_data(monkeypatch, workspace):
    # A wild eval value and an unreadable eval file must both change nothing.
    write_runs(workspace / "clean" / "eval.parquet", [make_row(2, -1e9, 1e9)])
    run_script(monkeypatch, workspace)
    with_eval = (workspace / "out" / "ranges.json").read_text()
    assert json.loads(with_eval)["points"]["stem"]["max"] == 2.0

    (workspace / "out" / "ranges.json").unlink()
    (workspace / "clean" / "eval.parquet").write_bytes(b"not parquet")
    run_script(monkeypatch, workspace)
    assert (workspace / "out" / "ranges.json").read_text() == with_eval


def test_script_refuses_to_replace_existing_ranges(monkeypatch, workspace):
    run_script(monkeypatch, workspace)
    with pytest.raises(SystemExit, match="already exists"):
        run_script(monkeypatch, workspace)


def test_script_stops_on_different_checkpoint(monkeypatch, workspace):
    (workspace / "fake.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checkpoint differs"):
        run_script(monkeypatch, workspace)
    assert not (workspace / "out").exists()


def test_script_stops_when_runs_dont_match_splits(monkeypatch, workspace):
    write_runs(workspace / "clean" / "fit.parquet", [make_row(0, -1.0, 2.0), make_row(9, -3.0, 1.0)])
    with pytest.raises(ValueError, match="don't match"):
        run_script(monkeypatch, workspace)
    assert not (workspace / "out").exists()


def test_stats_are_finite_for_real_shaped_values(tmp_path):
    bounds = reference_ranges(load_rows(tmp_path, [make_row(0, 0.0, 16.24)]))["layer4"]
    assert all(math.isfinite(v) for v in bounds.values())
