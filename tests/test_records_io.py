import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sdc.model.monitor import MONITORED_POINTS
from sdc.records import io
from sdc.records.io import ARROW_SCHEMA, completed_shards, load_runs, record_to_row, write_shard
from sdc.records.schema import (
    SCHEMA_VERSION,
    BitGroup,
    CrashReason,
    FaultSite,
    Label,
    LayerStats,
    RunRecord,
)


def make_record(plan_idx: int = 0, **overrides) -> RunRecord:
    values = dict(
        run_id=f"r{plan_idx}", plan_idx=plan_idx, seed=42, split="fit", image_id=7,
        image_path="Forest/Forest_1.jpg", true_class=1,
        fault_site=FaultSite.WEIGHT, layer="layer3.1.conv2", element=[0, 1, 2, 3],
        bit_position=30, bit_group=BitGroup.EXPONENT,
        original_value=0.5, flipped_value=1.5,
        clean_pred=1, faulty_pred=4, label=Label.SDC, crash_reason=None,
        faulty_logits=[float(i) for i in range(10)], top1_prob=0.9, entropy=0.3,
        checkpoint_sha256="abc", git_commit="def", device="cpu",
        timestamp="2026-11-03T00:00:00+00:00", inference_time_s=0.2,
        layer_stats={p: LayerStats(0.0, 1.0, 0.5, 0.25, 0) for p in MONITORED_POINTS},
    )
    return RunRecord(**{**values, **overrides})


def crashed_record(plan_idx: int = 1) -> RunRecord:
    return make_record(
        plan_idx, faulty_pred=None, label=Label.CRASH, crash_reason=CrashReason.EXCEPTION,
        faulty_logits=None, top1_prob=None, entropy=None, inference_time_s=None,
        original_value=None, flipped_value=None,
        layer_stats={"stem": LayerStats(0.0, 1.0, 0.5, 0.25, 0)},
    )


def test_row_has_exactly_the_schema_columns():
    assert set(record_to_row(make_record())) == set(ARROW_SCHEMA.names)


def test_row_flattens_stats_and_converts_enums_to_strings():
    row = record_to_row(make_record())
    assert row["stem_min"] == 0.0 and row["fc_nonfinite"] == 0
    assert "layer_stats" not in row
    assert (row["fault_site"], row["bit_group"], row["label"]) == ("weight", "exponent", "sdc")
    assert row["crash_reason"] is None
    assert row["schema_version"] == SCHEMA_VERSION


def test_row_for_unreached_point_is_empty_not_nan():
    row = record_to_row(crashed_record())
    assert row["stem_max"] == 1.0
    assert row["layer4_max"] is None
    assert row["crash_reason"] == "exception"


def test_unknown_point_is_rejected():
    with pytest.raises(ValueError, match="layer9"):
        record_to_row(make_record(layer_stats={"layer9": LayerStats(0, 1, 0, 0, 0)}))


def test_roundtrip_preserves_values(tmp_path):
    write_shard([make_record()], tmp_path, 0)
    run = load_runs(tmp_path).iloc[0]
    assert run["run_id"] == "r0"
    assert run["label"] == "sdc"
    assert run["layer3_std"] == 0.25
    assert list(run["element"]) == [0, 1, 2, 3]
    assert list(run["faulty_logits"]) == [float(i) for i in range(10)]


def test_shard_name_and_no_temp_file_left(tmp_path):
    path = write_shard([make_record()], tmp_path, 123)
    assert path.name == "shard_000123.parquet"
    assert [p.name for p in tmp_path.iterdir()] == ["shard_000123.parquet"]


def test_creates_missing_directory(tmp_path):
    write_shard([make_record()], tmp_path / "a" / "b", 0)
    assert completed_shards(tmp_path / "a" / "b") == {0}


def test_refuses_to_overwrite_and_keeps_original(tmp_path):
    write_shard([make_record(0)], tmp_path, 0)
    with pytest.raises(FileExistsError):
        write_shard([make_record(9)], tmp_path, 0)
    assert load_runs(tmp_path)["plan_idx"].tolist() == [0]


def test_empty_shard_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_shard([], tmp_path, 0)
    assert completed_shards(tmp_path) == set()


def test_failed_write_does_not_count_as_complete(tmp_path, monkeypatch):
    def fail(table, f):
        f.write(b"half a shard")
        raise OSError("power cut")

    monkeypatch.setattr(io.pq, "write_table", fail)
    with pytest.raises(OSError):
        write_shard([make_record()], tmp_path, 0)
    assert completed_shards(tmp_path) == set()
    assert len(load_runs(tmp_path)) == 0

    # The leftover temp file doesn't block the retry.
    monkeypatch.undo()
    write_shard([make_record()], tmp_path, 0)
    assert completed_shards(tmp_path) == {0}
    assert len(load_runs(tmp_path)) == 1


def test_completed_shards_ignores_other_files(tmp_path):
    write_shard([make_record()], tmp_path, 4)
    (tmp_path / "shard_000005.parquet.tmp").write_bytes(b"partial")
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "shard_x.parquet").write_text("hi")
    assert completed_shards(tmp_path) == {4}


def test_completed_shards_on_missing_directory(tmp_path):
    assert completed_shards(tmp_path / "nope") == set()


def test_load_returns_runs_in_shard_order(tmp_path):
    write_shard([make_record(2)], tmp_path, 10)
    write_shard([make_record(0), make_record(1)], tmp_path, 2)
    assert load_runs(tmp_path)["plan_idx"].tolist() == [0, 1, 2]


def test_shards_with_and_without_crashes_combine(tmp_path):
    write_shard([make_record(0)], tmp_path, 0)
    write_shard([crashed_record(1)], tmp_path, 1)
    runs = load_runs(tmp_path)
    assert runs["label"].tolist() == ["sdc", "crash"]
    assert runs["crash_reason"].isna().tolist() == [True, False]
    assert runs["faulty_pred"].isna().tolist() == [False, True]


def test_every_shard_has_the_same_layout_even_with_all_empty_columns(tmp_path):
    no_crash = write_shard([make_record(0)], tmp_path, 0)
    crash = write_shard([crashed_record(1)], tmp_path, 1)
    assert pq.read_schema(no_crash).equals(ARROW_SCHEMA)
    assert pq.read_schema(crash).equals(ARROW_SCHEMA)


def test_nan_and_empty_stay_different_after_loading(tmp_path):
    nan_stats = {"stem": LayerStats(math.nan, math.nan, math.nan, math.nan, 5)}
    write_shard([make_record(top1_prob=math.nan, layer_stats=nan_stats)], tmp_path, 0)
    run = load_runs(tmp_path)
    assert not run["stem_min"].isna().iloc[0]        # computed, and bad
    assert math.isnan(run["stem_min"].iloc[0])
    assert run["layer1_min"].isna().iloc[0]          # never computed
    assert run["stem_nonfinite"].iloc[0] == 5
    assert math.isnan(run["top1_prob"].iloc[0])


def test_infinite_flipped_value_survives(tmp_path):
    write_shard([make_record(flipped_value=math.inf)], tmp_path, 0)
    assert load_runs(tmp_path)["flipped_value"].iloc[0] == math.inf


def test_numpy_float32_from_bitflip_is_accepted(tmp_path):
    record = make_record(original_value=np.float32(1.0), flipped_value=np.float32(1.5))
    write_shard([record], tmp_path, 0)
    assert load_runs(tmp_path)["flipped_value"].iloc[0] == 1.5


def test_load_from_empty_or_missing_directory_has_all_columns(tmp_path):
    for directory in (tmp_path, tmp_path / "nope"):
        runs = load_runs(directory)
        assert len(runs) == 0
        assert list(runs.columns) == ARROW_SCHEMA.names


def test_shard_with_different_layout_fails_loudly(tmp_path):
    write_shard([make_record()], tmp_path, 0)
    pq.write_table(pa.table({"run_id": ["old"]}), tmp_path / "shard_000001.parquet")
    with pytest.raises(pa.ArrowInvalid):
        load_runs(tmp_path)


def test_label_filter_and_numeric_columns_work_for_analysis(tmp_path):
    write_shard([make_record(0), make_record(1, label=Label.MASKED, entropy=0.1)], tmp_path, 0)
    runs = load_runs(tmp_path)
    assert runs[runs["label"] == "sdc"]["run_id"].tolist() == ["r0"]
    assert runs["entropy"].to_numpy(dtype=float).tolist() == [0.3, 0.1]
