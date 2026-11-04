"""Parquet shard I/O for run records.

Runs are written in small shards (shard_000123.parquet) so the injection loop
can stop and resume between Pi sessions: a shard either exists completely or
not at all.
"""

import os
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sdc.model.monitor import MONITORED_POINTS
from sdc.records.schema import RunRecord

SHARD_PATTERN = re.compile(r"shard_(\d+)\.parquet")

STAT_COLUMNS = {
    "min": pa.float64(), "max": pa.float64(), "mean": pa.float64(),
    "std": pa.float64(), "nonfinite": pa.int64(),
}

# One fixed layout for every shard, so a shard with no crashes still has typed
# crash_reason and faulty_pred columns and shards can always be combined.
ARROW_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("plan_idx", pa.int64()),
        ("seed", pa.int64()),
        ("split", pa.string()),
        ("image_id", pa.int64()),
        ("image_path", pa.string()),
        ("true_class", pa.int64()),
        ("fault_site", pa.string()),
        ("layer", pa.string()),
        ("element", pa.list_(pa.int64())),
        ("bit_position", pa.int64()),
        ("bit_group", pa.string()),
        ("original_value", pa.float64()),
        ("flipped_value", pa.float64()),
        ("clean_pred", pa.int64()),
        ("faulty_pred", pa.int64()),
        ("label", pa.string()),
        ("crash_reason", pa.string()),
        ("faulty_logits", pa.list_(pa.float64())),
        ("top1_prob", pa.float64()),
        ("entropy", pa.float64()),
        ("checkpoint_sha256", pa.string()),
        ("git_commit", pa.string()),
        ("device", pa.string()),
        ("timestamp", pa.string()),
        ("inference_time_s", pa.float64()),
    ]
    + [
        (f"{point}_{stat}", arrow_type)
        for point in MONITORED_POINTS
        for stat, arrow_type in STAT_COLUMNS.items()
    ]
    + [
        ("model_format", pa.string()),
        ("schema_version", pa.int64()),
    ]
)


def record_to_row(record: RunRecord) -> dict:
    """Flatten a RunRecord into one Parquet row (layer_stats -> stem_min, stem_max, ...).

    Points a crashed run never reached become empty (None), not NaN: NaN means a
    value was computed and was bad.
    """
    row = asdict(record)
    layer_stats = row.pop("layer_stats")
    unknown = set(layer_stats) - set(MONITORED_POINTS)
    if unknown:
        raise ValueError(f"unknown monitored points: {sorted(unknown)}")
    for point in MONITORED_POINTS:
        stats = layer_stats.get(point, {})
        for stat in STAT_COLUMNS:
            row[f"{point}_{stat}"] = stats.get(stat)
    for key in ("fault_site", "bit_group", "label", "crash_reason"):
        if row[key] is not None:
            row[key] = row[key].value
    return row


def write_shard(records: list[RunRecord], out_dir: str | Path, shard_idx: int) -> Path:
    """Atomically write one shard (write to a temp file, then rename)."""
    if not records:
        raise ValueError("cannot write an empty shard")
    out_dir = Path(out_dir)
    path = out_dir / f"shard_{shard_idx:06d}.parquet"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist([record_to_row(r) for r in records], schema=ARROW_SCHEMA)
    write_table_atomic(table, path)
    return path


def write_table_atomic(table: pa.Table, path: Path) -> None:
    """Write a Parquet file so it either exists completely or not at all."""
    temp_path = path.with_name(f"{path.name}.tmp")
    with open(temp_path, "wb") as f:
        pq.write_table(table, f)
        f.flush()
        os.fsync(f.fileno())   # the data must reach the SD card before the rename
    os.replace(temp_path, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)   # and so must the rename itself
    finally:
        os.close(dir_fd)


def completed_shards(out_dir: str | Path) -> set[int]:
    """Return the indices of shards already on disk, for resuming."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return set()
    matches = (SHARD_PATTERN.fullmatch(p.name) for p in out_dir.iterdir())
    return {int(m.group(1)) for m in matches if m}


def load_runs(out_dir: str | Path) -> pd.DataFrame:
    """Concatenate all shards into one DataFrame, in shard order.

    Arrow-backed columns keep empty (<NA>) and NaN distinct; plain pandas floats
    would turn both into NaN. A shard with a different layout raises an error.
    """
    tables = [
        pq.read_table(Path(out_dir) / f"shard_{idx:06d}.parquet")
        for idx in sorted(completed_shards(out_dir))
    ]
    table = pa.concat_tables(tables) if tables else ARROW_SCHEMA.empty_table()
    return table.to_pandas(types_mapper=pd.ArrowDtype)
