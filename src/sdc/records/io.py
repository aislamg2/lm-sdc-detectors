"""Parquet shard I/O for run records.

Runs are written in small shards (shard_000123.parquet) so the injection loop
can stop and resume between Pi sessions: a shard either exists completely or
not at all.
"""

from pathlib import Path

import pandas as pd

from sdc.records.schema import RunRecord


def record_to_row(record: RunRecord) -> dict:
    """Flatten a RunRecord into one Parquet row (layer_stats -> stem_min, stem_max, ...).

    TODO: implement.
    """
    raise NotImplementedError


def write_shard(records: list[RunRecord], out_dir: str | Path, shard_idx: int) -> Path:
    """Atomically write one shard (write to a temp file, then rename).

    TODO: implement.
    """
    raise NotImplementedError


def completed_shards(out_dir: str | Path) -> set[int]:
    """Return the indices of shards already on disk, for resuming.

    TODO: implement.
    """
    raise NotImplementedError


def load_runs(out_dir: str | Path) -> pd.DataFrame:
    """Concatenate all shards into one DataFrame.

    TODO: implement.
    """
    raise NotImplementedError
