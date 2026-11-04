"""Fault-free runs: one row per image, the baselines that faulty runs compare against."""

from dataclasses import asdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Subset

from sdc.model.monitor import MONITORED_POINTS, ActivationMonitor
from sdc.observations.stats import blackbox_stats, whitebox_stats
from sdc.records.io import STAT_COLUMNS

CLEAN_ARROW_SCHEMA = pa.schema(
    [
        ("image_id", pa.int64()),
        ("image_path", pa.string()),
        ("true_class", pa.int64()),
        ("clean_pred", pa.int64()),
        ("clean_logits", pa.list_(pa.float64())),
        ("top1_prob", pa.float64()),
        ("entropy", pa.float64()),
    ]
    + [
        (f"{point}_{stat}", arrow_type)
        for point in MONITORED_POINTS
        for stat, arrow_type in STAT_COLUMNS.items()
    ]
)


def clean_run_row(model, image: torch.Tensor, entry: dict, device: str = "cpu") -> dict:
    """Run one image with the monitor attached and return its row.

    One image at a time, like the faulty runs, so the numbers are directly comparable.
    Raises if the model gets the image wrong: the splits only hold images it gets right,
    so this means the weights or preprocessing changed.
    """
    with torch.no_grad(), ActivationMonitor(model) as monitor:
        logits = model(image.unsqueeze(0).to(device))
    clean_pred = int(logits.argmax())
    if clean_pred != entry["true_class"]:
        raise ValueError(
            f"image_id {entry['image_id']} ({entry['image_path']}): clean prediction "
            f"{clean_pred} but true class {entry['true_class']}"
        )
    row = {
        "image_id": entry["image_id"],
        "image_path": entry["image_path"],
        "true_class": entry["true_class"],
        "clean_pred": clean_pred,
        "clean_logits": logits.flatten().tolist(),
        **blackbox_stats(logits),
    }
    for point, stats in whitebox_stats(monitor.activations).items():
        for stat, value in asdict(stats).items():
            row[f"{point}_{stat}"] = value
    return row


def run_clean(model, test_ds: Subset, entries: list[dict], device: str = "cpu") -> pa.Table:
    """Run every listed image and return the table of rows."""
    rows = [clean_run_row(model, test_ds[e["image_id"]][0], e, device) for e in entries]
    return pa.Table.from_pylist(rows, schema=CLEAN_ARROW_SCHEMA)


def load_clean_runs(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas(types_mapper=pd.ArrowDtype)
