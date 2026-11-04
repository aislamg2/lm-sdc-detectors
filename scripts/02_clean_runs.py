"""Run fault-free inference on both sets and record clean observations at the six monitored points.

Action item: Part I.1a Run fault-free inference

Refuses to replace existing files, and stops if the checkpoint or images differ from
the ones in the splits file. Nothing is written unless both sets finish.
"""

import argparse
import time
from pathlib import Path

from sdc.config import load_config
from sdc.data.eurosat import load_splits, load_test_split, verify_splits
from sdc.model.resnet18 import checkpoint_sha256, load_checkpoint
from sdc.observations.clean import run_clean
from sdc.records.io import write_table_atomic

SPLITS = ["fit", "eval"]


def main(config_path: str | None) -> None:
    config = load_config(config_path)
    paths, data = config["paths"], config["data"]

    out_dir = Path(paths["clean_runs"])
    existing = [out_dir / f"{s}.parquet" for s in SPLITS if (out_dir / f"{s}.parquet").exists()]
    if existing:
        raise SystemExit(f"{existing[0]} already exists; delete it to redo the clean runs")

    splits = load_splits(paths["splits"])
    use_indices = splits["split_source"] == "test_indices"
    test_ds = load_test_split(
        paths["eurosat_dir"], data["split_fractions"], data["split_seed"],
        indices_path=paths["test_indices"] if use_indices else None,
    )
    verify_splits(splits, test_ds, checkpoint_sha256(paths["checkpoint"]))
    model = load_checkpoint(paths["checkpoint"], config["device"])

    tables = {}
    for split in SPLITS:
        start = time.perf_counter()
        tables[split] = run_clean(model, test_ds, splits[split], config["device"])
        elapsed = time.perf_counter() - start
        print(f"{split}: {tables[split].num_rows} images in {elapsed:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, table in tables.items():
        write_table_atomic(table, out_dir / f"{split}.parquet")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    main(parser.parse_args().config)
