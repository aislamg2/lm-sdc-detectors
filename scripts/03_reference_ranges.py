"""Compute per-layer normal ranges from clean fitting-set runs.

Action item: Compute normal ranges from the fitting set

Reads only the fit file: using evaluation data here would leak it into the detectors.
Refuses to replace an existing ranges file, and stops if the checkpoint or images
differ from the splits file.
"""

import argparse
import json
import os
from pathlib import Path

from sdc.config import load_config
from sdc.data.eurosat import load_splits
from sdc.model.resnet18 import checkpoint_sha256
from sdc.observations.clean import load_clean_runs
from sdc.observations.stats import QUANTILE_HIGH, QUANTILE_LOW, reference_ranges


def main(config_path: str | None) -> None:
    paths = load_config(config_path)["paths"]

    out_path = Path(paths["reference_ranges"])
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; delete it to recompute the ranges")

    splits = load_splits(paths["splits"])
    checkpoint_hash = checkpoint_sha256(paths["checkpoint"])
    if splits["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint differs from the one used to build the splits; rebuild them")

    runs = load_clean_runs(str(Path(paths["clean_runs"]) / "fit.parquet"))
    expected_ids = [entry["image_id"] for entry in splits["fit"]]
    if runs["image_id"].tolist() != expected_ids:
        raise ValueError("clean fit runs don't match the fit images in the splits file")

    result = {
        "checkpoint_sha256": checkpoint_hash,
        "num_fit_images": len(runs),
        "quantile_low": QUANTILE_LOW,
        "quantile_high": QUANTILE_HIGH,
        "points": reference_ranges(runs),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f"{out_path.name}.tmp")
    temp_path.write_text(json.dumps(result, indent=2))
    os.replace(temp_path, out_path)

    for point, bounds in result["points"].items():
        print(f"{point}: [{bounds['min']:.3f}, {bounds['max']:.3f}]")
    print(f"from {len(runs)} fit images -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    main(parser.parse_args().config)
