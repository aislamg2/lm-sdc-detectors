"""Load the held-out test split, keep correctly classified images, balance classes, split fit/eval.

Action item: Part I.1 Clean runs; split images into fitting and evaluation sets

Refuses to replace an existing splits file: every later result depends on it, and
a new checkpoint means a new split. Delete the file yourself to rebuild it.
"""

import argparse
import json
import os
from pathlib import Path

from sdc.config import load_config
from sdc.data.eurosat import build_splits, load_test_split
from sdc.model.resnet18 import checkpoint_sha256, load_checkpoint


def main(config_path: str | None) -> None:
    config = load_config(config_path)
    paths, data = config["paths"], config["data"]

    out_path = Path(paths["splits"])
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; delete it to rebuild the splits")

    use_indices = Path(paths["test_indices"]).exists()
    test_ds = load_test_split(
        paths["eurosat_dir"], data["split_fractions"], data["split_seed"],
        indices_path=paths["test_indices"] if use_indices else None,
    )
    model = load_checkpoint(paths["checkpoint"], config["device"])
    splits = build_splits(
        model, test_ds, data["images_per_class"], data["fit_fraction"], config["seed"],
        device=config["device"],
    )
    splits["split_source"] = "test_indices" if use_indices else "reproduced_seed_split"
    splits["checkpoint_sha256"] = checkpoint_sha256(paths["checkpoint"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f"{out_path.name}.tmp")
    temp_path.write_text(json.dumps(splits, indent=2))
    os.replace(temp_path, out_path)

    print(f"clean accuracy: {splits['clean_accuracy']:.4f} "
          f"({splits['num_correct']}/{splits['num_test_images']}), source: {splits['split_source']}")
    print(f"{splits['per_class']} images per class: "
          f"{len(splits['fit'])} fit, {len(splits['eval'])} eval -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml")
    main(parser.parse_args().config)
