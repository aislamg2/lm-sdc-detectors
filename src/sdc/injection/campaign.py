"""Resumable fault-injection campaign.

For each image in a split and each planned fault: inject, run with the
ActivationMonitor attached, collect observations, label, and buffer a
RunRecord. Every `shard_size` records are flushed to a Parquet shard; on
restart, shards already on disk are skipped.
"""

from pathlib import Path


def plan_faults(image_ids: list[int], injections_per_image: int, sites, bit_groups,
                seed: int) -> list[dict]:
    """Deterministically enumerate every (image, site, bit_group, fault) to run.

    Determinism matters: resuming must regenerate the same plan so shard
    indices line up. TODO: implement.
    """
    raise NotImplementedError


def run_campaign(config: dict, split: str, out_dir: str | Path) -> None:
    """Execute the plan for `split` ("fit" or "eval"), skipping completed shards.

    TODO: implement.
    """
    raise NotImplementedError
