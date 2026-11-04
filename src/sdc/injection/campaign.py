"""Resumable fault-injection campaign.

For each image in a split and each planned fault: inject, run with the
ActivationMonitor attached, collect observations, label, and buffer a
RunRecord. Every `shard_size` records are flushed to a Parquet shard; on
restart, shards already on disk are skipped.

The plan is deterministic and shard k always covers the same runs, so stopping at any
moment loses at most the shard in progress. A saved campaign.json stops a resume that
would mix two different experiments.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from sdc.config import REPO_ROOT
from sdc.data.eurosat import load_splits, load_test_split, verify_splits
from sdc.injection.faults import (
    ActivationInjector,
    FaultSpec,
    activation_shapes,
    sample_fault,
    weight_fault,
    weight_shapes,
)
from sdc.injection.labeling import label_run
from sdc.model import resnet18
from sdc.model.monitor import ActivationMonitor
from sdc.observations.clean import load_clean_runs
from sdc.observations.stats import blackbox_stats, whitebox_stats
from sdc.records.io import completed_shards, write_shard
from sdc.records.schema import SCHEMA_VERSION, BitGroup, CrashReason, FaultSite, Label, RunRecord

SPLIT_NUMBERS = {"fit": 0, "eval": 1}


@dataclass
class PlannedRun:
    plan_idx: int
    image_id: int
    fault: FaultSpec


@dataclass
class RunContext:
    """What is the same for every run in a campaign."""
    seed: int
    split: str
    checkpoint_sha256: str
    git_commit: str
    device: str
    timeout_s: float


def plan_faults(image_ids: list[int], injections_per_image: int, sites: list[FaultSite],
                bit_groups: list[BitGroup], shapes: dict[FaultSite, dict], seed: int,
                split: str) -> list[PlannedRun]:
    """Deterministically enumerate every (image, site, bit_group, fault) to run.

    Determinism matters: resuming must regenerate the same plan so shard
    indices line up. Each fault gets its own generator, so it depends only on its own
    position in the plan.
    """
    plan = []
    for image_id in image_ids:
        for injection in range(injections_per_image):
            for site_idx, site in enumerate(sites):
                for group_idx, group in enumerate(bit_groups):
                    rng = np.random.default_rng(
                        [seed, SPLIT_NUMBERS[split], image_id, injection, site_idx, group_idx]
                    )
                    fault = sample_fault(shapes[site], site, group, rng)
                    plan.append(PlannedRun(len(plan), image_id, fault))
    return plan


def plan_hash(plan: list[PlannedRun]) -> str:
    digest = hashlib.sha256()
    for run in plan:
        fault = run.fault
        digest.update(repr((
            run.plan_idx, run.image_id, fault.site.value, fault.layer, fault.element,
            fault.bit_position, fault.bit_group.value,
        )).encode())
    return digest.hexdigest()


def weights_fingerprint(model: torch.nn.Module) -> str:
    """Hash of every weight's exact bits, to catch a fault that was not restored."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def current_git_commit() -> str:
    """The code version for run records, marked "-dirty" if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if changes else commit


@contextmanager
def time_limit(seconds: float):
    """Raise TimeoutError in the main thread after `seconds`.

    Python only acts on the alarm between operations, so this catches slow runs but
    can't interrupt a hang inside a single compiled operation.
    """
    def on_alarm(signum, frame):
        raise TimeoutError(f"run exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_one(run: PlannedRun, image: torch.Tensor, entry: dict, clean_pred: int,
            model: torch.nn.Module, injector: ActivationInjector, context: RunContext) -> RunRecord:
    """Do one faulty run and return its record.

    Exceptions and timeouts during inference become crash records. Errors while setting
    up the fault are bugs, not faults, so they stop the campaign.
    """
    fault = run.fault
    batch = image.unsqueeze(0).to(context.device)
    outcome = _Outcome()

    if fault.site == FaultSite.WEIGHT:
        with weight_fault(model, fault) as values:
            _infer(model, batch, context.timeout_s, outcome)
    else:
        corrupted, activation_values = injector.inject(fault)
        _infer(corrupted, batch, context.timeout_s, outcome)
        values = activation_values[0] if activation_values else None

    logits = outcome.logits
    label = label_run(clean_pred, logits, crashed=outcome.crash_reason is not None)
    crash_reason = outcome.crash_reason
    if label == Label.CRASH and crash_reason is None:
        crash_reason = CrashReason.NONFINITE
    black_box = blackbox_stats(logits) if logits is not None else {"top1_prob": None, "entropy": None}

    return RunRecord(
        run_id=f"{context.split}-{run.plan_idx:08d}",
        plan_idx=run.plan_idx,
        seed=context.seed,
        split=context.split,
        image_id=run.image_id,
        image_path=entry["image_path"],
        true_class=entry["true_class"],
        fault_site=fault.site,
        layer=fault.layer,
        element=list(fault.element),
        bit_position=fault.bit_position,
        bit_group=fault.bit_group,
        original_value=values.original if values else None,
        flipped_value=values.flipped if values else None,
        clean_pred=clean_pred,
        faulty_pred=None if label == Label.CRASH else int(logits.argmax()),
        label=label,
        crash_reason=crash_reason,
        faulty_logits=logits.flatten().tolist() if logits is not None else None,
        top1_prob=black_box["top1_prob"],
        entropy=black_box["entropy"],
        checkpoint_sha256=context.checkpoint_sha256,
        git_commit=context.git_commit,
        device=context.device,
        timestamp=datetime.now(timezone.utc).isoformat(),
        inference_time_s=outcome.elapsed,
        layer_stats=whitebox_stats(outcome.activations),
    )


@dataclass
class _Outcome:
    logits: torch.Tensor | None = None
    activations: dict | None = None
    crash_reason: CrashReason | None = None
    elapsed: float | None = None


def _infer(model: torch.nn.Module, batch: torch.Tensor, timeout_s: float, outcome: _Outcome) -> None:
    # The time limit sits inside weight_fault, so the alarm can't interrupt the restore.
    monitor = ActivationMonitor(model)
    start = time.perf_counter()
    try:
        with torch.no_grad(), monitor, time_limit(timeout_s):
            outcome.logits = model(batch)
    except TimeoutError:
        outcome.crash_reason = CrashReason.TIMEOUT
    except Exception as error:
        print(f"run raised {error!r}", file=sys.stderr)
        outcome.crash_reason = CrashReason.EXCEPTION
    outcome.elapsed = time.perf_counter() - start
    outcome.activations = monitor.activations


def run_campaign(config: dict, split: str, out_dir: str | Path,
                 max_shards: int | None = None) -> dict[str, int]:
    """Execute the plan for `split` ("fit" or "eval"), skipping completed shards.

    Stops after `max_shards` new shards if given. Returns counts of the plan's shards.
    """
    paths, data, injection = config["paths"], config["data"], config["injection"]
    if injection["injections_per_image"] is None:
        raise ValueError("set injection.injections_per_image first (the pilot decides it)")
    out_dir = Path(out_dir)

    splits = load_splits(paths["splits"])
    entries = {entry["image_id"]: entry for entry in splits[split]}
    use_indices = splits["split_source"] == "test_indices"
    test_ds = load_test_split(
        paths["eurosat_dir"], data["split_fractions"], data["split_seed"],
        indices_path=paths["test_indices"] if use_indices else None,
    )
    checkpoint_hash = resnet18.checkpoint_sha256(paths["checkpoint"])
    verify_splits(splits, test_ds, checkpoint_hash)
    clean_preds = _load_clean_preds(Path(paths["clean_runs"]) / f"{split}.parquet", entries)

    device = config["device"]
    model = resnet18.load_checkpoint(paths["checkpoint"], device)
    _check_first_clean_prediction(model, test_ds, clean_preds, device)

    sites = [FaultSite(site) for site in injection["sites"]]
    bit_groups = [BitGroup(group) for group in injection["bit_groups"]]
    shapes = {FaultSite.WEIGHT: weight_shapes(model), FaultSite.ACTIVATION: activation_shapes(model)}
    plan = plan_faults(
        list(entries), injection["injections_per_image"], sites, bit_groups, shapes,
        config["seed"], split,
    )
    shard_size = injection["shard_size"]
    num_shards = (len(plan) + shard_size - 1) // shard_size

    _check_manifest(out_dir, {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "seed": config["seed"],
        "injections_per_image": injection["injections_per_image"],
        "sites": injection["sites"],
        "bit_groups": injection["bit_groups"],
        "shard_size": shard_size,
        "checkpoint_sha256": checkpoint_hash,
        "plan_hash": plan_hash(plan),
        "num_runs": len(plan),
        "num_shards": num_shards,
    })

    context = RunContext(
        config["seed"], split, checkpoint_hash, current_git_commit(), device, injection["timeout_s"],
    )
    injector = ActivationInjector(model)
    fingerprint = weights_fingerprint(model)
    done = completed_shards(out_dir)
    written = 0
    image_cache = (None, None)

    for shard_idx in range(num_shards):
        if shard_idx in done:
            continue
        if max_shards is not None and written >= max_shards:
            break
        records = []
        for run in plan[shard_idx * shard_size:(shard_idx + 1) * shard_size]:
            if image_cache[0] != run.image_id:
                image_cache = (run.image_id, test_ds[run.image_id][0])
            records.append(run_one(
                run, image_cache[1], entries[run.image_id], clean_preds[run.image_id],
                model, injector, context,
            ))
        if all(record.crash_reason == CrashReason.EXCEPTION for record in records):
            raise RuntimeError(f"every run in shard {shard_idx} raised an exception: this looks like a bug")
        if weights_fingerprint(model) != fingerprint:
            raise RuntimeError(f"weights changed during shard {shard_idx}: a fault was not restored")
        write_shard(records, out_dir, shard_idx)
        written += 1
        print(f"{split}: shard {shard_idx + 1}/{num_shards} written")

    return {
        "total_shards": num_shards,
        "written": written,
        "skipped": len(done),
        "remaining": num_shards - len(done) - written,
    }


def _load_clean_preds(path: Path, entries: dict[int, dict]) -> dict[int, int]:
    runs = load_clean_runs(str(path))
    if runs["image_id"].tolist() != list(entries):
        raise ValueError(f"{path} doesn't match the images in the splits file")
    for image_id, image_path in zip(runs["image_id"].tolist(), runs["image_path"].tolist()):
        if image_path != entries[image_id]["image_path"]:
            raise ValueError(f"{path} lists {image_path} for image_id {image_id}: expected "
                             f"{entries[image_id]['image_path']}")
    return dict(zip(runs["image_id"].tolist(), runs["clean_pred"].tolist()))


def _check_first_clean_prediction(model, test_ds, clean_preds: dict[int, int], device: str) -> None:
    """Confirm the model and preprocessing still reproduce the saved clean runs."""
    image_id = next(iter(clean_preds))
    with torch.no_grad():
        pred = int(model(test_ds[image_id][0].unsqueeze(0).to(device)).argmax())
    if pred != clean_preds[image_id]:
        raise ValueError(f"image_id {image_id}: model predicts {pred}, but the clean run saved "
                         f"{clean_preds[image_id]}")


def _check_manifest(out_dir: Path, manifest: dict) -> None:
    """Save the campaign's settings, or stop if they differ from the ones already saved."""
    path = out_dir / "campaign.json"
    if path.exists():
        saved = json.loads(path.read_text())
        changed = sorted(key for key in manifest if saved.get(key) != manifest[key])
        if changed:
            raise ValueError(f"{out_dir} holds a different campaign (changed: {', '.join(changed)}); "
                             "restore those settings or use a new output folder")
        return
    if completed_shards(out_dir):
        raise ValueError(f"{out_dir} has shards but no campaign.json; use a new output folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2))
    os.replace(temp_path, path)
