"""The run-record format: what gets saved for every faulty run (proposal §4 "Records").

FROZEN at SCHEMA_VERSION 1. Changing a field after the injection campaign starts
means re-running it, so bump SCHEMA_VERSION and tell the team first.

Kept out of this record on purpose:
  * clean runs: one row per image, in their own table, joined on image_id
  * detector scores: their own table, joined on run_id, since detectors get re-run
  * raw activations and LM hidden states: far too large
"""

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = 1


class Label(str, Enum):
    MASKED = "masked"   # finite outputs, prediction unchanged
    SDC = "sdc"         # finite outputs, prediction changed
    CRASH = "crash"     # error, timeout, or non-finite output


class CrashReason(str, Enum):
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    NONFINITE = "nonfinite"


class FaultSite(str, Enum):
    WEIGHT = "weight"           # persistent for the whole inference
    ACTIVATION = "activation"   # transient, single intermediate value


class BitGroup(str, Enum):
    # For INT8 runs this still names the fault's pre-quantization group (proposal §3).
    SIGN = "sign"
    EXPONENT = "exponent"
    MANTISSA = "mantissa"


@dataclass
class LayerStats:
    """White-box statistics for one monitored point."""
    min: float
    max: float
    mean: float
    std: float
    nonfinite: int


@dataclass
class RunRecord:
    # Identity
    run_id: str
    plan_idx: int               # position in the deterministic fault plan
    seed: int                   # campaign seed that generated the plan
    split: str                  # "fit" or "eval"
    image_id: int               # index into the held-out test split
    image_path: str             # relative to the EuroSAT root
    true_class: int

    # Fault
    fault_site: FaultSite
    layer: str                  # module name, e.g. "layer3.1.conv2"
    element: list[int]          # tensor index of the flipped value
    bit_position: int           # 0..31 (FP32)
    bit_group: BitGroup
    original_value: float | None   # None if the run ended before the flip
    flipped_value: float | None

    # Outcome
    clean_pred: int
    faulty_pred: int | None     # None if crashed
    label: Label
    crash_reason: CrashReason | None    # None unless label is CRASH
    faulty_logits: list[float] | None   # all 10; None if no logits were produced

    # Observation o_i, black-box level
    top1_prob: float | None
    entropy: float | None

    # Reproducibility
    checkpoint_sha256: str
    git_commit: str
    device: str                 # e.g. "cpu"
    timestamp: str              # ISO 8601, UTC
    inference_time_s: float | None

    # Observation o_i, white-box level (keyed by monitored point)
    layer_stats: dict[str, LayerStats] = field(default_factory=dict)

    # Bookkeeping
    model_format: str = "pytorch_fp32"   # or "tflite_fp32", "tflite_int8"
    schema_version: int = SCHEMA_VERSION
