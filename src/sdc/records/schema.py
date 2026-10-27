"""The run-record format: what gets saved for every faulty run (proposal §4 "Records").

DRAFT: the team should review and freeze these fields before the campaign
starts (Action item: "Freeze the run-record format"). Bump SCHEMA_VERSION on
any change.
"""

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = 0


class Label(str, Enum):
    MASKED = "masked"   # finite outputs, prediction unchanged
    SDC = "sdc"         # finite outputs, prediction changed
    CRASH = "crash"     # error, timeout, or non-finite output


class FaultSite(str, Enum):
    WEIGHT = "weight"           # persistent for the whole inference
    ACTIVATION = "activation"   # transient, single intermediate value


class BitGroup(str, Enum):
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
    split: str                  # "fit" or "eval"
    image_id: int               # index into the held-out test split
    true_class: int

    # Fault
    fault_site: FaultSite
    layer: str                  # module name, e.g. "layer3.1.conv2"
    element: list[int]          # tensor index of the flipped value
    bit_position: int           # 0..31 (FP32)
    bit_group: BitGroup

    # Outcome
    clean_pred: int
    faulty_pred: int | None     # None if crashed
    label: Label

    # Observation o_i, black-box level
    top1_prob: float | None
    entropy: float | None

    # Observation o_i, white-box level (keyed by monitored point)
    layer_stats: dict[str, LayerStats] = field(default_factory=dict)

    # Bookkeeping
    model_format: str = "pytorch_fp32"   # or "tflite_fp32", "tflite_int8"
    schema_version: int = SCHEMA_VERSION
