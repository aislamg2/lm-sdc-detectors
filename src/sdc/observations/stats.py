"""Build the observation vector o_i at both access levels (proposal §6.2).

black-box: top-1 softmax probability and entropy of the output distribution.
white-box: black-box + per monitored point (min, max, mean, std, nonfinite count).
"""

import numpy as np
import torch

from sdc.records.schema import LayerStats


def blackbox_stats(logits: torch.Tensor) -> dict[str, float]:
    """Return {"top1_prob": ..., "entropy": ...}. TODO: implement."""
    raise NotImplementedError


def layer_stats(activation: torch.Tensor) -> LayerStats:
    """Summary statistics over finite values plus the non-finite count. TODO: implement."""
    raise NotImplementedError


def whitebox_stats(activations: dict[str, torch.Tensor]) -> dict[str, LayerStats]:
    """Apply layer_stats to every monitored point. TODO: implement."""
    raise NotImplementedError


def feature_vector(record_row: dict, access_level: str) -> np.ndarray:
    """Flatten an observation into a fixed-order numeric vector.

    Shared by the learned detector and anything else that needs o_i as numbers,
    so every detector sees identical information. TODO: implement.
    """
    raise NotImplementedError


def reference_ranges(fit_clean_runs) -> dict[str, dict[str, float]]:
    """Per-layer (a_l, b_l) min/max over fault-free runs in the fitting set.

    Used by the range check and by the LM "reference" prompt condition.
    TODO: implement (Action item: "Compute normal ranges from the fitting set").
    """
    raise NotImplementedError
