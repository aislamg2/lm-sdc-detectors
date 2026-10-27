"""Weight and activation fault injection via PyTorchFI.

Open questions from the planning notes: PyTorchFI x LiteRT (can a mutated
model be converted and still carry the fault?) and running FI on the Pi.
Activation faults happen during inference and cannot be carried into a
converted model, so they always run on the PyTorch model.
"""

from dataclasses import dataclass

import numpy as np
import torch

from sdc.records.schema import BitGroup, FaultSite


@dataclass
class FaultSpec:
    site: FaultSite
    layer: str
    element: tuple[int, ...]
    bit_position: int
    bit_group: BitGroup


def sample_fault(model: torch.nn.Module, site: FaultSite, group: BitGroup,
                 rng: np.random.Generator) -> FaultSpec:
    """Sample a layer and element uniformly at random, and a bit within `group`.

    TODO: implement. Decide whether "uniform across layer/element" means uniform
    over layers then elements, or uniform over all elements (weights by size).
    """
    raise NotImplementedError


def inject_weight_fault(model: torch.nn.Module, fault: FaultSpec) -> torch.nn.Module:
    """Return a copy of `model` with one weight bit flipped (persistent fault).

    TODO: implement with pytorchfi.core.fault_injection / declare_weight_fault_injection.
    """
    raise NotImplementedError


def inject_activation_fault(model: torch.nn.Module, fault: FaultSpec,
                            input_shape=(1, 3, 224, 224)) -> torch.nn.Module:
    """Return a model whose forward pass flips one activation bit (transient fault).

    TODO: implement with pytorchfi neuron injection and a custom bit-flip function.
    """
    raise NotImplementedError
