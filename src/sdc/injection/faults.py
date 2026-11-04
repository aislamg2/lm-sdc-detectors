"""Weight and activation fault injection.

Faults target the Conv2d and Linear layers' weights and outputs (21 layers in
ResNet-18). Batch norm parameters and the Linear bias are not targeted.

Weight faults are flipped in place and restored, which avoids PyTorchFI's copy of the
whole model per fault. Activation faults use PyTorchFI neuron injection with a custom
bit flip. Activation faults happen during inference and cannot be carried into a
converted model, so they always run on the PyTorch model.

Open question from the planning notes: PyTorchFI x LiteRT (can a mutated model be
converted and still carry the fault?).
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from pytorchfi.core import fault_injection

from sdc.injection.bitflip import bit_group_of, flip_bit_fp32, sample_bit
from sdc.records.schema import BitGroup, FaultSite

INJECTABLE_TYPES = (torch.nn.Conv2d, torch.nn.Linear)


@dataclass
class FaultSpec:
    site: FaultSite
    layer: str                  # module name, e.g. "layer3.1.conv2"
    element: tuple[int, ...]    # weight index, or activation index without the batch dim
    bit_position: int
    bit_group: BitGroup


@dataclass
class FaultValues:
    original: float
    flipped: float


def injectable_layers(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """The layers faults can target, by name, in a fixed order."""
    return {name: m for name, m in model.named_modules() if isinstance(m, INJECTABLE_TYPES)}


def weight_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(m.weight.shape) for name, m in injectable_layers(model).items()}


def activation_shapes(model: torch.nn.Module, input_shape=(3, 224, 224)) -> dict[str, tuple[int, ...]]:
    """Each layer's output shape without the batch dim, from one dummy forward pass."""
    shapes, handles = {}, []
    for name, module in injectable_layers(model).items():
        def record(module, inputs, output, name=name):
            shapes[name] = tuple(output.shape[1:])
        handles.append(module.register_forward_hook(record))
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape))
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def sample_fault(shapes: dict[str, tuple[int, ...]], site: FaultSite, group: BitGroup,
                 rng: np.random.Generator) -> FaultSpec:
    """Sample a layer uniformly, an element uniformly within it, and a bit within `group`.

    Every layer gets the same share of faults whatever its size, so per-layer results
    are well populated. An overall SDC rate is therefore not what uniform hits on the
    hardware would give, since a real hit is more likely to land in a large layer.
    `shapes` comes from weight_shapes or activation_shapes, matching `site`.
    """
    layer = str(rng.choice(list(shapes)))
    element = tuple(int(rng.integers(size)) for size in shapes[layer])
    return FaultSpec(site, layer, element, sample_bit(group, rng), group)


def _flip_in_place(tensor: torch.Tensor, index: tuple[int, ...], bit_position: int) -> FaultValues:
    if tensor.dtype != torch.float32:
        raise ValueError(f"expected a float32 tensor, got {tensor.dtype}")
    data = tensor.data
    original = data[index].cpu().numpy().copy()   # .numpy() alone shares memory with data
    flipped = flip_bit_fp32(original, bit_position)
    data[index] = torch.from_numpy(np.array(flipped, dtype=np.float32)).to(data.device)
    return FaultValues(float(original), float(flipped))


def _check_fault(fault: FaultSpec, site: FaultSite) -> None:
    if fault.site != site:
        raise ValueError(f"expected a {site.value} fault, got {fault.site.value}")
    if bit_group_of(fault.bit_position) != fault.bit_group:
        raise ValueError(f"bit {fault.bit_position} is not in group {fault.bit_group.value}")


@contextmanager
def weight_fault(model: torch.nn.Module, fault: FaultSpec):
    """Flip one weight bit for the duration of the `with` block, yielding FaultValues.

    The original bits are restored afterwards, even after a crash: a fault left in
    place would silently corrupt every later run.
    """
    _check_fault(fault, FaultSite.WEIGHT)
    weight = model.get_submodule(fault.layer).weight
    saved = weight.data[fault.element].clone()
    try:
        yield _flip_in_place(weight, fault.element, fault.bit_position)
    finally:
        weight.data[fault.element] = saved


class ActivationInjector:
    """Prepares PyTorchFI once per model, then gives a corrupted copy per fault.

    `model` must be in eval mode: PyTorchFI runs it once to learn the layer shapes.
    """

    def __init__(self, model: torch.nn.Module, input_shape=(3, 224, 224)):
        self._pfi = fault_injection(
            model, 1, input_shape=list(input_shape),
            layer_types=list(INJECTABLE_TYPES), use_cuda=False,
        )
        self._layer_numbers = {name: i for i, name in enumerate(injectable_layers(model))}

    def inject(self, fault: FaultSpec) -> tuple[torch.nn.Module, list[FaultValues]]:
        """Return a model that flips one activation bit each time it runs, and a list
        that receives the FaultValues when the flip happens.

        PyTorchFI hooks every injectable layer with the custom function, so the hook
        checks it is on the target. The fault hook is registered before any monitor is
        attached to the returned model, so a monitor on the same layer sees the fault.
        """
        _check_fault(fault, FaultSite.ACTIVATION)
        values: list[FaultValues] = []
        target = None

        def flip(module, inputs, output):
            if module is target:
                values.append(_flip_in_place(output[0], fault.element, fault.bit_position))

        dims = list(fault.element) + [None] * (3 - len(fault.element))
        corrupted = self._pfi.declare_neuron_fi(
            function=flip,
            batch=[0],
            layer_num=[self._layer_numbers[fault.layer]],
            dim1=[dims[0]], dim2=[dims[1]], dim3=[dims[2]],
        )
        target = corrupted.get_submodule(fault.layer)
        return corrupted, values
