"""Forward hooks on the six monitored points of ResNet-18.

stem   = output of conv1/bn1/relu/maxpool
layer1..layer4 = output of each residual stage
fc     = output logits
"""

import torch

MONITORED_POINTS = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]

# Module whose output is recorded for each monitored point.
POINT_TO_MODULE = {
    "stem": "maxpool",
    "layer1": "layer1",
    "layer2": "layer2",
    "layer3": "layer3",
    "layer4": "layer4",
    "fc": "fc",
}


class ActivationMonitor:
    """Context manager that captures monitored activations during one forward pass.

    Usage:
        with ActivationMonitor(model) as mon:
            logits = model(x)
        mon.activations  # {"stem": tensor, ..., "fc": tensor}
    """

    def __init__(self, model: torch.nn.Module, points=MONITORED_POINTS):
        unknown = [p for p in points if p not in POINT_TO_MODULE]
        if unknown:
            raise ValueError(f"unknown monitored points: {unknown}")
        self.model = model
        self.points = list(points)
        self.activations: dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        self.activations = {}
        # Look up every module first so a bad model leaves no hooks behind.
        modules = {p: self.model.get_submodule(POINT_TO_MODULE[p]) for p in self.points}
        for point, module in modules.items():
            self._handles.append(module.register_forward_hook(self._capture(point)))
        return self

    def __exit__(self, *exc):
        # Runs after a crash too: a leftover hook would contaminate later runs.
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _capture(self, point: str):
        def hook(module, inputs, output):
            self.activations[point] = output.detach()
        return hook
