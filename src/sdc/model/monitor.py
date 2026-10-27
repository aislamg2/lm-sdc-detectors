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
        self.model = model
        self.points = list(points)
        self.activations: dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        # TODO: register forward hooks on POINT_TO_MODULE[p] that store detached outputs.
        raise NotImplementedError

    def __exit__(self, *exc):
        # TODO: remove all hook handles.
        raise NotImplementedError
