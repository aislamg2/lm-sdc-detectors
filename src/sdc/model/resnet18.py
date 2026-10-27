"""The monitored ResNet-18 classifier (architecture ported from Reem's notebook)."""

import torch
from torchvision import models

NUM_CLASSES = 10


def build_resnet18(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """ResNet-18 with the final layer replaced for EuroSAT, as in the notebook."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: str, device: str = "cpu") -> torch.nn.Module:
    """Build the model, load the trained state_dict from `path`, and return it in eval mode.

    TODO: implement once the trained weights are exported from Reem's Colab run
    (the notebook never saves a .pt; add `torch.save(model.state_dict(), ...)`).
    """
    raise NotImplementedError
