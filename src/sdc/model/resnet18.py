"""The monitored ResNet-18 classifier (architecture ported from Reem's notebook)."""

import hashlib

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

    The file must be a plain state_dict: `torch.save(model.state_dict(), path)`.
    The notebook never saves one, so this has to be exported from Reem's Colab run.
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model = build_resnet18()
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def checkpoint_sha256(path: str) -> str:
    """Hash of the checkpoint file, recorded so results can be tied to exact weights."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
