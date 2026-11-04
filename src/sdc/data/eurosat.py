"""EuroSAT loading and the image splits used by every experiment.

Transforms and the train/val/test split are ported from Reem's notebook
(notebooks/reem_TFLite_EuroSAT_ResNet18.ipynb, from
github.com/reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset) so
that "held-out test split" refers to the same images the model never trained on.

Note: in the notebook all three subsets share one ImageFolder, so the last
`.transform` assignment (eval) applied to training too. The split *indices*
depend only on dataset length and seed, so we reproduce them exactly and apply
EVAL_TRANSFORM cleanly here.
"""

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture",
    "PermanentCrop", "Residential", "River", "SeaLake",
]

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Ported from the notebook (training only; kept for reference / retraining).
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_test_split(data_dir: str, fractions=(0.8, 0.1, 0.1), seed: int = 42) -> Subset:
    """Return the held-out test subset, identical to the notebook's `test_ds`."""
    full_dataset = datasets.ImageFolder(root=data_dir, transform=EVAL_TRANSFORM)
    train_size = int(fractions[0] * len(full_dataset))
    val_size = int(fractions[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return test_ds


def select_correct_images(model, test_ds: Subset) -> list[int]:
    """Run fault-free inference and return dataset indices the model gets right.

    Every SDC is then a fault that turns a correct prediction into a wrong one.
    TODO: implement (Action item: "Run clean inference ... keep only correctly
    classified images").
    """
    raise NotImplementedError


def balanced_sample(indices: list[int], labels: list[int], per_class: int, seed: int) -> list[int]:
    """Sample the same number of correctly classified images from each class.

    TODO: implement; `per_class` is set after the pilot.
    """
    raise NotImplementedError


def fit_eval_split(indices: list[int], fit_fraction: float, seed: int) -> dict[str, list[int]]:
    """Split image indices into disjoint {"fit": [...], "eval": [...]} sets.

    Stratify by class. The fitting set derives range bounds, trains the learned
    and probe detectors, and builds the reference ranges for LM prompts; all
    reported results use the evaluation set.
    TODO: implement and save to paths.splits.
    """
    raise NotImplementedError
