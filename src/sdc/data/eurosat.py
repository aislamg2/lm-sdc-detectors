"""EuroSAT loading and the image splits used by every experiment.

Transforms and the train/val/test split are ported from Reem's notebook
(notebooks/reem_TFLite_EuroSAT_ResNet18.ipynb, from
github.com/reemshehayib/TFLite-Deployment-of-ResNet18-on-EuroSAT-Dataset) so
that "held-out test split" refers to the same images the model never trained on.

Note: in the notebook all three subsets share one ImageFolder, so the last
`.transform` assignment (eval) applied to training too. The split *indices*
depend only on dataset length and seed, so we reproduce them exactly and apply
EVAL_TRANSFORM cleanly here.

An image_id is an image's position within the test split, not within the full
dataset. It is what run records store.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
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


def load_test_split(data_dir: str, fractions=(0.8, 0.1, 0.1), seed: int = 42,
                    indices_path: str | None = None) -> Subset:
    """Return the held-out test subset.

    With `indices_path` (a JSON list of dataset indices), use exactly those images.
    Otherwise reproduce the notebook's `test_ds`.
    """
    full_dataset = datasets.ImageFolder(root=data_dir, transform=EVAL_TRANSFORM)
    if full_dataset.classes != CLASS_NAMES:
        raise ValueError(f"expected classes {CLASS_NAMES}, found {full_dataset.classes}")
    if indices_path is not None:
        with open(indices_path) as f:
            indices = json.load(f)
        if not isinstance(indices, list):
            raise ValueError(f"{indices_path} must contain a JSON list of indices")
        return Subset(full_dataset, indices)
    train_size = int(fractions[0] * len(full_dataset))
    val_size = int(fractions[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return test_ds


def select_correct_images(model, test_ds: Subset, batch_size: int = 32,
                          device: str = "cpu") -> list[int]:
    """Run fault-free inference and return the image_ids the model gets right.

    Every SDC is then a fault that turns a correct prediction into a wrong one.
    """
    correct = []
    position = 0
    with torch.no_grad():
        for images, labels in DataLoader(test_ds, batch_size=batch_size, shuffle=False):
            preds = model(images.to(device)).argmax(dim=1).cpu()
            for pred, label in zip(preds.tolist(), labels.tolist()):
                if pred == label:
                    correct.append(position)
                position += 1
    return correct


def balanced_sample(indices: list[int], labels: list[int], per_class: int | None, seed: int,
                    num_classes: int = len(CLASS_NAMES)) -> list[int]:
    """Sample the same number of images from each class; labels[i] is the class of indices[i].

    `per_class=None` uses as many as the smallest class allows. Raises if any class
    has too few, rather than returning an unbalanced set.
    """
    by_class = _group_by_class(indices, labels, num_classes)
    counts = {cls: len(members) for cls, members in by_class.items()}
    available = min(counts.values())
    if per_class is None:
        per_class = available
    if per_class < 1 or per_class > available:
        raise ValueError(f"cannot sample {per_class} images per class; available: {counts}")
    sampled = []
    for cls, members in by_class.items():
        rng = np.random.default_rng([seed, cls])
        sampled.extend(rng.choice(members, size=per_class, replace=False).tolist())
    return sorted(sampled)


def fit_eval_split(indices: list[int], labels: list[int], fit_fraction: float, seed: int,
                   num_classes: int = len(CLASS_NAMES)) -> dict[str, list[int]]:
    """Split image ids into disjoint {"fit": [...], "eval": [...]} sets, within each class.

    The fitting set derives range bounds, trains the learned and probe detectors,
    and builds the reference ranges for LM prompts; all reported results use the
    evaluation set.
    """
    fit, evaluation = [], []
    for cls, members in _group_by_class(indices, labels, num_classes).items():
        shuffled = np.random.default_rng([seed, cls]).permutation(members).tolist()
        cut = int(len(shuffled) * fit_fraction)
        if cut == 0 or cut == len(shuffled):
            raise ValueError(f"class {cls} has {len(shuffled)} images: too few to split")
        fit.extend(shuffled[:cut])
        evaluation.extend(shuffled[cut:])
    return {"fit": sorted(fit), "eval": sorted(evaluation)}


def build_splits(model, test_ds: Subset, per_class: int | None, fit_fraction: float, seed: int,
                 batch_size: int = 32, device: str = "cpu") -> dict:
    """Keep correctly classified images, balance the classes, and split fit/eval.

    Returns a JSON-ready dict. Each image is listed with its path and class, so a
    later step can detect that someone's dataset differs.
    """
    labels = [test_ds.dataset.targets[i] for i in test_ds.indices]
    correct = select_correct_images(model, test_ds, batch_size, device)
    sampled = balanced_sample(correct, [labels[i] for i in correct], per_class, seed)
    parts = fit_eval_split(sampled, [labels[i] for i in sampled], fit_fraction, seed + 1)

    def describe(image_id: int) -> dict:
        path = test_ds.dataset.samples[test_ds.indices[image_id]][0]
        return {
            "image_id": image_id,
            "image_path": os.path.relpath(path, test_ds.dataset.root),
            "true_class": labels[image_id],
        }

    return {
        "seed": seed,
        "num_test_images": len(test_ds),
        "num_correct": len(correct),
        "clean_accuracy": len(correct) / len(test_ds),
        "per_class": len(sampled) // len(CLASS_NAMES),
        "fit": [describe(i) for i in parts["fit"]],
        "eval": [describe(i) for i in parts["eval"]],
    }


def load_splits(path: str) -> dict:
    """Read the file written by scripts/01_make_splits.py."""
    with open(path) as f:
        return json.load(f)


def verify_splits(splits: dict, test_ds: Subset, checkpoint_hash: str) -> None:
    """Raise if these weights or images are not the ones the splits were built from."""
    if splits["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint differs from the one used to build the splits; rebuild them")
    for entry in splits["fit"] + splits["eval"]:
        image_id = entry["image_id"]
        if not 0 <= image_id < len(test_ds):
            raise ValueError(f"image_id {image_id} is outside the test split")
        path, label = test_ds.dataset.samples[test_ds.indices[image_id]]
        relative = os.path.relpath(path, test_ds.dataset.root)
        if relative != entry["image_path"] or label != entry["true_class"]:
            raise ValueError(
                f"image_id {image_id} is {relative} (class {label}), but the splits say "
                f"{entry['image_path']} (class {entry['true_class']}): the dataset differs"
            )


def _group_by_class(indices: list[int], labels: list[int], num_classes: int) -> dict[int, list[int]]:
    if len(indices) != len(labels):
        raise ValueError(f"got {len(indices)} indices but {len(labels)} labels")
    by_class = {cls: [] for cls in range(num_classes)}
    for index, label in zip(indices, labels):
        by_class[label].append(index)
    return by_class
