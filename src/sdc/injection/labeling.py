"""Label a faulty run against its fault-free run on the same image (proposal §4)."""

import torch

from sdc.records.schema import Label


def label_run(clean_pred: int, faulty_logits: torch.Tensor | None, crashed: bool) -> Label:
    """masked / sdc / crash, for one image's logits (shape [10] or [1, 10]).

    crash: raised, timed out, or produced non-finite logits.
    sdc:   finite logits, argmax differs from clean_pred.
    masked: finite logits, same argmax.
    """
    # Check this first: argmax still returns a class when logits contain NaN.
    if crashed or faulty_logits is None or not torch.isfinite(faulty_logits).all():
        return Label.CRASH
    if int(faulty_logits.argmax()) != clean_pred:
        return Label.SDC
    return Label.MASKED
