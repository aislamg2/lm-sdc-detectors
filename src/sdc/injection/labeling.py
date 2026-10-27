"""Label a faulty run against its fault-free run on the same image (proposal §4)."""

import torch

from sdc.records.schema import Label


def label_run(clean_pred: int, faulty_logits: torch.Tensor | None, crashed: bool) -> Label:
    """masked / sdc / crash.

    crash: raised, timed out, or produced non-finite logits.
    sdc:   finite logits, argmax differs from clean_pred.
    masked: finite logits, same argmax.
    TODO: implement.
    """
    raise NotImplementedError
