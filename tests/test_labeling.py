import pytest
import torch

from sdc.injection.labeling import label_run
from sdc.records.schema import Label

CLEAN_PRED = 2


def logits(top_class: int = CLEAN_PRED) -> torch.Tensor:
    out = torch.zeros(10)
    out[top_class] = 5.0
    return out


def test_same_prediction_is_masked():
    assert label_run(CLEAN_PRED, logits(), crashed=False) == Label.MASKED


def test_changed_prediction_is_sdc():
    assert label_run(CLEAN_PRED, logits(top_class=7), crashed=False) == Label.SDC


def test_batch_dimension_gives_same_label():
    assert label_run(CLEAN_PRED, logits().unsqueeze(0), crashed=False) == Label.MASKED
    assert label_run(CLEAN_PRED, logits(7).unsqueeze(0), crashed=False) == Label.SDC


def test_crashed_flag_wins_even_with_finite_logits():
    assert label_run(CLEAN_PRED, logits(), crashed=True) == Label.CRASH


def test_missing_logits_is_crash():
    assert label_run(CLEAN_PRED, None, crashed=False) == Label.CRASH


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("top_class", [CLEAN_PRED, 7])
def test_non_finite_logit_is_crash_whatever_argmax_says(bad, top_class):
    out = logits(top_class)
    out[0] = bad
    assert label_run(CLEAN_PRED, out, crashed=False) == Label.CRASH


def test_large_finite_logits_are_not_a_crash():
    out = logits()
    out[CLEAN_PRED] = 1e30
    assert label_run(CLEAN_PRED, out, crashed=False) == Label.MASKED
