"""Detection-quality metrics (proposal §7.1).

Main task: SDC vs. masked runs. Crash runs are excluded and reported
separately as the share of all faulty runs that crash.
"""

import numpy as np
import pandas as pd


def sdc_vs_masked(runs: pd.DataFrame) -> pd.DataFrame:
    """Drop crash runs; add a binary `is_sdc` column. TODO: implement."""
    raise NotImplementedError


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """TODO: sklearn.metrics.roc_auc_score."""
    raise NotImplementedError


def recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """SDC recall at a threshold frozen on the fitting set (target 5% FPR). TODO: implement."""
    raise NotImplementedError


def crash_rate(runs: pd.DataFrame) -> float:
    """Share of all faulty runs labelled crash. TODO: implement."""
    raise NotImplementedError
