"""Common interface for every detector: higher score = more likely SDC."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Detector(ABC):
    name: str = "detector"
    supervised: bool = False   # True if fit() uses SDC labels

    def __init__(self, access_level: str = "whitebox"):
        self.access_level = access_level   # "blackbox" or "whitebox"
        self.threshold: float | None = None

    def fit(self, fit_runs: pd.DataFrame, reference_ranges: dict | None = None) -> "Detector":
        """Fit on the fitting set only. Default: nothing to fit."""
        return self

    @abstractmethod
    def score(self, runs: pd.DataFrame) -> np.ndarray:
        """Return one score s_i per run."""

    def freeze_threshold(self, fit_runs: pd.DataFrame, target_fpr: float = 0.05) -> float:
        """Choose the threshold giving `target_fpr` on masked fitting runs, and store it.

        TODO: implement (Action item: "Freeze every detector's threshold on the fitting set").
        """
        raise NotImplementedError
