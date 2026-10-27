"""95% bootstrap confidence intervals, resampling over images (not runs)."""

from typing import Callable

import pandas as pd


def bootstrap_ci(runs: pd.DataFrame, metric: Callable[[pd.DataFrame], float],
                 n_resamples: int = 1000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Return (point estimate, lower, upper), resampling image_id with replacement.

    TODO: implement.
    """
    raise NotImplementedError
