from sdc.detectors.base import Detector


class QuantileMonitor(Detector):
    """Like RangeCheck but with per-layer quantile bounds (e.g. 0.1% / 99.9%) on
    each statistic, which is more robust than raw min/max.

    Listed in the planning notes but not yet described in the proposal;
    TODO: agree on the exact definition and add it to the paper.
    """
    name = "quantile"

    def fit(self, fit_runs, reference_ranges=None):
        # TODO: implement.
        raise NotImplementedError

    def score(self, runs):
        # TODO: implement.
        raise NotImplementedError
