from sdc.detectors.base import Detector


class QuantileMonitor(Detector):
    """Like RangeCheck, but the bounds for layer l are the 0.1% and 99.9% quantiles
    of its per-run min and max across clean fitting runs, which makes them less
    sensitive to a single extreme clean run. Needs no fields beyond LayerStats.

    Listed in the planning notes but not in the proposal.
    TODO: confirm this definition with the team and add it to the paper.
    """
    name = "quantile"

    def fit(self, fit_runs, reference_ranges=None):
        # TODO: implement.
        raise NotImplementedError

    def score(self, runs):
        # TODO: implement.
        raise NotImplementedError
