from sdc.detectors.base import Detector


class RangeCheck(Detector):
    """Range restriction used as a detector (Chen et al., 2021), proposal Eq. 1.

    s_i = max_l max(0, m_il - b_l, a_l - n_il) / (b_l - a_l)
    where [a_l, b_l] is layer l's fault-free range on the fitting set and
    n_il, m_il are run i's min/max at layer l.
    """
    name = "range"

    def fit(self, fit_runs, reference_ranges=None):
        # TODO: store reference_ranges (or compute from clean fitting runs).
        raise NotImplementedError

    def score(self, runs):
        # TODO: implement Eq. 1.
        raise NotImplementedError
