from sdc.detectors.base import Detector


class SanityCheck(Detector):
    """s_i = 1 if the run crashed or produced NaN/inf, else 0."""
    name = "sanity"

    def score(self, runs):
        # TODO: implement.
        raise NotImplementedError
