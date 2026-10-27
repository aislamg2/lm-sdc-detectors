from sdc.detectors.base import Detector


class ConfidenceCheck(Detector):
    """s_i = entropy of the output distribution (black-box)."""
    name = "confidence"

    def score(self, runs):
        # TODO: implement.
        raise NotImplementedError
