from sdc.detectors.base import Detector


class ProbeDetector(Detector):
    """Logistic-regression probe on the LM's final hidden state at the answer
    position (Alain & Bengio, 2016): an open analog of a "System One" decision head.

    Same classifier as LearnedDetector; only the input representation differs.
    """
    name = "lm_probe"
    supervised = True

    def __init__(self, client, condition: str = "zero_shot", access_level: str = "whitebox"):
        super().__init__(access_level)
        self.client = client
        self.condition = condition

    def fit(self, fit_runs, reference_ranges=None):
        # TODO: extract hidden states for fitting runs (cache them to disk) and fit.
        raise NotImplementedError

    def score(self, runs):
        # TODO: implement.
        raise NotImplementedError
