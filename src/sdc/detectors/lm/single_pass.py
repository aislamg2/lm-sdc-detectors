from sdc.detectors.base import Detector


class SinglePassDetector(Detector):
    """Non-thinking mode, one forward pass: Eq. 2 with r_i omitted.

    No trained decision head, so the score need not be calibrated for detection.
    """
    name = "lm_single_pass"

    def __init__(self, client, condition: str = "zero_shot", access_level: str = "whitebox"):
        super().__init__(access_level)
        self.client = client
        self.condition = condition
        self.reference_text: str | None = None

    def fit(self, fit_runs, reference_ranges=None):
        # TODO: serialize reference_ranges when condition == "reference".
        raise NotImplementedError

    def score(self, runs):
        # TODO: implement.
        raise NotImplementedError
