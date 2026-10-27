from sdc.detectors.base import Detector


class GenerativeDetector(Detector):
    """Thinking mode: generate reasoning r_i, then
    s_i = P(yes|o_i,r_i) / (P(yes|o_i,r_i) + P(no|o_i,r_i))   (proposal Eq. 2).

    Runs with no parseable answer get s_i = 0.5 and are counted separately.
    """
    name = "lm_generative"

    def __init__(self, client, condition: str = "zero_shot", access_level: str = "whitebox"):
        super().__init__(access_level)
        self.client = client
        self.condition = condition
        self.reference_text: str | None = None

    def fit(self, fit_runs, reference_ranges=None):
        # TODO: serialize reference_ranges when condition == "reference".
        raise NotImplementedError

    def score(self, runs):
        # TODO: implement; also record the no-answer rate.
        raise NotImplementedError
