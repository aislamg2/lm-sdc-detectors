from sdc.detectors.base import Detector


class LearnedDetector(Detector):
    """Logistic regression on the raw observation features (supervised baseline).

    Uses the same classifier as the LM probe detector, so probe vs. learned
    isolates what the LM representation adds (ablation, RQ4).
    """
    name = "learned"
    supervised = True

    def fit(self, fit_runs, reference_ranges=None):
        # TODO: sklearn LogisticRegression on observations.stats.feature_vector,
        # trained on SDC vs. masked fitting runs.
        raise NotImplementedError

    def score(self, runs):
        # TODO: predict_proba[:, 1].
        raise NotImplementedError
