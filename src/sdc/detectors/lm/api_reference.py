"""Optional: a hosted-API model as a reference detector on a subset (planning notes).

Detection quality only; not deployable on the Pi, so no overhead numbers.
"""

from sdc.detectors.base import Detector


class APIReferenceDetector(Detector):
    name = "api_reference"

    def score(self, runs):
        # TODO: implement if the team decides to run it.
        raise NotImplementedError
