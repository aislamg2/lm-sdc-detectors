"""Serialize o_i as plain text for the language-model detectors (proposal §6.2).

Example white-box line:
    layer3: min=-2.10 max=11.84 mean=0.41 std=1.22 nonfinite=0
followed by the output statistics.
"""


def serialize_observation(record_row: dict, access_level: str) -> str:
    """One line per monitored point (white-box only), then top-1 prob and entropy.

    TODO: implement; keep number formatting fixed so prompts are reproducible.
    """
    raise NotImplementedError


def serialize_reference(ranges: dict[str, dict[str, float]]) -> str:
    """Per-layer normal ranges from fault-free runs, for the "reference" condition.

    TODO: implement.
    """
    raise NotImplementedError
