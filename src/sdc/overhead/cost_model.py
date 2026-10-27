"""Analytical cost model for LM detectors, to compare against measured Pi latency
(Action item Part II 2c).

e.g. latency ~ prompt_tokens * t_prefill + generated_tokens * t_decode
"""


def predicted_latency(prompt_tokens: int, generated_tokens: int,
                      t_prefill_s: float, t_decode_s: float) -> float:
    """TODO: implement; fit t_prefill / t_decode from the pilot."""
    raise NotImplementedError
