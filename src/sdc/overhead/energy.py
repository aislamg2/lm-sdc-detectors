"""Energy per inference, following Shehayib & Kogge (2026):

    E = (peak power during inference - idle power) * average time per inference

Power is read manually from an inline USB-C tester, so this module just
does the arithmetic on logged values.
"""


def energy_per_inference(peak_power_w: float, idle_power_w: float, avg_time_s: float) -> float:
    """Return joules per inference. TODO: implement."""
    raise NotImplementedError
