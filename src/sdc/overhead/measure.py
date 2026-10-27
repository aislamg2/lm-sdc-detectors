"""Per-detector overhead on the Raspberry Pi 5 (proposal §7.2).

Measured in a separate session from the detection runs, with temperature
logged to catch thermal throttling.
"""

from typing import Callable


def time_per_call(fn: Callable[[], object], warmup: int = 10, iterations: int = 100) -> dict:
    """Mean / p50 / p95 latency in seconds (time.perf_counter). TODO: implement."""
    raise NotImplementedError


def peak_memory(fn: Callable[[], object]) -> int:
    """Peak resident memory in bytes while running fn (resource / psutil). TODO: implement."""
    raise NotImplementedError


def read_cpu_temp_c() -> float:
    """Read /sys/class/thermal/thermal_zone0/temp. TODO: implement."""
    raise NotImplementedError
