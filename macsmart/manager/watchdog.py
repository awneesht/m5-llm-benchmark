"""Memory watchdog — monitor and alert on memory pressure during inference."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import psutil

from macsmart.profiler.memory import _bytes_to_gb, get_memory_pressure


@dataclass
class WatchdogEvent:
    """A memory pressure event recorded by the watchdog."""

    timestamp: float
    memory_used_gb: float
    memory_available_gb: float
    swap_used_gb: float
    pressure_level: str  # "nominal", "warn", "critical"


def sample_event() -> WatchdogEvent:
    """Take a single memory snapshot right now.

    Returns:
        WatchdogEvent with the current memory state.
    """
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    pressure = get_memory_pressure()

    return WatchdogEvent(
        timestamp=time.time(),
        memory_used_gb=_bytes_to_gb(vm.used),
        memory_available_gb=_bytes_to_gb(vm.available),
        swap_used_gb=_bytes_to_gb(swap.used),
        pressure_level=pressure,
    )


def start_watchdog(
    interval_sec: float = 1.0,
    swap_threshold_gb: float = 1.0,
    callback: Callable[[WatchdogEvent, list[str]], None] | None = None,
    max_events: int = 0,
) -> list[WatchdogEvent]:
    """Start monitoring memory pressure in a loop.

    Runs until KeyboardInterrupt (or until *max_events* samples are
    collected when max_events > 0). Each iteration samples memory,
    checks thresholds, and invokes the optional callback.

    Args:
        interval_sec: Polling interval in seconds.
        swap_threshold_gb: Swap usage threshold to trigger an alert.
        callback: Optional callable invoked each tick with
            (event, alerts) where *alerts* is a list of warning
            strings (empty when everything is fine).
        max_events: Stop after this many events. 0 means run forever.

    Returns:
        List of all collected WatchdogEvent objects.
    """
    events: list[WatchdogEvent] = []
    count = 0

    try:
        while True:
            event = sample_event()
            events.append(event)
            count += 1

            alerts: list[str] = []
            if event.swap_used_gb >= swap_threshold_gb:
                alerts.append(
                    f"Swap usage {event.swap_used_gb:.2f} GB exceeds "
                    f"threshold ({swap_threshold_gb:.1f} GB)"
                )
            if event.pressure_level == "critical":
                alerts.append("Memory pressure is CRITICAL")
            elif event.pressure_level == "warn":
                alerts.append("Memory pressure elevated (warn)")

            if callback is not None:
                callback(event, alerts)

            if max_events > 0 and count >= max_events:
                break

            time.sleep(interval_sec)
    except KeyboardInterrupt:
        pass

    return events


def get_memory_timeline(events: list[WatchdogEvent]) -> dict:
    """Summarize a timeline of watchdog events.

    Args:
        events: List of recorded watchdog events.

    Returns:
        Dict with min/max/avg memory stats over the timeline,
        plus duration and sample count.  Returns empty stats
        when the events list is empty.
    """
    if not events:
        return {
            "samples": 0,
            "duration_sec": 0.0,
            "memory_used_gb": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "memory_available_gb": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "swap_used_gb": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "pressure_levels": {},
        }

    used = [e.memory_used_gb for e in events]
    avail = [e.memory_available_gb for e in events]
    swap = [e.swap_used_gb for e in events]

    # Count how many samples fell into each pressure level
    pressure_counts: dict[str, int] = {}
    for e in events:
        pressure_counts[e.pressure_level] = pressure_counts.get(e.pressure_level, 0) + 1

    duration = events[-1].timestamp - events[0].timestamp if len(events) > 1 else 0.0

    def _stats(values: list[float]) -> dict:
        return {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "avg": round(sum(values) / len(values), 2),
        }

    return {
        "samples": len(events),
        "duration_sec": round(duration, 2),
        "memory_used_gb": _stats(used),
        "memory_available_gb": _stats(avail),
        "swap_used_gb": _stats(swap),
        "pressure_levels": pressure_counts,
    }
