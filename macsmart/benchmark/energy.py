"""Energy measurement using powermetrics wrapper."""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass


@dataclass
class EnergyMeasurement:
    """Energy consumption during a benchmark run."""

    cpu_power_watts: float | None
    gpu_power_watts: float | None
    ane_power_watts: float | None  # Apple Neural Engine
    total_power_watts: float | None
    duration_sec: float
    total_energy_joules: float | None


def _parse_powermetrics_output(output: str) -> dict[str, float]:
    """Parse power values from powermetrics output.

    Looks for lines like:
        CPU Power: 1234 mW
        GPU Power: 567 mW
        ANE Power: 89 mW
        Combined Power (CPU + GPU + ANE): 1890 mW

    Args:
        output: Raw stdout from powermetrics.

    Returns:
        Dict mapping component names to power in watts.
    """
    powers: dict[str, float] = {}
    for line in output.splitlines():
        # Match lines like "CPU Power: 1234 mW"
        match = re.match(r"\s*([\w\s\+\(\)]+?)\s*Power[^:]*:\s*([\d.]+)\s*mW", line, re.IGNORECASE)
        if match:
            name = match.group(1).strip().lower()
            mw = float(match.group(2))
            powers[name] = mw / 1000.0  # Convert mW to W
    return powers


def measure_energy(duration_sec: float = 10.0) -> EnergyMeasurement:
    """Measure energy consumption using powermetrics.

    Requires sudo access for powermetrics. Samples for the given
    duration and returns averaged power readings.

    Args:
        duration_sec: How long to measure in seconds.

    Returns:
        EnergyMeasurement with power data. Fields are None if
        powermetrics is unavailable or the data cannot be parsed.
    """
    if not is_powermetrics_available():
        return EnergyMeasurement(
            cpu_power_watts=None,
            gpu_power_watts=None,
            ane_power_watts=None,
            total_power_watts=None,
            duration_sec=duration_sec,
            total_energy_joules=None,
        )

    interval_ms = int(duration_sec * 1000)
    try:
        result = subprocess.run(
            [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power",
                "-i", str(interval_ms),
                "-n", "1",
            ],
            capture_output=True,
            text=True,
            timeout=duration_sec + 10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return EnergyMeasurement(
            cpu_power_watts=None,
            gpu_power_watts=None,
            ane_power_watts=None,
            total_power_watts=None,
            duration_sec=duration_sec,
            total_energy_joules=None,
        )

    powers = _parse_powermetrics_output(result.stdout)

    cpu_w = powers.get("cpu")
    gpu_w = powers.get("gpu")
    ane_w = powers.get("ane")

    # Look for combined/total power
    total_w: float | None = None
    for key, val in powers.items():
        if "combined" in key or "total" in key:
            total_w = val
            break

    if total_w is None and any(v is not None for v in [cpu_w, gpu_w, ane_w]):
        total_w = sum(v for v in [cpu_w, gpu_w, ane_w] if v is not None)

    energy_j = total_w * duration_sec if total_w is not None else None

    return EnergyMeasurement(
        cpu_power_watts=cpu_w,
        gpu_power_watts=gpu_w,
        ane_power_watts=ane_w,
        total_power_watts=total_w,
        duration_sec=duration_sec,
        total_energy_joules=round(energy_j, 2) if energy_j is not None else None,
    )


def is_powermetrics_available() -> bool:
    """Check if powermetrics is available and we have passwordless sudo.

    Returns:
        True if energy measurement is possible.
    """
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-i", "1", "-n", "1"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
