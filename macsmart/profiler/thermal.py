"""Thermal state monitoring using powermetrics and IOKit."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass
class ThermalInfo:
    """Current thermal state of the system."""

    thermal_level: str  # "nominal", "moderate", "heavy", "trapping", "sleeping"
    cpu_temp_celsius: float | None
    gpu_temp_celsius: float | None


def _parse_powermetrics_temps(output: str) -> dict[str, float]:
    """Extract temperature readings from powermetrics output.

    Args:
        output: Raw stdout from powermetrics.

    Returns:
        Dict mapping sensor names to temperatures in Celsius.
    """
    temps: dict[str, float] = {}
    for line in output.splitlines():
        # powermetrics lines like: "CPU die temperature: 45.3 C"
        if "temperature" in line.lower() and "C" in line:
            parts = line.rsplit(":", 1)
            if len(parts) == 2:
                name = parts[0].strip().lower()
                val_str = parts[1].strip().rstrip(" C").rstrip(" c")
                try:
                    temps[name] = float(val_str)
                except ValueError:
                    continue
    return temps


def get_thermal_state() -> ThermalInfo:
    """Get current thermal state.

    Attempts to read sensor data via `sudo powermetrics`. If sudo is
    unavailable, returns nominal with no temperature data. This is
    expected for non-root invocations.

    Returns:
        ThermalInfo with thermal level and temperatures.
    """
    cpu_temp: float | None = None
    gpu_temp: float | None = None
    level = "nominal"

    if platform.system() != "Darwin":
        return ThermalInfo(thermal_level=level, cpu_temp_celsius=None, gpu_temp_celsius=None)

    # Try powermetrics (requires sudo — will fail gracefully without it)
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "smc", "-i", "1000", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            temps = _parse_powermetrics_temps(result.stdout)
            for key, val in temps.items():
                if "cpu" in key and "die" in key:
                    cpu_temp = val
                elif "gpu" in key:
                    gpu_temp = val

            # Determine thermal level from CPU temp
            if cpu_temp is not None:
                if cpu_temp < 70:
                    level = "nominal"
                elif cpu_temp < 85:
                    level = "moderate"
                elif cpu_temp < 95:
                    level = "heavy"
                else:
                    level = "trapping"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ThermalInfo(
        thermal_level=level,
        cpu_temp_celsius=cpu_temp,
        gpu_temp_celsius=gpu_temp,
    )


def is_throttling() -> bool:
    """Check if the system is currently thermal-throttling.

    Returns:
        True if thermal state indicates throttling (heavy or worse).
    """
    state = get_thermal_state()
    return state.thermal_level in ("heavy", "trapping", "sleeping")
