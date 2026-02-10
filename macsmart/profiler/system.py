"""Chip detection, GPU core count, and memory bandwidth."""

from __future__ import annotations

import platform
import plistlib
import re
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class ChipInfo:
    """Apple Silicon chip details."""

    chip_name: str  # e.g. "Apple M5"
    chip_family: str  # e.g. "M5"
    cpu_cores: int
    gpu_cores: int
    neural_engine_cores: int | None
    memory_bandwidth_gbps: float | None
    has_neural_accelerator: bool


# Known memory bandwidth values (GB/s) for Apple Silicon chips.
# Source: Apple spec sheets. Keys are chip family prefixes.
_BANDWIDTH_TABLE: dict[str, float] = {
    "M1": 68.25,
    "M1 Pro": 200.0,
    "M1 Max": 400.0,
    "M1 Ultra": 800.0,
    "M2": 100.0,
    "M2 Pro": 200.0,
    "M2 Max": 400.0,
    "M2 Ultra": 800.0,
    "M3": 100.0,
    "M3 Pro": 150.0,
    "M3 Max": 400.0,
    "M3 Ultra": 800.0,
    "M4": 120.0,
    "M4 Pro": 273.0,
    "M4 Max": 546.0,
    "M4 Ultra": 819.0,
    "M5": 120.0,
    "M5 Pro": 273.0,
    "M5 Max": 546.0,
    "M5 Ultra": 819.0,
}

# Neural Engine core counts by chip family.
_NEURAL_ENGINE_CORES: dict[str, int] = {
    "M1": 16,
    "M2": 16,
    "M3": 16,
    "M4": 16,
    "M5": 16,
}


def _run_cmd(cmd: list[str], timeout: int = 5) -> str:
    """Run a shell command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _parse_chip_family(chip_name: str) -> str:
    """Extract the chip family from the full chip name.

    Examples:
        "Apple M5"       -> "M5"
        "Apple M4 Pro"   -> "M4 Pro"
        "Apple M3 Max"   -> "M3 Max"
        "Apple M1 Ultra" -> "M1 Ultra"
    """
    # Strip "Apple " prefix and return the rest
    name = chip_name.replace("Apple ", "").strip()
    return name if name else "Unknown"


def _get_gpu_cores_from_system_profiler() -> int | None:
    """Get GPU core count from system_profiler SPDisplaysDataType."""
    output = _run_cmd(["system_profiler", "SPDisplaysDataType"])
    for line in output.splitlines():
        if "Total Number of Cores" in line:
            match = re.search(r"(\d+)", line)
            if match:
                return int(match.group(1))
    return None


def _get_hardware_info_xml() -> dict:
    """Parse system_profiler SPHardwareDataType as structured data."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-xml"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        plist = plistlib.loads(result.stdout)
        for item in plist:
            items = item.get("_items", [])
            if items:
                return items[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, plistlib.InvalidFileException):
        pass
    return {}


def get_chip_info() -> ChipInfo:
    """Detect Apple Silicon chip and its capabilities.

    Uses sysctl and system_profiler to identify the chip,
    core counts, and memory bandwidth.

    Returns:
        ChipInfo with chip name, core counts, and capabilities.
    """
    if platform.system() != "Darwin":
        cpu_count = psutil.cpu_count(logical=True) or 1
        return ChipInfo(
            chip_name=platform.processor() or "Unknown",
            chip_family="Unknown",
            cpu_cores=cpu_count,
            gpu_cores=0,
            neural_engine_cores=None,
            memory_bandwidth_gbps=None,
            has_neural_accelerator=False,
        )

    # Chip name from sysctl
    chip_name = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    if not chip_name:
        hw_info = _get_hardware_info_xml()
        chip_name = hw_info.get("chip_type", "Unknown")

    chip_family = _parse_chip_family(chip_name)

    # CPU cores
    cpu_cores = psutil.cpu_count(logical=True) or 0

    # GPU cores from system_profiler
    gpu_cores = _get_gpu_cores_from_system_profiler() or 0

    # Neural engine cores (lookup table by base family)
    base_family = chip_family.split()[0] if chip_family else ""  # "M5 Pro" -> "M5"
    ne_cores = _NEURAL_ENGINE_CORES.get(base_family)

    # Memory bandwidth (lookup table by chip family)
    bandwidth = _BANDWIDTH_TABLE.get(chip_family)

    # Has neural accelerator: M-series chips all have ANE
    has_ane = base_family.startswith("M") and base_family[1:].split()[0].isdigit()

    return ChipInfo(
        chip_name=chip_name,
        chip_family=chip_family,
        cpu_cores=cpu_cores,
        gpu_cores=gpu_cores,
        neural_engine_cores=ne_cores,
        memory_bandwidth_gbps=bandwidth,
        has_neural_accelerator=has_ane,
    )


def get_total_memory_gb() -> float:
    """Get total system memory in GB.

    Returns:
        Total RAM in GB.
    """
    return round(psutil.virtual_memory().total / (1024**3), 2)


def get_system_profile() -> dict:
    """Get a complete system profile as a JSON-serializable dict.

    Combines chip, memory, thermal, and battery info.

    Returns:
        Dict with all system profiling data.
    """
    from macsmart.profiler.battery import get_battery_info
    from macsmart.profiler.memory import get_memory_info
    from macsmart.profiler.thermal import get_thermal_state

    chip = get_chip_info()
    mem = get_memory_info()
    thermal = get_thermal_state()
    battery = get_battery_info()

    return {
        "chip": {
            "name": chip.chip_name,
            "family": chip.chip_family,
            "cpu_cores": chip.cpu_cores,
            "gpu_cores": chip.gpu_cores,
            "neural_engine_cores": chip.neural_engine_cores,
            "memory_bandwidth_gbps": chip.memory_bandwidth_gbps,
            "has_neural_accelerator": chip.has_neural_accelerator,
        },
        "memory": {
            "total_gb": mem.total_gb,
            "available_gb": mem.available_gb,
            "used_gb": mem.used_gb,
            "swap_used_gb": mem.swap_used_gb,
            "pressure": mem.memory_pressure,
        },
        "thermal": {
            "level": thermal.thermal_level,
            "cpu_temp_celsius": thermal.cpu_temp_celsius,
            "gpu_temp_celsius": thermal.gpu_temp_celsius,
        },
        "battery": {
            "percent": battery.percent,
            "is_plugged_in": battery.is_plugged_in,
            "is_charging": battery.is_charging,
            "time_remaining_minutes": battery.time_remaining_minutes,
        },
    }
