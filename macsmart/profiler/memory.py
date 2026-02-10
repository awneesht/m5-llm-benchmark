"""Real-time memory profiling using vm_stat and psutil."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class MemoryInfo:
    """Snapshot of current memory state."""

    total_gb: float
    available_gb: float
    used_gb: float
    swap_used_gb: float
    memory_pressure: str  # "nominal", "warn", "critical"


def _bytes_to_gb(b: int) -> float:
    """Convert bytes to GB, rounded to 2 decimal places."""
    return round(b / (1024**3), 2)


def get_memory_info() -> MemoryInfo:
    """Get current memory usage from psutil.

    Returns:
        MemoryInfo with total, available, used, and swap memory in GB.
    """
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    pressure = get_memory_pressure()

    return MemoryInfo(
        total_gb=_bytes_to_gb(vm.total),
        available_gb=_bytes_to_gb(vm.available),
        used_gb=_bytes_to_gb(vm.used),
        swap_used_gb=_bytes_to_gb(swap.used),
        memory_pressure=pressure,
    )


def get_available_for_llm() -> float:
    """Calculate memory available for LLM inference.

    Uses actual available memory from psutil rather than a static
    budget, since app usage varies at runtime.

    Returns:
        Available memory in GB for loading a model.
    """
    vm = psutil.virtual_memory()
    return _bytes_to_gb(vm.available)


def get_memory_pressure() -> str:
    """Get macOS memory pressure level.

    Falls back to a heuristic based on psutil percent if the
    macOS `memory_pressure` command is unavailable.

    Returns:
        One of 'nominal', 'warn', or 'critical'.
    """
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["/usr/bin/memory_pressure"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout
            # Parse "System-wide memory free percentage: NN%"
            for line in output.splitlines():
                if "free percentage" in line.lower():
                    pct_str = line.rstrip().rstrip("%").rsplit(":", 1)[-1].strip()
                    free_pct = int(pct_str)
                    if free_pct > 30:
                        return "nominal"
                    elif free_pct > 10:
                        return "warn"
                    else:
                        return "critical"
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

    # Fallback: use psutil percent (percent *used*)
    vm = psutil.virtual_memory()
    if vm.percent < 70:
        return "nominal"
    elif vm.percent < 90:
        return "warn"
    else:
        return "critical"
