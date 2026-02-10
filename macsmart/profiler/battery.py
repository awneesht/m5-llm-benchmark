"""Battery level and power source detection."""

from __future__ import annotations

import platform
from dataclasses import dataclass

import psutil


@dataclass
class BatteryInfo:
    """Current battery and power state."""

    percent: float
    is_plugged_in: bool
    is_charging: bool
    time_remaining_minutes: int | None


def get_battery_info() -> BatteryInfo:
    """Get battery status from psutil.

    On desktops without a battery, returns 100% plugged-in.

    Returns:
        BatteryInfo with charge level and AC power status.
    """
    batt = psutil.sensors_battery()
    if batt is None:
        # Desktop Mac — no battery, always on AC
        return BatteryInfo(
            percent=100.0,
            is_plugged_in=True,
            is_charging=False,
            time_remaining_minutes=None,
        )

    # psutil.secsleft is -1 for "calculating", -2 for "unlimited" (plugged in)
    if batt.secsleft == psutil.POWER_TIME_UNLIMITED or batt.secsleft < 0:
        remaining = None
    else:
        remaining = batt.secsleft // 60

    return BatteryInfo(
        percent=batt.percent,
        is_plugged_in=batt.power_plugged,
        is_charging=batt.power_plugged and batt.percent < 100.0,
        time_remaining_minutes=remaining,
    )


def is_on_ac_power() -> bool:
    """Check if the Mac is plugged into AC power.

    Returns:
        True if running on AC power (or if no battery exists).
    """
    if platform.system() != "Darwin":
        return True

    batt = psutil.sensors_battery()
    if batt is None:
        return True
    return batt.power_plugged
