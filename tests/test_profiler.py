"""Tests for the system profiler module."""

from __future__ import annotations

import platform
from unittest.mock import MagicMock, patch

import pytest

from macsmart.profiler.battery import BatteryInfo, get_battery_info, is_on_ac_power
from macsmart.profiler.memory import MemoryInfo, get_available_for_llm, get_memory_info, get_memory_pressure
from macsmart.profiler.system import (
    ChipInfo,
    _parse_chip_family,
    get_chip_info,
    get_system_profile,
    get_total_memory_gb,
)
from macsmart.profiler.thermal import ThermalInfo, get_thermal_state, is_throttling


class TestMemoryProfiler:
    """Tests for memory profiling functions."""

    def test_get_memory_info(self) -> None:
        """Test that memory info returns valid data."""
        info = get_memory_info()
        assert isinstance(info, MemoryInfo)
        assert info.total_gb > 0
        assert info.available_gb >= 0
        assert info.used_gb >= 0
        assert info.swap_used_gb >= 0
        assert info.memory_pressure in ("nominal", "warn", "critical")

    def test_get_available_for_llm(self) -> None:
        """Test LLM-available memory calculation returns positive value."""
        available = get_available_for_llm()
        assert isinstance(available, float)
        assert available >= 0
        # Should be less than total RAM
        assert available < get_total_memory_gb() + 1  # small tolerance

    def test_memory_pressure_values(self) -> None:
        """Test that memory pressure returns a valid level."""
        pressure = get_memory_pressure()
        assert pressure in ("nominal", "warn", "critical")

    def test_memory_pressure_fallback(self) -> None:
        """Test that memory pressure falls back to psutil when command fails."""
        with patch("macsmart.profiler.memory.subprocess.run", side_effect=FileNotFoundError):
            pressure = get_memory_pressure()
            assert pressure in ("nominal", "warn", "critical")


class TestSystemProfiler:
    """Tests for chip and system detection."""

    def test_get_chip_info(self) -> None:
        """Test chip detection returns valid ChipInfo."""
        info = get_chip_info()
        assert isinstance(info, ChipInfo)
        assert info.chip_name != ""
        assert info.cpu_cores > 0

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_get_chip_info_macos(self) -> None:
        """Test chip detection on macOS returns Apple Silicon details."""
        info = get_chip_info()
        assert "Apple" in info.chip_name or info.chip_family != "Unknown"
        assert info.gpu_cores > 0

    def test_get_total_memory_gb(self) -> None:
        """Test total memory detection returns reasonable value."""
        total = get_total_memory_gb()
        assert isinstance(total, float)
        assert total > 0
        # Reasonable range: 4 GB to 256 GB
        assert 4 <= total <= 256

    def test_parse_chip_family(self) -> None:
        """Test chip family parsing from chip name."""
        assert _parse_chip_family("Apple M5") == "M5"
        assert _parse_chip_family("Apple M4 Pro") == "M4 Pro"
        assert _parse_chip_family("Apple M3 Max") == "M3 Max"
        assert _parse_chip_family("Apple M1 Ultra") == "M1 Ultra"
        assert _parse_chip_family("Apple M2") == "M2"

    def test_get_system_profile(self) -> None:
        """Test full system profile returns all sections."""
        profile = get_system_profile()
        assert isinstance(profile, dict)
        assert "chip" in profile
        assert "memory" in profile
        assert "thermal" in profile
        assert "battery" in profile
        # Verify chip section has expected keys
        assert "name" in profile["chip"]
        assert "cpu_cores" in profile["chip"]
        # Verify memory section
        assert "total_gb" in profile["memory"]
        assert "available_gb" in profile["memory"]
        assert "pressure" in profile["memory"]


class TestThermalProfiler:
    """Tests for thermal state monitoring."""

    def test_get_thermal_state(self) -> None:
        """Test thermal state detection returns valid ThermalInfo."""
        info = get_thermal_state()
        assert isinstance(info, ThermalInfo)
        assert info.thermal_level in ("nominal", "moderate", "heavy", "trapping", "sleeping")

    def test_get_thermal_state_without_sudo(self) -> None:
        """Test that thermal state works without sudo access."""
        with patch("macsmart.profiler.thermal.subprocess.run", side_effect=FileNotFoundError):
            info = get_thermal_state()
            assert info.thermal_level == "nominal"
            assert info.cpu_temp_celsius is None
            assert info.gpu_temp_celsius is None

    def test_is_throttling(self) -> None:
        """Test throttling detection returns boolean."""
        result = is_throttling()
        assert isinstance(result, bool)

    def test_is_throttling_detects_heavy(self) -> None:
        """Test that heavy thermal state is detected as throttling."""
        with patch(
            "macsmart.profiler.thermal.get_thermal_state",
            return_value=ThermalInfo(thermal_level="heavy", cpu_temp_celsius=90.0, gpu_temp_celsius=None),
        ):
            assert is_throttling() is True

    def test_is_throttling_nominal_is_false(self) -> None:
        """Test that nominal thermal state is not throttling."""
        with patch(
            "macsmart.profiler.thermal.get_thermal_state",
            return_value=ThermalInfo(thermal_level="nominal", cpu_temp_celsius=45.0, gpu_temp_celsius=None),
        ):
            assert is_throttling() is False


class TestBatteryProfiler:
    """Tests for battery detection."""

    def test_get_battery_info(self) -> None:
        """Test battery info retrieval returns valid BatteryInfo."""
        info = get_battery_info()
        assert isinstance(info, BatteryInfo)
        assert 0 <= info.percent <= 100
        assert isinstance(info.is_plugged_in, bool)
        assert isinstance(info.is_charging, bool)

    def test_get_battery_info_no_battery(self) -> None:
        """Test battery info on a desktop (no battery)."""
        with patch("macsmart.profiler.battery.psutil") as mock_psutil:
            mock_psutil.sensors_battery.return_value = None
            mock_psutil.POWER_TIME_UNLIMITED = -2
            info = get_battery_info()
            assert info.percent == 100.0
            assert info.is_plugged_in is True

    def test_is_on_ac_power(self) -> None:
        """Test AC power detection returns boolean."""
        result = is_on_ac_power()
        assert isinstance(result, bool)

    def test_is_on_ac_power_no_battery(self) -> None:
        """Test AC power returns True when no battery exists."""
        with patch("macsmart.profiler.battery.psutil") as mock_psutil:
            mock_psutil.sensors_battery.return_value = None
            assert is_on_ac_power() is True
