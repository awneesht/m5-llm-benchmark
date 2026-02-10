"""Tests for the memory watchdog module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from macsmart.manager.watchdog import (
    WatchdogEvent,
    get_memory_timeline,
    sample_event,
    start_watchdog,
)


def _make_event(
    timestamp: float = 1000.0,
    used: float = 8.0,
    available: float = 8.0,
    swap: float = 0.0,
    pressure: str = "nominal",
) -> WatchdogEvent:
    """Helper to build a WatchdogEvent with defaults."""
    return WatchdogEvent(
        timestamp=timestamp,
        memory_used_gb=used,
        memory_available_gb=available,
        swap_used_gb=swap,
        pressure_level=pressure,
    )


# ---------------------------------------------------------------------------
# sample_event tests
# ---------------------------------------------------------------------------


class TestSampleEvent:
    """Tests for sampling a single memory snapshot."""

    def test_sample_event_returns_watchdog_event(self) -> None:
        """sample_event returns a WatchdogEvent with realistic values."""
        event = sample_event()
        assert isinstance(event, WatchdogEvent)
        assert event.memory_used_gb > 0
        assert event.memory_available_gb >= 0
        assert event.swap_used_gb >= 0
        assert event.pressure_level in ("nominal", "warn", "critical")
        assert event.timestamp > 0

    def test_sample_event_timestamp_is_recent(self) -> None:
        """Timestamp should be close to current time."""
        before = time.time()
        event = sample_event()
        after = time.time()
        assert before <= event.timestamp <= after


# ---------------------------------------------------------------------------
# start_watchdog tests
# ---------------------------------------------------------------------------


class TestStartWatchdog:
    """Tests for the watchdog loop."""

    def test_max_events_stops_loop(self) -> None:
        """Watchdog stops after max_events samples."""
        events = start_watchdog(interval_sec=0.0, max_events=3)
        assert len(events) == 3
        for e in events:
            assert isinstance(e, WatchdogEvent)

    def test_callback_invoked_each_tick(self) -> None:
        """Callback is called once per sample with (event, alerts)."""
        calls: list[tuple] = []

        def on_tick(event: WatchdogEvent, alerts: list[str]) -> None:
            calls.append((event, alerts))

        start_watchdog(interval_sec=0.0, max_events=5, callback=on_tick)
        assert len(calls) == 5
        for event, alerts in calls:
            assert isinstance(event, WatchdogEvent)
            assert isinstance(alerts, list)

    def test_swap_alert_triggered(self) -> None:
        """Alert fires when swap exceeds threshold."""
        high_swap_event = _make_event(swap=3.5, pressure="nominal")

        with patch("macsmart.manager.watchdog.sample_event", return_value=high_swap_event):
            alerts_seen: list[list[str]] = []

            def on_tick(_event: WatchdogEvent, alerts: list[str]) -> None:
                alerts_seen.append(alerts)

            start_watchdog(
                interval_sec=0.0,
                swap_threshold_gb=1.0,
                callback=on_tick,
                max_events=1,
            )

        assert len(alerts_seen) == 1
        assert any("Swap usage" in a for a in alerts_seen[0])

    def test_critical_pressure_alert(self) -> None:
        """Alert fires when pressure is critical."""
        critical_event = _make_event(pressure="critical")

        with patch("macsmart.manager.watchdog.sample_event", return_value=critical_event):
            alerts_seen: list[list[str]] = []

            def on_tick(_event: WatchdogEvent, alerts: list[str]) -> None:
                alerts_seen.append(alerts)

            start_watchdog(interval_sec=0.0, callback=on_tick, max_events=1)

        assert any("CRITICAL" in a for a in alerts_seen[0])

    def test_warn_pressure_alert(self) -> None:
        """Alert fires when pressure is warn."""
        warn_event = _make_event(pressure="warn")

        with patch("macsmart.manager.watchdog.sample_event", return_value=warn_event):
            alerts_seen: list[list[str]] = []

            def on_tick(_event: WatchdogEvent, alerts: list[str]) -> None:
                alerts_seen.append(alerts)

            start_watchdog(interval_sec=0.0, callback=on_tick, max_events=1)

        assert any("warn" in a.lower() for a in alerts_seen[0])

    def test_no_alerts_when_nominal(self) -> None:
        """No alerts when swap is low and pressure is nominal."""
        ok_event = _make_event(swap=0.1, pressure="nominal")

        with patch("macsmart.manager.watchdog.sample_event", return_value=ok_event):
            alerts_seen: list[list[str]] = []

            def on_tick(_event: WatchdogEvent, alerts: list[str]) -> None:
                alerts_seen.append(alerts)

            start_watchdog(
                interval_sec=0.0,
                swap_threshold_gb=1.0,
                callback=on_tick,
                max_events=1,
            )

        assert alerts_seen[0] == []

    def test_no_callback_still_collects_events(self) -> None:
        """Events are collected even without a callback."""
        events = start_watchdog(interval_sec=0.0, max_events=3, callback=None)
        assert len(events) == 3

    def test_keyboard_interrupt_stops_loop(self) -> None:
        """KeyboardInterrupt is handled gracefully."""
        call_count = 0

        def exploding_sample() -> WatchdogEvent:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise KeyboardInterrupt
            return _make_event()

        with patch("macsmart.manager.watchdog.sample_event", side_effect=exploding_sample):
            events = start_watchdog(interval_sec=0.0)

        assert len(events) == 2


# ---------------------------------------------------------------------------
# get_memory_timeline tests
# ---------------------------------------------------------------------------


class TestGetMemoryTimeline:
    """Tests for timeline summarization."""

    def test_empty_events(self) -> None:
        """Empty event list returns zero-filled summary."""
        tl = get_memory_timeline([])
        assert tl["samples"] == 0
        assert tl["duration_sec"] == 0.0
        assert tl["memory_used_gb"] == {"min": 0.0, "max": 0.0, "avg": 0.0}

    def test_single_event(self) -> None:
        """Single event has zero duration and min==max==avg."""
        events = [_make_event(used=5.0, available=11.0, swap=0.5)]
        tl = get_memory_timeline(events)

        assert tl["samples"] == 1
        assert tl["duration_sec"] == 0.0
        assert tl["memory_used_gb"]["min"] == 5.0
        assert tl["memory_used_gb"]["max"] == 5.0
        assert tl["memory_used_gb"]["avg"] == 5.0

    def test_multiple_events_stats(self) -> None:
        """Min/max/avg are computed correctly across events."""
        events = [
            _make_event(timestamp=100, used=4.0, available=12.0, swap=0.0, pressure="nominal"),
            _make_event(timestamp=101, used=6.0, available=10.0, swap=1.0, pressure="nominal"),
            _make_event(timestamp=102, used=8.0, available=8.0, swap=2.0, pressure="warn"),
        ]
        tl = get_memory_timeline(events)

        assert tl["samples"] == 3
        assert tl["duration_sec"] == 2.0

        assert tl["memory_used_gb"]["min"] == 4.0
        assert tl["memory_used_gb"]["max"] == 8.0
        assert tl["memory_used_gb"]["avg"] == 6.0

        assert tl["swap_used_gb"]["min"] == 0.0
        assert tl["swap_used_gb"]["max"] == 2.0
        assert tl["swap_used_gb"]["avg"] == 1.0

    def test_pressure_level_counts(self) -> None:
        """Pressure levels are counted correctly."""
        events = [
            _make_event(pressure="nominal"),
            _make_event(pressure="nominal"),
            _make_event(pressure="warn"),
            _make_event(pressure="critical"),
        ]
        tl = get_memory_timeline(events)
        assert tl["pressure_levels"] == {"nominal": 2, "warn": 1, "critical": 1}

    def test_duration_calculation(self) -> None:
        """Duration is last timestamp minus first."""
        events = [
            _make_event(timestamp=1000.0),
            _make_event(timestamp=1005.5),
        ]
        tl = get_memory_timeline(events)
        assert tl["duration_sec"] == 5.5
