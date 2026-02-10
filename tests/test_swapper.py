"""Tests for the dynamic model swapper."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from macsmart.manager.swapper import DynamicSwapper, SwapEvent, SwapPolicy
from macsmart.manager.watchdog import WatchdogEvent, start_watchdog_background


# ---------------------------------------------------------------------------
# SwapPolicy tests
# ---------------------------------------------------------------------------


class TestSwapPolicy:
    """Tests for SwapPolicy defaults and configuration."""

    def test_defaults(self) -> None:
        policy = SwapPolicy()
        assert policy.swap_threshold_gb == 1.0
        assert policy.strategy == "smaller_quant"
        assert policy.cooldown_sec == 30.0
        assert policy.max_swaps == 3
        assert policy.monitor_interval_sec == 2.0

    def test_custom_values(self) -> None:
        policy = SwapPolicy(
            swap_threshold_gb=2.0,
            strategy="smaller_model",
            cooldown_sec=10.0,
            max_swaps=5,
            monitor_interval_sec=1.0,
        )
        assert policy.swap_threshold_gb == 2.0
        assert policy.strategy == "smaller_model"
        assert policy.cooldown_sec == 10.0
        assert policy.max_swaps == 5
        assert policy.monitor_interval_sec == 1.0


# ---------------------------------------------------------------------------
# SwapEvent tests
# ---------------------------------------------------------------------------


class TestSwapEvent:
    """Tests for SwapEvent dataclass."""

    def test_create_event(self) -> None:
        evt = SwapEvent(
            timestamp=1000.0,
            from_model="old-model",
            to_model="new-model",
            reason="High swap usage",
            memory_available_gb=3.5,
        )
        assert evt.from_model == "old-model"
        assert evt.to_model == "new-model"
        assert evt.reason == "High swap usage"
        assert evt.memory_available_gb == 3.5


# ---------------------------------------------------------------------------
# DynamicSwapper tests
# ---------------------------------------------------------------------------


def _make_watchdog_event(
    available_gb: float = 4.0,
    swap_gb: float = 0.0,
    pressure: str = "nominal",
) -> WatchdogEvent:
    return WatchdogEvent(
        timestamp=time.time(),
        memory_used_gb=16.0 - available_gb,
        memory_available_gb=available_gb,
        swap_used_gb=swap_gb,
        pressure_level=pressure,
    )


class TestDynamicSwapper:
    """Tests for DynamicSwapper behavior."""

    def _make_swapper(self, **policy_kwargs) -> tuple[DynamicSwapper, MagicMock]:
        session = MagicMock()
        session.selected_model = "mlx-community/Test-8bit"
        policy = SwapPolicy(**policy_kwargs)
        swapper = DynamicSwapper(session=session, policy=policy)
        return swapper, session

    def test_should_swap_respects_max_swaps(self) -> None:
        swapper, _ = self._make_swapper(max_swaps=2)
        event = _make_watchdog_event()
        assert swapper._should_swap(event) is True
        swapper._swap_count = 2
        assert swapper._should_swap(event) is False

    def test_should_swap_respects_cooldown(self) -> None:
        swapper, _ = self._make_swapper(cooldown_sec=10.0)
        event = _make_watchdog_event()
        swapper._last_swap_time = time.time()
        assert swapper._should_swap(event) is False

    def test_should_swap_after_cooldown(self) -> None:
        swapper, _ = self._make_swapper(cooldown_sec=0.01)
        event = _make_watchdog_event()
        swapper._last_swap_time = time.time() - 1.0
        assert swapper._should_swap(event) is True

    @patch("macsmart.manager.swapper.recommend_models")
    def test_find_downgrade_returns_fitting_model(self, mock_recs) -> None:
        rec = MagicMock()
        rec.fits_in_memory = True
        rec.hf_repo = "mlx-community/Test-4bit"
        mock_recs.return_value = [rec]

        swapper, _ = self._make_swapper(strategy="smaller_model")
        result = swapper._find_downgrade(4.0)
        assert result == "mlx-community/Test-4bit"

    @patch("macsmart.manager.swapper.recommend_models")
    def test_find_downgrade_skips_current_model(self, mock_recs) -> None:
        rec = MagicMock()
        rec.fits_in_memory = True
        rec.hf_repo = "mlx-community/Test-8bit"
        mock_recs.return_value = [rec]

        swapper, _ = self._make_swapper(strategy="smaller_model")
        result = swapper._find_downgrade(4.0)
        assert result is None

    @patch("macsmart.manager.swapper.recommend_models", return_value=[])
    def test_find_downgrade_no_models(self, mock_recs) -> None:
        swapper, _ = self._make_swapper()
        result = swapper._find_downgrade(4.0)
        assert result is None

    def test_find_downgrade_no_selected_model(self) -> None:
        swapper, session = self._make_swapper()
        session.selected_model = None
        result = swapper._find_downgrade(4.0)
        assert result is None

    @patch("macsmart.manager.swapper.recommend_models")
    def test_trigger_swap_records_event(self, mock_recs) -> None:
        swapper, session = self._make_swapper()
        swapper._trigger_swap("new-model", "memory pressure", 3.0)
        assert len(swapper.swap_events) == 1
        assert swapper.swap_events[0].to_model == "new-model"
        assert swapper.swap_count == 1
        session.request_swap.assert_called_once_with("new-model")

    @patch("macsmart.manager.swapper.recommend_models")
    def test_on_watchdog_tick_no_alerts(self, mock_recs) -> None:
        swapper, session = self._make_swapper()
        event = _make_watchdog_event()
        swapper._on_watchdog_tick(event, [])
        session.request_swap.assert_not_called()

    @patch("macsmart.manager.swapper.recommend_models")
    def test_on_watchdog_tick_with_alert_triggers_swap(self, mock_recs) -> None:
        rec = MagicMock()
        rec.fits_in_memory = True
        rec.hf_repo = "mlx-community/Smaller-4bit"
        mock_recs.return_value = [rec]

        swapper, session = self._make_swapper(strategy="smaller_model", cooldown_sec=0.0)
        event = _make_watchdog_event(available_gb=2.0, swap_gb=2.0, pressure="warn")
        swapper._on_watchdog_tick(event, ["High swap"])
        session.request_swap.assert_called_once_with("mlx-community/Smaller-4bit")

    @patch("macsmart.manager.swapper.recommend_models")
    def test_on_watchdog_tick_respects_max_swaps(self, mock_recs) -> None:
        rec = MagicMock()
        rec.fits_in_memory = True
        rec.hf_repo = "mlx-community/Smaller-4bit"
        mock_recs.return_value = [rec]

        swapper, session = self._make_swapper(max_swaps=1, cooldown_sec=0.0)
        event = _make_watchdog_event(swap_gb=2.0, pressure="warn")

        swapper._on_watchdog_tick(event, ["alert"])
        assert swapper.swap_count == 1

        swapper._on_watchdog_tick(event, ["alert"])
        assert swapper.swap_count == 1  # No second swap

    def test_is_running_before_start(self) -> None:
        swapper, _ = self._make_swapper()
        assert swapper.is_running is False

    @patch("macsmart.manager.swapper.start_watchdog_background")
    def test_start_and_stop(self, mock_bg) -> None:
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_bg.return_value = mock_thread

        swapper, _ = self._make_swapper()
        swapper.start()
        assert swapper.is_running is True

        swapper.stop()
        mock_thread.join.assert_called_once()

    def test_default_policy(self) -> None:
        session = MagicMock()
        swapper = DynamicSwapper(session=session)
        assert swapper.policy.max_swaps == 3
        assert swapper.swap_count == 0


# ---------------------------------------------------------------------------
# start_watchdog_background tests
# ---------------------------------------------------------------------------


class TestWatchdogBackground:
    """Tests for the background watchdog thread."""

    @patch("macsmart.manager.watchdog.sample_event")
    def test_background_watchdog_calls_callback(self, mock_sample) -> None:
        mock_sample.return_value = _make_watchdog_event()
        callback = MagicMock()
        stop = threading.Event()

        thread = start_watchdog_background(
            interval_sec=0.01,
            swap_threshold_gb=1.0,
            callback=callback,
            stop_event=stop,
        )

        time.sleep(0.1)
        stop.set()
        thread.join(timeout=2.0)

        assert callback.call_count >= 1
        assert not thread.is_alive()

    @patch("macsmart.manager.watchdog.sample_event")
    def test_background_watchdog_stops_on_event(self, mock_sample) -> None:
        mock_sample.return_value = _make_watchdog_event()
        stop = threading.Event()

        thread = start_watchdog_background(
            interval_sec=0.01,
            stop_event=stop,
        )

        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    @patch("macsmart.manager.watchdog.sample_event")
    def test_background_watchdog_is_daemon(self, mock_sample) -> None:
        mock_sample.return_value = _make_watchdog_event()
        stop = threading.Event()
        thread = start_watchdog_background(
            interval_sec=0.01,
            stop_event=stop,
        )
        assert thread.daemon is True
        stop.set()
        thread.join(timeout=2.0)

    @patch("macsmart.manager.watchdog.sample_event")
    def test_background_watchdog_generates_alerts(self, mock_sample) -> None:
        mock_sample.return_value = _make_watchdog_event(
            swap_gb=2.0, pressure="critical"
        )
        callback = MagicMock()
        stop = threading.Event()

        thread = start_watchdog_background(
            interval_sec=0.01,
            swap_threshold_gb=1.0,
            callback=callback,
            stop_event=stop,
        )

        time.sleep(0.1)
        stop.set()
        thread.join(timeout=2.0)

        # Check alerts were passed to callback
        args = callback.call_args_list[0]
        alerts = args[0][1]
        assert len(alerts) >= 1
