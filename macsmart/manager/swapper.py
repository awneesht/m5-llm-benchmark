"""Dynamic model swapper — auto-downgrade on memory pressure."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from macsmart.manager.watchdog import WatchdogEvent, start_watchdog_background
from macsmart.profiler.memory import get_available_for_llm
from macsmart.recommender.engine import recommend_models


@dataclass
class SwapPolicy:
    """Policy controlling when and how model swaps occur."""

    swap_threshold_gb: float = 1.0
    strategy: str = "smaller_quant"  # "smaller_quant" or "smaller_model"
    cooldown_sec: float = 30.0
    max_swaps: int = 3
    monitor_interval_sec: float = 2.0


@dataclass
class SwapEvent:
    """Record of a single model swap."""

    timestamp: float
    from_model: str
    to_model: str
    reason: str
    memory_available_gb: float


class DynamicSwapper:
    """Background monitor that auto-downsizes models on memory pressure.

    Runs the memory watchdog in a background thread and triggers
    model swaps when thresholds are exceeded.
    """

    def __init__(self, session: object, policy: SwapPolicy | None = None) -> None:
        self.session = session
        self.policy = policy or SwapPolicy()
        self.swap_events: list[SwapEvent] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_swap_time: float = 0.0
        self._swap_count: int = 0

    def start(self) -> None:
        """Start background memory monitoring."""
        self._thread = start_watchdog_background(
            interval_sec=self.policy.monitor_interval_sec,
            swap_threshold_gb=self.policy.swap_threshold_gb,
            callback=self._on_watchdog_tick,
            stop_event=self._stop_event,
        )

    def stop(self) -> None:
        """Stop background monitoring."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _on_watchdog_tick(self, event: WatchdogEvent, alerts: list[str]) -> None:
        """Handle a watchdog tick — check if swap is needed."""
        if not alerts:
            return

        if not self._should_swap(event):
            return

        target = self._find_downgrade(event.memory_available_gb)
        if target is None:
            return

        reason = "; ".join(alerts)
        self._trigger_swap(target, reason, event.memory_available_gb)

    def _should_swap(self, event: WatchdogEvent) -> bool:
        """Check cooldown and max swap limits."""
        if self._swap_count >= self.policy.max_swaps:
            return False

        elapsed = time.time() - self._last_swap_time
        if elapsed < self.policy.cooldown_sec:
            return False

        return True

    def _find_downgrade(self, available_gb: float) -> str | None:
        """Find a smaller model that fits in available memory.

        Args:
            available_gb: Currently available memory.

        Returns:
            HuggingFace repo ID or None if no downgrade found.
        """
        current = self.session.selected_model
        if current is None:
            return None

        recs = recommend_models(available_gb)
        for rec in recs:
            if rec.fits_in_memory and rec.hf_repo != current:
                if self.policy.strategy == "smaller_quant":
                    # Prefer same family with lower quant
                    current_name = current.rsplit("/", 1)[-1].lower()
                    rec_name = rec.hf_repo.rsplit("/", 1)[-1].lower()
                    # Check if same base model family
                    current_base = current_name.split("-")[0]
                    rec_base = rec_name.split("-")[0]
                    if current_base == rec_base:
                        return rec.hf_repo
                else:
                    return rec.hf_repo

        # Fallback: any model that fits
        for rec in recs:
            if rec.fits_in_memory and rec.hf_repo != current:
                return rec.hf_repo

        return None

    def _trigger_swap(self, target: str, reason: str, available_gb: float) -> None:
        """Initiate a model swap."""
        self._swap_count += 1
        self._last_swap_time = time.time()

        event = SwapEvent(
            timestamp=time.time(),
            from_model=self.session.selected_model or "",
            to_model=target,
            reason=reason,
            memory_available_gb=available_gb,
        )
        self.swap_events.append(event)

        self.session.request_swap(target)

    @property
    def is_running(self) -> bool:
        """Whether the background monitor is active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def swap_count(self) -> int:
        """Number of swaps triggered so far."""
        return self._swap_count
