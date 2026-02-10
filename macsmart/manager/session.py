"""Session manager for end-to-end model inference."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

from macsmart.manager.download import download_model, is_model_cached
from macsmart.manager.runtime import generate, load_model, unload_model
from macsmart.profiler.memory import get_available_for_llm
from macsmart.recommender.engine import recommend_models
from macsmart.recommender.task_router import detect_task_with_confidence, validate_task


class SessionState(Enum):
    """Lifecycle states for a Session."""

    IDLE = "idle"
    MODEL_SELECTED = "model_selected"
    LOADED = "loaded"
    GENERATING = "generating"
    SWAPPING = "swapping"
    ERROR = "error"


class ModelSwappedError(Exception):
    """Raised when a model swap interrupts generation."""


@dataclass
class SessionConfig:
    """Configuration for a Session run."""

    prompt: str
    task: str | None = None
    model: str | None = None
    max_tokens: int = 256
    memory_override: float | None = None
    auto_swap: bool = False


@dataclass
class Session:
    """Manages the full lifecycle of a single inference run.

    Handles model selection, loading, generation, and unloading.
    Thread-safe for use with DynamicSwapper.
    """

    config: SessionConfig
    state: SessionState = field(default=SessionState.IDLE, init=False)
    selected_model: str | None = field(default=None, init=False)
    selected_task: str | None = field(default=None, init=False)
    task_confidence: float = field(default=0.0, init=False)
    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _swap_pending: bool = field(default=False, init=False, repr=False)
    _swap_target: str | None = field(default=None, init=False, repr=False)
    swap_history: list[dict] = field(default_factory=list, init=False)

    def select_model(self) -> str:
        """Pick the best model based on task detection and memory.

        Returns:
            HuggingFace repo ID of the selected model.

        Raises:
            RuntimeError: If no suitable model is found.
        """
        with self._lock:
            if self.config.model:
                self.selected_model = self.config.model
                self.state = SessionState.MODEL_SELECTED
                return self.config.model

            # Detect or validate task
            if self.config.task:
                self.selected_task = validate_task(self.config.task)
                self.task_confidence = 1.0
            else:
                self.selected_task, self.task_confidence = detect_task_with_confidence(
                    self.config.prompt
                )

            available = (
                self.config.memory_override
                if self.config.memory_override is not None
                else get_available_for_llm()
            )

            recs = recommend_models(available, task=self.selected_task)
            if not recs:
                self.state = SessionState.ERROR
                raise RuntimeError(
                    f"No models found for task '{self.selected_task}' "
                    f"with {available:.1f} GB available"
                )

            self.selected_model = recs[0].hf_repo
            self.state = SessionState.MODEL_SELECTED
            return self.selected_model

    def load(self) -> None:
        """Download (if needed) and load the selected model into memory.

        Raises:
            RuntimeError: If no model has been selected.
        """
        with self._lock:
            if self.selected_model is None:
                raise RuntimeError("No model selected. Call select_model() first.")

            if not is_model_cached(self.selected_model):
                download_model(self.selected_model)

            self._model, self._tokenizer = load_model(self.selected_model)
            self.state = SessionState.LOADED

    def generate(self) -> object:
        """Run inference on the loaded model.

        Returns:
            GenerationResult from the runtime.

        Raises:
            RuntimeError: If model is not loaded.
            ModelSwappedError: If a swap was triggered during generation.
        """
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            self.state = SessionState.GENERATING
            model = self._model
            tokenizer = self._tokenizer

        # Run generation outside the lock so swapper can set _swap_pending
        result = generate(model, tokenizer, self.config.prompt, max_tokens=self.config.max_tokens)

        with self._lock:
            if self._swap_pending:
                self._swap_pending = False
                target = self._swap_target
                self._swap_target = None
                self.state = SessionState.SWAPPING
                raise ModelSwappedError(
                    f"Model swap requested to {target}"
                )
            self.state = SessionState.LOADED
        return result

    def unload(self) -> None:
        """Unload the current model and free memory."""
        with self._lock:
            if self._model is not None:
                unload_model(self._model)
                self._model = None
                self._tokenizer = None
            self.state = SessionState.IDLE

    def swap_to(self, new_model: str) -> None:
        """Swap the current model for a different one.

        Unloads the current model and loads the new one.

        Args:
            new_model: HuggingFace repo ID of the replacement model.
        """
        old_model = self.selected_model
        self.unload()
        self.selected_model = new_model
        self.load()
        self.swap_history.append({
            "from": old_model,
            "to": new_model,
        })

    def request_swap(self, target_model: str) -> None:
        """Request a mid-generation swap (called from swapper thread).

        Args:
            target_model: The model to swap to.
        """
        with self._lock:
            self._swap_pending = True
            self._swap_target = target_model

    def run(self) -> object:
        """Execute the full session lifecycle: select -> load -> generate -> unload.

        Returns:
            GenerationResult from inference.
        """
        self.select_model()
        self.load()
        try:
            result = self.generate()
        finally:
            self.unload()
        return result
