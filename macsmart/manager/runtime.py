"""MLX / Ollama runtime wrapper for model inference."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx_lm


@dataclass
class GenerationResult:
    """Result from a model generation call."""

    text: str
    prompt_tokens: int
    generation_tokens: int
    ttft_ms: float
    tokens_per_sec: float
    peak_memory_gb: float


def load_model(model_path: str) -> tuple:
    """Load a model into memory using MLX.

    Args:
        model_path: Path or HuggingFace repo ID for the model.

    Returns:
        Tuple of (model, tokenizer).
    """
    model, tokenizer = mlx_lm.load(model_path)
    mx.eval(model.parameters())
    return model, tokenizer


def generate(
    model: object,
    tokenizer: object,
    prompt: str,
    max_tokens: int = 256,
) -> GenerationResult:
    """Generate text from a loaded model, measuring performance.

    Uses mlx_lm.stream_generate to capture per-token timing
    for TTFT measurement and overall throughput.

    Args:
        model: Loaded MLX model.
        tokenizer: Associated tokenizer.
        prompt: Input prompt text.
        max_tokens: Maximum tokens to generate.

    Returns:
        GenerationResult with text and performance metrics.
    """
    text_parts: list[str] = []
    ttft_ms = 0.0
    prompt_tokens = 0
    generation_tokens = 0
    tokens_per_sec = 0.0
    peak_memory_gb = 0.0
    first_token = True
    t_start = time.perf_counter()

    for response in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        if first_token:
            ttft_ms = (time.perf_counter() - t_start) * 1000.0
            first_token = False

        text_parts.append(response.text)
        prompt_tokens = response.prompt_tokens
        generation_tokens = response.generation_tokens
        tokens_per_sec = response.generation_tps
        peak_memory_gb = response.peak_memory / (1024**3)

    return GenerationResult(
        text="".join(text_parts),
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        ttft_ms=round(ttft_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2),
        peak_memory_gb=round(peak_memory_gb, 2),
    )


def unload_model(model: object) -> None:
    """Unload a model from memory to free resources.

    Args:
        model: Model to unload.
    """
    del model
    gc.collect()
