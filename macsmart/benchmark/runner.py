"""Run inference benchmarks: TTFT, tokens/sec, memory peak."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

from macsmart.manager.runtime import generate, load_model, unload_model

# Standard benchmark prompt covering a mix of reasoning and generation.
DEFAULT_PROMPT = (
    "Explain how memory management works in modern operating systems. "
    "Cover virtual memory, paging, and swap. Be concise but thorough."
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model_name: str
    quantization: str
    ttft_ms: float  # Time to first token
    tokens_per_sec: float  # Generation speed
    peak_memory_gb: float
    swap_used_gb: float
    total_tokens: int
    prompt_tokens: int
    generation_tokens: int
    duration_sec: float


def _detect_quantization(model_repo: str) -> str:
    """Guess quantization from the repo name.

    Args:
        model_repo: HuggingFace repo ID.

    Returns:
        Quantization string like '4bit', '8bit', or 'unknown'.
    """
    repo_lower = model_repo.lower()
    for q in ("3bit", "4bit", "8bit", "bf16", "fp16", "f16"):
        if q in repo_lower:
            return q
    return "unknown"


def run_benchmark(
    model_repo: str,
    prompt: str | None = None,
    max_tokens: int = 256,
) -> BenchmarkResult:
    """Run a standardized inference benchmark on a model.

    Loads the model, generates tokens, measures timing and memory,
    then unloads the model.

    Args:
        model_repo: HuggingFace repo ID for the model.
        prompt: Optional custom prompt. Uses default benchmark prompt if None.
        max_tokens: Maximum tokens to generate.

    Returns:
        BenchmarkResult with timing and memory measurements.
    """
    prompt = prompt or DEFAULT_PROMPT
    quantization = _detect_quantization(model_repo)
    model_name = model_repo.rsplit("/", 1)[-1]

    swap_before = psutil.swap_memory().used

    model, tokenizer = load_model(model_repo)

    t_start = time.perf_counter()
    result = generate(model, tokenizer, prompt, max_tokens=max_tokens)
    duration = time.perf_counter() - t_start

    swap_after = psutil.swap_memory().used
    swap_delta_gb = max(0.0, (swap_after - swap_before) / (1024**3))

    unload_model(model)
    unload_model(tokenizer)

    return BenchmarkResult(
        model_name=model_name,
        quantization=quantization,
        ttft_ms=result.ttft_ms,
        tokens_per_sec=result.tokens_per_sec,
        peak_memory_gb=result.peak_memory_gb,
        swap_used_gb=round(swap_delta_gb, 2),
        total_tokens=result.prompt_tokens + result.generation_tokens,
        prompt_tokens=result.prompt_tokens,
        generation_tokens=result.generation_tokens,
        duration_sec=round(duration, 2),
    )


def compare_results(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
) -> dict:
    """Compare two benchmark results.

    Args:
        result_a: First benchmark result (baseline).
        result_b: Second benchmark result (comparison).

    Returns:
        Dict with comparison metrics: speedup ratios and absolute deltas.
    """

    def _safe_ratio(a: float, b: float) -> float | None:
        if a == 0:
            return None
        return round(b / a, 2)

    return {
        "baseline": result_a.model_name,
        "comparison": result_b.model_name,
        "tokens_per_sec": {
            "baseline": result_a.tokens_per_sec,
            "comparison": result_b.tokens_per_sec,
            "speedup": _safe_ratio(result_a.tokens_per_sec, result_b.tokens_per_sec),
        },
        "ttft_ms": {
            "baseline": result_a.ttft_ms,
            "comparison": result_b.ttft_ms,
            "delta_ms": round(result_b.ttft_ms - result_a.ttft_ms, 2),
        },
        "peak_memory_gb": {
            "baseline": result_a.peak_memory_gb,
            "comparison": result_b.peak_memory_gb,
            "delta_gb": round(result_b.peak_memory_gb - result_a.peak_memory_gb, 2),
        },
        "swap_used_gb": {
            "baseline": result_a.swap_used_gb,
            "comparison": result_b.swap_used_gb,
            "delta_gb": round(result_b.swap_used_gb - result_a.swap_used_gb, 2),
        },
        "duration_sec": {
            "baseline": result_a.duration_sec,
            "comparison": result_b.duration_sec,
            "speedup": _safe_ratio(result_a.duration_sec, result_b.duration_sec),
        },
    }
