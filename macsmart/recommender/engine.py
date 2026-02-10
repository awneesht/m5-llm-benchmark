"""Core recommendation logic for model selection."""

from __future__ import annotations

from dataclasses import dataclass

from macsmart.recommender.models_db import ModelEntry, load_registry, get_models_for_task
from macsmart.recommender.task_router import validate_task


@dataclass
class Recommendation:
    """A single model recommendation with reasoning."""

    model_name: str
    quantization: str
    format: str
    hf_repo: str
    memory_required_gb: float
    quality_score: int
    fits_in_memory: bool
    swap_warning: bool
    reason: str


# Models that need more than this much swap are flagged as risky.
_SWAP_HEADROOM_GB = 2.0


def recommend_models(
    available_memory_gb: float,
    task: str | None = None,
    prefer_mlx: bool = True,
) -> list[Recommendation]:
    """Recommend models ranked by quality that fit in available memory.

    Produces a ranked list where:
    - Models that fit cleanly in memory come first (sorted by quality).
    - Models that require some swap come next (with a warning).
    - Models that would exceed available memory + swap headroom are excluded.

    When prefer_mlx is True, MLX-format models get a small quality bonus
    to float them above equivalent non-MLX variants.

    Args:
        available_memory_gb: Memory available for LLM inference.
        task: Optional task type to filter models (coding, writing, etc.).
        prefer_mlx: Whether to prefer MLX format for Apple Neural Engine.

    Returns:
        List of Recommendation objects, ranked best-first.
    """
    # Load and optionally filter models
    models: list[ModelEntry] = load_registry()
    if task is not None:
        task = validate_task(task)
        models = get_models_for_task(task, models)

    max_memory = available_memory_gb + _SWAP_HEADROOM_GB
    recommendations: list[Recommendation] = []

    for model in models:
        for quant in model.quantizations:
            if quant.memory_gb > max_memory:
                continue

            fits = quant.memory_gb <= available_memory_gb
            swap_warning = not fits

            # Build a human-readable reason
            if fits:
                headroom = available_memory_gb - quant.memory_gb
                reason = f"Fits in memory with {headroom:.1f} GB headroom"
            else:
                overshoot = quant.memory_gb - available_memory_gb
                reason = f"Needs ~{overshoot:.1f} GB swap — may cause slowdowns"

            recommendations.append(
                Recommendation(
                    model_name=model.name,
                    quantization=quant.quant,
                    format=quant.format,
                    hf_repo=quant.hf_repo,
                    memory_required_gb=quant.memory_gb,
                    quality_score=quant.quality_score,
                    fits_in_memory=fits,
                    swap_warning=swap_warning,
                    reason=reason,
                )
            )

    # Sort: fits_in_memory first, then by quality (with MLX bonus), descending
    def _sort_key(rec: Recommendation) -> tuple[int, float]:
        bonus = 0.5 if prefer_mlx and rec.format == "mlx" else 0.0
        return (
            1 if rec.fits_in_memory else 0,
            rec.quality_score + bonus,
        )

    recommendations.sort(key=_sort_key, reverse=True)
    return recommendations


def calculate_memory_budget(total_memory_gb: float, reserved_gb: float = 4.5) -> float:
    """Calculate memory budget for LLM after reserving OS/app overhead.

    Args:
        total_memory_gb: Total system RAM.
        reserved_gb: Memory reserved for OS and apps (default 4.5 GB).

    Returns:
        Available memory in GB for model loading. Never negative.
    """
    return max(0.0, total_memory_gb - reserved_gb)
