"""Tests for the model recommendation engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from macsmart.recommender.engine import Recommendation, calculate_memory_budget, recommend_models
from macsmart.recommender.models_db import (
    ModelEntry,
    QuantizationVariant,
    get_models_for_task,
    load_registry,
)
from macsmart.recommender.task_router import (
    SUPPORTED_TASKS,
    detect_task_type,
    validate_task,
)


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "macsmart" / "data" / "registry.yaml"


class TestRecommendationEngine:
    """Tests for the core recommendation logic."""

    def test_recommend_models_basic(self) -> None:
        """Test basic model recommendation with plenty of memory."""
        recs = recommend_models(available_memory_gb=10.0)
        assert len(recs) > 0
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_returns_sorted_by_quality(self) -> None:
        """Test that fit-in-memory models are sorted by quality descending."""
        recs = recommend_models(available_memory_gb=10.0)
        fits = [r for r in recs if r.fits_in_memory]
        assert len(fits) >= 2
        for a, b in zip(fits, fits[1:]):
            assert a.quality_score >= b.quality_score

    def test_recommend_fits_before_swap(self) -> None:
        """Test that models fitting in memory rank above swap models."""
        recs = recommend_models(available_memory_gb=5.0)
        if not recs:
            pytest.skip("No recs at 5 GB")
        fits = [r for r in recs if r.fits_in_memory]
        swaps = [r for r in recs if r.swap_warning]
        if fits and swaps:
            last_fit_idx = max(recs.index(r) for r in fits)
            first_swap_idx = min(recs.index(r) for r in swaps)
            assert last_fit_idx < first_swap_idx

    def test_recommend_filters_by_memory(self) -> None:
        """Test that recommendations exclude models too large even with swap."""
        recs = recommend_models(available_memory_gb=3.0)
        # With 3 GB + 2 GB swap headroom = 5 GB max
        for r in recs:
            assert r.memory_required_gb <= 5.0

    def test_recommend_with_task_filter(self) -> None:
        """Test recommendation filtered by task type."""
        recs_coding = recommend_models(available_memory_gb=10.0, task="coding")
        recs_all = recommend_models(available_memory_gb=10.0)
        # Coding filter should return fewer or equal results
        assert len(recs_coding) <= len(recs_all)
        assert len(recs_coding) > 0

    def test_recommend_with_invalid_task(self) -> None:
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            recommend_models(available_memory_gb=10.0, task="dancing")

    def test_recommend_zero_memory(self) -> None:
        """Test recommendation with no available memory returns empty."""
        recs = recommend_models(available_memory_gb=0.0)
        # Only models <= 2.0 GB (swap headroom) should appear
        for r in recs:
            assert r.memory_required_gb <= 2.0

    def test_recommend_swap_warning_set_correctly(self) -> None:
        """Test that swap_warning is True only when model exceeds available."""
        recs = recommend_models(available_memory_gb=5.0)
        for r in recs:
            if r.memory_required_gb <= 5.0:
                assert r.fits_in_memory is True
                assert r.swap_warning is False
            else:
                assert r.fits_in_memory is False
                assert r.swap_warning is True

    def test_recommend_reason_has_headroom_info(self) -> None:
        """Test that the reason string contains useful info."""
        recs = recommend_models(available_memory_gb=10.0)
        for r in recs:
            if r.fits_in_memory:
                assert "headroom" in r.reason.lower()
            else:
                assert "swap" in r.reason.lower()

    def test_memory_budget_calculation(self) -> None:
        """Test memory budget calculation with reserved overhead."""
        assert calculate_memory_budget(16.0) == 11.5
        assert calculate_memory_budget(16.0, reserved_gb=5.0) == 11.0
        assert calculate_memory_budget(8.0, reserved_gb=4.5) == 3.5

    def test_memory_budget_never_negative(self) -> None:
        """Test that budget never goes below zero."""
        assert calculate_memory_budget(2.0, reserved_gb=10.0) == 0.0
        assert calculate_memory_budget(0.0) == 0.0


class TestModelsDB:
    """Tests for the model registry."""

    def test_load_registry(self) -> None:
        """Test loading the model registry from YAML."""
        models = load_registry(REGISTRY_PATH)
        assert len(models) >= 5
        for m in models:
            assert isinstance(m, ModelEntry)
            assert m.name != ""
            assert m.family != ""
            assert len(m.quantizations) > 0
            assert len(m.tasks) > 0

    def test_load_registry_default_path(self) -> None:
        """Test loading with default path resolution."""
        models = load_registry()
        assert len(models) >= 5

    def test_load_registry_invalid_path(self) -> None:
        """Test that loading a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_registry(Path("/nonexistent/registry.yaml"))

    def test_registry_quantization_fields(self) -> None:
        """Test that all quantization entries have required fields."""
        models = load_registry(REGISTRY_PATH)
        for m in models:
            for q in m.quantizations:
                assert isinstance(q, QuantizationVariant)
                assert q.quant in ("3bit", "4bit", "8bit", "bf16")
                assert q.format != ""
                assert q.hf_repo != ""
                assert q.memory_gb > 0
                assert 0 <= q.quality_score <= 100

    def test_filter_by_task(self) -> None:
        """Test filtering models by task type."""
        models = load_registry(REGISTRY_PATH)
        coding_models = get_models_for_task("coding", models)
        assert len(coding_models) > 0
        for m in coding_models:
            assert "coding" in m.tasks

    def test_filter_by_task_case_insensitive(self) -> None:
        """Test that task filtering is case-insensitive."""
        models = load_registry(REGISTRY_PATH)
        assert get_models_for_task("Coding", models) == get_models_for_task("coding", models)

    def test_filter_by_nonexistent_task(self) -> None:
        """Test that filtering by unknown task returns empty list."""
        models = load_registry(REGISTRY_PATH)
        result = get_models_for_task("underwater_basket_weaving", models)
        assert result == []

    def test_registry_has_expected_families(self) -> None:
        """Test that the registry contains all 9 expected model families."""
        models = load_registry(REGISTRY_PATH)
        families = {m.family for m in models}
        expected = {
            "qwen", "mistral", "llama", "phi", "gemma",
            "deepseek", "codellama", "starcoder", "smollm",
        }
        assert families == expected


class TestTaskRouter:
    """Tests for task type detection."""

    def test_detect_coding_task(self) -> None:
        """Test detection of coding-related prompts."""
        assert detect_task_type("Write a Python function to sort a list") == "coding"
        assert detect_task_type("Debug this JavaScript code") == "coding"
        assert detect_task_type("Implement a binary search algorithm") == "coding"

    def test_detect_writing_task(self) -> None:
        """Test detection of writing-related prompts."""
        assert detect_task_type("Write an essay about climate change") == "writing"
        assert detect_task_type("Draft a blog post about AI") == "writing"

    def test_detect_summarization_task(self) -> None:
        """Test detection of summarization prompts."""
        assert detect_task_type("Summarize this article for me") == "summarization"
        assert detect_task_type("Give me the TL;DR") == "summarization"

    def test_detect_chat_task(self) -> None:
        """Test detection of chat/conversation prompts."""
        assert detect_task_type("Hello, how are you?") == "chat"
        assert detect_task_type("Hey, let's chat") == "chat"

    def test_detect_general_fallback(self) -> None:
        """Test that unrecognized prompts fall back to general."""
        assert detect_task_type("What is the capital of France?") == "general"
        assert detect_task_type("Explain quantum mechanics") == "general"

    def test_validate_task(self) -> None:
        """Test task type validation."""
        for task in SUPPORTED_TASKS:
            assert validate_task(task) == task

    def test_validate_task_case_insensitive(self) -> None:
        """Test that validation is case-insensitive."""
        assert validate_task("Coding") == "coding"
        assert validate_task("WRITING") == "writing"
        assert validate_task("  Chat  ") == "chat"

    def test_validate_task_invalid(self) -> None:
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            validate_task("dancing")
