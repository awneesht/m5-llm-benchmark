"""Tests for the benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from macsmart.benchmark.energy import (
    EnergyMeasurement,
    _parse_powermetrics_output,
    is_powermetrics_available,
    measure_energy,
)
from macsmart.benchmark.report import (
    generate_markdown_report,
    load_results,
    save_result,
)
from macsmart.benchmark.runner import (
    BenchmarkResult,
    _detect_quantization,
    compare_results,
    run_benchmark,
)


def _make_result(**overrides) -> BenchmarkResult:
    """Helper to create a BenchmarkResult with sensible defaults."""
    defaults = {
        "model_name": "TestModel-7B-4bit",
        "quantization": "4bit",
        "ttft_ms": 150.0,
        "tokens_per_sec": 35.5,
        "peak_memory_gb": 4.2,
        "swap_used_gb": 0.0,
        "total_tokens": 280,
        "prompt_tokens": 24,
        "generation_tokens": 256,
        "duration_sec": 7.2,
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    """Tests for inference benchmarking."""

    def test_run_benchmark_mocked(self) -> None:
        """Test run_benchmark with a mocked runtime layer."""
        mock_gen_result = MagicMock()
        mock_gen_result.ttft_ms = 120.0
        mock_gen_result.tokens_per_sec = 40.0
        mock_gen_result.peak_memory_gb = 4.5
        mock_gen_result.prompt_tokens = 20
        mock_gen_result.generation_tokens = 100

        with (
            patch("macsmart.benchmark.runner.load_model", return_value=("mock_model", "mock_tok")),
            patch("macsmart.benchmark.runner.generate", return_value=mock_gen_result),
            patch("macsmart.benchmark.runner.unload_model"),
        ):
            result = run_benchmark("mlx-community/TestModel-4bit", max_tokens=100)

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "TestModel-4bit"
        assert result.quantization == "4bit"
        assert result.ttft_ms == 120.0
        assert result.tokens_per_sec == 40.0
        assert result.peak_memory_gb == 4.5
        assert result.prompt_tokens == 20
        assert result.generation_tokens == 100
        assert result.total_tokens == 120
        assert result.duration_sec >= 0
        assert result.swap_used_gb >= 0

    def test_detect_quantization(self) -> None:
        """Test quantization detection from repo name."""
        assert _detect_quantization("mlx-community/Model-4bit") == "4bit"
        assert _detect_quantization("mlx-community/Model-8bit") == "8bit"
        assert _detect_quantization("mlx-community/Model-3bit") == "3bit"
        assert _detect_quantization("mlx-community/Model-bf16") == "bf16"
        assert _detect_quantization("mlx-community/Model") == "unknown"

    def test_compare_results(self) -> None:
        """Test comparison between two benchmark results."""
        result_a = _make_result(
            model_name="ModelA",
            tokens_per_sec=30.0,
            ttft_ms=200.0,
            peak_memory_gb=4.0,
            swap_used_gb=0.0,
            duration_sec=10.0,
        )
        result_b = _make_result(
            model_name="ModelB",
            tokens_per_sec=60.0,
            ttft_ms=100.0,
            peak_memory_gb=5.0,
            swap_used_gb=0.5,
            duration_sec=5.0,
        )

        comp = compare_results(result_a, result_b)

        assert comp["baseline"] == "ModelA"
        assert comp["comparison"] == "ModelB"
        assert comp["tokens_per_sec"]["speedup"] == 2.0
        assert comp["ttft_ms"]["delta_ms"] == -100.0
        assert comp["peak_memory_gb"]["delta_gb"] == 1.0
        assert comp["swap_used_gb"]["delta_gb"] == 0.5
        assert comp["duration_sec"]["speedup"] == 0.5

    def test_compare_results_zero_baseline(self) -> None:
        """Test comparison handles zero values gracefully."""
        result_a = _make_result(tokens_per_sec=0.0, duration_sec=0.0)
        result_b = _make_result(tokens_per_sec=50.0, duration_sec=5.0)

        comp = compare_results(result_a, result_b)
        assert comp["tokens_per_sec"]["speedup"] is None
        assert comp["duration_sec"]["speedup"] is None


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """Tests for report generation."""

    def test_save_result(self, tmp_path: Path) -> None:
        """Test saving benchmark result to JSON."""
        result = _make_result()
        path = save_result(result, output_dir=tmp_path)

        assert path.exists()
        assert path.suffix == ".json"
        assert "TestModel-7B-4bit" in path.name
        assert "4bit" in path.name

        with open(path) as f:
            data = json.load(f)

        assert data["model_name"] == "TestModel-7B-4bit"
        assert data["tokens_per_sec"] == 35.5
        assert "timestamp" in data

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Test that save_result creates the output directory if needed."""
        nested = tmp_path / "deep" / "nested"
        result = _make_result()
        path = save_result(result, output_dir=nested)
        assert path.exists()
        assert nested.exists()

    def test_load_results(self, tmp_path: Path) -> None:
        """Test loading stored benchmark results."""
        # Save two results
        r1 = _make_result(model_name="Model1", tokens_per_sec=30.0)
        r2 = _make_result(model_name="Model2", tokens_per_sec=50.0)
        save_result(r1, output_dir=tmp_path)
        save_result(r2, output_dir=tmp_path)

        loaded = load_results(results_dir=tmp_path)
        assert len(loaded) == 2
        names = {r.model_name for r in loaded}
        assert names == {"Model1", "Model2"}

    def test_load_results_empty_dir(self, tmp_path: Path) -> None:
        """Test loading from an empty directory."""
        loaded = load_results(results_dir=tmp_path)
        assert loaded == []

    def test_load_results_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test loading from a nonexistent directory."""
        loaded = load_results(results_dir=tmp_path / "nonexistent")
        assert loaded == []

    def test_load_results_skips_corrupt_json(self, tmp_path: Path) -> None:
        """Test that corrupt JSON files are skipped."""
        # Write a valid result
        save_result(_make_result(), output_dir=tmp_path)
        # Write a corrupt file
        (tmp_path / "corrupt.json").write_text("{invalid json")

        loaded = load_results(results_dir=tmp_path)
        assert len(loaded) == 1

    def test_generate_markdown_report(self) -> None:
        """Test markdown report generation."""
        results = [
            _make_result(model_name="ModelA", quantization="4bit"),
            _make_result(model_name="ModelB", quantization="8bit"),
        ]
        md = generate_markdown_report(results)

        assert "# Benchmark Report" in md
        assert "ModelA" in md
        assert "ModelB" in md
        assert "4bit" in md
        assert "8bit" in md
        assert "| Model |" in md

    def test_generate_markdown_report_empty(self) -> None:
        """Test markdown report with no results."""
        md = generate_markdown_report([])
        assert "No benchmark results" in md


# ---------------------------------------------------------------------------
# Energy tests
# ---------------------------------------------------------------------------


class TestEnergyMeasurement:
    """Tests for energy measurement."""

    def test_is_powermetrics_available(self) -> None:
        """Test powermetrics availability check returns boolean."""
        result = is_powermetrics_available()
        assert isinstance(result, bool)

    def test_is_powermetrics_unavailable_on_non_darwin(self) -> None:
        """Test that powermetrics is unavailable on non-macOS."""
        with patch("macsmart.benchmark.energy.platform.system", return_value="Linux"):
            assert is_powermetrics_available() is False

    def test_parse_powermetrics_output(self) -> None:
        """Test parsing powermetrics power readings."""
        sample = (
            "CPU Power: 1500 mW\n"
            "GPU Power: 800 mW\n"
            "ANE Power: 200 mW\n"
            "Combined Power (CPU + GPU + ANE): 2500 mW\n"
        )
        powers = _parse_powermetrics_output(sample)
        assert abs(powers["cpu"] - 1.5) < 0.01
        assert abs(powers["gpu"] - 0.8) < 0.01
        assert abs(powers["ane"] - 0.2) < 0.01
        assert abs(powers["combined"] - 2.5) < 0.01

    def test_parse_powermetrics_empty(self) -> None:
        """Test parsing empty output returns empty dict."""
        assert _parse_powermetrics_output("") == {}
        assert _parse_powermetrics_output("no power data here\n") == {}

    def test_measure_energy_unavailable(self) -> None:
        """Test measure_energy when powermetrics is unavailable."""
        with patch("macsmart.benchmark.energy.is_powermetrics_available", return_value=False):
            result = measure_energy(duration_sec=1.0)

        assert isinstance(result, EnergyMeasurement)
        assert result.cpu_power_watts is None
        assert result.gpu_power_watts is None
        assert result.total_energy_joules is None
        assert result.duration_sec == 1.0

    def test_measure_energy_with_mock_powermetrics(self) -> None:
        """Test measure_energy with mocked powermetrics output."""
        mock_output = (
            "CPU Power: 2000 mW\n"
            "GPU Power: 1000 mW\n"
            "ANE Power: 500 mW\n"
        )
        mock_result = MagicMock()
        mock_result.stdout = mock_output
        mock_result.returncode = 0

        with (
            patch("macsmart.benchmark.energy.is_powermetrics_available", return_value=True),
            patch("macsmart.benchmark.energy.subprocess.run", return_value=mock_result),
        ):
            result = measure_energy(duration_sec=5.0)

        assert result.cpu_power_watts == 2.0
        assert result.gpu_power_watts == 1.0
        assert result.ane_power_watts == 0.5
        assert result.total_power_watts == 3.5  # sum of all
        assert result.total_energy_joules == 17.5  # 3.5W * 5s
        assert result.duration_sec == 5.0
