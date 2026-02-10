"""Tests for energy benchmark comparison."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from macsmart.benchmark.energy import EnergyMeasurement
from macsmart.benchmark.energy_compare import (
    EnergyBenchmarkResult,
    EnergyComparison,
    compare_energy,
    load_energy_results,
    run_energy_benchmark,
    save_energy_result,
)
from macsmart.benchmark.report import generate_energy_comparison_report
from macsmart.benchmark.runner import BenchmarkResult


def _make_benchmark(**overrides) -> BenchmarkResult:
    defaults = {
        "model_name": "TestModel-4bit",
        "quantization": "4bit",
        "ttft_ms": 150.0,
        "tokens_per_sec": 35.0,
        "peak_memory_gb": 4.0,
        "swap_used_gb": 0.0,
        "total_tokens": 280,
        "prompt_tokens": 24,
        "generation_tokens": 256,
        "duration_sec": 7.0,
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def _make_energy(**overrides) -> EnergyMeasurement:
    defaults = {
        "cpu_power_watts": 2.0,
        "gpu_power_watts": 1.0,
        "ane_power_watts": 0.5,
        "total_power_watts": 3.5,
        "duration_sec": 7.0,
        "total_energy_joules": 24.5,
    }
    defaults.update(overrides)
    return EnergyMeasurement(**defaults)


def _make_energy_result(power_source: str = "battery", **bm_overrides) -> EnergyBenchmarkResult:
    return EnergyBenchmarkResult(
        benchmark=_make_benchmark(**bm_overrides),
        energy=_make_energy(),
        power_source=power_source,
    )


# ---------------------------------------------------------------------------
# EnergyBenchmarkResult tests
# ---------------------------------------------------------------------------


class TestEnergyBenchmarkResult:
    """Tests for EnergyBenchmarkResult dataclass."""

    def test_create(self) -> None:
        result = _make_energy_result()
        assert result.power_source == "battery"
        assert result.benchmark.model_name == "TestModel-4bit"
        assert result.energy.total_power_watts == 3.5

    def test_ac_source(self) -> None:
        result = _make_energy_result(power_source="ac")
        assert result.power_source == "ac"


# ---------------------------------------------------------------------------
# compare_energy tests
# ---------------------------------------------------------------------------


class TestCompareEnergy:
    """Tests for energy comparison logic."""

    def test_basic_comparison(self) -> None:
        battery = _make_energy_result("battery", tokens_per_sec=30.0, ttft_ms=200.0)
        ac = _make_energy_result("ac", tokens_per_sec=40.0, ttft_ms=150.0)

        comp = compare_energy(battery, ac)

        assert isinstance(comp, EnergyComparison)
        assert comp.speed_ratio == round(40.0 / 30.0, 3)
        assert comp.ttft_delta_ms == -50.0

    def test_energy_ratio(self) -> None:
        battery = EnergyBenchmarkResult(
            benchmark=_make_benchmark(),
            energy=_make_energy(total_energy_joules=20.0),
            power_source="battery",
        )
        ac = EnergyBenchmarkResult(
            benchmark=_make_benchmark(),
            energy=_make_energy(total_energy_joules=30.0),
            power_source="ac",
        )
        comp = compare_energy(battery, ac)
        assert comp.energy_ratio == 1.5

    def test_efficiency_calculation(self) -> None:
        battery = EnergyBenchmarkResult(
            benchmark=_make_benchmark(generation_tokens=100),
            energy=_make_energy(total_energy_joules=10.0),
            power_source="battery",
        )
        ac = EnergyBenchmarkResult(
            benchmark=_make_benchmark(generation_tokens=100),
            energy=_make_energy(total_energy_joules=20.0),
            power_source="ac",
        )
        comp = compare_energy(battery, ac)
        assert comp.efficiency_battery == 10.0  # 100 tokens / 10 J
        assert comp.efficiency_ac == 5.0  # 100 tokens / 20 J

    def test_zero_baseline_speed(self) -> None:
        battery = _make_energy_result("battery", tokens_per_sec=0.0)
        ac = _make_energy_result("ac", tokens_per_sec=40.0)
        comp = compare_energy(battery, ac)
        assert comp.speed_ratio is None

    def test_none_energy(self) -> None:
        battery = EnergyBenchmarkResult(
            benchmark=_make_benchmark(),
            energy=EnergyMeasurement(
                cpu_power_watts=None, gpu_power_watts=None, ane_power_watts=None,
                total_power_watts=None, duration_sec=7.0, total_energy_joules=None,
            ),
            power_source="battery",
        )
        ac = _make_energy_result("ac")
        comp = compare_energy(battery, ac)
        assert comp.energy_ratio is None
        assert comp.efficiency_battery is None


# ---------------------------------------------------------------------------
# run_energy_benchmark tests
# ---------------------------------------------------------------------------


class TestRunEnergyBenchmark:
    """Tests for running energy benchmarks."""

    @patch("macsmart.benchmark.energy_compare.measure_energy")
    @patch("macsmart.benchmark.energy_compare.run_benchmark")
    def test_run_energy_benchmark(self, mock_bench, mock_energy) -> None:
        mock_bench.return_value = _make_benchmark()
        mock_energy.return_value = _make_energy()

        result = run_energy_benchmark("test-model", power_source="battery")

        assert isinstance(result, EnergyBenchmarkResult)
        assert result.power_source == "battery"
        mock_bench.assert_called_once()
        mock_energy.assert_called_once()

    @patch("macsmart.benchmark.energy_compare.measure_energy")
    @patch("macsmart.benchmark.energy_compare.run_benchmark")
    def test_run_energy_benchmark_ac(self, mock_bench, mock_energy) -> None:
        mock_bench.return_value = _make_benchmark()
        mock_energy.return_value = _make_energy()

        result = run_energy_benchmark("test-model", power_source="ac")
        assert result.power_source == "ac"


# ---------------------------------------------------------------------------
# save / load tests
# ---------------------------------------------------------------------------


class TestEnergySaveLoad:
    """Tests for energy result serialization."""

    def test_save_energy_result(self, tmp_path: Path) -> None:
        result = _make_energy_result()
        path = save_energy_result(result, output_dir=tmp_path)
        assert path.exists()
        assert "energy_battery_" in path.name
        with open(path) as f:
            data = json.load(f)
        assert data["power_source"] == "battery"
        assert data["benchmark"]["model_name"] == "TestModel-4bit"

    def test_save_ac_result(self, tmp_path: Path) -> None:
        result = _make_energy_result(power_source="ac")
        path = save_energy_result(result, output_dir=tmp_path)
        assert "energy_ac_" in path.name

    def test_load_energy_results(self, tmp_path: Path) -> None:
        r1 = _make_energy_result("battery")
        r2 = _make_energy_result("ac")
        save_energy_result(r1, output_dir=tmp_path)
        save_energy_result(r2, output_dir=tmp_path)

        loaded = load_energy_results(results_dir=tmp_path)
        assert len(loaded) == 2
        sources = {r.power_source for r in loaded}
        assert sources == {"battery", "ac"}

    def test_load_empty_dir(self, tmp_path: Path) -> None:
        loaded = load_energy_results(results_dir=tmp_path)
        assert loaded == []

    def test_load_nonexistent_dir(self, tmp_path: Path) -> None:
        loaded = load_energy_results(results_dir=tmp_path / "nope")
        assert loaded == []

    def test_load_skips_corrupt(self, tmp_path: Path) -> None:
        save_energy_result(_make_energy_result(), output_dir=tmp_path)
        (tmp_path / "energy_battery_corrupt.json").write_text("{bad json")
        loaded = load_energy_results(results_dir=tmp_path)
        assert len(loaded) == 1

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        path = save_energy_result(_make_energy_result(), output_dir=nested)
        assert path.exists()


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestEnergyComparisonReport:
    """Tests for energy comparison markdown report."""

    def test_report_generation(self) -> None:
        battery = _make_energy_result("battery", tokens_per_sec=30.0, ttft_ms=200.0)
        ac = _make_energy_result("ac", tokens_per_sec=40.0, ttft_ms=150.0)
        comp = compare_energy(battery, ac)
        md = generate_energy_comparison_report(comp)

        assert "# Energy Comparison Report" in md
        assert "TestModel-4bit" in md
        assert "Battery" in md
        assert "AC" in md
        assert "Tokens/s" in md
        assert "Speed ratio" in md

    def test_report_with_none_energy(self) -> None:
        battery = EnergyBenchmarkResult(
            benchmark=_make_benchmark(),
            energy=EnergyMeasurement(
                cpu_power_watts=None, gpu_power_watts=None, ane_power_watts=None,
                total_power_watts=None, duration_sec=7.0, total_energy_joules=None,
            ),
            power_source="battery",
        )
        ac = _make_energy_result("ac")
        comp = compare_energy(battery, ac)
        md = generate_energy_comparison_report(comp)
        assert "# Energy Comparison Report" in md


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestEnergyCompareCLI:
    """Tests for energy-compare CLI command."""

    def test_energy_compare_help(self) -> None:
        from macsmart.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["energy-compare", "--help"])
        assert result.exit_code == 0
        assert "MODEL" in result.output
        assert "--save" in result.output

    def test_energy_compare_appears_in_help(self) -> None:
        from macsmart.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "energy-compare" in result.output

    def test_benchmark_energy_flag_help(self) -> None:
        from macsmart.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert "--energy" in result.output
