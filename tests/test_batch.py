"""Tests for the batch benchmarking command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from macsmart.benchmark.runner import BenchmarkResult
from macsmart.cli import cli
from macsmart.ui.terminal import print_batch_summary


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


def _mock_run_benchmark(model_repo, prompt=None, max_tokens=256):
    """Mock run_benchmark that returns different results per model."""
    name = model_repo.rsplit("/", 1)[-1]
    return _make_result(
        model_name=name,
        tokens_per_sec=40.0 if "A" in name else 30.0,
    )


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestBatchCLI:
    """Tests for the batch CLI command."""

    def test_help(self) -> None:
        """Test that --help works and shows expected options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "MODELS" in result.output
        assert "--prompt" in result.output
        assert "--save" in result.output

    def test_json_output(self) -> None:
        """Test batch with --json-output returns valid JSON."""
        runner = CliRunner()
        with (
            patch("macsmart.benchmark.runner.run_benchmark", side_effect=_mock_run_benchmark),
            patch("macsmart.benchmark.report.save_result", return_value="/tmp/fake.json"),
        ):
            result = runner.invoke(cli, [
                "batch", "repo/ModelA-4bit", "repo/ModelB-4bit", "--json-output",
            ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["results"]) == 2
        assert "errors" not in data

    def test_calls_per_model(self) -> None:
        """Test that run_benchmark is called once per model."""
        mock_bench = MagicMock(side_effect=_mock_run_benchmark)
        runner = CliRunner()

        with (
            patch("macsmart.benchmark.runner.run_benchmark", mock_bench),
            patch("macsmart.benchmark.report.save_result", return_value="/tmp/fake.json"),
        ):
            result = runner.invoke(cli, [
                "batch", "repo/A-4bit", "repo/B-4bit", "repo/C-4bit",
                "--json-output",
            ])

        assert result.exit_code == 0
        assert mock_bench.call_count == 3

    def test_no_save(self) -> None:
        """Test that --no-save skips saving results."""
        mock_save = MagicMock()
        runner = CliRunner()

        with (
            patch("macsmart.benchmark.runner.run_benchmark", side_effect=_mock_run_benchmark),
            patch("macsmart.benchmark.report.save_result", mock_save),
        ):
            result = runner.invoke(cli, [
                "batch", "repo/A-4bit", "repo/B-4bit", "--no-save", "--json-output",
            ])

        assert result.exit_code == 0
        mock_save.assert_not_called()

    def test_partial_failure(self) -> None:
        """Test that batch continues when one model fails."""
        def _failing_bench(model_repo, prompt=None, max_tokens=256):
            if "fail" in model_repo.lower():
                raise RuntimeError("Model load failed")
            return _mock_run_benchmark(model_repo, prompt, max_tokens)

        runner = CliRunner()
        with (
            patch("macsmart.benchmark.runner.run_benchmark", side_effect=_failing_bench),
            patch("macsmart.benchmark.report.save_result", return_value="/tmp/fake.json"),
        ):
            result = runner.invoke(cli, [
                "batch", "repo/A-4bit", "repo/FAIL-model", "repo/B-4bit",
                "--json-output",
            ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["results"]) == 2
        assert "errors" in data
        assert "repo/FAIL-model" in data["errors"]

    def test_single_model_fails(self) -> None:
        """Test that a single model produces a usage error."""
        runner = CliRunner()
        with patch("macsmart.benchmark.runner.run_benchmark", side_effect=_mock_run_benchmark):
            result = runner.invoke(cli, ["batch", "repo/only-one"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# UI tests
# ---------------------------------------------------------------------------


class TestBatchSummaryUI:
    """Tests for print_batch_summary."""

    def test_render_without_crash(self) -> None:
        """Test that print_batch_summary renders without exceptions."""
        results = [
            _make_result(model_name="Fast", tokens_per_sec=50.0),
            _make_result(model_name="Slow", tokens_per_sec=20.0),
        ]
        # Should not raise
        print_batch_summary(results)

    def test_empty_list(self) -> None:
        """Test that empty results list renders without crash."""
        print_batch_summary([])
