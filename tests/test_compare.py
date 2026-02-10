"""Tests for the compare command and supporting helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from macsmart.benchmark.report import (
    get_latest_result_paths,
    load_result_from_file,
    save_result,
)
from macsmart.benchmark.runner import BenchmarkResult
from macsmart.cli import cli


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
# load_result_from_file tests
# ---------------------------------------------------------------------------


class TestLoadResultFromFile:
    """Tests for load_result_from_file."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        """Test loading a valid benchmark result JSON."""
        result = _make_result(model_name="MyModel")
        path = save_result(result, output_dir=tmp_path)
        loaded = load_result_from_file(path)
        assert loaded.model_name == "MyModel"
        assert loaded.tokens_per_sec == 35.5

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_result_from_file(tmp_path / "nope.json")

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test that corrupt JSON raises ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_result_from_file(bad)

    def test_load_wrong_schema(self, tmp_path: Path) -> None:
        """Test that valid JSON with wrong fields raises ValueError."""
        wrong = tmp_path / "wrong.json"
        wrong.write_text('{"foo": "bar"}')
        with pytest.raises(ValueError, match="Invalid benchmark result"):
            load_result_from_file(wrong)


# ---------------------------------------------------------------------------
# get_latest_result_paths tests
# ---------------------------------------------------------------------------


class TestGetLatestResultPaths:
    """Tests for get_latest_result_paths."""

    def test_returns_n_most_recent(self, tmp_path: Path) -> None:
        """Test that the N most recent paths are returned."""
        for i in range(3):
            save_result(_make_result(model_name=f"Model{i}"), output_dir=tmp_path)
            time.sleep(0.05)  # Ensure distinct mtime

        paths = get_latest_result_paths(n=2, results_dir=tmp_path)
        assert len(paths) == 2
        # Newest first
        assert paths[0].stat().st_mtime >= paths[1].stat().st_mtime

    def test_raises_if_too_few(self, tmp_path: Path) -> None:
        """Test ValueError when fewer than N results exist."""
        save_result(_make_result(), output_dir=tmp_path)
        with pytest.raises(ValueError, match="Need at least 2"):
            get_latest_result_paths(n=2, results_dir=tmp_path)

    def test_raises_if_dir_missing(self, tmp_path: Path) -> None:
        """Test ValueError when directory does not exist."""
        with pytest.raises(ValueError, match="does not exist"):
            get_latest_result_paths(n=2, results_dir=tmp_path / "nope")

    def test_excludes_energy_files(self, tmp_path: Path) -> None:
        """Test that energy result files are excluded."""
        save_result(_make_result(model_name="A"), output_dir=tmp_path)
        save_result(_make_result(model_name="B"), output_dir=tmp_path)
        # Create a fake energy file
        (tmp_path / "energy_result_20240101.json").write_text("{}")

        paths = get_latest_result_paths(n=2, results_dir=tmp_path)
        assert len(paths) == 2
        assert all("energy" not in p.name.lower() for p in paths)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCompareCLI:
    """Tests for the compare CLI command."""

    def test_help(self) -> None:
        """Test that --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--latest" in result.output
        assert "--results-dir" in result.output
        assert "FILES" in result.output

    def test_two_file_json(self, tmp_path: Path) -> None:
        """Test comparing two files with --json-output."""
        r1 = _make_result(model_name="ModelA", tokens_per_sec=30.0)
        r2 = _make_result(model_name="ModelB", tokens_per_sec=60.0)
        p1 = save_result(r1, output_dir=tmp_path)
        p2 = save_result(r2, output_dir=tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(p1), str(p2), "--json-output"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["baseline"] == "ModelA"
        assert data["comparison"] == "ModelB"
        assert data["tokens_per_sec"]["speedup"] == 2.0

    def test_no_args_fails(self) -> None:
        """Test that no arguments produces a usage error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["compare"])
        assert result.exit_code != 0

    def test_one_arg_fails(self, tmp_path: Path) -> None:
        """Test that a single file produces a usage error."""
        p = save_result(_make_result(), output_dir=tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(p)])
        assert result.exit_code != 0

    def test_latest(self, tmp_path: Path) -> None:
        """Test --latest flag to compare most recent results."""
        r1 = _make_result(model_name="Old")
        r2 = _make_result(model_name="New")
        save_result(r1, output_dir=tmp_path)
        time.sleep(0.05)
        save_result(r2, output_dir=tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "compare", "--latest", "2",
            "--results-dir", str(tmp_path),
            "--json-output",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "baseline" in data
        assert "comparison" in data
