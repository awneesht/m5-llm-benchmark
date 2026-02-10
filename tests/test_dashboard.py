"""Tests for the benchmark dashboard server."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from macsmart.benchmark.runner import BenchmarkResult
from macsmart.dashboard.server import (
    DashboardHandler,
    energy_results_to_json_api,
    results_to_json_api,
    start_server,
    stop_server,
)


def _make_result(**overrides) -> BenchmarkResult:
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


def _save_result(result: BenchmarkResult, output_dir: Path) -> None:
    from macsmart.benchmark.report import save_result
    save_result(result, output_dir=output_dir)


def _save_energy_result(output_dir: Path, power_source: str = "battery") -> None:
    from macsmart.benchmark.energy import EnergyMeasurement
    from macsmart.benchmark.energy_compare import EnergyBenchmarkResult, save_energy_result

    result = EnergyBenchmarkResult(
        benchmark=_make_result(),
        energy=EnergyMeasurement(
            cpu_power_watts=2.0, gpu_power_watts=1.0, ane_power_watts=0.5,
            total_power_watts=3.5, duration_sec=7.0, total_energy_joules=24.5,
        ),
        power_source=power_source,
    )
    save_energy_result(result, output_dir=output_dir)


# ---------------------------------------------------------------------------
# API serialization tests
# ---------------------------------------------------------------------------


class TestAPISerializers:
    """Tests for API JSON serialization."""

    def test_results_to_json_api(self, tmp_path: Path) -> None:
        _save_result(_make_result(model_name="A"), tmp_path)
        _save_result(_make_result(model_name="B"), tmp_path)
        data = results_to_json_api(results_dir=tmp_path)
        assert len(data) == 2
        assert all(isinstance(d, dict) for d in data)
        names = {d["model_name"] for d in data}
        assert names == {"A", "B"}

    def test_results_to_json_api_empty(self, tmp_path: Path) -> None:
        data = results_to_json_api(results_dir=tmp_path)
        assert data == []

    def test_energy_results_to_json_api(self, tmp_path: Path) -> None:
        _save_energy_result(tmp_path, "battery")
        _save_energy_result(tmp_path, "ac")
        data = energy_results_to_json_api(results_dir=tmp_path)
        assert len(data) == 2
        sources = {d["power_source"] for d in data}
        assert sources == {"battery", "ac"}

    def test_energy_results_to_json_api_empty(self, tmp_path: Path) -> None:
        data = energy_results_to_json_api(results_dir=tmp_path)
        assert data == []


# ---------------------------------------------------------------------------
# Server tests
# ---------------------------------------------------------------------------

# Use a port range unlikely to conflict
_TEST_PORT = 18377


class TestDashboardServer:
    """Tests for the dashboard HTTP server."""

    def test_start_and_stop(self, tmp_path: Path) -> None:
        server = start_server(port=_TEST_PORT, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT}/"
            resp = urllib.request.urlopen(url, timeout=5)
            assert resp.status == 200
            html = resp.read().decode()
            assert "MacSmart" in html
        finally:
            stop_server()

    def test_api_results_endpoint(self, tmp_path: Path) -> None:
        _save_result(_make_result(model_name="TestA"), tmp_path)
        server = start_server(port=_TEST_PORT + 1, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT + 1}/api/results"
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            assert len(data) == 1
            assert data[0]["model_name"] == "TestA"
        finally:
            stop_server()

    def test_api_energy_results_endpoint(self, tmp_path: Path) -> None:
        _save_energy_result(tmp_path)
        server = start_server(port=_TEST_PORT + 2, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT + 2}/api/energy-results"
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            assert len(data) == 1
            assert data[0]["power_source"] == "battery"
        finally:
            stop_server()

    def test_api_compare_endpoint(self, tmp_path: Path) -> None:
        _save_result(_make_result(model_name="A", tokens_per_sec=30.0), tmp_path)
        _save_result(_make_result(model_name="B", tokens_per_sec=60.0), tmp_path)
        server = start_server(port=_TEST_PORT + 3, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT + 3}/api/compare?a=0&b=1"
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            assert "tokens_per_sec" in data
            assert data["baseline"] in ("A", "B")
        finally:
            stop_server()

    def test_api_compare_missing_params(self, tmp_path: Path) -> None:
        server = start_server(port=_TEST_PORT + 4, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT + 4}/api/compare"
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            assert "error" in data
        finally:
            stop_server()

    def test_api_compare_out_of_range(self, tmp_path: Path) -> None:
        server = start_server(port=_TEST_PORT + 5, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT + 5}/api/compare?a=0&b=99"
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            assert "error" in data
        finally:
            stop_server()

    def test_static_css_served(self, tmp_path: Path) -> None:
        server = start_server(port=_TEST_PORT + 6, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT + 6}/style.css"
            resp = urllib.request.urlopen(url, timeout=5)
            assert resp.status == 200
            css = resp.read().decode()
            assert "--bg:" in css
        finally:
            stop_server()

    def test_static_js_served(self, tmp_path: Path) -> None:
        server = start_server(port=_TEST_PORT + 7, results_dir=tmp_path, open_browser=False)
        try:
            time.sleep(0.2)
            url = f"http://127.0.0.1:{_TEST_PORT + 7}/dashboard.js"
            resp = urllib.request.urlopen(url, timeout=5)
            assert resp.status == 200
            js = resp.read().decode()
            assert "fetchJSON" in js
        finally:
            stop_server()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestDashboardCLI:
    """Tests for the dashboard CLI command."""

    def test_dashboard_help(self) -> None:
        from macsmart.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--no-browser" in result.output
        assert "--results-dir" in result.output

    def test_dashboard_appears_in_help(self) -> None:
        from macsmart.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "dashboard" in result.output
