"""Generate benchmark reports in JSON and markdown formats."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from macsmart.benchmark.runner import BenchmarkResult

# Default output directory for benchmark results.
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"


def save_result(result: BenchmarkResult, output_dir: Path | None = None) -> Path:
    """Save a benchmark result to the benchmarks/ directory as JSON.

    Files are named with model name and timestamp for uniqueness.

    Args:
        result: The benchmark result to save.
        output_dir: Optional output directory. Defaults to benchmarks/.

    Returns:
        Path to the saved JSON file.
    """
    out = output_dir or _DEFAULT_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = result.model_name.replace("/", "_").replace(" ", "_")
    filename = f"{safe_name}_{result.quantization}_{timestamp}.json"

    data = asdict(result)
    data["timestamp"] = timestamp

    path = out / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def generate_markdown_report(results: list[BenchmarkResult]) -> str:
    """Generate a markdown comparison table from benchmark results.

    Args:
        results: List of benchmark results to compare.

    Returns:
        Markdown-formatted report string.
    """
    if not results:
        return "No benchmark results to report."

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(
        "| Model | Quant | TTFT (ms) | Tokens/s | Peak Mem (GB) | "
        "Swap (GB) | Gen Tokens | Duration (s) |"
    )
    lines.append(
        "|-------|-------|----------:|----------:|--------------:|"
        "----------:|-----------:|-------------:|"
    )

    for r in results:
        lines.append(
            f"| {r.model_name} | {r.quantization} | "
            f"{r.ttft_ms:.1f} | {r.tokens_per_sec:.1f} | "
            f"{r.peak_memory_gb:.2f} | {r.swap_used_gb:.2f} | "
            f"{r.generation_tokens} | {r.duration_sec:.1f} |"
        )

    lines.append("")
    return "\n".join(lines)


def load_results(results_dir: Path | None = None) -> list[BenchmarkResult]:
    """Load all stored benchmark results from the benchmarks/ directory.

    Args:
        results_dir: Optional directory to load from. Defaults to benchmarks/.

    Returns:
        List of previously saved BenchmarkResult objects, sorted by
        timestamp (newest first).
    """
    directory = results_dir or _DEFAULT_RESULTS_DIR
    if not directory.exists():
        return []

    results: list[tuple[str, BenchmarkResult]] = []
    for path in directory.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        timestamp = data.pop("timestamp", "")
        try:
            result = BenchmarkResult(**data)
            results.append((timestamp, result))
        except TypeError:
            continue

    # Sort newest first
    results.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in results]
