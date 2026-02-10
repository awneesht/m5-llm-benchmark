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


def generate_energy_comparison_report(comparison: object) -> str:
    """Generate a markdown report comparing battery vs AC energy usage.

    Args:
        comparison: EnergyComparison object.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = []
    lines.append("# Energy Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"**Model:** {comparison.battery.benchmark.model_name}")
    lines.append("")

    lines.append("| Metric | Battery | AC | Delta |")
    lines.append("|--------|--------:|---:|------:|")

    b = comparison.battery.benchmark
    a = comparison.ac.benchmark

    lines.append(
        f"| Tokens/s | {b.tokens_per_sec:.1f} | {a.tokens_per_sec:.1f} | "
        f"{a.tokens_per_sec - b.tokens_per_sec:+.1f} |"
    )
    lines.append(
        f"| TTFT (ms) | {b.ttft_ms:.1f} | {a.ttft_ms:.1f} | "
        f"{a.ttft_ms - b.ttft_ms:+.1f} |"
    )
    lines.append(
        f"| Peak Memory (GB) | {b.peak_memory_gb:.2f} | {a.peak_memory_gb:.2f} | "
        f"{a.peak_memory_gb - b.peak_memory_gb:+.2f} |"
    )

    be = comparison.battery.energy
    ae = comparison.ac.energy
    bw = be.total_power_watts if be.total_power_watts is not None else 0.0
    aw = ae.total_power_watts if ae.total_power_watts is not None else 0.0
    lines.append(
        f"| Power (W) | {bw:.1f} | {aw:.1f} | {aw - bw:+.1f} |"
    )

    bj = be.total_energy_joules if be.total_energy_joules is not None else 0.0
    aj = ae.total_energy_joules if ae.total_energy_joules is not None else 0.0
    lines.append(
        f"| Energy (J) | {bj:.1f} | {aj:.1f} | {aj - bj:+.1f} |"
    )

    lines.append("")

    if comparison.efficiency_battery is not None:
        lines.append(f"**Battery efficiency:** {comparison.efficiency_battery:.3f} tokens/J")
    if comparison.efficiency_ac is not None:
        lines.append(f"**AC efficiency:** {comparison.efficiency_ac:.3f} tokens/J")
    if comparison.speed_ratio is not None:
        lines.append(f"**Speed ratio (AC/Battery):** {comparison.speed_ratio:.3f}x")

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


def results_to_json_api(results_dir: Path | None = None) -> list[dict]:
    """Serialize benchmark results for the dashboard JSON API.

    Args:
        results_dir: Directory to load results from.

    Returns:
        List of result dicts ready for JSON serialization.
    """
    results = load_results(results_dir=results_dir)
    return [asdict(r) for r in results]
