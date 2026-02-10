"""Energy benchmark comparison between battery and AC power."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from macsmart.benchmark.energy import EnergyMeasurement, measure_energy
from macsmart.benchmark.runner import BenchmarkResult, run_benchmark

_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"


@dataclass
class EnergyBenchmarkResult:
    """Benchmark result combined with energy measurement and power source."""

    benchmark: BenchmarkResult
    energy: EnergyMeasurement
    power_source: str  # "battery" or "ac"


@dataclass
class EnergyComparison:
    """Side-by-side comparison of battery vs AC energy benchmarks."""

    battery: EnergyBenchmarkResult
    ac: EnergyBenchmarkResult
    speed_ratio: float | None  # ac.tokens_per_sec / battery.tokens_per_sec
    ttft_delta_ms: float  # ac.ttft - battery.ttft
    energy_ratio: float | None  # ac.total_energy / battery.total_energy
    efficiency_battery: float | None  # tokens / joule on battery
    efficiency_ac: float | None  # tokens / joule on AC


def run_energy_benchmark(
    model_repo: str,
    power_source: str,
    prompt: str | None = None,
    max_tokens: int = 256,
    energy_duration_sec: float = 10.0,
) -> EnergyBenchmarkResult:
    """Run a benchmark with energy measurement.

    Args:
        model_repo: HuggingFace repo ID.
        power_source: "battery" or "ac".
        prompt: Custom prompt.
        max_tokens: Maximum tokens to generate.
        energy_duration_sec: Duration for energy sampling.

    Returns:
        Combined benchmark + energy result.
    """
    benchmark = run_benchmark(model_repo, prompt=prompt, max_tokens=max_tokens)
    energy = measure_energy(duration_sec=energy_duration_sec)

    return EnergyBenchmarkResult(
        benchmark=benchmark,
        energy=energy,
        power_source=power_source,
    )


def compare_energy(
    battery: EnergyBenchmarkResult,
    ac: EnergyBenchmarkResult,
) -> EnergyComparison:
    """Compare battery vs AC energy benchmark results.

    Args:
        battery: Benchmark result on battery power.
        ac: Benchmark result on AC power.

    Returns:
        EnergyComparison with computed ratios and deltas.
    """
    batt_tps = battery.benchmark.tokens_per_sec
    ac_tps = ac.benchmark.tokens_per_sec

    speed_ratio = round(ac_tps / batt_tps, 3) if batt_tps > 0 else None
    ttft_delta = round(ac.benchmark.ttft_ms - battery.benchmark.ttft_ms, 2)

    batt_energy = battery.energy.total_energy_joules
    ac_energy = ac.energy.total_energy_joules

    energy_ratio: float | None = None
    if batt_energy and batt_energy > 0 and ac_energy is not None:
        energy_ratio = round(ac_energy / batt_energy, 3)

    eff_battery: float | None = None
    if batt_energy and batt_energy > 0:
        eff_battery = round(battery.benchmark.generation_tokens / batt_energy, 3)

    eff_ac: float | None = None
    if ac_energy and ac_energy > 0:
        eff_ac = round(ac.benchmark.generation_tokens / ac_energy, 3)

    return EnergyComparison(
        battery=battery,
        ac=ac,
        speed_ratio=speed_ratio,
        ttft_delta_ms=ttft_delta,
        energy_ratio=energy_ratio,
        efficiency_battery=eff_battery,
        efficiency_ac=eff_ac,
    )


def save_energy_result(
    result: EnergyBenchmarkResult,
    output_dir: Path | None = None,
) -> Path:
    """Save an energy benchmark result to JSON.

    Args:
        result: The energy benchmark result.
        output_dir: Output directory. Defaults to benchmarks/.

    Returns:
        Path to the saved JSON file.
    """
    out = output_dir or _DEFAULT_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = result.benchmark.model_name.replace("/", "_").replace(" ", "_")
    filename = f"energy_{result.power_source}_{safe_name}_{timestamp}.json"

    data = {
        "benchmark": asdict(result.benchmark),
        "energy": asdict(result.energy),
        "power_source": result.power_source,
        "timestamp": timestamp,
    }

    path = out / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_energy_results(
    results_dir: Path | None = None,
) -> list[EnergyBenchmarkResult]:
    """Load all energy benchmark results from disk.

    Args:
        results_dir: Directory to load from. Defaults to benchmarks/.

    Returns:
        List of EnergyBenchmarkResult objects.
    """
    directory = results_dir or _DEFAULT_RESULTS_DIR
    if not directory.exists():
        return []

    results: list[EnergyBenchmarkResult] = []
    for path in sorted(directory.glob("energy_*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        try:
            data.pop("timestamp", None)
            benchmark = BenchmarkResult(**data["benchmark"])
            energy = EnergyMeasurement(**data["energy"])
            result = EnergyBenchmarkResult(
                benchmark=benchmark,
                energy=energy,
                power_source=data["power_source"],
            )
            results.append(result)
        except (TypeError, KeyError):
            continue

    return results
