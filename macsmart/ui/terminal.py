"""Rich terminal UI for MacSmart LLM."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

_PRESSURE_COLORS = {"nominal": "green", "warn": "yellow", "critical": "red"}


def print_system_profile(profile: dict) -> None:
    """Pretty-print a system profile using Rich tables and panels.

    Args:
        profile: System profile dict from profiler.
    """
    chip = profile["chip"]
    mem = profile["memory"]
    thermal = profile["thermal"]
    battery = profile["battery"]

    # -- Chip info --
    chip_table = Table(show_header=False, box=None, padding=(0, 2))
    chip_table.add_column("Key", style="bold cyan")
    chip_table.add_column("Value")
    chip_table.add_row("Chip", chip["name"])
    chip_table.add_row("Family", chip["family"])
    chip_table.add_row("CPU Cores", str(chip["cpu_cores"]))
    chip_table.add_row("GPU Cores", str(chip["gpu_cores"]))

    ne = chip["neural_engine_cores"]
    chip_table.add_row("Neural Engine", f"{ne} cores" if ne else "N/A")

    bw = chip["memory_bandwidth_gbps"]
    chip_table.add_row("Memory Bandwidth", f"{bw} GB/s" if bw else "Unknown")

    ane_status = "[green]Yes[/green]" if chip["has_neural_accelerator"] else "[red]No[/red]"
    chip_table.add_row("Neural Accelerator", ane_status)

    console.print(Panel(chip_table, title="[bold]Chip[/bold]", border_style="blue"))

    # -- Memory info --
    pressure = mem["pressure"]
    pressure_color = _PRESSURE_COLORS.get(pressure, "white")

    mem_table = Table(show_header=False, box=None, padding=(0, 2))
    mem_table.add_column("Key", style="bold cyan")
    mem_table.add_column("Value")
    mem_table.add_row("Total", f"{mem['total_gb']} GB")
    mem_table.add_row("Available", f"{mem['available_gb']} GB")
    mem_table.add_row("Used", f"{mem['used_gb']} GB")
    mem_table.add_row("Swap Used", f"{mem['swap_used_gb']} GB")
    mem_table.add_row("Pressure", f"[{pressure_color}]{pressure}[/{pressure_color}]")

    console.print(Panel(mem_table, title="[bold]Memory[/bold]", border_style="green"))

    # -- Thermal info --
    level = thermal["level"]
    level_colors = {
        "nominal": "green",
        "moderate": "yellow",
        "heavy": "red",
        "trapping": "bold red",
        "sleeping": "bold red",
    }
    level_color = level_colors.get(level, "white")

    thermal_table = Table(show_header=False, box=None, padding=(0, 2))
    thermal_table.add_column("Key", style="bold cyan")
    thermal_table.add_column("Value")
    thermal_table.add_row("Thermal Level", f"[{level_color}]{level}[/{level_color}]")

    cpu_t = thermal["cpu_temp_celsius"]
    thermal_table.add_row("CPU Temp", f"{cpu_t} C" if cpu_t is not None else "N/A (needs sudo)")

    gpu_t = thermal["gpu_temp_celsius"]
    thermal_table.add_row("GPU Temp", f"{gpu_t} C" if gpu_t is not None else "N/A (needs sudo)")

    console.print(Panel(thermal_table, title="[bold]Thermal[/bold]", border_style="yellow"))

    # -- Battery info --
    batt_table = Table(show_header=False, box=None, padding=(0, 2))
    batt_table.add_column("Key", style="bold cyan")
    batt_table.add_column("Value")
    batt_table.add_row("Charge", f"{battery['percent']:.0f}%")

    plugged = "[green]Yes[/green]" if battery["is_plugged_in"] else "[red]No[/red]"
    batt_table.add_row("Plugged In", plugged)

    charging = "[green]Yes[/green]" if battery["is_charging"] else "No"
    batt_table.add_row("Charging", charging)

    remaining = battery["time_remaining_minutes"]
    if remaining is not None:
        hours, mins = divmod(remaining, 60)
        batt_table.add_row("Time Remaining", f"{hours}h {mins}m")
    else:
        batt_table.add_row("Time Remaining", "N/A")

    console.print(Panel(batt_table, title="[bold]Battery[/bold]", border_style="magenta"))


def print_recommendations(
    recommendations: list,
    available_memory_gb: float | None = None,
) -> None:
    """Pretty-print model recommendations as a ranked table.

    Args:
        recommendations: List of Recommendation objects.
        available_memory_gb: Current available memory (shown in header).
    """
    if available_memory_gb is not None:
        console.print(
            f"\n[bold]Available memory:[/bold] {available_memory_gb:.1f} GB\n"
        )

    table = Table(title="Model Recommendations", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="cyan")
    table.add_column("Quant", justify="center")
    table.add_column("Format", justify="center")
    table.add_column("Memory", justify="right")
    table.add_column("Quality", justify="center")
    table.add_column("Status")
    table.add_column("HuggingFace Repo", style="dim")

    for i, rec in enumerate(recommendations, 1):
        if rec.fits_in_memory:
            status = Text("OK", style="bold green")
        else:
            status = Text("SWAP", style="bold yellow")

        if rec.quality_score >= 90:
            q_style = "bold green"
        elif rec.quality_score >= 80:
            q_style = "green"
        elif rec.quality_score >= 70:
            q_style = "yellow"
        else:
            q_style = "red"

        table.add_row(
            str(i),
            rec.model_name,
            rec.quantization,
            rec.format.upper(),
            f"{rec.memory_required_gb:.1f} GB",
            Text(str(rec.quality_score), style=q_style),
            status,
            rec.hf_repo,
        )

    console.print(table)

    if recommendations:
        top = recommendations[0]
        console.print(
            f"\n[bold green]Top pick:[/bold green] {top.model_name} ({top.quantization}) "
            f"— {top.reason}"
        )


def print_benchmark_result(result: object) -> None:
    """Pretty-print a benchmark result with Rich formatting.

    Args:
        result: BenchmarkResult object.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", result.model_name)
    table.add_row("Quantization", result.quantization)
    table.add_row("TTFT", f"{result.ttft_ms:.1f} ms")
    table.add_row("Generation Speed", f"{result.tokens_per_sec:.1f} tokens/s")
    table.add_row("Peak Memory", f"{result.peak_memory_gb:.2f} GB")

    swap_style = "green" if result.swap_used_gb < 0.1 else "yellow"
    table.add_row("Swap Used", f"[{swap_style}]{result.swap_used_gb:.2f} GB[/{swap_style}]")

    table.add_row("Prompt Tokens", str(result.prompt_tokens))
    table.add_row("Generated Tokens", str(result.generation_tokens))
    table.add_row("Total Tokens", str(result.total_tokens))
    table.add_row("Duration", f"{result.duration_sec:.1f} s")

    console.print(Panel(table, title="[bold]Benchmark Result[/bold]", border_style="blue"))


def _build_watchdog_panel(event: object, alerts: list[str], sample_count: int) -> Panel:
    """Build a Rich Panel for a single watchdog tick.

    Args:
        event: WatchdogEvent with current memory state.
        alerts: List of alert strings for this tick.
        sample_count: Running count of samples taken.

    Returns:
        A Rich Panel renderable.
    """
    pressure = event.pressure_level
    p_color = _PRESSURE_COLORS.get(pressure, "white")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")

    table.add_row("Sample #", str(sample_count))
    table.add_row("Used", f"{event.memory_used_gb:.2f} GB")
    table.add_row("Available", f"{event.memory_available_gb:.2f} GB")

    swap_style = "green" if event.swap_used_gb < 0.5 else ("yellow" if event.swap_used_gb < 2.0 else "red")
    table.add_row("Swap", f"[{swap_style}]{event.swap_used_gb:.2f} GB[/{swap_style}]")
    table.add_row("Pressure", f"[{p_color}]{pressure}[/{p_color}]")

    if alerts:
        for alert in alerts:
            table.add_row("[bold red]Alert[/bold red]", f"[red]{alert}[/red]")

    border = "green" if pressure == "nominal" else ("yellow" if pressure == "warn" else "red")
    return Panel(table, title="[bold]Memory Watchdog[/bold]", subtitle="Ctrl+C to stop", border_style=border)


def print_watchdog_status(event: object, alerts: list[str], sample_count: int) -> None:
    """Print a single memory watchdog status snapshot (non-live mode).

    Args:
        event: WatchdogEvent with current memory state.
        alerts: List of alert strings.
        sample_count: Running count of samples.
    """
    console.print(_build_watchdog_panel(event, alerts, sample_count))


def run_watchdog_live(
    interval_sec: float = 1.0,
    swap_threshold_gb: float = 1.0,
) -> list:
    """Run the memory watchdog with a Rich Live-updating display.

    Blocks until the user presses Ctrl+C, then returns all
    collected events.

    Args:
        interval_sec: Polling interval in seconds.
        swap_threshold_gb: Swap usage threshold for alerts.

    Returns:
        List of WatchdogEvent objects collected during the run.
    """
    from macsmart.manager.watchdog import get_memory_timeline, start_watchdog

    sample_count = 0

    with Live(console=console, refresh_per_second=2, transient=True) as live:
        def _on_tick(event: object, alerts: list[str]) -> None:
            nonlocal sample_count
            sample_count += 1
            panel = _build_watchdog_panel(event, alerts, sample_count)
            live.update(panel)

        events = start_watchdog(
            interval_sec=interval_sec,
            swap_threshold_gb=swap_threshold_gb,
            callback=_on_tick,
        )

    # Print summary after stopping
    if events:
        timeline = get_memory_timeline(events)
        console.print()

        summary = Table(title="Watchdog Summary", show_header=False, box=None, padding=(0, 2))
        summary.add_column("Metric", style="bold cyan")
        summary.add_column("Value")
        summary.add_row("Samples", str(timeline["samples"]))
        summary.add_row("Duration", f"{timeline['duration_sec']:.1f} s")
        summary.add_row(
            "Used (min/avg/max)",
            "{min:.2f} / {avg:.2f} / {max:.2f} GB".format(**timeline["memory_used_gb"]),
        )
        summary.add_row(
            "Available (min/avg/max)",
            "{min:.2f} / {avg:.2f} / {max:.2f} GB".format(**timeline["memory_available_gb"]),
        )
        summary.add_row(
            "Swap (min/avg/max)",
            "{min:.2f} / {avg:.2f} / {max:.2f} GB".format(**timeline["swap_used_gb"]),
        )
        levels = timeline["pressure_levels"]
        if levels:
            parts = [f"{k}: {v}" for k, v in sorted(levels.items())]
            summary.add_row("Pressure", ", ".join(parts))

        console.print(Panel(summary, title="[bold]Session Complete[/bold]", border_style="blue"))

    return events


def print_download_status(status: object) -> None:
    """Pretty-print a model download/cache status.

    Args:
        status: DownloadStatus object.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Repository", status.hf_repo)

    if status.cached:
        table.add_row("Status", "[green]Already cached[/green]")
    else:
        table.add_row("Status", "[bold green]Downloaded[/bold green]")

    if status.local_path:
        table.add_row("Local Path", str(status.local_path))

    if status.size_gb is not None:
        table.add_row("Size", f"{status.size_gb:.2f} GB")

    console.print(Panel(table, title="[bold]Model Download[/bold]", border_style="green"))


def print_run_header(
    task: str,
    confidence: float,
    model: str,
    auto_swap: bool = False,
) -> None:
    """Print the header for a `macsmart run` invocation.

    Args:
        task: Detected or specified task type.
        confidence: Task detection confidence (0.0–1.0).
        model: Selected model repo ID.
        auto_swap: Whether auto-swap is enabled.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    conf_pct = f"{confidence * 100:.0f}%"
    table.add_row("Task", f"{task} ({conf_pct} confidence)")
    table.add_row("Model", model)
    if auto_swap:
        table.add_row("Auto-Swap", "[yellow]Enabled[/yellow]")

    console.print(Panel(table, title="[bold]Run Configuration[/bold]", border_style="blue"))


def print_generation_result(result: object) -> None:
    """Print the result of a generation run.

    Args:
        result: GenerationResult from runtime.
    """
    console.print(Panel(result.text.strip(), title="[bold]Output[/bold]", border_style="green"))

    stats = Table(show_header=False, box=None, padding=(0, 2))
    stats.add_column("Metric", style="bold cyan")
    stats.add_column("Value")
    stats.add_row("TTFT", f"{result.ttft_ms:.1f} ms")
    stats.add_row("Speed", f"{result.tokens_per_sec:.1f} tokens/s")
    stats.add_row("Tokens", f"{result.prompt_tokens} prompt + {result.generation_tokens} generated")
    stats.add_row("Peak Memory", f"{result.peak_memory_gb:.2f} GB")
    console.print(stats)


def print_energy_benchmark_result(result: object) -> None:
    """Print a combined benchmark + energy result.

    Args:
        result: EnergyBenchmarkResult object.
    """
    b = result.benchmark
    e = result.energy

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", b.model_name)
    table.add_row("Quantization", b.quantization)
    table.add_row("Power Source", f"[bold]{result.power_source.upper()}[/bold]")
    table.add_row("Tokens/s", f"{b.tokens_per_sec:.1f}")
    table.add_row("TTFT", f"{b.ttft_ms:.1f} ms")
    table.add_row("Peak Memory", f"{b.peak_memory_gb:.2f} GB")

    if e.total_power_watts is not None:
        table.add_row("Power", f"{e.total_power_watts:.1f} W")
    if e.total_energy_joules is not None:
        table.add_row("Energy", f"{e.total_energy_joules:.1f} J")

    border = "magenta" if result.power_source == "battery" else "green"
    console.print(Panel(table, title=f"[bold]Energy Benchmark ({result.power_source})[/bold]", border_style=border))


def print_energy_comparison(comparison: object) -> None:
    """Print a side-by-side energy comparison.

    Args:
        comparison: EnergyComparison object.
    """
    table = Table(title="Battery vs AC Comparison", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Battery", justify="right")
    table.add_column("AC", justify="right")
    table.add_column("Delta", justify="right")

    b = comparison.battery.benchmark
    a = comparison.ac.benchmark

    table.add_row(
        "Tokens/s",
        f"{b.tokens_per_sec:.1f}",
        f"{a.tokens_per_sec:.1f}",
        f"{a.tokens_per_sec - b.tokens_per_sec:+.1f}",
    )
    table.add_row(
        "TTFT (ms)",
        f"{b.ttft_ms:.1f}",
        f"{a.ttft_ms:.1f}",
        f"{a.ttft_ms - b.ttft_ms:+.1f}",
    )
    table.add_row(
        "Peak Memory",
        f"{b.peak_memory_gb:.2f} GB",
        f"{a.peak_memory_gb:.2f} GB",
        f"{a.peak_memory_gb - b.peak_memory_gb:+.2f} GB",
    )

    be = comparison.battery.energy
    ae = comparison.ac.energy

    bw = be.total_power_watts if be.total_power_watts is not None else 0.0
    aw = ae.total_power_watts if ae.total_power_watts is not None else 0.0
    table.add_row("Power (W)", f"{bw:.1f}", f"{aw:.1f}", f"{aw - bw:+.1f}")

    bj = be.total_energy_joules if be.total_energy_joules is not None else 0.0
    aj = ae.total_energy_joules if ae.total_energy_joules is not None else 0.0
    table.add_row("Energy (J)", f"{bj:.1f}", f"{aj:.1f}", f"{aj - bj:+.1f}")

    console.print(table)

    # Efficiency summary
    if comparison.efficiency_battery is not None:
        console.print(f"\n[bold]Battery efficiency:[/bold] {comparison.efficiency_battery:.3f} tokens/J")
    if comparison.efficiency_ac is not None:
        console.print(f"[bold]AC efficiency:[/bold] {comparison.efficiency_ac:.3f} tokens/J")
    if comparison.speed_ratio is not None:
        console.print(f"[bold]Speed ratio (AC/Battery):[/bold] {comparison.speed_ratio:.3f}x")


def prompt_power_switch(target_source: str) -> None:
    """Prompt the user to switch power source.

    Args:
        target_source: The power source to switch to ("battery" or "ac").
    """
    if target_source == "battery":
        console.print(
            "\n[bold yellow]Please unplug your Mac from AC power.[/bold yellow]"
        )
    else:
        console.print(
            "\n[bold yellow]Please plug your Mac into AC power.[/bold yellow]"
        )
    console.print("[dim]Press Enter when ready...[/dim]")


def print_swap_notification(from_model: str, to_model: str, reason: str) -> None:
    """Print a notification when a model swap occurs.

    Args:
        from_model: Previous model repo ID.
        to_model: New model repo ID.
        reason: Why the swap was triggered.
    """
    console.print(
        f"\n[bold yellow]Model Swap:[/bold yellow] "
        f"{from_model} -> {to_model}\n"
        f"[dim]Reason: {reason}[/dim]"
    )


def print_swap_summary(swap_events: list) -> None:
    """Print a summary of all swaps during a session.

    Args:
        swap_events: List of SwapEvent objects.
    """
    if not swap_events:
        return

    table = Table(title="Model Swaps", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("From", style="red")
    table.add_column("To", style="green")
    table.add_column("Available Memory", justify="right")
    table.add_column("Reason", style="dim")

    for i, evt in enumerate(swap_events, 1):
        table.add_row(
            str(i),
            evt.from_model,
            evt.to_model,
            f"{evt.memory_available_gb:.1f} GB",
            evt.reason,
        )

    console.print(table)


def print_batch_progress(model: str, index: int, total: int) -> None:
    """Print a progress line for batch benchmarking.

    Args:
        model: Model repo ID being benchmarked.
        index: 1-based index of current model.
        total: Total number of models in batch.
    """
    console.print(f"\n[bold][{index}/{total}][/bold] Benchmarking [cyan]{model}[/cyan]...")


def print_batch_summary(results: list) -> None:
    """Print a ranked summary table of batch benchmark results.

    Args:
        results: List of BenchmarkResult objects.
    """
    if not results:
        console.print("[dim]No successful benchmark results to summarize.[/dim]")
        return

    ranked = sorted(results, key=lambda r: r.tokens_per_sec, reverse=True)

    table = Table(title="Batch Benchmark Results", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="cyan")
    table.add_column("Quant", justify="center")
    table.add_column("Tokens/s", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Peak Mem (GB)", justify="right")
    table.add_column("Swap (GB)", justify="right")

    for i, r in enumerate(ranked, 1):
        tps_style = "bold green" if i == 1 else ""
        table.add_row(
            str(i),
            r.model_name,
            r.quantization,
            Text(f"{r.tokens_per_sec:.1f}", style=tps_style),
            f"{r.ttft_ms:.1f}",
            f"{r.peak_memory_gb:.2f}",
            f"{r.swap_used_gb:.2f}",
        )

    console.print(table)

    champion = ranked[0]
    console.print(
        f"\n[bold green]Champion:[/bold green] {champion.model_name} "
        f"({champion.quantization}) at {champion.tokens_per_sec:.1f} tokens/s"
    )


def print_comparison(comparison: dict) -> None:
    """Print a color-coded comparison of two benchmark results.

    Args:
        comparison: Dict from runner.compare_results() with baseline/comparison metrics.
    """
    table = Table(
        title=f"{comparison['baseline']} vs {comparison['comparison']}",
        show_lines=True,
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Comparison", justify="right")
    table.add_column("Delta", justify="right")

    def _delta_cell(val: float | None, label: str, higher_is_better: bool = True) -> str:
        if val is None:
            return "N/A"
        if val > 0:
            arrow = "^" if higher_is_better else "v"
            color = "green" if higher_is_better else "red"
        elif val < 0:
            arrow = "v" if higher_is_better else "^"
            color = "red" if higher_is_better else "green"
        else:
            return f"{val:+.2f} {label}"
        return f"[{color}]{val:+.2f} {label} {arrow}[/{color}]"

    # Tokens/sec (higher is better)
    tps = comparison["tokens_per_sec"]
    delta = tps["comparison"] - tps["baseline"]
    speedup = f" ({tps['speedup']}x)" if tps["speedup"] is not None else ""
    table.add_row(
        "Tokens/s",
        f"{tps['baseline']:.1f}",
        f"{tps['comparison']:.1f}",
        _delta_cell(delta, "tok/s", higher_is_better=True) + speedup,
    )

    # TTFT (lower is better)
    ttft = comparison["ttft_ms"]
    table.add_row(
        "TTFT (ms)",
        f"{ttft['baseline']:.1f}",
        f"{ttft['comparison']:.1f}",
        _delta_cell(ttft["delta_ms"], "ms", higher_is_better=False),
    )

    # Peak memory (lower is better)
    mem = comparison["peak_memory_gb"]
    table.add_row(
        "Peak Memory (GB)",
        f"{mem['baseline']:.2f}",
        f"{mem['comparison']:.2f}",
        _delta_cell(mem["delta_gb"], "GB", higher_is_better=False),
    )

    # Swap (lower is better)
    swap = comparison["swap_used_gb"]
    table.add_row(
        "Swap (GB)",
        f"{swap['baseline']:.2f}",
        f"{swap['comparison']:.2f}",
        _delta_cell(swap["delta_gb"], "GB", higher_is_better=False),
    )

    # Duration (lower is better)
    dur = comparison["duration_sec"]
    speedup_dur = f" ({dur['speedup']}x)" if dur["speedup"] is not None else ""
    delta_dur = dur["comparison"] - dur["baseline"]
    table.add_row(
        "Duration (s)",
        f"{dur['baseline']:.1f}",
        f"{dur['comparison']:.1f}",
        _delta_cell(delta_dur, "s", higher_is_better=False) + speedup_dur,
    )

    console.print(table)


def print_cached_models(models: list) -> None:
    """Pretty-print a list of cached models.

    Args:
        models: List of DownloadStatus objects.
    """
    total_gb = sum(m.size_gb for m in models if m.size_gb is not None)

    table = Table(title="Cached Models", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Repository", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    for i, m in enumerate(models, 1):
        size_str = f"{m.size_gb:.2f} GB" if m.size_gb is not None else "?"
        path_str = str(m.local_path) if m.local_path else "?"
        table.add_row(str(i), m.hf_repo, size_str, path_str)

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(models)} models, {total_gb:.2f} GB")
