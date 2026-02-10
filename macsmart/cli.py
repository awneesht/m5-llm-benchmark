"""CLI entry point for MacSmart LLM."""

from __future__ import annotations

import json

import click


@click.group()
@click.version_option(package_name="macsmart")
def cli() -> None:
    """MacSmart LLM — Smart local LLM manager for 16GB Macs."""


@cli.command()
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON instead of formatted tables")
def profile(as_json: bool) -> None:
    """Show system profile: chip, memory, thermal state, battery."""
    from macsmart.profiler.system import get_system_profile
    from macsmart.ui.terminal import print_system_profile

    data = get_system_profile()

    if as_json:
        click.echo(json.dumps(data, indent=2))
    else:
        print_system_profile(data)


@cli.command()
@click.option("--task", type=str, default=None, help="Task type: coding, writing, chat, summarization")
@click.option("--memory", "memory_override", type=float, default=None, help="Override available memory (GB)")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON instead of formatted tables")
def recommend(task: str | None, memory_override: float | None, as_json: bool) -> None:
    """Recommend the best model + quantization for current system state."""
    from macsmart.profiler.memory import get_available_for_llm
    from macsmart.recommender.engine import recommend_models
    from macsmart.ui.terminal import print_recommendations

    available = memory_override if memory_override is not None else get_available_for_llm()
    recs = recommend_models(available, task=task)

    if not recs:
        click.echo("No models found for the given constraints.")
        return

    if as_json:
        data = [
            {
                "model_name": r.model_name,
                "quantization": r.quantization,
                "format": r.format,
                "hf_repo": r.hf_repo,
                "memory_required_gb": r.memory_required_gb,
                "quality_score": r.quality_score,
                "fits_in_memory": r.fits_in_memory,
                "swap_warning": r.swap_warning,
                "reason": r.reason,
            }
            for r in recs
        ]
        click.echo(json.dumps(data, indent=2))
    else:
        print_recommendations(recs, available_memory_gb=available)


@cli.command()
@click.argument("prompt")
@click.option("--task", type=str, default=None, help="Task type: coding, writing, chat, summarization")
@click.option("--model", type=str, default=None, help="Specific model repo to use (skips auto-select)")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
@click.option("--memory", "memory_override", type=float, default=None, help="Override available memory (GB)")
@click.option("--auto-swap/--no-auto-swap", default=False, help="Enable dynamic model swapping on memory pressure")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON")
def run(
    prompt: str,
    task: str | None,
    model: str | None,
    max_tokens: int,
    memory_override: float | None,
    auto_swap: bool,
    as_json: bool,
) -> None:
    """Run inference: auto-detect task, pick model, generate.

    PROMPT is the text prompt to send to the model.
    """
    from macsmart.manager.session import Session, SessionConfig
    from macsmart.ui.terminal import print_generation_result, print_run_header

    config = SessionConfig(
        prompt=prompt,
        task=task,
        model=model,
        max_tokens=max_tokens,
        memory_override=memory_override,
        auto_swap=auto_swap,
    )
    session = Session(config=config)

    try:
        session.select_model()
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    if not as_json:
        print_run_header(
            task=session.selected_task or "general",
            confidence=session.task_confidence,
            model=session.selected_model,
            auto_swap=auto_swap,
        )

    swapper = None
    if auto_swap:
        from macsmart.manager.swapper import DynamicSwapper, SwapPolicy

        swapper = DynamicSwapper(session=session, policy=SwapPolicy())
        swapper.start()

    try:
        session.load()
        result = session.generate()
    except Exception as e:
        raise click.ClickException(f"Generation failed: {e}") from e
    finally:
        session.unload()
        if swapper:
            swapper.stop()

    if as_json:
        data = {
            "task": session.selected_task,
            "task_confidence": session.task_confidence,
            "model": session.selected_model,
            "text": result.text,
            "ttft_ms": result.ttft_ms,
            "tokens_per_sec": result.tokens_per_sec,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "peak_memory_gb": result.peak_memory_gb,
            "swaps": session.swap_history,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        print_generation_result(result)


@cli.command()
@click.argument("model")
@click.option("--prompt", type=str, default=None, help="Custom prompt (default: built-in benchmark prompt)")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
@click.option("--save/--no-save", default=True, help="Save results to benchmarks/ directory")
@click.option("--energy", is_flag=True, help="Also measure energy consumption")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON instead of formatted tables")
def benchmark(model: str, prompt: str | None, max_tokens: int, save: bool, energy: bool, as_json: bool) -> None:
    """Run inference benchmark on a model.

    MODEL is a HuggingFace repo ID (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit).
    """
    from dataclasses import asdict

    from macsmart.benchmark.report import save_result
    from macsmart.benchmark.runner import run_benchmark
    from macsmart.ui.terminal import print_benchmark_result

    click.echo(f"Benchmarking {model} (max_tokens={max_tokens})...")

    if energy:
        from macsmart.benchmark.energy_compare import run_energy_benchmark, save_energy_result
        from macsmart.profiler.battery import is_on_ac_power
        from macsmart.ui.terminal import print_energy_benchmark_result

        power_source = "ac" if is_on_ac_power() else "battery"
        try:
            result = run_energy_benchmark(model, power_source=power_source, prompt=prompt, max_tokens=max_tokens)
        except Exception as e:
            raise click.ClickException(f"Benchmark failed: {e}") from e

        if as_json:
            data = {
                "benchmark": asdict(result.benchmark),
                "energy": asdict(result.energy),
                "power_source": result.power_source,
            }
            click.echo(json.dumps(data, indent=2))
        else:
            print_energy_benchmark_result(result)

        if save:
            path = save_energy_result(result)
            click.echo(f"\nResults saved to {path}")
        return

    try:
        result = run_benchmark(model, prompt=prompt, max_tokens=max_tokens)
    except Exception as e:
        raise click.ClickException(f"Benchmark failed: {e}") from e

    if as_json:
        click.echo(json.dumps(asdict(result), indent=2))
    else:
        print_benchmark_result(result)

    if save:
        path = save_result(result)
        click.echo(f"\nResults saved to {path}")


@cli.command("energy-compare")
@click.argument("model")
@click.option("--prompt", type=str, default=None, help="Custom prompt")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
@click.option("--save/--no-save", default=True, help="Save results to benchmarks/")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON")
def energy_compare(model: str, prompt: str | None, max_tokens: int, save: bool, as_json: bool) -> None:
    """Compare inference on battery vs AC power.

    Runs the benchmark twice: once on current power source, then prompts
    you to switch, verifies, and runs again. MODEL is a HuggingFace repo ID.
    """
    from dataclasses import asdict

    from macsmart.benchmark.energy_compare import (
        compare_energy,
        run_energy_benchmark,
        save_energy_result,
    )
    from macsmart.profiler.battery import is_on_ac_power
    from macsmart.ui.terminal import (
        print_energy_benchmark_result,
        print_energy_comparison,
        prompt_power_switch,
    )

    # Determine current power source
    on_ac = is_on_ac_power()
    first_source = "ac" if on_ac else "battery"
    second_source = "battery" if on_ac else "ac"

    click.echo(f"Phase 1: Benchmarking on {first_source.upper()} power...")
    try:
        first_result = run_energy_benchmark(
            model, power_source=first_source, prompt=prompt, max_tokens=max_tokens,
        )
    except Exception as e:
        raise click.ClickException(f"First benchmark failed: {e}") from e

    if not as_json:
        print_energy_benchmark_result(first_result)

    if save:
        path = save_energy_result(first_result)
        click.echo(f"Saved to {path}")

    # Prompt to switch power source
    if not as_json:
        prompt_power_switch(second_source)
    click.pause()

    # Verify power switched
    switched = is_on_ac_power() != on_ac
    if not switched:
        click.echo(
            f"[Warning] Power source appears unchanged. "
            f"Expected {second_source}, still on {first_source}. "
            f"Continuing anyway..."
        )

    click.echo(f"\nPhase 2: Benchmarking on {second_source.upper()} power...")
    try:
        second_result = run_energy_benchmark(
            model, power_source=second_source, prompt=prompt, max_tokens=max_tokens,
        )
    except Exception as e:
        raise click.ClickException(f"Second benchmark failed: {e}") from e

    if not as_json:
        print_energy_benchmark_result(second_result)

    if save:
        path = save_energy_result(second_result)
        click.echo(f"Saved to {path}")

    # Compare
    if first_source == "battery":
        comparison = compare_energy(battery=first_result, ac=second_result)
    else:
        comparison = compare_energy(battery=second_result, ac=first_result)

    if as_json:
        data = {
            "battery": {
                "benchmark": asdict(comparison.battery.benchmark),
                "energy": asdict(comparison.battery.energy),
            },
            "ac": {
                "benchmark": asdict(comparison.ac.benchmark),
                "energy": asdict(comparison.ac.energy),
            },
            "speed_ratio": comparison.speed_ratio,
            "ttft_delta_ms": comparison.ttft_delta_ms,
            "energy_ratio": comparison.energy_ratio,
            "efficiency_battery": comparison.efficiency_battery,
            "efficiency_ac": comparison.efficiency_ac,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        print_energy_comparison(comparison)


@cli.command()
@click.option("--port", type=int, default=8377, help="Port for the dashboard server")
@click.option("--no-browser", is_flag=True, help="Do not auto-open the browser")
@click.option("--results-dir", type=click.Path(exists=False), default=None, help="Custom results directory")
def dashboard(port: int, no_browser: bool, results_dir: str | None) -> None:
    """Open the benchmark comparison dashboard in a browser.

    Starts a local web server showing charts and tables of all
    benchmark results. Press Ctrl+C to stop the server.
    """
    from pathlib import Path

    from macsmart.dashboard.server import start_server, stop_server

    rdir = Path(results_dir) if results_dir else None
    click.echo(f"Starting dashboard at http://127.0.0.1:{port}")
    click.echo("Press Ctrl+C to stop.")

    start_server(port=port, results_dir=rdir, open_browser=not no_browser)

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()
        click.echo("\nDashboard stopped.")


@cli.command()
@click.option("--interval", type=float, default=1.0, help="Polling interval in seconds")
@click.option("--swap-threshold", type=float, default=1.0, help="Swap usage threshold in GB for alerts")
@click.option("--json-output", "as_json", is_flag=True, help="Output JSON summary on exit instead of Rich UI")
def watch(interval: float, swap_threshold: float, as_json: bool) -> None:
    """Monitor memory pressure in real-time.

    Shows a live-updating display of memory usage, swap, and
    pressure level. Alerts when swap exceeds the threshold.
    Press Ctrl+C to stop and see a summary.
    """
    from macsmart.manager.watchdog import get_memory_timeline, start_watchdog
    from macsmart.ui.terminal import print_watchdog_status, run_watchdog_live

    if as_json:
        # Non-interactive mode: collect events and print JSON summary
        click.echo("Watching memory (Ctrl+C to stop)...")
        events = start_watchdog(
            interval_sec=interval,
            swap_threshold_gb=swap_threshold,
        )
        timeline = get_memory_timeline(events)
        click.echo(json.dumps(timeline, indent=2))
    else:
        run_watchdog_live(
            interval_sec=interval,
            swap_threshold_gb=swap_threshold,
        )


@cli.command()
@click.argument("model")
@click.option("--force", is_flag=True, help="Re-download even if already cached")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON instead of formatted display")
def download(model: str, force: bool, as_json: bool) -> None:
    """Download a model from HuggingFace Hub.

    MODEL is a HuggingFace repo ID (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit).
    """
    from macsmart.manager.download import download_model
    from macsmart.ui.terminal import print_download_status

    click.echo(f"Downloading {model}...")

    try:
        status = download_model(model, force=force)
    except Exception as e:
        raise click.ClickException(f"Download failed: {e}") from e

    if as_json:
        data = {
            "hf_repo": status.hf_repo,
            "local_path": str(status.local_path) if status.local_path else None,
            "cached": status.cached,
            "size_gb": status.size_gb,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        print_download_status(status)


@cli.command()
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON instead of formatted display")
def models(as_json: bool) -> None:
    """List all locally cached models."""
    from macsmart.manager.download import list_cached_models
    from macsmart.ui.terminal import print_cached_models

    cached = list_cached_models()

    if not cached:
        click.echo("No models cached locally.")
        return

    if as_json:
        data = [
            {
                "hf_repo": m.hf_repo,
                "local_path": str(m.local_path) if m.local_path else None,
                "size_gb": m.size_gb,
            }
            for m in cached
        ]
        click.echo(json.dumps(data, indent=2))
    else:
        print_cached_models(cached)


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--prompt", type=str, default=None, help="Custom prompt (default: built-in benchmark prompt)")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
@click.option("--save/--no-save", default=True, help="Save results to benchmarks/ directory")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON instead of formatted tables")
def batch(
    models: tuple[str, ...],
    prompt: str | None,
    max_tokens: int,
    save: bool,
    as_json: bool,
) -> None:
    """Benchmark multiple models sequentially and show a ranked comparison.

    MODELS is two or more HuggingFace repo IDs to benchmark.
    """
    from dataclasses import asdict

    from macsmart.benchmark.report import save_result
    from macsmart.benchmark.runner import run_benchmark
    from macsmart.ui.terminal import print_batch_progress, print_batch_summary

    if len(models) < 2:
        raise click.UsageError("Provide at least 2 models to batch benchmark.")

    results: list = []
    errors: dict[str, str] = {}
    total = len(models)

    for i, model in enumerate(models, 1):
        if not as_json:
            print_batch_progress(model, i, total)

        try:
            result = run_benchmark(model, prompt=prompt, max_tokens=max_tokens)
            results.append(result)

            if save:
                path = save_result(result)
                if not as_json:
                    click.echo(f"  Saved to {path}")
        except Exception as e:
            errors[model] = str(e)
            if not as_json:
                click.echo(f"  [Error] {model}: {e}")

    if as_json:
        data: dict = {"results": [asdict(r) for r in results]}
        if errors:
            data["errors"] = errors
        click.echo(json.dumps(data, indent=2))
    else:
        print_batch_summary(results)
        if errors:
            click.echo(f"\n{len(errors)} model(s) failed:")
            for model, err in errors.items():
                click.echo(f"  - {model}: {err}")


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--latest", "latest_n", type=int, default=None, help="Compare the N most recent results (default: 2)")
@click.option("--results-dir", type=click.Path(exists=False), default=None, help="Custom results directory")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON")
def compare(
    files: tuple[str, ...],
    latest_n: int | None,
    results_dir: str | None,
    as_json: bool,
) -> None:
    """Compare two benchmark results side-by-side.

    Provide two result FILES, or use --latest to compare the most recent results.
    """
    from pathlib import Path

    from macsmart.benchmark.report import get_latest_result_paths, load_result_from_file
    from macsmart.benchmark.runner import compare_results
    from macsmart.ui.terminal import print_comparison

    if files and latest_n is not None:
        raise click.UsageError("Cannot use both FILES and --latest at the same time.")

    if files:
        if len(files) != 2:
            raise click.UsageError("Provide exactly 2 result files to compare.")
        paths = [Path(f) for f in files]
    elif latest_n is not None:
        if latest_n < 2:
            raise click.UsageError("--latest must be at least 2.")
        rdir = Path(results_dir) if results_dir else None
        try:
            paths = get_latest_result_paths(n=latest_n, results_dir=rdir)[:2]
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    else:
        raise click.UsageError("Provide 2 result files or use --latest.")

    try:
        result_a = load_result_from_file(paths[0])
        result_b = load_result_from_file(paths[1])
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    comparison = compare_results(result_a, result_b)

    if as_json:
        click.echo(json.dumps(comparison, indent=2))
    else:
        print_comparison(comparison)


@cli.command()
@click.argument("model")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete(model: str, yes: bool) -> None:
    """Delete a cached model from disk.

    MODEL is a HuggingFace repo ID (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit).
    """
    from macsmart.manager.download import delete_cached_model, get_cached_size_gb

    size = get_cached_size_gb(model)
    if size is None:
        raise click.ClickException(f"Model {model} is not cached locally.")

    if not yes:
        click.confirm(
            f"Delete {model} ({size:.2f} GB) from cache?",
            abort=True,
        )

    deleted = delete_cached_model(model)
    if deleted:
        click.echo(f"Deleted {model} from cache.")
    else:
        raise click.ClickException(f"Failed to delete {model}.")
