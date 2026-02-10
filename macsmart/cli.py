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
@click.argument("model")
@click.option("--prompt", type=str, default=None, help="Custom prompt (default: built-in benchmark prompt)")
@click.option("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
@click.option("--save/--no-save", default=True, help="Save results to benchmarks/ directory")
@click.option("--json-output", "as_json", is_flag=True, help="Output raw JSON instead of formatted tables")
def benchmark(model: str, prompt: str | None, max_tokens: int, save: bool, as_json: bool) -> None:
    """Run inference benchmark on a model.

    MODEL is a HuggingFace repo ID (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit).
    """
    from dataclasses import asdict

    from macsmart.benchmark.report import save_result
    from macsmart.benchmark.runner import run_benchmark
    from macsmart.ui.terminal import print_benchmark_result

    click.echo(f"Benchmarking {model} (max_tokens={max_tokens})...")

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
