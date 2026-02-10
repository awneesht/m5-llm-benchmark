# MacSmart LLM

**Memory-intelligent local LLM orchestration for 16GB Apple Silicon Macs.**

Unlike LM Studio or Ollama which treat all Macs the same, MacSmart actively monitors available memory, thermal state, and battery level to recommend and manage the best model + quantization for your current system state.

## Features

- **System Profiler** — Detect chip, GPU cores, available memory, thermal state, battery
- **Model Recommender** — Auto-select the best model + quantization for your available memory
- **Benchmark Runner** — Measure TTFT, tokens/sec, peak memory, swap usage, energy
- **Memory Watchdog** — Real-time memory pressure monitoring during inference

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Usage

```bash
# Show system profile
macsmart profile

# Get model recommendations for your current memory
macsmart recommend

# Recommend for a specific task
macsmart recommend --task coding

# Benchmark a model
macsmart benchmark mlx-community/Qwen2.5-7B-Instruct-4bit

# Watch memory pressure in real-time
macsmart watch
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11+
- 16GB RAM (optimized for this config, works on others too)

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=macsmart
```
