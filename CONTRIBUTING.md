# M5 LLM Benchmark — Smart Local LLM Manager for 16GB Macs

## Project Overview

**MacSmart LLM** — A memory-intelligent local LLM orchestration tool optimized for 16GB Apple Silicon Macs (the most common config sold). Unlike LM Studio or Ollama which treat all Macs the same, this tool actively monitors available memory, thermal state, and battery level to recommend and manage the best model + quantization for your current system state.

## Why This Exists

- 16GB is the most common MacBook config but completely ignored by benchmarking research
- Existing tools (LM Studio, Ollama) let users download models that crash or swap-thrash on 16GB
- No tool auto-selects the best model/quantization based on *actual available* memory
- M5 Neural Accelerators (via MLX) are untapped on 16GB configs — Apple only tested 24GB

## Architecture

```
macsmart/
├── CONTRIBUTING.md             # This file
├── README.md                  # User-facing docs
├── pyproject.toml             # Project config (use uv or pip)
├── macsmart/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point (click or typer)
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── memory.py          # Real-time memory profiling (vm_stat, psutil)
│   │   ├── thermal.py         # Thermal state monitoring (powermetrics, IOKit)
│   │   ├── battery.py         # Battery level + plugged-in detection
│   │   └── system.py          # Chip detection (M5 vs M4 vs M3), GPU cores, bandwidth
│   ├── recommender/
│   │   ├── __init__.py
│   │   ├── engine.py          # Core recommendation logic
│   │   ├── models_db.py       # Database of models with memory requirements
│   │   └── task_router.py     # Task-type detection (coding, writing, chat, summarization)
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── runner.py          # Run inference benchmarks (TTFT, tokens/sec, memory peak)
│   │   ├── energy.py          # Energy measurement (powermetrics wrapper)
│   │   ├── energy_compare.py  # Battery vs AC power comparison
│   │   └── report.py          # Generate benchmark reports (JSON + markdown)
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── server.py          # Local web server for benchmark dashboard
│   │   └── static/            # HTML/CSS/JS for dashboard UI
│   ├── manager/
│   │   ├── __init__.py
│   │   ├── download.py        # Model download manager (HuggingFace hub)
│   │   ├── runtime.py         # MLX / Ollama runtime wrapper
│   │   ├── session.py         # Model session lifecycle (load, generate, swap)
│   │   ├── swapper.py         # Dynamic model swapping on memory pressure
│   │   └── watchdog.py        # Memory watchdog — alert/downsize if memory pressure rises
│   ├── data/
│   │   ├── __init__.py
│   │   └── registry.yaml      # Model definitions with memory profiles
│   └── ui/
│       ├── __init__.py
│       └── terminal.py        # Rich terminal UI (rich library)
├── benchmarks/                # Stored benchmark results
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_profiler.py
    ├── test_recommender.py
    └── test_benchmark.py
```

## Tech Stack

- **Python 3.11+** (ships with macOS)
- **MLX + mlx-lm** — Apple's ML framework, Neural Accelerator support
- **psutil** — Cross-platform system monitoring
- **huggingface_hub** — Model downloading
- **click** or **typer** — CLI framework
- **rich** — Beautiful terminal output
- **PyYAML** — Model registry
- **pytest** — Testing

## MVP Features (Phase 1)

### 1. System Profiler (`macsmart profile`)
- Detect chip (M1/M2/M3/M4/M5), GPU cores, memory bandwidth
- Report available memory (total minus OS/apps)
- Check thermal state
- Check battery level + AC power status
- Output: JSON summary of system capabilities

### 2. Model Recommender (`macsmart recommend`)
- Input: available memory + optional task type
- Output: ranked list of recommended models with quantization
- Logic:
  - Reserve 4-5GB for macOS + apps
  - Calculate max model size from remaining memory
  - Rank by quality within memory budget
  - Prefer MLX format for M5 (Neural Accelerator support)
  - Flag models that will require swap (with warning)

### 3. Benchmark Runner (`macsmart benchmark <model>`)
- Run standardized inference benchmark
- Measure: TTFT, tokens/sec (generation), peak memory, swap usage
- Optional: energy measurement (requires sudo for powermetrics)
- Save results to benchmarks/ directory
- Compare against stored results

### 4. Memory Watchdog (`macsmart watch`)
- Monitor memory pressure in real-time during LLM inference
- Alert when approaching swap threshold
- Log memory timeline for analysis

## Model Registry Format (registry.yaml)

```yaml
models:
  - name: "Qwen2.5-7B-Instruct"
    family: "qwen"
    params: "7B"
    quantizations:
      - quant: "4bit"
        format: "mlx"
        hf_repo: "mlx-community/Qwen2.5-7B-Instruct-4bit"
        memory_gb: 4.5
        quality_score: 82  # Relative score 0-100
      - quant: "8bit"
        format: "mlx"
        hf_repo: "mlx-community/Qwen2.5-7B-Instruct-8bit"
        memory_gb: 8.2
        quality_score: 90
    tasks: ["general", "coding", "writing"]
    
  - name: "Mistral-7B-Instruct-v0.3"
    family: "mistral"
    params: "7B"
    quantizations:
      - quant: "4bit"
        format: "mlx"
        hf_repo: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        memory_gb: 4.3
        quality_score: 80
    tasks: ["general", "coding"]
```

## Key Design Principles

1. **Memory-first**: Every decision starts with "how much memory is actually available right now?"
2. **No crashes**: Never recommend a model that will freeze or swap-thrash the system
3. **M5-aware**: Detect M5 Neural Accelerators and prefer MLX format when available
4. **Honest benchmarks**: Report real numbers including swap usage and thermal throttling
5. **Beautiful CLI**: Use rich library for colorful, informative terminal output
6. **Offline-capable**: Once models are downloaded, everything works without internet

## Development Guidelines

- Use type hints everywhere
- Write docstrings for public functions
- Keep functions small and testable
- Use pathlib for file paths
- Handle errors gracefully with informative messages
- Test on macOS (this is a macOS-specific tool, but profiler should gracefully degrade on Linux)
- Use `subprocess.run` for macOS commands like `vm_stat`, `sysctl`, `powermetrics`

## Phase 2 (Implemented)

- Dynamic model swapping (auto-downsize when memory pressure rises)
- Task-aware routing (auto-pick best model for detected task type)
- Benchmark comparison dashboard (web UI)
- Battery vs plugged-in energy comparison report
- Batch benchmarking and model comparison

## Phase 3 (Future)

- Menu bar app (Swift/SwiftUI wrapper)
- Community benchmark sharing

## Useful macOS Commands

```bash
# Memory info
vm_stat                          # Virtual memory stats
memory_pressure                  # Current memory pressure level
sysctl hw.memsize               # Total RAM in bytes

# Chip detection
sysctl -n machdep.cpu.brand_string  # CPU brand
system_profiler SPHardwareDataType   # Full hardware info

# Thermal
sudo powermetrics --samplers smc -i 1000 -n 1  # Thermal sensors

# Battery
pmset -g batt                    # Battery status
ioreg -l | grep -i capacity      # Battery capacity details

# GPU info
system_profiler SPDisplaysDataType  # GPU details
```

## Reference: 16GB Memory Budget

| Component | Memory |
|-----------|--------|
| macOS + system | ~3-4 GB |
| Typical user apps | ~1-2 GB |
| **Available for LLM** | **~10-12 GB** |
| Safe model size (no swap) | **~8-9 GB** |
| Max model size (some swap OK) | **~12-14 GB** |

## Reference: Model Sizes (Approximate)

| Model | BF16 | 8-bit | 4-bit | 3-bit |
|-------|------|-------|-------|-------|
| 1.5B  | 3 GB | 1.5 GB | 1 GB  | 0.75 GB |
| 3B    | 6 GB | 3 GB   | 2 GB  | 1.5 GB |
| 7B    | 14 GB | 7 GB  | 4.5 GB | 3.5 GB |
| 8B    | 16 GB | 8 GB  | 5 GB  | 4 GB |
| 14B   | 28 GB | 14 GB | 9 GB  | 7 GB |
| 30B MoE (3B active) | — | — | 18 GB | 14 GB |
