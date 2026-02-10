"""Database of models with memory requirements, loaded from registry.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass
class QuantizationVariant:
    """A specific quantization of a model."""

    quant: str
    format: str
    hf_repo: str
    memory_gb: float
    quality_score: int


@dataclass
class ModelEntry:
    """A model in the registry with its quantization variants."""

    name: str
    family: str
    params: str
    quantizations: list[QuantizationVariant] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)


_DEFAULT_REGISTRY = files("macsmart.data").joinpath("registry.yaml")


def load_registry(path=None) -> list[ModelEntry]:
    """Load model registry from YAML file.

    Args:
        path: Path to registry.yaml. Defaults to bundled package data.

    Returns:
        List of ModelEntry objects.

    Raises:
        FileNotFoundError: If the registry file does not exist.
    """
    registry_path = path or _DEFAULT_REGISTRY
    if isinstance(registry_path, Path):
        with open(registry_path) as f:
            data = yaml.safe_load(f)
    else:
        data = yaml.safe_load(registry_path.read_text())

    entries: list[ModelEntry] = []
    for item in data.get("models", []):
        quants = [
            QuantizationVariant(
                quant=q["quant"],
                format=q["format"],
                hf_repo=q["hf_repo"],
                memory_gb=q["memory_gb"],
                quality_score=q["quality_score"],
            )
            for q in item.get("quantizations", [])
        ]
        entries.append(
            ModelEntry(
                name=item["name"],
                family=item["family"],
                params=item["params"],
                quantizations=quants,
                tasks=item.get("tasks", []),
            )
        )
    return entries


def get_models_for_task(task: str, models: list[ModelEntry] | None = None) -> list[ModelEntry]:
    """Filter models that support a given task type.

    Args:
        task: Task type (e.g. 'coding', 'writing', 'general').
        models: Optional pre-loaded model list. Loads registry if None.

    Returns:
        Filtered list of models supporting the task.
    """
    if models is None:
        models = load_registry()

    task_lower = task.lower()
    return [m for m in models if task_lower in [t.lower() for t in m.tasks]]
