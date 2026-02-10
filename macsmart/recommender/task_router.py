"""Task-type detection for routing to the best model."""

from __future__ import annotations

import re

SUPPORTED_TASKS = ["general", "coding", "writing", "chat", "summarization"]

# Keyword patterns for each task type, checked in priority order.
_TASK_PATTERNS: list[tuple[str, list[str]]] = [
    ("coding", [
        r"\bcode\b", r"\bprogram\b", r"\bfunction\b", r"\bclass\b",
        r"\bbug\b", r"\bdebug\b", r"\brefactor\b", r"\bpython\b",
        r"\bjavascript\b", r"\btypescript\b", r"\brust\b", r"\bjava\b",
        r"\bimplementat", r"\balgorithm\b", r"\bapi\b", r"\bsql\b",
        r"\bhtml\b", r"\bcss\b", r"\bgit\b", r"\bcompile\b",
    ]),
    ("summarization", [
        r"\bsummar", r"\btl;?dr\b", r"\bdigest\b", r"\bcondense\b",
        r"\bkey\s*points?\b", r"\bhighlights?\b", r"\bbrief\b",
    ]),
    ("writing", [
        r"\bwrite\b", r"\bessay\b", r"\bstory\b", r"\barticle\b",
        r"\bblog\b", r"\bletter\b", r"\bemail\b", r"\bcreative\b",
        r"\bpoem\b", r"\bscript\b", r"\bdraft\b", r"\bcompose\b",
    ]),
    ("chat", [
        r"\bchat\b", r"\btalk\b", r"\bconvers", r"\bhello\b",
        r"\bhi\b", r"\bhey\b", r"\bwhat do you think\b",
    ]),
]


def detect_task_type(prompt: str) -> str:
    """Detect the task type from a user prompt.

    Uses keyword matching to classify the prompt. Returns 'general'
    if no specific task pattern matches.

    Args:
        prompt: User input text.

    Returns:
        Detected task type string.
    """
    prompt_lower = prompt.lower()
    for task, patterns in _TASK_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                return task
    return "general"


def detect_task_with_confidence(prompt: str) -> tuple[str, float]:
    """Detect the task type from a prompt with a confidence score.

    Confidence is the ratio of matched keywords to total keywords
    for the winning task category. Returns ("general", 0.0) when
    no specific task matches.

    Args:
        prompt: User input text.

    Returns:
        Tuple of (task_type, confidence) where confidence is 0.0–1.0.
    """
    prompt_lower = prompt.lower()
    best_task = "general"
    best_confidence = 0.0

    for task, patterns in _TASK_PATTERNS:
        matches = sum(1 for p in patterns if re.search(p, prompt_lower))
        if matches > 0:
            confidence = matches / len(patterns)
            if confidence > best_confidence:
                best_confidence = confidence
                best_task = task

    return best_task, round(best_confidence, 3)


def validate_task(task: str) -> str:
    """Validate and normalize a task type string.

    Args:
        task: Task type to validate.

    Returns:
        Normalized task type string (lowercase).

    Raises:
        ValueError: If task type is not recognized.
    """
    normalized = task.lower().strip()
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task type '{task}'. "
            f"Supported tasks: {', '.join(SUPPORTED_TASKS)}"
        )
    return normalized
