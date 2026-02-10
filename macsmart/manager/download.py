"""Model download manager using HuggingFace Hub."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import (
    repo_info,
    scan_cache_dir,
    snapshot_download,
)

logger = logging.getLogger(__name__)


@dataclass
class DownloadStatus:
    """Status of a model download or cache lookup."""

    hf_repo: str
    local_path: Path | None
    cached: bool
    size_gb: float | None


def download_model(
    hf_repo: str,
    cache_dir: Path | None = None,
    force: bool = False,
) -> DownloadStatus:
    """Download a model from HuggingFace Hub.

    Uses ``snapshot_download`` to fetch all model files into the
    HuggingFace cache. If the model is already cached and *force*
    is False, the cached path is returned immediately.

    Args:
        hf_repo: HuggingFace repository ID
            (e.g. ``mlx-community/Qwen2.5-7B-Instruct-4bit``).
        cache_dir: Optional local cache directory override.
        force: Re-download even if already cached.

    Returns:
        DownloadStatus with the local path and metadata.

    Raises:
        Exception: If the download fails (network error, repo not found, etc.).
    """
    if not force and is_model_cached(hf_repo, cache_dir=cache_dir):
        local_path = _get_cached_path(hf_repo, cache_dir=cache_dir)
        size = get_cached_size_gb(hf_repo, cache_dir=cache_dir)
        return DownloadStatus(
            hf_repo=hf_repo,
            local_path=local_path,
            cached=True,
            size_gb=size,
        )

    kwargs: dict = {"repo_id": hf_repo, "force_download": force}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    local_dir = snapshot_download(**kwargs)
    local_path = Path(local_dir)

    size = get_cached_size_gb(hf_repo, cache_dir=cache_dir)

    return DownloadStatus(
        hf_repo=hf_repo,
        local_path=local_path,
        cached=False,
        size_gb=size,
    )


def is_model_cached(hf_repo: str, cache_dir: Path | None = None) -> bool:
    """Check if a model is already downloaded locally.

    Scans the HuggingFace cache directory for the given repo ID.

    Args:
        hf_repo: HuggingFace repository ID.
        cache_dir: Optional cache directory to check.

    Returns:
        True if the model is cached locally.
    """
    try:
        cache = scan_cache_dir(cache_dir=cache_dir)
    except Exception:
        return False

    return any(r.repo_id == hf_repo for r in cache.repos)


def get_model_size_gb(hf_repo: str) -> float | None:
    """Get the approximate size of a model from HuggingFace Hub metadata.

    Makes an API call to fetch file metadata and sums up sizes.

    Args:
        hf_repo: HuggingFace repository ID.

    Returns:
        Model size in GB, or None if unavailable.
    """
    try:
        info = repo_info(hf_repo, files_metadata=True)
    except Exception:
        logger.debug("Failed to fetch repo info for %s", hf_repo)
        return None

    total_bytes = sum(s.size for s in info.siblings if s.size)
    if total_bytes == 0:
        return None

    return round(total_bytes / (1024**3), 2)


def get_cached_size_gb(hf_repo: str, cache_dir: Path | None = None) -> float | None:
    """Get the on-disk size of a cached model.

    Args:
        hf_repo: HuggingFace repository ID.
        cache_dir: Optional cache directory.

    Returns:
        Size in GB, or None if not cached.
    """
    try:
        cache = scan_cache_dir(cache_dir=cache_dir)
    except Exception:
        return None

    for r in cache.repos:
        if r.repo_id == hf_repo:
            return round(r.size_on_disk / (1024**3), 2)

    return None


def list_cached_models(cache_dir: Path | None = None) -> list[DownloadStatus]:
    """List all models present in the HuggingFace cache.

    Args:
        cache_dir: Optional cache directory to scan.

    Returns:
        List of DownloadStatus for each cached model repo.
    """
    try:
        cache = scan_cache_dir(cache_dir=cache_dir)
    except Exception:
        return []

    results: list[DownloadStatus] = []
    for r in cache.repos:
        results.append(
            DownloadStatus(
                hf_repo=r.repo_id,
                local_path=Path(r.repo_path),
                cached=True,
                size_gb=round(r.size_on_disk / (1024**3), 2),
            )
        )

    return results


def delete_cached_model(hf_repo: str, cache_dir: Path | None = None) -> bool:
    """Delete a cached model from disk.

    Removes all revisions of the specified repo from the cache.

    Args:
        hf_repo: HuggingFace repository ID.
        cache_dir: Optional cache directory.

    Returns:
        True if the model was found and deleted, False if not cached.
    """
    try:
        cache = scan_cache_dir(cache_dir=cache_dir)
    except Exception:
        return False

    revision_hashes: list[str] = []
    for r in cache.repos:
        if r.repo_id == hf_repo:
            for rev in r.revisions:
                revision_hashes.append(rev.commit_hash)

    if not revision_hashes:
        return False

    strategy = cache.delete_revisions(*revision_hashes)
    strategy.execute()
    return True


def _get_cached_path(hf_repo: str, cache_dir: Path | None = None) -> Path | None:
    """Get the local path for a cached model.

    Args:
        hf_repo: HuggingFace repository ID.
        cache_dir: Optional cache directory.

    Returns:
        Path to the cached snapshot, or None if not cached.
    """
    try:
        cache = scan_cache_dir(cache_dir=cache_dir)
    except Exception:
        return None

    for r in cache.repos:
        if r.repo_id == hf_repo:
            # Return the path of the most recent revision snapshot
            for rev in sorted(r.revisions, key=lambda x: x.last_modified, reverse=True):
                return Path(rev.snapshot_path)

    return None
