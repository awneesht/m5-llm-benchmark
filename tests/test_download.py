"""Tests for the model download manager."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from macsmart.cli import cli
from macsmart.manager.download import (
    DownloadStatus,
    _get_cached_path,
    delete_cached_model,
    download_model,
    get_cached_size_gb,
    get_model_size_gb,
    is_model_cached,
    list_cached_models,
)


# ---------------------------------------------------------------------------
# Helpers to build mock HuggingFace cache structures
# ---------------------------------------------------------------------------


def _make_mock_revision(commit_hash: str = "abc123", snapshot_path: str = "/cache/snap", last_modified: float = 1000.0):
    rev = MagicMock()
    rev.commit_hash = commit_hash
    rev.snapshot_path = snapshot_path
    rev.last_modified = last_modified
    return rev


def _make_mock_repo(repo_id: str, size_on_disk: int = 5_000_000_000, repo_path: str = "/cache/repo", revisions=None):
    repo = MagicMock()
    repo.repo_id = repo_id
    repo.size_on_disk = size_on_disk
    repo.repo_path = repo_path
    repo.nb_files = 10
    repo.revisions = revisions or [_make_mock_revision()]
    return repo


def _make_mock_cache(repos: list | None = None):
    cache = MagicMock()
    cache.repos = repos or []
    cache.size_on_disk = sum(r.size_on_disk for r in cache.repos)
    return cache


# ---------------------------------------------------------------------------
# is_model_cached
# ---------------------------------------------------------------------------


class TestIsModelCached:
    """Tests for cache checking."""

    def test_cached_model_found(self) -> None:
        repo = _make_mock_repo("mlx-community/TestModel-4bit")
        cache = _make_mock_cache([repo])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            assert is_model_cached("mlx-community/TestModel-4bit") is True

    def test_model_not_cached(self) -> None:
        cache = _make_mock_cache([])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            assert is_model_cached("mlx-community/Missing-Model") is False

    def test_cache_scan_failure(self) -> None:
        with patch("macsmart.manager.download.scan_cache_dir", side_effect=OSError("fail")):
            assert is_model_cached("any/model") is False


# ---------------------------------------------------------------------------
# get_model_size_gb
# ---------------------------------------------------------------------------


class TestGetModelSizeGB:
    """Tests for remote model size lookup."""

    def test_returns_size(self) -> None:
        mock_info = MagicMock()
        sibling1 = MagicMock()
        sibling1.size = 2 * 1024**3  # 2 GB
        sibling2 = MagicMock()
        sibling2.size = 1 * 1024**3  # 1 GB
        mock_info.siblings = [sibling1, sibling2]

        with patch("macsmart.manager.download.repo_info", return_value=mock_info):
            size = get_model_size_gb("mlx-community/TestModel")

        assert size == 3.0

    def test_returns_none_on_failure(self) -> None:
        with patch("macsmart.manager.download.repo_info", side_effect=Exception("network error")):
            assert get_model_size_gb("bad/repo") is None

    def test_returns_none_for_zero_size(self) -> None:
        mock_info = MagicMock()
        sibling = MagicMock()
        sibling.size = 0
        mock_info.siblings = [sibling]

        with patch("macsmart.manager.download.repo_info", return_value=mock_info):
            assert get_model_size_gb("empty/repo") is None

    def test_handles_none_size_siblings(self) -> None:
        mock_info = MagicMock()
        s1 = MagicMock()
        s1.size = 1 * 1024**3
        s2 = MagicMock()
        s2.size = None  # Some siblings may lack size
        mock_info.siblings = [s1, s2]

        with patch("macsmart.manager.download.repo_info", return_value=mock_info):
            size = get_model_size_gb("partial/repo")

        assert size == 1.0


# ---------------------------------------------------------------------------
# get_cached_size_gb
# ---------------------------------------------------------------------------


class TestGetCachedSizeGB:
    """Tests for cached model size."""

    def test_returns_size_for_cached_model(self) -> None:
        repo = _make_mock_repo("test/model", size_on_disk=4 * 1024**3)
        cache = _make_mock_cache([repo])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            size = get_cached_size_gb("test/model")

        assert size == 4.0

    def test_returns_none_for_uncached(self) -> None:
        cache = _make_mock_cache([])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            assert get_cached_size_gb("missing/model") is None

    def test_returns_none_on_scan_failure(self) -> None:
        with patch("macsmart.manager.download.scan_cache_dir", side_effect=OSError):
            assert get_cached_size_gb("any/model") is None


# ---------------------------------------------------------------------------
# download_model
# ---------------------------------------------------------------------------


class TestDownloadModel:
    """Tests for model downloading."""

    def test_download_new_model(self) -> None:
        cache_empty = _make_mock_cache([])
        repo_after = _make_mock_repo("test/model", size_on_disk=3 * 1024**3)
        cache_after = _make_mock_cache([repo_after])

        scan_calls = [0]

        def mock_scan(cache_dir=None):
            scan_calls[0] += 1
            # First call in is_model_cached -> empty, rest -> has model
            if scan_calls[0] <= 1:
                return cache_empty
            return cache_after

        with (
            patch("macsmart.manager.download.scan_cache_dir", side_effect=mock_scan),
            patch("macsmart.manager.download.snapshot_download", return_value="/cache/snap") as mock_dl,
        ):
            status = download_model("test/model")

        assert isinstance(status, DownloadStatus)
        assert status.hf_repo == "test/model"
        assert status.local_path == Path("/cache/snap")
        assert status.cached is False
        mock_dl.assert_called_once()

    def test_returns_cached_without_download(self) -> None:
        rev = _make_mock_revision(snapshot_path="/cache/snap/abc")
        repo = _make_mock_repo("test/cached", size_on_disk=2 * 1024**3, revisions=[rev])
        cache = _make_mock_cache([repo])

        with (
            patch("macsmart.manager.download.scan_cache_dir", return_value=cache),
            patch("macsmart.manager.download.snapshot_download") as mock_dl,
        ):
            status = download_model("test/cached")

        assert status.cached is True
        assert status.local_path == Path("/cache/snap/abc")
        mock_dl.assert_not_called()

    def test_force_redownload(self) -> None:
        repo = _make_mock_repo("test/model", size_on_disk=2 * 1024**3)
        cache = _make_mock_cache([repo])

        with (
            patch("macsmart.manager.download.scan_cache_dir", return_value=cache),
            patch("macsmart.manager.download.snapshot_download", return_value="/new/path") as mock_dl,
        ):
            status = download_model("test/model", force=True)

        assert status.cached is False
        mock_dl.assert_called_once_with(repo_id="test/model", force_download=True)

    def test_cache_dir_override(self) -> None:
        cache_empty = _make_mock_cache([])
        repo_after = _make_mock_repo("test/model")
        cache_after = _make_mock_cache([repo_after])

        scan_calls = [0]

        def mock_scan(cache_dir=None):
            scan_calls[0] += 1
            if scan_calls[0] <= 1:
                return cache_empty
            return cache_after

        with (
            patch("macsmart.manager.download.scan_cache_dir", side_effect=mock_scan),
            patch("macsmart.manager.download.snapshot_download", return_value="/tmp/model") as mock_dl,
        ):
            status = download_model("test/model", cache_dir=Path("/tmp/cache"))

        mock_dl.assert_called_once_with(
            repo_id="test/model",
            force_download=False,
            cache_dir="/tmp/cache",
        )


# ---------------------------------------------------------------------------
# list_cached_models
# ---------------------------------------------------------------------------


class TestListCachedModels:
    """Tests for listing cached models."""

    def test_lists_all_cached(self) -> None:
        repos = [
            _make_mock_repo("model/a", size_on_disk=1 * 1024**3, repo_path="/cache/a"),
            _make_mock_repo("model/b", size_on_disk=2 * 1024**3, repo_path="/cache/b"),
        ]
        cache = _make_mock_cache(repos)

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            result = list_cached_models()

        assert len(result) == 2
        assert result[0].hf_repo == "model/a"
        assert result[0].size_gb == 1.0
        assert result[0].cached is True
        assert result[1].hf_repo == "model/b"
        assert result[1].size_gb == 2.0

    def test_empty_cache(self) -> None:
        cache = _make_mock_cache([])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            assert list_cached_models() == []

    def test_scan_failure_returns_empty(self) -> None:
        with patch("macsmart.manager.download.scan_cache_dir", side_effect=OSError):
            assert list_cached_models() == []


# ---------------------------------------------------------------------------
# delete_cached_model
# ---------------------------------------------------------------------------


class TestDeleteCachedModel:
    """Tests for deleting cached models."""

    def test_delete_existing_model(self) -> None:
        rev = _make_mock_revision(commit_hash="deadbeef")
        repo = _make_mock_repo("test/model", revisions=[rev])
        cache = _make_mock_cache([repo])

        mock_strategy = MagicMock()
        cache.delete_revisions = MagicMock(return_value=mock_strategy)

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            result = delete_cached_model("test/model")

        assert result is True
        cache.delete_revisions.assert_called_once_with("deadbeef")
        mock_strategy.execute.assert_called_once()

    def test_delete_nonexistent_model(self) -> None:
        cache = _make_mock_cache([])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            result = delete_cached_model("missing/model")

        assert result is False

    def test_delete_scan_failure(self) -> None:
        with patch("macsmart.manager.download.scan_cache_dir", side_effect=OSError):
            assert delete_cached_model("any/model") is False

    def test_delete_multiple_revisions(self) -> None:
        rev1 = _make_mock_revision(commit_hash="aaa111")
        rev2 = _make_mock_revision(commit_hash="bbb222")
        repo = _make_mock_repo("test/model", revisions=[rev1, rev2])
        cache = _make_mock_cache([repo])

        mock_strategy = MagicMock()
        cache.delete_revisions = MagicMock(return_value=mock_strategy)

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            result = delete_cached_model("test/model")

        assert result is True
        cache.delete_revisions.assert_called_once_with("aaa111", "bbb222")


# ---------------------------------------------------------------------------
# _get_cached_path
# ---------------------------------------------------------------------------


class TestGetCachedPath:
    """Tests for resolving cached model paths."""

    def test_returns_snapshot_path(self) -> None:
        rev = _make_mock_revision(snapshot_path="/cache/models--test--model/snapshots/abc123")
        repo = _make_mock_repo("test/model", revisions=[rev])
        cache = _make_mock_cache([repo])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            path = _get_cached_path("test/model")

        assert path == Path("/cache/models--test--model/snapshots/abc123")

    def test_returns_most_recent_revision(self) -> None:
        rev_old = _make_mock_revision(snapshot_path="/old", last_modified=100.0)
        rev_new = _make_mock_revision(snapshot_path="/new", last_modified=200.0)
        repo = _make_mock_repo("test/model", revisions=[rev_old, rev_new])
        cache = _make_mock_cache([repo])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            path = _get_cached_path("test/model")

        assert path == Path("/new")

    def test_returns_none_for_uncached(self) -> None:
        cache = _make_mock_cache([])

        with patch("macsmart.manager.download.scan_cache_dir", return_value=cache):
            assert _get_cached_path("missing/model") is None

    def test_returns_none_on_scan_failure(self) -> None:
        with patch("macsmart.manager.download.scan_cache_dir", side_effect=OSError):
            assert _get_cached_path("any/model") is None


# ---------------------------------------------------------------------------
# CLI: macsmart download
# ---------------------------------------------------------------------------


class TestDownloadCLI:
    """Tests for the `macsmart download` CLI command."""

    def test_download_new_model_rich(self) -> None:
        """Download command prints Rich output for a new model."""
        status = DownloadStatus(
            hf_repo="mlx-community/TestModel-4bit",
            local_path=Path("/cache/snap/abc"),
            cached=False,
            size_gb=4.5,
        )

        with patch("macsmart.manager.download.download_model", return_value=status):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "mlx-community/TestModel-4bit"])

        assert result.exit_code == 0
        assert "Downloading mlx-community/TestModel-4bit" in result.output

    def test_download_cached_model_rich(self) -> None:
        """Download command handles already-cached models."""
        status = DownloadStatus(
            hf_repo="mlx-community/TestModel-4bit",
            local_path=Path("/cache/snap/abc"),
            cached=True,
            size_gb=4.5,
        )

        with patch("macsmart.manager.download.download_model", return_value=status):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "mlx-community/TestModel-4bit"])

        assert result.exit_code == 0

    def test_download_json_output(self) -> None:
        """Download command with --json-output emits valid JSON."""
        status = DownloadStatus(
            hf_repo="mlx-community/TestModel-4bit",
            local_path=Path("/cache/snap/abc"),
            cached=False,
            size_gb=4.5,
        )

        with patch("macsmart.manager.download.download_model", return_value=status):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "mlx-community/TestModel-4bit", "--json-output"])

        assert result.exit_code == 0
        # First line is the "Downloading..." message; JSON follows
        lines = result.output.strip().split("\n")
        json_str = "\n".join(lines[1:])
        data = json.loads(json_str)
        assert data["hf_repo"] == "mlx-community/TestModel-4bit"
        assert data["cached"] is False
        assert data["size_gb"] == 4.5
        assert data["local_path"] == "/cache/snap/abc"

    def test_download_json_null_path(self) -> None:
        """JSON output shows null for local_path when None."""
        status = DownloadStatus(
            hf_repo="test/model",
            local_path=None,
            cached=False,
            size_gb=None,
        )

        with patch("macsmart.manager.download.download_model", return_value=status):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "test/model", "--json-output"])

        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        data = json.loads("\n".join(lines[1:]))
        assert data["local_path"] is None
        assert data["size_gb"] is None

    def test_download_force_flag(self) -> None:
        """--force flag is passed through to download_model."""
        status = DownloadStatus(
            hf_repo="test/model",
            local_path=Path("/cache/snap"),
            cached=False,
            size_gb=2.0,
        )

        with patch("macsmart.manager.download.download_model", return_value=status) as mock_dl:
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "test/model", "--force"])

        assert result.exit_code == 0
        mock_dl.assert_called_once_with("test/model", force=True)

    def test_download_failure(self) -> None:
        """Download command reports errors via ClickException."""
        with patch("macsmart.manager.download.download_model", side_effect=RuntimeError("network timeout")):
            runner = CliRunner()
            result = runner.invoke(cli, ["download", "bad/model"])

        assert result.exit_code != 0
        assert "Download failed" in result.output
        assert "network timeout" in result.output

    def test_download_missing_argument(self) -> None:
        """Download command requires a MODEL argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["download"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: macsmart models
# ---------------------------------------------------------------------------


class TestModelsCLI:
    """Tests for the `macsmart models` CLI command."""

    def test_models_empty(self) -> None:
        """Shows message when no models are cached."""
        with patch("macsmart.manager.download.list_cached_models", return_value=[]):
            runner = CliRunner()
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0
        assert "No models cached locally" in result.output

    def test_models_rich_output(self) -> None:
        """Models command prints Rich table when models exist."""
        cached = [
            DownloadStatus(hf_repo="model/a", local_path=Path("/a"), cached=True, size_gb=1.0),
            DownloadStatus(hf_repo="model/b", local_path=Path("/b"), cached=True, size_gb=2.5),
        ]

        with patch("macsmart.manager.download.list_cached_models", return_value=cached):
            runner = CliRunner()
            result = runner.invoke(cli, ["models"])

        assert result.exit_code == 0

    def test_models_json_output(self) -> None:
        """Models command with --json-output emits valid JSON array."""
        cached = [
            DownloadStatus(hf_repo="model/a", local_path=Path("/a"), cached=True, size_gb=1.0),
            DownloadStatus(hf_repo="model/b", local_path=Path("/b"), cached=True, size_gb=2.5),
        ]

        with patch("macsmart.manager.download.list_cached_models", return_value=cached):
            runner = CliRunner()
            result = runner.invoke(cli, ["models", "--json-output"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["hf_repo"] == "model/a"
        assert data[0]["size_gb"] == 1.0
        assert data[0]["local_path"] == "/a"
        assert data[1]["hf_repo"] == "model/b"
        assert data[1]["size_gb"] == 2.5

    def test_models_json_empty_returns_message(self) -> None:
        """JSON mode with no models still shows the text message, not JSON."""
        with patch("macsmart.manager.download.list_cached_models", return_value=[]):
            runner = CliRunner()
            result = runner.invoke(cli, ["models", "--json-output"])

        assert result.exit_code == 0
        assert "No models cached locally" in result.output


# ---------------------------------------------------------------------------
# CLI: macsmart delete
# ---------------------------------------------------------------------------


class TestDeleteCLI:
    """Tests for the `macsmart delete` CLI command."""

    def test_delete_with_yes_flag(self) -> None:
        """Delete command with -y skips confirmation."""
        with (
            patch("macsmart.manager.download.get_cached_size_gb", return_value=4.5),
            patch("macsmart.manager.download.delete_cached_model", return_value=True),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "test/model", "-y"])

        assert result.exit_code == 0
        assert "Deleted test/model from cache" in result.output

    def test_delete_with_confirmation(self) -> None:
        """Delete command prompts for confirmation and proceeds on 'y'."""
        with (
            patch("macsmart.manager.download.get_cached_size_gb", return_value=4.5),
            patch("macsmart.manager.download.delete_cached_model", return_value=True),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "test/model"], input="y\n")

        assert result.exit_code == 0
        assert "Deleted test/model from cache" in result.output

    def test_delete_confirmation_abort(self) -> None:
        """Delete command aborts when user declines confirmation."""
        with patch("macsmart.manager.download.get_cached_size_gb", return_value=4.5):
            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "test/model"], input="n\n")

        assert result.exit_code != 0

    def test_delete_model_not_cached(self) -> None:
        """Delete command errors when model is not cached."""
        with patch("macsmart.manager.download.get_cached_size_gb", return_value=None):
            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "missing/model", "-y"])

        assert result.exit_code != 0
        assert "not cached locally" in result.output

    def test_delete_failure(self) -> None:
        """Delete command errors when deletion fails."""
        with (
            patch("macsmart.manager.download.get_cached_size_gb", return_value=2.0),
            patch("macsmart.manager.download.delete_cached_model", return_value=False),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "test/model", "-y"])

        assert result.exit_code != 0
        assert "Failed to delete" in result.output

    def test_delete_missing_argument(self) -> None:
        """Delete command requires a MODEL argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete"])

        assert result.exit_code != 0

    def test_delete_shows_size_in_prompt(self) -> None:
        """Confirmation prompt includes the model size."""
        with (
            patch("macsmart.manager.download.get_cached_size_gb", return_value=3.75),
            patch("macsmart.manager.download.delete_cached_model", return_value=True),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "test/model"], input="y\n")

        assert result.exit_code == 0
        assert "3.75 GB" in result.output
