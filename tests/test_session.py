"""Tests for the Session manager and task-aware routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from macsmart.manager.session import (
    ModelSwappedError,
    Session,
    SessionConfig,
    SessionState,
)
from macsmart.recommender.task_router import detect_task_with_confidence


# ---------------------------------------------------------------------------
# detect_task_with_confidence tests
# ---------------------------------------------------------------------------


class TestDetectTaskWithConfidence:
    """Tests for confidence-scored task detection."""

    def test_coding_prompt(self) -> None:
        task, conf = detect_task_with_confidence("Write a Python function to sort a list")
        assert task == "coding"
        assert conf > 0.0

    def test_writing_prompt(self) -> None:
        task, conf = detect_task_with_confidence("Write an essay about climate change")
        assert task == "writing"
        assert conf > 0.0

    def test_summarization_prompt(self) -> None:
        task, conf = detect_task_with_confidence("Summarize this article")
        assert task == "summarization"
        assert conf > 0.0

    def test_chat_prompt(self) -> None:
        task, conf = detect_task_with_confidence("Hello, how are you?")
        assert task == "chat"
        assert conf > 0.0

    def test_general_fallback(self) -> None:
        task, conf = detect_task_with_confidence("What is the capital of France?")
        assert task == "general"
        assert conf == 0.0

    def test_higher_confidence_with_more_keywords(self) -> None:
        """More keywords matching should yield higher confidence."""
        _, conf_single = detect_task_with_confidence("Write a function")
        _, conf_multi = detect_task_with_confidence(
            "Write a Python function to implement a class using an algorithm"
        )
        assert conf_multi > conf_single

    def test_returns_tuple(self) -> None:
        result = detect_task_with_confidence("hello")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_confidence_range(self) -> None:
        """Confidence should always be between 0 and 1."""
        for prompt in [
            "Write code",
            "Summarize this",
            "Hello there",
            "Random stuff",
            "Debug the JavaScript function and refactor the Python class",
        ]:
            _, conf = detect_task_with_confidence(prompt)
            assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# SessionConfig tests
# ---------------------------------------------------------------------------


class TestSessionConfig:
    """Tests for SessionConfig defaults."""

    def test_defaults(self) -> None:
        config = SessionConfig(prompt="hello")
        assert config.prompt == "hello"
        assert config.task is None
        assert config.model is None
        assert config.max_tokens == 256
        assert config.memory_override is None
        assert config.auto_swap is False

    def test_custom_values(self) -> None:
        config = SessionConfig(
            prompt="test",
            task="coding",
            model="mlx-community/Model-4bit",
            max_tokens=128,
            memory_override=8.0,
            auto_swap=True,
        )
        assert config.task == "coding"
        assert config.model == "mlx-community/Model-4bit"
        assert config.max_tokens == 128
        assert config.memory_override == 8.0
        assert config.auto_swap is True


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------


def _mock_recommendation(hf_repo: str = "mlx-community/Test-4bit") -> MagicMock:
    rec = MagicMock()
    rec.hf_repo = hf_repo
    return rec


class TestSessionSelectModel:
    """Tests for Session.select_model()."""

    def test_explicit_model_skips_detection(self) -> None:
        config = SessionConfig(prompt="hello", model="mlx-community/Explicit-4bit")
        session = Session(config=config)
        result = session.select_model()
        assert result == "mlx-community/Explicit-4bit"
        assert session.state == SessionState.MODEL_SELECTED

    @patch("macsmart.manager.session.recommend_models")
    @patch("macsmart.manager.session.get_available_for_llm", return_value=8.0)
    def test_auto_select_coding(self, mock_mem, mock_recs) -> None:
        mock_recs.return_value = [_mock_recommendation()]
        config = SessionConfig(prompt="Write a Python function")
        session = Session(config=config)
        session.select_model()
        assert session.selected_task == "coding"
        assert session.task_confidence > 0.0
        assert session.selected_model == "mlx-community/Test-4bit"

    @patch("macsmart.manager.session.recommend_models")
    @patch("macsmart.manager.session.get_available_for_llm", return_value=8.0)
    def test_auto_select_with_task_override(self, mock_mem, mock_recs) -> None:
        mock_recs.return_value = [_mock_recommendation()]
        config = SessionConfig(prompt="hello", task="writing")
        session = Session(config=config)
        session.select_model()
        assert session.selected_task == "writing"
        assert session.task_confidence == 1.0

    @patch("macsmart.manager.session.recommend_models", return_value=[])
    @patch("macsmart.manager.session.get_available_for_llm", return_value=1.0)
    def test_no_models_found_raises(self, mock_mem, mock_recs) -> None:
        config = SessionConfig(prompt="hello")
        session = Session(config=config)
        with pytest.raises(RuntimeError, match="No models found"):
            session.select_model()
        assert session.state == SessionState.ERROR

    @patch("macsmart.manager.session.recommend_models")
    def test_memory_override(self, mock_recs) -> None:
        mock_recs.return_value = [_mock_recommendation()]
        config = SessionConfig(prompt="What is the capital of France?", memory_override=12.0)
        session = Session(config=config)
        session.select_model()
        mock_recs.assert_called_once_with(12.0, task="general")

    def test_invalid_task_raises(self) -> None:
        config = SessionConfig(prompt="hello", task="dancing")
        session = Session(config=config)
        with pytest.raises(ValueError, match="Unknown task type"):
            session.select_model()


class TestSessionLoad:
    """Tests for Session.load()."""

    def test_load_without_select_raises(self) -> None:
        config = SessionConfig(prompt="hello")
        session = Session(config=config)
        with pytest.raises(RuntimeError, match="No model selected"):
            session.load()

    @patch("macsmart.manager.session.load_model", return_value=("mock_model", "mock_tok"))
    @patch("macsmart.manager.session.is_model_cached", return_value=True)
    def test_load_cached_model(self, mock_cached, mock_load) -> None:
        config = SessionConfig(prompt="hello", model="mlx-community/Test-4bit")
        session = Session(config=config)
        session.select_model()
        session.load()
        assert session.state == SessionState.LOADED
        mock_load.assert_called_once_with("mlx-community/Test-4bit")

    @patch("macsmart.manager.session.load_model", return_value=("mock_model", "mock_tok"))
    @patch("macsmart.manager.session.download_model")
    @patch("macsmart.manager.session.is_model_cached", return_value=False)
    def test_load_downloads_if_not_cached(self, mock_cached, mock_dl, mock_load) -> None:
        config = SessionConfig(prompt="hello", model="mlx-community/Test-4bit")
        session = Session(config=config)
        session.select_model()
        session.load()
        mock_dl.assert_called_once_with("mlx-community/Test-4bit")


class TestSessionGenerate:
    """Tests for Session.generate()."""

    def test_generate_without_load_raises(self) -> None:
        config = SessionConfig(prompt="hello", model="test")
        session = Session(config=config)
        session.select_model()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            session.generate()

    @patch("macsmart.manager.session.generate")
    @patch("macsmart.manager.session.load_model", return_value=("m", "t"))
    @patch("macsmart.manager.session.is_model_cached", return_value=True)
    def test_generate_returns_result(self, mock_cached, mock_load, mock_gen) -> None:
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_gen.return_value = mock_result

        config = SessionConfig(prompt="hello", model="test")
        session = Session(config=config)
        session.select_model()
        session.load()
        result = session.generate()
        assert result.text == "Hello world"

    @patch("macsmart.manager.session.generate")
    @patch("macsmart.manager.session.load_model", return_value=("m", "t"))
    @patch("macsmart.manager.session.is_model_cached", return_value=True)
    def test_swap_pending_raises(self, mock_cached, mock_load, mock_gen) -> None:
        mock_gen.return_value = MagicMock()
        config = SessionConfig(prompt="hello", model="test")
        session = Session(config=config)
        session.select_model()
        session.load()
        session.request_swap("new-model")
        with pytest.raises(ModelSwappedError):
            session.generate()


class TestSessionUnload:
    """Tests for Session.unload()."""

    @patch("macsmart.manager.session.unload_model")
    @patch("macsmart.manager.session.load_model", return_value=("m", "t"))
    @patch("macsmart.manager.session.is_model_cached", return_value=True)
    def test_unload_resets_state(self, mock_cached, mock_load, mock_unload) -> None:
        config = SessionConfig(prompt="hello", model="test")
        session = Session(config=config)
        session.select_model()
        session.load()
        session.unload()
        assert session.state == SessionState.IDLE

    def test_unload_when_not_loaded(self) -> None:
        """Unloading without a model should not error."""
        config = SessionConfig(prompt="hello")
        session = Session(config=config)
        session.unload()
        assert session.state == SessionState.IDLE


class TestSessionSwapTo:
    """Tests for Session.swap_to()."""

    @patch("macsmart.manager.session.unload_model")
    @patch("macsmart.manager.session.load_model", return_value=("m", "t"))
    @patch("macsmart.manager.session.is_model_cached", return_value=True)
    def test_swap_to_records_history(self, mock_cached, mock_load, mock_unload) -> None:
        config = SessionConfig(prompt="hello", model="old-model")
        session = Session(config=config)
        session.select_model()
        session.load()
        session.swap_to("new-model")
        assert len(session.swap_history) == 1
        assert session.swap_history[0]["from"] == "old-model"
        assert session.swap_history[0]["to"] == "new-model"
        assert session.selected_model == "new-model"


class TestSessionRun:
    """Tests for the full Session.run() lifecycle."""

    @patch("macsmart.manager.session.unload_model")
    @patch("macsmart.manager.session.generate")
    @patch("macsmart.manager.session.load_model", return_value=("m", "t"))
    @patch("macsmart.manager.session.is_model_cached", return_value=True)
    def test_run_full_lifecycle(self, mock_cached, mock_load, mock_gen, mock_unload) -> None:
        mock_result = MagicMock()
        mock_result.text = "output"
        mock_gen.return_value = mock_result

        config = SessionConfig(prompt="hello", model="test-model")
        session = Session(config=config)
        result = session.run()
        assert result.text == "output"
        assert session.state == SessionState.IDLE
        mock_load.assert_called_once()
        mock_gen.assert_called_once()
        mock_unload.assert_called_once()


# ---------------------------------------------------------------------------
# CLI run command tests
# ---------------------------------------------------------------------------


class TestRunCLI:
    """Tests for the `macsmart run` CLI command."""

    def test_run_json_output(self) -> None:
        mock_result = MagicMock()
        mock_result.text = "Generated text"
        mock_result.ttft_ms = 100.0
        mock_result.tokens_per_sec = 40.0
        mock_result.prompt_tokens = 10
        mock_result.generation_tokens = 50
        mock_result.peak_memory_gb = 3.5

        with (
            patch("macsmart.manager.session.is_model_cached", return_value=True),
            patch("macsmart.manager.session.load_model", return_value=("m", "t")),
            patch("macsmart.manager.session.generate", return_value=mock_result),
            patch("macsmart.manager.session.unload_model"),
        ):
            from macsmart.cli import cli

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["run", "hello world", "--model", "test-model", "--json-output"],
            )

        assert result.exit_code == 0, result.output
        import json

        data = json.loads(result.output)
        assert data["text"] == "Generated text"
        assert data["model"] == "test-model"

    def test_run_help(self) -> None:
        from macsmart.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "PROMPT" in result.output
        assert "--task" in result.output
        assert "--model" in result.output
        assert "--auto-swap" in result.output

    def test_run_appears_in_help(self) -> None:
        from macsmart.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "run" in result.output
