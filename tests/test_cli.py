"""Tests for the CLI interface."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tinbox.cli import app
from tinbox.core.cost import CostEstimate
from tinbox.core.processor import DocumentContent
from tinbox.core.types import FileType, ModelType, TranslationResult, TranslationConfig
from tinbox.core.translation.interface import ModelInterface


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_console():
    """Create a mock console for testing."""
    return MagicMock(spec=Console)


@pytest.fixture
def mock_cost_estimate():
    """Create a mock cost estimate."""
    return CostEstimate(
        estimated_tokens=1000,
        estimated_cost=0.03,
        estimated_time=60.0,
        warnings=[],
    )


@pytest.fixture
def mock_translation_result():
    """Create a mock translation result."""
    return TranslationResult(
        text="Translated text",
        tokens_used=1000,
        cost=0.03,
        time_taken=30.0,
    )


@pytest.fixture
def mock_document_content():
    """Create a mock document content."""
    return DocumentContent(
        pages=["Test content"],
        content_type="text/plain",
        metadata={"test": "metadata"},
    )


@pytest.fixture
def mock_model_interface():
    """Create a mock model interface."""
    interface = AsyncMock(spec=ModelInterface)
    interface.translate = AsyncMock(
        return_value=TranslationResult(
            text="Translated text",
            tokens_used=1000,
            cost=0.03,
            time_taken=30.0,
        )
    )
    return interface


def test_version(cli_runner):
    """Test version command."""
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Tinbox version:" in result.stdout


def test_translate_basic(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test basic translation command."""
    # Create a temporary test file
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--model",
                    "anthropic:claude-3-sonnet",
                ],
            )
            assert result.exit_code == 0

    finally:
        # Clean up test file
        test_file.unlink()


def test_translate_dry_run(cli_runner, mock_cost_estimate):
    """Test dry run mode."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch("tinbox.cli.console"),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--dry-run",
                ],
            )
            assert result.exit_code == 0

    finally:
        test_file.unlink()


def test_translate_with_warnings(cli_runner):
    """Test translation with cost warnings."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    # Create cost estimate with warnings
    estimate = CostEstimate(
        estimated_tokens=100_000,
        estimated_cost=3.0,
        estimated_time=600.0,
        warnings=["Large document detected (100,000 tokens)"],
    )

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=estimate),
            patch("tinbox.cli.console.print") as mock_print,
            patch("typer.confirm", return_value=False),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                ],
            )
            assert result.exit_code == 1
            # Check if the cancellation message was printed
            mock_print.assert_any_call("\nTranslation cancelled.")

    finally:
        test_file.unlink()


def test_translate_with_output_file(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test translation with output file."""
    input_file = Path("input.txt")
    output_file = Path("output.txt")
    input_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(input_file),
                    "--output",
                    str(output_file),
                    "--to",
                    "es",
                ],
            )
            assert result.exit_code == 0
            assert output_file.exists()
            assert output_file.read_text() == mock_translation_result.text

    finally:
        input_file.unlink()
        if output_file.exists():
            output_file.unlink()


def test_translate_invalid_file(cli_runner):
    """Test translation with non-existent file."""
    result = cli_runner.invoke(
        app,
        [
            "translate",
            "nonexistent.txt",
            "--to",
            "es",
        ],
    )
    assert result.exit_code != 0


def test_translate_verbose_mode(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test translation in verbose mode."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--verbose",
                ],
            )
            assert result.exit_code == 0

    finally:
        test_file.unlink()


def test_translate_force_mode(
    cli_runner,
    mock_cost_estimate,
    mock_translation_result,
    mock_document_content,
    mock_model_interface,
):
    """Test translation in force mode."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=mock_cost_estimate),
            patch(
                "tinbox.cli.translate_document",
                new_callable=AsyncMock,
                return_value=mock_translation_result,
            ),
            patch(
                "tinbox.cli.load_document",
                new_callable=AsyncMock,
                return_value=mock_document_content,
            ),
            patch("tinbox.cli.create_translator", return_value=mock_model_interface),
            patch("tinbox.cli.console"),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--force",
                ],
            )
            assert result.exit_code == 0

    finally:
        test_file.unlink()


def test_translate_max_cost(cli_runner):
    """Test translation with max cost threshold."""
    test_file = Path("test.txt")
    test_file.write_text("Test content")

    # Create cost estimate exceeding threshold
    estimate = CostEstimate(
        estimated_tokens=100_000,
        estimated_cost=3.0,
        estimated_time=600.0,
        warnings=[
            "Estimated cost ($3.00) exceeds maximum threshold ($1.00)",
        ],
    )

    try:
        with (
            patch("tinbox.cli.estimate_cost", return_value=estimate),
            patch("tinbox.cli.console.print") as mock_print,
            patch("typer.confirm", return_value=False),
        ):
            result = cli_runner.invoke(
                app,
                [
                    "translate",
                    str(test_file),
                    "--to",
                    "es",
                    "--max-cost",
                    "1.0",
                ],
            )
            assert result.exit_code == 1
            # Check if the cancellation message was printed
            mock_print.assert_any_call("\nTranslation cancelled.")

    finally:
        test_file.unlink()
