"""Tests for cost estimation functionality."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from tinbox.core.cost import (
    CostEstimate,
    CostLevel,
    estimate_cost,
    estimate_document_tokens,
    get_cost_level,
)
from tinbox.core.types import FileType, ModelType


def test_cost_level_thresholds():
    """Test cost level threshold calculations."""
    assert get_cost_level(0.5) == CostLevel.LOW
    assert get_cost_level(2.0) == CostLevel.MEDIUM
    assert get_cost_level(10.0) == CostLevel.HIGH
    assert get_cost_level(25.0) == CostLevel.VERY_HIGH


@pytest.mark.parametrize(
    "file_type,content,expected_tokens",
    [
        (
            FileType.PDF,
            b"PDF content",  # Mock PDF with 2 pages
            1000,  # 2 pages * 500 tokens per page
        ),
        (
            FileType.DOCX,
            "This is a test document",  # 5 words
            7,  # 5 words * 1.3 tokens per word, rounded up
        ),
        (
            FileType.TXT,
            "This is a test",  # 16 characters
            4,  # 16 characters / 4, rounded up
        ),
    ],
)
def test_estimate_document_tokens(tmp_path, file_type, content, expected_tokens):
    """Test token estimation for different file types."""
    file_path = tmp_path / f"test.{file_type.value}"

    if file_type == FileType.PDF:
        # Create a mock PDF file
        file_path.write_bytes(content)
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]  # 2 pages
        with patch("pypdf.PdfReader", return_value=mock_pdf) as mock_reader:
            mock_reader.return_value = mock_pdf
            tokens = estimate_document_tokens(file_path)
            assert tokens == expected_tokens
    elif file_type == FileType.DOCX:
        # Create a mock DOCX file
        file_path.write_bytes(b"dummy docx content")
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text=content)]
        with patch("docx.Document", return_value=mock_doc):
            tokens = estimate_document_tokens(file_path)
            assert tokens == expected_tokens
    else:  # TXT
        file_path.write_text(content)
        tokens = estimate_document_tokens(file_path)
        assert tokens == expected_tokens


@pytest.mark.parametrize(
    "model,tokens,expected_cost,expected_warnings",
    [
        (
            ModelType.OPENAI,
            100_000,
            1.125,  # 100K tokens * ($0.00125 input + $0.01 output) per 1K tokens = 100 * 0.01125 = 1.125
            [
                "Large document detected (100,000 tokens). Consider using Ollama for better performance and no cost."
            ],
        ),
        (
            ModelType.ANTHROPIC,
            25_000,
            0.45,  # 25K tokens * ($0.003 input + $0.015 output) per 1K tokens = 25 * 0.018 = 0.45
            [],
        ),
        (
            ModelType.OLLAMA,
            200_000,
            0.0,  # Local model, no cost
            [],
        ),
    ],
)
def test_estimate_cost(tmp_path, model, tokens, expected_cost, expected_warnings):
    """Test cost estimation for different models and token counts."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test content")

    with patch(
        "tinbox.core.cost.estimate_document_tokens",
        return_value=tokens,
    ):
        estimate = estimate_cost(file_path, model)
        assert estimate.estimated_tokens == tokens
        assert (
            abs(estimate.estimated_cost - expected_cost) < 0.001
        )  # Allow small floating-point differences
        assert all(
            any(expected in warning for warning in estimate.warnings)
            for expected in expected_warnings
        )


def test_cost_threshold_warning():
    """Test cost threshold warning."""
    file_path = Path("test.txt")
    max_cost = 0.5

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=100_000):
        estimate = estimate_cost(
            file_path,
            ModelType.OPENAI,
            max_cost=max_cost,
        )
        assert any(
            f"Estimated cost ($1.12) exceeds maximum threshold (${max_cost:.2f})"
            in warning
            for warning in estimate.warnings
        )


def test_ollama_suggestion():
    """Test Ollama suggestion for large documents."""
    file_path = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=100_000):
        estimate = estimate_cost(file_path, ModelType.OPENAI)
        assert any(
            "Consider using Ollama for better performance and no cost" in warning
            for warning in estimate.warnings
        )
