"""Tests for translation algorithms."""

import asyncio
from datetime import datetime
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import (
    translate_page_by_page,
    translate_sliding_window,
    repair_seams,
    create_windows,
    merge_chunks,
)
from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import TranslationConfig, ModelType


@pytest.fixture
def mock_translator():
    """Create a mock translator."""
    translator = AsyncMock(spec=ModelInterface)

    async def mock_translate(request, stream=False):
        if hasattr(request, "content") and request.content == "Page 2":
            if getattr(mock_translate, "should_fail", False):
                raise TranslationError("Failed to translate page 2")
        return TranslationResponse(
            text="Translated text",
            tokens_used=10,
            cost=0.001,
            time_taken=0.5,
        )

    translator.translate = AsyncMock(side_effect=mock_translate)
    return translator


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock checkpoint manager."""
    manager = AsyncMock(spec=CheckpointManager)

    async def mock_load_checkpoint(*args, **kwargs):
        return TranslationState(
            source_lang="en",
            target_lang="fr",
            algorithm="page",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={1: "Translated page 1"},
            token_usage=10,
            cost=0.001,
            time_taken=1.0,
        )

    manager.load_checkpoint = AsyncMock(side_effect=mock_load_checkpoint)
    manager.save_checkpoint = AsyncMock()
    manager.cleanup_old_checkpoints = AsyncMock()
    return manager


@pytest.fixture
def test_content():
    """Create test document content."""
    pages = ["Page 1", "Page 2", "Page 3"]
    return DocumentContent(
        pages=pages,
        content_type="text/plain",
        metadata={"test": "metadata"},
    )


@pytest.fixture
def test_config(tmp_path):
    """Create test translation config with checkpointing."""
    return TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=tmp_path / "test.txt",
        checkpoint_dir=tmp_path,
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )


async def test_translate_page_by_page_with_checkpointing(
    test_content,
    test_config,
    mock_translator,
    mock_checkpoint_manager,
):
    """Test page-by-page translation with checkpointing."""
    # Translate document
    result = await translate_page_by_page(
        test_content,
        test_config,
        mock_translator,
        checkpoint_manager=mock_checkpoint_manager,
    )

    # Verify checkpoint loading
    mock_checkpoint_manager.load_checkpoint.assert_called_once_with(
        test_config.input_file,
    )

    # Count translation calls for actual pages (excluding seam repair)
    page_translation_calls = [
        call
        for call in mock_translator.translate.call_args_list
        if "Page" in str(call[0][0].content)
    ]
    assert len(page_translation_calls) == 2  # Only pages 2 and 3

    # Verify checkpoint saving
    assert mock_checkpoint_manager.save_checkpoint.call_count >= 2
    saved_states = [
        call[0][0] for call in mock_checkpoint_manager.save_checkpoint.call_args_list
    ]
    assert len(saved_states[-1].completed_pages) == 3  # All pages completed

    # Verify cleanup
    mock_checkpoint_manager.cleanup_old_checkpoints.assert_called_once_with(
        test_config.input_file,
    )

    # Verify final result
    assert result.text  # Result after seam repair
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0


async def test_translate_sliding_window_with_checkpointing(
    test_content,
    test_config,
    mock_translator,
    mock_checkpoint_manager,
):
    """Test sliding window translation with checkpointing."""
    # Create sliding window config
    sliding_config = test_config.model_copy(
        update={
            "algorithm": "sliding-window",
            "window_size": 10,  # Small window for testing
            "overlap_size": 3,  # Small overlap for testing
        }
    )

    # Create test content with known overlapping sections
    test_content = DocumentContent(
        pages=["This is a test content with some overlap text here"],
        content_type="text/plain",
        metadata={"test": "metadata"},
    )

    # Set up mock translator with proper async response
    async def mock_translate(request, stream=False):
        # Simulate translation by adding a prefix
        translated = "TR: " + request.content
        return TranslationResponse(
            text=translated,
            tokens_used=len(request.content.split()),
            cost=0.001,
            time_taken=0.1,
        )

    mock_translator.translate = AsyncMock(side_effect=mock_translate)

    # Translate document
    result = await translate_sliding_window(
        test_content,
        sliding_config,
        mock_translator,
        checkpoint_manager=mock_checkpoint_manager,
    )

    # Verify basic result properties
    assert result.text
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0

    # Verify checkpoint handling
    assert mock_checkpoint_manager.save_checkpoint.called
    saved_states = [
        call[0][0] for call in mock_checkpoint_manager.save_checkpoint.call_args_list
    ]
    assert len(saved_states) > 0
    assert all(state.algorithm == "sliding-window" for state in saved_states)

    # Verify translation content
    assert "TR: " in result.text  # Our mock translation prefix is present
    assert len(result.text.split("TR: ")) > 1  # Multiple chunks were translated


async def test_translation_without_checkpointing(
    test_content,
    mock_translator,
):
    """Test translation without checkpointing enabled."""
    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        checkpoint_dir=None,
    )

    # Set up mock translator with proper response
    async def mock_translate(request, stream=False):
        return TranslationResponse(
            text="Translated text",
            tokens_used=10,
            cost=0.001,
            time_taken=0.5,
        )

    mock_translator.translate = AsyncMock(side_effect=mock_translate)

    # Translate document
    result = await translate_page_by_page(
        test_content,
        config,
        mock_translator,
    )

    # Count translation calls for actual pages (excluding seam repair)
    page_translation_calls = [
        call
        for call in mock_translator.translate.call_args_list
        if "Page" in str(call[0][0].content)
    ]
    assert len(page_translation_calls) == 3  # All three pages

    # Verify result
    assert result.text
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0


async def test_translation_with_failed_pages(
    test_content,
    test_config,
    mock_translator,
    mock_checkpoint_manager,
):
    """Test translation with some failed pages."""
    # Create a new mock translator that fails on page 2
    translator = AsyncMock(spec=ModelInterface)

    async def mock_translate(request, stream=False):
        if hasattr(request, "content") and request.content == "Page 2":
            raise TranslationError("Failed to translate page 2")
        return TranslationResponse(
            text="Translated text",
            tokens_used=10,
            cost=0.001,
            time_taken=0.5,
        )

    translator.translate = AsyncMock(side_effect=mock_translate)

    # Translate document - should succeed but mark page 2 as failed
    result = await translate_page_by_page(
        test_content,
        test_config,
        translator,
        checkpoint_manager=mock_checkpoint_manager,
    )

    # Verify checkpoint saving includes failed page
    assert mock_checkpoint_manager.save_checkpoint.called
    saved_states = [
        call[0][0] for call in mock_checkpoint_manager.save_checkpoint.call_args_list
    ]
    assert any(2 in state.failed_pages for state in saved_states)
    assert any(
        1 in state.completed_pages for state in saved_states
    )  # Page 1 should succeed

    # Verify result contains successful translations
    assert result.text
    assert result.tokens_used > 0
    assert result.cost > 0
    assert result.time_taken > 0


async def test_translation_all_pages_fail(
    test_content,
    test_config,
    mock_translator,
):
    """Test handling of all pages failing."""
    # Make all translations fail
    mock_translator.translate.side_effect = TranslationError(
        "No pages were successfully translated"
    )

    # Verify translation fails with appropriate error
    with pytest.raises(TranslationError) as exc_info:
        await translate_page_by_page(
            test_content,
            test_config,
            mock_translator,
        )

    # Check error message
    assert "No pages were successfully translated" in str(exc_info.value)
    assert "Failed pages: [1, 2, 3]" in str(exc_info.value)


# Helper function tests
async def test_repair_seams(mock_translator):
    """Test seam repair functionality."""
    pages = ["First page content.", "Second page content.", "Third page content."]
    config = TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="page",
        input_file=Path("test.txt"),
        page_seam_overlap=10,
    )

    result = await repair_seams(pages, config, mock_translator)
    assert result
    assert isinstance(result, str)


def test_create_windows():
    """Test window creation for sliding window translation."""
    text = "This is a test text for window creation."
    windows = create_windows(text, window_size=10, overlap_size=3)

    assert len(windows) > 0
    assert all(isinstance(w, str) for w in windows)
    assert len(windows[0]) <= 10  # Should not exceed window size


def test_merge_chunks():
    """Test chunk merging for sliding window translation."""
    chunks = [
        "First chunk with overlap",
        "with overlap and more",
        "and more final content",
    ]

    # Test with smaller overlap to ensure proper merging
    result = merge_chunks(chunks, overlap_size=12)
    assert result
    assert isinstance(result, str)

    # Verify content preservation
    assert "First chunk" in result
    assert "and more" in result
    assert "final content" in result

    # Verify proper overlap handling
    assert result.count("with overlap") == 1  # Should only appear once after merging
    assert result.count("and more") == 1  # Should only appear once after merging

    # Test with no overlap
    no_overlap_result = merge_chunks(chunks, overlap_size=0)
    assert no_overlap_result == "".join(chunks)

    # Test with full overlap
    full_overlap_result = merge_chunks(chunks, overlap_size=len(chunks[0]))
    assert len(full_overlap_result) > 0
    assert "First chunk" in full_overlap_result
    assert "final content" in full_overlap_result
