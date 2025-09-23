"""Tests for context-aware translation algorithm."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tinbox.core.processor import DocumentContent
from tinbox.core.translation.algorithms import (
    build_translation_context,
    smart_text_split,
    translate_context_aware,
)
from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationResponse,
)
from tinbox.core.types import ModelType, TranslationConfig


@pytest.fixture
def mock_translator():
    """Create a mock translator for context-aware testing."""
    translator = AsyncMock(spec=ModelInterface)

    async def mock_translate(request, stream=False):
        # Extract the content between [TRANSLATE_THIS] tags
        content = request.content
        if "[TRANSLATE_THIS]" in content and "[/TRANSLATE_THIS]" in content:
            start = content.find("[TRANSLATE_THIS]") + len("[TRANSLATE_THIS]")
            end = content.find("[/TRANSLATE_THIS]")
            actual_content = content[start:end].strip()
            return TranslationResponse(
                text=f"Translated: {actual_content}",
                tokens_used=len(actual_content.split()) * 2,  # Simulate token usage
                cost=0.001,
                time_taken=0.1,
            )
        else:
            # Fallback for content without tags
            return TranslationResponse(
                text=f"Translated: {content}",
                tokens_used=len(content.split()) * 2,
                cost=0.001,
                time_taken=0.1,
            )

    translator.translate = AsyncMock(side_effect=mock_translate)
    return translator


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock checkpoint manager."""
    manager = AsyncMock(spec=CheckpointManager)
    manager.load = AsyncMock(return_value=None)  # No existing checkpoint by default
    manager.save = AsyncMock()
    manager.cleanup_old_checkpoints = AsyncMock()
    return manager


@pytest.fixture
def context_aware_config(tmp_path):
    """Create test configuration for context-aware algorithm."""
    return TranslationConfig(
        source_lang="en",
        target_lang="fr",
        model=ModelType.ANTHROPIC,
        model_name="claude-3-sonnet",
        algorithm="context-aware",
        input_file=tmp_path / "test.txt",
        context_size=50,  # Small size for testing
        custom_split_token=None,
        checkpoint_dir=tmp_path,
        checkpoint_frequency=1,
        resume_from_checkpoint=True,
    )


class TestSmartTextSplit:
    """Test smart text splitting functionality."""

    def test_split_empty_text(self):
        """Test splitting empty text."""
        result = smart_text_split("", 100)
        assert result == []

    def test_split_invalid_target_size(self):
        """Test splitting with invalid target size."""
        with pytest.raises(ValueError, match="target_size must be positive"):
            smart_text_split("test", 0)

        with pytest.raises(ValueError, match="target_size must be positive"):
            smart_text_split("test", -1)

    def test_split_text_smaller_than_target(self):
        """Test splitting text smaller than target size."""
        text = "Short text"
        result = smart_text_split(text, 100)
        assert result == [text]

    def test_split_by_custom_token(self):
        """Test splitting by custom split token."""
        text = "First part|||Second part|||Third part"
        result = smart_text_split(text, 100, custom_split_token="|||")
        expected = ["First part", "Second part", "Third part"]
        assert result == expected

    def test_split_by_custom_token_ignores_target_size(self):
        """Test that custom token splitting ignores target size."""
        text = "Very long first part that exceeds target|||Short second"
        result = smart_text_split(text, 10, custom_split_token="|||")
        expected = ["Very long first part that exceeds target", "Short second"]
        assert result == expected

    def test_split_by_paragraphs(self):
        """Test splitting by paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph."
        result = smart_text_split(text, 30)
        assert len(result) >= 2
        assert "First paragraph." in result[0]
        assert "Second paragraph" in result[1]

    def test_split_by_sentences(self):
        """Test splitting by sentence endings."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        result = smart_text_split(text, 25)
        assert len(result) >= 2
        assert any("First sentence." in chunk for chunk in result)
        assert any("Second sentence!" in chunk for chunk in result)

    def test_split_by_line_breaks(self):
        """Test splitting by line breaks."""
        text = "First line\nSecond line\nThird line\nFourth line"
        result = smart_text_split(text, 20)
        assert len(result) >= 2
        assert "First line" in result[0]
        assert "Second line" in result[0] or "Second line" in result[1]

    def test_split_by_clause_boundaries(self):
        """Test splitting by clause boundaries."""
        text = "First clause; second clause: third clause, fourth clause"
        result = smart_text_split(text, 25)
        assert len(result) >= 2
        assert "First clause;" in result[0]

    def test_split_by_word_boundaries(self):
        """Test splitting by word boundaries as fallback."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        result = smart_text_split(text, 20)
        assert len(result) >= 2
        assert all(chunk.strip() for chunk in result)  # No empty chunks

    def test_split_fallback_to_character_boundary(self):
        """Test fallback to character boundary when no word boundaries."""
        text = "verylongwordwithoutspacesorpunctuation"
        result = smart_text_split(text, 10)
        assert len(result) >= 2
        assert all(len(chunk) <= 10 for chunk in result)


class TestBuildTranslationContext:
    """Test translation context building."""

    def test_build_context_without_previous(self):
        """Test building context for first chunk (no previous context)."""
        context = build_translation_context(
            source_lang="en",
            target_lang="fr",
            current_chunk="Hello world",
        )

        assert "Translate the following text from en to fr" in context
        assert "[TRANSLATE_THIS]" in context
        assert "Hello world" in context
        assert "[/TRANSLATE_THIS]" in context
        assert "Only return the translation" in context
        assert "[PREVIOUS_SOURCE]" not in context
        assert "[PREVIOUS_TRANSLATION]" not in context

    def test_build_context_with_previous(self):
        """Test building context with previous chunk and translation."""
        context = build_translation_context(
            source_lang="en",
            target_lang="fr",
            current_chunk="Second sentence",
            previous_chunk="First sentence",
            previous_translation="Première phrase",
        )

        assert "Translate the following text from en to fr" in context
        assert "[PREVIOUS_SOURCE]" in context
        assert "First sentence" in context
        assert "[/PREVIOUS_SOURCE]" in context
        assert "[PREVIOUS_TRANSLATION]" in context
        assert "Première phrase" in context
        assert "[/PREVIOUS_TRANSLATION]" in context
        assert "[TRANSLATE_THIS]" in context
        assert "Second sentence" in context
        assert "[/TRANSLATE_THIS]" in context

    def test_build_context_with_partial_previous(self):
        """Test building context with only previous chunk (no translation)."""
        context = build_translation_context(
            source_lang="en",
            target_lang="fr",
            current_chunk="Current text",
            previous_chunk="Previous text",
            previous_translation=None,
        )

        # Should not include previous context if translation is missing
        assert "[PREVIOUS_SOURCE]" not in context
        assert "[PREVIOUS_TRANSLATION]" not in context
        assert "[TRANSLATE_THIS]" in context
        assert "Current text" in context


class TestTranslateContextAware:
    """Test context-aware translation algorithm."""

    async def test_translate_single_chunk(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translating content that fits in a single chunk."""
        content = DocumentContent(
            pages=["Short text that fits in one chunk."],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should have one translation call
        assert mock_translator.translate.call_count == 1
        
        # Verify the call was made with proper context structure
        call_args = mock_translator.translate.call_args[0][0]
        assert "[TRANSLATE_THIS]" in call_args.content
        assert "Short text that fits in one chunk." in call_args.content
        
        # Should not have previous context for first chunk
        assert "[PREVIOUS_SOURCE]" not in call_args.content

        # Verify result
        assert result.text
        assert result.tokens_used > 0
        assert result.cost > 0
        assert result.time_taken >= 0

    async def test_translate_multiple_chunks(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translating content that requires multiple chunks."""
        # Create content that will definitely be split into multiple chunks
        long_content = "This is the first sentence. " * 10 + "This is the second sentence. " * 10
        content = DocumentContent(
            pages=[long_content],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should have multiple translation calls
        assert mock_translator.translate.call_count >= 2

        # First call should not have previous context
        first_call = mock_translator.translate.call_args_list[0][0][0]
        assert "[TRANSLATE_THIS]" in first_call.content
        assert "[PREVIOUS_SOURCE]" not in first_call.content

        # Second call should have previous context
        if mock_translator.translate.call_count > 1:
            second_call = mock_translator.translate.call_args_list[1][0][0]
            assert "[TRANSLATE_THIS]" in second_call.content
            assert "[PREVIOUS_SOURCE]" in second_call.content
            assert "[PREVIOUS_TRANSLATION]" in second_call.content

        # Verify result
        assert result.text
        assert result.tokens_used > 0
        assert result.cost > 0

    async def test_translate_with_custom_split_token(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translation with custom split token."""
        config = context_aware_config.model_copy(
            update={"custom_split_token": "|||"}
        )
        
        content = DocumentContent(
            pages=["First part|||Second part|||Third part"],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should have exactly 3 translation calls (one per part)
        assert mock_translator.translate.call_count == 3

        # Verify each call contains the expected content
        calls = mock_translator.translate.call_args_list
        assert "First part" in calls[0][0][0].content
        assert "Second part" in calls[1][0][0].content
        assert "Third part" in calls[2][0][0].content

        # Verify result
        assert result.text
        assert "Translated: First part" in result.text
        assert "Translated: Second part" in result.text
        assert "Translated: Third part" in result.text

    async def test_translate_with_checkpointing(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translation with checkpoint saving."""
        content = DocumentContent(
            pages=["Content that will be split into multiple chunks for testing checkpointing functionality."],
            content_type="text/plain",
            metadata={},
        )

        await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Verify checkpoint was loaded
        mock_checkpoint_manager.load.assert_called_once()

        # Verify checkpoint was saved (at least once)
        assert mock_checkpoint_manager.save.call_count >= 1

        # Verify cleanup was called
        mock_checkpoint_manager.cleanup_old_checkpoints.assert_called_once_with(
            context_aware_config.input_file
        )

    async def test_translate_resume_from_checkpoint(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test resuming translation from checkpoint."""
        # Configure checkpoint manager to return existing checkpoint
        existing_checkpoint = TranslationState(
            source_lang="en",
            target_lang="fr",
            algorithm="context-aware",
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={1: "Previously translated content"},
            token_usage=50,
            cost=0.05,
            time_taken=5.0,
        )
        mock_checkpoint_manager.load = AsyncMock(return_value=existing_checkpoint)

        content = DocumentContent(
            pages=["Any content"],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should not make any new translation calls
        mock_translator.translate.assert_not_called()

        # Should return checkpoint data
        assert result.text == "Previously translated content"
        assert result.tokens_used == 50
        assert result.cost == 0.05

    async def test_translate_image_content_error(
        self, context_aware_config, mock_translator
    ):
        """Test that context-aware algorithm rejects image content."""
        content = DocumentContent(
            pages=[b"binary image data"],
            content_type="image/png",
            metadata={},
        )

        with pytest.raises(TranslationError, match="Context-aware algorithm not supported for image content"):
            await translate_context_aware(
                content,
                context_aware_config,
                mock_translator,
            )

    async def test_translate_with_translation_error(
        self, context_aware_config, mock_checkpoint_manager
    ):
        """Test handling of translation errors."""
        # Create a translator that always fails
        failing_translator = AsyncMock(spec=ModelInterface)
        failing_translator.translate.side_effect = TranslationError("Translation failed")

        content = DocumentContent(
            pages=["Test content"],
            content_type="text/plain",
            metadata={},
        )

        with pytest.raises(TranslationError, match="Translation failed"):
            await translate_context_aware(
                content,
                context_aware_config,
                failing_translator,
                checkpoint_manager=mock_checkpoint_manager,
            )

    async def test_translate_multiple_pages(
        self, context_aware_config, mock_translator, mock_checkpoint_manager
    ):
        """Test translation with multiple pages."""
        content = DocumentContent(
            pages=["First page content.", "Second page content.", "Third page content."],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=mock_checkpoint_manager,
        )

        # Should translate the combined content
        assert result.text
        assert "Translated:" in result.text

        # Verify the combined text was processed
        first_call = mock_translator.translate.call_args_list[0][0][0]
        combined_text = "First page content.\n\nSecond page content.\n\nThird page content."
        assert combined_text in first_call.content

    async def test_translate_without_checkpoint_manager(
        self, context_aware_config, mock_translator
    ):
        """Test translation without checkpoint manager."""
        content = DocumentContent(
            pages=["Simple content"],
            content_type="text/plain",
            metadata={},
        )

        result = await translate_context_aware(
            content,
            context_aware_config,
            mock_translator,
            checkpoint_manager=None,
        )

        # Should still work without checkpoint manager
        assert result.text
        assert result.tokens_used > 0
        assert result.cost > 0
