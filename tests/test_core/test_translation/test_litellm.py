"""Tests for the LiteLLM translator implementation."""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
import asyncio

import pytest
from PIL import Image

from tinbox.core.translation import (
    LiteLLMTranslator,
    TranslationError,
    TranslationRequest,
)
from tinbox.core.types import ModelType


class MockStreamingResponse:
    """Mock streaming response for testing."""

    def __init__(self, text: str = "Translated text"):
        self._chunks = [c for c in text]  # Split text into characters for streaming
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration

        chunk = self._chunks[self._index]
        self._index += 1

        return type(
            "StreamChunk",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"delta": type("Delta", (), {"content": chunk})},
                    )
                ]
            },
        )


class MockResponse:
    """Mock response for testing."""

    def __init__(self, text: str = "Translated text", tokens: int = 30):
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "message": type("Message", (), {"content": text}),
                    "finish_reason": "stop",
                },
            )
        ]
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": tokens,
            },
        )


class MockTimeoutResponse:
    """Mock response that simulates a timeout."""

    def __init__(self):
        raise asyncio.TimeoutError("Request timed out")


class MockRateLimitResponse:
    """Mock response that simulates rate limiting."""

    def __init__(self):
        raise RuntimeError("Rate limit exceeded. Please try again in 60 seconds.")


@pytest.fixture
def translator() -> LiteLLMTranslator:
    """Fixture providing a LiteLLM translator instance."""
    return LiteLLMTranslator()


@pytest.fixture
def mock_completion(monkeypatch):
    """Fixture to mock LiteLLM completion calls."""

    def mock_completion(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        if kwargs.get("stream", False):
            return MockStreamingResponse()
        return MockResponse()

    # Patch both the module and the imported function
    import litellm

    monkeypatch.setattr(litellm, "completion", mock_completion)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_completion)


@pytest.mark.asyncio
async def test_text_translation(translator: LiteLLMTranslator, mock_completion):
    """Test translating text content."""
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )

    response = await translator.translate(request)

    assert response.text == "Translated text"
    assert response.tokens_used == 30
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_image_translation(
    translator: LiteLLMTranslator, mock_completion, tmp_path: Path
):
    """Test translating image content."""
    # Create a valid PNG image
    image = Image.new("RGB", (100, 100), color="white")
    image_path = tmp_path / "test.png"
    image.save(image_path)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=image_path.read_bytes(),
        content_type="image/png",
        model=ModelType.GPT4_VISION,
    )

    response = await translator.translate(request)

    assert response.text == "Translated text"
    assert response.tokens_used == 30
    assert response.time_taken > 0


@pytest.mark.asyncio
async def test_streaming_translation(translator: LiteLLMTranslator, mock_completion):
    """Test streaming translation responses."""
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )

    response_gen = await translator.translate(request, stream=True)
    responses: List[TranslationResponse] = []
    async for response in response_gen:
        responses.append(response)

    # Each character in "Translated text" should be a response
    assert len(responses) == len("Translated text")
    assert responses[-1].text == "Translated text"
    assert all(r.tokens_used > 0 for r in responses)
    assert all(r.time_taken > 0 for r in responses)


@pytest.mark.asyncio
async def test_translation_error_handling(translator: LiteLLMTranslator, monkeypatch):
    """Test error handling during translation."""

    def mock_error(*args: Any, **kwargs: Dict[str, Any]) -> None:
        raise RuntimeError("API error")

    # Patch both the module and the imported function
    import litellm

    monkeypatch.setattr(litellm, "completion", mock_error)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_error)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )

    with pytest.raises(TranslationError, match="Translation failed"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_model_validation(translator: LiteLLMTranslator, mock_completion):
    """Test model validation."""
    assert await translator.validate_model()


@pytest.mark.asyncio
async def test_model_validation_failure(translator: LiteLLMTranslator, monkeypatch):
    """Test model validation failure."""

    def mock_error(*args: Any, **kwargs: Dict[str, Any]) -> None:
        raise RuntimeError("API error")

    # Patch both the module and the imported function
    import litellm

    monkeypatch.setattr(litellm, "completion", mock_error)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_error)

    assert not await translator.validate_model()


@pytest.mark.asyncio
async def test_empty_content(translator: LiteLLMTranslator, mock_completion):
    """Test handling of empty or whitespace-only content."""
    # Empty string
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response

    # Whitespace only
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="   \n\t   ",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response


@pytest.mark.asyncio
async def test_long_content(translator: LiteLLMTranslator, mock_completion):
    """Test handling of very long content."""
    long_text = "Hello, world! " * 1000  # Create long repetitive text
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=long_text,
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response
    assert response.tokens_used == 30  # Mock token count


@pytest.mark.asyncio
async def test_special_characters(translator: LiteLLMTranslator, mock_completion):
    """Test handling of special characters and emojis."""
    special_text = "Hello! 👋 This is a test with special chars: ¡¢£¤¥¦§¨©ª"
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=special_text,
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response


@pytest.mark.asyncio
async def test_invalid_model_params(translator: LiteLLMTranslator, mock_completion):
    """Test handling of invalid model parameters."""
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
        model_params={"invalid_param": "value"},  # Invalid parameter
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response


@pytest.mark.asyncio
async def test_malformed_image(translator: LiteLLMTranslator, mock_completion):
    """Test handling of malformed image data."""
    invalid_image_data = b"This is not a valid image"
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=invalid_image_data,
        content_type="image/png",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response


@pytest.mark.asyncio
async def test_timeout_handling(translator: LiteLLMTranslator, monkeypatch):
    """Test handling of network timeouts."""

    def mock_timeout(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        raise asyncio.TimeoutError("Request timed out")

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_timeout)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_timeout)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )

    with pytest.raises(TranslationError, match="Translation failed"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_rate_limit_handling(translator: LiteLLMTranslator, monkeypatch):
    """Test handling of rate limiting."""

    def mock_rate_limit(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        return MockRateLimitResponse()

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_rate_limit)
    monkeypatch.setattr("tinbox.core.translation.litellm.completion", mock_rate_limit)

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )

    with pytest.raises(TranslationError, match="Translation failed"):
        await translator.translate(request)


@pytest.mark.asyncio
async def test_invalid_language_codes(translator: LiteLLMTranslator, mock_completion):
    """Test handling of invalid language codes."""
    # Invalid source language
    request = TranslationRequest(
        source_lang="invalid",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response

    # Invalid target language
    request = TranslationRequest(
        source_lang="en",
        target_lang="invalid",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response


@pytest.mark.asyncio
async def test_mixed_content_handling(translator: LiteLLMTranslator, mock_completion):
    """Test handling of mixed content types."""
    # Text with embedded image-like content
    mixed_content = "Hello, world! data:image/png;base64,invalid_base64"
    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content=mixed_content,
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )
    response = await translator.translate(request)
    assert response.text == "Translated text"  # Mock response


@pytest.mark.asyncio
async def test_response_validation(translator: LiteLLMTranslator, monkeypatch):
    """Test validation of malformed responses."""

    def mock_malformed_response(*args: Any, **kwargs: Dict[str, Any]) -> Any:
        return type(
            "MalformedResponse",
            (),
            {
                "choices": [],  # Empty choices
                "usage": None,  # Missing usage info
            },
        )

    import litellm

    monkeypatch.setattr(litellm, "completion", mock_malformed_response)
    monkeypatch.setattr(
        "tinbox.core.translation.litellm.completion", mock_malformed_response
    )

    request = TranslationRequest(
        source_lang="en",
        target_lang="es",
        content="Hello, world!",
        content_type="text/plain",
        model=ModelType.GPT4_VISION,
    )

    with pytest.raises(TranslationError, match="No response from model"):
        await translator.translate(request)
