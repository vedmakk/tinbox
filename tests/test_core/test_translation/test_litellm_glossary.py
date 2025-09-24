import pytest
from unittest.mock import patch

from tinbox.core.translation.interface import TranslationRequest, TranslationError
from tinbox.core.translation.litellm import LiteLLMTranslator
from tinbox.core.types import ModelType, Glossary, GlossaryEntry


@pytest.mark.asyncio
async def test_litellm_structured_response_with_glossary():
    translator = LiteLLMTranslator()

    # Mock a structured response with parsed field
    with patch("tinbox.core.translation.litellm.completion") as mock:
        class Parsed:
            translation = "Translated structured"
            glossary_extension = [GlossaryEntry(term="AI", translation="IA")]

        mock.return_value = type(
            "CompletionResponse",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "finish_reason": "stop",
                            "message": type("Message", (), {"parsed": Parsed})(),
                        },
                    )
                ],
                "usage": type("Usage", (), {"total_tokens": 12})(),
                "_hidden_params": {"response_cost": 0.002},
            },
        )

        req = TranslationRequest(
            source_lang="en",
            target_lang="fr",
            content="Hello",
            context=None,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
            glossary=Glossary(entries={"CPU": "Processeur"}),
        )

        resp = await translator.translate(req)
        assert resp.text.endswith("Translated structured")
        assert len(resp.glossary_updates) == 1
        assert resp.glossary_updates[0].term == "AI"


@pytest.mark.asyncio
async def test_litellm_raises_on_missing_parsed():
    translator = LiteLLMTranslator()

    with patch("tinbox.core.translation.litellm.completion") as mock:
        mock.return_value = type(
            "CompletionResponse",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "finish_reason": "stop",
                            "message": type("Message", (), {"content": "Plain translated"})(),
                        },
                    )
                ],
                "usage": type("Usage", (), {"total_tokens": 8})(),
                "_hidden_params": {"response_cost": 0.001},
            },
        )

        req = TranslationRequest(
            source_lang="en",
            target_lang="fr",
            content="Hi",
            context=None,
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-sonnet"},
        )

        with pytest.raises(TranslationError, match="Invalid response format"):
            await translator.translate(req)
