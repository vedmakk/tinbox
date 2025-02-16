"""LiteLLM-based translation implementation."""

import base64
from datetime import datetime
from typing import AsyncIterator, Union

from litellm import completion

from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import ModelType
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class LiteLLMTranslator(ModelInterface):
    """LiteLLM-based implementation of the model interface."""

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the translator.

        Args:
            temperature: Model temperature (randomness). Defaults to 0.3.
            max_tokens: Maximum tokens in response. Defaults to 4096.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._logger = logger

    def _get_model_string(self, request: TranslationRequest) -> str:
        """Get the model string for LiteLLM.

        Args:
            request: The translation request

        Returns:
            Model string for LiteLLM

        Raises:
            TranslationError: If model name is missing
        """
        if not request.model_params.get("model_name"):
            raise TranslationError("No model name provided")

        model_name = request.model_params["model_name"]

        # If model name already includes provider prefix, use it directly
        if "/" in model_name:  # Using / instead of : as that's what litellm expects
            return model_name

        # Handle provider-specific model strings
        if request.model == ModelType.OLLAMA:
            return f"ollama/{model_name}"
        elif request.model == ModelType.OPENAI:
            return model_name  # OpenAI models use their names directly
        elif request.model == ModelType.ANTHROPIC:
            # Anthropic models need anthropic/ prefix according to litellm docs
            # Replace : with / if present in the model name
            clean_name = model_name.replace(":", "/")
            return f"anthropic/{clean_name}"
        elif request.model == ModelType.GEMINI:
            # Gemini models need gemini/ prefix according to litellm docs
            # Replace : with / if present in the model name
            clean_name = model_name.replace(":", "/")
            return f"gemini/{clean_name}"
        else:
            raise TranslationError(f"Unsupported model provider: {request.model}")

    def _create_prompt(self, request: TranslationRequest) -> list[dict]:
        """Create the prompt for the model.

        Args:
            request: The translation request

        Returns:
            List of message dictionaries for the model
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. Translate the following content "
                    f"from {request.source_lang} to {request.target_lang}. "
                    f"Maintain the original formatting and structure. "
                    f"Translate only the content, do not add any explanations or notes."
                ),
            }
        ]

        if request.content_type.startswith("text/"):
            # Text content
            messages.append(
                {
                    "role": "user",
                    "content": f"{request.content}\n\nTranslation without commentary:",
                }
            )
        else:
            # Image content
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Translate the contents of the image from {request.source_lang} to {request.target_lang}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(request.content).decode()}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Translation without commentary:",
                        },
                    ],
                }
            )

        return messages

    async def translate(
        self,
        request: TranslationRequest,
        stream: bool = False,
    ) -> Union[TranslationResponse, AsyncIterator[TranslationResponse]]:
        """Translate content using LiteLLM.

        Args:
            request: The translation request configuration
            stream: Whether to stream the response

        Returns:
            Either a single response or an async iterator of responses if streaming

        Raises:
            TranslationError: If translation fails
        """
        start_time = datetime.now()

        try:
            if stream:
                # For streaming, return an async generator
                async def response_generator() -> AsyncIterator[TranslationResponse]:
                    total_tokens = 0
                    accumulated_text = ""

                    try:
                        response = completion(
                            model=self._get_model_string(request),
                            messages=self._create_prompt(request),
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            stream=True,
                            **{
                                k: v
                                for k, v in request.model_params.items()
                                if k != "model_name"
                            },
                        )

                        async for chunk in response:
                            if not hasattr(chunk, "choices") or not chunk.choices:
                                continue

                            delta = chunk.choices[0].delta
                            if not hasattr(delta, "content") or not delta.content:
                                continue

                            accumulated_text += delta.content
                            total_tokens += 1  # Approximate token count for streaming

                            yield TranslationResponse(
                                text=accumulated_text,
                                tokens_used=total_tokens,
                                cost=total_tokens * 0.001,  # Approximate cost
                                time_taken=(
                                    datetime.now() - start_time
                                ).total_seconds(),
                            )
                    except Exception as e:
                        raise TranslationError(f"Streaming failed: {str(e)}") from e

                return response_generator()
            else:
                # For non-streaming, make a single request
                try:
                    response = completion(
                        model=self._get_model_string(request),
                        messages=self._create_prompt(request),
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False,
                        **{
                            k: v
                            for k, v in request.model_params.items()
                            if k != "model_name"
                        },
                    )

                    if not hasattr(response, "choices") or not response.choices:
                        raise TranslationError("No response from model")

                    text = response.choices[0].message.content
                    tokens = getattr(response.usage, "total_tokens", len(text.split()))
                    cost = tokens * 0.001  # Approximate cost if not provided

                    return TranslationResponse(
                        text=text,
                        tokens_used=tokens,
                        cost=cost,
                        time_taken=(datetime.now() - start_time).total_seconds(),
                    )
                except Exception as e:
                    raise TranslationError(f"Translation failed: {str(e)}") from e

        except Exception as e:
            # Catch any remaining errors
            raise TranslationError(f"Translation failed: {str(e)}") from e

    async def validate_model(self) -> bool:
        """Verify the model is available and properly configured.

        Returns:
            True if the model is available and configured correctly
        """
        try:
            # Make a minimal test request
            response = completion(
                model="gpt-3.5-turbo",  # Use a simple model for validation
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return hasattr(response, "choices") and len(response.choices) > 0
        except Exception as e:
            self._logger.error(f"Model validation failed: {str(e)}")
            return False
