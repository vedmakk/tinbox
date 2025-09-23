"""Translation interface definitions."""

from typing import AsyncIterator, Protocol, Union

from pydantic import BaseModel, Field, ConfigDict

from tinbox.core.types import ModelType


class TranslationError(Exception):
    """Base class for translation-related errors."""

    pass


class TranslationRequest(BaseModel):
    """Configuration for a translation request."""

    source_lang: str
    target_lang: str
    content: Union[str, bytes]  # text or image bytes
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    model: ModelType
    model_params: dict = Field(
        default_factory=dict
    )  # Additional model-specific parameters

    model_config = ConfigDict(frozen=True, protected_namespaces=())


class TranslationResponse(BaseModel):
    """Response from a translation request."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)

    model_config = ConfigDict(frozen=True)


class ModelInterface(Protocol):
    """Protocol for interacting with LLMs."""

    async def translate(
        self,
        request: TranslationRequest,
        stream: bool = False,
    ) -> Union[TranslationResponse, AsyncIterator[TranslationResponse]]:
        """Translate content using the model.

        Args:
            request: The translation request configuration
            stream: Whether to stream the response

        Returns:
            Either a single response or an async iterator of responses if streaming

        Raises:
            TranslationError: If translation fails
        """
        ...

    async def validate_model(self) -> bool:
        """Check if the model is available and properly configured.

        Returns:
            True if the model is available and configured correctly
        """
        ...
