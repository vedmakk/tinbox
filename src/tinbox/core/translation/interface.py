"""Translation interface definitions."""

from typing import Optional, Protocol, Union, List

from pydantic import BaseModel, Field, ConfigDict

from tinbox.core.types import ModelType, Glossary, GlossaryEntry


class TranslationError(Exception):
    """Base class for translation-related errors."""

    pass


class TranslationRequest(BaseModel):
    """Configuration for a translation request."""

    source_lang: str
    target_lang: str
    content: Union[str, bytes]  # Pure content to translate (text or image bytes)
    context: Optional[str] = None  # Supporting context information for better translation
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    model: ModelType
    model_params: dict = Field(
        default_factory=dict
    )  # Additional model-specific parameters
    glossary: Optional[Glossary] = Field(
        default=None,
        description="Optional glossary for consistent translations",
    )

    model_config = ConfigDict(frozen=True, protected_namespaces=())


class TranslationResponse(BaseModel):
    """Response from a translation request."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)
    glossary_updates: List[GlossaryEntry] = Field(
        default_factory=list,
        description="New glossary entries discovered during this translation",
    )

    model_config = ConfigDict(frozen=True)


class ModelInterface(Protocol):
    """Protocol for interacting with LLMs."""

    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate content using the model.

        Args:
            request: The translation request configuration

        Returns:
            A single translation response

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


class TranslationWithGlossaryResponse(BaseModel):
    """Structured LLM response when glossary is enabled."""

    translation: str = Field(description="The translated text")
    glossary_extension: Optional[List[GlossaryEntry]] = Field(
        default=None,
        description="New glossary entries discovered during translation (optional)",
    )

    model_config = ConfigDict(frozen=True)


class TranslationWithoutGlossaryResponse(BaseModel):
    """Structured LLM response when glossary is not enabled."""

    translation: str = Field(description="The translated text")

    model_config = ConfigDict(frozen=True)
