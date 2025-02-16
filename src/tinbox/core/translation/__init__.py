"""Translation engine for Tinbox."""

from tinbox.core.translation.interface import (
    TranslationRequest,
    TranslationResponse,
    ModelInterface,
    TranslationError,
)
from tinbox.core.translation.litellm import LiteLLMTranslator

__all__ = [
    "TranslationRequest",
    "TranslationResponse",
    "ModelInterface",
    "TranslationError",
    "LiteLLMTranslator",
] 
