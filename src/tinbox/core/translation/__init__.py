"""Translation engine for Tinbox."""

from tinbox.core.translation.interface import (
    TranslationRequest,
    TranslationResponse,
    ModelInterface,
    TranslationError,
)
from tinbox.core.translation.litellm import LiteLLMTranslator
from tinbox.core.types import TranslationConfig


def create_translator(config: TranslationConfig) -> ModelInterface:
    """Create a translator instance based on configuration.

    Args:
        config: Translation configuration

    Returns:
        Configured translator instance
    """
    translator = LiteLLMTranslator()

    # Create translation request with model-specific parameters
    model_params = {}
    if config.model_name:
        model_params["model_name"] = config.model_name

    return translator


__all__ = [
    "TranslationRequest",
    "TranslationResponse",
    "ModelInterface",
    "TranslationError",
    "LiteLLMTranslator",
    "create_translator",
]
