"""Core functionality for Tinbox."""

from tinbox.core.types import (
    FileType,
    ModelType,
    TranslationConfig,
    TranslationResult,
)
from tinbox.core.processor import DocumentContent, load_document
from tinbox.core.translation import (
    ModelInterface,
    TranslationRequest,
    TranslationResponse,
    create_translator,
)
from tinbox.core.translation.algorithms import translate_document

__all__ = [
    "FileType",
    "ModelType",
    "TranslationConfig",
    "TranslationResult",
    "DocumentContent",
    "load_document",
    "translate_document",
]
