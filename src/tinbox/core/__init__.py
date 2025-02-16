"""Core functionality for the Tinbox translation tool."""

from tinbox.core.types import FileType, ModelType, TranslationConfig, TranslationResult
from tinbox.core.processor import (
    DocumentContent,
    DocumentMetadata,
    DocumentProcessor,
    BaseDocumentProcessor,
    ProcessingError,
)

__all__ = [
    "FileType",
    "ModelType",
    "TranslationConfig",
    "TranslationResult",
    "DocumentContent",
    "DocumentMetadata",
    "DocumentProcessor",
    "BaseDocumentProcessor",
    "ProcessingError",
] 
