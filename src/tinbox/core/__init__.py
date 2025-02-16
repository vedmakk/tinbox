"""Core functionality for Tinbox."""

from tinbox.core.types import FileType, ModelType, TranslationConfig, TranslationResult
from tinbox.core.cost import CostEstimate, CostLevel, estimate_cost
from tinbox.core.progress import ProgressStats, ProgressTracker
from tinbox.core.translation.algorithms import translate_document
from tinbox.core.processor import (
    DocumentContent,
    DocumentMetadata,
    DocumentProcessor,
    BaseDocumentProcessor,
    ProcessingError,
    load_document,
)

__all__ = [
    "CostEstimate",
    "CostLevel",
    "estimate_cost",
    "FileType",
    "ModelType",
    "ProgressStats",
    "ProgressTracker",
    "TranslationConfig",
    "TranslationResult",
    "translate_document",
    "DocumentContent",
    "DocumentMetadata",
    "DocumentProcessor",
    "BaseDocumentProcessor",
    "ProcessingError",
    "load_document",
]
