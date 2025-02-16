"""Core data types for the Tinbox translation tool."""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class FileType(str, Enum):
    """Supported file types for translation."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class ModelType(str, Enum):
    """Supported LLM models for translation."""

    GPT4_VISION = "gpt-4o"  # OpenAI GPT-4 with vision capabilities
    CLAUDE_3_5_SONNET = "claude-3.5-sonnet-latest"  # Anthropic Claude 3 Sonnet
    CLAUDE_3_5_OPUS = "claude-3.5-opus-latest"  # Anthropic Claude 3 Opus
    OLLAMA = "ollama"  # For local models


class TranslationConfig(BaseModel):
    """Configuration for translation tasks."""

    source_lang: str
    target_lang: str
    model: ModelType
    algorithm: Literal["page", "sliding-window"]
    input_file: Path
    output_file: Optional[Path] = None
    benchmark: bool = False

    # Algorithm-specific settings
    page_seam_overlap: int = Field(default=200, gt=0)
    window_size: int = Field(default=2000, gt=0)
    overlap_size: int = Field(default=200, gt=0)

    model_config = ConfigDict(frozen=True)  # Make config immutable


class TranslationResult(BaseModel):
    """Result of a translation operation."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)

    model_config = ConfigDict(frozen=True)
