"""Core data types for the Tinbox translation tool."""

from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class FileType(str, Enum):
    """Supported file types for translation."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class ModelType(str, Enum):
    """Supported LLM model providers."""

    OPENAI = "openai"  # OpenAI models (GPT-4, etc.)
    ANTHROPIC = "anthropic"  # Anthropic models (Claude)
    OLLAMA = "ollama"  # Local models via Ollama
    GEMINI = "gemini"  # Google's Gemini models


class TranslationConfig(BaseModel):
    """Configuration for translation tasks."""

    # Basic settings
    source_lang: str
    target_lang: str
    model: ModelType
    model_name: str = Field(
        description="Specific model name (e.g., 'gpt-4o' for OpenAI, 'claude-3-sonnet' for Anthropic)",
    )
    algorithm: Literal["page", "sliding-window"]
    input_file: Path
    output_file: Optional[Path] = None

    # UI and progress settings
    verbose: bool = Field(
        default=False,
        description="Whether to show detailed progress information",
    )
    progress_callback: Optional[Callable[[int], None]] = Field(
        default=None,
        description="Callback function to update progress (receives tokens processed)",
    )

    # Cost control settings
    max_cost: Optional[float] = Field(
        default=None,
        description="Maximum cost threshold in USD",
        ge=0.0,
    )
    force: bool = Field(
        default=False,
        description="Whether to skip cost and size warnings",
    )

    # Algorithm-specific settings
    page_seam_overlap: int = Field(
        default=200,
        gt=0,
        description="Token overlap for page-by-page translation",
    )
    window_size: int = Field(
        default=2000,
        gt=0,
        description="Window size for sliding window translation",
    )
    overlap_size: int = Field(
        default=200,
        gt=0,
        description="Overlap size for sliding window translation",
    )

    # Checkpoint settings
    checkpoint_dir: Optional[Path] = Field(
        default=None,
        description="Directory to store checkpoints",
    )
    checkpoint_frequency: int = Field(
        default=1,
        gt=0,
        description="Save checkpoint every N pages/chunks",
    )
    resume_from_checkpoint: bool = Field(
        default=True,
        description="Whether to try resuming from checkpoint",
    )

    model_config = ConfigDict(
        frozen=True,  # Make config immutable
        arbitrary_types_allowed=True,  # Allow Callable type for progress_callback
    )


class TranslationResult(BaseModel):
    """Result of a translation operation."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)

    model_config = ConfigDict(frozen=True)
