"""Document processing interfaces and implementations for Tinbox."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, Union, List, Dict, Type

from pydantic import BaseModel, Field, ConfigDict

from tinbox.core.types import FileType
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentContent(BaseModel):
    """Represents a single page of document content ready for translation."""

    content: Union[str, bytes]
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    page_number: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class DocumentPages(BaseModel):
    """Represents a multi-page document ready for translation."""

    content: Union[str, bytes]  # Full document content for sliding window
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    pages: List[Union[str, bytes]]  # Individual pages for page-by-page translation
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class DocumentMetadata(BaseModel):
    """Metadata about a document being processed."""

    file_type: FileType
    total_pages: int = Field(ge=1)
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class DocumentProcessor(Protocol):
    """Protocol defining the interface for document processors."""

    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor."""
        ...

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from the document.

        Args:
            file_path: Path to the document file

        Returns:
            Document metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        ...

    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[DocumentContent]:
        """Extract content from the document page by page.

        Args:
            file_path: Path to the document file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (inclusive), or None for all pages

        Yields:
            Document content for each page

        Raises:
            ProcessingError: If content extraction fails
        """
        ...


class BaseDocumentProcessor(ABC):
    """Base class for document processors implementing common functionality."""

    def __init__(self) -> None:
        """Initialize the document processor."""
        self._logger = get_logger(self.__class__.__name__)

    @property
    @abstractmethod
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor."""
        ...

    def supports_file_type(self, file_type: FileType) -> bool:
        """Check if this processor supports a given file type.

        Args:
            file_type: The file type to check

        Returns:
            True if the file type is supported, False otherwise
        """
        return file_type in self.supported_types

    async def validate_file(self, file_path: Path) -> None:
        """Validate that a file can be processed.

        Args:
            file_path: Path to the file to validate

        Raises:
            ProcessingError: If the file is invalid or cannot be processed
        """
        if not file_path.exists():
            raise ProcessingError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ProcessingError(f"Not a file: {file_path}")

        try:
            file_type = FileType(file_path.suffix.lstrip(".").lower())
        except ValueError as e:
            raise ProcessingError(f"Unsupported file type: {file_path.suffix}") from e

        if not self.supports_file_type(file_type):
            raise ProcessingError(
                f"File type {file_type} not supported by {self.__class__.__name__}"
            )


class ProcessingError(Exception):
    """Exception raised when document processing fails."""

    pass


def get_processor_for_file_type(file_type: FileType) -> DocumentProcessor:
    """Get the appropriate processor for a file type.

    Args:
        file_type: The file type to get a processor for

    Returns:
        A document processor instance

    Raises:
        ProcessingError: If no processor is available for the file type
    """
    # Import processors here to avoid circular imports
    from tinbox.core.processors.pdf import PDFProcessor
    from tinbox.core.processors.text import TextProcessor
    from tinbox.core.processors.word import WordProcessor

    processors = {
        FileType.PDF: PDFProcessor(),
        FileType.DOCX: WordProcessor(),
        FileType.TXT: TextProcessor(),
    }

    processor = processors.get(file_type)
    if not processor:
        raise ProcessingError(f"Unsupported file type: {file_type}")

    return processor


async def load_document(file_path: Path) -> DocumentPages:
    """Load a document and prepare it for translation.

    Args:
        file_path: Path to the document file

    Returns:
        Document pages ready for translation

    Raises:
        ProcessingError: If document loading fails
    """
    # Determine file type
    file_type = FileType(file_path.suffix.lstrip(".").lower())

    # Get appropriate processor
    processor = get_processor_for_file_type(file_type)

    # Validate file
    await processor.validate_file(file_path)

    # Get metadata
    metadata = await processor.get_metadata(file_path)

    # Extract content
    pages = []
    content = ""
    content_type = "text/plain"

    async for page in processor.extract_content(file_path):
        pages.append(page.content)
        if isinstance(page.content, str):
            content += page.content + "\n\n"
        content_type = page.content_type

    if isinstance(content, str) and content.endswith("\n\n"):
        content = content[:-2]  # Remove trailing newlines

    return DocumentPages(
        content=content or pages[0],  # Use first page if no combined content
        content_type=content_type,
        pages=pages,
        metadata=metadata.model_dump(),
    )
