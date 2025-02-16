"""Text document processor implementation."""

from pathlib import Path
from typing import AsyncIterator, Union

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentMetadata,
    FileType,
    ProcessingError,
)
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class TextProcessor(BaseDocumentProcessor):
    """Processor for text files."""

    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor.

        Returns:
            Set containing only FileType.TXT
        """
        return {FileType.TXT}

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from a text document.

        Args:
            file_path: Path to the text file

        Returns:
            Document metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        try:
            # For text files, we'll count paragraphs (double newlines) as pages
            text = file_path.read_text()
            paragraphs = [p for p in text.split("\n\n") if p.strip()]

            return DocumentMetadata(
                file_type=FileType.TXT,
                total_pages=len(paragraphs) or 1,  # Ensure at least 1 page
                title=file_path.name,
                author=None,
                creation_date=None,
                modification_date=str(file_path.stat().st_mtime),
            )
        except Exception as e:
            logger.exception("Failed to extract text metadata")
            raise ProcessingError(f"Failed to extract text metadata: {str(e)}") from e

    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[Union[str, bytes]]:
        """Extract content from a text document.

        Args:
            file_path: Path to the text file
            start_page: First paragraph to extract (1-indexed)
            end_page: Last paragraph to extract (inclusive), or None for all paragraphs

        Yields:
            Text content of each paragraph

        Raises:
            ProcessingError: If content extraction fails
        """
        try:
            text = file_path.read_text()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            selected_paragraphs = paragraphs[start_page - 1 : end_page]

            for para in selected_paragraphs:
                yield para

        except Exception as e:
            logger.exception("Failed to extract text content")
            raise ProcessingError(f"Failed to extract text content: {str(e)}") from e
