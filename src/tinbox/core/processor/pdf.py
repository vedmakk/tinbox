"""PDF document processor implementation."""

from pathlib import Path
from typing import AsyncIterator, Union

import pypdf
from pdf2image import convert_from_path
import io

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentMetadata,
    FileType,
    ProcessingError,
)
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class PdfProcessor(BaseDocumentProcessor):
    """Processor for PDF files."""

    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor.

        Returns:
            Set containing only FileType.PDF
        """
        return {FileType.PDF}

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        try:
            with open(file_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                info = pdf.metadata

                return DocumentMetadata(
                    file_type=FileType.PDF,
                    total_pages=len(pdf.pages),
                    title=info.get("/Title", None),
                    author=info.get("/Author", None),
                    creation_date=info.get("/CreationDate", None),
                    modification_date=info.get("/ModDate", None),
                )
        except Exception as e:
            logger.exception("Failed to extract PDF metadata")
            raise ProcessingError(f"Failed to extract PDF metadata: {str(e)}") from e

    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[Union[str, bytes]]:
        """Extract content from a PDF document.

        Args:
            file_path: Path to the PDF file
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (inclusive), or None for all pages

        Yields:
            PNG image bytes for each page

        Raises:
            ProcessingError: If content extraction fails
        """
        try:
            # Convert pages to images
            pages = convert_from_path(
                file_path,
                first_page=start_page,
                last_page=end_page,
            )

            for page in pages:
                with io.BytesIO() as bio:
                    page.save(bio, format="PNG")
                    yield bio.getvalue()

        except Exception as e:
            logger.exception("Failed to extract PDF content")
            raise ProcessingError(f"Failed to extract PDF content: {str(e)}") from e
