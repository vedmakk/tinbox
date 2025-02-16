"""Text document processor implementation."""

import re
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentContent,
    DocumentMetadata,
    ProcessingError,
)
from tinbox.core.types import FileType
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class TextProcessor(BaseDocumentProcessor):
    """Processor for plain text documents."""
    
    def __init__(self) -> None:
        """Initialize the text processor."""
        super().__init__()
        self._rtl_pattern = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
    
    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor."""
        return {FileType.TXT}
    
    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from the text document.
        
        Args:
            file_path: Path to the text document
            
        Returns:
            Document metadata
            
        Raises:
            ProcessingError: If metadata extraction fails
        """
        await self.validate_file(file_path)
        
        try:
            # Read file to check encoding and RTL content
            text = file_path.read_text(encoding="utf-8")
            has_rtl = bool(self._rtl_pattern.search(text))
            
            # Get file stats
            stats = file_path.stat()
            
            return DocumentMetadata(
                file_type=FileType.TXT,
                total_pages=1,  # Treat as single text file
                title=file_path.name,
                creation_date=datetime.fromtimestamp(stats.st_ctime).isoformat(),
                modification_date=datetime.fromtimestamp(stats.st_mtime).isoformat(),
                custom_metadata={
                    "size_bytes": stats.st_size,
                    "encoding": "utf-8",
                    "contains_rtl": has_rtl,
                },
            )
            
        except UnicodeDecodeError as e:
            raise ProcessingError(
                f"File is not valid UTF-8 encoded text: {str(e)}"
            ) from e
        except Exception as e:
            raise ProcessingError(f"Failed to extract text metadata: {str(e)}") from e
    
    async def extract_content(
        self,
        file_path: Path,
        *,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> AsyncIterator[DocumentContent]:
        """Extract content from the text document.
        
        Since text documents are treated as single files, page parameters are ignored.
        The sliding window algorithm will handle chunking the text appropriately.
        
        Args:
            file_path: Path to the text document
            start_page: Ignored (kept for interface compatibility)
            end_page: Ignored (kept for interface compatibility)
            
        Yields:
            Document content
            
        Raises:
            ProcessingError: If content extraction fails
        """
        await self.validate_file(file_path)
        
        try:
            # Read file content
            text = file_path.read_text(encoding="utf-8")
            
            # Detect if text contains RTL content
            has_rtl = bool(self._rtl_pattern.search(text))
            
            yield DocumentContent(
                content=text,
                content_type="text/plain",
                page_number=1,  # Single "page"
                metadata={
                    "length": len(text),
                    "contains_rtl": has_rtl,
                    "encoding": "utf-8",
                },
            )
                
        except UnicodeDecodeError as e:
            raise ProcessingError(
                f"File is not valid UTF-8 encoded text: {str(e)}"
            ) from e
        except Exception as e:
            raise ProcessingError(f"Failed to extract text content: {str(e)}") from e 
