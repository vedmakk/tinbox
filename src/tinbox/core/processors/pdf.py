"""PDF document processor implementation."""

import io
from pathlib import Path
from typing import AsyncIterator

from pdf2image import convert_from_path
from PIL import Image

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentContent,
    DocumentMetadata,
    ProcessingError,
)
from tinbox.core.types import FileType
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class PDFProcessor(BaseDocumentProcessor):
    """Processor for PDF documents.
    
    This processor converts PDF pages to images for processing by multimodal models.
    It uses pdf2image (which requires poppler) for the conversion.
    """
    
    def __init__(
        self,
        dpi: int = 200,
        image_format: str = "PNG",
        thread_count: int = 4,
    ) -> None:
        """Initialize the PDF processor.
        
        Args:
            dpi: Resolution for image conversion. Higher values mean better quality
                but larger files. Defaults to 200.
            image_format: Output image format. Defaults to "PNG".
            thread_count: Number of threads to use for PDF conversion.
                Defaults to 4.
        """
        super().__init__()
        self.dpi = dpi
        self.image_format = image_format
        self.thread_count = thread_count
    
    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor."""
        return {FileType.PDF}
    
    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from the PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document metadata
            
        Raises:
            ProcessingError: If metadata extraction fails
        """
        await self.validate_file(file_path)
        
        try:
            # Convert first page to get total page count
            images = convert_from_path(
                file_path,
                first_page=1,
                last_page=1,
                dpi=self.dpi,
                thread_count=self.thread_count,
            )
            
            # Get total page count from PDF
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(file_path))
            total_pages = info["Pages"]
            
            return DocumentMetadata(
                file_type=FileType.PDF,
                total_pages=total_pages,
                title=info.get("Title"),
                author=info.get("Author"),
                creation_date=info.get("CreationDate"),
                modification_date=info.get("ModDate"),
                custom_metadata={
                    "producer": info.get("Producer"),
                    "creator": info.get("Creator"),
                },
            )
            
        except Exception as e:
            raise ProcessingError(f"Failed to extract PDF metadata: {str(e)}") from e
    
    async def extract_content(
        self,
        file_path: Path,
        *,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> AsyncIterator[DocumentContent]:
        """Extract content from the PDF document page by page.
        
        Args:
            file_path: Path to the PDF file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (inclusive), or None for all pages
            
        Yields:
            Document content for each page
            
        Raises:
            ProcessingError: If content extraction fails
        """
        await self.validate_file(file_path)
        
        try:
            # Get metadata for total pages if end_page not specified
            if end_page is None:
                metadata = await self.get_metadata(file_path)
                end_page = metadata.total_pages
            
            # Validate page range
            if start_page < 1:
                raise ProcessingError("start_page must be >= 1")
            if end_page < start_page:
                raise ProcessingError("end_page must be >= start_page")
            
            # Convert PDF pages to images
            images = convert_from_path(
                file_path,
                first_page=start_page,
                last_page=end_page,
                dpi=self.dpi,
                thread_count=self.thread_count,
            )
            
            # Process each page
            for i, image in enumerate(images, start=start_page):
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=self.image_format)
                img_byte_arr = img_byte_arr.getvalue()
                
                yield DocumentContent(
                    content=img_byte_arr,
                    content_type=f"image/{self.image_format.lower()}",
                    page_number=i,
                    metadata={
                        "dpi": self.dpi,
                        "format": self.image_format,
                        "width": image.width,
                        "height": image.height,
                    },
                )
                
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to extract PDF content: {str(e)}") from e 
