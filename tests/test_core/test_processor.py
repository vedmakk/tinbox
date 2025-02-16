"""Tests for document processing interfaces and base functionality."""

from pathlib import Path
from typing import AsyncIterator

import pytest
from pydantic import ValidationError

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentContent,
    DocumentMetadata,
    FileType,
    ProcessingError,
)


class MockProcessor(BaseDocumentProcessor):
    """Mock document processor for testing."""
    
    @property
    def supported_types(self) -> set[FileType]:
        """Get supported file types."""
        return {FileType.TXT, FileType.PDF}
    
    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Mock metadata extraction."""
        return DocumentMetadata(
            file_type=FileType.TXT,
            total_pages=1,
            title="Test Document",
        )
    
    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[DocumentContent]:
        """Mock content extraction."""
        yield DocumentContent(
            content="Test content",
            content_type="text/plain",
            page_number=1,
        )


@pytest.fixture
def processor() -> MockProcessor:
    """Fixture providing a mock processor instance."""
    return MockProcessor()


def test_document_content_validation():
    """Test DocumentContent validation."""
    # Valid content
    content = DocumentContent(
        content="Test content",
        content_type="text/plain",
        page_number=1,
    )
    assert content.content == "Test content"
    assert content.content_type == "text/plain"
    assert content.page_number == 1
    
    # Invalid content type
    with pytest.raises(ValidationError):
        DocumentContent(
            content="Test content",
            content_type="invalid",  # Doesn't match regex
            page_number=1,
        )
    
    # Invalid page number
    with pytest.raises(ValidationError):
        DocumentContent(
            content="Test content",
            content_type="text/plain",
            page_number=0,  # Must be >= 1
        )


def test_document_metadata_validation():
    """Test DocumentMetadata validation."""
    # Valid metadata
    metadata = DocumentMetadata(
        file_type=FileType.TXT,
        total_pages=1,
        title="Test Document",
    )
    assert metadata.file_type == FileType.TXT
    assert metadata.total_pages == 1
    assert metadata.title == "Test Document"
    
    # Invalid total pages
    with pytest.raises(ValidationError):
        DocumentMetadata(
            file_type=FileType.TXT,
            total_pages=0,  # Must be >= 1
        )


def test_processor_supported_types(processor: MockProcessor):
    """Test supported file type checking."""
    assert processor.supports_file_type(FileType.TXT)
    assert processor.supports_file_type(FileType.PDF)
    assert not processor.supports_file_type(FileType.DOCX)


@pytest.mark.asyncio
async def test_processor_validate_file(processor: MockProcessor, tmp_path: Path):
    """Test file validation."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    # Valid file
    await processor.validate_file(test_file)
    
    # Non-existent file
    with pytest.raises(ProcessingError, match="File does not exist"):
        await processor.validate_file(tmp_path / "nonexistent.txt")
    
    # Directory instead of file
    with pytest.raises(ProcessingError, match="Not a file"):
        await processor.validate_file(tmp_path)
    
    # Unsupported file type
    unsupported = tmp_path / "test.docx"
    unsupported.write_text("Test content")
    with pytest.raises(ProcessingError, match="not supported"):
        await processor.validate_file(unsupported)


@pytest.mark.asyncio
async def test_processor_get_metadata(processor: MockProcessor, tmp_path: Path):
    """Test metadata extraction."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    metadata = await processor.get_metadata(test_file)
    assert isinstance(metadata, DocumentMetadata)
    assert metadata.file_type == FileType.TXT
    assert metadata.title == "Test Document"


@pytest.mark.asyncio
async def test_processor_extract_content(processor: MockProcessor, tmp_path: Path):
    """Test content extraction."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    content_stream = processor.extract_content(test_file)
    content = await content_stream.__anext__()
    
    assert isinstance(content, DocumentContent)
    assert content.content == "Test content"
    assert content.content_type == "text/plain"
    assert content.page_number == 1 
