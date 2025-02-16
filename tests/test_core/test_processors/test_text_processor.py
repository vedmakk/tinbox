"""Tests for the text document processor."""

from pathlib import Path

import pytest

from tinbox.core.processor import DocumentContent, DocumentMetadata, ProcessingError
from tinbox.core.processors.text import TextProcessor
from tinbox.core.types import FileType


@pytest.fixture
def processor() -> TextProcessor:
    """Fixture providing a text processor instance."""
    return TextProcessor()


@pytest.fixture
def sample_text(tmp_path: Path) -> Path:
    """Fixture providing a temporary text file with mixed content."""
    content = """This is a test document.
With multiple lines of text.

It also includes some Arabic:
هذا نص عربي للاختبار
مزيد من النص العربي

And some more English text.
"""
    file_path = tmp_path / "test.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def invalid_encoding_text(tmp_path: Path) -> Path:
    """Fixture providing a text file with invalid UTF-8 encoding."""
    file_path = tmp_path / "invalid.txt"
    with file_path.open("wb") as f:
        f.write(b"This has invalid UTF-8 \xFF\xFE bytes")
    return file_path


@pytest.mark.asyncio
async def test_text_processor_metadata(processor: TextProcessor, sample_text: Path):
    """Test text document metadata extraction."""
    metadata = await processor.get_metadata(sample_text)
    
    assert isinstance(metadata, DocumentMetadata)
    assert metadata.file_type == FileType.TXT
    assert metadata.total_pages == 1  # Single text file
    assert metadata.title == "test.txt"
    
    # File timestamps should be present
    assert metadata.creation_date is not None
    assert metadata.modification_date is not None
    
    # Custom metadata
    assert metadata.custom_metadata["size_bytes"] > 0
    assert metadata.custom_metadata["encoding"] == "utf-8"
    assert metadata.custom_metadata["contains_rtl"]  # Should be True for Arabic text


@pytest.mark.asyncio
async def test_text_processor_content_extraction(
    processor: TextProcessor, sample_text: Path
):
    """Test text document content extraction."""
    content_stream = processor.extract_content(sample_text)
    content = await content_stream.__anext__()
    
    assert isinstance(content, DocumentContent)
    assert content.content_type == "text/plain"
    assert content.page_number == 1  # Single text file
    
    # Verify content
    assert "This is a test document" in content.content
    assert "هذا نص عربي" in content.content  # Arabic text should be present
    
    # Verify metadata
    assert content.metadata["length"] > 0
    assert content.metadata["encoding"] == "utf-8"
    assert content.metadata["contains_rtl"]  # Should have RTL text


@pytest.mark.asyncio
async def test_text_processor_rtl_content(processor: TextProcessor, sample_text: Path):
    """Test handling of right-to-left text."""
    content_stream = processor.extract_content(sample_text)
    content = await content_stream.__anext__()
    
    assert "هذا نص عربي" in content.content
    assert content.metadata["contains_rtl"]


@pytest.mark.asyncio
async def test_text_processor_invalid_encoding(
    processor: TextProcessor, invalid_encoding_text: Path
):
    """Test handling of invalid UTF-8 encoding."""
    with pytest.raises(ProcessingError, match="not valid UTF-8"):
        await processor.get_metadata(invalid_encoding_text)
    
    with pytest.raises(ProcessingError, match="not valid UTF-8"):
        async for _ in processor.extract_content(invalid_encoding_text):
            pass


@pytest.mark.asyncio
async def test_text_processor_invalid_file(processor: TextProcessor, tmp_path: Path):
    """Test handling of invalid files."""
    # Non-existent file
    with pytest.raises(ProcessingError, match="does not exist"):
        await processor.get_metadata(tmp_path / "nonexistent.txt")
    
    # Invalid file type
    invalid_file = tmp_path / "test.docx"
    invalid_file.write_text("Not a text file")
    with pytest.raises(ProcessingError, match="not supported"):
        await processor.get_metadata(invalid_file) 
