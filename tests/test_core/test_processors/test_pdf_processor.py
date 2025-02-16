"""Tests for the PDF document processor."""

import io
from pathlib import Path

import pytest
from PIL import Image

from tinbox.core.processor import DocumentContent, DocumentMetadata, ProcessingError
from tinbox.core.processors.pdf import PDFProcessor
from tinbox.core.types import FileType


@pytest.fixture
def processor() -> PDFProcessor:
    """Fixture providing a PDF processor instance."""
    return PDFProcessor()


@pytest.fixture
def sample_pdf() -> Path:
    """Fixture providing the path to the sample PDF file."""
    return Path("tests/data/sample_ar.pdf")


@pytest.mark.asyncio
async def test_pdf_processor_metadata(processor: PDFProcessor, sample_pdf: Path):
    """Test PDF metadata extraction."""
    metadata = await processor.get_metadata(sample_pdf)

    assert isinstance(metadata, DocumentMetadata)
    assert metadata.file_type == FileType.PDF
    assert metadata.total_pages > 0  # Should have at least one page

    # These might be None, but should exist
    assert hasattr(metadata, "title")
    assert hasattr(metadata, "author")
    assert hasattr(metadata, "creation_date")
    assert hasattr(metadata, "modification_date")

    # Custom metadata should include PDF-specific fields
    assert "producer" in metadata.custom_metadata
    assert "creator" in metadata.custom_metadata


@pytest.mark.asyncio
async def test_pdf_processor_content_extraction(
    processor: PDFProcessor, sample_pdf: Path
):
    """Test PDF content extraction."""
    # Get first page
    content_stream = processor.extract_content(sample_pdf, start_page=1, end_page=1)
    content = await content_stream.__anext__()

    assert isinstance(content, DocumentContent)
    assert content.content_type == "image/png"
    assert content.page_number == 1

    # Verify image metadata
    assert "dpi" in content.metadata
    assert "format" in content.metadata
    assert "width" in content.metadata
    assert "height" in content.metadata

    # Verify image content
    image_bytes = content.content
    assert isinstance(image_bytes, bytes)

    # Should be able to load as PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    assert image.format == "PNG"
    assert image.width > 0
    assert image.height > 0


@pytest.mark.asyncio
async def test_pdf_processor_multi_page(processor: PDFProcessor, sample_pdf: Path):
    """Test multi-page PDF processing."""
    metadata = await processor.get_metadata(sample_pdf)
    total_pages = metadata.total_pages

    # Process all pages
    pages = []
    async for content in processor.extract_content(sample_pdf):
        pages.append(content)

    assert len(pages) == total_pages
    assert all(isinstance(p, DocumentContent) for p in pages)
    assert [p.page_number for p in pages] == list(range(1, total_pages + 1))


@pytest.mark.asyncio
async def test_pdf_processor_page_range(processor: PDFProcessor, sample_pdf: Path):
    """Test PDF processing with specific page ranges."""
    # Process pages 2-3
    pages = []
    async for content in processor.extract_content(
        sample_pdf, start_page=2, end_page=3
    ):
        pages.append(content)

    assert len(pages) == 2
    assert [p.page_number for p in pages] == [2, 3]


@pytest.mark.asyncio
async def test_pdf_processor_invalid_file(processor: PDFProcessor, tmp_path: Path):
    """Test handling of invalid files."""
    # Non-existent file
    with pytest.raises(ProcessingError, match="does not exist"):
        await processor.get_metadata(tmp_path / "nonexistent.pdf")

    # Invalid file type
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("Not a PDF")
    with pytest.raises(ProcessingError, match="not supported"):
        await processor.get_metadata(invalid_file)


@pytest.mark.asyncio
async def test_pdf_processor_invalid_page_range(
    processor: PDFProcessor, sample_pdf: Path
):
    """Test handling of invalid page ranges."""
    # Start page < 1
    with pytest.raises(ProcessingError, match="start_page must be >= 1"):
        async for _ in processor.extract_content(sample_pdf, start_page=0):
            pass

    # End page < start page
    with pytest.raises(ProcessingError, match="end_page must be >= start_page"):
        async for _ in processor.extract_content(sample_pdf, start_page=2, end_page=1):
            pass


@pytest.mark.asyncio
async def test_pdf_processor_dpi_settings(sample_pdf: Path):
    """Test PDF processing with different DPI settings."""
    # Create processors with different DPI settings
    low_dpi = PDFProcessor(dpi=72)
    high_dpi = PDFProcessor(dpi=300)

    # Get first page from each
    low_content = await low_dpi.extract_content(sample_pdf, end_page=1).__anext__()
    high_content = await high_dpi.extract_content(sample_pdf, end_page=1).__anext__()

    # High DPI should result in larger images
    low_image = Image.open(io.BytesIO(low_content.content))
    high_image = Image.open(io.BytesIO(high_content.content))

    assert high_image.width > low_image.width
    assert high_image.height > low_image.height
