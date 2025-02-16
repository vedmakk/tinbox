"""Example script demonstrating document processing."""

import asyncio
from pathlib import Path

from rich.console import Console

from tinbox.core.processor import ProcessingError
from tinbox.core.processors import PDFProcessor, TextProcessor, WordProcessor
from tinbox.core.types import FileType

console = Console()


async def process_document(file_path: Path) -> None:
    """Process a document and display its content and metadata.

    Args:
        file_path: Path to the document to process
    """
    # Determine file type
    try:
        file_type = FileType(file_path.suffix.lstrip(".").lower())
    except ValueError:
        console.print(f"[red]Unsupported file type: {file_path.suffix}[/red]")
        return

    # Select appropriate processor
    processors = {
        FileType.PDF: PDFProcessor(),
        FileType.DOCX: WordProcessor(),
        FileType.TXT: TextProcessor(),
    }
    processor = processors.get(file_type)
    if not processor:
        console.print(f"[red]No processor available for {file_type}[/red]")
        return

    try:
        # Get metadata
        console.print("\n[bold blue]Document Metadata:[/bold blue]")
        metadata = await processor.get_metadata(file_path)
        console.print(metadata.model_dump_json(indent=2))

        # Extract content
        console.print("\n[bold blue]Document Content:[/bold blue]")
        async for content in processor.extract_content(file_path):
            console.print(f"\n[bold]Page {content.page_number}:[/bold]")
            console.print(f"Content Type: {content.content_type}")
            console.print(f"Metadata: {content.metadata}")

            if content.content_type == "text/plain":
                # For text content, display it directly
                console.print("\n[bold green]Text Content:[/bold green]")
                console.print(content.content)
            else:
                # For binary content (e.g., PDF images), show size
                console.print(
                    f"\n[bold yellow]Binary content of size {len(content.content)} bytes[/bold yellow]"
                )

    except ProcessingError as e:
        console.print(f"[red]Error processing document: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")


async def main() -> None:
    """Main entry point."""
    # Process sample Arabic Word document
    sample_path = Path("tests/data/sample_ar.docx")
    if not sample_path.exists():
        console.print("[red]Sample document not found![/red]")
        return

    console.print(f"[bold]Processing {sample_path}...[/bold]")
    await process_document(sample_path)


if __name__ == "__main__":
    asyncio.run(main())
