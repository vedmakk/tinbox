"""Command-line interface for Tinbox."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from tinbox.core import FileType, ModelType, TranslationConfig, translate_document
from tinbox.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="tinbox",
    help="A CLI tool for translating documents using LLMs",
    add_completion=False,
)
console = Console()
logger = get_logger()


def version_callback(value: bool) -> None:
    """Print version information and exit."""
    if value:
        from tinbox import __version__
        console.print(f"Tinbox version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version information and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set the logging level.",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output logs in JSON format.",
    ),
) -> None:
    """Tinbox - A CLI Translation Tool using LLMs."""
    configure_logging(level=log_level, json=json_logs)


@app.command()
def translate(
    input_file: Path = typer.Argument(
        ...,
        help="The input file to translate.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="The output file. If not specified, prints to stdout.",
    ),
    source_lang: str = typer.Option(
        None,
        "--from",
        "-f",
        help="Source language code (e.g., 'en', 'zh'). Defaults to auto-detect.",
    ),
    target_lang: str = typer.Option(
        "en",
        "--to",
        "-t",
        help="Target language code. Defaults to 'en' (English).",
    ),
    model: ModelType = typer.Option(
        ModelType.CLAUDE_3_SONNET,
        "--model",
        "-m",
        help="The model to use for translation.",
    ),
    algorithm: str = typer.Option(
        "page",
        "--algorithm",
        "-a",
        help="Translation algorithm to use: 'page' or 'sliding-window'.",
    ),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        "-b",
        help="Enable benchmarking mode.",
    ),
) -> None:
    """Translate a document using LLMs."""
    try:
        # Determine file type
        file_type = FileType(input_file.suffix.lstrip(".").lower())
        
        # Create configuration
        config = TranslationConfig(
            source_lang=source_lang or "auto",
            target_lang=target_lang,
            model=model,
            algorithm=algorithm,
            input_file=input_file,
            output_file=output_file,
            benchmark=benchmark,
        )
        
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Translating document...", total=None)
            result = translate_document(config)
        
        # Output results
        if output_file:
            output_file.write_text(result.text)
            console.print(f"Translation saved to: {output_file}")
        else:
            console.print(result.text)
        
        # Show benchmarks if requested
        if benchmark:
            console.print("\n[bold]Benchmark Results:[/bold]")
            console.print(f"Time taken: {result.time_taken:.2f}s")
            console.print(f"Tokens used: {result.tokens_used}")
            console.print(f"Estimated cost: ${result.cost:.4f}")
            
    except Exception as e:
        logger.exception("Translation failed")
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app() 
