"""Translation algorithms for Tinbox."""

import asyncio
from datetime import datetime
from typing import Optional

from rich.progress import Progress, TaskID

from tinbox.core.processor import DocumentContent, DocumentPages
from tinbox.core.progress import ProgressTracker
from tinbox.core.translation.checkpoint import CheckpointManager, TranslationState
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import TranslationConfig, TranslationResult
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class TranslationProgress:
    """Track translation progress and statistics."""

    def __init__(
        self,
        total_pages: int,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> None:
        """Initialize translation progress tracking.

        Args:
            total_pages: Total number of pages to translate
            progress: Optional Rich progress instance
            task_id: Optional task ID for the progress bar
        """
        self.total_pages = total_pages
        self.completed_pages = 0
        self.failed_pages: list[int] = []
        self.token_usage = 0
        self.cost = 0.0
        self.start_time = datetime.now()
        self._progress = progress
        self._task_id = task_id

    @property
    def time_taken(self) -> float:
        """Get the total time taken so far."""
        return (datetime.now() - self.start_time).total_seconds()

    def update(
        self,
        page_number: int,
        response: Optional[TranslationResponse] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Update progress with page results.

        Args:
            page_number: The page number that was processed
            response: Optional translation response
            error: Optional error that occurred
        """
        if error:
            self.failed_pages.append(page_number)
            logger.error(
                f"Translation failed for page {page_number}",
                error=str(error),
                page=page_number,
            )
        else:
            self.completed_pages += 1
            if response:
                self.token_usage += response.tokens_used
                self.cost += response.cost

        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.completed_pages,
                total=self.total_pages,
            )


async def translate_document(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
) -> TranslationResult:
    """Main translation function that delegates to appropriate algorithm.

    Args:
        content: Document content to translate
        config: Translation configuration
        translator: Model interface for translation
        progress: Optional progress tracking

    Returns:
        Translation result with combined text and metadata

    Raises:
        TranslationError: If translation fails
    """
    if config.algorithm == "page":
        return await translate_page_by_page(content.pages, config, translator, progress)
    else:
        return await translate_sliding_window(
            content.pages, config, translator, progress
        )


async def translate_page_by_page(
    content: DocumentPages,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
) -> TranslationResult:
    """Translate document page by page with seam repair.

    Args:
        content: Document content to translate
        config: Translation configuration
        translator: Model interface for translation
        progress: Optional progress tracking

    Returns:
        Translation result with combined text and metadata

    Raises:
        TranslationError: If translation fails
    """
    # Initialize checkpoint manager if configured
    checkpoint_manager = None
    if config.checkpoint_dir:
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    # Try to resume from checkpoint
    initial_state = None
    if checkpoint_manager and config.resume_from_checkpoint:
        initial_state = checkpoint_manager.load_checkpoint(config.input_file)

    # Initialize progress tracking
    task_id = None
    if progress:
        task_id = progress.add_task(
            description="Translating pages...",
            total=len(content.pages),
        )

    # Estimate total tokens (rough estimate)
    total_tokens = len(content.pages) * 500  # Rough estimate of 500 tokens per page

    tracker = ProgressTracker(
        total_tokens=total_tokens,
        total_pages=len(content.pages),
        progress=progress,
        task_id=task_id,
        callback=config.progress_callback,
        verbose=config.verbose,
    )

    # Initialize state from checkpoint if available
    translated_chunks = {}
    if initial_state:
        for i in range(1, len(initial_state.completed_pages) + 1):
            translated_chunks[i] = initial_state.translated_chunks[i]
            tracker.update(
                tokens_processed=500,  # Rough estimate
                cost=initial_state.cost / len(initial_state.completed_pages),
                page_completed=True,
            )

    try:
        # Translate each page
        for i, page in enumerate(content.pages, start=1):
            # Skip already translated pages
            if initial_state and i in initial_state.completed_pages:
                continue

            try:
                # Create translation request
                request = TranslationRequest(
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    content=page,
                    content_type=content.content_type,
                    model=config.model,
                )

                # Translate page
                response = await translator.translate(request)
                translated_chunks[i] = response.text
                tracker.update(
                    tokens_processed=response.tokens_used,
                    cost=response.cost,
                    page_completed=True,
                )

                # Save checkpoint if configured
                if checkpoint_manager and i % config.checkpoint_frequency == 0:
                    state = TranslationState(
                        input_file=config.input_file,
                        source_lang=config.source_lang,
                        target_lang=config.target_lang,
                        algorithm="page",
                        completed_pages=list(range(1, i + 1)),
                        failed_pages=tracker.stats.failed_pages,  # Include failed pages from tracker
                        translated_chunks=translated_chunks,
                        token_usage=tracker.stats.tokens_processed,
                        cost=tracker.stats.cost,
                        time_taken=tracker.stats.time_taken,
                    )
                    checkpoint_manager.save_checkpoint(state)

            except Exception as e:
                logger.error(
                    f"Failed to translate page {i}",
                    error=str(e),
                    page=i,
                )
                # Track failed page
                tracker.update(
                    tokens_processed=0,
                    cost=0.0,
                    page_failed=i,
                )
                # Continue with next page
                continue

        if not translated_chunks:
            failed_pages = sorted(tracker.stats.failed_pages)
            raise TranslationError(
                f"No pages were successfully translated. Failed pages: {failed_pages}"
            )

        # Convert chunks to list in order
        translated_pages = [
            translated_chunks[i]
            for i in range(1, len(content.pages) + 1)
            if i in translated_chunks
        ]

        # Repair seams if needed
        if len(translated_pages) > 1:
            final_text = await repair_seams(
                translated_pages,
                config,
                translator,
            )
        elif translated_pages:  # Check if we have at least one page
            final_text = translated_pages[0]
        else:
            raise TranslationError("No valid translations available")

        # Clean up old checkpoints
        if checkpoint_manager:
            checkpoint_manager.cleanup_old_checkpoints(config.input_file)

        return TranslationResult(
            text=final_text,
            tokens_used=tracker.stats.tokens_processed,
            cost=tracker.stats.cost,
            time_taken=tracker.stats.time_taken,
        )

    except Exception as e:
        raise TranslationError(f"Translation failed: {str(e)}") from e


async def translate_sliding_window(
    content: DocumentPages,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
) -> TranslationResult:
    """Translate document using sliding window approach.

    Args:
        content: Document content to translate
        config: Translation configuration
        translator: Model interface for translation
        progress: Optional progress tracking

    Returns:
        Translation result with combined text and metadata

    Raises:
        TranslationError: If translation fails
    """
    if not isinstance(content.content, str):
        raise TranslationError("Sliding window translation only supports text content")

    # Initialize checkpoint manager if configured
    checkpoint_manager = None
    if config.checkpoint_dir:
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    # Try to resume from checkpoint
    initial_state = None
    if checkpoint_manager and config.resume_from_checkpoint:
        initial_state = checkpoint_manager.load_checkpoint(config.input_file)

    # Create windows
    text = content.content
    windows = create_windows(
        text,
        window_size=config.window_size,
        overlap_size=config.overlap_size,
    )

    # Initialize progress tracking
    task_id = None
    if progress:
        task_id = progress.add_task(
            description="Translating chunks...",
            total=len(windows),
        )

    # Estimate total tokens (rough estimate)
    total_tokens = len(text) // 4  # Rough estimate of 1 token per 4 characters

    tracker = ProgressTracker(
        total_tokens=total_tokens,
        total_pages=len(windows),  # Treat windows as pages
        progress=progress,
        task_id=task_id,
        callback=config.progress_callback,
        verbose=config.verbose,
    )

    # Initialize state from checkpoint if available
    translated_chunks = {}
    if initial_state:
        for i in range(1, len(initial_state.completed_pages) + 1):
            translated_chunks[i] = initial_state.translated_chunks[i]
            tracker.update(
                tokens_processed=len(initial_state.translated_chunks[i])
                // 4,  # Rough estimate
                cost=initial_state.cost / len(initial_state.completed_pages),
                page_completed=True,
            )

    try:
        # Translate each window
        for i, window in enumerate(windows, start=1):
            # Skip already translated windows
            if initial_state and i in initial_state.completed_pages:
                continue

            try:
                # Create translation request
                request = TranslationRequest(
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    content=window,
                    content_type="text/plain",
                    model=config.model,
                )

                # Translate window
                response = await translator.translate(request)
                translated_chunks[i] = response.text
                tracker.update(
                    tokens_processed=response.tokens_used,
                    cost=response.cost,
                    page_completed=True,
                )

                # Save checkpoint if configured
                if checkpoint_manager and i % config.checkpoint_frequency == 0:
                    state = TranslationState(
                        input_file=config.input_file,
                        source_lang=config.source_lang,
                        target_lang=config.target_lang,
                        algorithm="sliding-window",
                        completed_pages=list(range(1, i + 1)),
                        failed_pages=tracker.stats.failed_pages,  # Include failed pages from tracker
                        translated_chunks=translated_chunks,
                        token_usage=tracker.stats.tokens_processed,
                        cost=tracker.stats.cost,
                        time_taken=tracker.stats.time_taken,
                    )
                    checkpoint_manager.save_checkpoint(state)

            except Exception as e:
                logger.error(
                    f"Failed to translate window {i}",
                    error=str(e),
                    window=i,
                )
                # Track failed page
                tracker.update(
                    tokens_processed=0,
                    cost=0.0,
                    page_failed=i,
                )
                # Continue with next window
                continue

        if not translated_chunks:
            failed_pages = sorted(tracker.stats.failed_pages)
            raise TranslationError(
                f"No chunks were successfully translated. Failed pages: {failed_pages}"
            )

        # Convert chunks to list in order
        translated_chunks_list = [
            translated_chunks[i]
            for i in range(1, len(windows) + 1)
            if i in translated_chunks
        ]

        # Merge chunks
        final_text = merge_chunks(
            translated_chunks_list,
            overlap_size=config.overlap_size,
        )

        # Clean up old checkpoints
        if checkpoint_manager:
            checkpoint_manager.cleanup_old_checkpoints(config.input_file)

        return TranslationResult(
            text=final_text,
            tokens_used=tracker.stats.tokens_processed,
            cost=tracker.stats.cost,
            time_taken=tracker.stats.time_taken,
        )

    except Exception as e:
        raise TranslationError(f"Translation failed: {str(e)}") from e


async def repair_seams(
    pages: list[str],
    config: TranslationConfig,
    translator: ModelInterface,
) -> str:
    """Repair seams between translated pages.

    Args:
        pages: List of translated pages
        config: Translation configuration
        translator: Model interface for translation

    Returns:
        Combined text with repaired seams

    Raises:
        TranslationError: If seam repair fails
    """
    if len(pages) <= 1:
        return pages[0] if pages else ""

    try:
        result = [pages[0]]
        for i in range(1, len(pages)):
            # Extract overlapping content
            seam = extract_seam(result[-1], pages[i], config.page_seam_overlap)
            if seam:
                # Update page with repaired seam
                result.append(update_page_with_seam(pages[i], seam))
            else:
                result.append(pages[i])

        return "\n\n".join(result)

    except Exception as e:
        raise TranslationError(f"Failed to repair seams: {str(e)}") from e


def create_windows(
    text: str,
    window_size: int,
    overlap_size: int,
) -> list[str]:
    """Create overlapping windows from text.

    Args:
        text: Input text
        window_size: Size of each window in characters
        overlap_size: Size of overlap between windows

    Returns:
        List of text windows
    """
    if not text:
        return []

    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if overlap_size < 0:
        raise ValueError("Overlap size must be non-negative")
    if overlap_size >= window_size:
        raise ValueError("Overlap size must be less than window size")

    windows = []
    start = 0

    while start < len(text):
        # Calculate end position
        end = min(start + window_size, len(text))

        # Extract window
        window = text[start:end]
        windows.append(window)

        # If we've reached the end, break
        if end == len(text):
            break

        # Move start position, ensuring we make progress
        start = end - min(overlap_size, end - start)
        if start <= 0 or start >= end:
            break

    return windows


def merge_chunks(chunks: list[str], overlap_size: int) -> str:
    """Merge translated chunks, handling overlaps.

    Args:
        chunks: List of translated chunks
        overlap_size: Size of overlap between chunks

    Returns:
        Merged text
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    # If no overlap is requested, just join the chunks
    if overlap_size == 0:
        return "".join(chunks)

    result = chunks[0]
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]

        # Try to find the overlap at the end of the result and start of current chunk
        for overlap_len in range(
            min(len(result), len(current_chunk), overlap_size), 0, -1
        ):
            end_of_result = result[-overlap_len:]
            start_of_chunk = current_chunk[:overlap_len]

            if end_of_result == start_of_chunk:
                # Found overlap, append only the non-overlapping part
                result += "\n\n" + current_chunk[overlap_len:]
                break
        else:
            # No overlap found, append entire chunk
            result += "\n\n" + current_chunk

    return result


def extract_seam(text1: str, text2: str, overlap_size: int) -> str:
    """Extract overlapping content between two texts.

    Args:
        text1: First text
        text2: Second text
        overlap_size: Size of overlap to look for

    Returns:
        Overlapping content or empty string if no overlap found
    """
    if not text1 or not text2 or overlap_size <= 0:
        return ""

    # Get end of first text
    end1 = text1[-overlap_size:]
    if not end1:
        return ""

    # Look for overlap in second text
    start_pos = text2.find(end1)
    if start_pos >= 0:
        return text2[start_pos : start_pos + overlap_size]

    return ""


def update_page_with_seam(page: str, seam: str) -> str:
    """Update page text with repaired seam.

    Args:
        page: Page text
        seam: Seam content

    Returns:
        Updated page text
    """
    if not seam:
        return page

    # Find seam position
    pos = page.find(seam)
    if pos >= 0:
        # Keep text after seam
        return page[pos + len(seam) :]

    return page
