"""Translation algorithms for Tinbox."""

import asyncio
import re
from datetime import datetime
from typing import List, Optional

from rich.progress import Progress, TaskID

from tinbox.core.processor import DocumentContent
from tinbox.core.progress import ProgressTracker
from tinbox.core.translation.checkpoint import (
    CheckpointManager,
    TranslationState,
    should_resume,
    load_checkpoint,
    save_checkpoint,
)
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
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> TranslationResponse:
    """Translate a document using the specified algorithm.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    if config.algorithm == "page":
        return await translate_page_by_page(
            content,
            config,
            translator,
            progress,
            checkpoint_manager,
        )
    elif config.algorithm == "sliding-window":
        return await translate_sliding_window(
            content,
            config,
            translator,
            progress,
            checkpoint_manager,
        )
    elif config.algorithm == "context-aware":
        return await translate_context_aware(
            content,
            config,
            translator,
            progress,
            checkpoint_manager,
        )
    else:
        raise TranslationError(f"Unknown algorithm: {config.algorithm}")


async def translate_page_by_page(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> TranslationResponse:
    """Translate a document page by page.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    total_tokens = 0
    total_cost = 0.0
    translated_pages = []
    task_id: Optional[TaskID] = None
    start_time = datetime.now()

    try:
        # Set up progress tracking
        if progress:
            task_id = progress.add_task(
                "Translating pages...",
                total=len(content.pages),
            )

        # Check for checkpoint and resume if available
        translated_pages = []
        if checkpoint_manager and config.resume_from_checkpoint:
            checkpoint = await checkpoint_manager.load()
            if checkpoint:
                # Load existing translated pages in order
                translated_pages = [
                    checkpoint.translated_chunks[str(i)]
                    for i in range(1, len(checkpoint.translated_chunks) + 1)
                    if str(i) in checkpoint.translated_chunks
                ]
                total_tokens = checkpoint.token_usage
                total_cost = checkpoint.cost
                logger.info(f"Resumed with {len(translated_pages)} completed pages")
                if progress and task_id is not None:
                    progress.update(task_id, completed=len(translated_pages))

        # Track failed pages
        failed_pages: list[int] = []

        # Translate remaining pages
        for i, page in enumerate(
            content.pages[len(translated_pages) :], len(translated_pages)
        ):
            try:
                # Create translation request
                request = TranslationRequest(
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    content=page,
                    content_type=content.content_type,
                    model=config.model,
                    model_params={"model_name": config.model_name}
                    if config.model_name
                    else {},
                )

                # Translate page
                response = await translator.translate(request)
                translated_pages.append(response.text)
                total_tokens += response.tokens_used
                total_cost += response.cost

                # Update progress
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

                # Save checkpoint if needed
                if (
                    checkpoint_manager
                    and config.checkpoint_dir
                    and (i + 1) % config.checkpoint_frequency == 0
                ):
                    state = TranslationState(
                        source_lang=config.source_lang,
                        target_lang=config.target_lang,
                        algorithm="page",
                        completed_pages=list(range(1, len(translated_pages) + 1)),
                        failed_pages=failed_pages,
                        translated_chunks={
                            i + 1: text for i, text in enumerate(translated_pages)
                        },
                        token_usage=total_tokens,
                        cost=total_cost,
                        time_taken=(datetime.now() - start_time).total_seconds(),
                    )
                    await checkpoint_manager.save(state)

            except Exception as e:
                logger.error(f"Failed to translate page {i + 1}: {str(e)}")
                failed_pages.append(i + 1)

        if not translated_pages:
            if failed_pages:
                raise TranslationError(
                    f"Translation failed: No pages were successfully translated. Failed pages: {failed_pages}"
                )
            else:
                raise TranslationError(
                    f"Translation failed: No pages were successfully translated"
                )

        if failed_pages:
            logger.warning(f"Failed pages: {failed_pages}")

        # Join pages with double newlines
        final_text = "\n\n".join(translated_pages)

        time_taken = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            text=final_text,
            tokens_used=total_tokens,
            cost=total_cost,
            time_taken=time_taken,
        )

    except Exception as e:
        raise TranslationError(f"Translation failed: {str(e)}") from e


async def translate_sliding_window(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> TranslationResponse:
    """Translate a document using sliding window algorithm.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    logger.info("Starting sliding window translation")

    if isinstance(content.pages[0], bytes):
        raise TranslationError(
            "Sliding window algorithm not supported for image content"
        )

    start_time = datetime.now()
    total_tokens = 0
    total_cost = 0.0

    try:
        # Join all pages into single text
        text = "\n\n".join(content.pages)
        logger.debug(f"Combined text: {text}")

        # Use configured window size and overlap size, with fallbacks
        window_size = config.window_size if config.window_size is not None else 1000
        overlap_size = (
            config.overlap_size
            if config.overlap_size is not None
            else min(100, window_size // 4)
        )
        logger.info(
            f"Using window size: {window_size}, overlap size: {overlap_size}"
        )

        # Create windows
        windows = create_windows(
            text,
            window_size,
            overlap_size,
        )
        logger.info(f"Created {len(windows)} windows")

        # Check for checkpoint and resume if available
        translated_windows = []
        if checkpoint_manager and config.resume_from_checkpoint:
            logger.info("Checking for checkpoint")
            checkpoint = await checkpoint_manager.load()
            if checkpoint and checkpoint.translated_chunks:
                logger.info("Found valid checkpoint, resuming from saved state")
                # Load existing translated windows in order
                translated_windows = [
                    checkpoint.translated_chunks[str(i)]
                    for i in range(1, len(checkpoint.translated_chunks) + 1)
                    if str(i) in checkpoint.translated_chunks
                ]
                total_tokens = checkpoint.token_usage
                total_cost = checkpoint.cost
                logger.info(f"Resumed with {len(translated_windows)} completed windows")

        # Set up progress tracking
        if progress:
            task_id = progress.add_task(
                "Translating windows...",
                total=len(windows),
                completed=len(translated_windows),
            )

        # Translate remaining windows
        for i, window in enumerate(windows[len(translated_windows):], len(translated_windows)):
            logger.debug(f"Translating window {i + 1}: {window}")
            # Create translation request
            request = TranslationRequest(
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                content=window,
                content_type="text/plain",
                model=config.model,
                model_params={"model_name": config.model_name}
                if config.model_name
                else {},
            )

            # Translate window
            response = await translator.translate(request)
            translated_windows.append(response.text)
            total_tokens += response.tokens_used
            total_cost += response.cost

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            # Save checkpoint if needed
            if (
                checkpoint_manager
                and config.checkpoint_dir
                and (i + 1) % config.checkpoint_frequency == 0
            ):
                logger.debug(f"Saving checkpoint after window {i + 1}")
                state = TranslationState(
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    algorithm="sliding-window",
                    completed_pages=[
                        1
                    ],  # Sliding window treats the whole document as one page
                    failed_pages=[],
                    translated_chunks={
                        j + 1: text for j, text in enumerate(translated_windows)
                    },
                    token_usage=total_tokens,
                    cost=total_cost,
                    time_taken=(datetime.now() - start_time).total_seconds(),
                )
                await checkpoint_manager.save(state)

        # Merge windows
        final_text = merge_chunks(translated_windows, overlap_size)
        logger.debug(f"Final merged text: {final_text}")

        time_taken = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            text=final_text,
            tokens_used=total_tokens,
            cost=total_cost,
            time_taken=time_taken,
        )

    except Exception as e:
        logger.error(f"Error in sliding window translation: {str(e)}")
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
    logger.debug(f"Creating windows for text: {text}")
    logger.debug(f"Window size: {window_size}, Overlap size: {overlap_size}")

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
        logger.debug(f"Created window: {window}")
        windows.append(window)

        # If we've reached the end, break
        if end == len(text):
            break

        # Move start position, ensuring we make progress
        start = end - min(overlap_size, end - start)
        if start <= 0 or start >= end:
            break

    logger.debug(f"Created {len(windows)} windows")
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


def smart_text_split(
    text: str,
    target_size: int,
    custom_split_token: Optional[str] = None
) -> List[str]:
    """Split text at natural boundaries or custom tokens.

    Priority order:
    1. Custom split token (if provided) - ignores target_size
    2. Paragraph breaks (\\n\\n)
    3. Sentence endings ([.!?]\\s+)
    4. Line breaks (\\n)
    5. Clause boundaries ([;:,]\\s+)
    6. Word boundaries (\\s+)

    Args:
        text: Input text to split
        target_size: Target size for each chunk (characters)
        custom_split_token: Optional custom token to split on (ignores target_size)

    Returns:
        List of text chunks

    Raises:
        ValueError: If target_size is not positive or text is empty
    """
    if not text:
        return []
    
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    # If custom split token is provided, split on it and ignore target_size
    if custom_split_token:
        chunks = text.split(custom_split_token)
        # Remove empty chunks
        return [chunk for chunk in chunks if chunk]

    # If text is smaller than target size, return as single chunk
    if len(text) <= target_size:
        return [text]

    chunks = []
    current_pos = 0

    while current_pos < len(text):
        # Calculate the end position for this chunk
        end_pos = min(current_pos + target_size, len(text))
        
        # If we're at the end of the text, take the remaining text
        if end_pos == len(text):
            chunks.append(text[current_pos:end_pos])
            break

        # Try to find a good split point within the target size
        chunk_text = text[current_pos:end_pos]
        best_split_pos = len(chunk_text)  # Default to target size
        
        # Priority 1: Paragraph breaks (\n\n)
        paragraph_matches = list(re.finditer(r'\n\n', chunk_text))
        if paragraph_matches:
            # Take the last paragraph break within the chunk
            best_split_pos = paragraph_matches[-1].end()
        else:
            # Priority 2: Sentence endings ([.!?]\s+)
            sentence_matches = list(re.finditer(r'[.!?]\s+', chunk_text))
            if sentence_matches:
                best_split_pos = sentence_matches[-1].end()
            else:
                # Priority 3: Line breaks (\n)
                line_matches = list(re.finditer(r'\n', chunk_text))
                if line_matches:
                    best_split_pos = line_matches[-1].end()
                else:
                    # Priority 4: Clause boundaries ([;:,]\s+)
                    clause_matches = list(re.finditer(r'[;:,]\s+', chunk_text))
                    if clause_matches:
                        best_split_pos = clause_matches[-1].end()
                    else:
                        # Priority 5: Word boundaries (\s+)
                        word_matches = list(re.finditer(r'\s+', chunk_text))
                        if word_matches:
                            best_split_pos = word_matches[-1].end()
                        # If no word boundaries found, split at target size (fallback)

        # Extract the chunk up to the best split position
        actual_end = current_pos + best_split_pos
        chunk = text[current_pos:actual_end]
        
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move to the next position
        # Only skip ahead if we found no natural split point (best_split_pos == target_size)
        if best_split_pos == len(chunk_text) and current_pos + best_split_pos < len(text):
            # No natural boundary found, ensure we make progress by skipping whitespace
            current_pos = actual_end
            while current_pos < len(text) and text[current_pos].isspace():
                current_pos += 1
        else:
            # Natural boundary found, continue from there
            current_pos = actual_end

    return chunks


def build_translation_context(
    source_lang: str,
    target_lang: str,
    current_chunk: str,
    previous_chunk: Optional[str] = None,
    previous_translation: Optional[str] = None,
) -> str:
    """Build translation context with previous chunk and translation for better consistency.

    Args:
        source_lang: Source language code
        target_lang: Target language code  
        current_chunk: Current chunk to translate
        previous_chunk: Previous chunk (for context)
        previous_translation: Previous translation (for consistency)

    Returns:
        Formatted translation prompt with context
    """
    context = f"Translate the following text from {source_lang} to {target_lang}.\n\n"

    if previous_chunk and previous_translation:
        context += f"[PREVIOUS_SOURCE]\n{previous_chunk}\n[/PREVIOUS_SOURCE]\n\n"
        context += f"[PREVIOUS_TRANSLATION]\n{previous_translation}\n[/PREVIOUS_TRANSLATION]\n\n"

    context += f"[TRANSLATE_THIS]\n{current_chunk}\n[/TRANSLATE_THIS]\n\n"
    context += "Only return the translation of the text between [TRANSLATE_THIS] tags. Do not include the tags or any other content."

    return context


async def translate_context_aware(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> TranslationResponse:
    """Translate using context-aware algorithm with natural boundary splitting.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    if isinstance(content.pages[0], bytes):
        raise TranslationError(
            "Context-aware algorithm not supported for image content"
        )

    start_time = datetime.now()
    total_tokens = 0
    total_cost = 0.0
    task_id: Optional[TaskID] = None

    try:
        # Join all pages into single text
        text = "\n\n".join(content.pages)
        logger.info(f"Combined text length: {len(text)} characters")

        # Use configured context size with fallback
        context_size = config.context_size if config.context_size is not None else 2000
        logger.info(f"Using context size: {context_size} characters")

        # Split text using smart splitting
        chunks = smart_text_split(
            text,
            context_size,
            config.custom_split_token,
        )
        logger.debug(f"Chunks: {chunks}")
        logger.info(f"Created {len(chunks)} chunks using context-aware splitting")

        # Check for checkpoint and resume if available
        translated_chunks = []
        previous_chunk: Optional[str] = None
        previous_translation: Optional[str] = None
        
        if checkpoint_manager and config.resume_from_checkpoint:
            logger.info("Checking for checkpoint")
            checkpoint = await checkpoint_manager.load()
            if checkpoint and checkpoint.translated_chunks:
                logger.info("Found valid checkpoint, resuming from saved state")
                # Load existing translated chunks in order
                translated_chunks = [
                    checkpoint.translated_chunks[str(i)]
                    for i in range(1, len(checkpoint.translated_chunks) + 1)
                    if str(i) in checkpoint.translated_chunks
                ]
                total_tokens = checkpoint.token_usage
                total_cost = checkpoint.cost
                logger.info(f"Resumed with {len(translated_chunks)} completed chunks")
                
                # Set up context from the last completed chunk
                if translated_chunks and len(translated_chunks) > 0:
                    chunk_index = len(translated_chunks) - 1
                    if chunk_index < len(chunks):
                        previous_chunk = chunks[chunk_index]
                        previous_translation = translated_chunks[-1]

        # Set up progress tracking
        if progress:
            task_id = progress.add_task(
                "Translating chunks...",
                total=len(chunks),
                completed=len(translated_chunks),
            )

        # Translate remaining chunks with context
        for i, current_chunk in enumerate(chunks[len(translated_chunks):], len(translated_chunks)):
            logger.debug(f"Translating chunk {i + 1}/{len(chunks)}")
            
            # Build context for this chunk
            context_content = build_translation_context(
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                current_chunk=current_chunk,
                previous_chunk=previous_chunk,
                previous_translation=previous_translation,
            )

            # Create translation request
            request = TranslationRequest(
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                content=context_content,
                content_type="text/plain",
                model=config.model,
                model_params={"model_name": config.model_name}
                if config.model_name
                else {},
            )

            # Translate chunk
            response = await translator.translate(request)
            translated_chunks.append(response.text)
            total_tokens += response.tokens_used
            total_cost += response.cost

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            # Save checkpoint if needed
            if (
                checkpoint_manager
                and config.checkpoint_dir
                and (i + 1) % config.checkpoint_frequency == 0
            ):
                logger.debug(f"Saving checkpoint after chunk {i + 1}")
                state = TranslationState(
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    algorithm="context-aware",
                    completed_pages=[1],  # Context-aware treats the whole document as one page
                    failed_pages=[],
                    translated_chunks={
                        j + 1: text for j, text in enumerate(translated_chunks)
                    },
                    token_usage=total_tokens,
                    cost=total_cost,
                    time_taken=(datetime.now() - start_time).total_seconds(),
                )
                await checkpoint_manager.save(state)

            # Update context for next iteration
            previous_chunk = current_chunk
            previous_translation = response.text

        # Direct concatenation (no complex merging needed)
        final_text = "".join(translated_chunks)
        logger.info(f"Final translated text length: {len(final_text)} characters")

        time_taken = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            text=final_text,
            tokens_used=total_tokens,
            cost=total_cost,
            time_taken=time_taken,
        )

    except Exception as e:
        logger.error(f"Error in context-aware translation: {str(e)}")
        raise TranslationError(f"Translation failed: {str(e)}") from e
