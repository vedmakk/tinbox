"""Progress tracking functionality for Tinbox."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from rich.progress import Progress, TaskID

from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressStats:
    """Statistics for translation progress."""

    tokens_processed: int = 0
    tokens_total: int = 0
    pages_processed: int = 0
    pages_total: int = 0
    failed_pages: list[int] = field(default_factory=list)
    cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def time_taken(self) -> float:
        """Get the total time taken so far."""
        current_time = datetime.now()
        delta = current_time - self.start_time
        return max(delta.total_seconds(), 0.001)  # Avoid division by zero

    @property
    def percent_complete(self) -> float:
        """Get the percentage complete."""
        if self.tokens_total == 0:
            return 0.0
        return (self.tokens_processed / self.tokens_total) * 100

    @property
    def tokens_per_second(self) -> float:
        """Get the tokens processed per second."""
        time_taken = self.time_taken
        if time_taken <= 0:
            return 0.0
        return self.tokens_processed / time_taken

    @property
    def estimated_time_remaining(self) -> float:
        """Get the estimated time remaining in seconds."""
        if self.tokens_per_second == 0:
            return float("inf")
        tokens_remaining = self.tokens_total - self.tokens_processed
        return tokens_remaining / self.tokens_per_second


class ProgressTracker:
    """Track translation progress and update UI."""

    def __init__(
        self,
        total_tokens: int,
        total_pages: int,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
        callback: Optional[Callable[[int], None]] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize progress tracking.

        Args:
            total_tokens: Total number of tokens to process
            total_pages: Total number of pages to process
            progress: Optional Rich progress instance
            task_id: Optional task ID for the progress bar
            callback: Optional callback function for progress updates
            verbose: Whether to show detailed progress information
        """
        self.stats = ProgressStats(
            tokens_total=total_tokens,
            pages_total=total_pages,
        )
        self._progress = progress
        self._task_id = task_id
        self._callback = callback
        self._verbose = verbose

        # Initialize progress bar if available
        if self._progress and self._task_id:
            self._progress.update(
                self._task_id,
                total=total_tokens,
                description="Translating..." if not verbose else "",
            )

    def update(
        self,
        tokens_processed: int,
        cost: float = 0.0,
        page_completed: bool = False,
        page_failed: Optional[int] = None,
    ) -> None:
        """Update progress tracking.

        Args:
            tokens_processed: Number of new tokens processed
            cost: Cost of processing these tokens
            page_completed: Whether a full page was completed
            page_failed: Optional page number that failed
        """
        self.stats.tokens_processed += tokens_processed
        self.stats.cost += cost
        if page_completed:
            self.stats.pages_processed += 1
        if page_failed is not None:
            self.stats.failed_pages.append(page_failed)

        # Update progress bar if available
        if self._progress and self._task_id is not None:
            description = (
                f"Processed {self.stats.tokens_processed:,} tokens ({self.stats.percent_complete:.1f}%)"
                if self._verbose
                else "Translating..."
            )
            if self.stats.failed_pages:
                description += f" ({len(self.stats.failed_pages)} pages failed)"
            self._progress.update(
                self._task_id,
                completed=self.stats.pages_processed,
                description=description,
            )

        # Call progress callback if provided
        if self._callback:
            self._callback(self.stats.tokens_processed)

        # Log progress in verbose mode
        if self._verbose:
            logger.info(
                "Translation progress",
                tokens_processed=self.stats.tokens_processed,
                tokens_total=self.stats.tokens_total,
                pages_processed=self.stats.pages_processed,
                pages_total=self.stats.pages_total,
                cost=self.stats.cost,
                time_taken=self.stats.time_taken,
                estimated_remaining=self.stats.estimated_time_remaining,
            )
