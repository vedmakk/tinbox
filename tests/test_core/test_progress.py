"""Tests for progress tracking functionality."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from rich.progress import Progress

from tinbox.core.progress import ProgressStats, ProgressTracker


def test_progress_stats_calculations():
    """Test progress statistics calculations."""
    # Test percentage calculation
    stats = ProgressStats(
        tokens_processed=500,
        tokens_total=1000,
        pages_processed=2,
        pages_total=4,
        cost=0.05,
    )
    assert stats.percent_complete == 50.0

    # Test tokens per second with mocked time
    with patch("tinbox.core.progress.datetime") as mock_datetime:
        # Create fixed timestamps for testing
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        current_time = datetime(2024, 1, 1, 12, 0, 10)  # 10 seconds later

        # Mock datetime.now() to return our fixed timestamps
        mock_datetime.now = MagicMock(return_value=current_time)

        # Create new stats object with explicit start_time
        stats = ProgressStats(
            tokens_processed=500,
            tokens_total=1000,
            start_time=start_time,  # Explicitly set start_time
        )

        # Calculate tokens per second (should be 500 tokens / 10 seconds = 50 tokens/sec)
        assert abs(stats.tokens_per_second - 50.0) < 0.1

    # Test edge cases
    with patch("tinbox.core.progress.datetime") as mock_datetime:
        now = datetime.now()
        mock_datetime.now = MagicMock(return_value=now)
        stats = ProgressStats(tokens_total=0)
        assert stats.percent_complete == 0.0
        assert stats.tokens_per_second >= 0.0
        assert stats.estimated_time_remaining == float("inf")


def test_progress_tracker_initialization():
    """Test progress tracker initialization."""
    progress = Progress()
    task_id = progress.add_task("Test")
    callback = MagicMock()

    tracker = ProgressTracker(
        total_tokens=1000,
        total_pages=4,
        progress=progress,
        task_id=task_id,
        callback=callback,
        verbose=True,
    )

    assert tracker.stats.tokens_total == 1000
    assert tracker.stats.pages_total == 4
    assert tracker._progress == progress
    assert tracker._task_id == task_id
    assert tracker._callback == callback
    assert tracker._verbose is True


def test_progress_tracker_updates():
    """Test progress tracker updates."""
    progress = Progress()
    task_id = progress.add_task("Test")
    callback = MagicMock()

    tracker = ProgressTracker(
        total_tokens=1000,
        total_pages=4,
        progress=progress,
        task_id=task_id,
        callback=callback,
        verbose=True,
    )

    # Test basic update
    tracker.update(tokens_processed=100, cost=0.01)
    assert tracker.stats.tokens_processed == 100
    assert tracker.stats.cost == 0.01
    assert tracker.stats.pages_processed == 0
    callback.assert_called_once_with(100)

    # Test page completion
    tracker.update(tokens_processed=150, cost=0.015, page_completed=True)
    assert tracker.stats.tokens_processed == 250
    assert tracker.stats.cost == 0.025
    assert tracker.stats.pages_processed == 1
    callback.assert_called_with(250)


def test_progress_tracker_without_progress_bar():
    """Test progress tracker without progress bar."""
    tracker = ProgressTracker(
        total_tokens=1000,
        total_pages=4,
        verbose=False,
    )

    # Should not raise any errors when updating without progress bar
    tracker.update(tokens_processed=100, cost=0.01)
    assert tracker.stats.tokens_processed == 100


@pytest.mark.parametrize(
    "verbose,expected_description",
    [
        (True, "Processed 100 tokens (10.0%)"),
        (False, "Translating..."),
    ],
)
def test_progress_tracker_verbose_mode(verbose, expected_description):
    """Test progress tracker verbose mode."""
    mock_progress = MagicMock(spec=Progress)
    task_id = mock_progress.add_task.return_value = 1

    tracker = ProgressTracker(
        total_tokens=1000,
        total_pages=4,
        progress=mock_progress,
        task_id=task_id,
        verbose=verbose,
    )

    # Reset mock for testing update
    mock_progress.reset_mock()

    # Test update
    tracker.update(tokens_processed=100)
    mock_progress.update.assert_called_once()
    description = mock_progress.update.call_args[1]["description"]
    assert description.startswith(expected_description)
