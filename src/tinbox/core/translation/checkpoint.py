"""Checkpoint management for translation tasks."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class TranslationState(BaseModel):
    """State of a translation task for checkpointing."""

    # Task identification
    input_file: Path
    source_lang: str
    target_lang: str
    algorithm: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Progress tracking
    completed_pages: List[int]
    failed_pages: List[int]
    translated_chunks: Dict[int, str]  # page/chunk number -> translated text
    token_usage: int = 0
    cost: float = 0.0
    time_taken: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class CheckpointManager:
    """Manages saving and loading translation checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self._logger = logger

    def _get_checkpoint_path(self, input_file: Path) -> Path:
        """Get the checkpoint file path for a given input file.

        Args:
            input_file: The input file being translated

        Returns:
            Path to the checkpoint file
        """
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use input filename with timestamp for checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{input_file.stem}_{timestamp}.checkpoint.json"
        return self.checkpoint_dir / checkpoint_name

    def _get_latest_checkpoint(self, input_file: Path) -> Optional[Path]:
        """Get the most recent checkpoint for a given input file.

        Args:
            input_file: The input file being translated

        Returns:
            Path to the latest checkpoint file, or None if no checkpoint exists
        """
        if not self.checkpoint_dir.exists():
            return None

        # Find all checkpoints for this file
        checkpoints = list(
            self.checkpoint_dir.glob(f"{input_file.stem}_*.checkpoint.json")
        )
        if not checkpoints:
            return None

        # Return most recent checkpoint
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def save_checkpoint(self, state: TranslationState) -> None:
        """Save the current translation state to a checkpoint file.

        Args:
            state: The current translation state
        """
        try:
            checkpoint_path = self._get_checkpoint_path(state.input_file)

            # Convert state to JSON
            state_dict = state.model_dump()
            # Convert paths to strings
            state_dict["input_file"] = str(state_dict["input_file"])

            # Save to file
            checkpoint_path.write_text(json.dumps(state_dict, indent=2))

            self._logger.info(
                "Saved translation checkpoint",
                checkpoint=str(checkpoint_path),
                pages_completed=len(state.completed_pages),
            )

        except Exception as e:
            self._logger.error(
                "Failed to save checkpoint",
                error=str(e),
                checkpoint=str(checkpoint_path),
            )
            raise

    def load_checkpoint(self, input_file: Path) -> Optional[TranslationState]:
        """Load the most recent checkpoint for a given input file.

        Args:
            input_file: The input file being translated

        Returns:
            The loaded translation state, or None if no checkpoint exists
        """
        try:
            checkpoint_path = self._get_latest_checkpoint(input_file)
            if not checkpoint_path:
                return None

            # Load and parse JSON
            state_dict = json.loads(checkpoint_path.read_text())
            # Convert path strings back to Path objects
            state_dict["input_file"] = Path(state_dict["input_file"])

            # Create TranslationState from dict
            state = TranslationState(**state_dict)

            self._logger.info(
                "Loaded translation checkpoint",
                checkpoint=str(checkpoint_path),
                pages_completed=len(state.completed_pages),
            )

            return state

        except Exception as e:
            self._logger.error(
                "Failed to load checkpoint",
                error=str(e),
                input_file=str(input_file),
            )
            return None

    def cleanup_old_checkpoints(self, input_file: Path, keep_latest: int = 5) -> None:
        """Clean up old checkpoints, keeping only the N most recent.

        Args:
            input_file: The input file to clean checkpoints for
            keep_latest: Number of recent checkpoints to keep
        """
        try:
            if not self.checkpoint_dir.exists():
                return

            # Find all checkpoints for this file
            checkpoints = list(
                self.checkpoint_dir.glob(f"{input_file.stem}_*.checkpoint.json")
            )
            if len(checkpoints) <= keep_latest:
                return

            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old checkpoints
            for checkpoint in checkpoints[keep_latest:]:
                checkpoint.unlink()
                self._logger.info(
                    "Removed old checkpoint",
                    checkpoint=str(checkpoint),
                )

        except Exception as e:
            self._logger.error(
                "Failed to cleanup checkpoints",
                error=str(e),
                input_file=str(input_file),
            )
