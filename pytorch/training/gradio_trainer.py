"""
Gradio-compatible training wrapper.

This module provides a non-blocking training interface specifically designed
for Gradio GUIs. It wraps the standard Trainer class to run in a background
thread with real-time progress callbacks.
"""

import torch
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass, asdict

from .trainer import Trainer
from ..config.training_config import TrainingConfig
from ..model.transformer import MusicTransformer
from torch.utils.data import DataLoader


@dataclass
class TrainingMetrics:
    """Real-time training metrics for GUI display."""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    perplexity: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    epoch_time: float = 0.0
    elapsed_time: float = 0.0
    status: str = "idle"  # idle, running, paused, completed, error
    message: str = ""
    best_val_loss: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GradioTrainer:
    """
    Non-blocking trainer wrapper for Gradio interfaces.

    Features:
    - Runs training in background thread
    - Real-time metric updates via callbacks
    - Pause/resume/stop controls
    - Thread-safe metric collection
    - Graceful shutdown with checkpoint saving

    Example:
        >>> trainer = GradioTrainer(model, config, train_loader, val_loader)
        >>> trainer.start_training(progress_callback=update_gui)
        >>> # ... later ...
        >>> trainer.pause_training()
        >>> trainer.resume_training()
        >>> trainer.stop_training()
    """

    def __init__(
        self,
        model: MusicTransformer,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Initialize Gradio trainer.

        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training thread
        self.training_thread: Optional[threading.Thread] = None
        self.metrics_queue: queue.Queue = queue.Queue()
        self.log_queue: queue.Queue = queue.Queue()

        # Control flags (thread-safe)
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._running = threading.Event()

        # Current metrics
        self.current_metrics = TrainingMetrics()

        # Trainer instance (created when training starts)
        self.trainer: Optional[Trainer] = None

        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable] = None

        # Training start time
        self.start_time: Optional[float] = None

    def start_training(
        self,
        progress_callback: Optional[Callable[[TrainingMetrics], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Start training in background thread.

        Args:
            progress_callback: Function called with updated metrics
            log_callback: Function called with log messages

        Returns:
            True if training started successfully, False otherwise
        """
        if self._running.is_set():
            self._log("Training is already running", "WARNING")
            return False

        # Store callbacks
        self.progress_callback = progress_callback
        self.log_callback = log_callback

        # Reset flags
        self._stop_flag.clear()
        self._pause_flag.clear()
        self._running.set()

        # Reset metrics
        self.current_metrics = TrainingMetrics(
            status="running",
            message="Training started",
            total_steps=self.config.get_total_steps(len(self.train_loader.dataset))
        )
        self.start_time = time.time()

        # Create training thread
        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True
        )
        self.training_thread.start()

        self._log("Training started in background thread", "INFO")
        return True

    def pause_training(self) -> bool:
        """
        Pause training.

        Returns:
            True if paused successfully, False otherwise
        """
        if not self._running.is_set():
            self._log("Training is not running", "WARNING")
            return False

        if self._pause_flag.is_set():
            self._log("Training is already paused", "WARNING")
            return False

        self._pause_flag.set()
        self.current_metrics.status = "paused"
        self.current_metrics.message = "Training paused"
        self._log("Training paused", "INFO")
        return True

    def resume_training(self) -> bool:
        """
        Resume paused training.

        Returns:
            True if resumed successfully, False otherwise
        """
        if not self._running.is_set():
            self._log("Training is not running", "WARNING")
            return False

        if not self._pause_flag.is_set():
            self._log("Training is not paused", "WARNING")
            return False

        self._pause_flag.clear()
        self.current_metrics.status = "running"
        self.current_metrics.message = "Training resumed"
        self._log("Training resumed", "INFO")
        return True

    def stop_training(self, save_checkpoint: bool = True) -> bool:
        """
        Stop training gracefully.

        Args:
            save_checkpoint: Whether to save checkpoint before stopping

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self._running.is_set():
            self._log("Training is not running", "WARNING")
            return False

        self._log("Stopping training...", "INFO")
        self._stop_flag.set()

        # Wait for training thread to finish (with timeout)
        if self.training_thread is not None:
            self.training_thread.join(timeout=10.0)

        # Save checkpoint if requested
        if save_checkpoint and self.trainer is not None:
            try:
                checkpoint_path = self.config.checkpoint_dir / "stopped_checkpoint.pt"
                from ..utils.checkpoint_utils import save_checkpoint
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=self.trainer.model,
                    optimizer=self.trainer.optimizer,
                    scheduler=self.trainer.scheduler,
                    epoch=self.trainer.current_epoch,
                    step=self.trainer.global_step,
                    best_val_loss=self.trainer.best_val_loss,
                    config=self.config.to_dict()
                )
                self._log(f"Checkpoint saved to: {checkpoint_path}", "INFO")
            except Exception as e:
                self._log(f"Failed to save checkpoint: {e}", "ERROR")

        self._running.clear()
        self.current_metrics.status = "completed"
        self.current_metrics.message = "Training stopped by user"
        self._log("Training stopped", "INFO")
        return True

    def is_running(self) -> bool:
        """Check if training is currently running."""
        return self._running.is_set()

    def is_paused(self) -> bool:
        """Check if training is currently paused."""
        return self._pause_flag.is_set()

    def get_metrics(self) -> TrainingMetrics:
        """
        Get current training metrics.

        Returns:
            Current metrics
        """
        # Update elapsed time
        if self.start_time is not None and self._running.is_set():
            self.current_metrics.elapsed_time = time.time() - self.start_time

        return self.current_metrics

    def get_logs(self, max_lines: int = 100) -> list[str]:
        """
        Get recent log messages.

        Args:
            max_lines: Maximum number of log lines to return

        Returns:
            List of log messages
        """
        logs = []
        try:
            while not self.log_queue.empty() and len(logs) < max_lines:
                logs.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        return logs

    def _training_loop(self):
        """Main training loop (runs in background thread)."""
        try:
            self._log("Initializing trainer...", "INFO")

            # Create trainer instance
            self.trainer = Trainer(
                model=self.model,
                config=self.config,
                train_loader=self.train_loader,
                val_loader=self.val_loader
            )

            # Monkey-patch trainer methods to add our callbacks
            self._patch_trainer_methods()

            # Start training
            self._log("Starting training loop...", "INFO")
            self.trainer.train()

            # Training completed normally
            self.current_metrics.status = "completed"
            self.current_metrics.message = "Training completed successfully"
            self._log("Training completed successfully", "INFO")

        except Exception as e:
            self.current_metrics.status = "error"
            self.current_metrics.message = f"Training error: {str(e)}"
            self._log(f"Training error: {e}", "ERROR")
            import traceback
            self._log(traceback.format_exc(), "ERROR")

        finally:
            self._running.clear()

    def _patch_trainer_methods(self):
        """Patch trainer methods to add pause/stop checks and metric updates."""
        if self.trainer is None:
            return

        # Store original train_epoch method
        original_train_epoch = self.trainer.train_epoch

        def patched_train_epoch():
            """Patched train_epoch with pause/stop checks."""
            # Check for pause
            while self._pause_flag.is_set() and not self._stop_flag.is_set():
                time.sleep(0.5)  # Sleep while paused

            # Check for stop
            if self._stop_flag.is_set():
                self._log("Training stopped by user during epoch", "INFO")
                return {'loss': 0.0, 'epoch_time': 0.0}

            # Run original method
            result = original_train_epoch()

            # Update metrics
            self._update_metrics_from_trainer(result)

            return result

        # Replace method
        self.trainer.train_epoch = patched_train_epoch

    def _update_metrics_from_trainer(self, train_metrics: Dict[str, float]):
        """Update metrics from trainer results."""
        if self.trainer is None:
            return

        self.current_metrics.epoch = self.trainer.current_epoch + 1
        self.current_metrics.step = self.trainer.global_step
        self.current_metrics.train_loss = train_metrics.get('loss', 0.0)
        self.current_metrics.perplexity = train_metrics.get('perplexity', 0.0)
        self.current_metrics.learning_rate = train_metrics.get('learning_rate', 0.0)
        self.current_metrics.grad_norm = train_metrics.get('grad_norm', 0.0)
        self.current_metrics.epoch_time = train_metrics.get('epoch_time', 0.0)
        self.current_metrics.best_val_loss = self.trainer.best_val_loss

        # Call progress callback if set
        if self.progress_callback is not None:
            try:
                self.progress_callback(self.current_metrics)
            except Exception as e:
                self._log(f"Error in progress callback: {e}", "WARNING")

    def _log(self, message: str, level: str = "INFO"):
        """Add message to log queue and call log callback."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"

        # Add to queue
        try:
            self.log_queue.put(formatted_message)
        except queue.Full:
            pass  # Queue full, skip this log

        # Call callback if set
        if self.log_callback is not None:
            try:
                self.log_callback(formatted_message)
            except Exception as e:
                print(f"Error in log callback: {e}")


__all__ = ['GradioTrainer', 'TrainingMetrics']
