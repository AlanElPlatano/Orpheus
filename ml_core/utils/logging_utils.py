"""
Logging utilities for training.

This module provides functions for logging training metrics, setting up
TensorBoard, and managing log files.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class TrainingLogger:
    """
    Logger for training metrics and events.

    Supports:
    - Console logging
    - File logging
    - TensorBoard logging
    - Weights & Biases logging (optional)
    """

    def __init__(
        self,
        log_dir: Path,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        console: bool = True
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Directory for log files
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            console: Whether to log to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.console = console
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.log"

        # Initialize TensorBoard writer
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.log("TensorBoard logging enabled")
            except ImportError:
                self.log("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

        # Initialize Weights & Biases
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    dir=str(self.log_dir)
                )
                self.log("Weights & Biases logging enabled")
            except ImportError:
                self.log("Warning: Weights & Biases not available. Install with: pip install wandb")
                self.use_wandb = False
            except Exception as e:
                self.log(f"Warning: Failed to initialize W&B: {e}")
                self.use_wandb = False

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"

        # Console logging
        if self.console:
            print(formatted_message)

        # File logging
        with open(self.log_file, 'a') as f:
            f.write(formatted_message + '\n')

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = ""
    ):
        """
        Log metrics to all enabled loggers.

        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        # Console/file logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in metrics.items()])
        self.log(f"Step {step} - {prefix}{metrics_str}")

        # TensorBoard logging
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}{key}", value, step)

        # W&B logging
        if self.wandb_run is not None:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            wandb_metrics["step"] = step
            self.wandb_run.log(wandb_metrics)

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            hparams: Dictionary of hyperparameters
        """
        self.log("Hyperparameters:")
        for key, value in hparams.items():
            self.log(f"  {key}: {value}")

        # W&B logging
        if self.wandb_run is not None:
            self.wandb_run.config.update(hparams)

    def close(self):
        """Close all loggers."""
        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.wandb_run is not None:
            self.wandb_run.finish()


class MetricsTracker:
    """
    Track and compute running averages of metrics during training.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float]):
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def get_averages(self) -> Dict[str, float]:
        """
        Get average values for all metrics.

        Returns:
            Dictionary of average metric values
        """
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_number(num: float) -> str:
    """
    Format large numbers with K/M/B suffixes.

    Args:
        num: Number to format

    Returns:
        Formatted number string (e.g., "1.5M", "3.2K")
    """
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    else:
        return f"{num:.0f}"


def print_training_header(config: Any):
    """
    Print training configuration header.

    Args:
        config: Training configuration object
    """
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)

    print(f"\nModel:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Vocab size: {config.vocab_size}")

    print(f"\nTraining:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate:.2e}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Warmup steps: {config.warmup_steps}")

    print(f"\nLogging:")
    print(f"  Log dir: {config.log_dir}")
    print(f"  TensorBoard: {config.use_tensorboard}")
    print(f"  Weights & Biases: {config.use_wandb}")

    print(f"\nCheckpointing:")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Checkpoint interval: {config.checkpoint_interval}")
    print(f"  Max checkpoints: {config.max_checkpoints_to_keep}")

    print("=" * 80 + "\n")


def print_epoch_summary(
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    learning_rate: float,
    epoch_time: float
):
    """
    Print summary of an epoch.

    Args:
        epoch: Epoch number
        train_loss: Training loss
        val_loss: Validation loss (optional)
        learning_rate: Current learning rate
        epoch_time: Time taken for epoch (seconds)
    """
    print("\n" + "-" * 80)
    print(f"Epoch {epoch} Summary")
    print("-" * 80)

    print(f"Training loss: {train_loss:.4f}")
    if val_loss is not None:
        print(f"Validation loss: {val_loss:.4f}")
    print(f"Learning rate: {learning_rate:.2e}")
    print(f"Epoch time: {format_time(epoch_time)}")

    print("-" * 80 + "\n")


def print_progress(
    step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Print training progress for current step.

    Args:
        step: Current step
        total_steps: Total number of steps
        loss: Current loss value
        learning_rate: Current learning rate
        metrics: Additional metrics to display (optional)
    """
    progress = step / total_steps * 100
    progress_bar = "=" * int(progress / 2) + ">" + "." * (50 - int(progress / 2))

    metrics_str = ""
    if metrics:
        metrics_str = " | " + " | ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ])

    print(f"\rStep {step}/{total_steps} [{progress_bar}] {progress:.1f}% | "
          f"Loss: {loss:.4f} | LR: {learning_rate:.2e}{metrics_str}", end="")

    if step == total_steps:
        print()  # New line at end


__all__ = [
    'TrainingLogger',
    'MetricsTracker',
    'format_time',
    'format_number',
    'print_training_header',
    'print_epoch_summary',
    'print_progress'
]
