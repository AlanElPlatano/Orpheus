"""
Checkpoint management utilities for training.

This module provides functions to save and load model checkpoints,
manage checkpoint directories, and handle training state persistence.
"""

import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    best_val_loss: Optional[float] = None,
    config: Optional[Dict] = None,
    model_config: Optional[Dict] = None,
    extra_state: Optional[Dict] = None
):
    """
    Save training checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save (optional)
        epoch: Current epoch
        step: Current step
        best_val_loss: Best validation loss so far
        config: Training configuration (optional)
        model_config: Model architecture configuration (optional)
        extra_state: Any additional state to save (optional, e.g., vocab_info, tokenizer_config)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now().isoformat()
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config

    if model_config is not None:
        checkpoint["model_config"] = model_config

    if extra_state is not None:
        checkpoint["extra_state"] = extra_state

    # Create directory if it doesn't exist
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on (optional)
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary with checkpoint metadata (epoch, step, etc.)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return metadata
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "config": checkpoint.get("config"),
        "extra_state": checkpoint.get("extra_state"),
        "timestamp": checkpoint.get("timestamp")
    }


def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Get path to the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def get_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Get path to the best model checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to best checkpoint, or None if not found
    """
    best_path = checkpoint_dir / "best_model.pt"
    return best_path if best_path.exists() else None


def list_checkpoints(checkpoint_dir: Path) -> list:
    """
    List all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint paths, sorted by step number
    """
    if not checkpoint_dir.exists():
        return []

    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

    # Sort by step number
    def get_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    return sorted(checkpoints, key=get_step)


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    max_to_keep: int = 5,
    keep_best: bool = True
):
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        keep_best: Whether to always keep the best model checkpoint
    """
    checkpoints = list_checkpoints(checkpoint_dir)

    if len(checkpoints) <= max_to_keep:
        return  # Nothing to clean up

    # Keep the most recent max_to_keep checkpoints
    to_delete = checkpoints[:-max_to_keep]

    for checkpoint_path in to_delete:
        try:
            checkpoint_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete checkpoint {checkpoint_path}: {e}")


def save_best_model(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    val_loss: float = float('inf'),
    config: Optional[Dict] = None,
    model_config: Optional[Dict] = None,
    extra_state: Optional[Dict] = None
):
    """
    Save the best model checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint in
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state (optional)
        epoch: Current epoch
        step: Current step
        val_loss: Validation loss (becomes best_val_loss)
        config: Training configuration (optional)
        model_config: Model architecture configuration (optional)
        extra_state: Any additional state to save (optional)
    """
    best_path = checkpoint_dir / "best_model.pt"

    save_checkpoint(
        checkpoint_path=best_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        step=step,
        best_val_loss=val_loss,
        config=config,
        model_config=model_config,
        extra_state=extra_state
    )


def save_training_state(
    checkpoint_dir: Path,
    state: Dict[str, Any],
    filename: str = "training_state.json"
):
    """
    Save training state as JSON.

    Useful for saving metrics history, best scores, etc.

    Args:
        checkpoint_dir: Directory to save state in
        state: Dictionary with training state
        filename: Filename for state file
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / filename

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def load_training_state(
    checkpoint_dir: Path,
    filename: str = "training_state.json"
) -> Optional[Dict[str, Any]]:
    """
    Load training state from JSON.

    Args:
        checkpoint_dir: Directory containing state file
        filename: Filename for state file

    Returns:
        Dictionary with training state, or None if not found
    """
    state_path = checkpoint_dir / filename

    if not state_path.exists():
        return None

    with open(state_path, 'r') as f:
        return json.load(f)


def get_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    info = {
        "path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "timestamp": checkpoint.get("timestamp"),
        "file_size_mb": checkpoint_path.stat().st_size / (1024 ** 2)
    }

    # Count model parameters if state dict is present
    if "model_state_dict" in checkpoint:
        num_params = sum(
            p.numel() for p in checkpoint["model_state_dict"].values()
        )
        info["num_parameters"] = num_params

    return info


def print_checkpoint_info(checkpoint_path: Path):
    """
    Print information about a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    info = get_checkpoint_info(checkpoint_path)

    print("\n" + "=" * 60)
    print("Checkpoint Information")
    print("=" * 60)

    print(f"\nPath: {info['path']}")
    print(f"Epoch: {info['epoch']}")
    print(f"Step: {info['step']}")

    if info['best_val_loss'] is not None:
        print(f"Best validation loss: {info['best_val_loss']:.4f}")

    if info['timestamp']:
        print(f"Timestamp: {info['timestamp']}")

    print(f"File size: {info['file_size_mb']:.2f} MB")

    if 'num_parameters' in info:
        print(f"Model parameters: {info['num_parameters']:,}")

    print("=" * 60)


__all__ = [
    'save_checkpoint',
    'load_checkpoint',
    'get_latest_checkpoint',
    'get_best_checkpoint',
    'list_checkpoints',
    'cleanup_old_checkpoints',
    'save_best_model',
    'save_training_state',
    'load_training_state',
    'get_checkpoint_info',
    'print_checkpoint_info'
]
