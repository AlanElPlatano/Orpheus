"""
Training configuration for music generation model.

This module defines all training-related hyperparameters and settings
in a centralized location for easy experimentation and reproducibility.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from ..data.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    WARMUP_STEPS,
    GRADIENT_CLIP,
    WEIGHT_DECAY,
    LOG_INTERVAL,
    VALIDATION_INTERVAL,
    CHECKPOINT_INTERVAL,
    MAX_CHECKPOINTS_TO_KEEP
)


@dataclass
class TrainingConfig:
    """
    Training configuration parameters.

    All hyperparameters for training in one place, making it easy to
    save, load, and modify training settings.
    """

    # ========================================================================
    # Data settings
    # ========================================================================
    data_dir: Path = Path("processed")
    split_manifest_path: Path = Path("pytorch/data/splits/split_manifest.json")
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 0  # 0 = load data in main process (safer on Windows)
    use_cache: bool = False  # Cache dataset in memory

    # ========================================================================
    # Model settings
    # ========================================================================
    vocab_size: int = 531
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_dim: int = 2048
    context_length: int = 2048
    dropout: float = 0.1

    # Track-aware model settings
    use_track_embeddings: bool = True  # Use track type embeddings (melody vs chord)
    num_track_types: int = 2  # Number of track types

    # ========================================================================
    # Optimization settings
    # ========================================================================
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = GRADIENT_CLIP

    # ========================================================================
    # Learning rate schedule
    # ========================================================================
    warmup_steps: int = WARMUP_STEPS
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of max LR

    # ========================================================================
    # Training loop settings
    # ========================================================================
    num_epochs: int = 50
    max_steps: Optional[int] = None  # If set, overrides num_epochs
    gradient_accumulation_steps: int = 1  # Simulate larger batch size

    # ========================================================================
    # Loss settings
    # ========================================================================
    loss_type: str = "weighted"  # "weighted", "constraint_aware", or "track_aware"
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 = no smoothing)
    ignore_index: int = -100  # Padding token index to ignore in loss

    # Track-aware loss settings (only used when loss_type="track_aware")
    melody_violation_weight: float = 10.0  # Weight for melody constraint violations
    chord_violation_weight: float = 5.0  # Weight for chord constraint violations

    # ========================================================================
    # Validation and evaluation
    # ========================================================================
    do_validation: bool = True
    validation_interval: int = VALIDATION_INTERVAL
    validation_batches: Optional[int] = None  # If set, limit validation batches

    # ========================================================================
    # Logging
    # ========================================================================
    log_interval: int = LOG_INTERVAL
    log_dir: Path = Path("pytorch/logs")
    use_tensorboard: bool = True
    use_wandb: bool = False  # Weights & Biases integration
    wandb_project: Optional[str] = "Orpheus"
    wandb_run_name: Optional[str] = None

    # ========================================================================
    # Checkpointing
    # ========================================================================
    checkpoint_dir: Path = Path("pytorch/checkpoints")
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    save_best_only: bool = False  # If True, only save best model
    max_checkpoints_to_keep: int = MAX_CHECKPOINTS_TO_KEEP
    resume_from_checkpoint: Optional[Path] = None

    # ========================================================================
    # Early stopping
    # ========================================================================
    early_stopping: bool = True
    early_stopping_patience: int = 10  # Validation intervals without improvement
    early_stopping_min_delta: float = 0.001  # Minimum improvement to reset patience

    # ========================================================================
    # Hardware settings
    # ========================================================================
    device: str = "cuda"  # "cuda", "cpu", or "auto"
    mixed_precision: bool = True  # Use automatic mixed precision (FP16)
    compile_model: bool = False  # Use torch.compile (PyTorch 2.0+)

    # ========================================================================
    # Reproducibility
    # ========================================================================
    seed: int = 42
    deterministic: bool = False  # More reproducible but slower

    # ========================================================================
    # Debugging
    # ========================================================================
    debug: bool = False  # Enable debug mode (more logging, checks)
    overfit_batch: bool = False  # Train on single batch (for debugging)
    profile: bool = False  # Enable profiling

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Convert string paths to Path objects
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.split_manifest_path, str):
            self.split_manifest_path = Path(self.split_manifest_path)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.resume_from_checkpoint and isinstance(self.resume_from_checkpoint, str):
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)

        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Validate settings
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0 or self.max_steps is not None, \
            "Must specify either num_epochs or max_steps"
        assert self.gradient_accumulation_steps > 0, \
            "Gradient accumulation steps must be positive"

    def get_effective_batch_size(self) -> int:
        """Get the effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def get_total_steps(self, num_train_samples: int) -> int:
        """
        Calculate total training steps.

        Args:
            num_train_samples: Number of training samples

        Returns:
            Total number of training steps
        """
        if self.max_steps is not None:
            return self.max_steps

        steps_per_epoch = num_train_samples // self.get_effective_batch_size()
        return steps_per_epoch * self.num_epochs

    def to_dict(self) -> dict:
        """Convert config to dictionary (for logging/saving)."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# ============================================================================
# Preset Configurations
# ============================================================================

def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_quick_test_config() -> TrainingConfig:
    """
    Get configuration for quick testing/debugging.

    Useful for:
    - Testing training pipeline
    - Debugging issues
    - Quick iterations during development
    """
    return TrainingConfig(
        num_epochs=1,
        max_steps=100,
        batch_size=2,
        validation_interval=50,
        checkpoint_interval=50,
        log_interval=10,
        use_tensorboard=False,
        use_wandb=False,
        save_best_only=True,
        debug=True
    )


def get_overfit_config() -> TrainingConfig:
    """
    Get configuration for overfitting a single batch.

    Useful for:
    - Verifying model can learn
    - Debugging training issues
    - Testing constraint loss weighting
    """
    return TrainingConfig(
        num_epochs=100,
        batch_size=2,
        learning_rate=1e-3,  # Higher LR for faster overfitting
        validation_interval=50,
        checkpoint_interval=1000,
        log_interval=10,
        use_tensorboard=True,
        use_wandb=False,
        overfit_batch=True,
        early_stopping=False,
        debug=True
    )


def get_production_config() -> TrainingConfig:
    """
    Get configuration for production training.

    Optimized settings for final model training with full dataset.
    """
    return TrainingConfig(
        num_epochs=100,
        batch_size=16,  # Larger batch size for stable training
        learning_rate=1e-4,
        warmup_steps=1000,
        gradient_accumulation_steps=1,
        validation_interval=500,
        checkpoint_interval=2000,
        log_interval=100,
        use_tensorboard=True,
        use_wandb=True,
        early_stopping=True,
        early_stopping_patience=10,
        mixed_precision=True,
        max_checkpoints_to_keep=5
    )


def get_track_aware_config() -> TrainingConfig:
    """
    Get configuration for track-aware training.

    Uses track-aware loss and embeddings to learn different patterns
    for melody vs chord tracks. Optimized for 2-track music generation.
    """
    return TrainingConfig(
        # Model settings
        use_track_embeddings=True,
        num_track_types=2,

        # Loss settings
        loss_type="track_aware",
        melody_violation_weight=10.0,  # Heavy penalty for melody polyphony
        chord_violation_weight=5.0,    # Medium penalty for chord rhythm

        # Training settings
        num_epochs=100,
        batch_size=12,  # Slightly smaller to accommodate track embeddings
        learning_rate=1e-4,
        warmup_steps=1000,
        gradient_accumulation_steps=1,

        # Validation and logging
        validation_interval=500,
        checkpoint_interval=2000,
        log_interval=100,

        # Logging
        use_tensorboard=True,
        use_wandb=True,

        # Early stopping
        early_stopping=True,
        early_stopping_patience=10,

        # Performance
        mixed_precision=True,
        max_checkpoints_to_keep=5
    )


def get_low_memory_config() -> TrainingConfig:
    """
    Get configuration optimized for low VRAM systems (4GB or less).

    Key optimizations:
    - Reduced batch size (1)
    - Shorter context length (512 instead of 2048)
    - Smaller model dimensions
    - Gradient accumulation to simulate larger batches
    - Mixed precision enabled

    Useful for:
    - Training on laptops with integrated or low-end GPUs
    - Development on memory-constrained systems
    """
    return TrainingConfig(
        # Model settings - smaller to fit in memory
        hidden_dim=256,           # Reduced from 512
        num_layers=4,             # Reduced from 8
        num_heads=4,              # Reduced from 8
        ff_dim=1024,              # Reduced from 2048
        context_length=512,       # Reduced from 2048 (4x smaller!)
        dropout=0.1,

        # Training settings
        batch_size=1,             # Minimal batch size
        gradient_accumulation_steps=8,  # Simulate batch_size=8
        num_epochs=50,
        learning_rate=1e-4,
        warmup_steps=500,

        # Validation and logging
        validation_interval=200,  # Less frequent validation to save memory
        checkpoint_interval=1000,
        log_interval=50,

        # Memory optimization
        mixed_precision=True,     # Essential for memory savings
        num_workers=0,            # No multiprocessing overhead
        use_cache=False,          # Don't cache dataset in memory

        # Logging
        use_tensorboard=True,
        use_wandb=True,

        # Early stopping
        early_stopping=True,
        early_stopping_patience=10,

        # Checkpointing
        max_checkpoints_to_keep=3  # Save disk space
    )

def get_optimized_default_config() -> TrainingConfig:
    """
    Balanced config that should work well without requiring much VRAM.
    
    Uses gradient accumulation to maintain effective batch size while
    keeping memory footprint low. Maintains default architecture to
    ensure training quality.
    
    Key features:
    - Full-size architecture (512 hidden dim, 8 layers)
    - Small batch size (2) with gradient accumulation (2x) = effective batch of 4
    - Mixed precision for memory efficiency
    - Suitable for GPUs with 6-8GB VRAM
    
    Useful for:
    - Mid-range GPUs (GTX 1060/1070, RTX 2060, etc.)
    - Training with quality comparable to default config
    - Development on memory-constrained systems
    """
    return TrainingConfig(
        # Default architecture - full quality
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        context_length=2048,
        dropout=0.1,
        
        # Training - smaller batch, use accumulation
        batch_size=2,  # Reduced for memory
        gradient_accumulation_steps=2,  # Effective batch = 4
        num_epochs=50,
        learning_rate=1e-4,
        warmup_steps=1000,
        
        # Memory optimization
        mixed_precision=True,  # Essential for memory savings
        num_workers=0,  # Avoid multiprocessing overhead
        use_cache=False,  # Don't cache dataset in memory
        
        # Validation and logging
        validation_interval=500,
        checkpoint_interval=2000,
        log_interval=100,
        
        # Logging
        use_tensorboard=True,
        use_wandb=True,
        
        # Early stopping
        early_stopping=True,
        early_stopping_patience=10,
        
        # Checkpointing
        max_checkpoints_to_keep=5
    )

def get_config_by_name(name: str) -> TrainingConfig:
    """
    Get configuration by preset name.

    Args:
        name: One of "default", "quick_test", "overfit", "production", 
              "track_aware", "optimized_default", "low_memory"

    Returns:
        TrainingConfig instance
    """
    configs = {
        "default": get_default_config,
        "quick_test": get_quick_test_config,
        "overfit": get_overfit_config,
        "production": get_production_config,
        "track_aware": get_track_aware_config,
        "optimized_default": get_optimized_default_config,
        "low_memory": get_low_memory_config
    }

    if name not in configs:
        raise ValueError(
            f"Unknown config name '{name}'. "
            f"Available: {list(configs.keys())}"
        )

    return configs[name]()


__all__ = [
    'TrainingConfig',
    'get_default_config',
    'get_quick_test_config',
    'get_overfit_config',
    'get_production_config',
    'get_track_aware_config',
    'get_optimized_default_config',
    'get_low_memory_config',
    'get_config_by_name'
]
