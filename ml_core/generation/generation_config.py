"""
Configuration for music generation.

Defines generation parameters, presets, and result structures.
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

from ..data.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    QUALITY_MODE_PARAMS,
    CREATIVE_MODE_PARAMS,
    MAX_GENERATION_RETRIES,
    CONTEXT_LENGTH
)


@dataclass
class GenerationConfig:
    """
    Configuration for music generation.

    Defines all parameters for the generation process including sampling,
    conditioning, constraints, and output settings.
    """

    # ========== Model Loading ==========
    checkpoint_path: Optional[Path] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Sampling Parameters ==========
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: Optional[int] = None
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

    # ========== Conditioning Parameters (requires model trained with use_conditioning=True) ==========
    key: Optional[str] = None  # e.g., "Fm", "C", "Am"
    tempo: Optional[float] = None  # e.g., 125.0 BPM
    time_signature: Optional[Tuple[int, int]] = None  # e.g., (6, 8) or (4, 4)

    # ========== Generation Constraints ==========
    max_length: int = CONTEXT_LENGTH  # Maximum token sequence length
    max_generation_bars: int = 32  # Maximum bars to generate
    min_generation_bars: int = 8   # Minimum bars for valid output

    # ========== Retry Logic ==========
    max_retries: int = MAX_GENERATION_RETRIES
    retry_temperature_decay: float = 0.95  # Reduce temperature on retry

    # ========== Output Settings ==========
    output_dir: Path = field(default_factory=lambda: Path("./generated"))
    save_intermediate_tokens: bool = True  # Save token sequence for debugging
    filename_template: str = "generated_{timestamp}_{index}"

    # ========== Advanced Settings ==========
    enforce_constraints: bool = True  # Apply musical constraints during generation
    validate_output: bool = True      # Validate constraints after generation
    seed: Optional[int] = None        # Random seed for reproducibility

    def __post_init__(self):
        """Validate and convert paths."""
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        self.output_dir = Path(self.output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "device": self.device,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "key": self.key,
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "max_length": self.max_length,
            "max_generation_bars": self.max_generation_bars,
            "min_generation_bars": self.min_generation_bars,
            "max_retries": self.max_retries,
            "retry_temperature_decay": self.retry_temperature_decay,
            "output_dir": str(self.output_dir),
            "save_intermediate_tokens": self.save_intermediate_tokens,
            "enforce_constraints": self.enforce_constraints,
            "validate_output": self.validate_output,
            "seed": self.seed
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GenerationConfig':
        """Create config from dictionary."""
        # Convert string paths back to Path objects
        if "checkpoint_path" in config_dict and config_dict["checkpoint_path"]:
            config_dict["checkpoint_path"] = Path(config_dict["checkpoint_path"])
        if "output_dir" in config_dict:
            config_dict["output_dir"] = Path(config_dict["output_dir"])

        return cls(**config_dict)


@dataclass
class GenerationResult:
    """
    Result of a single generation attempt.

    Contains all information about the generated file including tokens,
    MIDI path, validation results, and timing information.
    """

    # ========== Generation Status ==========
    success: bool = False
    error_message: Optional[str] = None
    num_attempts: int = 1

    # ========== Generated Content ==========
    token_ids: Optional[List[int]] = None
    token_sequence_path: Optional[Path] = None  # Path to saved tokens
    midi_path: Optional[Path] = None

    # ========== Validation Results ==========
    is_valid: bool = False
    constraint_violations: List[str] = field(default_factory=list)
    num_violations: int = 0

    # ========== Metadata ==========
    generation_time: float = 0.0  # Seconds
    sequence_length: int = 0
    num_bars: int = 0

    # ========== Configuration Used ==========
    config: Optional[GenerationConfig] = None
    temperature_used: Optional[float] = None

    # ========== Timestamp ==========
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "num_attempts": self.num_attempts,
            "token_sequence_path": str(self.token_sequence_path) if self.token_sequence_path else None,
            "midi_path": str(self.midi_path) if self.midi_path else None,
            "is_valid": self.is_valid,
            "constraint_violations": self.constraint_violations,
            "num_violations": self.num_violations,
            "generation_time": self.generation_time,
            "sequence_length": self.sequence_length,
            "num_bars": self.num_bars,
            "temperature_used": self.temperature_used,
            "timestamp": self.timestamp
        }

    def get_summary(self) -> str:
        """Get human-readable summary of generation result."""
        if self.success:
            return (
                f"✓ Success | {self.sequence_length} tokens | "
                f"{self.num_bars} bars | {self.generation_time:.1f}s | "
                f"{self.num_violations} violations"
            )
        else:
            return f"✗ Failed | {self.error_message} | {self.num_attempts} attempts"


# ============================================================================
# Preset Factory Functions
# ============================================================================

def create_quality_config(**overrides) -> GenerationConfig:
    """
    Create configuration for quality-focused generation.

    This mode stays closer to training patterns and produces
    highly usable, conservative outputs.

    Args:
        **overrides: Override any default parameters

    Returns:
        GenerationConfig with quality preset parameters
    """
    config = GenerationConfig(
        temperature=QUALITY_MODE_PARAMS['temperature'],
        top_p=QUALITY_MODE_PARAMS['top_p'],
        repetition_penalty=QUALITY_MODE_PARAMS['repetition_penalty']
    )

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def create_creative_config(**overrides) -> GenerationConfig:
    """
    Create configuration for creative/experimental generation.

    This mode is more exploratory and produces higher variety outputs,
    with some potentially experimental results.

    Args:
        **overrides: Override any default parameters

    Returns:
        GenerationConfig with creative preset parameters
    """
    config = GenerationConfig(
        temperature=CREATIVE_MODE_PARAMS['temperature'],
        top_p=CREATIVE_MODE_PARAMS['top_p'],
        repetition_penalty=CREATIVE_MODE_PARAMS['repetition_penalty']
    )

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def create_custom_config(
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.1,
    **overrides
) -> GenerationConfig:
    """
    Create configuration with custom sampling parameters.

    Args:
        temperature: Sampling temperature (0.1-2.0)
        top_p: Nucleus sampling threshold (0.0-1.0)
        repetition_penalty: Penalty for repeated tokens (1.0-2.0)
        **overrides: Override any other parameters

    Returns:
        GenerationConfig with custom parameters
    """
    config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


__all__ = [
    'GenerationConfig',
    'GenerationResult',
    'create_quality_config',
    'create_creative_config',
    'create_custom_config'
]
