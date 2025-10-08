"""
Default configuration constants and validation for MIDI parser.

This module provides the core configuration structure, default values,
and validation functions for the MIDI tokenization system.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Core Configuration Classes
# ============================================================================

@dataclass
class TokenizerConfig:
    """Configuration for MidiTok tokenizers."""
    pitch_range: Tuple[int, int] = (21, 109)  # A0 to C8
    beat_resolution: int = 8  # 16th note resolution
    num_velocities: int = 64  # Velocity quantization levels
    max_seq_length: int = 2048
    additional_tokens: Dict[str, bool] = field(default_factory=lambda: {
        'Chord': True,
        'Rest': True,
        'Tempo': True,
        'TimeSignature': True,
        'Program': True,
        'Pedal': False,
        'PitchBend': False,
    })
    single_stream_mode: bool = False
    
    def __post_init__(self):
        """Validate tokenizer configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate tokenizer configuration parameters."""
        if not (0 <= self.pitch_range[0] < self.pitch_range[1] <= 127):
            raise ValueError(f"Invalid pitch_range: {self.pitch_range}. Must be (0-127, 0-127) with first < second")
        
        if self.beat_resolution <= 0 or self.beat_resolution > 32:
            raise ValueError(f"Invalid beat_resolution: {self.beat_resolution}. Must be 1-32")
        
        if self.num_velocities <= 0 or self.num_velocities > 128:
            raise ValueError(f"Invalid num_velocities: {self.num_velocities}. Must be 1-128")
        
        if self.max_seq_length <= 0:
            raise ValueError(f"Invalid max_seq_length: {self.max_seq_length}. Must be > 0")


@dataclass
class TrackClassificationConfig:
    """Configuration for track analysis and classification."""
    chord_threshold: int = 3  # Min simultaneous notes to classify as chord track
    min_notes_per_track: int = 10  # Min notes to consider track valid
    max_empty_ratio: float = 0.8  # Max ratio of empty space allowed
    melody_max_polyphony: int = 2  # Max simultaneous notes for melody classification
    bass_pitch_threshold: int = 48  # C3 - notes below this more likely bass
    drum_channel: int = 9  # Standard MIDI drum channel (0-indexed)
    
    def __post_init__(self):
        """Validate track classification configuration."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate track classification parameters."""
        if self.chord_threshold < 2:
            raise ValueError("chord_threshold must be >= 2")
        
        if not (0.0 <= self.max_empty_ratio <= 1.0):
            raise ValueError("max_empty_ratio must be between 0.0 and 1.0")
        
        if not (0 <= self.bass_pitch_threshold <= 127):
            raise ValueError("bass_pitch_threshold must be between 0 and 127")


@dataclass
class OutputConfig:
    """Configuration for output formatting and serialization."""
    compress_json: bool = True  # Use gzip compression for large files
    include_vocabulary: bool = True  # Include tokenizer vocabulary in output
    pretty_print: bool = False  # Pretty-print JSON (increases file size)
    file_naming_template: str = "{key}-{tempo}bpm-{tokenization}-{title}.json"
    max_filename_length: int = 100  # Max characters in filename
    
    def __post_init__(self):
        """Validate output configuration."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate output configuration parameters."""
        if self.max_filename_length <= 0:
            raise ValueError("max_filename_length must be > 0")
        
        # Validate template has required placeholders
        required_placeholders = ['{key}', '{tempo}', '{tokenization}', '{title}']
        for placeholder in required_placeholders:
            if placeholder not in self.file_naming_template:
                logger.warning(f"Missing placeholder {placeholder} in file_naming_template")


@dataclass
class ProcessingConfig:
    """Configuration for MIDI processing pipeline."""
    max_duration_seconds: float = 600.0  # 10 minutes max per file
    max_file_size_mb: float = 10.0  # 10MB max file size
    chunk_size_seconds: float = 30.0  # Chunk size for long files
    parallel_processing: bool = True  # Enable parallel processing
    max_workers: Optional[int] = None  # None = auto-detect CPU count
    
    def __post_init__(self):
        """Validate processing configuration."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate processing configuration parameters."""
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be > 0")
        
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be > 0")
        
        if self.chunk_size_seconds <= 0:
            raise ValueError("chunk_size_seconds must be > 0")


@dataclass
class ValidationConfig:
    """Configuration for round-trip validation and quality checks."""
    tolerances: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        'note_start_tick': 2,           # Max 2 tick difference
        'note_duration': 4,             # Max 4 ticks difference  
        'velocity_bin': 2,              # Max 2 velocity bin difference
        'missing_notes_ratio': 0.01,    # Max 1% notes missing
        'extra_notes_ratio': 0.01,      # Max 1% extra notes
        'tempo_bpm_diff': 3.0,          # Max 3 BPM difference
    })
    enable_round_trip_test: bool = True  # Enable round-trip validation
    quality_threshold: float = 0.95  # Min quality score to pass validation
    
    def __post_init__(self):
        """Validate validation configuration."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate validation configuration parameters."""
        if not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0.0 and 1.0")
        
        for key, value in self.tolerances.items():
            if value < 0:
                raise ValueError(f"Tolerance {key} must be >= 0, got {value}")


@dataclass
class MidiParserConfig:
    """Main configuration class that combines all sub-configurations."""

    tokenization: str = "REMI"  # Default tokenization strategy
    
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    track_classification: TrackClassificationConfig = field(default_factory=TrackClassificationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    def __post_init__(self):
        """Validate main configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate main configuration parameters."""
        available_strategies = get_available_strategies()
        if self.tokenization not in available_strategies:
            raise ValueError(f"Invalid tokenization strategy: {self.tokenization}. "
                           f"Must be one of: {available_strategies}")


# ============================================================================
# YAML Configuration Loading
# ============================================================================

import yaml
from pathlib import Path
from functools import lru_cache

# Path to strategies YAML file
_STRATEGIES_YAML_PATH = Path(__file__).parent / "strategies.yaml"

@lru_cache(maxsize=1)
def _load_strategies_yaml() -> Dict[str, Any]:
    """
    Load and cache the strategies YAML file.
    
    Returns:
        Dictionary containing all YAML configuration data
        
    Raises:
        FileNotFoundError: If strategies.yaml is not found
        yaml.YAMLError: If YAML file is malformed
    """
    try:
        with open(_STRATEGIES_YAML_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Strategies YAML file not found at: {_STRATEGIES_YAML_PATH}")
        raise FileNotFoundError(f"Required configuration file not found: {_STRATEGIES_YAML_PATH}")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse strategies YAML file: {e}")
        raise yaml.YAMLError(f"Malformed YAML configuration: {e}")

def get_available_strategies() -> List[str]:
    """
    Get list of available tokenization strategies from YAML config.
    
    Returns:
        List of strategy names
    """
    yaml_data = _load_strategies_yaml()
    return list(yaml_data.get('tokenization_strategies', {}).keys())

def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a tokenization strategy.
    
    Args:
        strategy_name: Name of the tokenization strategy
        
    Returns:
        Dictionary containing strategy information
        
    Raises:
        ValueError: If strategy name is invalid
    """
    yaml_data = _load_strategies_yaml()
    strategies = yaml_data.get('tokenization_strategies', {})
    
    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(f"Invalid strategy '{strategy_name}'. Available: {available}")
    
    return strategies[strategy_name]

def get_available_genres() -> List[str]:
    """
    Get list of available genre presets from YAML config.
    
    Returns:
        List of genre names (empty if no genre presets defined)
    """
    yaml_data = _load_strategies_yaml()
    genre_presets = yaml_data.get('genre_presets', {})
    return list(genre_presets.keys()) if genre_presets else []

def get_available_use_cases() -> List[str]:
    """
    Get list of available use case presets from YAML config.
    
    Returns:
        List of use case names
    """
    yaml_data = _load_strategies_yaml()
    return list(yaml_data.get('use_case_presets', {}).keys())

def get_available_instruments() -> List[str]:
    """
    Get list of available instrument presets from YAML config.
    
    Returns:
        List of instrument preset names (empty if no instrument presets defined)
    """
    yaml_data = _load_strategies_yaml()
    instrument_presets = yaml_data.get('instrument_presets', {})
    return list(instrument_presets.keys()) if instrument_presets else []

def get_available_complexity_levels() -> List[str]:
    """
    Get list of available complexity level presets from YAML config.
    
    Returns:
        List of complexity level names (empty if no complexity presets defined)
    """
    yaml_data = _load_strategies_yaml()
    complexity_presets = yaml_data.get('complexity_presets', {})
    return list(complexity_presets.keys()) if complexity_presets else []

def get_available_templates() -> List[str]:
    """
    Get list of available configuration templates from YAML config.
    
    Returns:
        List of template names (empty if no templates defined)
    """
    yaml_data = _load_strategies_yaml()
    templates = yaml_data.get('templates', {})
    return list(templates.keys()) if templates else []

def get_preset_config(preset_type: str, preset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific preset.
    
    Args:
        preset_type: Type of preset ('use_case_presets', 'genre_presets', etc.)
        preset_name: Name of the specific preset
        
    Returns:
        Dictionary containing preset configuration
        
    Raises:
        ValueError: If preset type doesn't exist or preset name is invalid
    """
    yaml_data = _load_strategies_yaml()
    
    # Check if preset type exists
    if preset_type not in yaml_data:
        available_types = [k for k in yaml_data.keys() if k.endswith('_presets') or k == 'templates']
        raise ValueError(
            f"Preset type '{preset_type}' not found. "
            f"Available types: {available_types}. "
            f"This preset category may have been removed from the simplified configuration."
        )
    
    presets = yaml_data[preset_type]
    
    # Handle empty preset categories
    if not presets:
        raise ValueError(
            f"No presets defined for '{preset_type}'. "
            f"This preset category exists but contains no entries."
        )
    
    if preset_name not in presets:
        available_presets = list(presets.keys())
        raise ValueError(
            f"Preset '{preset_name}' not found in {preset_type}. "
            f"Available presets: {available_presets}"
        )
    
    return presets[preset_name]

# ============================================================================
# Error Handling Strategies
# ============================================================================

ERROR_HANDLING_STRATEGIES = {
    "corrupted_midi": "skip",           # Skip unreadable files
    "empty_tracks": "remove",           # Remove tracks with no notes
    "extreme_duration": "truncate",     # Handle excessively long MIDIs
    "unsupported_events": "filter",     # Remove non-standard MIDI events
    "memory_overflow": "chunk",         # Process in chunks for large files
    "tokenization_failure": "fallback", # Try alternative tokenization
    "invalid_metadata": "default",     # Use default values for missing metadata
}

# ============================================================================
# Validation Functions
# ============================================================================

def validate_config(config: MidiParserConfig) -> List[str]:
    """
    Validate a complete configuration object.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    try:
        # Validation is handled in __post_init__ methods
        # This will raise exceptions if invalid
        MidiParserConfig(
            tokenization=config.tokenization,
            tokenizer=config.tokenizer,
            track_classification=config.track_classification,
            output=config.output,
            processing=config.processing,
            validation=config.validation
        )
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Unexpected validation error: {e}")
    
    return errors


def get_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """
    Get configuration overrides for a specific tokenization strategy.
    
    Args:
        strategy_name: Name of the tokenization strategy
        
    Returns:
        Dictionary of configuration overrides
        
    Raises:
        ValueError: If strategy name is invalid
    """
    strategy_info = get_strategy_info(strategy_name)
    return strategy_info.get("config_overrides", {})


def get_genre_preset(genre: str) -> Dict[str, Any]:
    """
    Get configuration preset for a specific genre.
    
    Args:
        genre: Name of the genre preset
        
    Returns:
        Dictionary of configuration settings for the genre
        
    Raises:
        ValueError: If genre presets don't exist or genre is not found
    """
    yaml_data = _load_strategies_yaml()
    
    # Check if genre_presets exists in YAML
    if 'genre_presets' not in yaml_data or not yaml_data['genre_presets']:
        raise ValueError(
            "Genre presets are not defined in the current configuration. "
            "Consider using 'corrido_demo' or 'general_purpose' from use_case_presets instead."
        )
    
    return get_preset_config("genre_presets", genre)


def apply_config_overrides(base_config: MidiParserConfig, overrides: Dict[str, Any]) -> MidiParserConfig:
    """
    Apply configuration overrides to a base configuration.
    
    Args:
        base_config: Base configuration object
        overrides: Dictionary of configuration overrides
        
    Returns:
        New configuration object with overrides applied
    """
    import copy
    
    # Deep copy to avoid modifying original
    new_config = copy.deepcopy(base_config)
    
    # Apply top-level overrides
    for key, value in overrides.items():
        if hasattr(new_config, key):
            if isinstance(getattr(new_config, key), dict):
                # Merge dictionaries
                current_dict = getattr(new_config, key)
                if isinstance(value, dict):
                    current_dict.update(value)
                else:
                    setattr(new_config, key, value)
            else:
                setattr(new_config, key, value)
    
    # Re-validate after applying overrides
    new_config._validate()
    
    return new_config


# ============================================================================
# Default Configuration Instance
# ============================================================================

DEFAULT_CONFIG = MidiParserConfig()

# Export commonly used configurations
__all__ = [
    'MidiParserConfig',
    'TokenizerConfig', 
    'TrackClassificationConfig',
    'OutputConfig',
    'ProcessingConfig',
    'ValidationConfig',
    'ERROR_HANDLING_STRATEGIES',
    'DEFAULT_CONFIG',
    'validate_config',
    'get_strategy_config',
    'get_genre_preset',
    'apply_config_overrides',
    # YAML loading functions
    'get_available_strategies',
    'get_available_genres',
    'get_available_use_cases',
    'get_available_instruments', 
    'get_available_complexity_levels',
    'get_available_templates',
    'get_strategy_info',
    'get_preset_config',
]