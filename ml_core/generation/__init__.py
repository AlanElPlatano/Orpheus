"""
Music generation infrastructure for trained transformer models.

This module provides complete generation pipeline:
- Configuration management
- Advanced sampling strategies
- Constrained decoding for musical constraints
- Two-stage generation (chords → melody)
- Token-to-MIDI conversion
- Post-generation validation
"""

from .generation_config import (
    GenerationConfig,
    GenerationResult,
    create_quality_config,
    create_creative_config,
    create_custom_config
)

from .sampling import (
    sample_with_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
    sample_next_token
)

from .constrained_decode import (
    update_generation_state,
    apply_all_constraints,
    is_sequence_complete
)

from .two_stage import (
    TwoStageGenerator
)

from .midi_export import (
    tokens_to_midi,
    save_token_sequence
)

from .validator import (
    ConstraintValidator,
    ValidationReport,
    validate_generated_sequence
)

from .generator import (
    MusicGenerator,
    load_generator_from_checkpoint
)

__all__ = [
    # Configuration
    'GenerationConfig',
    'GenerationResult',
    'create_quality_config',
    'create_creative_config',
    'create_custom_config',

    # Sampling
    'sample_with_temperature',
    'apply_top_k',
    'apply_top_p',
    'apply_repetition_penalty',
    'sample_next_token',

    # Constrained decoding
    'update_generation_state',
    'apply_all_constraints',
    'is_sequence_complete',

    # Two-stage generation
    'TwoStageGenerator',

    # MIDI export
    'tokens_to_midi',
    'save_token_sequence',

    # Validation
    'ConstraintValidator',
    'ValidationReport',
    'validate_generated_sequence',

    # Main generator
    'MusicGenerator',
    'load_generator_from_checkpoint'
]
