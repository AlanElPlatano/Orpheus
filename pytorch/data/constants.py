"""
Constants and special token definitions for the REMI tokenization vocabulary.

This module defines all special tokens, token categories, and other constants
used throughout the PyTorch training pipeline.

Basically defines all the "magic numbers" used throughout the entire project
instead of having them scattered randomly throughout 50 files
"""

from enum import IntEnum
from typing import Dict, List


# ============================================================================
# Special Token IDs
# ============================================================================

class SpecialTokens(IntEnum):
    """
    Special token IDs from the REMI vocabulary.
    These are fixed across all files.
    """
    PAD = 0      # Padding token for batch processing
    BOS = 1      # Beginning of sequence
    EOS = 2      # End of sequence
    MASK = 3     # Mask token for masked language modeling
    BAR = 4      # Bar/measure separator


# Convenience constants
PAD_TOKEN_ID = SpecialTokens.PAD
BOS_TOKEN_ID = SpecialTokens.BOS
EOS_TOKEN_ID = SpecialTokens.EOS
MASK_TOKEN_ID = SpecialTokens.MASK
BAR_TOKEN_ID = SpecialTokens.BAR


# ============================================================================
# Token Range Constants
# ============================================================================

# Based on the vocabulary structure from the JSON
TOKEN_RANGES = {
    'pitch': (5, 53),           # Pitch_36 to Pitch_84
    'velocity': (54, 61),       # Velocity_15 to Velocity_127
    'duration': (62, 77),       # Duration_0.1.4 to Duration_4.0.4
    'position': (78, 125),      # Position_0 to Position_47
    'pitch_drum': (126, 187),   # PitchDrum_27 to PitchDrum_88
    'chord': (188, 201),        # Chord tokens
    'tempo': (202, 265),        # Tempo_40.0 to Tempo_250.0
    'program': (266, 394),      # Program_0 to Program_127 and Program_-1
    'time_sig': (395, 403),     # TimeSig_3/8 to TimeSig_4/4
}


# ============================================================================
# Musical Constants
# ============================================================================

# MIDI pitch range (based on vocabulary)
MIN_PITCH = 36  # C2
MAX_PITCH = 84  # C6

# Velocity bins (8 bins based on vocabulary)
VELOCITY_BINS = [15, 31, 47, 63, 79, 95, 111, 127]
NUM_VELOCITY_BINS = len(VELOCITY_BINS)

# Beat resolution (from tokenizer config)
BEAT_RESOLUTION = 4

# Position range (number of position tokens)
NUM_POSITIONS = 48  # Position_0 to Position_47

# Duration range
MIN_DURATION_TICKS = 0.1 * BEAT_RESOLUTION  # 0.1 beats
MAX_DURATION_TICKS = 4.0 * BEAT_RESOLUTION  # 4.0 beats


# ============================================================================
# Model Architecture Constants
# ============================================================================

# From the actual vocabulary extracted from tokenizer
VOCAB_SIZE = 580  # Total vocabulary size (580 tokens in REMI vocab)
CONTEXT_LENGTH = 2048  # Maximum sequence length
HIDDEN_DIM = 512  # Model dimension
NUM_LAYERS = 8  # Transformer layers
NUM_HEADS = 8  # Attention heads
FF_DIM = 2048  # Feedforward dimension (4x hidden_dim)
DROPOUT = 0.1  # Dropout rate


# ============================================================================
# Training Constants
# ============================================================================

# From the design document
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 0.01


# ============================================================================
# Generation Constants
# ============================================================================

# Sampling parameters
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.1

# Quality vs Creative modes
QUALITY_MODE_PARAMS = {
    'temperature': 0.8,
    'top_p': 0.95,
    'repetition_penalty': 1.1
}

CREATIVE_MODE_PARAMS = {
    'temperature': 1.1,
    'top_p': 0.92,
    'repetition_penalty': 1.05
}

# Retry logic
MAX_GENERATION_RETRIES = 2


# ============================================================================
# Constraint Weights (for weighted loss)
# ============================================================================

# From the design document
LOSS_WEIGHTS = {
    'monophony_violation': 10.0,      # Heavy penalty for melody polyphony
    'chord_duration_violation': 5.0,   # Medium penalty for chord rhythm
    'diatonic_violation': 3.0,         # Encourage key adherence
    'normal_token': 1.0,               # Standard weight
}


# ============================================================================
# Data Augmentation Constants
# ============================================================================

# Transposition
NUM_TRANSPOSITION_KEYS = 12  # Transpose to all 12 chromatic keys
SEMITONES_RANGE = range(-11, 1)  # -11 to 0 semitones

# Data split ratios (by original song, not by transposition)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


# ============================================================================
# Musical Key Definitions
# ============================================================================

# Major keys
MAJOR_KEYS = [
    'C', 'Db', 'D', 'Eb', 'E', 'F',
    'F#', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'
]

# Minor keys
MINOR_KEYS = [
    'Am', 'Bbm', 'Bm', 'Cm', 'C#m', 'Dm',
    'Ebm', 'Em', 'Fm', 'F#m', 'Gm', 'G#m'
]

# Diatonic scale degrees (for constraint enforcement)
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]  # Semitones from root
MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 10]  # Natural minor


# ============================================================================
# Time Signature Definitions
# ============================================================================

# From vocabulary
SUPPORTED_TIME_SIGNATURES = [
    (1, 4), (2, 4), (3, 4), (4, 4),  # Simple time
    (3, 8), (6, 8), (12, 8),          # Compound time
    (5, 4), (6, 4)                     # Complex time
]

# Most common for corridos
DEFAULT_TIME_SIGNATURE = (6, 8)


# ============================================================================
# Tempo Definitions
# ============================================================================

# From vocabulary
MIN_TEMPO = 40.0
MAX_TEMPO = 250.0
TEMPO_BINS = 64  # Number of tempo tokens

# Typical range for corridos
CORRIDOS_MIN_TEMPO = 90
CORRIDOS_MAX_TEMPO = 140
CORRIDOS_DEFAULT_TEMPO = 125


# ============================================================================
# Track Type Definitions
# ============================================================================

class TrackType(IntEnum):
    """
    Track type identifiers for multi-track music generation.

    These are used to distinguish between different types of tracks
    (melody vs chords) in the model.
    """
    MELODY = 0  # Monophonic melody track
    CHORD = 1   # Chord/harmony track (no rhythmic variation)


# Convenience constants
TRACK_TYPE_MELODY = TrackType.MELODY
TRACK_TYPE_CHORD = TrackType.CHORD
NUM_TRACK_TYPES = 2  # Number of distinct track types


# ============================================================================
# Program (Instrument) Definitions
# ============================================================================

# Common programs used in corridos
CORRIDOS_MELODY_PROGRAM = 98   # Lead 2 (sawtooth) - used for "Voz"
CORRIDOS_CHORD_PROGRAM = 29    # Overdriven Guitar - used for "Armonía"

# MIDI program ranges
PIANO_RANGE = range(0, 8)
CHROMATIC_PERCUSSION_RANGE = range(8, 16)
ORGAN_RANGE = range(16, 24)
GUITAR_RANGE = range(24, 32)
BASS_RANGE = range(32, 40)
STRINGS_RANGE = range(40, 48)
ENSEMBLE_RANGE = range(48, 56)
BRASS_RANGE = range(56, 64)
REED_RANGE = range(64, 72)
PIPE_RANGE = range(72, 80)
SYNTH_LEAD_RANGE = range(80, 88)
SYNTH_PAD_RANGE = range(88, 96)
SYNTH_EFFECTS_RANGE = range(96, 104)
ETHNIC_RANGE = range(104, 112)
PERCUSSIVE_RANGE = range(112, 120)
SOUND_EFFECTS_RANGE = range(120, 128)


# ============================================================================
# File Naming Constants
# ============================================================================

# From the design document
GENERATED_FILE_PREFIX = "generated"
GENERATED_FILE_TEMPLATE = "{prefix}_{key}_{tempo}bpm_{timestamp}"

# Filename parsing regex patterns
KEY_PATTERN = r'([A-G][#b]?m?)'
TEMPO_PATTERN = r'(\d+)bpm'
TRANSPOSE_PATTERN = r'transpose([+-]\d+)'


# ============================================================================
# Validation Constants
# ============================================================================

# Constraints for validation
MAX_MELODY_POLYPHONY = 1  # Strict monophony
MIN_CHORD_POLYPHONY = 2   # At least 2 notes for chords
MAX_CHORD_POLYPHONY = 6   # Upper limit for chords

# Sequence validation
MIN_SEQUENCE_LENGTH = 50   # Minimum tokens for valid sequence
MAX_SEQUENCE_LENGTH = 2048 # Maximum tokens (context length)

# Quality thresholds
MIN_QUALITY_SCORE = 0.7
MIN_CONSTRAINT_ADHERENCE = 0.9  # 90% of constraints should be met


# ============================================================================
# Logging and Checkpointing
# ============================================================================

# How often to log during training
LOG_INTERVAL = 100  # Log every N steps
VALIDATION_INTERVAL = 500  # Validate every N steps
CHECKPOINT_INTERVAL = 2000  # Save checkpoint every N steps

# How many checkpoints to keep
MAX_CHECKPOINTS_TO_KEEP = 5

# Checkpoint naming
CHECKPOINT_PREFIX = "checkpoint"
BEST_MODEL_NAME = "best_model.pt"


# ============================================================================
# Utility Functions
# ============================================================================

def get_token_type(token_id: int) -> str:
    """
    Get the type of a token based on its ID.

    Args:
        token_id: Token ID

    Returns:
        Token type string
    """
    if token_id == PAD_TOKEN_ID:
        return 'PAD'
    elif token_id == BOS_TOKEN_ID:
        return 'BOS'
    elif token_id == EOS_TOKEN_ID:
        return 'EOS'
    elif token_id == MASK_TOKEN_ID:
        return 'MASK'
    elif token_id == BAR_TOKEN_ID:
        return 'BAR'

    for token_type, (start, end) in TOKEN_RANGES.items():
        if start <= token_id <= end:
            return token_type

    return 'UNKNOWN'


def is_special_token(token_id: int) -> bool:
    """Check if token is a special token."""
    return token_id in {PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, MASK_TOKEN_ID, BAR_TOKEN_ID}


def is_pitch_token(token_id: int) -> bool:
    """Check if token is a pitch token."""
    start, end = TOKEN_RANGES['pitch']
    return start <= token_id <= end


def is_drum_token(token_id: int) -> bool:
    """Check if token is a drum pitch token."""
    start, end = TOKEN_RANGES['pitch_drum']
    return start <= token_id <= end


def is_duration_token(token_id: int) -> bool:
    """Check if token is a duration token."""
    start, end = TOKEN_RANGES['duration']
    return start <= token_id <= end


def is_position_token(token_id: int) -> bool:
    """Check if token is a position token."""
    start, end = TOKEN_RANGES['position']
    return start <= token_id <= end


def is_metadata_token(token_id: int) -> bool:
    """Check if token is a metadata token (tempo, time sig, program)."""
    tempo_start, tempo_end = TOKEN_RANGES['tempo']
    timesig_start, timesig_end = TOKEN_RANGES['time_sig']
    program_start, program_end = TOKEN_RANGES['program']

    return (
        (tempo_start <= token_id <= tempo_end) or
        (timesig_start <= token_id <= timesig_end) or
        (program_start <= token_id <= program_end)
    )


def get_track_type_from_program(program_id: int) -> int:
    """
    Determine track type (MELODY or CHORD) based on program/instrument ID.

    Args:
        program_id: MIDI program number (0-127)

    Returns:
        TrackType.MELODY or TrackType.CHORD
    """
    # For corridos: melody is program 98, chords is program 29
    if program_id == CORRIDOS_MELODY_PROGRAM:
        return TRACK_TYPE_MELODY
    elif program_id == CORRIDOS_CHORD_PROGRAM:
        return TRACK_TYPE_CHORD
    else:
        # Default heuristic: higher program numbers tend to be leads/melody
        # Lower numbers tend to be harmonic instruments
        return TRACK_TYPE_MELODY if program_id >= 80 else TRACK_TYPE_CHORD


__all__ = [
    # Special tokens
    'SpecialTokens',
    'PAD_TOKEN_ID',
    'BOS_TOKEN_ID',
    'EOS_TOKEN_ID',
    'MASK_TOKEN_ID',
    'BAR_TOKEN_ID',

    # Token ranges
    'TOKEN_RANGES',

    # Track types
    'TrackType',
    'TRACK_TYPE_MELODY',
    'TRACK_TYPE_CHORD',
    'NUM_TRACK_TYPES',

    # Musical constants
    'MIN_PITCH',
    'MAX_PITCH',
    'VELOCITY_BINS',
    'NUM_VELOCITY_BINS',
    'BEAT_RESOLUTION',
    'NUM_POSITIONS',

    # Model architecture
    'VOCAB_SIZE',
    'CONTEXT_LENGTH',
    'HIDDEN_DIM',
    'NUM_LAYERS',
    'NUM_HEADS',
    'FF_DIM',
    'DROPOUT',

    # Training
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_LEARNING_RATE',
    'WARMUP_STEPS',
    'GRADIENT_CLIP',
    'WEIGHT_DECAY',

    # Generation
    'DEFAULT_TEMPERATURE',
    'DEFAULT_TOP_P',
    'QUALITY_MODE_PARAMS',
    'CREATIVE_MODE_PARAMS',
    'MAX_GENERATION_RETRIES',

    # Constraints
    'LOSS_WEIGHTS',

    # Data splits
    'TRAIN_SPLIT',
    'VAL_SPLIT',
    'TEST_SPLIT',

    # Musical definitions
    'MAJOR_KEYS',
    'MINOR_KEYS',
    'MAJOR_SCALE_INTERVALS',
    'MINOR_SCALE_INTERVALS',
    'SUPPORTED_TIME_SIGNATURES',
    'DEFAULT_TIME_SIGNATURE',
    'CORRIDOS_MIN_TEMPO',
    'CORRIDOS_MAX_TEMPO',

    # Corridos programs
    'CORRIDOS_MELODY_PROGRAM',
    'CORRIDOS_CHORD_PROGRAM',

    # Utility functions
    'get_token_type',
    'is_special_token',
    'is_pitch_token',
    'is_duration_token',
    'is_position_token',
    'is_metadata_token',
    'get_track_type_from_program',
]
