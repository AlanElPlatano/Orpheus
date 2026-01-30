"""
Chord injection module for custom chord-based melody generation.

This module handles parsing user-provided MIDI files containing chord tracks,
validating them, and preparing them for melody generation.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from miditoolkit import MidiFile

from midi_parser.core.tokenizer_manager import TokenizerManager
from midi_parser.config.defaults import MidiParserConfig
from ml_core.data.constants import (
    TOKEN_RANGES,
    VOCAB_SIZE,
    CORRIDOS_CHORD_PROGRAM,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    BAR_TOKEN_ID,
    KEY_TO_ID,
    TIME_SIG_TO_ID,
    ID_TO_KEY,
    ID_TO_TIME_SIG,
    CONDITION_NONE_ID,
)
from ml_core.data.vocab import VocabularyInfo

logger = logging.getLogger(__name__)


@dataclass
class ChordMetadata:
    """Metadata extracted from user-provided chord MIDI."""
    key: Optional[str] = None
    tempo: Optional[float] = None
    time_signature: Optional[Tuple[int, int]] = None
    token_count: int = 0
    has_program_token: bool = False
    original_program: Optional[int] = None


@dataclass
class ChordValidationResult:
    """Result of chord MIDI validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Optional[ChordMetadata] = None
    tokens: Optional[List[int]] = None


def parse_chord_midi(
    midi_path: Path,
    temp_dir: Path,
    max_tokens: int = 1536
) -> ChordValidationResult:
    """
    Parse a user-provided MIDI file and extract chord tokens.

    Args:
        midi_path: Path to MIDI file
        temp_dir: Temporary directory for processing
        max_tokens: Maximum allowed tokens (default: 1536, leaving room for melody)

    Returns:
        ChordValidationResult with tokens and metadata
    """
    result = ChordValidationResult(
        is_valid=False,
        errors=[],
        warnings=[],
        metadata=ChordMetadata()
    )

    try:
        # Load MIDI file
        logger.info(f"Loading MIDI file: {midi_path}")
        midi = MidiFile(str(midi_path))

        # Validate track count
        num_tracks = len([t for t in midi.instruments if not t.is_drum])
        if num_tracks == 0:
            result.errors.append("No non-drum tracks found in MIDI file")
            return result

        if num_tracks > 1:
            result.errors.append(
                f"Expected exactly 1 track, found {num_tracks}. "
                "Please provide a MIDI file with only chord track."
            )
            return result

        logger.info(f"✓ Single track validation passed")

        # Initialize tokenizer
        parser_config = MidiParserConfig()
        tokenizer_manager = TokenizerManager(parser_config)

        # Tokenize MIDI
        logger.info("Tokenizing MIDI file...")
        tokenization_result = tokenizer_manager.tokenize_midi(
            midi,
            strategy="REMI",
            max_seq_length=max_tokens + 100  # Allow some buffer for validation
        )

        if not tokenization_result.success:
            result.errors.append(
                f"Failed to tokenize MIDI: {tokenization_result.error_message}"
            )
            return result

        tokens = tokenization_result.tokens
        logger.info(f"✓ Tokenization successful: {len(tokens)} tokens")

        # Validate tokens are within vocabulary
        invalid_tokens = [t for t in tokens if t < 0 or t >= VOCAB_SIZE]
        if invalid_tokens:
            result.errors.append(
                f"Found {len(invalid_tokens)} invalid tokens (out of vocab range 0-{VOCAB_SIZE-1})"
            )
            return result

        logger.info(f"✓ All tokens within vocabulary range")

        # Extract metadata from tokens
        metadata = extract_metadata(tokens)
        result.metadata = metadata

        # Check token count
        # Remove special tokens for counting (BOS, EOS, PAD)
        content_tokens = [
            t for t in tokens
            if t not in {BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID}
        ]
        metadata.token_count = len(content_tokens)

        if metadata.token_count > max_tokens:
            result.errors.append(
                f"Chord sequence too long: {metadata.token_count} tokens "
                f"(max: {max_tokens}). This would leave insufficient space for melody generation."
            )
            return result
        elif metadata.token_count > max_tokens * 0.8:
            result.warnings.append(
                f"Chord sequence is quite long ({metadata.token_count}/{max_tokens} tokens). "
                "This may limit melody generation space."
            )

        logger.info(f"✓ Token count validation passed: {metadata.token_count}/{max_tokens}")

        # Force chord program if not present
        tokens_with_program = force_chord_program(tokens, metadata)
        result.tokens = tokens_with_program

        # All validations passed
        result.is_valid = True
        logger.info("✓ All validations passed successfully")

    except FileNotFoundError:
        result.errors.append(f"MIDI file not found: {midi_path}")
    except Exception as e:
        logger.error(f"Error parsing chord MIDI: {e}", exc_info=True)
        result.errors.append(f"Unexpected error: {str(e)}")

    return result


def extract_metadata(tokens: List[int]) -> ChordMetadata:
    """
    Extract metadata (key, tempo, time signature) from token sequence.

    Args:
        tokens: List of token IDs

    Returns:
        ChordMetadata with extracted information
    """
    metadata = ChordMetadata()

    tempo_start, tempo_end = TOKEN_RANGES['tempo']
    timesig_start, timesig_end = TOKEN_RANGES['time_sig']
    program_start, program_end = TOKEN_RANGES['program']

    for token_id in tokens:
        # Extract tempo
        if tempo_start <= token_id <= tempo_end:
            # Tempo tokens are Tempo_40.0 to Tempo_250.0
            # Token 202 = Tempo_40.0, Token 265 = Tempo_250.0
            # Linear mapping: tempo = 40 + (token_id - 202) * (210 / 63)
            tempo_value = 40.0 + (token_id - tempo_start) * (210.0 / (tempo_end - tempo_start))
            metadata.tempo = round(tempo_value, 1)
            logger.debug(f"Extracted tempo: {metadata.tempo} BPM from token {token_id}")

        # Extract time signature
        elif timesig_start <= token_id <= timesig_end:
            # Map token ID back to time signature
            # Need to find which time signature this corresponds to
            # Time sig tokens are 395-403, corresponding to 9 time signatures
            for time_sig_tuple, ts_id in TIME_SIG_TO_ID.items():
                if time_sig_tuple == (0, 0):  # Skip the "none" entry
                    continue
                # TIME_SIG_TO_ID maps to 1-indexed IDs, need to reverse
                # Token 395 = first time sig, token 403 = last time sig
                if token_id == timesig_start + (ts_id - 1):
                    metadata.time_signature = time_sig_tuple
                    logger.debug(f"Extracted time signature: {time_sig_tuple} from token {token_id}")
                    break

        # Extract program
        elif program_start <= token_id <= program_end:
            metadata.has_program_token = True
            if token_id == program_end:
                program_num = -1
            else:
                program_num = token_id - program_start
            metadata.original_program = program_num
            logger.debug(f"Found program token: Program_{program_num} (token {token_id})")

    # Key extraction is not directly available from tokens
    # (key is passed as conditioning tensor, not in token sequence)
    # We'll leave it as None and let the user specify it in the UI

    logger.info(f"Metadata extraction complete: tempo={metadata.tempo}, "
                f"time_sig={metadata.time_signature}, "
                f"has_program={metadata.has_program_token}")

    return metadata


def force_chord_program(tokens: List[int], metadata: ChordMetadata) -> List[int]:
    """
    Ensure the token sequence uses the chord program (Program_29).

    If no program token exists, inject Program_29 at the beginning.
    If a different program token exists, replace it with Program_29.

    Args:
        tokens: Original token sequence
        metadata: Metadata with program information

    Returns:
        Token sequence with forced chord program
    """
    program_start, program_end = TOKEN_RANGES['program']
    chord_program_token = program_start + CORRIDOS_CHORD_PROGRAM  # 266 + 29 = 295

    # If already has the correct program, return as-is
    if metadata.has_program_token and chord_program_token in tokens:
        logger.info("✓ Chord program (Program_29) already present")
        return tokens

    # Find position to inject/replace program token
    # Program tokens typically appear early in the sequence, after BOS
    modified_tokens = []
    program_injected = False

    for i, token_id in enumerate(tokens):
        # If we find any program token, replace it with chord program
        if program_start <= token_id <= program_end:
            if not program_injected:
                modified_tokens.append(chord_program_token)
                program_injected = True
                logger.info(f"✓ Replaced Program_{metadata.original_program} with Program_29")
            # Skip this token (replaced)
        else:
            modified_tokens.append(token_id)

    # If no program token was found, inject after BOS token
    if not program_injected:
        # Find BOS token position (should be first)
        if tokens and tokens[0] == BOS_TOKEN_ID:
            modified_tokens = [BOS_TOKEN_ID, chord_program_token] + tokens[1:]
            logger.info("✓ Injected Program_29 after BOS token")
        else:
            # No BOS, inject at the very beginning
            modified_tokens = [chord_program_token] + tokens
            logger.info("✓ Injected Program_29 at beginning")

    return modified_tokens


def validate_chord_tokens(
    tokens: List[int],
    vocab_info: VocabularyInfo
) -> Tuple[bool, List[str]]:
    """
    Validate that chord tokens are valid for generation.

    Args:
        tokens: Token sequence to validate
        vocab_info: Vocabulary information

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check all tokens are in vocabulary
    for token_id in tokens:
        if token_id >= vocab_info.vocab_size:
            errors.append(f"Token ID {token_id} exceeds vocabulary size {vocab_info.vocab_size}")

    # Check for required elements
    has_bar_token = BAR_TOKEN_ID in tokens
    if not has_bar_token:
        errors.append("No bar tokens found - chord sequence may lack rhythmic structure")

    # Check for chord program
    program_start, program_end = TOKEN_RANGES['program']
    chord_program_token = program_start + CORRIDOS_CHORD_PROGRAM
    has_chord_program = chord_program_token in tokens

    if not has_chord_program:
        errors.append("Chord program (Program_29) not found in token sequence")

    is_valid = len(errors) == 0
    return is_valid, errors


def get_token_count_with_context(tokens: List[int], max_context: int = 2048) -> Dict[str, Any]:
    """
    Get token count information with context limits.

    Args:
        tokens: Token sequence
        max_context: Maximum context length (default: 2048)

    Returns:
        Dictionary with count information
    """
    # Remove special tokens for accurate counting
    content_tokens = [
        t for t in tokens
        if t not in {BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID}
    ]

    # Calculate reserved space for melody (25% of context)
    melody_reservation = max(int(max_context * 0.25), 64)
    max_chord_tokens = max_context - melody_reservation

    chord_count = len(content_tokens)
    remaining_for_melody = max_chord_tokens - chord_count

    return {
        'chord_tokens': chord_count,
        'max_chord_tokens': max_chord_tokens,
        'remaining_for_melody': remaining_for_melody,
        'percentage_used': (chord_count / max_chord_tokens) * 100,
        'is_within_limit': chord_count <= max_chord_tokens
    }


__all__ = [
    'ChordMetadata',
    'ChordValidationResult',
    'parse_chord_midi',
    'extract_metadata',
    'force_chord_program',
    'validate_chord_tokens',
    'get_token_count_with_context',
]
