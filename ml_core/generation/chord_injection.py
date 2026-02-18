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
    VOCAB_SIZE,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    BAR_TOKEN_ID,
    CHORD_START_TOKEN_NAME,
    MELODY_START_TOKEN_NAME,
)
from ml_core.data.vocab import VocabularyInfo
from midi_parser.core.token_reorderer import reorder_bar_tokens

logger = logging.getLogger(__name__)


@dataclass
class ChordMetadata:
    """Metadata extracted from user-provided chord MIDI."""
    key: Optional[str] = None
    tempo: Optional[float] = None
    time_signature: Optional[Tuple[int, int]] = None
    token_count: int = 0
    has_chord_start: bool = False


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
        vocabulary = tokenization_result.vocabulary
        logger.info(f"✓ Tokenization successful: {len(tokens)} tokens")

        # Reorder tokens into bar-level chord/melody structure
        # This also extends vocabulary with CHORD_START/MELODY_START
        tokens, vocabulary = reorder_bar_tokens(tokens, vocabulary)
        logger.info(f"✓ Reordered tokens: {len(tokens)} tokens")

        # Validate tokens are within vocabulary
        vocab_size = len(vocabulary)
        invalid_tokens = [t for t in tokens if t < 0 or t >= vocab_size]
        if invalid_tokens:
            result.errors.append(
                f"Found {len(invalid_tokens)} invalid tokens (out of vocab range 0-{vocab_size-1})"
            )
            return result

        logger.info(f"✓ All tokens within vocabulary range")

        # Extract metadata from reordered tokens
        metadata = extract_metadata(tokens, vocabulary)
        result.metadata = metadata

        # Check token count (exclude special tokens)
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

        result.tokens = tokens

        # All validations passed
        result.is_valid = True
        logger.info("✓ All validations passed successfully")

    except FileNotFoundError:
        result.errors.append(f"MIDI file not found: {midi_path}")
    except Exception as e:
        logger.error(f"Error parsing chord MIDI: {e}", exc_info=True)
        result.errors.append(f"Unexpected error: {str(e)}")

    return result


def extract_metadata(tokens: List[int], vocabulary: Dict[str, int]) -> ChordMetadata:
    """
    Extract metadata (tempo, time signature) from reordered token sequence.

    Uses vocabulary lookup instead of hardcoded token ranges.

    Args:
        tokens: List of token IDs (in new bar-level format)
        vocabulary: Token name to ID mapping

    Returns:
        ChordMetadata with extracted information
    """
    metadata = ChordMetadata()

    # Build reverse vocabulary for token name lookup
    id_to_name = {v: k for k, v in vocabulary.items()}
    chord_start_id = vocabulary.get(CHORD_START_TOKEN_NAME)

    for token_id in tokens:
        token_name = id_to_name.get(token_id, "")

        if token_name.startswith("Tempo_"):
            try:
                metadata.tempo = float(token_name.split("_")[1])
                logger.debug(f"Extracted tempo: {metadata.tempo} BPM")
            except (ValueError, IndexError):
                pass

        elif token_name.startswith("TimeSig_"):
            try:
                parts = token_name.split("_")[1].split("/")
                metadata.time_signature = (int(parts[0]), int(parts[1]))
                logger.debug(f"Extracted time signature: {metadata.time_signature}")
            except (ValueError, IndexError):
                pass

        elif token_id == chord_start_id:
            metadata.has_chord_start = True

    logger.info(f"Metadata extraction complete: tempo={metadata.tempo}, "
                f"time_sig={metadata.time_signature}, "
                f"has_chord_start={metadata.has_chord_start}")

    return metadata


def validate_chord_tokens(
    tokens: List[int],
    vocab_info: VocabularyInfo
) -> Tuple[bool, List[str]]:
    """
    Validate that chord tokens are valid for generation.

    Checks for CHORD_START structural marker and basic structure.

    Args:
        tokens: Token sequence to validate (in new bar-level format)
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

    # Check for CHORD_START structural marker
    if vocab_info.chord_start_token_id is None:
        errors.append("CHORD_START token not found in vocabulary")
    elif vocab_info.chord_start_token_id not in tokens:
        errors.append("CHORD_START marker not found in token sequence")

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
    'validate_chord_tokens',
    'get_token_count_with_context',
]
