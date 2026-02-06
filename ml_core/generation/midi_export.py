"""
Token-to-MIDI export functionality.

Converts generated token sequences directly to MIDI files using miditok's
decode function, bypassing the intermediate JSON step.
"""

import torch
import json
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

from miditoolkit import MidiFile

from ..data.vocab import VocabularyInfo
from ..data.constants import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID
)
from .chord_sustain import apply_chord_sustain

logger = logging.getLogger(__name__)


def tokens_to_midi(
    token_ids: List[int],
    vocab_info: VocabularyInfo,
    output_path: Path,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    remove_special_tokens: bool = True,
    apply_chord_sustain: bool = True
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Convert token sequence directly to MIDI file using miditok.

    Args:
        token_ids: List of integer token IDs
        vocab_info: Vocabulary information
        output_path: Where to save MIDI file
        tokenizer_config: Tokenizer configuration (pitch_range, beat_res, etc.)
        remove_special_tokens: Whether to remove BOS/EOS/PAD tokens before decoding

    Returns:
        Tuple of (success, midi_path, error_message)
    """
    try:
        # Remove special tokens if requested
        if remove_special_tokens:
            cleaned_tokens = [
                token_id for token_id in token_ids
                if token_id not in {BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID}
            ]
        else:
            cleaned_tokens = token_ids

        if not cleaned_tokens:
            return False, None, "No tokens to decode after removing special tokens"

        logger.info(f"Converting {len(cleaned_tokens)} tokens to MIDI")

        # Create miditok tokenizer with same config used for training
        from midi_parser.core.tokenizer_manager import TokenizerManager
        from midi_parser.config.defaults import MidiParserConfig, TokenizerConfig

        # Build tokenizer config
        if tokenizer_config is not None:
            tok_config = TokenizerConfig(
                pitch_range=tuple(tokenizer_config.get('pitch_range', (36, 84))),
                beat_resolution=tokenizer_config.get('beat_resolution', 4),
                num_velocities=tokenizer_config.get('num_velocities', 8),
                additional_tokens=tokenizer_config.get('additional_tokens', {
                    "Chord": True,
                    "Rest": True,
                    "Tempo": True,
                    "TimeSignature": True
                }),
                max_seq_length=tokenizer_config.get('max_seq_length', 2048)
            )
        else:
            # Use default config matching training data
            logger.warning("No tokenizer config provided, using default config")
            tok_config = TokenizerConfig(
                pitch_range=(36, 84),
                beat_resolution=4,
                num_velocities=8,
                additional_tokens={
                    "Chord": True,
                    "Rest": True,
                    "Tempo": True,
                    "TimeSignature": True
                },
                max_seq_length=2048
            )

        # Create parser config and tokenizer manager
        parser_config = MidiParserConfig(
            tokenization="REMI",
            tokenizer=tok_config
        )

        tokenizer_manager = TokenizerManager(parser_config)
        tokenizer = tokenizer_manager.create_tokenizer("REMI", tok_config)

        logger.info("Decoding tokens to symusic.Score...")

        # Decode tokens to symusic.Score
        try:
            score = tokenizer.decode(cleaned_tokens)
        except Exception as e:
            logger.error(f"Failed to decode tokens: {e}")
            return False, None, f"Token decoding failed: {str(e)}"

        if score is None:
            return False, None, "Tokenizer returned None score"

        logger.info(f"Decoded score with {len(score.tracks)} tracks")

        # Convert symusic.Score to miditoolkit.MidiFile via temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.mid', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Dump score to temporary MIDI file
            score.dump_midi(tmp_path)

            # Load with miditoolkit
            midi = MidiFile(tmp_path)

            logger.info(f"Loaded MIDI with {len(midi.instruments)} instruments")

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {e}")

        # Optionally apply chord sustain post-processing
        # This extends chord durations to the start of the next chord
        if apply_chord_sustain:
            midi = apply_chord_sustain(midi)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save MIDI file
        midi.dump(str(output_path))

        logger.info(f"Successfully saved MIDI to {output_path}")

        # Log statistics
        total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
        logger.info(f"  Total tracks: {len(midi.instruments)}")
        logger.info(f"  Total notes: {total_notes}")

        return True, output_path, None

    except Exception as e:
        error_msg = f"Error converting tokens to MIDI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def save_token_sequence(
    token_ids: List[int],
    output_path: Path,
    vocab_info: Optional[VocabularyInfo] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Save token sequence to JSON file for debugging/analysis.

    Args:
        token_ids: List of token IDs
        output_path: Where to save JSON file
        vocab_info: Vocabulary information for token names (optional)
        metadata: Additional metadata to include (optional)

    Returns:
        Tuple of (success, error_message)
    """
    try:
        data = {
            "token_ids": token_ids,
            "sequence_length": len(token_ids),
            "metadata": metadata or {}
        }

        # Add token names if vocab_info provided
        if vocab_info is not None:
            data["token_names"] = [
                vocab_info.get_token_name(token_id)
                for token_id in token_ids
            ]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved token sequence to {output_path}")

        return True, None

    except Exception as e:
        error_msg = f"Error saving token sequence: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def load_token_sequence(
    json_path: Path
) -> Tuple[bool, Optional[List[int]], Optional[str]]:
    """
    Load token sequence from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Tuple of (success, token_ids, error_message)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        token_ids = data.get('token_ids', [])

        if not token_ids:
            return False, None, "No token_ids found in JSON file"

        logger.info(f"Loaded {len(token_ids)} tokens from {json_path}")

        return True, token_ids, None

    except Exception as e:
        error_msg = f"Error loading token sequence: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg


def batch_tokens_to_midi(
    token_sequences: List[List[int]],
    vocab_info: VocabularyInfo,
    output_dir: Path,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    filename_template: str = "generated_{index:03d}",
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Convert multiple token sequences to MIDI files.

    Args:
        token_sequences: List of token ID sequences
        vocab_info: Vocabulary information
        output_dir: Output directory for MIDI files
        tokenizer_config: Tokenizer configuration
        filename_template: Template for output filenames
        progress_callback: Callback function(current, total, filename)

    Returns:
        List of result dictionaries with success status and paths
    """
    results = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, token_ids in enumerate(token_sequences, 1):
        # Generate filename
        filename = filename_template.format(index=i) + ".mid"
        output_path = output_dir / filename

        # Call progress callback if provided
        if progress_callback:
            progress_callback(i, len(token_sequences), filename)

        # Convert to MIDI
        success, midi_path, error = tokens_to_midi(
            token_ids,
            vocab_info,
            output_path,
            tokenizer_config
        )

        results.append({
            "index": i,
            "filename": filename,
            "success": success,
            "midi_path": str(midi_path) if midi_path else None,
            "error": error,
            "num_tokens": len(token_ids)
        })

    return results


__all__ = [
    'tokens_to_midi',
    'save_token_sequence',
    'load_token_sequence',
    'batch_tokens_to_midi'
]