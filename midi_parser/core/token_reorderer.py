"""
Post-processing module that restructures REMI token sequences for bar-level
chord/melody separation.

Transforms MidiTok's interleaved token output into a structured format where
each bar has explicit CHORD_START and MELODY_START boundaries, replacing
redundant per-note Program tokens.

Before: Bar | Pos | Tempo | Program_98 | Pitch | Vel | Dur | Program_29 | Pitch | Vel | Dur ...
After:  Bar | TimeSig | Tempo | CHORD_START | notes... | MELODY_START | Pos | notes...
"""

import logging
from typing import Dict, List, Tuple, Set

from ml_core.data.constants import (
    CHORD_START_TOKEN_NAME,
    MELODY_START_TOKEN_NAME,
    CORRIDOS_CHORD_PROGRAM,
    CORRIDOS_MELODY_PROGRAM,
)

logger = logging.getLogger(__name__)


def reorder_bar_tokens(
    tokens: List[int],
    vocabulary: Dict[str, int],
) -> Tuple[List[int], Dict[str, int]]:
    """
    Restructure a flat REMI token sequence into bar-level chord/melody sections.

    Replaces per-note Program tokens with CHORD_START and MELODY_START boundary
    markers, grouping all chord notes before melody notes within each bar.

    Args:
        tokens: Raw MidiTok token IDs
        vocabulary: Token name -> ID mapping from MidiTok

    Returns:
        Tuple of (reordered token list, updated vocabulary with new tokens)
    """
    updated_vocab = _extend_vocabulary(vocabulary)
    reverse_vocab = {token_id: name for name, token_id in updated_vocab.items()}

    chord_start_id = updated_vocab[CHORD_START_TOKEN_NAME]
    melody_start_id = updated_vocab[MELODY_START_TOKEN_NAME]

    program_ids = _find_program_ids(updated_vocab)
    chord_program_id = updated_vocab.get(f"Program_{CORRIDOS_CHORD_PROGRAM}")
    melody_program_id = updated_vocab.get(f"Program_{CORRIDOS_MELODY_PROGRAM}")

    bar_id = updated_vocab.get("Bar_None")

    bars = _split_into_bars(tokens, bar_id)

    reordered = []
    for bar_tokens in bars:
        reordered.extend(
            _reorder_single_bar(
                bar_tokens,
                reverse_vocab,
                program_ids,
                chord_program_id,
                melody_program_id,
                chord_start_id,
                melody_start_id,
                bar_id,
            )
        )

    token_savings = len(tokens) - len(reordered)
    logger.info(
        f"Token reordering complete: {len(tokens)} -> {len(reordered)} "
        f"(saved {token_savings} tokens)"
    )

    return reordered, updated_vocab


def _extend_vocabulary(vocabulary: Dict[str, int]) -> Dict[str, int]:
    """Append CHORD_START and MELODY_START tokens to the vocabulary."""
    extended = dict(vocabulary)
    base_size = len(vocabulary)

    extended[CHORD_START_TOKEN_NAME] = base_size
    extended[MELODY_START_TOKEN_NAME] = base_size + 1

    logger.info(
        f"Extended vocabulary: {base_size} -> {len(extended)} "
        f"({CHORD_START_TOKEN_NAME}={base_size}, "
        f"{MELODY_START_TOKEN_NAME}={base_size + 1})"
    )

    return extended


def _find_program_ids(vocabulary: Dict[str, int]) -> Set[int]:
    """Collect all Program token IDs from the vocabulary."""
    return {
        token_id
        for name, token_id in vocabulary.items()
        if name.startswith("Program_")
    }


def _split_into_bars(tokens: List[int], bar_id: int) -> List[List[int]]:
    """
    Split a flat token sequence into per-bar segments.

    Each segment starts with the Bar token. Tokens before the first Bar
    (if any) are placed in the first segment.
    """
    bars = []
    current_bar = []

    for token in tokens:
        if token == bar_id and current_bar:
            bars.append(current_bar)
            current_bar = [token]
        else:
            current_bar.append(token)

    if current_bar:
        bars.append(current_bar)

    return bars


def _classify_token(name: str) -> str:
    """Classify a token name into its functional category."""
    if name.startswith("TimeSig_") or name.startswith("Tempo_"):
        return "metadata"
    if name.startswith("Position_"):
        return "position"
    if name.startswith("Program_"):
        return "program"
    if name.startswith("Pitch_"):
        return "pitch"
    if name.startswith("Velocity_"):
        return "velocity"
    if name.startswith("Duration_"):
        return "duration"
    if name == "Bar_None":
        return "bar"
    if name.startswith("Chord_"):
        return "chord_annotation"
    return "other"


def _reorder_single_bar(
    bar_tokens: List[int],
    reverse_vocab: Dict[int, str],
    program_ids: Set[int],
    chord_program_id: int,
    melody_program_id: int,
    chord_start_id: int,
    melody_start_id: int,
    bar_id: int,
) -> List[int]:
    """
    Reorder tokens within a single bar into chord-first, melody-second structure.

    Walks through bar tokens tracking the active program to classify each note
    group (Pitch+Velocity+Duration) as chord or melody. Strips Program tokens
    and replaces them with CHORD_START/MELODY_START boundary markers.
    """
    bar_marker = []
    metadata_tokens = []
    chord_notes = []
    melody_notes = []

    current_program = None
    current_position_token = None

    for token_id in bar_tokens:
        name = reverse_vocab.get(token_id, "<UNK>")
        category = _classify_token(name)

        if category == "bar":
            bar_marker.append(token_id)
            continue

        if category == "metadata" or category == "chord_annotation":
            metadata_tokens.append(token_id)
            continue

        if category == "program":
            current_program = token_id
            continue

        if category == "position":
            current_position_token = token_id
            continue

        if category == "pitch":
            is_chord = (current_program == chord_program_id)
            target = chord_notes if is_chord else melody_notes

            if not is_chord and current_position_token is not None:
                target.append(current_position_token)
                current_position_token = None

            target.append(token_id)
            continue

        if category in ("velocity", "duration"):
            is_chord = (current_program == chord_program_id)
            target = chord_notes if is_chord else melody_notes
            target.append(token_id)
            continue

        # Fallback: treat unknown tokens as metadata
        metadata_tokens.append(token_id)

    result = bar_marker + metadata_tokens
    result.append(chord_start_id)
    result.extend(chord_notes)
    result.append(melody_start_id)
    result.extend(melody_notes)

    return result


def restore_program_tokens(
    tokens: List[int],
    vocabulary: Dict[str, int],
) -> List[int]:
    """
    Reverse the reordering transformation for MIDI export via miditok.

    Replaces CHORD_START/MELODY_START markers with per-note Program tokens
    and re-inserts Position_0 for chord sections, producing a token sequence
    that miditok's decoder can process.

    Args:
        tokens: Reordered token IDs containing structural markers
        vocabulary: Vocabulary mapping (must include structural tokens)

    Returns:
        Token sequence with Program tokens restored and structural markers removed
    """
    chord_start_id = vocabulary.get(CHORD_START_TOKEN_NAME)
    melody_start_id = vocabulary.get(MELODY_START_TOKEN_NAME)

    if chord_start_id is None or melody_start_id is None:
        logger.warning("Structural tokens not found in vocabulary, returning tokens unchanged")
        return tokens

    chord_program_id = vocabulary.get(f"Program_{CORRIDOS_CHORD_PROGRAM}")
    melody_program_id = vocabulary.get(f"Program_{CORRIDOS_MELODY_PROGRAM}")
    position_0_id = vocabulary.get("Position_0")

    reverse_vocab = {token_id: name for name, token_id in vocabulary.items()}
    restored = []
    current_section = None
    chord_position_injected = False

    for token_id in tokens:
        if token_id == chord_start_id:
            current_section = "chord"
            chord_position_injected = False
            continue

        if token_id == melody_start_id:
            current_section = "melody"
            continue

        name = reverse_vocab.get(token_id, "<UNK>")

        if name.startswith("Pitch_") and current_section is not None:
            if current_section == "chord":
                if not chord_position_injected and position_0_id is not None:
                    restored.append(position_0_id)
                    chord_position_injected = True
                if chord_program_id is not None:
                    restored.append(chord_program_id)
            elif current_section == "melody":
                if melody_program_id is not None:
                    restored.append(melody_program_id)

        restored.append(token_id)

    return restored


__all__ = [
    "reorder_bar_tokens",
    "restore_program_tokens",
]
