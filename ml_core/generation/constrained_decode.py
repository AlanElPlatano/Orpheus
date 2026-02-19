"""
Constrained decoding for music generation.

Enhanced constraint implementations for diatonic scales, chord sustain
validation, and sequence completion detection. Uses the monophony
constraint and GenerationState from ml_core/model/constraints.py.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Set, Dict
import logging

from ..data.constants import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    BAR_TOKEN_ID,
    PAD_TOKEN_ID,
    MASK_TOKEN_ID,
    MAJOR_SCALE_INTERVALS,
    MINOR_SCALE_INTERVALS,
    is_pitch_token,
    is_duration_token,
    is_position_token,
    is_special_token
)
from ..model.constraints import GenerationState

logger = logging.getLogger(__name__)


# ============================================================================
# Scale and Key Utilities
# ============================================================================

# MIDI note names to chromatic pitch class (C=0, C#=1, ..., B=11)
NOTE_TO_PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11
}


def parse_key_signature(key: str) -> tuple[int, bool]:
    """
    Parse key signature string to root pitch class and major/minor.

    Args:
        key: Key signature string (e.g., "Fm", "C", "Db", "G#m")

    Returns:
        Tuple of (root_pitch_class, is_minor)

    Examples:
        >>> parse_key_signature("Fm")
        (5, True)  # F=5, minor=True
        >>> parse_key_signature("C")
        (0, False)  # C=0, major=True
    """
    key = key.strip()
    is_minor = key.endswith('m')

    # Remove 'm' suffix if present
    root_name = key[:-1] if is_minor else key

    if root_name not in NOTE_TO_PITCH_CLASS:
        raise ValueError(f"Invalid key signature: {key}")

    root_pitch_class = NOTE_TO_PITCH_CLASS[root_name]

    return root_pitch_class, is_minor


def get_diatonic_pitches(key: str) -> Set[int]:
    """
    Get set of MIDI pitches that are diatonic (in-scale) for a given key.

    Args:
        key: Key signature string (e.g., "Fm", "C")

    Returns:
        Set of MIDI pitch numbers that are in the scale (all octaves)

    Example:
        >>> pitches = get_diatonic_pitches("C")
        >>> 60 in pitches  # Middle C
        True
        >>> 61 in pitches  # C#
        False
    """
    root_pitch_class, is_minor = parse_key_signature(key)

    # Select scale intervals
    intervals = MINOR_SCALE_INTERVALS if is_minor else MAJOR_SCALE_INTERVALS

    # Generate pitch classes in the scale
    scale_pitch_classes = {(root_pitch_class + interval) % 12 for interval in intervals}

    # Generate MIDI pitches across all octaves (MIDI range 0-127)
    diatonic_pitches = set()
    for pitch in range(128):
        pitch_class = pitch % 12
        if pitch_class in scale_pitch_classes:
            diatonic_pitches.add(pitch)

    return diatonic_pitches


def get_diatonic_token_ids(
    key: str,
    pitch_token_to_midi: Dict[int, int]
) -> Set[int]:
    """
    Get set of pitch token IDs that are diatonic for a given key.

    Args:
        key: Key signature string
        pitch_token_to_midi: Mapping from pitch token ID to MIDI pitch number

    Returns:
        Set of pitch token IDs that are in-scale
    """
    diatonic_pitches = get_diatonic_pitches(key)

    diatonic_token_ids = {
        token_id
        for token_id, midi_pitch in pitch_token_to_midi.items()
        if midi_pitch in diatonic_pitches
    }

    return diatonic_token_ids


# ============================================================================
# State Management
# ============================================================================

def update_generation_state(
    state: GenerationState,
    token_id: int,
    vocab_info: 'VocabularyInfo'
) -> None:
    """
    Update generation state based on newly generated token.

    Tracks active notes, position, track context, and structural markers.

    Args:
        state: Current generation state
        token_id: Newly generated token ID
        vocab_info: Vocabulary information for token categorization
    """
    # Track switching via structural markers
    if token_id == vocab_info.chord_start_token_id:
        state.current_track = 'chord'
        return

    if token_id == vocab_info.melody_start_token_id:
        state.current_track = 'melody'
        return

    # Update based on token type
    if vocab_info.is_pitch_token(token_id):
        token_name = vocab_info.get_token_name(token_id)
        try:
            midi_pitch = int(token_name.split('_')[1])
            state.note_on(midi_pitch, state.current_track or 'melody')
        except (IndexError, ValueError):
            logger.warning(f"Could not parse pitch from token: {token_name}")

    elif vocab_info.is_duration_token(token_id):
        # Duration comes after pitch in REMI, clear active notes
        if state.current_track == 'melody':
            state.active_melody_notes.clear()
        elif state.current_track == 'chord':
            state.active_chord_notes.clear()

    elif vocab_info.is_position_token(token_id):
        token_name = vocab_info.get_token_name(token_id)
        try:
            relative_position = int(token_name.split('_')[1])
            absolute_position = state.current_bar * 48 + relative_position
            state.current_position = absolute_position
        except (IndexError, ValueError):
            pass

    elif token_id == BAR_TOKEN_ID:
        state.current_bar += 1
        state.current_position = state.current_bar * 48


# ============================================================================
# Constraint Application
# ============================================================================

def apply_grammar_constraint(
    logits: torch.Tensor,
    generated_tokens: List[int],
    vocab_info: 'VocabularyInfo',
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Enforce REMI note event grammar: Pitch -> Velocity -> Duration.

    After a Pitch token, only Velocity tokens are allowed.
    After a Velocity token, only Duration tokens are allowed.
    This prevents the model from generating broken note events.

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        generated_tokens: Previously generated token IDs
        vocab_info: Vocabulary information
        mask_value: Value to use for masked positions

    Returns:
        Grammar-constrained logits
    """
    if not generated_tokens:
        return logits

    last_token = generated_tokens[-1]

    # After a Pitch token, only Velocity tokens are allowed
    if vocab_info.is_pitch_token(last_token):
        constrained = torch.full_like(logits, mask_value)
        for token_id in vocab_info.velocity_tokens:
            constrained[:, token_id] = logits[:, token_id]
        return constrained

    # After a Velocity token, only Duration tokens are allowed
    if last_token in vocab_info.velocity_tokens:
        constrained = torch.full_like(logits, mask_value)
        for token_id in vocab_info.duration_tokens:
            constrained[:, token_id] = logits[:, token_id]
        return constrained

    return logits


def apply_consecutive_repetition_constraint(
    logits: torch.Tensor,
    generated_tokens: List[int],
    max_consecutive: int = 3,
    max_cycle_length: int = 8,
    min_cycle_repetitions: int = 3,
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Hard constraint: prevent both single-token loops and short repeating cycles.

    Detects two types of degenerate patterns:
    1. Single token repeated N+ times: AAAA... -> mask A
    2. Short cycle repeated M+ times: ABCDABCDABCD... -> mask next token in cycle

    Special tokens (BOS, EOS, PAD, BAR) are exempt since they may legitimately repeat.

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        generated_tokens: List of previously generated token IDs
        max_consecutive: Max allowed consecutive repeats of same token (default: 3)
        max_cycle_length: Max cycle length to check for (default: 8)
        min_cycle_repetitions: Min repetitions to trigger cycle detection (default: 3)
        mask_value: Value to use for masked positions

    Returns:
        Logits with degenerate pattern tokens masked
    """
    if len(generated_tokens) < max_consecutive:
        return logits

    exempt_tokens = {BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, BAR_TOKEN_ID}
    cloned = False

    # 1) Check single-token repetition (AAAA...)
    recent = generated_tokens[-max_consecutive:]
    if len(set(recent)) == 1:
        repeated_token = recent[0]
        if repeated_token not in exempt_tokens:
            if not cloned:
                logits = logits.clone()
                cloned = True
            logits[:, repeated_token] = mask_value
            logger.debug(
                f"Blocked token {repeated_token} after {max_consecutive} consecutive repetitions"
            )
            return logits

    # 2) Check short cycle repetition (ABCDABCDABCD...)
    for cycle_len in range(2, max_cycle_length + 1):
        needed = cycle_len * min_cycle_repetitions
        if len(generated_tokens) < needed:
            continue

        tail = generated_tokens[-needed:]
        pattern = tail[:cycle_len]

        # Verify the pattern repeats exactly min_cycle_repetitions times
        is_cycle = True
        for i in range(cycle_len, needed):
            if tail[i] != pattern[i % cycle_len]:
                is_cycle = False
                break

        if is_cycle:
            # The next token the model would generate to continue the cycle
            next_in_cycle = pattern[0]
            if next_in_cycle not in exempt_tokens:
                if not cloned:
                    logits = logits.clone()
                    cloned = True
                logits[:, next_in_cycle] = mask_value
                logger.debug(
                    f"Blocked repeating cycle of length {cycle_len} "
                    f"(pattern: {pattern}), masked token {next_in_cycle}"
                )
            break  # Handle shortest detected cycle only

    return logits


def apply_diatonic_boost_enhanced(
    logits: torch.Tensor,
    key: Optional[str],
    pitch_token_to_midi: Dict[int, int],
    boost_weight: float = 2.0,
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Boost probabilities of diatonic pitches with full implementation.

    This is a soft constraint - we don't forbid non-diatonic notes,
    but we make diatonic notes more likely.

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        key: Key signature (e.g., "Fm", "C"), None to skip
        pitch_token_to_midi: Mapping from pitch token ID to MIDI pitch
        boost_weight: Multiplicative boost for diatonic pitches
        mask_value: Not used (kept for API compatibility)

    Returns:
        Logits with diatonic pitches boosted
    """
    if key is None:
        return logits

    try:
        # Get diatonic token IDs
        diatonic_token_ids = get_diatonic_token_ids(key, pitch_token_to_midi)

        # Boost diatonic pitch logits
        boosted_logits = logits.clone()
        for token_id in diatonic_token_ids:
            boosted_logits[:, token_id] *= boost_weight

        return boosted_logits

    except Exception as e:
        logger.warning(f"Failed to apply diatonic boost: {e}")
        return logits


def apply_chord_sustain_constraint_enhanced(
    logits: torch.Tensor,
    state: GenerationState,
    vocab_info: 'VocabularyInfo',
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Enforce sustained chord constraint with full implementation.

    Masks short duration tokens when generating chords to ensure
    they sustain throughout the bar.

    Args:
        logits: Model output logits
        state: Current generation state
        vocab_info: Vocabulary information
        mask_value: Value to use for masked positions

    Returns:
        Masked logits with chord sustain constraint applied
    """
    if state.current_track != 'chord':
        return logits

    # When we're in chord generation and have active chord notes,
    # we should only allow long durations (quarter note or longer)
    if state.has_active_chord_notes():
        # Mask short duration tokens
        # In REMI tokenization, duration tokens are like "Duration_0.1.4", "Duration_1.0.4", etc.
        # We want to keep only durations >= 1.0 beats for chords

        for token_id in vocab_info.duration_tokens:
            token_name = vocab_info.get_token_name(token_id)

            try:
                # Parse duration (e.g., "Duration_0.1.4" -> 0.1 beats)
                duration_str = token_name.split('_')[1]  # "0.1.4"
                duration_beats = float(duration_str.split('.')[0] + '.' + duration_str.split('.')[1])

                # Mask durations shorter than 1.0 beats
                if duration_beats < 1.0:
                    logits[:, token_id] = mask_value

            except (IndexError, ValueError):
                # Could not parse duration, skip
                pass

    return logits


def apply_all_constraints(
    logits: torch.Tensor,
    state: GenerationState,
    vocab_info: 'VocabularyInfo',
    pitch_token_to_midi: Optional[Dict[int, int]] = None,
    key: Optional[str] = None,
    diatonic_boost_weight: float = 2.0,
    generated_tokens: Optional[List[int]] = None,
    max_consecutive_repetitions: int = 5
) -> torch.Tensor:
    """
    Apply all musical constraints to logits.

    Combines monophony, chord sustain, diatonic boosting, and repetition prevention.

    Args:
        logits: Model output logits
        state: Current generation state
        vocab_info: Vocabulary information
        pitch_token_to_midi: Mapping for diatonic boosting (optional)
        key: Key signature for diatonic boosting (optional)
        diatonic_boost_weight: Boost weight for diatonic pitches
        generated_tokens: Previously generated tokens for repetition constraint
        max_consecutive_repetitions: Max allowed consecutive repeats of same token

    Returns:
        Constrained logits
    """
    from ..model.constraints import apply_monophony_constraint

    # Apply REMI grammar constraint (Pitch -> Velocity -> Duration)
    if generated_tokens is not None:
        logits = apply_grammar_constraint(logits, generated_tokens, vocab_info)

    logits = apply_monophony_constraint(logits, state, vocab_info=vocab_info)

    # Apply chord sustain constraint (enhanced version)
    logits = apply_chord_sustain_constraint_enhanced(logits, state, vocab_info)

    # Apply diatonic boost (enhanced version)
    if pitch_token_to_midi is not None and key is not None:
        logits = apply_diatonic_boost_enhanced(
            logits, key, pitch_token_to_midi, diatonic_boost_weight
        )

    # Apply consecutive repetition constraint (prevents infinite loops)
    if generated_tokens is not None:
        logits = apply_consecutive_repetition_constraint(
            logits, generated_tokens, max_consecutive=max_consecutive_repetitions
        )

    return logits


# ============================================================================
# Sequence Completion Detection
# ============================================================================

def is_sequence_complete(
    generated_tokens: List[int],
    max_length: int,
    min_bars: int = 8,
    vocab_info: Optional['VocabularyInfo'] = None
) -> bool:
    """
    Check if generated sequence is complete.

    A sequence is complete if:
    1. EOS token is generated, OR
    2. Maximum length is reached, OR
    3. Minimum number of bars have been generated

    Args:
        generated_tokens: List of generated token IDs
        max_length: Maximum sequence length
        min_bars: Minimum number of bars for valid sequence
        vocab_info: Vocabulary information (optional, for bar counting)

    Returns:
        True if sequence is complete
    """
    # Check for EOS token
    if EOS_TOKEN_ID in generated_tokens:
        return True

    # Check for maximum length
    if len(generated_tokens) >= max_length:
        return True

    # Check for minimum bars (if vocab_info provided)
    if vocab_info is not None:
        num_bars = sum(1 for token in generated_tokens if token == BAR_TOKEN_ID)
        if num_bars >= min_bars:
            return True

    return False


def should_stop_generation(
    generated_tokens: List[int],
    config: 'GenerationConfig',
    vocab_info: Optional['VocabularyInfo'] = None
) -> tuple[bool, str]:
    """
    Determine if generation should stop.

    Args:
        generated_tokens: List of generated token IDs
        config: Generation configuration
        vocab_info: Vocabulary information (optional)

    Returns:
        Tuple of (should_stop, reason)
    """
    # Check for EOS token
    if EOS_TOKEN_ID in generated_tokens:
        return True, "EOS token generated"

    # Check for maximum length
    if len(generated_tokens) >= config.max_length:
        return True, "Maximum length reached"

    # Check for maximum bars
    if vocab_info is not None:
        num_bars = sum(1 for token in generated_tokens if token == BAR_TOKEN_ID)
        if num_bars >= config.max_generation_bars:
            return True, f"Maximum bars ({config.max_generation_bars}) reached"

    return False, ""


__all__ = [
    'parse_key_signature',
    'get_diatonic_pitches',
    'get_diatonic_token_ids',
    'update_generation_state',
    'apply_grammar_constraint',
    'apply_consecutive_repetition_constraint',
    'apply_diatonic_boost_enhanced',
    'apply_chord_sustain_constraint_enhanced',
    'apply_all_constraints',
    'is_sequence_complete',
    'should_stop_generation'
]
