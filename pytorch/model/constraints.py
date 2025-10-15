"""
Constraint masking functions for constrained music generation.

This module provides functions to enforce musical constraints during generation:
- Monophony for melody tracks
- Sustained chords for chord tracks
- Diatonic note preferences
- Track separation

The purpose of this script is to enforce several music theory aspects.

NOTE: This is a basic structure for Phase 2. Full implementation will be done
in Phase 4 (Generation Pipeline).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
from ..data.constants import TOKEN_RANGES, VOCAB_SIZE


class GenerationState:
    """
    Tracks the current state during music generation.

    This helps us make decisions about which tokens are allowed based on
    what has already been generated.
    """

    def __init__(self):
        """Initialize generation state."""
        self.active_melody_notes: List[int] = []
        self.active_chord_notes: List[int] = []
        self.current_track: Optional[str] = None  # 'melody' or 'chord'
        self.current_key: Optional[int] = None
        self.current_position: int = 0

    def note_on(self, pitch: int, track: str):
        """Register a note-on event."""
        if track == 'melody':
            self.active_melody_notes.append(pitch)
        elif track == 'chord':
            self.active_chord_notes.append(pitch)

    def note_off(self, pitch: int, track: str):
        """Register a note-off event."""
        if track == 'melody' and pitch in self.active_melody_notes:
            self.active_melody_notes.remove(pitch)
        elif track == 'chord' and pitch in self.active_chord_notes:
            self.active_chord_notes.remove(pitch)

    def has_active_melody_notes(self) -> bool:
        """Check if there are active melody notes."""
        return len(self.active_melody_notes) > 0

    def has_active_chord_notes(self) -> bool:
        """Check if there are active chord notes."""
        return len(self.active_chord_notes) > 0


def apply_monophony_constraint(
    logits: torch.Tensor,
    state: GenerationState,
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Enforce monophony constraint for melody track.

    When a melody note is already active, prevent generating another pitch token
    until the current note ends (duration token is generated).

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        state: Current generation state
        mask_value: Value to use for masked positions

    Returns:
        Masked logits with monophony constraint applied

    Due to this, our AI specific implementation remains mostly
    constrained to this specific use case.
    """
    if state.current_track != 'melody':
        return logits

    if state.has_active_melody_notes():
        # Block all pitch tokens
        pitch_start, pitch_end = TOKEN_RANGES['pitch']
        logits[:, pitch_start:pitch_end+1] = mask_value

    return logits


def apply_chord_sustain_constraint(
    logits: torch.Tensor,
    state: GenerationState,
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Enforce sustained chord constraint.

    Chords should sustain (no short rhythmic variations).
    This masks short duration tokens when generating chords.

    Args:
        logits: Model output logits
        state: Current generation state
        mask_value: Value to use for masked positions

    Returns:
        Masked logits with chord sustain constraint applied
    """
    if state.current_track != 'chord':
        return logits

    # In Phase 4, we'll implement logic to mask short duration tokens
    # For now, just return unchanged
    return logits


def apply_diatonic_boost(
    logits: torch.Tensor,
    key_signature: Optional[int],
    boost_weight: float = 2.0
) -> torch.Tensor:
    """
    Boost probabilities of diatonic pitches (notes in the current key).

    This is a soft constraint - we don't forbid non-diatonic notes, but
    we make diatonic notes more likely. This way we allow the model to
    make unconscious diatonic decisions.

    Args:
        logits: Model output logits
        key_signature: Current key (if known)
        boost_weight: Multiplicative boost for diatonic pitches

    Returns:
        Logits with diatonic pitches boosted
    """
    if key_signature is None:
        return logits

    # In Phase 4, we'll implement the full diatonic scale logic
    # For now, just return unchanged
    return logits


def constrained_decode_step(
    logits: torch.Tensor,
    state: GenerationState,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: Optional[int] = None
) -> torch.Tensor:
    """
    Perform one step of constrained decoding.

    Applies all relevant constraints and then samples from the filtered distribution.

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        state: Current generation state
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k filtering threshold

    Returns:
        Sampled token IDs, shape [batch_size, 1]
    """
    # Apply constraints
    logits = apply_monophony_constraint(logits, state)
    logits = apply_chord_sustain_constraint(logits, state)
    logits = apply_diatonic_boost(logits, state.current_key)

    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering if specified
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

    # Apply nucleus (top-p) filtering if specified
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to original positions
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def create_constraint_validator():
    """
    Factory function to create a constraint validator.

    This will be fully implemented in Phase 4.

    Returns:
        Constraint validator object
    """
    # Placeholder for Phase 4
    return None


__all__ = [
    'GenerationState',
    'apply_monophony_constraint',
    'apply_chord_sustain_constraint',
    'apply_diatonic_boost',
    'constrained_decode_step',
    'create_constraint_validator'
]
