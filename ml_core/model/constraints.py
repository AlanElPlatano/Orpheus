"""
Base constraint primitives for constrained music generation.

Provides GenerationState tracking and the monophony constraint used by
the generation pipeline. Enhanced constraint variants (diatonic boost,
chord sustain) live in ml_core/generation/constrained_decode.py.
"""

import torch
from typing import Optional, List, Set
from ..data.constants import TOKEN_RANGES


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
        self.current_bar: int = 0  # Track which bar we're in
        self.current_position: int = 0  # Absolute position in ticks

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
    vocab_info: Optional['VocabularyInfo'] = None,
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Enforce monophony constraint for melody track.

    When a melody note is already active, prevent generating another pitch token
    until the current note ends (duration token is generated).

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        state: Current generation state
        vocab_info: Vocabulary info with accurate pitch token set (preferred).
                    Falls back to TOKEN_RANGES if not provided.
        mask_value: Value to use for masked positions

    Returns:
        Masked logits with monophony constraint applied
    """
    if state.current_track != 'melody':
        return logits

    if state.has_active_melody_notes():
        if vocab_info is not None:
            # Use accurate pitch token set from vocabulary
            for token_id in vocab_info.pitch_tokens:
                logits[:, token_id] = mask_value
        else:
            # Fallback to TOKEN_RANGES
            pitch_start, pitch_end = TOKEN_RANGES['pitch']
            logits[:, pitch_start:pitch_end+1] = mask_value

    return logits


__all__ = [
    'GenerationState',
    'apply_monophony_constraint'
]