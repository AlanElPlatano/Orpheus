"""
Two-stage music generation orchestrator.

Implements the two-stage generation process:
1. Stage 1: Generate chord progression
2. Stage 2: Generate melody conditioned on chords

This matches the design document's approach where chords are generated first,
then the melody is generated with awareness of the harmonic context.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
import logging

from ..data.constants import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    BAR_TOKEN_ID,
    TRACK_TYPE_MELODY,
    TRACK_TYPE_CHORD,
    is_pitch_token,
    is_duration_token
)
from ..data.vocab import VocabularyInfo
from ..model.constraints import GenerationState
from .generation_config import GenerationConfig
from .sampling import sample_next_token, sample_next_token_track_aware
from .constrained_decode import (
    update_generation_state,
    apply_all_constraints,
    should_stop_generation
)

logger = logging.getLogger(__name__)


class TwoStageGenerator:
    """
    Two-stage music generation orchestrator.

    Generates complete MIDI sequences in two stages:
    1. Chord generation (with melody tokens masked)
    2. Melody generation (conditioned on generated chords)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vocab_info: VocabularyInfo,
        config: GenerationConfig,
        device: str = 'cuda',
        use_track_aware_sampling: bool = True
    ):
        """
        Initialize two-stage generator.

        Args:
            model: Trained transformer model
            vocab_info: Vocabulary information
            config: Generation configuration
            device: Device to run on ('cuda' or 'cpu')
            use_track_aware_sampling: Whether to use track-aware constraints (default: True)
        """
        self.model = model
        self.vocab_info = vocab_info
        self.config = config
        self.device = device
        self.use_track_aware_sampling = use_track_aware_sampling

        # Build pitch token to MIDI mapping for diatonic constraints
        self.pitch_token_to_midi = self._build_pitch_mapping()

        # Model should be in eval mode
        self.model.eval()

        logger.info(f"TwoStageGenerator initialized on {device}")
        logger.info(f"Track-aware sampling: {use_track_aware_sampling}")

    def _build_pitch_mapping(self) -> Dict[int, int]:
        """
        Build mapping from pitch token ID to MIDI pitch number.

        Returns:
            Dictionary mapping token ID to MIDI pitch
        """
        pitch_mapping = {}

        for token_id in self.vocab_info.pitch_tokens:
            token_name = self.vocab_info.get_token_name(token_id)

            try:
                # Parse pitch from token name (e.g., "Pitch_60" -> 60)
                midi_pitch = int(token_name.split('_')[1])
                pitch_mapping[token_id] = midi_pitch
            except (IndexError, ValueError):
                logger.warning(f"Could not parse pitch from token: {token_name}")

        return pitch_mapping

    def generate_complete_sequence(
        self,
        prompt_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        key_ids: Optional[torch.Tensor] = None,
        tempo_values: Optional[torch.Tensor] = None,
        time_sig_ids: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Generate a complete music sequence using two-stage generation.

        Args:
            prompt_tokens: Optional conditioning tokens to start with
            seed: Optional random seed for reproducibility
            temperature: Optional temperature override (uses config.temperature if None)
            key_ids: Optional key signature conditioning tensor, shape [1]
            tempo_values: Optional tempo conditioning tensor, shape [1]
            time_sig_ids: Optional time signature conditioning tensor, shape [1]

        Returns:
            List of generated token IDs
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Use provided temperature or fall back to config
        effective_temperature = temperature if temperature is not None else self.config.temperature

        # Stage 1: Generate chord progression
        logger.info("Stage 1: Generating chord progression...")
        chord_tokens = self._generate_chords(
            prompt_tokens,
            effective_temperature,
            key_ids,
            tempo_values,
            time_sig_ids
        )

        logger.info(f"Stage 1 complete: {len(chord_tokens)} chord tokens generated")

        # Check if chord sequence is too long for Stage 2
        # Reserve space for melody generation (25% of context length)
        # For default 2048 context: 512 tokens (original behavior)
        # For low_memory 512 context: 128 tokens (scaled appropriately)
        melody_reservation = max(int(self.config.max_length * 0.25), 64)
        max_chord_context = self.config.max_length - melody_reservation

        # Sanity check: ensure we have at least some space for chords
        if max_chord_context < 32:
            raise ValueError(
                f"Context length ({self.config.max_length}) is too small for two-stage generation. "
                f"Need at least {melody_reservation + 32} tokens."
            )

        if len(chord_tokens) > max_chord_context:
            original_length = len(chord_tokens)
            logger.warning(
                f"Chord sequence too long ({original_length} tokens), "
                f"truncating to {max_chord_context} tokens to reserve space for melody generation. "
                f"Keeping most recent {max_chord_context} tokens for harmonic context."
            )
            # Keep the beginning of the tokens
            chord_tokens = chord_tokens[:max_chord_context]

        # Stage 2: Generate melody conditioned on chords
        logger.info("Stage 2: Generating melody...")
        full_sequence = self._generate_melody(
            chord_tokens,
            effective_temperature,
            key_ids,
            tempo_values,
            time_sig_ids
        )

        logger.info(f"Generation complete: {len(full_sequence)} total tokens")

        return full_sequence

    def _generate_chords(
        self,
        prompt_tokens: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        key_ids: Optional[torch.Tensor] = None,
        tempo_values: Optional[torch.Tensor] = None,
        time_sig_ids: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Generate chord progression (Stage 1).

        In this stage, we mask melody-related tokens to ensure only
        chord progressions are generated.

        Args:
            prompt_tokens: Optional conditioning tokens
            temperature: Temperature for sampling (uses config if None)
            key_ids: Optional key signature conditioning tensor, shape [1]
            tempo_values: Optional tempo conditioning tensor, shape [1]
            time_sig_ids: Optional time signature conditioning tensor, shape [1]

        Returns:
            List of chord token IDs
        """
        # Initialize sequence with BOS token
        if prompt_tokens is not None:
            generated_tokens = [BOS_TOKEN_ID] + prompt_tokens
        else:
            generated_tokens = [BOS_TOKEN_ID]

        # Initialize generation state
        state = GenerationState()
        state.current_track = 'chord'
        state.current_key = self.config.key  # For future conditional generation

        # Use provided temperature or fall back to config
        effective_temperature = temperature if temperature is not None else self.config.temperature

        # Convert to tensor
        input_ids = torch.tensor([generated_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            while True:
                # Check stop conditions
                should_stop, reason = should_stop_generation(
                    generated_tokens, self.config, self.vocab_info
                )

                if should_stop:
                    logger.info(f"Chord generation stopped: {reason}")
                    break

                # Forward pass through model (with conditioning if available)
                # Note: track_ids are None for generation (model adds them internally if needed)
                logits, _ = self.model(
                    input_ids,
                    key_ids=key_ids,
                    tempo_values=tempo_values,
                    time_sig_ids=time_sig_ids
                )

                # Get logits for last token
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]

                # Apply musical constraints
                # For chord generation, we don't apply monophony but do apply sustain
                constrained_logits = apply_all_constraints(
                    next_token_logits,
                    state,
                    self.vocab_info,
                    self.pitch_token_to_midi,
                    self.config.key
                )

                # Sample next token with track-aware constraints
                if self.use_track_aware_sampling:
                    # Use track-aware sampling for CHORD track
                    next_token, probs = sample_next_token_track_aware(
                        constrained_logits,
                        track_type=TRACK_TYPE_CHORD,  # Chord track constraints
                        temperature=effective_temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        generated_tokens=input_ids,
                        repetition_penalty=self.config.repetition_penalty,
                        apply_constraints=True
                    )
                else:
                    # Use standard sampling without track constraints
                    next_token, probs = sample_next_token(
                        constrained_logits,
                        temperature=effective_temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        generated_tokens=input_ids,
                        repetition_penalty=self.config.repetition_penalty
                    )

                next_token_id = next_token.item()

                # Update generation state
                update_generation_state(state, next_token_id, self.vocab_info)

                # Add to sequence
                generated_tokens.append(next_token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # For two-stage generation, we could stop chord generation after
                # a certain number of bars. For now, we'll use the standard stopping.
                # In a more advanced implementation, we might stop when we detect
                # a good breakpoint for adding melody.

        return generated_tokens

    def _generate_melody(
        self,
        chord_tokens: List[int],
        temperature: Optional[float] = None,
        key_ids: Optional[torch.Tensor] = None,
        tempo_values: Optional[torch.Tensor] = None,
        time_sig_ids: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Generate melody conditioned on chord progression (Stage 2).

        Uses the chord tokens as context and generates melody on top.

        Args:
            chord_tokens: Generated chord tokens from Stage 1
            temperature: Temperature for sampling (uses config if None)
            key_ids: Optional key signature conditioning tensor, shape [1]
            tempo_values: Optional tempo conditioning tensor, shape [1]
            time_sig_ids: Optional time signature conditioning tensor, shape [1]

        Returns:
            Complete sequence with chords and melody
        """
        # Start with chord tokens as context
        generated_tokens = chord_tokens.copy()

        # Initialize generation state
        state = GenerationState()
        state.current_track = 'melody'
        state.current_key = self.config.key

        # Use provided temperature or fall back to config
        effective_temperature = temperature if temperature is not None else self.config.temperature

        # Convert to tensor
        input_ids = torch.tensor([generated_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            melody_tokens_generated = 0
            max_melody_tokens = self.config.max_length - len(chord_tokens)

            while melody_tokens_generated < max_melody_tokens:
                # Check stop conditions
                should_stop, reason = should_stop_generation(
                    generated_tokens, self.config, self.vocab_info
                )

                if should_stop:
                    logger.info(f"Melody generation stopped: {reason}")
                    break

                # Truncate input if it exceeds context length
                if input_ids.size(1) > self.model.max_len:
                    input_ids = input_ids[:, -self.model.max_len:]

                # Forward pass through model (with conditioning if available)
                logits, _ = self.model(
                    input_ids,
                    key_ids=key_ids,
                    tempo_values=tempo_values,
                    time_sig_ids=time_sig_ids
                )

                # Get logits for last token
                next_token_logits = logits[:, -1, :]

                # Apply musical constraints
                # For melody, we apply monophony constraint
                constrained_logits = apply_all_constraints(
                    next_token_logits,
                    state,
                    self.vocab_info,
                    self.pitch_token_to_midi,
                    self.config.key
                )

                # Sample next token with track-aware constraints
                if self.use_track_aware_sampling:
                    # Use track-aware sampling for MELODY track
                    next_token, probs = sample_next_token_track_aware(
                        constrained_logits,
                        track_type=TRACK_TYPE_MELODY,  # Melody track constraints
                        temperature=effective_temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        generated_tokens=input_ids,
                        repetition_penalty=self.config.repetition_penalty,
                        apply_constraints=True
                    )
                else:
                    # Use standard sampling without track constraints
                    next_token, probs = sample_next_token(
                        constrained_logits,
                        temperature=effective_temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        generated_tokens=input_ids,
                        repetition_penalty=self.config.repetition_penalty
                    )

                next_token_id = next_token.item()

                # Update generation state
                update_generation_state(state, next_token_id, self.vocab_info)

                # Add to sequence
                generated_tokens.append(next_token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                melody_tokens_generated += 1

        # Add EOS token if not present
        if EOS_TOKEN_ID not in generated_tokens:
            generated_tokens.append(EOS_TOKEN_ID)

        return generated_tokens

    def generate_batch(
        self,
        num_sequences: int,
        prompt_tokens: Optional[List[int]] = None
    ) -> List[List[int]]:
        """
        Generate multiple sequences.

        Args:
            num_sequences: Number of sequences to generate
            prompt_tokens: Optional conditioning tokens for all sequences

        Returns:
            List of generated token sequences
        """
        sequences = []

        for i in range(num_sequences):
            logger.info(f"Generating sequence {i+1}/{num_sequences}")

            # Use different seed for each sequence
            seed = self.config.seed + i if self.config.seed is not None else None

            sequence = self.generate_complete_sequence(prompt_tokens, seed)
            sequences.append(sequence)

        return sequences


__all__ = [
    'TwoStageGenerator'
]
