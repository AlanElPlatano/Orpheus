"""
Loss functions for music generation training.

Implements weighted cross-entropy loss with support for constraint-based
loss weighting to encourage the model to follow musical rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from ..data.constants import (
    PAD_TOKEN_ID,
    TRACK_TYPE_MELODY,
    TRACK_TYPE_CHORD,
    TOKEN_RANGES,
    is_pitch_token,
    is_duration_token
)


class WeightedMusicLoss(nn.Module):
    """
    Weighted cross-entropy loss for music generation.

    Supports different loss weights for different types of tokens to
    encourage constraint adherence during training.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize weighted music loss.

        Args:
            ignore_index: Token ID to ignore in loss calculation (default: -100)
            reduction: Reduction method ('mean', 'sum', or 'none')
            label_smoothing: Label smoothing factor (default: 0.0)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: Model output logits, shape [batch_size, seq_len, vocab_size]
            targets: Target token IDs, shape [batch_size, seq_len]
            sample_weights: Per-sample weights, shape [batch_size] (optional)

        Returns:
            Scalar loss value
        """
        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction='none',  # We'll apply reduction manually
            label_smoothing=self.label_smoothing
        )

        # Reshape back to [batch_size, seq_len]
        loss = loss.view(batch_size, seq_len)

        # Apply sample weights if provided
        if sample_weights is not None:
            # sample_weights shape: [batch_size]
            # Broadcast to [batch_size, seq_len]
            sample_weights = sample_weights.unsqueeze(1)
            loss = loss * sample_weights

        # Apply reduction
        if self.reduction == 'mean':
            # Only average over non-ignored positions
            mask = (targets != self.ignore_index).float()
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        elif self.reduction == 'sum':
            loss = loss.sum()
        # else: reduction == 'none', return unreduced loss

        return loss


class ConstraintAwareLoss(nn.Module):
    """
    Loss function with constraint-aware weighting.

    Applies different loss weights based on musical constraints:
    - Higher weight for constraint violations
    - Normal weight for regular tokens

    NOTE: This is a basic implementation. Full constraint detection will
    be added when we have more training experience.
    """

    def __init__(
        self,
        base_loss_fn: Optional[nn.Module] = None,
        constraint_weights: Optional[Dict[str, float]] = None,
        ignore_index: int = -100
    ):
        """
        Initialize constraint-aware loss.

        Args:
            base_loss_fn: Base loss function (default: WeightedMusicLoss)
            constraint_weights: Dictionary of constraint weights
            ignore_index: Token ID to ignore in loss calculation
        """
        super().__init__()

        self.base_loss_fn = base_loss_fn or WeightedMusicLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

        # Default constraint weights
        self.constraint_weights = constraint_weights or {
            'monophony_violation': 10.0,
            'chord_duration_violation': 5.0,
            'diatonic_violation': 3.0,
            'normal_token': 1.0
        }

    def detect_constraint_violations(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect constraint violations in the predictions.

        This is a placeholder for Phase 3. We'll implement proper constraint
        detection when we have training data and understand violation patterns.

        Args:
            predictions: Predicted token IDs
            targets: Target token IDs

        Returns:
            Weight tensor, shape [batch_size, seq_len]
        """
        # For now, return uniform weights
        # In Phase 3, we'll add logic to detect:
        # - Monophony violations (multiple simultaneous melody notes)
        # - Chord duration violations (short chord durations)
        # - Diatonic violations (notes outside the key)

        return torch.ones_like(targets, dtype=torch.float)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute constraint-aware loss.

        Args:
            logits: Model output logits
            targets: Target token IDs
            sample_weights: Per-sample weights (optional)

        Returns:
            Scalar loss value
        """
        # Get predictions (for constraint detection)
        predictions = logits.argmax(dim=-1)

        # Detect constraint violations (future implementation)
        constraint_weights = self.detect_constraint_violations(predictions, targets)

        # Combine with sample weights if provided
        if sample_weights is not None:
            # Broadcast sample_weights to match constraint_weights shape
            sample_weights = sample_weights.unsqueeze(1)
            combined_weights = constraint_weights * sample_weights
        else:
            combined_weights = constraint_weights

        # For now, just use the base loss without constraint weighting
        # We'll add proper constraint-based weighting in Phase 3 after
        # observing actual training behavior
        return self.base_loss_fn(logits, targets, sample_weights)


class TrackAwareLoss(nn.Module):
    """
    Track-aware loss function that applies different constraints for melody vs chord tracks.

    This loss function:
    1. Applies higher weights to constraint violations specific to each track type
    2. For melody tracks: penalizes polyphony violations
    3. For chord tracks: penalizes short durations and rhythmic variation
    4. Uses track_ids to determine which constraints to apply
    """

    def __init__(
        self,
        base_loss_fn: Optional[nn.Module] = None,
        melody_violation_weight: float = 10.0,
        chord_violation_weight: float = 5.0,
        ignore_index: int = -100
    ):
        """
        Initialize track-aware loss.

        Args:
            base_loss_fn: Base loss function (default: WeightedMusicLoss)
            melody_violation_weight: Weight for melody constraint violations
            chord_violation_weight: Weight for chord constraint violations
            ignore_index: Token ID to ignore in loss calculation
        """
        super().__init__()

        self.base_loss_fn = base_loss_fn or WeightedMusicLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.melody_violation_weight = melody_violation_weight
        self.chord_violation_weight = chord_violation_weight

    def detect_track_violations(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        track_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect track-specific constraint violations.

        For melody tracks (track_id=0):
        - Penalize very short durations (indicates potential polyphony)

        For chord tracks (track_id=1):
        - Penalize very short durations (chords should sustain)

        Args:
            predictions: Predicted token IDs, shape [batch_size, seq_len]
            targets: Target token IDs, shape [batch_size, seq_len]
            track_ids: Track type IDs, shape [batch_size, seq_len]

        Returns:
            Weight tensor, shape [batch_size, seq_len]
        """
        batch_size, seq_len = targets.shape
        weights = torch.ones_like(targets, dtype=torch.float)

        # Duration token range
        duration_start, duration_end = TOKEN_RANGES['duration']

        # Process each position
        for b in range(batch_size):
            for t in range(seq_len):
                target_token = targets[b, t].item()
                track_id = track_ids[b, t].item()

                # Skip padding and ignored tokens
                if target_token == self.ignore_index:
                    continue

                # Check if this is a duration token
                if duration_start <= target_token <= duration_end:
                    # Duration tokens are Duration_0.1.4 to Duration_4.0.4
                    # Token IDs: 62-77
                    # Shortest durations are at the beginning of the range
                    duration_idx = target_token - duration_start

                    # If it's a very short duration (first 3 tokens)
                    # These represent durations < 0.5 beats
                    if duration_idx < 3:
                        if track_id == TRACK_TYPE_MELODY:
                            # Short durations in melody might indicate polyphony
                            weights[b, t] = self.melody_violation_weight
                        elif track_id == TRACK_TYPE_CHORD:
                            # Short durations in chords violate our constraint
                            weights[b, t] = self.chord_violation_weight

        return weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        track_ids: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute track-aware loss.

        Args:
            logits: Model output logits, shape [batch_size, seq_len, vocab_size]
            targets: Target token IDs, shape [batch_size, seq_len]
            track_ids: Track type IDs, shape [batch_size, seq_len]
            sample_weights: Per-sample weights (optional)

        Returns:
            Scalar loss value
        """
        # Get predictions (for constraint detection)
        predictions = logits.argmax(dim=-1)

        # Detect track-specific violations
        violation_weights = self.detect_track_violations(predictions, targets, track_ids)

        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Compute per-token cross-entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        # Reshape back to [batch_size, seq_len]
        loss = loss.view(batch_size, seq_len)

        # Apply violation weights
        loss = loss * violation_weights

        # Apply sample weights if provided
        if sample_weights is not None:
            sample_weights = sample_weights.unsqueeze(1)
            loss = loss * sample_weights

        # Apply reduction (mean over non-ignored positions)
        mask = (targets != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

        return loss


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from loss.

    Perplexity is a measure of how well the model predicts the target.
    Lower perplexity = better predictions.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    return torch.exp(loss)


def create_loss_function(
    loss_type: str = 'weighted',
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    constraint_weights: Optional[Dict[str, float]] = None,
    melody_violation_weight: float = 10.0,
    chord_violation_weight: float = 5.0
) -> nn.Module:
    """
    Factory function to create a loss function.

    Args:
        loss_type: Type of loss ('weighted', 'constraint_aware', or 'track_aware')
        ignore_index: Token ID to ignore in loss calculation
        label_smoothing: Label smoothing factor
        constraint_weights: Dictionary of constraint weights
        melody_violation_weight: Weight for melody constraint violations
        chord_violation_weight: Weight for chord constraint violations

    Returns:
        Loss function module
    """
    if loss_type == 'track_aware':
        return TrackAwareLoss(
            melody_violation_weight=melody_violation_weight,
            chord_violation_weight=chord_violation_weight,
            ignore_index=ignore_index
        )
    elif loss_type == 'constraint_aware':
        base_loss = WeightedMusicLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        return ConstraintAwareLoss(
            base_loss_fn=base_loss,
            constraint_weights=constraint_weights,
            ignore_index=ignore_index
        )
    else:  # 'weighted' or default
        return WeightedMusicLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )


__all__ = [
    'WeightedMusicLoss',
    'ConstraintAwareLoss',
    'TrackAwareLoss',
    'compute_perplexity',
    'create_loss_function'
]
