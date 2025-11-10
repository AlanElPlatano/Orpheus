"""
Token and positional embeddings for the music generation transformer.

This module provides embedding layers that convert discrete token IDs into
continuous vectors that the transformer can process.

Neural networks can't work directly with discrete tokens (numbers 0-403).
They need continuous vectors that can be mathematically manipulated so that it can
understand these relationships
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from ..data.constants import (
    VOCAB_SIZE,
    HIDDEN_DIM,
    CONTEXT_LENGTH,
    DROPOUT,
    NUM_TRACK_TYPES,
    NUM_KEY_CONDITIONS,
    NUM_TIME_SIG_CONDITIONS,
    TEMPO_EMBEDDING_DIM,
    CONDITION_EMBED_DIM,
    TEMPO_NONE_VALUE,
    MIN_TEMPO_CONDITION,
    MAX_TEMPO_CONDITION
)


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to dense vectors.

    Each of the 531 tokens in our vocabulary gets mapped to a learned
    HIDDEN_DIM-dimensional vector.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM
    ):
        """
        Initialize token embeddings.

        Args:
            vocab_size: Size of vocabulary (default: 531)
            hidden_dim: Dimension of embedding vectors (default: 512)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Learnable embedding matrix: [vocab_size, hidden_dim]
        # When a row is looked up, returns a 512 dimensional vector
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # During training the vectors adjust so that similar concepts become closer
        # And different concepts become further apart

        # Initialize embeddings with small random values
        # Using Xavier/Glorot initialization for better training stability
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            token_ids: Token IDs, shape [batch_size, seq_len]

        Returns:
            Embeddings, shape [batch_size, seq_len, hidden_dim]
        """
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions (as in original Transformer paper).

    Since transformers have no inherent notion of sequence order, we add positional
    information so the model knows which token comes where.

    Uses the sinusoidal encoding from "Attention is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        max_len: int = CONTEXT_LENGTH,
        dropout: float = DROPOUT
    ):
        """
        Initialize positional encoding.

        Args:
            hidden_dim: Dimension of embeddings (default: 512)
            max_len: Maximum sequence length (default: 2048)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix: [max_len, hidden_dim]
        pe = torch.zeros(max_len, hidden_dim)

        # Position indices: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division term for the sinusoidal functions
        # Creates: [hidden_dim // 2] with values 10000^(2i/hidden_dim)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() *
            (-math.log(10000.0) / hidden_dim)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_len, hidden_dim]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state_dict)
        # This means it will be moved to GPU with the model but won't be trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings, shape [batch_size, seq_len, hidden_dim]

        Returns:
            Embeddings with positional encoding added, same shape as input
        """
        # Add positional encoding (broadcasting handles batch dimension)
        # x.size(1) is seq_len
        x = x + self.pe[:, :x.size(1), :]

        # Apply dropout for regularization
        return self.dropout(x)


class TrackEmbedding(nn.Module):
    """
    Track type embedding layer that converts track type IDs to dense vectors.

    Similar to positional embeddings, but for track types (melody vs chord).
    This helps the model learn different generation patterns for different tracks.
    """

    def __init__(
        self,
        num_track_types: int = NUM_TRACK_TYPES,
        hidden_dim: int = HIDDEN_DIM
    ):
        """
        Initialize track embeddings.

        Args:
            num_track_types: Number of track types (default: 2 for melody/chord)
            hidden_dim: Dimension of embedding vectors (default: 512)
        """
        super().__init__()
        self.num_track_types = num_track_types
        self.hidden_dim = hidden_dim

        # Learnable embedding matrix: [num_track_types, hidden_dim]
        self.embedding = nn.Embedding(num_track_types, hidden_dim)

        # Initialize embeddings with small random values
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, track_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert track type IDs to embeddings.

        Args:
            track_ids: Track type IDs, shape [batch_size, seq_len]
                      Each element is 0 (MELODY) or 1 (CHORD)

        Returns:
            Embeddings, shape [batch_size, seq_len, hidden_dim]
        """
        return self.embedding(track_ids)


class ConditionEmbedding(nn.Module):
    """
    Conditional generation embedding layer that encodes key, tempo, and time signature.

    This allows the model to generate music conditioned on specific musical characteristics:
    - Key signature (e.g., C major, F# minor)
    - Tempo (BPM, e.g., 125)
    - Time signature (e.g., 4/4, 6/8)

    Each condition can be independently specified or set to "none" for unconditioned generation.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        condition_embed_dim: int = CONDITION_EMBED_DIM,
        num_key_conditions: int = NUM_KEY_CONDITIONS,
        num_time_sig_conditions: int = NUM_TIME_SIG_CONDITIONS,
        tempo_embed_dim: int = TEMPO_EMBEDDING_DIM
    ):
        """
        Initialize condition embeddings.

        Args:
            hidden_dim: Dimension to project final conditioning to (default: 512)
            condition_embed_dim: Dimension for individual condition embeddings (default: 64)
            num_key_conditions: Number of key options including "none" (default: 26)
            num_time_sig_conditions: Number of time sig options including "none" (default: 10)
            tempo_embed_dim: Dimension for tempo MLP hidden layer (default: 32)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.condition_embed_dim = condition_embed_dim

        # Key signature embedding: discrete (C, Dm, F#, etc.)
        # Maps key ID -> embedding vector
        self.key_embedding = nn.Embedding(num_key_conditions, condition_embed_dim)

        # Time signature embedding: discrete (4/4, 6/8, etc.)
        # Maps time sig ID -> embedding vector
        self.time_sig_embedding = nn.Embedding(num_time_sig_conditions, condition_embed_dim)

        # Tempo embedding: continuous value -> MLP -> embedding
        # Handles continuous BPM values (90-140)
        self.tempo_mlp = nn.Sequential(
            nn.Linear(1, tempo_embed_dim),  # 1D input (BPM value)
            nn.ReLU(),
            nn.Linear(tempo_embed_dim, condition_embed_dim)  # Project to condition_embed_dim
        )

        # Project combined conditions to model hidden dimension
        # Concatenates all three condition embeddings (3 * condition_embed_dim) -> hidden_dim
        self.projection = nn.Linear(3 * condition_embed_dim, hidden_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.key_embedding.weight)
        nn.init.xavier_uniform_(self.time_sig_embedding.weight)

        # Initialize MLP weights
        for module in self.tempo_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize projection
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        key_ids: torch.Tensor,
        tempo_values: torch.Tensor,
        time_sig_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conditioning embeddings from key, tempo, and time signature.

        Args:
            key_ids: Key signature IDs, shape [batch_size]
                    0 = "none" (unconditioned), 1-25 = specific keys
            tempo_values: Tempo values in BPM, shape [batch_size]
                         0.0 = "none" (unconditioned), 90-140 = specific tempo
            time_sig_ids: Time signature IDs, shape [batch_size]
                         0 = "none" (unconditioned), 1-9 = specific time sigs

        Returns:
            Conditioning embedding, shape [batch_size, hidden_dim]
        """
        # Get key embedding: [batch_size] -> [batch_size, condition_embed_dim]
        key_emb = self.key_embedding(key_ids)

        # Get time signature embedding: [batch_size] -> [batch_size, condition_embed_dim]
        time_sig_emb = self.time_sig_embedding(time_sig_ids)

        # Normalize tempo to [0, 1] range for better MLP performance
        # tempo_values are either 0.0 (none) or 90-140 (actual tempo)
        # We normalize non-zero values to [0, 1] range
        tempo_normalized = torch.where(
            tempo_values > 0,
            (tempo_values - MIN_TEMPO_CONDITION) / (MAX_TEMPO_CONDITION - MIN_TEMPO_CONDITION),
            torch.zeros_like(tempo_values)
        )

        # Get tempo embedding via MLP: [batch_size] -> [batch_size, 1] -> [batch_size, condition_embed_dim]
        tempo_emb = self.tempo_mlp(tempo_normalized.unsqueeze(-1))

        # Concatenate all condition embeddings: [batch_size, 3 * condition_embed_dim]
        combined = torch.cat([key_emb, tempo_emb, time_sig_emb], dim=-1)

        # Project to hidden dimension: [batch_size, hidden_dim]
        conditioning = self.projection(combined)

        return conditioning


class MusicEmbedding(nn.Module):
    """
    Combined embedding layer for music tokens.

    Combines token embeddings with positional encodings and optionally
    track type embeddings to create the final input representation for the transformer.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        max_len: int = CONTEXT_LENGTH,
        dropout: float = DROPOUT,
        use_track_embeddings: bool = True,
        num_track_types: int = NUM_TRACK_TYPES,
        use_conditioning: bool = False
    ):
        """
        Initialize music embedding.

        Args:
            vocab_size: Size of vocabulary (default: 531)
            hidden_dim: Dimension of embeddings (default: 512)
            max_len: Maximum sequence length (default: 2048)
            dropout: Dropout probability (default: 0.1)
            use_track_embeddings: Whether to include track type embeddings (default: True)
            num_track_types: Number of track types (default: 2)
            use_conditioning: Whether to include conditional generation embeddings (default: False)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.use_track_embeddings = use_track_embeddings
        self.use_conditioning = use_conditioning

        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len, dropout)

        # Track type embeddings (optional)
        if use_track_embeddings:
            self.track_embedding = TrackEmbedding(num_track_types, hidden_dim)
        else:
            self.track_embedding = None

        # Conditional generation embeddings (optional)
        if use_conditioning:
            self.condition_embedding = ConditionEmbedding(hidden_dim)
        else:
            self.condition_embedding = None

        # Scaling factor (as in original Transformer paper)
        # Helps stabilize training by preventing embeddings from being too large
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        track_ids: Optional[torch.Tensor] = None,
        key_ids: Optional[torch.Tensor] = None,
        tempo_values: Optional[torch.Tensor] = None,
        time_sig_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional, track, and conditioning information.

        Args:
            token_ids: Token IDs, shape [batch_size, seq_len]
            track_ids: Track type IDs, shape [batch_size, seq_len] (optional)
                      Each element is 0 (MELODY) or 1 (CHORD)
            key_ids: Key signature condition IDs, shape [batch_size] (optional)
                    0 = "none", 1-25 = specific keys
            tempo_values: Tempo condition values in BPM, shape [batch_size] (optional)
                         0.0 = "none", 90-140 = specific tempo
            time_sig_ids: Time signature condition IDs, shape [batch_size] (optional)
                         0 = "none", 1-9 = specific time signatures

        Returns:
            Embeddings with all encoding applied, shape [batch_size, seq_len, hidden_dim]
        """
        # Get token embeddings and scale them
        token_emb = self.token_embedding(token_ids) * self.scale

        # Add track embeddings if provided and enabled
        if self.use_track_embeddings and track_ids is not None:
            track_emb = self.track_embedding(track_ids)
            # Add track embeddings to token embeddings (similar to segment embeddings in BERT)
            token_emb = token_emb + track_emb

        # Add conditioning if provided and enabled
        if self.use_conditioning and key_ids is not None and tempo_values is not None and time_sig_ids is not None:
            # Get conditioning embedding: [batch_size, hidden_dim]
            condition_emb = self.condition_embedding(key_ids, tempo_values, time_sig_ids)

            # Broadcast conditioning to all sequence positions: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
            # Then broadcast to [batch_size, seq_len, hidden_dim] via addition
            condition_emb = condition_emb.unsqueeze(1)

            # Add conditioning to token embeddings
            token_emb = token_emb + condition_emb

        # Add positional encoding (includes dropout)
        return self.positional_encoding(token_emb)


def get_embedding_layer(
    vocab_size: int = VOCAB_SIZE,
    hidden_dim: int = HIDDEN_DIM,
    max_len: int = CONTEXT_LENGTH,
    dropout: float = DROPOUT,
    use_track_embeddings: bool = True,
    num_track_types: int = NUM_TRACK_TYPES,
    use_conditioning: bool = False
) -> MusicEmbedding:
    """
    Factory function to create a music embedding layer.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Dimension of embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
        use_track_embeddings: Whether to include track type embeddings
        num_track_types: Number of track types
        use_conditioning: Whether to include conditional generation embeddings

    Returns:
        MusicEmbedding layer ready to use
    """
    return MusicEmbedding(
        vocab_size,
        hidden_dim,
        max_len,
        dropout,
        use_track_embeddings,
        num_track_types,
        use_conditioning
    )


__all__ = [
    'TokenEmbedding',
    'PositionalEncoding',
    'TrackEmbedding',
    'ConditionEmbedding',
    'MusicEmbedding',
    'get_embedding_layer'
]
