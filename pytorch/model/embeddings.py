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

from ..data.constants import VOCAB_SIZE, HIDDEN_DIM, CONTEXT_LENGTH, DROPOUT


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to dense vectors.

    Each of the 404 tokens in our vocabulary gets mapped to a learned
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
            vocab_size: Size of vocabulary (default: 404)
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


class MusicEmbedding(nn.Module):
    """
    Combined embedding layer for music tokens.

    Combines token embeddings with positional encodings to create the final
    input representation for the transformer.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        max_len: int = CONTEXT_LENGTH,
        dropout: float = DROPOUT
    ):
        """
        Initialize music embedding.

        Args:
            vocab_size: Size of vocabulary (default: 404)
            hidden_dim: Dimension of embeddings (default: 512)
            max_len: Maximum sequence length (default: 2048)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len, dropout)

        # Scaling factor (as in original Transformer paper)
        # Helps stabilize training by preventing embeddings from being too large
        self.scale = math.sqrt(hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional information.

        Args:
            token_ids: Token IDs, shape [batch_size, seq_len]

        Returns:
            Embeddings with positional encoding, shape [batch_size, seq_len, hidden_dim]
        """
        # Get token embeddings and scale them
        token_emb = self.token_embedding(token_ids) * self.scale

        # Add positional encoding (includes dropout)
        return self.positional_encoding(token_emb)


def get_embedding_layer(
    vocab_size: int = VOCAB_SIZE,
    hidden_dim: int = HIDDEN_DIM,
    max_len: int = CONTEXT_LENGTH,
    dropout: float = DROPOUT
) -> MusicEmbedding:
    """
    Factory function to create a music embedding layer.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Dimension of embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability

    Returns:
        MusicEmbedding layer ready to use
    """
    return MusicEmbedding(vocab_size, hidden_dim, max_len, dropout)


__all__ = [
    'TokenEmbedding',
    'PositionalEncoding',
    'MusicEmbedding',
    'get_embedding_layer'
]
