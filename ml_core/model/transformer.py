"""
Transformer model for music generation.

This module implements a decoder-only transformer architecture (like GPT but)
for autoregressive music token generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

from .embeddings import MusicEmbedding
from ..data.constants import (
    VOCAB_SIZE,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    FF_DIM,
    CONTEXT_LENGTH,
    DROPOUT,
    NUM_TRACK_TYPES
)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Splits the hidden dimension into multiple heads, allowing the model to
    attend to different aspects of the input simultaneously.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
        use_flash_attention: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            hidden_dim: Dimension of hidden states (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_flash_attention: Whether to use PyTorch's scaled_dot_product_attention
                                (FlashAttention when available, more memory efficient)
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_flash_attention = use_flash_attention

        # Linear projections for Q, K, V: Query, Key, Value
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        # Scaling factor for attention scores (only used in manual attention)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Apply multi-head attention.

        Args:
            x: Input tensor, shape [batch_size, seq_len, hidden_dim]
            attention_mask: Mask for padding tokens, shape [batch_size, seq_len]
                           1 for real tokens, 0 for padding
            causal_mask: Whether to apply causal masking (default: True)

        Returns:
            Output tensor, shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, hidden_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into multiple heads
        # Reshape: [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to: [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's optimized scaled_dot_product_attention if available (PyTorch 2.0+)
        # This uses FlashAttention when possible, which is much more memory efficient
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Prepare attention mask for scaled_dot_product_attention
            # It expects: [batch_size, num_heads, seq_len, seq_len] or broadcastable
            attn_mask = None
            if attention_mask is not None:
                # Convert padding mask to attention mask
                # attention_mask is [batch_size, seq_len], we need [batch_size, 1, 1, seq_len]
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # Convert to float and set padding positions to -inf
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf'))
                attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

            # scaled_dot_product_attention handles causal masking internally
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal_mask and attention_mask is None  # Only use is_causal if no custom mask
            )
        else:
            # Fall back to manual attention computation
            # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask (prevent attending to future tokens)
            if causal_mask:
                # Create lower triangular matrix
                causal_mask_matrix = torch.tril(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
                )
                attn_scores = attn_scores.masked_fill(~causal_mask_matrix, float('-inf'))

            # Apply padding mask if provided
            if attention_mask is not None:
                # Reshape attention_mask: [batch_size, 1, 1, seq_len]
                attention_mask_reshaped = attention_mask.unsqueeze(1).unsqueeze(2)
                # Mask out padding positions
                attn_scores = attn_scores.masked_fill(attention_mask_reshaped == 0, float('-inf'))

            # Compute attention weights (softmax over key dimension)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Apply dropout to attention weights
            attn_weights = self.attn_dropout(attn_weights)

            # Apply attention to values
            # [batch_size, num_heads, seq_len, head_dim]
            attn_output = torch.matmul(attn_weights, v)

        # Transpose back: [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)

        # Concatenate heads: [batch_size, seq_len, hidden_dim]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Apply output projection
        output = self.out_proj(attn_output)

        # Apply residual dropout
        output = self.resid_dropout(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two-layer MLP with GELU activation applied independently to each position.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT
    ):
        """
        Initialize feed-forward network.

        Args:
            hidden_dim: Dimension of hidden states
            ff_dim: Dimension of feed-forward layer (typically 4x hidden_dim)
            dropout: Dropout probability
        """
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.

        Args:
            x: Input tensor, shape [batch_size, seq_len, hidden_dim]

        Returns:
            Output tensor, same shape as input
        """
        x = self.fc1(x)
        x = F.gelu(x)  # GELU (Gaussian Error Linear Unit) activation (smoother than ReLU)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Consists of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        use_flash_attention: bool = True
    ):
        """
        Initialize transformer block.

        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward layer
            dropout: Dropout probability
            use_flash_attention: Whether to use FlashAttention-compatible attention
        """
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout, use_flash_attention)

        # Feed-forward network
        self.feed_forward = FeedForward(hidden_dim, ff_dim, dropout)

        # Layer normalization (applied before attention and FF, "pre-norm" style)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply transformer block.

        Args:
            x: Input tensor, shape [batch_size, seq_len, hidden_dim]
            attention_mask: Mask for padding tokens

        Returns:
            Output tensor, same shape as input
        """
        # Self-attention with residual connection (pre-norm)
        x = x + self.attention(self.ln1(x), attention_mask)

        # Feed-forward with residual connection (pre-norm)
        x = x + self.feed_forward(self.ln2(x))

        return x


class MusicTransformer(nn.Module):
    """
    Complete transformer model for music generation.

    Decoder-only transformer (like GPT) for autoregressive token generation.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        ff_dim: int = FF_DIM,
        max_len: int = CONTEXT_LENGTH,
        dropout: float = DROPOUT,
        use_track_embeddings: bool = True,
        num_track_types: int = NUM_TRACK_TYPES,
        use_conditioning: bool = False,
        use_gradient_checkpointing: bool = False,
        use_flash_attention: bool = True
    ):
        """
        Initialize music transformer.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Dimension of hidden states
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward layer
            max_len: Maximum sequence length
            dropout: Dropout probability
            use_track_embeddings: Whether to use track type embeddings
            num_track_types: Number of track types
            use_conditioning: Whether to use conditional generation embeddings
            use_gradient_checkpointing: Whether to use gradient checkpointing (saves memory)
            use_flash_attention: Whether to use FlashAttention-compatible attention (saves memory)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.dropout = dropout
        self.use_track_embeddings = use_track_embeddings
        self.use_conditioning = use_conditioning
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Embedding layer (token + positional + track + conditioning)
        self.embedding = MusicEmbedding(
            vocab_size,
            hidden_dim,
            max_len,
            dropout,
            use_track_embeddings,
            num_track_types,
            use_conditioning
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout, use_flash_attention)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(hidden_dim)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tie output projection weights with input embeddings (weight sharing)
        # This is a common technique that reduces parameters and often improves performance
        # This also works because the embedding matrix and
        # the output projection matrix share the same weights
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        track_ids: Optional[torch.Tensor] = None,
        key_ids: Optional[torch.Tensor] = None,
        tempo_values: Optional[torch.Tensor] = None,
        time_sig_ids: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape [batch_size, seq_len]
            attention_mask: Mask for padding, shape [batch_size, seq_len]
            track_ids: Track type IDs, shape [batch_size, seq_len] (optional)
            key_ids: Key signature condition IDs, shape [batch_size] (optional)
            tempo_values: Tempo condition values in BPM, shape [batch_size] (optional)
            time_sig_ids: Time signature condition IDs, shape [batch_size] (optional)
            return_hidden_states: Whether to return final hidden states

        Returns:
            Tuple of:
            - logits: Output logits, shape [batch_size, seq_len, vocab_size]
            - hidden_states (optional): Final hidden states before LM head
        """
        # Get embeddings (with track and conditioning information if provided)
        x = self.embedding(
            input_ids,
            track_ids,
            key_ids,
            tempo_values,
            time_sig_ids
        )  # [batch_size, seq_len, hidden_dim]

        # Apply transformer blocks with optional gradient checkpointing
        # Gradient checkpointing trades compute for memory by not storing
        # intermediate activations during forward pass, recomputing them during backward
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing (memory efficient but slower)
            for block in self.blocks:
                # checkpoint requires a function that takes tensors and returns tensors
                # We create a wrapper function that calls the block
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                x = checkpoint(create_custom_forward(block), x, attention_mask, use_reentrant=False)
        else:
            # Normal forward pass (faster but uses more memory)
            for block in self.blocks:
                x = block(x, attention_mask)

        # Apply final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]

        if return_hidden_states:
            return logits, x
        else:
            return logits, None

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.embedding.token_embedding.embedding.weight.numel()

        return n_params

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        This is a simple generation method. For full generation with constraints,
        use the generation/ module (Phase 4).

        Args:
            input_ids: Initial token IDs, shape [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top K logits (optional)
            top_p: Nucleus sampling threshold (optional)
            eos_token_id: End-of-sequence token ID (optional)

        Returns:
            Generated token IDs, shape [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds max_len
            input_ids_crop = input_ids if input_ids.size(1) <= self.max_len \
                else input_ids[:, -self.max_len:]

            # Forward pass
            logits, _ = self.forward(input_ids_crop)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original positions
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


def create_model(
    vocab_size: int = VOCAB_SIZE,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    num_heads: int = NUM_HEADS,
    ff_dim: int = FF_DIM,
    max_len: int = CONTEXT_LENGTH,
    dropout: float = DROPOUT,
    use_track_embeddings: bool = True,
    num_track_types: int = NUM_TRACK_TYPES,
    use_conditioning: bool = False,
    use_gradient_checkpointing: bool = False,
    use_flash_attention: bool = True
) -> MusicTransformer:
    """
    Factory function to create a MusicTransformer model.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Dimension of hidden states
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        max_len: Maximum sequence length
        dropout: Dropout probability
        use_track_embeddings: Whether to use track type embeddings
        num_track_types: Number of track types
        use_conditioning: Whether to use conditional generation embeddings
        use_gradient_checkpointing: Whether to use gradient checkpointing (saves memory)
        use_flash_attention: Whether to use FlashAttention-compatible attention (saves memory)

    Returns:
        MusicTransformer model
    """
    return MusicTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_len=max_len,
        dropout=dropout,
        use_track_embeddings=use_track_embeddings,
        num_track_types=num_track_types,
        use_conditioning=use_conditioning,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_flash_attention=use_flash_attention
    )


__all__ = [
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    'MusicTransformer',
    'create_model'
]