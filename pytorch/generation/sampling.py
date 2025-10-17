"""
Advanced sampling strategies for token generation.

Implements various sampling techniques including temperature scaling,
top-k filtering, nucleus (top-p) sampling, and repetition penalties.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Higher temperature makes distribution more uniform (more random).
    Lower temperature makes distribution more peaked (more conservative).

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        temperature: Temperature value (typically 0.1-2.0)

    Returns:
        Scaled logits, same shape as input
    """
    if temperature == 1.0:
        return logits

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    return logits / temperature


def apply_top_k(
    logits: torch.Tensor,
    top_k: int,
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Apply top-k filtering to logits.

    Keeps only the top K highest probability tokens, masking all others.

    Args:
        logits: Input logits, shape [batch_size, vocab_size]
        top_k: Number of top tokens to keep
        mask_value: Value to assign to masked tokens

    Returns:
        Filtered logits, same shape as input
    """
    if top_k <= 0:
        return logits

    # Ensure top_k doesn't exceed vocab size
    vocab_size = logits.size(-1)
    top_k = min(top_k, vocab_size)

    # Get top k values and indices
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)

    # Get the k-th largest value (threshold)
    threshold = top_k_values[..., -1:]

    # Mask all values below threshold
    return torch.where(
        logits < threshold,
        torch.full_like(logits, mask_value),
        logits
    )


def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
    mask_value: float = float('-inf')
) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling to logits.

    Keeps the smallest set of tokens whose cumulative probability exceeds top_p.

    Args:
        logits: Input logits, shape [batch_size, vocab_size]
        top_p: Cumulative probability threshold (0.0-1.0)
        mask_value: Value to assign to masked tokens

    Returns:
        Filtered logits, same shape as input
    """
    if top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find indices to remove (cumulative prob > top_p)
    # Shift right to keep at least one token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter back to original positions
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )

    # Mask removed indices
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = mask_value

    return filtered_logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 1.1
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage token repetition.

    Reduces probability of tokens that have already been generated.

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        generated_tokens: Previously generated tokens, shape [batch_size, seq_len]
        penalty: Penalty factor (>1.0 penalizes, <1.0 encourages repetition)

    Returns:
        Penalized logits, same shape as input
    """
    if penalty == 1.0:
        return logits

    if penalty <= 0:
        raise ValueError(f"Penalty must be positive, got {penalty}")

    batch_size, vocab_size = logits.shape

    # For each batch, get unique tokens that have been generated
    penalized_logits = logits.clone()

    for batch_idx in range(batch_size):
        # Get unique tokens in this sequence
        unique_tokens = generated_tokens[batch_idx].unique()

        # Apply penalty
        for token_id in unique_tokens:
            if 0 <= token_id < vocab_size:
                # If logit is positive, divide by penalty (reduce probability)
                # If logit is negative, multiply by penalty (reduce probability further)
                if penalized_logits[batch_idx, token_id] > 0:
                    penalized_logits[batch_idx, token_id] /= penalty
                else:
                    penalized_logits[batch_idx, token_id] *= penalty

    return penalized_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generated_tokens: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
    deterministic: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample next token using combined sampling strategies.

    Applies temperature scaling, top-k filtering, top-p filtering,
    and repetition penalty in sequence, then samples from the filtered distribution.

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        temperature: Temperature for scaling (default: 1.0)
        top_k: Top-k filtering (optional)
        top_p: Nucleus sampling threshold (optional)
        generated_tokens: Previously generated tokens for repetition penalty (optional)
        repetition_penalty: Repetition penalty factor (default: 1.0)
        deterministic: If True, use argmax instead of sampling

    Returns:
        Tuple of:
        - next_token: Sampled token IDs, shape [batch_size, 1]
        - probs: Token probabilities after all filtering, shape [batch_size, vocab_size]
    """
    # Apply repetition penalty first (if applicable)
    if generated_tokens is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

    # Apply temperature scaling
    logits = sample_with_temperature(logits, temperature)

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k)

    # Apply top-p filtering
    if top_p is not None and top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample or select deterministically
    if deterministic:
        # Take argmax (most likely token)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
    else:
        # Sample from distribution
        try:
            next_token = torch.multinomial(probs, num_samples=1)
        except RuntimeError as e:
            # Handle edge case where all probabilities are masked/zero
            logger.warning(f"Sampling failed: {e}. Using argmax fallback.")
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

    return next_token, probs


def compute_token_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of token distribution.

    Higher entropy indicates more uncertainty/randomness in the distribution.

    Args:
        probs: Token probabilities, shape [batch_size, vocab_size]

    Returns:
        Entropy values, shape [batch_size]
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)

    # Entropy = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy


def compute_perplexity_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from token probabilities.

    Lower perplexity indicates higher confidence in predictions.

    Args:
        probs: Token probabilities, shape [batch_size, vocab_size]

    Returns:
        Perplexity values, shape [batch_size]
    """
    entropy = compute_token_entropy(probs)
    perplexity = torch.exp(entropy)

    return perplexity


def get_top_k_tokens(
    logits: torch.Tensor,
    k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top K most likely tokens and their probabilities.

    Useful for debugging and analysis.

    Args:
        logits: Model output logits, shape [batch_size, vocab_size]
        k: Number of top tokens to return

    Returns:
        Tuple of:
        - top_k_tokens: Token IDs, shape [batch_size, k]
        - top_k_probs: Token probabilities, shape [batch_size, k]
    """
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_tokens = torch.topk(probs, k, dim=-1)

    return top_k_tokens, top_k_probs


__all__ = [
    'sample_with_temperature',
    'apply_top_k',
    'apply_top_p',
    'apply_repetition_penalty',
    'sample_next_token',
    'compute_token_entropy',
    'compute_perplexity_from_probs',
    'get_top_k_tokens'
]
