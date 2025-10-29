"""
VRAM estimation utility for transformer models.

Provides accurate memory usage estimates for training configurations
to help prevent OOM errors.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import torch


@dataclass
class VRAMEstimate:
    """Container for VRAM usage breakdown."""
    model_mb: float
    optimizer_mb: float
    activations_mb: float
    gradients_mb: float
    overhead_mb: float
    total_mb: float
    
    @property
    def total_gb(self) -> float:
        """Total VRAM in GB."""
        return self.total_mb / 1024
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get memory breakdown as dictionary."""
        return {
            'Model Parameters': self.model_mb,
            'Optimizer States': self.optimizer_mb,
            'Activations': self.activations_mb,
            'Gradients': self.gradients_mb,
            'PyTorch Overhead': self.overhead_mb,
            'Total': self.total_mb,
        }
    
    def get_percentage_breakdown(self) -> Dict[str, float]:
        """Get memory breakdown as percentages."""
        if self.total_mb == 0:
            return {k: 0.0 for k in self.get_breakdown().keys()}
        
        breakdown = self.get_breakdown()
        return {
            k: (v / self.total_mb * 100) 
            for k, v in breakdown.items()
        }


def estimate_vram_usage(
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    context_length: int,
    batch_size: int,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    use_track_embeddings: bool = False,
    num_track_types: int = 2,
) -> VRAMEstimate:
    """
    Estimate VRAM usage for transformer training.
    
    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        context_length: Maximum sequence length
        batch_size: Batch size
        mixed_precision: Whether using FP16/mixed precision
        gradient_accumulation_steps: Gradient accumulation steps
        use_track_embeddings: Whether using track embeddings
        num_track_types: Number of track types (if using track embeddings)
    
    Returns:
        VRAMEstimate with detailed breakdown
    """
    # Bytes per parameter
    bytes_per_param = 2 if mixed_precision else 4
    
    # Effective batch size (before gradient accumulation)
    # Note: gradient accumulation doesn't affect memory much since
    # we only store gradients once and accumulate them
    effective_batch = batch_size
    
    # ========================================================================
    # 1. MODEL PARAMETERS MEMORY
    # ========================================================================
    
    # Token embeddings: vocab_size × hidden_dim
    token_embedding_params = vocab_size * hidden_dim
    
    # Position embeddings: context_length × hidden_dim
    position_embedding_params = context_length * hidden_dim
    
    # Track embeddings (optional): num_track_types × hidden_dim
    track_embedding_params = 0
    if use_track_embeddings:
        track_embedding_params = num_track_types * hidden_dim
    
    # Transformer layers
    layer_params = 0
    for _ in range(num_layers):
        # Self-attention: Q, K, V, output projections
        # Each is hidden_dim × hidden_dim
        attention_params = 4 * (hidden_dim * hidden_dim)
        
        # Attention biases (if used)
        attention_bias_params = 4 * hidden_dim
        
        # Layer norms (2 per layer): each has 2 × hidden_dim params (scale + bias)
        layer_norm_params = 2 * (2 * hidden_dim)
        
        # Feed-forward network
        # First layer: hidden_dim × ff_dim
        # Second layer: ff_dim × hidden_dim
        ffn_params = (hidden_dim * ff_dim) + (ff_dim * hidden_dim)
        
        # FFN biases
        ffn_bias_params = ff_dim + hidden_dim
        
        layer_params += (
            attention_params + 
            attention_bias_params + 
            layer_norm_params + 
            ffn_params + 
            ffn_bias_params
        )
    
    # Output layer: hidden_dim × vocab_size (often shares weights with input embedding)
    output_layer_params = hidden_dim * vocab_size
    
    # Final layer norm
    final_layer_norm_params = 2 * hidden_dim
    
    # Total parameters
    total_params = (
        token_embedding_params +
        position_embedding_params +
        track_embedding_params +
        layer_params +
        output_layer_params +
        final_layer_norm_params
    )
    
    # Model memory in MB
    model_mb = (total_params * bytes_per_param) / (1024 ** 2)
    
    # ========================================================================
    # 2. OPTIMIZER STATES MEMORY
    # ========================================================================
    
    # Adam/AdamW stores 2 states per parameter (momentum + variance)
    # These are always stored in FP32, even with mixed precision
    optimizer_mb = (total_params * 4 * 2) / (1024 ** 2)
    
    # ========================================================================
    # 3. GRADIENTS MEMORY
    # ========================================================================
    
    # Gradients are same size as model parameters
    # Even with gradient accumulation, we only store one copy of gradients
    gradients_mb = (total_params * bytes_per_param) / (1024 ** 2)
    
    # ========================================================================
    # 4. ACTIVATIONS MEMORY (THE BIG ONE)
    # ========================================================================
    
    b = effective_batch
    s = context_length
    h = hidden_dim
    l = num_layers
    f = ff_dim
    n = num_heads
    v = vocab_size
    
    # Input embeddings activation
    input_embedding_mb = (b * s * h * bytes_per_param) / (1024 ** 2)
    
    # Per-layer activations
    per_layer_mb = 0
    
    # Attention scores: batch × num_heads × seq_len × seq_len
    # This is the O(n²) term that dominates for long sequences
    attention_scores_mb = (b * n * s * s * bytes_per_param) / (1024 ** 2)
    
    # Attention output: batch × seq_len × hidden_dim
    attention_output_mb = (b * s * h * bytes_per_param) / (1024 ** 2)
    
    # Q, K, V projections: 3 × (batch × seq_len × hidden_dim)
    qkv_mb = 3 * (b * s * h * bytes_per_param) / (1024 ** 2)
    
    # Layer norm outputs: 2 per layer
    layer_norm_mb = 2 * (b * s * h * bytes_per_param) / (1024 ** 2)
    
    # FFN intermediate: batch × seq_len × ff_dim
    ffn_intermediate_mb = (b * s * f * bytes_per_param) / (1024 ** 2)
    
    # FFN output: batch × seq_len × hidden_dim
    ffn_output_mb = (b * s * h * bytes_per_param) / (1024 ** 2)
    
    per_layer_mb = (
        attention_scores_mb +
        attention_output_mb +
        qkv_mb +
        layer_norm_mb +
        ffn_intermediate_mb +
        ffn_output_mb
    )
    
    # Total for all layers
    layers_activations_mb = per_layer_mb * l
    
    # Output logits: batch × seq_len × vocab_size
    # This can be large for big vocabularies!
    output_logits_mb = (b * s * v * bytes_per_param) / (1024 ** 2)
    
    # Total activations
    activations_mb = (
        input_embedding_mb +
        layers_activations_mb +
        output_logits_mb
    )
    
    # ========================================================================
    # 5. PYTORCH OVERHEAD
    # ========================================================================
    
    # PyTorch allocates extra memory for:
    # - Memory fragmentation
    # - Caching allocator
    # - Temporary buffers
    # - CUDA context
    # Conservative estimate: 15-20% overhead
    base_memory = model_mb + optimizer_mb + gradients_mb + activations_mb
    overhead_mb = base_memory * 0.20  # 20% overhead
    
    # ========================================================================
    # TOTAL VRAM
    # ========================================================================
    
    total_mb = base_memory + overhead_mb
    
    return VRAMEstimate(
        model_mb=model_mb,
        optimizer_mb=optimizer_mb,
        activations_mb=activations_mb,
        gradients_mb=gradients_mb,
        overhead_mb=overhead_mb,
        total_mb=total_mb
    )


def get_available_vram_mb() -> float:
    """
    Get available VRAM on the current device.
    
    Returns:
        Available VRAM in MB, or 0 if CUDA not available
    """
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        # Get total and allocated memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)
        
        # Available = total - max(allocated, reserved)
        # We use reserved because PyTorch's caching allocator may reserve more than allocated
        used_memory = max(allocated_memory, reserved_memory)
        available_memory = total_memory - used_memory
        
        return available_memory / (1024 ** 2)  # Convert to MB
    except Exception:
        return 0.0


def get_total_vram_mb() -> float:
    """
    Get total VRAM on the current device.
    
    Returns:
        Total VRAM in MB, or 0 if CUDA not available
    """
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return total_memory / (1024 ** 2)
    except Exception:
        return 0.0


def check_vram_availability(
    estimated_mb: float,
    safety_margin: float = 0.9
) -> Tuple[bool, str, str]:
    """
    Check if estimated VRAM usage fits within available memory.
    
    Args:
        estimated_mb: Estimated VRAM usage in MB
        safety_margin: Safety margin (0.9 = use max 90% of available VRAM)
    
    Returns:
        Tuple of (fits, status_level, message)
        - fits: Whether the config will fit in VRAM
        - status_level: "safe", "warning", or "error"
        - message: Human-readable status message
    """
    if not torch.cuda.is_available():
        return False, "error", "⚠️ CUDA not available. Training will run on CPU (very slow)."
    
    total_vram = get_total_vram_mb()
    available_vram = get_available_vram_mb()
    
    # Use available VRAM, but cap at total (in case available > total due to measurement issues)
    usable_vram = min(available_vram, total_vram * safety_margin)
    
    estimated_gb = estimated_mb / 1024
    usable_gb = usable_vram / 1024
    total_gb = total_vram / 1024
    
    # Calculate usage percentage
    usage_percent = (estimated_mb / total_vram) * 100 if total_vram > 0 else 100
    
    if estimated_mb <= usable_vram:
        # Safe - plenty of room
        if usage_percent < 70:
            return True, "safe", (
                f"✅ **VRAM: Safe**\n"
                f"Estimated: {estimated_gb:.2f} GB / {total_gb:.2f} GB ({usage_percent:.1f}%)\n"
                f"You have plenty of headroom for this configuration."
            )
        else:
            return True, "warning", (
                f"⚠️ **VRAM: Tight but OK**\n"
                f"Estimated: {estimated_gb:.2f} GB / {total_gb:.2f} GB ({usage_percent:.1f}%)\n"
                f"This should work, but you're using most of your VRAM."
            )
    else:
        # Not enough VRAM
        deficit_gb = (estimated_mb - usable_vram) / 1024
        return False, "error", (
            f"❌ **VRAM: Insufficient**\n"
            f"Estimated: {estimated_gb:.2f} GB / {total_gb:.2f} GB ({usage_percent:.1f}%)\n"
            f"You need approximately {deficit_gb:.2f} GB more VRAM.\n\n"
            f"**Suggestions:**\n"
            f"• Reduce batch size\n"
            f"• Reduce context length\n"
            f"• Use gradient accumulation\n"
            f"• Try 'low_memory' preset"
        )


def format_vram_breakdown(estimate: VRAMEstimate) -> str:
    """
    Format VRAM breakdown for display.
    
    Args:
        estimate: VRAM estimate
    
    Returns:
        Formatted string with breakdown
    """
    breakdown = estimate.get_breakdown()
    percentages = estimate.get_percentage_breakdown()
    
    lines = [
        "**VRAM Breakdown:**",
        ""
    ]
    
    for component, mb in breakdown.items():
        if component == 'Total':
            lines.append("---")
        
        gb = mb / 1024
        pct = percentages[component]
        
        # Create a simple bar
        bar_length = int(pct / 5)  # 5% per character
        bar = "█" * bar_length
        
        lines.append(f"{component:.<20} {gb:>6.2f} GB ({pct:>5.1f}%) {bar}")
    
    return "\n".join(lines)


# Quick rule of thumb for rough estimates (from your document)
def quick_estimate_gb(
    batch_size: int,
    context_length: int,
    num_layers: int,
    num_heads: int,
    mixed_precision: bool = True
) -> float:
    """
    Quick and dirty VRAM estimate.
    
    This is less accurate but useful for quick checks.
    
    Args:
        batch_size: Batch size
        context_length: Context length
        num_layers: Number of layers
        num_heads: Number of attention heads
        mixed_precision: Whether using FP16
    
    Returns:
        Estimated VRAM in GB
    """
    estimate_gb = (batch_size * context_length ** 2 * num_layers * num_heads) / 50_000_000
    
    if mixed_precision:
        estimate_gb *= 0.5
    
    return estimate_gb
