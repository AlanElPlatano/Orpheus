"""
Optimizer setup for training.

This module provides functions to create optimizers with appropriate
settings for music generation model training.
"""

import torch
import torch.optim as optim
from typing import Optional, List, Dict
from torch.nn import Module


def create_optimizer(
    model: Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    no_decay_params: Optional[List[str]] = None
) -> optim.Optimizer:
    """
    Create optimizer for model training.

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient (L2 regularization)
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        epsilon: Adam epsilon parameter
        no_decay_params: List of parameter name patterns that should not have weight decay
                        (typically bias and LayerNorm parameters)

    Returns:
        Optimizer instance
    """
    # Default parameters that should not have weight decay
    if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm", "layer_norm", "ln"]

    # Separate parameters into those with and without weight decay
    decay_params = []
    no_decay_params_list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        # Check if this parameter should have weight decay
        should_decay = True
        for no_decay_pattern in no_decay_params:
            if no_decay_pattern in name:
                should_decay = False
                break

        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params_list.append(param)

    # Create parameter groups
    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay
        },
        {
            "params": no_decay_params_list,
            "weight_decay": 0.0
        }
    ]

    # Create optimizer
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=0.9
        )
    else:
        raise ValueError(
            f"Unknown optimizer type '{optimizer_type}'. "
            f"Available: 'adam', 'adamw', 'sgd'"
        )

    return optimizer


def get_optimizer_info(optimizer: optim.Optimizer) -> Dict:
    """
    Get information about optimizer configuration.

    Args:
        optimizer: Optimizer instance

    Returns:
        Dictionary with optimizer information
    """
    info = {
        "optimizer_type": type(optimizer).__name__,
        "num_param_groups": len(optimizer.param_groups),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }

    # Add parameter counts for each group
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group["params"])
        weight_decay = group.get("weight_decay", 0.0)
        info[f"group_{i}_num_params"] = num_params
        info[f"group_{i}_weight_decay"] = weight_decay

    return info


def print_optimizer_info(optimizer: optim.Optimizer):
    """
    Print information about optimizer configuration.

    Args:
        optimizer: Optimizer instance
    """
    print("\n" + "=" * 60)
    print("Optimizer Configuration")
    print("=" * 60)

    info = get_optimizer_info(optimizer)

    print(f"\nOptimizer type: {info['optimizer_type']}")
    print(f"Number of parameter groups: {info['num_param_groups']}")
    print(f"Learning rate: {info['learning_rate']:.2e}")

    for i in range(info['num_param_groups']):
        print(f"\nParameter group {i}:")
        print(f"  Number of parameters: {info[f'group_{i}_num_params']:,}")
        print(f"  Weight decay: {info[f'group_{i}_weight_decay']}")

    print("=" * 60)


def get_trainable_params(model: Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: Model to count parameters in

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model: Module, layer_names: List[str]):
    """
    Freeze specific layers in the model.

    Useful for:
    - Fine-tuning only specific parts of the model
    - Transfer learning
    - Debugging training issues

    Args:
        model: Model to freeze layers in
        layer_names: List of layer name patterns to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break


def unfreeze_layers(model: Module, layer_names: List[str]):
    """
    Unfreeze specific layers in the model.

    Args:
        model: Model to unfreeze layers in
        layer_names: List of layer name patterns to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                break


def clip_gradients(
    model: Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by global norm.

    This prevents exploding gradients, which can destabilize training.

    Args:
        model: Model to clip gradients in
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use (default: 2.0 for L2 norm)

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type
    )


def get_gradient_norm(model: Module, norm_type: float = 2.0) -> float:
    """
    Compute global gradient norm.

    Useful for monitoring gradient flow during training.

    Args:
        model: Model to compute gradient norm for
        norm_type: Type of norm to use (default: 2.0 for L2 norm)

    Returns:
        Global gradient norm
    """
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ]),
        norm_type
    )

    return total_norm.item()


__all__ = [
    'create_optimizer',
    'get_optimizer_info',
    'print_optimizer_info',
    'get_trainable_params',
    'freeze_layers',
    'unfreeze_layers',
    'clip_gradients',
    'get_gradient_norm'
]
