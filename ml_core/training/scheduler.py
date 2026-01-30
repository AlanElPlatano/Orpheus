"""
Learning rate schedulers for training.

This module provides various learning rate scheduling strategies
optimized for transformer training.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from typing import Optional


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and linear decay.

    Learning rate schedule:
    1. Linear warmup from 0 to max_lr over num_warmup_steps
    2. Linear decay from max_lr to min_lr over remaining steps

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Index of last epoch (for resuming)
        min_lr_ratio: Minimum learning rate as ratio of max LR (default: 0.0)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.

    Learning rate schedule:
    1. Linear warmup from 0 to max_lr over num_warmup_steps
    2. Cosine decay from max_lr to min_lr over remaining steps

    This is the most common schedule for transformer training.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (default: 0.5 for single decay)
        last_epoch: Index of last epoch (for resuming)
        min_lr_ratio: Minimum learning rate as ratio of max LR (default: 0.1)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and constant LR.

    Learning rate schedule:
    1. Linear warmup from 0 to max_lr over num_warmup_steps
    2. Constant max_lr for remaining steps

    Useful for:
    - Initial exploration of hyperparameters
    - Debugging training issues
    - Fine-tuning with stable learning rate

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        last_epoch: Index of last epoch (for resuming)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 1.0,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and polynomial decay.

    Learning rate schedule:
    1. Linear warmup from 0 to max_lr over num_warmup_steps
    2. Polynomial decay from max_lr to min_lr over remaining steps

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        power: Power of polynomial decay (1.0 = linear, 2.0 = quadratic, etc.)
        last_epoch: Index of last epoch (for resuming)
        min_lr_ratio: Minimum learning rate as ratio of max LR (default: 0.0)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Polynomial decay phase
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        lr_scale = (1.0 - progress) ** power
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * lr_scale)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1
):
    """
    Factory function to create a learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("linear", "cosine", "constant", "polynomial")
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of max LR
        last_epoch: Index of last epoch (for resuming)

    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            last_epoch,
            min_lr_ratio
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            num_cycles=0.5,
            last_epoch=last_epoch,
            min_lr_ratio=min_lr_ratio
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            last_epoch
        )
    elif scheduler_type == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            power=1.0,
            last_epoch=last_epoch,
            min_lr_ratio=min_lr_ratio
        )
    else:
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Available: 'linear', 'cosine', 'constant', 'polynomial'"
        )


def get_current_lr(optimizer: Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: Optimizer instance

    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def get_all_lrs(optimizer: Optimizer) -> list:
    """
    Get learning rates for all parameter groups.

    Args:
        optimizer: Optimizer instance

    Returns:
        List of learning rates for each parameter group
    """
    return [group['lr'] for group in optimizer.param_groups]


__all__ = [
    'get_linear_schedule_with_warmup',
    'get_cosine_schedule_with_warmup',
    'get_constant_schedule_with_warmup',
    'get_polynomial_decay_schedule_with_warmup',
    'create_scheduler',
    'get_current_lr',
    'get_all_lrs'
]
