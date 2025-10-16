"""
Training infrastructure and utilities.
"""

from .loss import (
    WeightedMusicLoss,
    ConstraintAwareLoss,
    compute_perplexity,
    create_loss_function
)

from .optimizer import (
    create_optimizer,
    get_optimizer_info,
    print_optimizer_info,
    clip_gradients,
    get_gradient_norm
)

from .scheduler import (
    create_scheduler,
    get_current_lr,
    get_all_lrs
)

from .trainer import Trainer
from .gradio_trainer import GradioTrainer, TrainingMetrics

__all__ = [
    # Loss functions
    'WeightedMusicLoss',
    'ConstraintAwareLoss',
    'compute_perplexity',
    'create_loss_function',

    # Optimizer
    'create_optimizer',
    'get_optimizer_info',
    'print_optimizer_info',
    'clip_gradients',
    'get_gradient_norm',

    # Scheduler
    'create_scheduler',
    'get_current_lr',
    'get_all_lrs',

    # Trainer
    'Trainer',
    'GradioTrainer',
    'TrainingMetrics'
]
