"""
Main training loop orchestrator.

This module implements the Trainer class that handles the complete training
workflow including training loop, validation, checkpointing, and logging.
"""

import torch
import time
import json
import logging
import gc
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ..model.transformer import MusicTransformer
from ..config.training_config import TrainingConfig
from .loss import create_loss_function, compute_perplexity
from .optimizer import create_optimizer, clip_gradients, get_gradient_norm
from .scheduler import create_scheduler, get_current_lr
from ..utils.device_utils import get_device, to_device, print_device_info, print_memory_info
from ..utils.checkpoint_utils import (
    save_checkpoint, load_checkpoint, save_best_model,
    cleanup_old_checkpoints, get_latest_checkpoint
)
from ..utils.logging_utils import (
    TrainingLogger, MetricsTracker, format_time,
    print_training_header, print_epoch_summary, print_progress
)
from ..data.vocab import load_vocabulary, VocabularyInfo
from ..data.constants import CONDITION_NONE_ID, TEMPO_NONE_VALUE

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for music generation model.

    Handles:
    - Training loop with gradient accumulation
    - Validation loop
    - Checkpointing and model saving
    - Learning rate scheduling
    - Mixed precision training
    - Early stopping
    - Progress logging
    """

    def __init__(
        self,
        model: MusicTransformer,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup device
        self.device = get_device(config.device)
        self.model = model.to(self.device)

        # Setup loss function
        self.loss_fn = create_loss_function(
            loss_type=config.loss_type,
            ignore_index=config.ignore_index,
            label_smoothing=config.label_smoothing,
            melody_violation_weight=getattr(config, 'melody_violation_weight', 10.0),
            chord_violation_weight=getattr(config, 'chord_violation_weight', 5.0)
        )

        # Check if loss function is track-aware
        self.use_track_aware_loss = config.loss_type == 'track_aware'

        # Setup optimizer
        self.optimizer = create_optimizer(
            model=self.model,
            optimizer_type="adamw",
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon
        )

        # Setup learning rate scheduler
        num_training_steps = config.get_total_steps(len(train_loader.dataset))
        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=config.lr_scheduler_type,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=config.min_lr_ratio
        )

        # Setup mixed precision training
        self.scaler = GradScaler('cuda') if config.mixed_precision else None

        # Setup logging
        self.logger = TrainingLogger(
            log_dir=config.log_dir,
            use_tensorboard=config.use_tensorboard,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            wandb_run_name=config.wandb_run_name
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Load vocabulary info and tokenizer config for checkpoint saving
        self._load_vocab_and_tokenizer_config()

        # Log configuration
        self.logger.log_hyperparameters(config.to_dict())

        # Print configuration
        print_training_header(config)
        print_device_info(self.device)
        if self.device.type == "cuda":
            print_memory_info(self.device)

    def _load_vocab_and_tokenizer_config(self):
        """Load vocabulary and tokenizer config from data directory for checkpoint saving."""
        try:
            # Load vocabulary from processed JSON files
            logger.info(f"Loading vocabulary from {self.config.data_dir}")
            self.vocab_info = load_vocabulary(
                json_dir=self.config.data_dir,
                verify_consistency=True,
                num_files_to_check=5
            )
            logger.info(f"Loaded vocabulary: {self.vocab_info.vocab_size} tokens")

            # Load tokenizer config from first JSON file
            json_files = list(self.config.data_dir.glob('*.json'))
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tokenizer_config = data.get('tokenizer_config', {})
                    if self.tokenizer_config:
                        logger.info(f"Loaded tokenizer config from {json_files[0].name}")
                    else:
                        logger.warning("No tokenizer_config found in JSON, using defaults")
                        self.tokenizer_config = self._get_default_tokenizer_config()
            else:
                logger.warning("No JSON files found, using default tokenizer config")
                self.tokenizer_config = self._get_default_tokenizer_config()

        except Exception as e:
            logger.error(f"Failed to load vocab/tokenizer config: {e}")
            # Set to None so checkpoints work but warn user
            self.vocab_info = None
            self.tokenizer_config = None

    def _get_default_tokenizer_config(self) -> Dict:
        """Get default tokenizer configuration."""
        return {
            'pitch_range': (36, 84),
            'beat_resolution': 4,
            'num_velocities': 8,
            'additional_tokens': {
                'Chord': True,
                'Rest': True,
                'Tempo': True,
                'TimeSignature': True
            }
        }

    def _get_model_config(self) -> Dict:
        """Extract model configuration for checkpoint saving."""
        return {
            'vocab_size': self.config.vocab_size,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'ff_dim': self.config.ff_dim,
            'max_len': self.config.context_length,
            'dropout': self.config.dropout,
            'use_track_embeddings': self.config.use_track_embeddings,
            'num_track_types': self.config.num_track_types
        }

    def _get_extra_state(self) -> Dict:
        """Get extra state (vocab_info, tokenizer_config) for checkpoint saving."""
        extra_state = {}

        if self.vocab_info is not None:
            extra_state['vocab_info'] = self.vocab_info.to_dict()

        if self.tokenizer_config is not None:
            extra_state['tokenizer_config'] = self.tokenizer_config

        return extra_state

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()

        epoch_start_time = time.time()
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = to_device(batch, self.device)

            # Forward pass with optional mixed precision
            with autocast('cuda', enabled=self.config.mixed_precision):
                # Pass track_ids if available
                track_ids = batch.get('track_ids', None)

                # Extract and apply conditioning dropout if enabled
                key_ids = None
                tempo_values = None
                time_sig_ids = None

                if self.config.use_conditioning:
                    # Extract conditioning tensors from batch
                    key_ids = batch.get('key_id', None)
                    tempo_values = batch.get('tempo_value', None)
                    time_sig_ids = batch.get('time_sig_id', None)

                    # Apply conditioning dropout (randomly set conditions to "none")
                    # This teaches the model to generate both with and without conditions
                    if self.config.conditioning_dropout > 0 and self.model.training:
                        dropout_mask = torch.rand(key_ids.size(0), device=self.device) < self.config.conditioning_dropout

                        # Apply dropout to each conditioning type independently
                        if key_ids is not None:
                            key_ids = torch.where(dropout_mask, torch.tensor(CONDITION_NONE_ID, device=self.device), key_ids)

                        if tempo_values is not None:
                            tempo_values = torch.where(dropout_mask, torch.tensor(TEMPO_NONE_VALUE, device=self.device), tempo_values)

                        if time_sig_ids is not None:
                            time_sig_ids = torch.where(dropout_mask, torch.tensor(CONDITION_NONE_ID, device=self.device), time_sig_ids)

                logits, _ = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    track_ids=track_ids,
                    key_ids=key_ids,
                    tempo_values=tempo_values,
                    time_sig_ids=time_sig_ids
                )

                # Compute loss (pass track_ids if using track-aware loss)
                if self.use_track_aware_loss and track_ids is not None:
                    loss = self.loss_fn(logits, batch['labels'], track_ids)
                else:
                    loss = self.loss_fn(logits, batch['labels'])

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Determine if we should update weights
            # Update if reached accumulation steps (1) or last batch in epoch (2)
            is_accumulation_step = (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            should_update = is_accumulation_step or is_last_batch

            # Update weights after accumulation steps or at end of epoch
            if should_update:
                # Clip gradients
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = clip_gradients(
                    self.model,
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                self.scheduler.step()

                # Zero gradients (set_to_none=True is more memory efficient)
                self.optimizer.zero_grad(set_to_none=True)

                # Update global step
                self.global_step += 1

                # More aggressive memory cleanup to prevent VRAM accumulation
                # Clean up more frequently (every 50 steps instead of 100)
                if self.global_step % 50 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

                # Track metrics (ensure all values are detached scalars)
                metrics_tracker.update({
                    'loss': loss.item() * self.config.gradient_accumulation_steps,
                    'perplexity': compute_perplexity(loss.detach()).item(),
                    'grad_norm': grad_norm,
                    'learning_rate': get_current_lr(self.optimizer)
                })

                # Log progress
                if self.global_step % self.config.log_interval == 0:
                    avg_metrics = metrics_tracker.get_averages()
                    self.logger.log_metrics(
                        avg_metrics,
                        step=self.global_step,
                        prefix="train/"
                    )
                    metrics_tracker.reset()

                # Validation
                if self.config.do_validation and \
                   self.val_loader is not None and \
                   self.global_step % self.config.validation_interval == 0:
                    try:
                        val_metrics = self.validate()
                        self.logger.log_metrics(
                            val_metrics,
                            step=self.global_step,
                            prefix="val/"
                        )

                        # Save best model
                        if val_metrics['loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['loss']
                            self.patience_counter = 0
                            save_best_model(
                                checkpoint_dir=self.config.checkpoint_dir,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                epoch=self.current_epoch,
                                step=self.global_step,
                                val_loss=self.best_val_loss,
                                config=self.config.to_dict(),
                                model_config=self._get_model_config(),
                                extra_state=self._get_extra_state()
                            )
                            self.logger.log(f"New best model saved! Val loss: {self.best_val_loss:.4f}")
                        else:
                            self.patience_counter += 1

                        # Early stopping check
                        if self.config.early_stopping and \
                           self.patience_counter >= self.config.early_stopping_patience:
                            self.logger.log(
                                f"Early stopping triggered after {self.patience_counter} "
                                f"validation intervals without improvement"
                            )
                            return metrics_tracker.get_averages()

                    finally:
                        # Always return to training mode, even if validation fails
                        self.model.train()
                        # Aggressive memory cleanup after validation to free memory
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            gc.collect()

                # Save checkpoint
                if not self.config.save_best_only and \
                   self.global_step % self.config.checkpoint_interval == 0:
                    checkpoint_path = self.config.checkpoint_dir / \
                                    f"checkpoint_step_{self.global_step}.pt"
                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=self.current_epoch,
                        step=self.global_step,
                        best_val_loss=self.best_val_loss,
                        config=self.config.to_dict(),
                        model_config=self._get_model_config(),
                        extra_state=self._get_extra_state()
                    )

                    # Cleanup old checkpoints
                    cleanup_old_checkpoints(
                        self.config.checkpoint_dir,
                        max_to_keep=self.config.max_checkpoints_to_keep
                    )

                    # Memory cleanup after checkpoint save
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # Check max steps
                if self.config.max_steps is not None and \
                   self.global_step >= self.config.max_steps:
                    self.logger.log(f"Reached max steps: {self.config.max_steps}")
                    break

            # Overfit batch mode (for debugging)
            if self.config.overfit_batch:
                break

        epoch_time = time.time() - epoch_start_time
        return {
            **metrics_tracker.get_averages(),
            'epoch_time': epoch_time
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()

        num_batches = len(self.val_loader)
        if self.config.validation_batches is not None:
            num_batches = min(num_batches, self.config.validation_batches)

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= num_batches:
                break

            # Move batch to device
            batch = to_device(batch, self.device)

            # Forward pass
            with autocast('cuda', enabled=self.config.mixed_precision):
                # Pass track_ids if available
                track_ids = batch.get('track_ids', None)

                # Extract conditioning tensors if enabled (no dropout during validation)
                key_ids = None
                tempo_values = None
                time_sig_ids = None

                if self.config.use_conditioning:
                    key_ids = batch.get('key_id', None)
                    tempo_values = batch.get('tempo_value', None)
                    time_sig_ids = batch.get('time_sig_id', None)

                logits, _ = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    track_ids=track_ids,
                    key_ids=key_ids,
                    tempo_values=tempo_values,
                    time_sig_ids=time_sig_ids
                )

                # Compute loss (pass track_ids if using track-aware loss)
                if self.use_track_aware_loss and track_ids is not None:
                    loss = self.loss_fn(logits, batch['labels'], track_ids)
                else:
                    loss = self.loss_fn(logits, batch['labels'])

            # Track metrics (ensure all values are detached scalars)
            metrics_tracker.update({
                'loss': loss.item(),
                'perplexity': compute_perplexity(loss.detach()).item()
            })

        return metrics_tracker.get_averages()

    def train(self):
        """
        Main training loop.

        Runs training for the specified number of epochs or steps,
        with validation, checkpointing, and early stopping.
        """
        self.logger.log("Starting training...")
        training_start_time = time.time()

        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch

                self.logger.log(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

                # Train for one epoch
                train_metrics = self.train_epoch()

                # Validate at end of epoch
                val_metrics = None
                if self.config.do_validation and self.val_loader is not None:
                    val_metrics = self.validate()
                    self.logger.log_metrics(
                        val_metrics,
                        step=self.global_step,
                        prefix="val/"
                    )

                # Print epoch summary
                print_epoch_summary(
                    epoch=epoch + 1,
                    train_loss=train_metrics.get('loss', 0.0),
                    val_loss=val_metrics.get('loss', 0.0) if val_metrics else None,
                    learning_rate=get_current_lr(self.optimizer),
                    epoch_time=train_metrics.get('epoch_time', 0)
                )

                # Check max steps
                if self.config.max_steps is not None and \
                   self.global_step >= self.config.max_steps:
                    break

                # Check early stopping
                if self.config.early_stopping and \
                   self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.log("Early stopping triggered")
                    break

        except KeyboardInterrupt:
            self.logger.log("\nTraining interrupted by user")

        finally:
            # Save final checkpoint
            final_checkpoint_path = self.config.checkpoint_dir / "final_checkpoint.pt"
            save_checkpoint(
                checkpoint_path=final_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                step=self.global_step,
                best_val_loss=self.best_val_loss,
                config=self.config.to_dict(),
                model_config=self._get_model_config(),
                extra_state=self._get_extra_state()
            )

            training_time = time.time() - training_start_time
            self.logger.log(f"\nTraining completed!")
            self.logger.log(f"Total training time: {format_time(training_time)}")
            self.logger.log(f"Best validation loss: {self.best_val_loss:.4f}")
            self.logger.log(f"Total steps: {self.global_step}")

            # Close loggers
            self.logger.close()

    def resume_from_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (optional, uses latest if not provided)
        """
        if checkpoint_path is None:
            checkpoint_path = get_latest_checkpoint(self.config.checkpoint_dir)

        if checkpoint_path is None:
            self.logger.log("No checkpoint found to resume from")
            return

        self.logger.log(f"Resuming from checkpoint: {checkpoint_path}")

        metadata = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )

        self.current_epoch = metadata['epoch']
        self.global_step = metadata['step']
        self.best_val_loss = metadata.get('best_val_loss', float('inf'))

        self.logger.log(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def cleanup(self):
        """
        Clean up GPU memory and resources.

        Call this method before destroying the trainer to properly release GPU memory.
        """
        try:
            # Move model to CPU to free GPU memory
            if hasattr(self, 'model') and self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None

            # Delete optimizer
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                del self.optimizer
                self.optimizer = None

            # Delete scheduler
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                del self.scheduler
                self.scheduler = None

            # Delete scaler
            if hasattr(self, 'scaler') and self.scaler is not None:
                del self.scaler
                self.scaler = None

            # Clear data loader references
            if hasattr(self, 'train_loader'):
                self.train_loader = None
            if hasattr(self, 'val_loader'):
                self.val_loader = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("Trainer cleanup completed - GPU memory released")

        except Exception as e:
            logger.error(f"Error during trainer cleanup: {e}")


__all__ = ['Trainer']
