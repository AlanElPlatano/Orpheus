"""
Main training script for music generation model.

Usage:
    python train.py --config default
    python train.py --config production --resume
    python train.py --config quick_test
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import pytorch modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from pytorch.model.transformer import create_model
from pytorch.config.training_config import get_config_by_name, TrainingConfig
from pytorch.data.dataloader import create_dataloaders
from pytorch.training.trainer import Trainer
from pytorch.utils.device_utils import set_seed, print_device_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train music generation model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "quick_test", "overfit", "production"],
        help="Training configuration preset"
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("pytorch/data/splits/split_manifest.json"),
        help="Path to split manifest file"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to specific checkpoint to resume from"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps from config"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Device to train on"
    )

    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    print(f"\nLoading configuration: {args.config}")
    config = get_config_by_name(args.config)

    # Override config with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.device is not None:
        config.device = args.device
    if args.no_tensorboard:
        config.use_tensorboard = False
    if args.no_wandb:
        config.use_wandb = False
    if args.seed is not None:
        config.seed = args.seed
    if args.debug:
        config.debug = True
    if args.checkpoint is not None:
        config.resume_from_checkpoint = args.checkpoint
    if args.split_manifest is not None:
        config.split_manifest_path = args.split_manifest

    # Set random seed
    print(f"Setting random seed: {config.seed}")
    set_seed(config.seed, config.deterministic)

    # Print device info
    print_device_info()

    # Create data loaders
    print(f"\nCreating data loaders from: {config.split_manifest_path}")
    train_loader, val_loader, test_loader = create_dataloaders(
        split_manifest_path=config.split_manifest_path,
        batch_size=config.batch_size,
        max_length=config.context_length,
        num_workers=config.num_workers,
        use_cache=config.use_cache
    )

    if train_loader is None:
        print("ERROR: No training data found!")
        return

    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)}")
    else:
        print("No validation data found")

    # Create model
    print(f"\nCreating model...")
    model = create_model(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        max_len=config.context_length,
        dropout=config.dropout
    )

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")

    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Resume from checkpoint if requested
    if args.resume or config.resume_from_checkpoint is not None:
        trainer.resume_from_checkpoint(config.resume_from_checkpoint)

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    try:
        trainer.train()
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print(f"Total steps: {trainer.global_step}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: {config.log_dir}")


if __name__ == "__main__":
    main()
