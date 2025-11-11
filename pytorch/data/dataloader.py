"""
DataLoader setup and utilities for training.

This module provides functions to create DataLoaders with appropriate
settings for training, validation, and testing.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple
from functools import partial
from torch.utils.data import DataLoader, Dataset

from .dataset import create_datasets, collate_fn, MusicTokenDataset
from .constants import DEFAULT_BATCH_SIZE, CONTEXT_LENGTH


def create_dataloaders(
    split_manifest_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = CONTEXT_LENGTH,
    num_workers: int = 0,
    use_cache: bool = False,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    dynamic_padding: bool = True
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        split_manifest_path: Path to split_manifest.json
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading (0 = main process)
        use_cache: Whether to cache dataset in memory
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        dynamic_padding: If True, pad to longest sequence in batch (saves memory)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        val_loader and test_loader may be None if no files in those splits
    """
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        split_manifest_path=split_manifest_path,
        max_length=max_length,
        use_cache=use_cache
    )

    # Create collate function with dynamic_padding parameter
    collate_fn_with_padding = partial(collate_fn, dynamic_padding=dynamic_padding)

    # Create training DataLoader (if training set exists)
    train_loader = None
    if train_dataset and len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=collate_fn_with_padding,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=True  # Drop last incomplete batch for stable training
        )

    # Create validation DataLoader (if validation set exists)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            collate_fn=collate_fn_with_padding,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False  # Keep all validation samples
        )

    # Create test DataLoader (if test set exists)
    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test
            num_workers=num_workers,
            collate_fn=collate_fn_with_padding,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False  # Keep all test samples
        )

    return train_loader, val_loader, test_loader


def get_dataloader_info(dataloader: DataLoader) -> dict:
    """
    Get information about a DataLoader.

    Args:
        dataloader: DataLoader to inspect

    Returns:
        Dictionary with DataLoader information
    """
    dataset = dataloader.dataset
    return {
        'num_samples': len(dataset),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last
    }


def print_dataloader_info(
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None
):
    """
    Print information about DataLoaders.

    Args:
        train_loader: Training DataLoader (optional)
        val_loader: Validation DataLoader (optional)
        test_loader: Test DataLoader (optional)
    """
    print("\n" + "=" * 60)
    print("DataLoader Information")
    print("=" * 60)

    if train_loader:
        print("\nTraining DataLoader:")
        info = get_dataloader_info(train_loader)
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("\nTraining DataLoader: None (no training files)")

    if val_loader:
        print("\nValidation DataLoader:")
        info = get_dataloader_info(val_loader)
        for key, value in info.items():
            print(f"  {key}: {value}")

    if test_loader:
        print("\nTest DataLoader:")
        info = get_dataloader_info(test_loader)
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("=" * 60)


def estimate_memory_usage(
    num_samples: int,
    sequence_length: int,
    batch_size: int,
    dtype_size: int = 8  # 8 bytes for int64
) -> dict:
    """
    Estimate memory usage for a dataset.

    Args:
        num_samples: Number of samples in dataset
        sequence_length: Sequence length
        batch_size: Batch size
        dtype_size: Size of data type in bytes (default: 8 for int64)

    Returns:
        Dictionary with memory estimates in different units
    """
    # Memory per sample (input_ids + attention_mask + labels)
    memory_per_sample = 3 * sequence_length * dtype_size

    # Total dataset memory
    total_dataset_memory = num_samples * memory_per_sample

    # Memory per batch
    memory_per_batch = batch_size * memory_per_sample

    return {
        'memory_per_sample_bytes': memory_per_sample,
        'memory_per_sample_mb': memory_per_sample / (1024 ** 2),
        'total_dataset_bytes': total_dataset_memory,
        'total_dataset_mb': total_dataset_memory / (1024 ** 2),
        'total_dataset_gb': total_dataset_memory / (1024 ** 3),
        'memory_per_batch_bytes': memory_per_batch,
        'memory_per_batch_mb': memory_per_batch / (1024 ** 2)
    }


def print_memory_estimates(
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None
):
    """
    Print memory usage estimates for DataLoaders.

    Args:
        train_loader: Training DataLoader (optional)
        val_loader: Validation DataLoader (optional)
        test_loader: Test DataLoader (optional)
    """
    print("\n" + "=" * 60)
    print("Memory Usage Estimates")
    print("=" * 60)

    # Use the first available loader to get batch info
    loader_to_use = train_loader or val_loader or test_loader
    if not loader_to_use:
        print("\nNo loaders available!")
        return

    # Get sequence length from first batch
    first_batch = next(iter(loader_to_use))
    sequence_length = first_batch['input_ids'].shape[1]

    print(f"\nSequence length: {sequence_length}")
    print(f"Batch size: {loader_to_use.batch_size}")

    if train_loader:
        print("\nTraining set:")
        train_mem = estimate_memory_usage(
            len(train_loader.dataset),
            sequence_length,
            train_loader.batch_size
        )
        print(f"  Total dataset: {train_mem['total_dataset_mb']:.2f} MB ({train_mem['total_dataset_gb']:.3f} GB)")
        print(f"  Per batch: {train_mem['memory_per_batch_mb']:.2f} MB")

    if val_loader:
        print("\nValidation set:")
        val_mem = estimate_memory_usage(
            len(val_loader.dataset),
            sequence_length,
            val_loader.batch_size
        )
        print(f"  Total dataset: {val_mem['total_dataset_mb']:.2f} MB ({val_mem['total_dataset_gb']:.3f} GB)")
        print(f"  Per batch: {val_mem['memory_per_batch_mb']:.2f} MB")

    if test_loader:
        print("\nTest set:")
        test_mem = estimate_memory_usage(
            len(test_loader.dataset),
            sequence_length,
            test_loader.batch_size
        )
        print(f"  Total dataset: {test_mem['total_dataset_mb']:.2f} MB ({test_mem['total_dataset_gb']:.3f} GB)")
        print(f"  Per batch: {test_mem['memory_per_batch_mb']:.2f} MB")

    print("\nNote: These are estimates for data only. Model parameters and gradients")
    print("      will require additional memory during training.")
    print("=" * 60)


__all__ = [
    'create_dataloaders',
    'get_dataloader_info',
    'print_dataloader_info',
    'estimate_memory_usage',
    'print_memory_estimates'
]
