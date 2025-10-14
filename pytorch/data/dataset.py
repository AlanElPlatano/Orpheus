"""
PyTorch Dataset class for loading tokenized JSON files.

This module provides a Dataset class that loads pre-tokenized JSON files
and returns token sequences ready for training.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from .constants import (
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    CONTEXT_LENGTH,
    VOCAB_SIZE
)


class MusicTokenDataset(Dataset):
    """
    PyTorch Dataset for loading tokenized music JSON files.

    Each item contains:
    - input_ids: Token sequence (truncated/padded to context_length)
    - attention_mask: Mask indicating real tokens vs padding
    - labels: Target tokens for next-token prediction (shifted input_ids)
    - metadata: Original file metadata (key, tempo, time signature, etc.)
    """

    def __init__(
        self,
        file_paths: List[Path],
        max_length: int = CONTEXT_LENGTH,
        add_bos: bool = True,
        add_eos: bool = True,
        pad_to_max_length: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            file_paths: List of paths to tokenized JSON files
            max_length: Maximum sequence length (default: CONTEXT_LENGTH=2048)
            add_bos: Whether to prepend BOS token (default: True)
            add_eos: Whether to append EOS token (default: True)
            pad_to_max_length: Whether to pad sequences to max_length (default: True)
        """
        self.file_paths = [Path(p) for p in file_paths]
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_to_max_length = pad_to_max_length

        # Verify all files exist
        self._verify_files()

        print(f"Dataset initialized with {len(self.file_paths)} files")

    def _verify_files(self):
        """Verify that all files exist and are readable."""
        missing_files = []
        for file_path in self.file_paths:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            raise FileNotFoundError(
                f"Missing {len(missing_files)} files:\n" +
                "\n".join(missing_files[:5]) +
                ("\n..." if len(missing_files) > 5 else "")
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Dictionary containing:
            - input_ids: Token sequence tensor [seq_len]
            - attention_mask: Attention mask tensor [seq_len]
            - labels: Target labels tensor [seq_len] (shifted input_ids)
            - metadata: Original file metadata (dict)
        """
        # Load JSON file
        file_path = self.file_paths[idx]
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract token sequence
        tokens = data['global_tokens']

        # Add special tokens
        if self.add_bos:
            tokens = [BOS_TOKEN_ID] + tokens
        if self.add_eos:
            tokens = tokens + [EOS_TOKEN_ID]

        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)

        # Pad if necessary
        if self.pad_to_max_length and len(tokens) < self.max_length:
            num_padding = self.max_length - len(tokens)
            tokens = tokens + [PAD_TOKEN_ID] * num_padding
            attention_mask = attention_mask + [0] * num_padding

        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Create labels for next-token prediction
        # Labels are the input_ids shifted by 1 position
        # We set padding positions to -100 (ignored by PyTorch loss functions)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'file_path': str(file_path),
            'metadata': data.get('metadata', {})
        }

    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with statistics like sequence lengths, vocab usage, etc.
        """
        sequence_lengths = []
        vocab_usage = set()

        print("Computing dataset statistics...")
        for idx in range(len(self)):
            file_path = self.file_paths[idx]
            with open(file_path, 'r') as f:
                data = json.load(f)

            tokens = data['global_tokens']
            sequence_lengths.append(len(tokens))
            vocab_usage.update(tokens)

        return {
            'num_samples': len(self),
            'min_length': min(sequence_lengths),
            'max_length': max(sequence_lengths),
            'avg_length': sum(sequence_lengths) / len(sequence_lengths),
            'median_length': sorted(sequence_lengths)[len(sequence_lengths) // 2],
            'vocab_coverage': len(vocab_usage),
            'vocab_coverage_pct': len(vocab_usage) / VOCAB_SIZE * 100
        }


class CachedMusicTokenDataset(MusicTokenDataset):
    """
    Cached version of MusicTokenDataset that loads all files into memory.

    Use this for smaller datasets that fit in RAM for faster training.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and cache all data in memory."""
        super().__init__(*args, **kwargs)
        print("Caching dataset in memory...")
        self.cache = [super().__getitem__(idx) for idx in range(len(self))]
        print(f"Cached {len(self.cache)} samples")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from cache."""
        return self.cache[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.

    This is used by the DataLoader to combine individual samples into a batch.
    Handles variable-length sequences by padding to the longest sequence in the batch.

    Args:
        batch: List of sample dictionaries from __getitem__

    Returns:
        Batched dictionary with tensors of shape [batch_size, seq_len]
    """
    # Extract fields
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Stack tensors (they should all be the same length if pad_to_max_length=True)
    batched = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }

    # Keep metadata as list (not batched)
    batched['metadata'] = [item['metadata'] for item in batch]
    batched['file_paths'] = [item['file_path'] for item in batch]

    return batched


def load_split_manifest(manifest_path: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Load the train/val/test split manifest.

    Args:
        manifest_path: Path to split_manifest.json

    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    train_files = [Path(p) for p in manifest['train']]
    val_files = [Path(p) for p in manifest['val']]
    test_files = [Path(p) for p in manifest['test']]

    return train_files, val_files, test_files


def create_datasets(
    split_manifest_path: Path,
    max_length: int = CONTEXT_LENGTH,
    use_cache: bool = False
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train, validation, and test datasets from split manifest.

    Args:
        split_manifest_path: Path to split_manifest.json
        max_length: Maximum sequence length
        use_cache: Whether to use cached dataset (loads all into memory)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load split manifest
    train_files, val_files, test_files = load_split_manifest(split_manifest_path)

    # Choose dataset class
    dataset_class = CachedMusicTokenDataset if use_cache else MusicTokenDataset

    # Create datasets
    print(f"\nCreating datasets with {dataset_class.__name__}...")
    train_dataset = dataset_class(train_files, max_length=max_length)
    val_dataset = dataset_class(val_files, max_length=max_length) if val_files else None
    test_dataset = dataset_class(test_files, max_length=max_length) if test_files else None

    print(f"Train: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val: {len(val_dataset)} samples")
    if test_dataset:
        print(f"Test: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset


__all__ = [
    'MusicTokenDataset',
    'CachedMusicTokenDataset',
    'collate_fn',
    'load_split_manifest',
    'create_datasets'
]
