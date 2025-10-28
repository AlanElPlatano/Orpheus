"""
PyTorch Dataset class for loading tokenized JSON files.

This module provides a Dataset class that loads pre-tokenized JSON files
and returns token sequences ready for training.
"""

import json
import torch
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from .constants import (
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    CONTEXT_LENGTH,
    VOCAB_SIZE,
    TRACK_TYPE_MELODY,
    TRACK_TYPE_CHORD,
    CORRIDOS_MELODY_PROGRAM,
    CORRIDOS_CHORD_PROGRAM,
    TOKEN_RANGES,
    get_track_type_from_program
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

    def _generate_track_ids(
        self,
        tokens: List[int],
        tracks_info: List[Dict]
    ) -> List[int]:
        """
        Generate track type IDs for each token in the sequence.

        This function analyzes the token sequence to determine which track
        (melody or chord) each token belongs to, based on Program tokens.

        Args:
            tokens: List of token IDs
            tracks_info: List of track dictionaries from JSON metadata

        Returns:
            List of track type IDs (0 for melody, 1 for chord)
        """
        # Create a mapping from program number to track type
        program_to_track = {}
        for track in tracks_info:
            program = track.get('program', -1)
            track_type = track.get('type', '').lower()

            if track_type == 'melody':
                program_to_track[program] = TRACK_TYPE_MELODY
            elif track_type == 'chord':
                program_to_track[program] = TRACK_TYPE_CHORD
            else:
                # Fallback: use heuristic
                program_to_track[program] = get_track_type_from_program(program)

        # Program token range
        program_start, program_end = TOKEN_RANGES['program']

        # Track the current track type (default to melody)
        current_track_type = TRACK_TYPE_MELODY
        track_ids = []

        for token in tokens:
            # Check if this is a Program token
            if program_start <= token <= program_end:
                # Extract program number from token
                # Program tokens are Program_0 to Program_127 and Program_-1
                # Token IDs: 266-394
                # Program_0 is 266, Program_127 is 393, Program_-1 is 394
                if token == 394:  # Program_-1
                    program_num = -1
                else:
                    program_num = token - 266

                # Update current track type based on program
                current_track_type = program_to_track.get(
                    program_num,
                    get_track_type_from_program(program_num)
                )

            track_ids.append(current_track_type)

        return track_ids

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
            - track_ids: Track type IDs tensor [seq_len]
            - metadata: Original file metadata (dict)
        """
        # Load JSON file
        file_path = self.file_paths[idx]
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract token sequence and track information
        tokens = data['global_tokens']
        tracks_info = data.get('tracks', [])

        # Generate track IDs BEFORE adding special tokens
        track_ids = self._generate_track_ids(tokens, tracks_info)

        # Add special tokens
        if self.add_bos:
            tokens = [BOS_TOKEN_ID] + tokens
            # BOS token gets the track type of the first real token
            # (or melody as default)
            first_track = track_ids[0] if track_ids else TRACK_TYPE_MELODY
            track_ids = [first_track] + track_ids

        if self.add_eos:
            tokens = tokens + [EOS_TOKEN_ID]
            # EOS token gets the track type of the last real token
            last_track = track_ids[-1] if track_ids else TRACK_TYPE_MELODY
            track_ids = track_ids + [last_track]

        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            track_ids = track_ids[:self.max_length]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)

        # Pad if necessary
        if self.pad_to_max_length and len(tokens) < self.max_length:
            num_padding = self.max_length - len(tokens)
            tokens = tokens + [PAD_TOKEN_ID] * num_padding
            attention_mask = attention_mask + [0] * num_padding
            # Padding tokens get a default track type (doesn't matter since they're masked)
            track_ids = track_ids + [TRACK_TYPE_MELODY] * num_padding

        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        track_ids_tensor = torch.tensor(track_ids, dtype=torch.long)

        # Create labels for next-token prediction
        # Labels should be input_ids shifted left by 1 (predict the next token)
        # For input [BOS, t1, t2, t3], labels should be [t1, t2, t3, EOS]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left: copy positions 1-end to 0-end-1
        labels[-1] = -100  # Last position has no next token to predict
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'track_ids': track_ids_tensor,
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


def _estimate_cache_memory_bytes(num_samples: int, max_length: int) -> int:
    """
    Estimate memory required to cache dataset in bytes.

    Args:
        num_samples: Number of samples in dataset
        max_length: Maximum sequence length

    Returns:
        Estimated memory in bytes
    """
    # Each sample contains 4 tensors (input_ids, attention_mask, labels, track_ids)
    # Each tensor has max_length elements of int64 (8 bytes each)
    bytes_per_tensor = max_length * 8
    bytes_per_sample_tensors = bytes_per_tensor * 4

    # Add overhead for metadata dict and file path string (~1 KB per sample)
    bytes_per_sample_overhead = 1024

    bytes_per_sample = bytes_per_sample_tensors + bytes_per_sample_overhead
    total_bytes = num_samples * bytes_per_sample

    return total_bytes


def _get_available_memory_bytes() -> Tuple[int, int]:
    """
    Get available system memory in bytes.

    Returns:
        Tuple of (available_bytes, total_bytes)
    """
    mem = psutil.virtual_memory()
    return mem.available, mem.total


def _format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


class CachedMusicTokenDataset(MusicTokenDataset):
    """
    Cached version of MusicTokenDataset that loads all files into memory.

    Use this for smaller datasets that fit in RAM for faster training.

    WARNING: This will load all samples into memory. For large datasets,
    this may cause out-of-memory errors. The initialization will fail
    if estimated memory usage exceeds 50% of available RAM.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize and cache all data in memory.

        Raises:
            MemoryError: If estimated memory usage exceeds 50% of available RAM
        """
        super().__init__(*args, **kwargs)

        # Estimate memory requirements
        estimated_bytes = _estimate_cache_memory_bytes(
            num_samples=len(self),
            max_length=self.max_length
        )
        available_bytes, total_bytes = _get_available_memory_bytes()

        # Check if estimated usage exceeds 80% of available RAM
        threshold = available_bytes * 0.8
        if estimated_bytes > threshold:
            raise MemoryError(
                f"Cannot cache dataset - estimated memory usage "
                f"({_format_bytes(estimated_bytes)}) exceeds 50% of available RAM "
                f"({_format_bytes(available_bytes)} available out of "
                f"{_format_bytes(total_bytes)} total).\n"
                f"Please use MusicTokenDataset instead of CachedMusicTokenDataset, "
                f"or reduce dataset size."
            )

        # Log memory usage estimate
        usage_pct = (estimated_bytes / available_bytes) * 100
        print(f"Estimated cache memory: {_format_bytes(estimated_bytes)} "
              f"({usage_pct:.1f}% of available RAM)")

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
    track_ids = [item['track_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Stack tensors (they should all be the same length if pad_to_max_length=True)
    batched = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'track_ids': torch.stack(track_ids),
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
