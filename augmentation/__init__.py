"""
Data augmentation module for MIDI tokenized files.

This module provides functionality to augment the training dataset by transposing
tokenized JSON files to different keys.
"""

from .transpose_tokenized_json import (
    transpose_json_file,
    batch_transpose_directory,
    calculate_new_key,
    SEMITONE_OFFSETS
)

__all__ = [
    'transpose_json_file',
    'batch_transpose_directory',
    'calculate_new_key',
    'SEMITONE_OFFSETS'
]
