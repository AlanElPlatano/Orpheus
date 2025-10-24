"""
Data splitting script for train/validation/test sets.

Splits the augmented dataset by original song to avoid data leakage.
Each original song has 12 transpositions (including the original key).
We ensure that all transpositions of the same song stay together in the same split.
"""

# python pytorch/data/split.py

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def extract_original_id(filename: str) -> str:
    """
    Extract the original song identifier from a filename.

    Original files: "Fm-125bpm-remi-dime.json"
    Transposed files: "Bm-125bpm-remi-dime-transpose-6.json" or "Am-125bpm-remi-dime-transpose+4.json"

    We extract the base name by removing the key prefix, tempo, 'remi',
    and transpose suffix to get the original song name.

    Args:
        filename: Name of the JSON file

    Returns:
        Original song identifier (e.g., "dime")
    """
    # Remove .json extension
    name = filename.replace('.json', '')

    # Remove transpose suffix if present (handles both transpose-X and transpose+X)
    if '-transpose' in name:
        # Find the position of '-transpose'
        transpose_pos = name.find('-transpose')
        # Remove everything from '-transpose' onwards
        name = name[:transpose_pos]

    # Now we have something like "Fm-125bpm-remi-dime"
    # Split by hyphens
    parts = name.split('-')

    # Remove key (first part), tempo (second part with 'bpm'), and 'remi'
    filtered_parts = []
    for i, part in enumerate(parts):
        if i == 0:  # Skip key (e.g., "Fm", "C#m")
            continue
        if 'bpm' in part:  # Skip tempo (e.g., "125bpm")
            continue
        if part == 'remi':  # Skip tokenization type
            continue
        filtered_parts.append(part)

    # Join remaining parts to get original song ID
    original_id = '-'.join(filtered_parts)
    return original_id


def group_files_by_original(processed_dir: Path) -> Dict[str, List[Path]]:
    """
    Group all JSON files by their original song ID.

    Args:
        processed_dir: Directory containing processed JSON files

    Returns:
        Dictionary mapping original song ID to list of file paths
    """
    groups = defaultdict(list)

    for json_file in processed_dir.glob('*.json'):
        original_id = extract_original_id(json_file.name)
        groups[original_id].append(json_file)

    return dict(groups)


def split_dataset(
    processed_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split dataset into train/validation/test sets by original song.

    All transpositions of the same original song stay together in the same split
    to avoid data leakage.

    Args:
        processed_dir: Directory containing processed JSON files
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Group files by original song
    groups = group_files_by_original(processed_dir)
    original_ids = list(groups.keys())

    print(f"Found {len(original_ids)} original songs")
    print(f"Total files: {sum(len(files) for files in groups.values())}")

    # Shuffle original IDs with seed for reproducibility
    random.seed(seed)
    random.shuffle(original_ids)

    # Calculate split indices
    num_songs = len(original_ids)

    # Handle small datasets - ensure at least 1 song per split when possible
    if num_songs < 3:
        print(f"\nWARNING: Very small dataset ({num_songs} songs)!")
        print("    Recommendation: Need at least 3 original songs for proper train/val/test split.")
        if num_songs == 1:
            print("    Assigning all files to training set (no validation/test).")
            train_count = 1
            val_count = 0
            test_count = 0
        elif num_songs == 2:
            print("    Assigning 1 song to train, 1 song to test (no validation).")
            train_count = 1
            val_count = 0
            test_count = 1
    else:
        # Use proportional split, but ensure at least 1 song in each split
        train_count = max(1, int(num_songs * train_ratio))
        val_count = max(1, int(num_songs * val_ratio))
        test_count = num_songs - train_count - val_count

        # If test_count is 0, take one from train (train should be largest)
        if test_count < 1 and train_count > 1:
            train_count -= 1
            test_count = 1

        # Double-check we haven't over-allocated
        if train_count + val_count + test_count != num_songs:
            # Adjust test to take any remaining
            test_count = num_songs - train_count - val_count

    # Split original IDs
    train_ids = original_ids[:train_count]
    val_ids = original_ids[train_count:train_count + val_count]
    test_ids = original_ids[train_count + val_count:]

    # Collect all files for each split
    train_files = []
    val_files = []
    test_files = []

    for song_id in train_ids:
        train_files.extend(groups[song_id])

    for song_id in val_ids:
        val_files.extend(groups[song_id])

    for song_id in test_ids:
        test_files.extend(groups[song_id])

    # Print split statistics
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_ids)} songs, {len(train_files)} files ({len(train_files)/sum(len(files) for files in groups.values())*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} songs, {len(val_files)} files ({len(val_files)/sum(len(files) for files in groups.values())*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} songs, {len(test_files)} files ({len(test_files)/sum(len(files) for files in groups.values())*100:.1f}%)")

    return train_files, val_files, test_files


def save_split_manifest(
    train_files: List[Path],
    val_files: List[Path],
    test_files: List[Path],
    output_dir: Path
):
    """
    Save the split manifest to JSON files for later use.

    Args:
        train_files: List of training file paths
        val_files: List of validation file paths
        test_files: List of test file paths
        output_dir: Directory to save manifest files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to strings for JSON serialization
    manifest = {
        'train': [str(f) for f in train_files],
        'val': [str(f) for f in val_files],
        'test': [str(f) for f in test_files]
    }

    # Save combined manifest
    manifest_path = output_dir / 'split_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved split manifest to {manifest_path}")

    # Also save individual split files for convenience
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_path = output_dir / f'{split_name}_files.txt'
        with open(split_path, 'w') as f:
            for file_path in files:
                f.write(str(file_path) + '\n')
        print(f"Saved {split_name} file list to {split_path}")


def main():
    """Main function to perform data splitting."""
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / 'processed'
    output_dir = project_root / 'pytorch' / 'data' / 'splits'

    print("=" * 60)
    print("Data Splitting Script")
    print("=" * 60)
    print(f"Processed directory: {processed_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Perform split
    train_files, val_files, test_files = split_dataset(
        processed_dir=processed_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    # Save manifest
    save_split_manifest(train_files, val_files, test_files, output_dir)

    print("\n" + "=" * 60)
    print("Split complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
