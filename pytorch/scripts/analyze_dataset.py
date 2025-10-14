"""
Dataset analysis script for tokenized JSON files.

This script analyzes the tokenized dataset to understand:
- Sequence length distribution
- Token usage statistics
- Key/tempo/time signature distributions
- Track structure patterns

This information helps plan the PyTorch Dataset/DataLoader configuration.

"""

# python pytorch/scripts/analyze_dataset.py

# Future Tweaks
# Return instead of print: The print_results() function currently prints to console. 
# For Gradio, we'd want a format_results_html() function that returns an HTML string instead
# Progress callback: Currently uses print() for progress.
# For Gradio, you'd pass a callback function to update a progress bar:
# def analyze_dataset(json_dir, progress_callback=None):
#     for i, file in enumerate(files):
#         # ... processing ...
#         if progress_callback:
#             progress_callback(i / len(files))
#
# When you implement this, use gr.JSON() first, then later upgrade to gr.HTML() for a polished look.

import sys
import json
from pathlib import Path
from collections import Counter
from typing import Dict
import statistics

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pytorch.data.constants import (
    CONTEXT_LENGTH,
    is_pitch_token,
    is_duration_token,
    is_position_token,
    get_token_type
)


def analyze_single_file(json_path: Path) -> Dict:
    """
    Analyze a single tokenized JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary with file statistics
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        'file_name': json_path.name,
        'sequence_length': data.get('sequence_length', 0),
        'num_tracks': len(data.get('tracks', [])),
        'key': data.get('metadata', {}).get('key_signature', 'Unknown'),
        'tempo': None,
        'time_sig': None,
        'duration_seconds': data.get('metadata', {}).get('duration_seconds', 0),
        'note_count': data.get('metadata', {}).get('note_count', 0),
    }

    # Extract tempo (average)
    tempo_changes = data.get('metadata', {}).get('tempo_changes', [])
    if tempo_changes:
        stats['tempo'] = tempo_changes[0].get('bpm', 120)

    # Extract time signature
    time_sigs = data.get('metadata', {}).get('time_signatures', [])
    if time_sigs:
        ts = time_sigs[0]
        stats['time_sig'] = f"{ts.get('numerator', 4)}/{ts.get('denominator', 4)}"

    return stats


def analyze_dataset(json_dir: Path, max_files: int = None) -> Dict:
    """
    Analyze the entire dataset.

    Args:
        json_dir: Directory containing JSON files
        max_files: Maximum number of files to analyze (None = all)

    Returns:
        Dictionary with dataset statistics
    """
    json_files = list(json_dir.glob('*.json'))

    if max_files:
        json_files = json_files[:max_files]

    print(f"Analyzing {len(json_files)} JSON files...")
    print()

    # Collect statistics
    sequence_lengths = []
    keys = []
    tempos = []
    time_sigs = []
    durations = []
    note_counts = []
    track_counts = []

    # Token usage counter
    token_type_counts = Counter()

    for i, json_file in enumerate(json_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(json_files)} files...", end='\r')

        try:
            stats = analyze_single_file(json_file)

            sequence_lengths.append(stats['sequence_length'])
            keys.append(stats['key'])
            if stats['tempo']:
                tempos.append(stats['tempo'])
            if stats['time_sig']:
                time_sigs.append(stats['time_sig'])
            durations.append(stats['duration_seconds'])
            note_counts.append(stats['note_count'])
            track_counts.append(stats['num_tracks'])

            # Count token types (sample first 100 files to save time)
            if i < 100:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tokens = data.get('global_tokens', [])
                for token_id in tokens:
                    token_type = get_token_type(token_id)
                    token_type_counts[token_type] += 1

        except Exception as e:
            print(f"\nWarning: Failed to process {json_file.name}: {e}")

    print(f"\n  Completed analysis of {len(json_files)} files")
    print()

    # Compile results
    results = {
        'total_files': len(json_files),
        'sequence_lengths': {
            'min': min(sequence_lengths) if sequence_lengths else 0,
            'max': max(sequence_lengths) if sequence_lengths else 0,
            'mean': statistics.mean(sequence_lengths) if sequence_lengths else 0,
            'median': statistics.median(sequence_lengths) if sequence_lengths else 0,
            'stdev': statistics.stdev(sequence_lengths) if len(sequence_lengths) > 1 else 0,
            'percentile_95': sorted(sequence_lengths)[int(len(sequence_lengths) * 0.95)] if sequence_lengths else 0,
            'exceeds_context': sum(1 for s in sequence_lengths if s > CONTEXT_LENGTH),
        },
        'keys': dict(Counter(keys).most_common()),
        'tempos': {
            'min': min(tempos) if tempos else 0,
            'max': max(tempos) if tempos else 0,
            'mean': statistics.mean(tempos) if tempos else 0,
        },
        'time_signatures': dict(Counter(time_sigs).most_common()),
        'durations': {
            'min': min(durations) if durations else 0,
            'max': max(durations) if durations else 0,
            'mean': statistics.mean(durations) if durations else 0,
        },
        'note_counts': {
            'min': min(note_counts) if note_counts else 0,
            'max': max(note_counts) if note_counts else 0,
            'mean': statistics.mean(note_counts) if note_counts else 0,
        },
        'track_counts': dict(Counter(track_counts).most_common()),
        'token_types': dict(token_type_counts.most_common()),
    }

    return results


def print_results(results: Dict):
    """Print analysis results in a readable format."""

    print("=" * 70)
    print("DATASET ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\n{'Total Files:':<30} {results['total_files']}")

    # Sequence lengths
    print(f"\n{'─'*70}")
    print("SEQUENCE LENGTHS")
    print(f"{'─'*70}")
    seq_stats = results['sequence_lengths']
    print(f"{'Min:':<30} {seq_stats['min']} tokens")
    print(f"{'Max:':<30} {seq_stats['max']} tokens")
    print(f"{'Mean:':<30} {seq_stats['mean']:.1f} tokens")
    print(f"{'Median:':<30} {seq_stats['median']} tokens")
    print(f"{'Std Dev:':<30} {seq_stats['stdev']:.1f} tokens")
    print(f"{'95th Percentile:':<30} {seq_stats['percentile_95']} tokens")
    print(f"{'Context Length Limit:':<30} {CONTEXT_LENGTH} tokens")
    print(f"{'Exceeds Context Length:':<30} {seq_stats['exceeds_context']} files ({seq_stats['exceeds_context']/results['total_files']*100:.1f}%)")

    if seq_stats['exceeds_context'] > 0:
        print(f"\n⚠️  WARNING: {seq_stats['exceeds_context']} files exceed context length!")
        print(f"   These will need to be truncated during training.")

    # Musical characteristics
    print(f"\n{'─'*70}")
    print("MUSICAL CHARACTERISTICS")
    print(f"{'─'*70}")

    # Keys
    print(f"\nKey Distribution (top 10):")
    for key, count in list(results['keys'].items())[:10]:
        percentage = count / results['total_files'] * 100
        bar = '█' * int(percentage / 2)
        print(f"  {key:10s} {count:4d} ({percentage:5.1f}%) {bar}")

    # Tempo
    print(f"\nTempo Range:")
    tempo_stats = results['tempos']
    print(f"  Min:  {tempo_stats['min']:.1f} BPM")
    print(f"  Max:  {tempo_stats['max']:.1f} BPM")
    print(f"  Mean: {tempo_stats['mean']:.1f} BPM")

    # Time signatures
    print(f"\nTime Signature Distribution:")
    for time_sig, count in results['time_signatures'].items():
        percentage = count / results['total_files'] * 100
        bar = '█' * int(percentage / 2)
        print(f"  {time_sig:10s} {count:4d} ({percentage:5.1f}%) {bar}")

    # Durations
    print(f"\nDuration Range:")
    dur_stats = results['durations']
    print(f"  Min:  {dur_stats['min']:.1f} seconds")
    print(f"  Max:  {dur_stats['max']:.1f} seconds")
    print(f"  Mean: {dur_stats['mean']:.1f} seconds")

    # Note counts
    print(f"\nNote Count Range:")
    note_stats = results['note_counts']
    print(f"  Min:  {note_stats['min']} notes")
    print(f"  Max:  {note_stats['max']} notes")
    print(f"  Mean: {note_stats['mean']:.1f} notes")

    # Track counts
    print(f"\nTrack Count Distribution:")
    for track_count, count in results['track_counts'].items():
        percentage = count / results['total_files'] * 100
        print(f"  {track_count} tracks: {count:4d} ({percentage:5.1f}%)")

    # Token types
    print(f"\n{'─'*70}")
    print("TOKEN TYPE DISTRIBUTION (sampled from first 100 files)")
    print(f"{'─'*70}")
    total_tokens = sum(results['token_types'].values())
    for token_type, count in results['token_types'].items():
        percentage = count / total_tokens * 100
        print(f"  {token_type:15s} {count:7d} ({percentage:5.1f}%)")

    # Recommendations
    print(f"\n{'─'*70}")
    print("RECOMMENDATIONS FOR PYTORCH DATASET")
    print(f"{'─'*70}")

    # Context length
    if seq_stats['percentile_95'] < CONTEXT_LENGTH:
        print(f"✓ 95% of sequences fit within context length ({CONTEXT_LENGTH})")
        print(f"  Recommended: Use padding for shorter sequences")
    else:
        print(f"⚠ 95th percentile ({seq_stats['percentile_95']}) exceeds context length")
        print(f"  Recommended: Truncate longer sequences or increase context length")

    # Batch size
    avg_seq_len = seq_stats['mean']
    if avg_seq_len < 1000:
        recommended_batch_size = 16
    elif avg_seq_len < 1500:
        recommended_batch_size = 8
    else:
        recommended_batch_size = 4

    print(f"\n✓ Average sequence length: {avg_seq_len:.0f} tokens")
    print(f"  Recommended batch size: {recommended_batch_size} (adjust based on GPU memory)")

    # Data diversity
    num_unique_keys = len(results['keys'])
    num_unique_time_sigs = len(results['time_signatures'])

    print(f"\n✓ Dataset diversity:")
    print(f"  {num_unique_keys} unique keys")
    print(f"  {num_unique_time_sigs} unique time signatures")
    print(f"  Tempo range: {tempo_stats['max'] - tempo_stats['min']:.1f} BPM")


def main():
    """Main entry point."""
    processed_dir = project_root / 'processed'

    if not processed_dir.exists():
        print(f"ERROR: Processed directory not found at {processed_dir}")
        return 1

    print(f"Dataset directory: {processed_dir}")
    print()

    # Analyze dataset
    results = analyze_dataset(processed_dir)

    # Print results
    print_results(results)

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
