from typing import Optional, Tuple, Dict
import pretty_midi

from get_preprocessing_stats import get_preprocessing_stats
from preprocess_notes import preprocess_notes
from quantize_midi_timing import quantize_midi_timing
from remove_empty_tracks import remove_empty_tracks


def process_midi_file(file_path: str, quantize_grid: Optional[int] = None, remove_empty: bool = True, verbose: bool = False) -> Tuple[bool, pretty_midi.PrettyMIDI, Dict]:
    """
    Complete preprocessing pipeline for a MIDI file.

    Args:
        file_path: Path to the MIDI file
        quantize_grid: Grid for quantization (None to skip)
        remove_empty: Whether to remove empty tracks
        verbose: If True, print detailed processing information

    Returns:
        Tuple of (success, processed_midi, statistics)
    """
    try:
        if verbose:
            print(f"\nProcessing: {file_path}")

        # Load MIDI file
        midi_file = pretty_midi.PrettyMIDI(file_path)

        # Get initial stats
        initial_stats = get_preprocessing_stats(midi_file)
        if verbose:
            print(f"Initial: {initial_stats['total_notes']} notes, "
                  f"{initial_stats['melodic_tracks']} melodic tracks")

        # Apply preprocessing steps
        midi_file = preprocess_notes(midi_file, verbose=verbose)

        if quantize_grid is not None:
            midi_file = quantize_midi_timing(midi_file, quantize_grid=quantize_grid, verbose=verbose)

        if remove_empty:
            midi_file = remove_empty_tracks(midi_file, verbose=verbose)

        # Get final stats
        final_stats = get_preprocessing_stats(midi_file)
        final_stats['preprocessing_applied'] = {
            'quantization': quantize_grid is not None,
            'quantize_grid': quantize_grid,
            'empty_tracks_removed': remove_empty,
            'notes_removed': initial_stats['total_notes'] - final_stats['total_notes']
        }

        if verbose:
            print(f"Final: {final_stats['total_notes']} notes, "
                  f"{final_stats['melodic_tracks']} melodic tracks")

        return True, midi_file, final_stats

    except Exception as e:
        if verbose:
            print(f"Error processing {file_path}: {str(e)}")
        return False, None, {'error': str(e)}


