from typing import Optional, Tuple, Dict
import pretty_midi

from bpm_reader import get_tempo_from_midi
from stats import get_preprocessing_stats
from cleanup import preprocess_notes
from quantizer import quantize_midi_timing
from remove_empty_tracks import remove_empty_tracks
from remove_bass_tracks import remove_bass_tracks


def process_midi_file(
    file_path: str, 
    quantize_grid: Optional[int] = None, 
    remove_empty: bool = True,
    remove_bass: bool = False,
    bass_threshold: int = 36,
    verbose: bool = False
) -> Tuple[bool, pretty_midi.PrettyMIDI, Dict]:
    """
    Complete preprocessing pipeline for a MIDI file.

    Args:
        file_path: Path to the MIDI file
        quantize_grid: Grid for quantization (None to skip)
        remove_empty: Whether to remove empty tracks
        remove_bass: Whether to remove bass tracks
        bass_threshold: MIDI note threshold for bass removal (default 36 = C2)
        verbose: If True, print detailed processing information

    Returns:
        Tuple of (success, processed_midi, statistics)
    """
    try:
        if verbose:
            print(f"\nProcessing: {file_path}")

        # Load MIDI file
        midi_file = pretty_midi.PrettyMIDI(file_path)
        
        # Calculate tempo once at the beginning
        tempo_bpm = get_tempo_from_midi(midi_file)
        if verbose:
            print(f"Detected tempo: {tempo_bpm} BPM")

        # Get initial stats (passing the calculated tempo)
        initial_stats = get_preprocessing_stats(midi_file, tempo_bpm=tempo_bpm)
        if verbose:
            print(f"Initial: {initial_stats['total_notes']} notes, "
                  f"{initial_stats['melodic_tracks']} melodic tracks")

        # Apply preprocessing steps (passing tempo to avoid recalculation)
        midi_file = preprocess_notes(midi_file, tempo_bpm=tempo_bpm, verbose=verbose)

        if quantize_grid is not None:
            midi_file = quantize_midi_timing(
                midi_file, 
                quantize_grid=quantize_grid, 
                tempo_bpm=tempo_bpm,
                verbose=verbose
            )

        if remove_bass:
            midi_file = remove_bass_tracks(
                midi_file,
                threshold_note=bass_threshold,
                verbose=verbose
            )

        if remove_empty:
            midi_file = remove_empty_tracks(midi_file, verbose=verbose)

        # Get final stats (passing the calculated tempo)
        final_stats = get_preprocessing_stats(midi_file, tempo_bpm=tempo_bpm)
        final_stats['preprocessing_applied'] = {
            'quantization': quantize_grid is not None,
            'quantize_grid': quantize_grid,
            'bass_tracks_removed': remove_bass,
            'bass_threshold': bass_threshold if remove_bass else None,
            'empty_tracks_removed': remove_empty,
            'notes_removed': initial_stats['total_notes'] - final_stats['total_notes'],
            'tracks_removed': initial_stats['melodic_tracks'] - final_stats['melodic_tracks']
        }

        if verbose:
            print(f"Final: {final_stats['total_notes']} notes, "
                  f"{final_stats['melodic_tracks']} melodic tracks")

        return True, midi_file, final_stats

    except Exception as e:
        if verbose:
            print(f"Error processing {file_path}: {str(e)}")
        return False, None, {'error': str(e)}