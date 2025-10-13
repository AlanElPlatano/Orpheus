import warnings
import pretty_midi
from typing import Optional
from .bpm_reader import get_tempo_from_midi


def quantize_midi_timing(
    midi_file: pretty_midi.PrettyMIDI, 
    quantize_grid: int = None,
    tempo_bpm: Optional[float] = None,
    verbose: bool = False
) -> pretty_midi.PrettyMIDI:
    """
    Quantize note timing to user-specified grid divisions.

    Args:
        midi_file: PrettyMIDI object to quantize
        quantize_grid: Grid division (e.g., 16 for 1/16 notes, 32 for 1/32 notes)
                      None to skip quantization
        tempo_bpm: Pre-calculated tempo in BPM (if None, will be calculated)
        verbose: If True, print detailed processing information

    Returns:
        Quantized PrettyMIDI object (or original if quantize_grid is None)
    """
    if quantize_grid is None:
        if verbose:
            print("Skipping quantization (quantize_grid=None)")
        return midi_file

    if verbose:
        print(f"Applying quantization with grid={quantize_grid}")

    # Use provided tempo or calculate again if not provided
    if tempo_bpm is None:
        tempo = get_tempo_from_midi(midi_file)
    else:
        tempo = tempo_bpm
        
    # Logic limits for tempo
    if tempo <= 0 or tempo >= 360:
        warnings.warn("Invalid tempo detected, using default 120 BPM")
        tempo = 120.0
    
    if verbose:
        print(f"Using tempo: {tempo:.2f} BPM")

    beats_per_second = tempo / 60.0
    # Duration of one grid unit (4.0 represents a whole note)
    grid_duration = (4.0 / quantize_grid) / beats_per_second

    # I used this a lot during development but it is not so useful now
    # if verbose:
    #    print(f"Grid duration: {grid_duration:.4f} seconds")

    notes_quantized = 0

    for inst_idx, instrument in enumerate(midi_file.instruments):
        if instrument.is_drum:
            if verbose:
                print(f"Skipping drum track {inst_idx}")
            continue

        for note in instrument.notes:
            original_start = note.start
            original_end = note.end

            # Quantize note start time
            note.start = round(note.start / grid_duration) * grid_duration

            # Quantize note end time
            note.end = round(note.end / grid_duration) * grid_duration

            # Ensure note has minimum duration (1/64 note)
            min_duration = (4.0 / 64) / beats_per_second
            if (note.end - note.start) < min_duration:
                note.end = note.start + min_duration

            # Verify that a change has been made
            if original_start != note.start or original_end != note.end:
                # Count of total notes quantized
                notes_quantized += 1

    if verbose:
        print(f"Quantized {notes_quantized} notes")

    return midi_file