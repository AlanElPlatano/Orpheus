from typing import Dict, Optional
import pretty_midi
from bpm_reader import get_tempo_from_midi


def get_preprocessing_stats(
    midi_file: pretty_midi.PrettyMIDI, 
    tempo_bpm: Optional[float] = None,
    verbose: bool = False
) -> Dict:
    """
    Get statistics about the MIDI file for validation purposes.

    Args:
        midi_file: PrettyMIDI object
        tempo_bpm: Pre-calculated tempo in BPM (if None, will be calculated)

    Returns:
        Dictionary containing preprocessing statistics
    """
    total_notes = sum(len(inst.notes) for inst in midi_file.instruments 
                      if not inst.is_drum)

    drum_tracks = sum(1 for inst in midi_file.instruments if inst.is_drum)
    melodic_tracks = len(midi_file.instruments) - drum_tracks

    duration = midi_file.get_end_time()
    
    # Use provided tempo or calculate if not provided
    if tempo_bpm is None:
        tempo = get_tempo_from_midi(midi_file)
        if verbose:
            print(f"BPM not provided as an argument, recalculated as {tempo_bpm}")
    else:
        tempo = tempo_bpm

    # Calculate note density
    notes_per_second = total_notes / duration if duration > 0 else 0

    # Find pitch range
    all_pitches = []
    for inst in midi_file.instruments:
        if not inst.is_drum:
            all_pitches.extend([note.pitch for note in inst.notes])

    min_pitch = min(all_pitches) if all_pitches else 0
    max_pitch = max(all_pitches) if all_pitches else 0

    # Check for extremely short notes (that survived preprocessing)
    tempo_safe = tempo if tempo > 0 else 120.0
    min_duration = (4.0 / 64) / (tempo_safe / 60.0)
    short_notes = 0

    for inst in midi_file.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                if (note.end - note.start) < min_duration * 1.1:  # Small tolerance
                    short_notes += 1

    return {
        'total_notes': total_notes,
        'drum_tracks': drum_tracks,
        'melodic_tracks': melodic_tracks,
        'duration_seconds': duration,
        'tempo_bpm': tempo,
        'notes_per_second': notes_per_second,
        'pitch_range': (min_pitch, max_pitch),
        'pitch_span': max_pitch - min_pitch if all_pitches else 0,
        'short_notes_remaining': short_notes,
        'time_signatures': len(midi_file.time_signature_changes)
    }