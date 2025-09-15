import warnings
import pretty_midi
import miditoolkit
from typing import Optional
from bpm_reader import get_tempo_from_midi


def preprocess_notes(
    midi_file: pretty_midi.PrettyMIDI, 
    tempo_bpm: Optional[float] = None,
    verbose: bool = False
) -> pretty_midi.PrettyMIDI:
    """
    Clean up notes before validation:
    - Remove notes shorter than 1/64 note duration
    - Remove empty notes (start == end)
    - Trim overlapping notes (cut sustain when new note starts)

    Args:
        midi_file: PrettyMIDI object to preprocess
        tempo_bpm: Pre-calculated tempo in BPM (if None, will be re-calculated)
        verbose: If True, print detailed processing information

    Returns:
        Preprocessed PrettyMIDI object
    """
    # Use provided tempo or calculate if not provided
    if tempo_bpm is None:
        tempo = get_tempo_from_midi(midi_file)
    else:
        tempo = tempo_bpm
    
    if verbose:
        print(f"Using tempo: {tempo} BPM")

    beats_per_second = tempo / 60.0
    min_duration = (4.0 / 64) / beats_per_second  # 1/64 note duration

    total_removed = 0
    total_trimmed = 0

    for inst_idx, instrument in enumerate(midi_file.instruments):
        if instrument.is_drum:
            if verbose:
                print(f"Skipping drum track {inst_idx}")
            continue

        original_note_count = len(instrument.notes)

        # Remove extremely short and empty notes
        valid_notes = []
        for note in instrument.notes:
            duration = note.end - note.start
            if duration >= min_duration and note.start >= 0:
                valid_notes.append(note)
            else:
                total_removed += 1
                if verbose and total_removed <= 5:  # Limit verbose output
                    print(f"  Removed note: pitch={note.pitch}, "
                          f"duration={duration:.4f}s (min={min_duration:.4f}s)")

        instrument.notes = valid_notes

        # Sort notes by start time for overlap processing
        instrument.notes.sort(key=lambda x: (x.start, x.pitch))

        # Trim overlapping notes (same pitch only)
        notes_by_pitch = {}
        for note in instrument.notes:
            if note.pitch not in notes_by_pitch:
                notes_by_pitch[note.pitch] = []
            notes_by_pitch[note.pitch].append(note)

        # Process overlaps for each pitch separately
        for pitch, pitch_notes in notes_by_pitch.items():
            for i in range(len(pitch_notes) - 1):
                current_note = pitch_notes[i]
                next_note = pitch_notes[i + 1]

                # If next note starts before current ends (overlap), trim current
                if next_note.start < current_note.end:
                    old_end = current_note.end
                    current_note.end = next_note.start

                    # Ensure minimum duration after trimming
                    if (current_note.end - current_note.start) < min_duration:
                        current_note.end = current_note.start + min_duration

                    total_trimmed += 1
                    if verbose and total_trimmed <= 5:
                        print(f"  Trimmed note: pitch={pitch}, "
                              f"end {old_end:.4f}s -> {current_note.end:.4f}s")

        if verbose:
            removed_count = original_note_count - len(instrument.notes)
            if removed_count > 0:
                print(f"Track {inst_idx}: Removed {removed_count} notes, "
                      f"{len(instrument.notes)} remaining")

    if verbose:
        print(f"Preprocessing complete: Removed {total_removed} notes, "
              f"Trimmed {total_trimmed} overlapping notes")

    return midi_file