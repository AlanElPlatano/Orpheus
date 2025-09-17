import pretty_midi
from typing import Optional


def remove_bass_tracks(
    midi_file: pretty_midi.PrettyMIDI,
    threshold_note: int = 36,  # Default to C2 (MIDI note 36)
    verbose: bool = False
) -> pretty_midi.PrettyMIDI:
    """
    Remove tracks where any of the following are true:
    - Contain any word for "bass" in the track name
    - If more than 35% of its notes are below the threshold.
        (Most bass tracks don't play all the time in their lower registry, hence the tolerance).
    
    Args:
        midi_file: PrettyMIDI object to process
        threshold_note: MIDI note number threshold (default 36 = C2)
                        Tracks with majority of notes below this are removed
        verbose: If True, print detailed processing information
    
    Returns:
        PrettyMIDI object with bass tracks removed
    """
    original_track_count = len(midi_file.instruments)
    tracks_to_keep = []
    
    # Bass-related terms in multiple languages, we check for track names to detect bass tracks
    # If we detect a bass track by its name, we avoid wasting resources by checking the notes themselves
    bass_terms = ['bass', 'bajo', 'basso', 'contrabajo', 'tololoche', 'upright']
    # Declared outside the loop to avoid redeclaring every iteration
    
    for inst_idx, instrument in enumerate(midi_file.instruments):
        # Get track name (program name or generic name)
        if instrument.name:
            track_name = instrument.name
        else:
            # If it fails, try to get program name from MIDI program number
            try:
                track_name = pretty_midi.program_to_instrument_name(instrument.program)
            except:
                track_name = f"Track {inst_idx}"

        # Skip drum tracks, follows the same logic as cleanup.py
        if instrument.is_drum:
            if verbose:
                print(f"Track {inst_idx}: Skipping drum track (will not be included)")
            continue
        
        # Check if track name contains bass-related terms
        # If the name gives out it is a bass track, we don't have to check the notes themselves
        track_name_lower = track_name.lower()
        is_named_bass = any(term in track_name_lower for term in bass_terms)
        
        if is_named_bass:
            if verbose:
                print(f"Track {inst_idx} '{track_name}': Removing bass track by name")
            continue  # Skip this track (don't add to tracks_to_keep)
        
        # Analyze note amount for non-bass-named tracks
        total_notes = len(instrument.notes)
        
        if total_notes == 0:
            # Empty track, will be handled by remove_empty_tracks
            tracks_to_keep.append(instrument)
            continue

        # Count notes below threshold
        bass_notes = sum(1 for note in instrument.notes if note.pitch < threshold_note)
        bass_percentage = (bass_notes / total_notes) * 100
        
        # Determine if this is a bass track based on note range
        is_bass_track = bass_percentage > 35  # More than 35% of notes are below threshold
        
        if is_bass_track:
            if verbose:
                avg_pitch = sum(note.pitch for note in instrument.notes) / total_notes
                # Convert threshold MIDI note to name for clarity
                note_name = pretty_midi.note_number_to_name(threshold_note)
                print(f"Track {inst_idx} '{track_name}': Removing bass track - "
                      f"{bass_percentage:.1f}% notes below {note_name} (MIDI {threshold_note}), "
                      f"avg pitch: {avg_pitch:.1f}")
        else:
            tracks_to_keep.append(instrument)
            if verbose and bass_percentage > 25:  # Notable bass notes but not enough to be removed, just a warning
                note_name = pretty_midi.note_number_to_name(threshold_note)
                print(f"Track {inst_idx} '{track_name}': Keeping mixed track - "
                      f"{bass_percentage:.1f}% notes below {note_name} (MIDI {threshold_note})")
    
    # Put the remaining tracks into the file
    midi_file.instruments = tracks_to_keep
    
    removed_count = original_track_count - len(midi_file.instruments)
    if verbose:
        print(f"Removed {removed_count} bass tracks (threshold: MIDI note {threshold_note})")
    
    return midi_file