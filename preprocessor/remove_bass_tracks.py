import pretty_midi
from typing import Optional


def remove_bass_tracks(
    midi_file: pretty_midi.PrettyMIDI,
    threshold_note: int = 48,  # Default to C3 (MIDI note 48)
    verbose: bool = False
) -> pretty_midi.PrettyMIDI:
    """
    Remove tracks that primarily contain bass notes below a specified threshold.
    A track is considered a bass track if more than 50% of its notes are below the threshold.
    Most bass tracks don't play all the time in their lower registry, hence the tolerance
    
    Args:
        midi_file: PrettyMIDI object to process
        threshold_note: MIDI note number threshold (default 48 = C3)
                       Tracks with majority of notes below this are removed
        verbose: If True, print detailed processing information
    
    Returns:
        PrettyMIDI object with bass tracks removed
    """
    original_track_count = len(midi_file.instruments)
    tracks_to_keep = []
    
    for inst_idx, instrument in enumerate(midi_file.instruments):
        # Always keep drum tracks
        if instrument.is_drum:
            tracks_to_keep.append(instrument)
            if verbose:
                print(f"Track {inst_idx}: Keeping drum track")
            continue
        
        # Analyze note distribution
        total_notes = len(instrument.notes)
        
        if total_notes == 0:
            # Empty track, will be handled by remove_empty_tracks
            tracks_to_keep.append(instrument)
            continue
        
        # Count notes below threshold
        bass_notes = sum(1 for note in instrument.notes if note.pitch < threshold_note)
        bass_percentage = (bass_notes / total_notes) * 100
        
        # Determine if this is a bass track
        is_bass_track = bass_percentage > 30  # More than 80% of notes are below threshold
        
        if is_bass_track:
            if verbose:
                avg_pitch = sum(note.pitch for note in instrument.notes) / total_notes
                print(f"Track {inst_idx}: Removing bass track - "
                      f"{bass_percentage:.1f}% notes below {threshold_note} "
                      f"(avg pitch: {avg_pitch:.1f})")
        else:
            tracks_to_keep.append(instrument)
            if verbose and bass_percentage > 50:  # Notable bass content but not removed
                print(f"Track {inst_idx}: Keeping mixed track - "
                      f"{bass_percentage:.1f}% notes below {threshold_note}")
    
    midi_file.instruments = tracks_to_keep
    
    removed_count = original_track_count - len(midi_file.instruments)
    if verbose:
        print(f"Removed {removed_count} bass tracks (threshold: MIDI note {threshold_note})")
    
    return midi_file