import pretty_midi


def remove_empty_tracks(midi_file: pretty_midi.PrettyMIDI, verbose: bool = False) -> pretty_midi.PrettyMIDI:
    """
    Remove tracks with no notes.

    Args:
        midi_file: PrettyMIDI object
        verbose: If True, print detailed processing information

    Returns:
        PrettyMIDI object with empty tracks removed
    """
    original_track_count = len(midi_file.instruments)
    midi_file.instruments = [inst for inst in midi_file.instruments 
                            if len(inst.notes) > 0]

    removed_count = original_track_count - len(midi_file.instruments)
    if verbose and removed_count > 0:
        print(f"Removed {removed_count} empty tracks")

    return midi_file