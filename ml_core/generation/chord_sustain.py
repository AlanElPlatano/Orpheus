"""
Chord sustain post-processing for MIDI files.

Extends chord note durations to the start of the next chord, ensuring
continuous harmonic support without gaps between chords.
"""

import logging
from typing import List, Tuple, Optional
from miditoolkit import MidiFile, Instrument, Note

logger = logging.getLogger(__name__)


def identify_chord_track(midi: MidiFile) -> Optional[int]:
    """
    Identify the chord track in a MIDI file.

    The chord track is identified by:
    1. Program number 29 (Electric Guitar Clean - commonly used for chords)
    2. High polyphony (3+ notes playing simultaneously)
    3. Track name containing "chord", "harmony", "accomp", or "pad"

    Args:
        midi: MidiFile object from miditoolkit

    Returns:
        Index of chord track, or None if not found
    """
    for idx, instrument in enumerate(midi.instruments):
        # Check for common chord indicators
        is_chord_program = instrument.program == 29  # Electric Guitar Clean

        # Check track name
        track_name_lower = instrument.name.lower() if instrument.name else ""
        has_chord_name = any(keyword in track_name_lower
                            for keyword in ["chord", "harmony", "accomp", "pad"])

        # Check polyphony (chords typically have 3+ simultaneous notes)
        if len(instrument.notes) > 0:
            max_polyphony = calculate_max_polyphony(instrument.notes)
            has_high_polyphony = max_polyphony >= 3
        else:
            has_high_polyphony = False

        # If any strong indicator is present, consider it the chord track
        if is_chord_program or (has_chord_name and has_high_polyphony):
            logger.info(f"Identified chord track at index {idx}: {instrument.name} "
                       f"(program={instrument.program}, polyphony={max_polyphony if has_high_polyphony else 'N/A'})")
            return idx

    logger.warning("Could not identify chord track in MIDI file")
    return None


def calculate_max_polyphony(notes: List[Note]) -> int:
    """
    Calculate maximum polyphony (simultaneous notes) in a note sequence.

    Args:
        notes: List of Note objects

    Returns:
        Maximum number of notes playing simultaneously
    """
    if not notes:
        return 0

    # Create events for note on/off
    events = []
    for note in notes:
        events.append((note.start, 1))  # Note on
        events.append((note.end, -1))   # Note off

    # Sort by time, with note-offs before note-ons at same time
    events.sort(key=lambda x: (x[0], x[1]))

    max_poly = 0
    current_poly = 0

    for time, delta in events:
        current_poly += delta
        max_poly = max(max_poly, current_poly)

    return max_poly


def get_chord_events(notes: List[Note]) -> List[Tuple[int, List[Note]]]:
    """
    Group notes into chord events based on simultaneous note starts.

    Args:
        notes: List of Note objects sorted by start time

    Returns:
        List of (start_time, chord_notes) tuples
    """
    if not notes:
        return []

    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda n: n.start)

    chord_events = []
    current_chord_start = sorted_notes[0].start
    current_chord_notes = []

    # Group notes that start at the same time (within a small tolerance)
    tolerance = 10  # ticks tolerance for simultaneous starts

    for note in sorted_notes:
        if abs(note.start - current_chord_start) <= tolerance:
            # Same chord
            current_chord_notes.append(note)
        else:
            # New chord starts
            if current_chord_notes:
                chord_events.append((current_chord_start, current_chord_notes))
            current_chord_start = note.start
            current_chord_notes = [note]

    # Add last chord
    if current_chord_notes:
        chord_events.append((current_chord_start, current_chord_notes))

    return chord_events


def extend_chord_durations(midi: MidiFile, chord_track_idx: int) -> None:
    """
    Extend chord note durations to the start of the next chord.

    Modifies the MIDI file in-place by extending the duration of each
    chord's notes to end when the next chord begins.

    Args:
        midi: MidiFile object from miditoolkit
        chord_track_idx: Index of the chord track to process
    """
    if chord_track_idx >= len(midi.instruments):
        logger.error(f"Invalid chord track index: {chord_track_idx}")
        return

    chord_track = midi.instruments[chord_track_idx]

    if not chord_track.notes:
        logger.warning("Chord track has no notes")
        return

    # Get chord events (groups of simultaneous notes)
    chord_events = get_chord_events(chord_track.notes)

    if len(chord_events) <= 1:
        logger.info("Only one chord found, no sustain adjustment needed")
        return

    logger.info(f"Found {len(chord_events)} chord events")

    # Extend each chord to the start of the next chord
    notes_extended = 0

    for i in range(len(chord_events) - 1):
        current_start, current_notes = chord_events[i]
        next_start, _ = chord_events[i + 1]

        # Extend all notes in current chord to next chord start
        for note in current_notes:
            if note.end < next_start:
                # Only extend if the note doesn't already reach the next chord
                note.end = next_start
                notes_extended += 1

    # For the last chord, extend to a reasonable length if it's too short
    # (e.g., at least 1 bar worth of ticks)
    last_start, last_notes = chord_events[-1]
    min_duration = midi.ticks_per_beat * 4  # 1 bar in 4/4 time

    for note in last_notes:
        current_duration = note.end - note.start
        if current_duration < min_duration:
            note.end = note.start + min_duration
            notes_extended += 1

    logger.info(f"Extended {notes_extended} notes across {len(chord_events)} chords")


def apply_chord_sustain(midi: MidiFile) -> MidiFile:
    """
    Apply chord sustain post-processing to a MIDI file.

    This function identifies the chord track and extends chord note
    durations to the start of the next chord, ensuring continuous
    harmonic support.

    Args:
        midi: MidiFile object from miditoolkit

    Returns:
        Modified MidiFile object (modified in-place, but returned for convenience)
    """
    logger.info("Applying chord sustain post-processing...")

    # Identify chord track
    chord_track_idx = identify_chord_track(midi)

    if chord_track_idx is None:
        logger.warning("No chord track found, skipping chord sustain processing")
        return midi

    # Extend chord durations
    extend_chord_durations(midi, chord_track_idx)

    logger.info("Chord sustain post-processing complete")

    return midi


__all__ = [
    'apply_chord_sustain',
    'identify_chord_track',
    'extend_chord_durations',
    'get_chord_events'
]