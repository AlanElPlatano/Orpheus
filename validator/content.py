import pretty_midi
from typing import Tuple, List

from stats import get_preprocessing_stats


def validate_musical_content(midi_file: pretty_midi.PrettyMIDI, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate that the MIDI file contains meaningful musical content.
    
    Args:
        midi_file: PrettyMIDI object to validate
        verbose: If True, print detailed validation information
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        # Get basic statistics
        stats = get_preprocessing_stats(midi_file)
        
        # Check minimum duration
        if stats['duration_seconds'] < 1.0:
            issues.append(f"File too short: {stats['duration_seconds']:.2f} seconds")
        elif stats['duration_seconds'] > 1800:  # 30 minutes
            issues.append(f"File unusually long: {stats['duration_seconds']:.1f} seconds")
            
        # Check note count
        if stats['total_notes'] < 10:
            issues.append(f"Too few notes: {stats['total_notes']}")
        elif stats['total_notes'] > 20000:
            issues.append(f"Excessive note count: {stats['total_notes']}")
            
        # Check melodic tracks
        if stats['melodic_tracks'] == 0:
            issues.append("No melodic tracks found")
        elif stats['melodic_tracks'] > 50:
            issues.append(f"Unusually many melodic tracks: {stats['melodic_tracks']}")
            
        # Check tempo
        if stats['tempo_bpm'] <= 0:
            issues.append("Invalid tempo detected")
        elif stats['tempo_bpm'] < 20:
            issues.append(f"Unusually slow tempo: {stats['tempo_bpm']:.1f} BPM")
        elif stats['tempo_bpm'] > 300:
            issues.append(f"Unusually fast tempo: {stats['tempo_bpm']:.1f} BPM")
            
        # Check note density
        if stats['notes_per_second'] > 50:
            issues.append(f"Extremely high note density: {stats['notes_per_second']:.1f} notes/sec")
        elif stats['notes_per_second'] < 0.1:
            issues.append(f"Very low note density: {stats['notes_per_second']:.2f} notes/sec")
            
        # Check pitch range
        pitch_span = stats['pitch_span']
        if pitch_span == 0:
            issues.append("All notes have the same pitch")
        elif pitch_span > 100:
            issues.append(f"Extremely wide pitch range: {pitch_span} semitones")
            
        if verbose:
            if issues:
                print(f"⚠ Found {len(issues)} musical content issues")
            else:
                print("✓ Musical content validation passed")
                
    except Exception as e:
        issues.append(f"Error during musical content validation: {str(e)}")
        
    return len(issues) == 0, issues