import pretty_midi
from typing import Tuple, List


def detect_advanced_corruption(midi_file: pretty_midi.PrettyMIDI, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Detect subtle corruption patterns that basic parsers might miss.
    
    Args:
        midi_file: PrettyMIDI object to validate
        verbose: If True, print detailed validation information
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    file_end_time = midi_file.get_end_time()
    
    # Track statistics for anomaly detection
    total_notes = 0
    pitch_counts = {}
    velocity_counts = {}
    duration_stats = []
    
    for inst_idx, inst in enumerate(midi_file.instruments):
        inst_issues = []
            
        for note_idx, note in enumerate(inst.notes):
            total_notes += 1
            
            # Basic temporal validations

            # Negative duration notes
            if note.end <= note.start:
                inst_issues.append(f"Note {note_idx} has negative/zero duration "
                                 f"(start: {note.start:.4f}, end: {note.end:.4f})")
                
            # Notes before the start of the file
            if note.start < 0:
                inst_issues.append(f"Note {note_idx} starts before file beginning "
                                 f"(start: {note.start:.4f})")
            
            # If notes end *after* the file ends
            if note.end > file_end_time + 0.1:  # Small tolerance for rounding
                inst_issues.append(f"Note {note_idx} extends beyond file end "
                                 f"(end: {note.end:.4f}, file_end: {file_end_time:.4f})")
            
            # Pitch validation
            if not (0 <= note.pitch <= 127):
                inst_issues.append(f"Note {note_idx} has invalid pitch: {note.pitch}")
            else:
                pitch_counts[note.pitch] = pitch_counts.get(note.pitch, 0) + 1
                
            # Velocity validation
            if not (0 <= note.velocity <= 127):
                inst_issues.append(f"Note {note_idx} has invalid velocity: {note.velocity}")
            else:
                velocity_counts[note.velocity] = velocity_counts.get(note.velocity, 0) + 1
                
            # Duration statistics
            duration = note.end - note.start
            if duration > 0:
                duration_stats.append(duration)
                
            # Check for extremely long notes (potential corruption)
            if duration > 300:  # 5 minutes
                inst_issues.append(f"Note {note_idx} has unusually long duration: "
                                 f"{duration:.2f} seconds")
                
            # Check for extremely short notes (should be caught by preprocessing anyway)
            if 0 < duration < 0.001:  # 1ms
                inst_issues.append(f"Note {note_idx} has extremely short duration: "
                                 f"{duration:.6f} seconds")
        
        # Add instrument-specific issues
        if inst_issues:
            inst_name = inst.name if inst.name else f"Instrument {inst_idx}"
            for issue in inst_issues[:5]:  # Limit to first 5 issues per instrument
                issues.append(f"{inst_name}: {issue}")
            if len(inst_issues) > 5:
                issues.append(f"{inst_name}: ... and {len(inst_issues) - 5} more issues")
    
    # Check for suspicious patterns
    if total_notes == 0:
        issues.append("File contains absolutely no notes at all")
    elif total_notes > 50000:
        issues.append(f"Unusually high note count: {total_notes}")
        
    # Check pitch distribution anomalies
    if pitch_counts:
        unique_pitches = len(pitch_counts)
        if unique_pitches == 1:
            issues.append(f"All notes have the same pitch: {list(pitch_counts.keys())[0]}")
        elif unique_pitches > 100:
            issues.append(f"Unusually wide pitch range: {unique_pitches} unique pitches")
            
    # Check velocity distribution anomalies
    if velocity_counts:
        unique_velocities = len(velocity_counts)
        if unique_velocities == 1 and list(velocity_counts.keys())[0] in [0, 127]:
            issues.append(f"All notes have extreme velocity: {list(velocity_counts.keys())[0]}")
            
    # Check duration statistics
    if duration_stats:
        avg_duration = sum(duration_stats) / len(duration_stats)
        max_duration = max(duration_stats)
        min_duration = min(duration_stats)
        
        if max_duration / min_duration > 10000:  # Huge duration variance
            issues.append(f"Extreme duration variance: {min_duration:.6f}s to {max_duration:.2f}s")
            
    if verbose:
        if issues:
            print(f"⚠ Found {len(issues)} corruption issues")
        else:
            print("✓ No corruption patterns detected")
            
    return len(issues) == 0, issues