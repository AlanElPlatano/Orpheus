import os
import mido
from typing import Tuple, List


def validate_file_headers(file_path: str, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate MIDI file headers and basic format compliance using mido.
    
    Args:
        file_path: Path to the MIDI file
        verbose: If True, print detailed validation information
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            issues.append("File does not exist")
            return False, issues
            
        if not os.access(file_path, os.R_OK):
            issues.append("File is not readable")
            return False, issues
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            issues.append("File is empty (0 bytes)")
            return False, issues
        elif file_size < 14:  # Minimum size for MIDI header
            issues.append(f"File too small ({file_size} bytes) for valid MIDI")
            return False, issues
            
        # Try to read with mido for low-level validation
        try:
            midi_file = mido.MidiFile(file_path)
        except Exception as e:
            issues.append(f"Failed to parse MIDI with mido: {str(e)}")
            return False, issues
            
        # Check MIDI format type
        if midi_file.type not in [0, 1, 2]:
            issues.append(f"Invalid MIDI format type: {midi_file.type}")
            
        # Check number of tracks
        if len(midi_file.tracks) == 0:
            issues.append("MIDI file contains no tracks")
        elif len(midi_file.tracks) > 1000:  # Sanity check
            issues.append(f"Unusually high number of tracks: {len(midi_file.tracks)}")
            
        # Check ticks per beat
        if midi_file.ticks_per_beat <= 0:
            issues.append(f"Invalid ticks per beat: {midi_file.ticks_per_beat}")
        elif midi_file.ticks_per_beat > 10000:  # Sanity check
            issues.append(f"Unusually high ticks per beat: {midi_file.ticks_per_beat}")
            
        # Check for basic track structure
        for i, track in enumerate(midi_file.tracks):
            if len(track) == 0:
                issues.append(f"Track {i} is completely empty")
                
        if verbose and not issues:
            print(f"✓ Header validation passed - Type: {midi_file.type}, "
                  f"Tracks: {len(midi_file.tracks)}, "
                  f"TPB: {midi_file.ticks_per_beat}")
                  
    except Exception as e:
        issues.append(f"Unexpected error during header validation: {str(e)}")
        
    return len(issues) == 0, issues