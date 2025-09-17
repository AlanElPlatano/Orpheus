import warnings
import pretty_midi
from typing import Tuple, List, Optional


def validate_pretty_midi_parsing(file_path: str, verbose: bool = False) -> Tuple[bool, List[str], Optional[pretty_midi.PrettyMIDI]]:
    """
    Validate that the file can be parsed by pretty_midi without errors.
    
    Args:
        file_path: Path to the MIDI file
        verbose: If True, print detailed validation information
        
    Returns:
        Tuple of (is_valid, list_of_issues, parsed_midi_or_none)
    """
    issues = []
    midi_file = None
    
    try:
        # Suppress warnings temporarily to catch them ourselves
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            midi_file = pretty_midi.PrettyMIDI(file_path)
            
            # Check if any warnings were raised
            if w:
                for warning in w:
                    issues.append(f"Warning during parsing: {str(warning.message)}")
                    
    except Exception as e:
        issues.append(f"Failed to parse with pretty_midi: {str(e)}")
        return False, issues, None
        
    if verbose and midi_file:
        print(f"✓ pretty_midi parsing successful - Duration: {midi_file.get_end_time():.2f}s")
        
    return len(issues) == 0, issues, midi_file