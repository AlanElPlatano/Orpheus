from pathlib import Path
from typing import List, Dict, Any

from validator import validate_midi_file


def validate_batch(directory_path: str, extensions: List[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate all MIDI files in a directory in batch.
    It just calls the function from validator, gets the stats and prints them.
    
    Args:
        directory_path: Path to directory containing MIDI files
        extensions: List of file extensions to check (default: common MIDI extensions)
        verbose: If True, print detailed validation information
        
    Returns:
        Dictionary containing batch validation results
    """
    if extensions is None:
        extensions = ['.mid', '.midi']
        
    # Receives folder as argument
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        return {'error': f"Directory does not exist: {directory_path}"}
        
    # Find all MIDI files in the folder
    midi_files = []
    for ext in extensions:
        midi_files.extend(directory_path.glob(f"*{ext}"))
        
    if verbose:
        print(f"\n🔍 Found {len(midi_files)} MIDI files in {directory_path}")
        
    batch_results = {
        'directory': str(directory_path),
        'total_files': len(midi_files),
        'valid_files': 0,
        'invalid_files': 0,
        'files_with_warnings': 0,
        'results': {}
    }
    
    # For each file, it attaches the information into batch_results
    for midi_file in midi_files:
        result = validate_midi_file(midi_file, verbose=verbose)
        batch_results['results'][str(midi_file)] = result
        
        if result['is_valid']:
            batch_results['valid_files'] += 1
        else:
            batch_results['invalid_files'] += 1
            
        if result['warnings']:
            batch_results['files_with_warnings'] += 1
            
    if verbose:
        # With spaces at the beginning of each print to organize every line better
        print(f"\n📊 Batch validation complete:")
        print(f"  Valid: {batch_results['valid_files']}")
        print(f"  Invalid: {batch_results['invalid_files']}")
        print(f"  With warnings: {batch_results['files_with_warnings']}")
        
    return batch_results