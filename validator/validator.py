from pathlib import Path
from typing import Dict, Any

from headers import validate_file_headers
from parsing import validate_pretty_midi_parsing
from corruption import detect_advanced_corruption
from content import validate_musical_content
from stats import get_preprocessing_stats


def validate_midi_file(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Comprehensive MIDI file validation using all the other files from the validator module
    
    Args:
        file_path: Path to the MIDI file
        verbose: If True, print detailed validation information
        
    Returns:
        Dictionary containing validation results
    """
    file_path = str(file_path)  # Ensure string path
    
    if verbose:
        print(f"\n🔍 Validating: {Path(file_path).name}")
        
    validation_result = {
        'file_path': file_path,
        'is_valid': False,
        'issues': [],
        'warnings': [],
        'stats': None,
        'validation_steps': {
            'headers': False,
            'parsing': False,
            'corruption': False,
            'content': False
        }
    }
    
    # Step 1: Validate file headers
    headers_valid, header_issues = validate_file_headers(file_path, verbose)
    validation_result['validation_steps']['headers'] = headers_valid
    if not headers_valid:
        validation_result['issues'].extend(header_issues)
        return validation_result
        
    # Step 2: Validate pretty_midi parsing
    parsing_valid, parsing_issues, midi_file = validate_pretty_midi_parsing(file_path, verbose)
    validation_result['validation_steps']['parsing'] = parsing_valid
    
    if parsing_issues:
        # Distinguish between fatal errors and warnings
        fatal_issues = [issue for issue in parsing_issues if "Failed to parse" in issue]
        warning_issues = [issue for issue in parsing_issues if "Warning" in issue]
        
        validation_result['issues'].extend(fatal_issues)
        validation_result['warnings'].extend(warning_issues)
        
        if fatal_issues:
            return validation_result
            
    # Step 3: Advanced corruption detection
    if midi_file:
        corruption_valid, corruption_issues = detect_advanced_corruption(midi_file, verbose)
        validation_result['validation_steps']['corruption'] = corruption_valid
        if corruption_issues:
            validation_result['issues'].extend(corruption_issues)
            
        # Step 4: Musical content validation
        content_valid, content_issues = validate_musical_content(midi_file, verbose)
        validation_result['validation_steps']['content'] = content_valid
        
        # Content issues are often warnings rather than fatal errors
        validation_result['warnings'].extend(content_issues)
        
        # Get file statistics
        try:
            validation_result['stats'] = get_preprocessing_stats(midi_file)
        except Exception as e:
            validation_result['warnings'].append(f"Could not generate statistics: {str(e)}")
    
    # Determine overall validity
    # File is valid if headers and parsing work, even with warnings
    validation_result['is_valid'] = (
        validation_result['validation_steps']['headers'] and 
        validation_result['validation_steps']['parsing'] and
        validation_result['validation_steps']['corruption']
    )
    
    if verbose:
        if validation_result['is_valid']:
            print("✅ File is valid")
        else:
            print("❌ File validation failed")
            
    return validation_result