# MIDI File Parsing and Validation System - Technical Specification

## Overview
This document outlines the comprehensive MIDI file parsing and validation system for the AI melody generation project. The goal is to create a robust pipeline that filters out corrupt, invalid, or unsuitable MIDI files before they contaminate the training dataset.

## Library Capabilities vs Custom Implementation

### What The Library Handles Automatically

#### `pretty_midi` Library
**Built-in Protections:**
- Basic MIDI format validation (checks file header structure)
- Automatic handling of malformed note events
- Time signature and tempo change parsing
- Invalid instrument filtering
- Basic file corruption detection (won't crash on most corrupt files)

**Limitations:**
- Doesn't validate musical coherence
- May silently ignore problematic sections
- No detection of empty or near-empty files
- Limited validation of note timing relationships

### What We Need to Implement Ourselves

## Custom Validation Rules

### 1. File Structure Validation
```python
def validate_file_structure(midi_file):
    """
    Validate basic MIDI file structure and content
    """
    checks = {
        'file_size': midi_file.size > 100,  # Minimum reasonable size
        'has_tracks': len(midi_file.instruments) > 0,
        'has_notes': any(len(inst.notes) > 0 for inst in midi_file.instruments),
        'reasonable_duration': 5 <= midi_file.get_end_time() <= 300,  # 5s to 5min
        'valid_tempo': 60 <= midi_file.estimate_tempo() <= 200
    }
    return all(checks.values()), checks
```

### 2. Musical Content Validation

#### Note Density and Distribution
**Problem**: Files with extreme note densities (too sparse/dense) won't train well
**Custom Rules Needed:**
- Minimum notes per second (avoid near-empty files)
- Maximum notes per second (avoid unplayable dense passages)  
- Note distribution across octaves (avoid single-note files)
- Reasonable note duration ranges

```python
def validate_note_density(midi_file):
    """
    Check if note density is within reasonable bounds
    """
    duration = midi_file.get_end_time()
    total_notes = sum(len(inst.notes) for inst in midi_file.instruments)
    
    notes_per_second = total_notes / duration
    
    return {
        'min_density': notes_per_second >= 0.5,  # At least 1 note per 2 seconds
        'max_density': notes_per_second <= 20,   # No more than 20 notes per second
        'total_notes': total_notes >= 10         # Minimum meaningful content
    }
```

#### Time Signature Consistency
**Problem**: Libraries parse time signatures but don't validate musical consistency
**Custom Rules Needed:**
- Detect time signature changes mid-song
- Validate that note patterns align with declared time signature
- Check for reasonable time signature values (4/4, 3/4, 6/8, etc.)

```python
def validate_time_signatures(midi_file):
    """
    Ensure time signature consistency and validity
    """
    time_sigs = midi_file.time_signature_changes
    
    if len(time_sigs) == 0:
        return False, "No time signature found"
    
    if len(time_sigs) > 1:
        return False, "Multiple time signature changes detected"
    
    ts = time_sigs[0]
    valid_signatures = [(4, 4), (3, 4), (2, 4), (6, 8), (9, 8), (12, 8)]
    
    return (ts.numerator, ts.denominator) in valid_signatures
```

#### Complete Time Signature Validation
**Problem**: Generated files must maintain specified time signature throughout entire duration
**Custom Rules Needed:**
- Validate every measure follows the specified time signature pattern
- Detect mid-song time signature violations
- Ensure note timing aligns perfectly with specified signature

```python
def validate_complete_time_signature_consistency(midi_file, expected_time_sig):
    """
    Validate that the entire file follows the specified time signature
    without any deviations throughout the complete duration
    """
    # Get the expected time signature (numerator, denominator)
    expected_num, expected_den = expected_time_sig
    
    # Calculate beats per measure and beat duration
    beats_per_measure = expected_num
    beat_duration = 4.0 / expected_den  # Quarter note = 1.0
    measure_duration = beats_per_measure * beat_duration
    
    # Analyze every measure in the file
    file_duration = midi_file.get_end_time()
    num_measures = int(file_duration / measure_duration) + 1
    
    violations = []
    
    for measure_idx in range(num_measures):
        measure_start = measure_idx * measure_duration
        measure_end = (measure_idx + 1) * measure_duration
        
        # Check all notes in this measure
        for inst in midi_file.instruments:
            if inst.is_drum:
                continue
                
            measure_notes = [note for note in inst.notes 
                           if measure_start <= note.start < measure_end]
            
            # Validate note timing within measure constraints
            for note in measure_notes:
                relative_start = (note.start - measure_start) % beat_duration
                relative_end = (note.end - measure_start) % beat_duration
                
                # Check if note aligns with time signature grid
                if not is_valid_timing_for_signature(relative_start, relative_end, expected_time_sig):
                    violations.append(f"Measure {measure_idx}: Invalid note timing")
    
    return len(violations) == 0, violations

def is_valid_timing_for_signature(note_start, note_end, time_sig):
    """
    Check if note timing is valid for the specified time signature
    """
    num, den = time_sig
    
    # Define valid subdivisions for different time signatures
    valid_subdivisions = {
        (4, 4): [0.0, 0.25, 0.5, 0.75, 1.0],  # Quarter note subdivisions
        (3, 4): [0.0, 0.333, 0.667, 1.0],     # Triplet subdivisions  
        (6, 8): [0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]  # Eighth note triplets
    }
    
    allowed_positions = valid_subdivisions.get((num, den), [0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Check if note start aligns with valid positions (with small tolerance)
    tolerance = 0.01
    return any(abs(note_start - pos) < tolerance for pos in allowed_positions)

### 3. Track Structure Validation

#### Two-Track Requirement Validation
**Problem**: We need exactly chord + melody tracks
**Custom Rules Needed:**
- Identify which tracks contain chords vs melody
- Ensure we have exactly 2 meaningful tracks
- Validate track separation (no melody notes in chord track, etc.)

```python
def validate_track_structure(midi_file):
    """
    Ensure we have proper chord + melody track separation
    """
    active_instruments = [inst for inst in midi_file.instruments 
                         if len(inst.notes) > 0 and not inst.is_drum]
    
    if len(active_instruments) != 2:
        return False, f"Expected 2 tracks, found {len(active_instruments)}"
    
    # Identify chord vs melody tracks
    track_analysis = []
    for inst in active_instruments:
        simultaneous_notes = count_simultaneous_notes(inst)
        avg_simultaneous = sum(simultaneous_notes) / len(simultaneous_notes)
        
        track_analysis.append({
            'instrument': inst,
            'avg_simultaneous': avg_simultaneous,
            'is_chord_track': avg_simultaneous > 1.5  # Chords have multiple simultaneous notes
        })
    
    chord_tracks = [t for t in track_analysis if t['is_chord_track']]
    melody_tracks = [t for t in track_analysis if not t['is_chord_track']]
    
    return len(chord_tracks) == 1 and len(melody_tracks) == 1
```

### 4. Data Quality Validation

#### Pitch Range Validation
**Problem**: Extreme pitch ranges suggest corrupted data
**Custom Rules Needed:**
- Reasonable pitch ranges (C2 to C7 for most instruments)
- No pitch bend extremes
- Consistent instrument assignments

```python
def validate_pitch_ranges(midi_file):
    """
    Check for reasonable pitch ranges and no extreme outliers
    """
    all_pitches = []
    for inst in midi_file.instruments:
        if not inst.is_drum:
            all_pitches.extend([note.pitch for note in inst.notes])
    
    if not all_pitches:
        return False, "No pitched notes found"
    
    min_pitch, max_pitch = min(all_pitches), max(all_pitches)
    
    return {
        'reasonable_range': 24 <= min_pitch <= 96 and 36 <= max_pitch <= 108,  # C2 to C7 range
        'not_too_wide': (max_pitch - min_pitch) <= 60,  # Max 5 octave range
        'no_extreme_low': min_pitch >= 21,  # Above A0
        'no_extreme_high': max_pitch <= 108  # Below C8
    }
```

### 5. Corruption Detection

#### Advanced Corruption Patterns
**Problem**: Libraries catch basic corruption but miss subtle issues
**Custom Rules Needed:**
- Detect impossible note timing (negative durations, overlaps)
- Identify truncated files
- Catch encoding issues

```python
def detect_advanced_corruption(midi_file):
    """
    Detect subtle corruption patterns that basic parsers miss
    """
    issues = []
    
    for inst in midi_file.instruments:
        for note in inst.notes:
            # Check for impossible note durations
            if note.end <= note.start:
                issues.append("Negative or zero duration note")
            
            # Check for extremely short notes (likely corruption)
            if (note.end - note.start) < 0.01:  # Less than 10ms
                issues.append("Extremely short note duration")
            
            # Check for notes starting before file begins
            if note.start < 0:
                issues.append("Note starts before file beginning")
    
    # Check for abrupt file endings (truncation)
    if midi_file.get_end_time() < 5:  # Suspiciously short
        last_events = []
        for inst in midi_file.instruments:
            if inst.notes:
                last_events.append(max(note.end for note in inst.notes))
        
        if len(set(last_events)) == 1:  # All tracks end at exactly same time
            issues.append("Possible file truncation detected")
    
    return len(issues) == 0, issues
```

## Comprehensive Validation Pipeline

### Pipeline Architecture
```python
class MIDIValidator:
    def __init__(self):
        self.validation_steps = [
            self.basic_file_validation,
            self.structure_validation, 
            self.musical_content_validation,
            self.track_structure_validation,
            self.pitch_range_validation,
            self.corruption_detection
        ]
    
    def validate_file(self, file_path):
        """
        Run complete validation pipeline
        Returns: (is_valid, detailed_report)
        """
        try:
            midi_file = pretty_midi.PrettyMIDI(file_path)
        except Exception as e:
            return False, {"error": f"Failed to parse MIDI file: {str(e)}"}
        
        validation_report = {}
        overall_valid = True
        
        for step in self.validation_steps:
            step_name = step.__name__
            try:
                is_valid, details = step(midi_file)
                validation_report[step_name] = {
                    'valid': is_valid,
                    'details': details
                }
                if not is_valid:
                    overall_valid = False
            except Exception as e:
                validation_report[step_name] = {
                    'valid': False,
                    'error': str(e)
                }
                overall_valid = False
        
        return overall_valid, validation_report

    def __init__(self, expected_time_signature=None):
        self.expected_time_signature = expected_time_signature  # e.g., (4, 4)
        self.validation_steps = [
            self.basic_file_validation,
            self.structure_validation, 
            self.musical_content_validation,
            self.track_structure_validation,
            self.pitch_range_validation,
            self.time_signature_consistency_validation,  # New step
            self.corruption_detection
        ]
    
    def time_signature_consistency_validation(self, midi_file):
        """
        Validate complete time signature consistency if specified
        """
        if self.expected_time_signature is None:
            return True, "No time signature requirement specified"
        
        return validate_complete_time_signature_consistency(midi_file, self.expected_time_signature)
```


## Implementation Strategy

### Phase 1: Basic Validation
- Implement file structure and basic corruption detection
- Use `pretty_midi` built-in capabilities where possible
- Focus on filtering obviously corrupt files

### Phase 2: Musical Validation  
- Add note density and pitch range validation
- Implement time signature consistency checking
- Add track structure validation for chord/melody separation

### Phase 3: Advanced Quality Control
- Implement sophisticated corruption detection
- Add musical coherence validation
- Performance optimization for batch processing

## Error Handling and Logging

### Logging Strategy
```python
import logging

class MIDIValidationLogger:
    def __init__(self):
        self.logger = logging.getLogger('midi_validation')
        
    def log_validation_results(self, file_path, is_valid, report):
        if is_valid:
            self.logger.info(f"✓ {file_path}: Valid")
        else:
            failed_checks = [k for k, v in report.items() if not v.get('valid', True)]
            self.logger.warning(f"✗ {file_path}: Failed checks: {failed_checks}")
            
    def log_batch_summary(self, total_files, valid_files, invalid_files):
        success_rate = (valid_files / total_files) * 100
        self.logger.info(f"Batch validation complete: {valid_files}/{total_files} valid ({success_rate:.1f}%)")
```

## Performance Considerations

### Batch Processing Optimization
- Process files in parallel where possible
- Implement early termination for obviously invalid files
- Cache validation results to avoid re-processing
- Memory management for large datasets

### Validation Speed vs. Thoroughness Trade-offs
- Quick validation mode for initial filtering
- Detailed validation mode for final dataset preparation
- Configurable validation strictness levels

## Integration with Main Pipeline

### Dataset Preparation Workflow
1. **Initial Scan**: Quick validation to identify obviously corrupt files
2. **Detailed Validation**: Comprehensive checks on remaining files  
3. **Quality Metrics**: Generate dataset quality reports
4. **Final Filtering**: Apply project-specific criteria (track count, duration, etc.)

### Validation Configuration
```python
VALIDATION_CONFIG = {
    'min_duration': 10,      # seconds
    'max_duration': 180,     # seconds  
    'min_tempo': 60,         # BPM
    'max_tempo': 180,        # BPM
    'required_tracks': 2,    # chord + melody
    'max_pitch_range': 60,   # semitones (5 octaves)
    'min_notes_per_track': 5,
    'allowed_time_signatures': [(4, 4), (3, 4), (2, 4), (6, 8)],
    'enforce_time_signature': True,  # Validate complete consistency
    'time_signature_tolerance': 0.01  # Timing tolerance in beats
}
```
