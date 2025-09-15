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

## Pre-Processing Steps

### 1. MIDI Quantization (Optional)
```python
def quantize_midi_timing(midi_file, quantize_grid=None):
    """
    Quantize note timing to specified grid divisions
    
    Args:
        midi_file: PrettyMIDI object
        quantize_grid: Grid division (e.g., 16 for 1/16 notes, 32 for 1/32 notes)
                      None to skip quantization
    
    Returns:
        Quantized PrettyMIDI object (or original if quantize_grid is None)
    """
    if quantize_grid is None:
        return midi_file  # Skip quantization
    
    # Calculate grid size in seconds
    tempo = midi_file.estimate_tempo()
    beats_per_second = tempo / 60.0
    grid_duration = (4.0 / quantize_grid) / beats_per_second  # Duration of one grid unit
    
    for instrument in midi_file.instruments:
        for note in instrument.notes:
            # Quantize note start time
            note.start = round(note.start / grid_duration) * grid_duration
            
            # Quantize note end time
            note.end = round(note.end / grid_duration) * grid_duration
            
            # Ensure note has minimum duration (1/64 note)
            min_duration = (4.0 / 64) / beats_per_second
            if (note.end - note.start) < min_duration:
                note.end = note.start + min_duration
    
    return midi_file
```

### 2. Note Preprocessing
```python
def preprocess_notes(midi_file):
    """
    Clean up notes before validation:
    - Remove notes shorter than 1/64 note duration
    - Remove empty notes (start == end)
    - Trim overlapping notes (cut sustain when new note starts)
    """
    tempo = midi_file.estimate_tempo()
    beats_per_second = tempo / 60.0
    min_duration = (4.0 / 64) / beats_per_second  # 1/64 note duration
    
    for instrument in midi_file.instruments:
        # Remove extremely short and empty notes
        instrument.notes = [note for note in instrument.notes 
                           if (note.end - note.start) >= min_duration]
        
        # Sort notes by start time for overlap processing
        instrument.notes.sort(key=lambda x: x.start)
        
        # Trim overlapping notes
        for i in range(len(instrument.notes) - 1):
            current_note = instrument.notes[i]
            next_note = instrument.notes[i + 1]
            
            # If next note starts before current ends, trim current
            if next_note.start < current_note.end:
                current_note.end = next_note.start
                
                # Ensure minimum duration after trimming
                if (current_note.end - current_note.start) < min_duration:
                    current_note.end = current_note.start + min_duration
    
    return midi_file
```

## Custom Validation Rules

### 1. File Structure Validation
```python
def validate_file_structure(midi_file):
    """
    Validate basic MIDI file structure and content
    """
    checks = {
        'file_size': midi_file.get_end_time() > 0,  # File has content
        'has_tracks': len(midi_file.instruments) > 0,
        'has_notes': any(len(inst.notes) > 0 for inst in midi_file.instruments),
        'valid_tempo': 60 <= midi_file.estimate_tempo() <= 200
    }
    return all(checks.values()), checks
```

### 2. Musical Content Validation

#### Note Density and Distribution
```python
def validate_note_density(midi_file):
    """
    Check if note density is within reasonable bounds
    """
    duration = midi_file.get_end_time()
    if duration == 0:
        return False, "Zero duration file"
    
    total_notes = sum(len(inst.notes) for inst in midi_file.instruments)
    notes_per_second = total_notes / duration
    
    return {
        'min_density': notes_per_second >= 0.5,  # At least 1 note per 2 seconds
        'max_density': notes_per_second <= 20,   # No more than 20 notes per second
        'total_notes': total_notes >= 10         # Minimum meaningful content
    }
```

#### Time Signature Consistency
```python
def validate_time_signatures(midi_file, allowed_signatures=None):
    """
    Ensure time signature consistency and validity
    """
    if allowed_signatures is None:
        allowed_signatures = [(4, 4), (3, 4), (2, 4), (6, 8), (9, 8), (12, 8)]
    
    time_sigs = midi_file.time_signature_changes
    
    if len(time_sigs) == 0:
        return False, "No time signature found"
    
    if len(time_sigs) > 1:
        return False, "Multiple time signature changes detected"
    
    ts = time_sigs[0]
    signature_tuple = (ts.numerator, ts.denominator)
    
    if signature_tuple not in allowed_signatures:
        return False, f"Time signature {signature_tuple} not in allowed list: {allowed_signatures}"
    
    return True, f"Valid time signature: {signature_tuple}"
```

#### Complete Time Signature Validation (Conditional)
```python
def validate_complete_time_signature_consistency(midi_file, expected_time_sig, quantization_enabled=True):
    """
    Validate that the entire file follows the specified time signature
    Only performed if quantization was enabled
    """
    if not quantization_enabled:
        return True, "Skipped - quantization disabled"
    
    # Get the expected time signature (numerator, denominator)
    expected_num, expected_den = expected_time_sig
    
    # Calculate beats per measure and beat duration in seconds
    tempo = midi_file.estimate_tempo()
    beats_per_second = tempo / 60.0
    beat_duration = 1.0 / beats_per_second  # Duration of one beat in seconds
    measure_duration = expected_num * beat_duration
    
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
                
                # Check if note aligns with time signature grid
                if not is_valid_timing_for_signature(relative_start, expected_time_sig, beat_duration):
                    violations.append(f"Measure {measure_idx}: Invalid note timing at {note.start}")
    
    return len(violations) == 0, violations

def is_valid_timing_for_signature(relative_start, time_sig, beat_duration):
    """
    Check if note timing is valid for the specified time signature
    """
    num, den = time_sig
    
    # Define valid subdivisions based on common musical practice
    # Using 1/16 note as finest subdivision for most time signatures
    subdivision_size = beat_duration / 4  # 1/16 note duration
    
    # Check if note start aligns with subdivision grid (with small tolerance)
    tolerance = subdivision_size * 0.1  # 10% tolerance
    grid_position = round(relative_start / subdivision_size) * subdivision_size
    
    return abs(relative_start - grid_position) <= tolerance
```

### 3. Track Structure Validation

#### Bass Track Removal and Two-Track Requirement
```python
def remove_bass_tracks(midi_file, bass_threshold_pitch=36):
    """
    Remove tracks that contain a majority of notes below the specified pitch threshold
    Default: C2 (MIDI note 36)
    
    Args:
        midi_file: PrettyMIDI object
        bass_threshold_pitch: MIDI note number threshold (notes below this are considered bass)
    
    Returns:
        Modified PrettyMIDI object with bass tracks removed
    """
    instruments_to_keep = []
    
    for inst in midi_file.instruments:
        if inst.is_drum:
            instruments_to_keep.append(inst)
            continue
        
        # Check if track has any notes below threshold
        has_bass_notes = any(note.pitch < bass_threshold_pitch for note in inst.notes)
        
        if not has_bass_notes:
            instruments_to_keep.append(inst)
    
    # Update the MIDI file with filtered instruments
    midi_file.instruments = instruments_to_keep
    return midi_file

def validate_track_structure(midi_file):
    """
    Ensure we have proper chord + melody track separation after bass removal
    Uses triad detection: 3+ simultaneous notes = chord track
    REQUIRES both chord and melody tracks to be present
    """
    active_instruments = [inst for inst in midi_file.instruments 
                         if len(inst.notes) > 0 and not inst.is_drum]
    
    # Must have at least 2 tracks for chord + melody
    if len(active_instruments) < 2:
        return False, f"Insufficient tracks after bass removal, found {len(active_instruments)}, need at least 2"
    
    # Identify chord vs melody tracks using triad detection
    chord_tracks = []
    melody_tracks = []
    
    for inst in active_instruments:
        chord_moments = count_chord_moments(inst)  # Count times with 3+ simultaneous notes
        
        if chord_moments > 0:  # Any triads = chord track
            chord_tracks.append(inst)
        else:
            melody_tracks.append(inst)
    
    # Must have exactly 1 chord track and at least 1 melody track
    if len(chord_tracks) != 1:
        return False, f"Expected exactly 1 chord track, found {len(chord_tracks)}"
    
    if len(melody_tracks) < 1:
        return False, f"Expected at least 1 melody track, found {len(melody_tracks)}"
    
    # If more than 1 melody track, keep only the first one (or implement selection logic)
    if len(melody_tracks) > 1:
        # Keep the melody track with the most notes
        best_melody = max(melody_tracks, key=lambda x: len(x.notes))
        melody_tracks = [best_melody]
        
        # Remove extra melody tracks from the MIDI file
        midi_file.instruments = [inst for inst in midi_file.instruments 
                               if inst in chord_tracks or inst == best_melody or inst.is_drum]
    
    return True, f"Valid track structure: 1 chord track, 1 melody track (removed {len(active_instruments) - 2} extra tracks)"

def count_chord_moments(instrument):
    """
    Count moments where 3 or more notes are playing simultaneously (triads)
    """
    if len(instrument.notes) < 3:
        return 0
    
    chord_count = 0
    
    # Check each note's start time for simultaneous notes
    for i, note in enumerate(instrument.notes):
        simultaneous_count = 1  # Count the current note
        
        # Check how many other notes are playing at this moment
        for j, other_note in enumerate(instrument.notes):
            if i != j and other_note.start <= note.start < other_note.end:
                simultaneous_count += 1
        
        # If 3+ notes are simultaneous, it's a chord moment
        if simultaneous_count >= 3:
            chord_count += 1
    
    return chord_count
```

### 4. Data Quality Validation

#### Pitch Range Validation
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
    }, f"Pitch range: {min_pitch}-{max_pitch}"
```

### 5. Corruption Detection

#### Advanced Corruption Patterns
```python
def detect_advanced_corruption(midi_file):
    """
    Detect subtle corruption patterns that basic parsers miss
    """
    issues = []
    
    for inst in midi_file.instruments:
        for note in inst.notes:
            # Check for impossible note durations (should be caught by preprocessing)
            if note.end <= note.start:
                issues.append("Negative or zero duration note")
            
            # Check for notes starting before file begins
            if note.start < 0:
                issues.append("Note starts before file beginning")
            
            # Check for notes extending beyond file end
            if note.end > midi_file.get_end_time() + 0.1:  # Small tolerance
                issues.append("Note extends beyond file end")
    
    # Check for completely empty instruments that passed initial filtering
    empty_instruments = [inst for inst in midi_file.instruments if len(inst.notes) == 0]
    if empty_instruments:
        issues.append(f"Found {len(empty_instruments)} empty instruments")
    
    return len(issues) == 0, issues
```

## Token Representation System

### Beat-Based Timing
```python
def convert_to_tokens(midi_file):
    """
    Convert MIDI file to token sequence using beats as time units
    
    Token format:
    - NOTE_ON_<pitch>: Start playing note
    - NOTE_OFF_<pitch>: Stop playing note  
    - CHORD_<note1>_<note2>_<note3>: Chord with specific notes
    - WAIT_<beats>: Wait for specified number of beats
    - TEMPO_CHANGE_<bpm>: Tempo change marker
    """
    tempo = midi_file.estimate_tempo()
    beats_per_second = tempo / 60.0
    
    # Collect all events with timing
    all_events = []
    
    # Add tempo changes
    for tempo_change in midi_file.tempo_changes:
        beat_time = tempo_change.time * beats_per_second
        all_events.append({
            'time': beat_time,
            'type': 'tempo_change',
            'tempo': tempo_change.tempo
        })
    
    # Add note events
    for inst_idx, instrument in enumerate(midi_file.instruments):
        if instrument.is_drum:
            continue
            
        for note in instrument.notes:
            start_beat = note.start * beats_per_second
            end_beat = note.end * beats_per_second
            
            all_events.append({
                'time': start_beat,
                'type': 'note_on',
                'pitch': note.pitch,
                'track': inst_idx
            })
            
            all_events.append({
                'time': end_beat,
                'type': 'note_off', 
                'pitch': note.pitch,
                'track': inst_idx
            })
    
    # Sort events by time
    all_events.sort(key=lambda x: x['time'])
    
    # Convert to token sequence
    tokens = []
    current_time = 0.0
    
    for event in all_events:
        # Add wait token if time has passed
        if event['time'] > current_time:
            wait_beats = round((event['time'] - current_time) * 4) / 4  # Quantize to 1/4 beat
            if wait_beats > 0:
                tokens.append(f"WAIT_{wait_beats}")
                current_time = event['time']
        
        # Add event token
        if event['type'] == 'note_on':
            tokens.append(f"NOTE_ON_{event['pitch']}")
        elif event['type'] == 'note_off':
            tokens.append(f"NOTE_OFF_{event['pitch']}")
        elif event['type'] == 'tempo_change':
            tokens.append(f"TEMPO_CHANGE_{event['tempo']}")
    
    return tokens

def detect_chords_and_tokenize(midi_file):
    """
    Enhanced tokenization that detects and labels chord moments
    """
    # Implementation for chord detection and specialized chord tokens
    # This would analyze simultaneous notes and create CHORD_<notes> tokens
    pass
```

## Comprehensive Validation Pipeline

### Memory Management and Batch Processing
```python
def load_dataset_to_memory(dataset_folder):
    """
    Load entire MIDI dataset into system RAM for faster processing
    Returns list of (filepath, midi_data) tuples
    
    Note: Skip this implementation if too complex - SSD speed is sufficient
    """
    # Implementation optional - can process files directly from disk instead
    pass

### Pipeline Architecture
```python
class MIDIValidator:
    def __init__(self, config=None):
        """
        Initialize validator with configuration
        
        Args:
            config: Dict with validation parameters including:
                   - quantize_grid: Grid for quantization (None to skip)
                   - allowed_time_signatures: List of allowed time signatures
                   - enable_timing_validation: Whether to check note timing consistency
                   - bass_threshold_pitch: MIDI note below which tracks are removed
        """
        self.config = config or self._default_config()
        self.validation_steps = [
            self.basic_file_validation,
            self.structure_validation, 
            self.musical_content_validation,
            self.bass_track_removal_and_validation,  # New step
            self.pitch_range_validation,
            self.time_signature_consistency_validation,
            self.corruption_detection
        ]
    
    def _default_config(self):
        return {
            'quantize_grid': None,  # No quantization by default
            'allowed_time_signatures': [(4, 4), (3, 4), (2, 4), (6, 8)],
            'enable_timing_validation': True,
            'bass_threshold_pitch': 36,  # C2 - notes below this are considered bass
            'min_notes_per_second': 0.5,
            'max_notes_per_second': 20,
            'min_total_notes': 10,
            'min_tempo': 60,
            'max_tempo': 200
        }
    
    def validate_file(self, file_path):
        """
        Run complete validation pipeline with preprocessing
        Returns: (is_valid, detailed_report, processed_midi)
        """
        try:
            # Load MIDI file
            midi_file = pretty_midi.PrettyMIDI(file_path)
            
            # Apply preprocessing
            midi_file = preprocess_notes(midi_file)
            
            # Apply quantization if enabled
            quantization_applied = self.config.get('quantize_grid') is not None
            if quantization_applied:
                midi_file = quantize_midi_timing(midi_file, self.config['quantize_grid'])
            
        except Exception as e:
            return False, {"error": f"Failed to parse MIDI file: {str(e)}"}, None
        
        validation_report = {
            'preprocessing': {
                'quantization_applied': quantization_applied,
                'quantize_grid': self.config.get('quantize_grid')
            }
        }
        overall_valid = True
        
        # Run validation steps
        for step in self.validation_steps:
            step_name = step.__name__
            try:
                if step_name == 'time_signature_consistency_validation':
                    # Pass quantization status to timing validation
                    is_valid, details = step(midi_file, quantization_applied)
                else:
                    is_valid, details = step(midi_file)
                    
                validation_report[step_name] = {
                    'valid': is_valid,
                    'details': details
                }
                if not is_valid:
                    overall_valid = False
                    break  # Early termination - don't try to salvage borderline cases
            except Exception as e:
                validation_report[step_name] = {
                    'valid': False,
                    'error': str(e)
                }
                overall_valid = False
                break  # Early termination on error
        
        return overall_valid, validation_report, midi_file if overall_valid else None

    def basic_file_validation(self, midi_file):
        return validate_file_structure(midi_file)
    
    def structure_validation(self, midi_file):
        return validate_file_structure(midi_file)
    
    def musical_content_validation(self, midi_file):
        return validate_note_density(midi_file)
    
    def track_structure_validation(self, midi_file):
        return validate_track_structure(midi_file)
    
    def pitch_range_validation(self, midi_file):
        return validate_pitch_ranges(midi_file)
    
    def time_signature_consistency_validation(self, midi_file, quantization_applied):
        # First validate basic time signature
        basic_valid, basic_details = validate_time_signatures(midi_file, self.config['allowed_time_signatures'])
        if not basic_valid:
            return basic_valid, basic_details
        
        # Then validate complete consistency if quantization was applied
        if quantization_applied and self.config.get('enable_timing_validation', True):
            time_sig = midi_file.time_signature_changes[0]
            return validate_complete_time_signature_consistency(
                midi_file, 
                (time_sig.numerator, time_sig.denominator), 
                quantization_applied
            )
        else:
            return True, "Time signature validation skipped (no quantization or disabled)"
    
    def corruption_detection(self, midi_file):
        return detect_advanced_corruption(midi_file)
```

## Implementation Strategy

### Phase 1: Core Validation with Preprocessing
- Implement note preprocessing (cleanup, overlap trimming)
- Implement optional quantization
- Add basic file structure and corruption detection
- Add track structure validation with triad detection

### Phase 2: Musical Validation  
- Add note density validation
- Implement time signature consistency checking (conditional on quantization)
- Add pitch range validation

### Phase 3: Token Conversion System
- Implement beat-based tokenization
- Add chord detection and specialized chord tokens
- Implement bidirectional conversion (tokens ↔ MIDI)

## Updated Validation Configuration
```python
VALIDATION_CONFIG = {
    # Preprocessing options
    'quantize_grid': 16,             # 1/16 note quantization (None to skip)
    'enable_timing_validation': True, # Only applies if quantization enabled
    
    # Track filtering
    'bass_threshold_pitch': 36,      # C2 - remove tracks with notes below this
    
    # Time signature constraints  
    'allowed_time_signatures': [(4, 4), (3, 4), (2, 4), (6, 8)],
    
    # Quality thresholds
    'min_tempo': 60,                 # BPM
    'max_tempo': 180,                # BPM
    'min_notes_per_second': 0.5,     # Note density bounds
    'max_notes_per_second': 20,      
    'min_total_notes': 10,           # Minimum meaningful content
    
    # Pitch constraints
    'min_pitch': 21,                 # A0
    'max_pitch': 108,                # C8
    'max_pitch_range': 60,           # Max range in semitones
    
    # Processing options
    'remove_drums': True,            # Filter out drum tracks
    'merge_overlapping_notes': True, # Trim overlapping notes
    'early_termination': True        # Don't try to salvage borderline cases
}
```