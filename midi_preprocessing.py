import pretty_midi
import numpy as np
from typing import Optional, Tuple, List, Dict
import warnings

class MIDIPreprocessor:
    """
    MIDI file preprocessing pipeline for cleaning and preparing MIDI files
    for AI training. Handles quantization and note cleanup operations.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the MIDI preprocessor.
        
        Args:
            verbose: If True, print detailed processing information
        """
        self.verbose = verbose
    
    def quantize_midi_timing(self, midi_file: pretty_midi.PrettyMIDI, 
                           quantize_grid: Optional[int] = None) -> pretty_midi.PrettyMIDI:
        """
        Quantize note timing to specified grid divisions.
        
        Args:
            midi_file: PrettyMIDI object to quantize
            quantize_grid: Grid division (e.g., 16 for 1/16 notes, 32 for 1/32 notes)
                          None to skip quantization
        
        Returns:
            Quantized PrettyMIDI object (or original if quantize_grid is None)
        """
        if quantize_grid is None:
            if self.verbose:
                print("Skipping quantization (quantize_grid=None)")
            return midi_file
        
        if self.verbose:
            print(f"Applying quantization with grid={quantize_grid}")
        
        # Calculate grid size in seconds
        tempo = midi_file.estimate_tempo()
        if tempo <= 0:
            warnings.warn("Invalid tempo detected, using default 120 BPM")
            tempo = 120.0
        
        beats_per_second = tempo / 60.0
        # Duration of one grid unit (4.0 represents a whole note)
        grid_duration = (4.0 / quantize_grid) / beats_per_second
        
        if self.verbose:
            print(f"Tempo: {tempo:.2f} BPM, Grid duration: {grid_duration:.4f} seconds")
        
        notes_quantized = 0
        
        for inst_idx, instrument in enumerate(midi_file.instruments):
            if instrument.is_drum:
                if self.verbose:
                    print(f"Skipping drum track {inst_idx}")
                continue
            
            for note in instrument.notes:
                original_start = note.start
                original_end = note.end
                
                # Quantize note start time
                note.start = round(note.start / grid_duration) * grid_duration
                
                # Quantize note end time
                note.end = round(note.end / grid_duration) * grid_duration
                
                # Ensure note has minimum duration (1/64 note)
                min_duration = (4.0 / 64) / beats_per_second
                if (note.end - note.start) < min_duration:
                    note.end = note.start + min_duration
                
                if original_start != note.start or original_end != note.end:
                    # Count of total notes quantized
                    notes_quantized += 1
        
        if self.verbose:
            print(f"Quantized {notes_quantized} notes")
        
        return midi_file
    
    def preprocess_notes(self, midi_file: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Clean up notes before validation:
        - Remove notes shorter than 1/64 note duration
        - Remove empty notes (start == end)
        - Trim overlapping notes (cut sustain when new note starts)
        
        Args:
            midi_file: PrettyMIDI object to preprocess
        
        Returns:
            Preprocessed PrettyMIDI object
        """
        if self.verbose:
            print("Starting note preprocessing")
        
        tempo = midi_file.estimate_tempo()
        if tempo <= 0:
            warnings.warn("Invalid tempo detected, using default 120 BPM")
            tempo = 120.0
        
        beats_per_second = tempo / 60.0
        min_duration = (4.0 / 64) / beats_per_second  # 1/64 note duration
        
        total_removed = 0
        total_trimmed = 0
        
        for inst_idx, instrument in enumerate(midi_file.instruments):
            if instrument.is_drum:
                if self.verbose:
                    print(f"Skipping drum track {inst_idx}")
                continue
            
            original_note_count = len(instrument.notes)
            
            # Remove extremely short and empty notes
            valid_notes = []
            for note in instrument.notes:
                duration = note.end - note.start
                if duration >= min_duration and note.start >= 0:
                    valid_notes.append(note)
                else:
                    total_removed += 1
                    if self.verbose and total_removed <= 5:  # Limit verbose output
                        print(f"  Removed note: pitch={note.pitch}, "
                              f"duration={duration:.4f}s (min={min_duration:.4f}s)")
            
            instrument.notes = valid_notes
            
            # Sort notes by start time for overlap processing
            instrument.notes.sort(key=lambda x: (x.start, x.pitch))
            
            # Trim overlapping notes (same pitch only)
            notes_by_pitch = {}
            for note in instrument.notes:
                if note.pitch not in notes_by_pitch:
                    notes_by_pitch[note.pitch] = []
                notes_by_pitch[note.pitch].append(note)
            
            # Process overlaps for each pitch separately
            for pitch, pitch_notes in notes_by_pitch.items():
                for i in range(len(pitch_notes) - 1):
                    current_note = pitch_notes[i]
                    next_note = pitch_notes[i + 1]
                    
                    # If next note starts before current ends (overlap), trim current
                    if next_note.start < current_note.end:
                        old_end = current_note.end
                        current_note.end = next_note.start
                        
                        # Ensure minimum duration after trimming
                        if (current_note.end - current_note.start) < min_duration:
                            current_note.end = current_note.start + min_duration
                        
                        total_trimmed += 1
                        if self.verbose and total_trimmed <= 5:
                            print(f"  Trimmed note: pitch={pitch}, "
                                  f"end {old_end:.4f}s -> {current_note.end:.4f}s")
            
            if self.verbose:
                removed_count = original_note_count - len(instrument.notes)
                if removed_count > 0:
                    print(f"Track {inst_idx}: Removed {removed_count} notes, "
                          f"{len(instrument.notes)} remaining")
        
        if self.verbose:
            print(f"Preprocessing complete: Removed {total_removed} notes, "
                  f"Trimmed {total_trimmed} overlapping notes")
        
        return midi_file
    
    def remove_empty_tracks(self, midi_file: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Remove tracks with no notes.
        
        Args:
            midi_file: PrettyMIDI object
        
        Returns:
            PrettyMIDI object with empty tracks removed
        """
        original_track_count = len(midi_file.instruments)
        midi_file.instruments = [inst for inst in midi_file.instruments 
                                if len(inst.notes) > 0]
        
        removed_count = original_track_count - len(midi_file.instruments)
        if self.verbose and removed_count > 0:
            print(f"Removed {removed_count} empty tracks")
        
        return midi_file
    
    def get_preprocessing_stats(self, midi_file: pretty_midi.PrettyMIDI) -> Dict:
        """
        Get statistics about the MIDI file for validation purposes.
        
        Args:
            midi_file: PrettyMIDI object
        
        Returns:
            Dictionary containing preprocessing statistics
        """
        total_notes = sum(len(inst.notes) for inst in midi_file.instruments 
                         if not inst.is_drum)
        
        drum_tracks = sum(1 for inst in midi_file.instruments if inst.is_drum)
        melodic_tracks = len(midi_file.instruments) - drum_tracks
        
        duration = midi_file.get_end_time()
        tempo = midi_file.estimate_tempo()
        
        # Calculate note density
        notes_per_second = total_notes / duration if duration > 0 else 0
        
        # Find pitch range
        all_pitches = []
        for inst in midi_file.instruments:
            if not inst.is_drum:
                all_pitches.extend([note.pitch for note in inst.notes])
        
        min_pitch = min(all_pitches) if all_pitches else 0
        max_pitch = max(all_pitches) if all_pitches else 0
        
        # Check for extremely short notes (that survived preprocessing)
        tempo_safe = tempo if tempo > 0 else 120.0
        min_duration = (4.0 / 64) / (tempo_safe / 60.0)
        short_notes = 0
        
        for inst in midi_file.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    if (note.end - note.start) < min_duration * 1.1:  # Small tolerance
                        short_notes += 1
        
        return {
            'total_notes': total_notes,
            'drum_tracks': drum_tracks,
            'melodic_tracks': melodic_tracks,
            'duration_seconds': duration,
            'tempo_bpm': tempo,
            'notes_per_second': notes_per_second,
            'pitch_range': (min_pitch, max_pitch),
            'pitch_span': max_pitch - min_pitch if all_pitches else 0,
            'short_notes_remaining': short_notes,
            'time_signatures': len(midi_file.time_signature_changes)
        }
    
    def process_midi_file(self, file_path: str, 
                         quantize_grid: Optional[int] = None,
                         remove_empty: bool = True) -> Tuple[bool, pretty_midi.PrettyMIDI, Dict]:
        """
        Complete preprocessing pipeline for a MIDI file.
        
        Args:
            file_path: Path to the MIDI file
            quantize_grid: Grid for quantization (None to skip)
            remove_empty: Whether to remove empty tracks
        
        Returns:
            Tuple of (success, processed_midi, statistics)
        """
        try:
            if self.verbose:
                print(f"\nProcessing: {file_path}")
            
            # Load MIDI file
            midi_file = pretty_midi.PrettyMIDI(file_path)
            
            # Get initial stats
            initial_stats = self.get_preprocessing_stats(midi_file)
            if self.verbose:
                print(f"Initial: {initial_stats['total_notes']} notes, "
                      f"{initial_stats['melodic_tracks']} melodic tracks")
            
            # Apply preprocessing steps
            midi_file = self.preprocess_notes(midi_file)
            
            if quantize_grid is not None:
                midi_file = self.quantize_midi_timing(midi_file, quantize_grid)
            
            if remove_empty:
                midi_file = self.remove_empty_tracks(midi_file)
            
            # Get final stats
            final_stats = self.get_preprocessing_stats(midi_file)
            final_stats['preprocessing_applied'] = {
                'quantization': quantize_grid is not None,
                'quantize_grid': quantize_grid,
                'empty_tracks_removed': remove_empty,
                'notes_removed': initial_stats['total_notes'] - final_stats['total_notes']
            }
            
            if self.verbose:
                print(f"Final: {final_stats['total_notes']} notes, "
                      f"{final_stats['melodic_tracks']} melodic tracks")
            
            return True, midi_file, final_stats
            
        except Exception as e:
            if self.verbose:
                print(f"Error processing {file_path}: {str(e)}")
            return False, None, {'error': str(e)}


# Example usage and testing
def test_preprocessing():
    """
    Example function showing how to use the preprocessor.
    """
    # Initialize preprocessor
    preprocessor = MIDIPreprocessor(verbose=True)
    
    # Process a single file
    file_path = "example.mid"  # Replace with your MIDI file path
    
    # Process with 1/16 note quantization
    success, processed_midi, stats = preprocessor.process_midi_file(
        file_path,
        quantize_grid=16,  # 1/16 note quantization
        remove_empty=True
    )
    
    if success:
        print("\nProcessing successful!")
        print(f"Statistics: {stats}")
        
        # Save processed file
        output_path = "processed_" + file_path
        processed_midi.write(output_path)
        print(f"Saved to: {output_path}")
    else:
        print(f"Processing failed: {stats.get('error', 'Unknown error')}")


# Batch processing example
def batch_process_midi_files(input_folder: str, output_folder: str, 
                            quantize_grid: Optional[int] = 16):
    """
    Process multiple MIDI files in batch.
    
    Args:
        input_folder: Folder containing MIDI files
        output_folder: Folder to save processed files
        quantize_grid: Quantization grid (None to skip)
    """
    import os
    from pathlib import Path
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    preprocessor = MIDIPreprocessor(verbose=False)
    
    success_count = 0
    fail_count = 0
    
    # Find all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(Path(input_folder).glob(ext))
    
    print(f"Found {len(midi_files)} MIDI files to process")
    
    for midi_path in midi_files:
        success, processed_midi, stats = preprocessor.process_midi_file(
            str(midi_path),
            quantize_grid=quantize_grid,
            remove_empty=True
        )
        
        if success:
            output_path = Path(output_folder) / midi_path.name
            processed_midi.write(str(output_path))
            success_count += 1
            print(f"✓ {midi_path.name} - {stats['total_notes']} notes")
        else:
            fail_count += 1
            print(f"✗ {midi_path.name} - {stats.get('error', 'Unknown error')}")
    
    print(f"\nBatch processing complete: {success_count} successful, {fail_count} failed")


if __name__ == "__main__":
    # Uncomment to test
    # test_preprocessing()
    pass