from typing import Optional
from pathlib import Path
import os

from preprocessor import MIDIPreprocessor


def test_preprocessing():
    """
    Example function showing how to use the preprocessor.
    Processes all MIDI files in the source_midis folder.
    """
    # Initialize preprocessor
    preprocessor = MIDIPreprocessor(verbose=True)

    # Declaration of the folder with the source MIDI files
    file_path = Path(__file__).resolve().parent
    file_path = file_path.parent
    file_path = file_path / "source_midis"

    # Check if the folder exists
    if not os.path.isdir(file_path):
        return "The source midi folder doesn't exist yet, nothing to process"

    # Create output folder if it doesn't exist
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    processed_dir = parent_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    # Find all MIDI files
    midi_files = set()
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.update(file_path.glob(ext))
    midi_files = list(midi_files)

    print(f"Found {len(midi_files)} MIDI files to process")

    for midi_path in midi_files:
        success, processed_midi, stats = preprocessor.process_midi_file(
            str(midi_path),
            quantize_grid=None,  # Set to 16 for 1/16 note quantization
            remove_empty=True,
            remove_bass=True,  # Enable bass track removal
            bass_threshold=36  # Remove tracks primarily below C2 (MIDI note 36)
        )

        if success:
            # Create output filename with _processed suffix
            input_stem = midi_path.stem
            input_suffix = midi_path.suffix
            output_file = f"{input_stem}_processed{input_suffix}"
            output_path = processed_dir / output_file

            processed_midi.write(str(output_path))
            success_count += 1
            
            # Display processing results
            preprocessing_info = stats.get('preprocessing_applied', {})
            print(f"✓ {midi_path.name}")
            print(f"  - Notes: {stats['total_notes']}")
            print(f"  - Tempo: {stats['tempo_bpm']:.1f} BPM")
            print(f"  - Tracks: {stats['melodic_tracks']} melodic, {stats['drum_tracks']} drum")
            if preprocessing_info.get('tracks_removed', 0) > 0:
                print(f"  - Removed: {preprocessing_info['tracks_removed']} tracks")
            print(f"  - Saved to: {output_path.name}")
        else:
            fail_count += 1
            print(f"✗ {midi_path.name} - {stats.get('error', 'Unknown error')}")

    print(f"\nBatch processing complete: {success_count} successful, {fail_count} failed")


def test_bass_removal_only():
    """
    Test function specifically for bass track removal feature.
    Processes only the 'example.mid' file from the source_midis folder.
    """
    from preprocessor import MIDIPreprocessor
    import pretty_midi
    
    preprocessor = MIDIPreprocessor(verbose=True)
    
    # Get path to source_midis folder
    file_path = Path(__file__).resolve().parent
    file_path = file_path.parent
    file_path = file_path / "source_midis" / "example.mid"
    
    if not os.path.exists(file_path):
        print(f"Example file not found: {file_path}")
        print("Please ensure 'example.mid' exists in the source_midis folder")
        return
    
    print(f"Testing bass removal on: {file_path.name}\n")
    print("Testing with different thresholds:\n")
    
    # Create output folder for bass removal tests
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    bass_test_dir = parent_dir / "processed" / "bass_removal_tests"
    bass_test_dir.mkdir(parents=True, exist_ok=True)
    
    for threshold in [36, 48, 60]:  # C2, C3, C4
        note_names = {36: "C2", 48: "C3", 60: "C4"}
        print(f"\n--- Testing with threshold: {note_names[threshold]} (MIDI note {threshold}) ---")
        
        success, processed_midi, stats = preprocessor.process_midi_file(
            str(file_path),
            quantize_grid=None,
            remove_empty=True,
            remove_bass=True,
            bass_threshold=threshold
        )
        
        if success:
            # Save with threshold in filename
            output_file = f"example_bass_threshold_{threshold}.mid"
            output_path = bass_test_dir / output_file
            processed_midi.write(str(output_path))
            
            preprocessing = stats.get('preprocessing_applied', {})
            print(f"  Tracks remaining: {stats['melodic_tracks']}")
            print(f"  Notes remaining: {stats['total_notes']}")
            print(f"  Tracks removed: {preprocessing.get('tracks_removed', 0)}")
            print(f"  Saved to: {output_path.name}")
        else:
            print(f"  Error: {stats.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_preprocessing()
    # Uncomment to test bass removal specifically:
    # test_bass_removal_only()