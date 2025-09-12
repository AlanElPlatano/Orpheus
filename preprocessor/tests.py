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
            quantize_grid=None,  # 1/16 note quantization
            remove_empty=True
        )

        if success:
            # Create output filename with _processed suffix
            input_stem = midi_path.stem
            input_suffix = midi_path.suffix
            output_file = f"{input_stem}_processed{input_suffix}"
            output_path = processed_dir / output_file

            processed_midi.write(str(output_path))
            success_count += 1
            print(f"✓ {midi_path.name} - {stats['total_notes']} notes - Saved to: {output_path}")
        else:
            fail_count += 1
            print(f"✗ {midi_path.name} - {stats.get('error', 'Unknown error')}")

    print(f"\nBatch processing complete: {success_count} successful, {fail_count} failed")


if __name__ == "__main__":
	test_preprocessing()
	pass