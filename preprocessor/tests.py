from typing import Optional
from pathlib import Path
import os

from preprocessor import MIDIPreprocessor


def test_preprocessing():
    """
    Example function showing how to use the preprocessor.
    """
    # Initialize preprocessor
    preprocessor = MIDIPreprocessor(verbose=True)

    # Process a single file
    # This repo includes a sample MIDI file sourced from Wikipedia commons
    # https://en.wikipedia.org/wiki/File:MIDI_sample.mid
    file_path = "example.mid"

    file_path = os.path.join(os.path.dirname(__file__), file_path)

    # Process with 1/16 note quantization
    success, processed_midi, stats = preprocessor.process_midi_file(
        file_path,
        quantize_grid=None,  # 1/16 note quantization
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


def batch_process_midi_files(input_folder: str, output_folder: str, quantize_grid: Optional[int] = 16):
    """
    Process multiple MIDI files in batch.

    Args:
        input_folder: Folder containing MIDI files
        output_folder: Folder to save processed files
        quantize_grid: Quantization grid (None to skip)
    """
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
	# Uncomment to run examples
	# test_preprocessing()
	pass


