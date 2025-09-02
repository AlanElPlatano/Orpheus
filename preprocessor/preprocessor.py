import pretty_midi
from typing import Optional, Tuple, Dict

from quantize_midi_timing import quantize_midi_timing
from preprocess_notes import preprocess_notes
from remove_empty_tracks import remove_empty_tracks
from get_preprocessing_stats import get_preprocessing_stats


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

    def quantize_midi_timing(self, midi_file: pretty_midi.PrettyMIDI, quantize_grid: Optional[int] = None) -> pretty_midi.PrettyMIDI:
        return quantize_midi_timing(midi_file, quantize_grid=quantize_grid, verbose=self.verbose)

    def preprocess_notes(self, midi_file: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        return preprocess_notes(midi_file, verbose=self.verbose)

    def remove_empty_tracks(self, midi_file: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        return remove_empty_tracks(midi_file, verbose=self.verbose)

    def get_preprocessing_stats(self, midi_file: pretty_midi.PrettyMIDI) -> Dict:
        return get_preprocessing_stats(midi_file)

    def process_midi_file(self, file_path: str, quantize_grid: Optional[int] = None, remove_empty: bool = True) -> Tuple[bool, pretty_midi.PrettyMIDI, Dict]:
        from process_midi_file import process_midi_file as _process
        return _process(file_path, quantize_grid=quantize_grid, remove_empty=remove_empty, verbose=self.verbose)


