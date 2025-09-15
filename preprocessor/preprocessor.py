import pretty_midi
from typing import Optional, Tuple, Dict

from bpm_reader import get_tempo_from_midi
from quantizer import quantize_midi_timing
from cleanup import preprocess_notes
from remove_empty_tracks import remove_empty_tracks
from remove_bass_tracks import remove_bass_tracks
from stats import get_preprocessing_stats


class MIDIPreprocessor:
    """
    MIDI file preprocessing pipeline for cleaning and preparing MIDI files
    for AI training. Handles quantization, note cleanup, and bass track removal.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the MIDI preprocessor.

        Args:
            verbose: If True, print detailed processing information
        """
        self.verbose = verbose
        self._cached_tempo = None  # Cache tempo to avoid recalculation

    def get_tempo(self, midi_file: pretty_midi.PrettyMIDI) -> float:
        """
        Get tempo from MIDI file, using cache if available.
        
        Args:
            midi_file: PrettyMIDI object
            
        Returns:
            Tempo in BPM
        """
        if self._cached_tempo is None:
            self._cached_tempo = get_tempo_from_midi(midi_file)
        return self._cached_tempo

    def clear_tempo_cache(self):
        """Clear the cached tempo value."""
        self._cached_tempo = None

    def quantize_midi_timing(
        self, 
        midi_file: pretty_midi.PrettyMIDI, 
        quantize_grid: Optional[int] = None,
        tempo_bpm: Optional[float] = None
    ) -> pretty_midi.PrettyMIDI:
        """Quantize MIDI timing with optional pre-calculated tempo."""
        if tempo_bpm is None:
            tempo_bpm = self.get_tempo(midi_file)
        return quantize_midi_timing(
            midi_file, 
            quantize_grid=quantize_grid, 
            tempo_bpm=tempo_bpm,
            verbose=self.verbose
        )

    def preprocess_notes(
        self, 
        midi_file: pretty_midi.PrettyMIDI,
        tempo_bpm: Optional[float] = None
    ) -> pretty_midi.PrettyMIDI:
        """Preprocess notes with optional pre-calculated tempo."""
        if tempo_bpm is None:
            tempo_bpm = self.get_tempo(midi_file)
        return preprocess_notes(midi_file, tempo_bpm=tempo_bpm, verbose=self.verbose)

    def remove_empty_tracks(self, midi_file: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Remove empty tracks from MIDI file."""
        return remove_empty_tracks(midi_file, verbose=self.verbose)

    def remove_bass_tracks(
        self, 
        midi_file: pretty_midi.PrettyMIDI,
        threshold_note: int = 36
    ) -> pretty_midi.PrettyMIDI:
        """
        Remove bass tracks based on note threshold.
        
        Args:
            midi_file: PrettyMIDI object
            threshold_note: MIDI note threshold (default 36 = C2)
            
        Returns:
            PrettyMIDI object with bass tracks removed
        """
        return remove_bass_tracks(
            midi_file, 
            threshold_note=threshold_note, 
            verbose=self.verbose
        )

    def get_preprocessing_stats(
        self, 
        midi_file: pretty_midi.PrettyMIDI,
        tempo_bpm: Optional[float] = None
    ) -> Dict:
        """Get preprocessing statistics with optional pre-calculated tempo."""
        if tempo_bpm is None:
            tempo_bpm = self.get_tempo(midi_file)
        return get_preprocessing_stats(midi_file, tempo_bpm=tempo_bpm)

    def process_midi_file(
        self, 
        file_path: str, 
        quantize_grid: Optional[int] = None, 
        remove_empty: bool = True,
        remove_bass: bool = False,
        bass_threshold: int = 36
    ) -> Tuple[bool, pretty_midi.PrettyMIDI, Dict]:
        """
        Complete preprocessing pipeline for a MIDI file.
        
        Args:
            file_path: Path to the MIDI file
            quantize_grid: Grid for quantization (None to skip)
            remove_empty: Whether to remove empty tracks
            remove_bass: Whether to remove bass tracks
            bass_threshold: MIDI note threshold for bass removal (default 36 = C2)
            
        Returns:
            Tuple of (success, processed_midi, statistics)
        """
        # Clear tempo cache for new file
        self.clear_tempo_cache()
        
        from process_midi_file import process_midi_file as _process
        return _process(
            file_path, 
            quantize_grid=quantize_grid, 
            remove_empty=remove_empty,
            remove_bass=remove_bass,
            bass_threshold=bass_threshold,
            verbose=self.verbose
        )