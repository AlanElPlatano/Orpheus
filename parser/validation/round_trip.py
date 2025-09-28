"""
Round-trip validation module for MIDI tokenization.

This module provides comprehensive validation by tokenizing MIDI files,
detokenizing them back, and comparing the results to ensure fidelity
within specified tolerances.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import time
from collections import defaultdict
import numpy as np

# MidiTok imports for detokenization
try:
    from miditok import MIDITokenizer
    MIDITOK_AVAILABLE = True
except ImportError:
    MIDITOK_AVAILABLE = False
    logging.warning("MidiTok not installed. Round-trip validation unavailable.")

from miditoolkit import MidiFile, Note, Instrument

from parser.config.defaults import (
    MidiParserConfig,
    ValidationConfig,
    DEFAULT_CONFIG
)
from parser.core.midi_loader import (
    MidiMetadata,
    ValidationResult,
    extract_metadata,
    clean_midi_data
)
from parser.core.tokenizer_manager import (
    TokenizerManager,
    TokenizationResult
)
from parser.core.track_analyzer import TrackInfo

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class NoteComparison:
    """Detailed comparison of a single note."""
    original_note: Note
    reconstructed_note: Optional[Note]
    pitch_match: bool = True
    velocity_diff: int = 0
    start_diff: int = 0
    duration_diff: int = 0
    is_missing: bool = False
    is_extra: bool = False
    
    @property
    def is_within_tolerance(self) -> bool:
        """Check if differences are within tolerance."""
        return (self.pitch_match and 
                not self.is_missing and 
                not self.is_extra)


@dataclass
class TrackComparison:
    """Comparison results for a single track."""
    track_index: int
    track_name: str
    original_note_count: int
    reconstructed_note_count: int
    note_comparisons: List[NoteComparison] = field(default_factory=list)
    missing_notes: int = 0
    extra_notes: int = 0
    timing_errors: int = 0
    velocity_errors: int = 0
    program_match: bool = True
    is_drum_match: bool = True
    
    @property
    def accuracy_score(self) -> float:
        """Calculate accuracy score for the track."""
        if self.original_note_count == 0:
            return 1.0 if self.reconstructed_note_count == 0 else 0.0
        
        correct_notes = sum(1 for nc in self.note_comparisons 
                           if nc.is_within_tolerance)
        return correct_notes / max(self.original_note_count, 
                                  self.reconstructed_note_count)


@dataclass
class RoundTripMetrics:
    """Complete metrics from round-trip validation."""
    total_notes_original: int = 0
    total_notes_reconstructed: int = 0
    missing_notes: int = 0
    extra_notes: int = 0
    timing_errors: int = 0
    duration_errors: int = 0
    velocity_errors: int = 0
    pitch_errors: int = 0
    
    # Aggregate metrics
    missing_notes_ratio: float = 0.0
    extra_notes_ratio: float = 0.0
    timing_accuracy: float = 1.0
    velocity_accuracy: float = 1.0
    overall_accuracy: float = 1.0
    
    # Timing statistics
    mean_start_diff: float = 0.0
    max_start_diff: int = 0
    mean_duration_diff: float = 0.0
    max_duration_diff: int = 0
    
    # Velocity statistics
    mean_velocity_diff: float = 0.0
    max_velocity_diff: int = 0
    
    # Track comparisons
    track_comparisons: List[TrackComparison] = field(default_factory=list)
    
    # Processing info
    tokenization_strategy: str = "REMI"
    processing_time: float = 0.0
    token_count: int = 0
    
    def calculate_aggregates(self) -> None:
        """Calculate aggregate metrics from detailed comparisons."""
        if self.total_notes_original > 0:
            self.missing_notes_ratio = self.missing_notes / self.total_notes_original
            self.extra_notes_ratio = self.extra_notes / max(self.total_notes_original, 1)
            
            correct_notes = self.total_notes_original - self.missing_notes
            self.timing_accuracy = 1.0 - (self.timing_errors / max(self.total_notes_original, 1))
            self.velocity_accuracy = 1.0 - (self.velocity_errors / max(self.total_notes_original, 1))
            
            # Overall accuracy
            total_errors = (self.missing_notes + self.extra_notes + 
                          self.timing_errors + self.velocity_errors + self.pitch_errors)
            max_possible_errors = self.total_notes_original * 4  # 4 aspects per note
            self.overall_accuracy = 1.0 - (total_errors / max(max_possible_errors, 1))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "notes": {
                "original": self.total_notes_original,
                "reconstructed": self.total_notes_reconstructed,
                "missing": self.missing_notes,
                "extra": self.extra_notes,
                "missing_ratio": round(self.missing_notes_ratio, 4),
                "extra_ratio": round(self.extra_notes_ratio, 4)
            },
            "errors": {
                "timing": self.timing_errors,
                "duration": self.duration_errors,
                "velocity": self.velocity_errors,
                "pitch": self.pitch_errors
            },
            "accuracy": {
                "overall": round(self.overall_accuracy, 4),
                "timing": round(self.timing_accuracy, 4),
                "velocity": round(self.velocity_accuracy, 4)
            },
            "timing_stats": {
                "mean_start_diff": round(self.mean_start_diff, 2),
                "max_start_diff": self.max_start_diff,
                "mean_duration_diff": round(self.mean_duration_diff, 2),
                "max_duration_diff": self.max_duration_diff
            },
            "velocity_stats": {
                "mean_diff": round(self.mean_velocity_diff, 2),
                "max_diff": self.max_velocity_diff
            },
            "processing": {
                "strategy": self.tokenization_strategy,
                "token_count": self.token_count,
                "time_seconds": round(self.processing_time, 3)
            }
        }


# ============================================================================
# Round-Trip Validator Class
# ============================================================================

class RoundTripValidator:
    """
    Validates MIDI tokenization fidelity through round-trip conversion.
    
    This class implements the validation system specified in Section 5,
    ensuring that tokenization→detokenization preserves musical content
    within acceptable tolerances.
    """
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize the round-trip validator.
        
        Args:
            config: Parser configuration with validation tolerances
        """
        if not MIDITOK_AVAILABLE:
            raise ImportError("MidiTok is required for round-trip validation")
        
        self.config = config or DEFAULT_CONFIG
        self.validation_config = self.config.validation
        self.tokenizer_manager = TokenizerManager(self.config)
        
        # Cache for tokenizers
        self._tokenizer_cache = {}
        
    def validate_round_trip(
        self,
        midi: MidiFile,
        strategy: Optional[str] = None,
        track_infos: Optional[List[TrackInfo]] = None,
        detailed_report: bool = True
    ) -> Tuple[ValidationResult, RoundTripMetrics]:
        """
        Perform complete round-trip validation on a MIDI file.
        
        This is the main validation function referenced in Section 5.
        
        Args:
            midi: Original MIDI file
            strategy: Tokenization strategy to test
            track_infos: Optional track analysis information
            detailed_report: Whether to generate detailed comparison data
            
        Returns:
            Tuple of (ValidationResult, RoundTripMetrics)
        """
        start_time = time.time()
        
        strategy = strategy or self.config.tokenization
        validation_result = ValidationResult(is_valid=True)
        metrics = RoundTripMetrics(tokenization_strategy=strategy)
        
        try:
            # Step 1: Tokenize the original MIDI
            logger.info(f"Starting round-trip validation with {strategy}")
            tokenization_result = self._tokenize_midi(midi, strategy, track_infos)
            
            if not tokenization_result.success:
                validation_result.add_error(f"Tokenization failed: {tokenization_result.error_message}")
                return validation_result, metrics
            
            metrics.token_count = len(tokenization_result.tokens)
            
            # Step 2: Detokenize back to MIDI
            reconstructed_midi = self._detokenize_tokens(
                tokenization_result.tokens,
                strategy,
                midi.ticks_per_beat
            )
            
            if reconstructed_midi is None:
                validation_result.add_error("Detokenization failed")
                return validation_result, metrics
            
            # Step 3: Compare original and reconstructed MIDI
            metrics = self._compare_midi_files(
                midi,
                reconstructed_midi,
                strategy,
                detailed_report
            )
            
            # Step 4: Apply tolerance checks
            validation_result = self._apply_tolerance_checks(metrics)
            
            # Calculate processing time
            metrics.processing_time = time.time() - start_time
            
            # Log summary
            if validation_result.is_valid:
                logger.info(f"Round-trip validation PASSED for {strategy} "
                          f"(accuracy: {metrics.overall_accuracy:.2%})")
            else:
                logger.warning(f"Round-trip validation FAILED for {strategy}: "
                             f"{validation_result.errors}")
            
        except Exception as e:
            logger.error(f"Round-trip validation error: {e}")
            validation_result.add_error(f"Validation error: {str(e)}")
            metrics.processing_time = time.time() - start_time
        
        return validation_result, metrics
    
    def _tokenize_midi(
        self,
        midi: MidiFile,
        strategy: str,
        track_infos: Optional[List[TrackInfo]] = None
    ) -> TokenizationResult:
        """
        Tokenize MIDI file using specified strategy.
        
        Args:
            midi: MIDI file to tokenize
            strategy: Tokenization strategy
            track_infos: Optional track information
            
        Returns:
            TokenizationResult with tokens
        """
        return self.tokenizer_manager.tokenize_midi(
            midi,
            strategy=strategy,
            track_infos=track_infos,
            max_seq_length=None  # No truncation for validation
        )
    
    def _detokenize_tokens(
        self,
        tokens: List[int],
        strategy: str,
        ticks_per_beat: int = 480
    ) -> Optional[MidiFile]:
        """
        Detokenize tokens back to MIDI format.
        
        Args:
            tokens: Token sequence
            strategy: Tokenization strategy used
            ticks_per_beat: PPQ for the MIDI file
            
        Returns:
            Reconstructed MidiFile or None if failed
        """
        try:
            # Get or create tokenizer
            tokenizer = self.tokenizer_manager.create_tokenizer(strategy)
            
            # MidiTok detokenization
            # The exact method depends on MidiTok version and strategy
            if hasattr(tokenizer, 'tokens_to_midi'):
                # Newer MidiTok versions
                reconstructed = tokenizer.tokens_to_midi(tokens)
            elif hasattr(tokenizer, 'detokenize'):
                # Alternative method name
                reconstructed = tokenizer.detokenize(tokens)
            else:
                # Fallback: try calling tokenizer directly
                reconstructed = tokenizer(tokens, _=None)  # Some versions use this
            
            # Ensure we have a MidiFile object
            if isinstance(reconstructed, MidiFile):
                return reconstructed
            elif hasattr(reconstructed, 'to_midi'):
                return reconstructed.to_midi()
            else:
                logger.error(f"Unexpected detokenization output type: {type(reconstructed)}")
                return None
                
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            return None
    
    def _compare_midi_files(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        strategy: str,
        detailed: bool = True
    ) -> RoundTripMetrics:
        """
        Compare original and reconstructed MIDI files.
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file
            strategy: Tokenization strategy used
            detailed: Whether to generate detailed comparisons
            
        Returns:
            RoundTripMetrics with comparison results
        """
        metrics = RoundTripMetrics(tokenization_strategy=strategy)
        
        # Compare global properties
        self._compare_global_properties(original, reconstructed, metrics)
        
        # Compare tracks
        metrics.track_comparisons = self._compare_tracks(
            original,
            reconstructed,
            detailed
        )
        
        # Aggregate statistics
        self._calculate_aggregate_metrics(metrics)
        
        return metrics
    
    def _compare_global_properties(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        metrics: RoundTripMetrics
    ) -> None:
        """
        Compare global MIDI properties (tempo, time signatures).
        
        Args:
            original: Original MIDI
            reconstructed: Reconstructed MIDI
            metrics: Metrics object to update
        """
        # Compare tempo changes
        tolerances = self.validation_config.tolerances
        
        if len(original.tempo_changes) != len(reconstructed.tempo_changes):
            logger.warning(f"Tempo change count mismatch: "
                         f"{len(original.tempo_changes)} vs {len(reconstructed.tempo_changes)}")
        
        for i, orig_tempo in enumerate(original.tempo_changes):
            if i < len(reconstructed.tempo_changes):
                recon_tempo = reconstructed.tempo_changes[i]
                tempo_diff = abs(orig_tempo.tempo - recon_tempo.tempo)
                
                if tempo_diff > tolerances.get('tempo_bpm_diff', 1.0):
                    logger.warning(f"Tempo difference at tick {orig_tempo.time}: "
                                 f"{tempo_diff:.1f} BPM")
        
        # Compare time signatures
        if len(original.time_signature_changes) != len(reconstructed.time_signature_changes):
            logger.warning(f"Time signature change count mismatch")
    
    def _compare_tracks(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        detailed: bool
    ) -> List[TrackComparison]:
        """
        Compare individual tracks between MIDI files.
        
        Args:
            original: Original MIDI
            reconstructed: Reconstructed MIDI
            detailed: Whether to generate detailed note comparisons
            
        Returns:
            List of TrackComparison objects
        """
        comparisons = []
        tolerances = self.validation_config.tolerances
        
        # Match tracks between original and reconstructed
        track_pairs = self._match_tracks(original.instruments, reconstructed.instruments)
        
        for orig_idx, recon_idx in track_pairs:
            if orig_idx is None:
                # Extra track in reconstructed
                recon_track = reconstructed.instruments[recon_idx]
                comparison = TrackComparison(
                    track_index=recon_idx,
                    track_name=recon_track.name or f"Track_{recon_idx}",
                    original_note_count=0,
                    reconstructed_note_count=len(recon_track.notes),
                    extra_notes=len(recon_track.notes)
                )
                comparisons.append(comparison)
                continue
            
            orig_track = original.instruments[orig_idx]
            
            if recon_idx is None:
                # Missing track in reconstructed
                comparison = TrackComparison(
                    track_index=orig_idx,
                    track_name=orig_track.name or f"Track_{orig_idx}",
                    original_note_count=len(orig_track.notes),
                    reconstructed_note_count=0,
                    missing_notes=len(orig_track.notes)
                )
                comparisons.append(comparison)
                continue
            
            recon_track = reconstructed.instruments[recon_idx]
            
            # Compare the matched tracks
            comparison = TrackComparison(
                track_index=orig_idx,
                track_name=orig_track.name or f"Track_{orig_idx}",
                original_note_count=len(orig_track.notes),
                reconstructed_note_count=len(recon_track.notes),
                program_match=(orig_track.program == recon_track.program),
                is_drum_match=(orig_track.is_drum == recon_track.is_drum)
            )
            
            if detailed:
                # Detailed note-by-note comparison
                note_comparisons = self._compare_notes(
                    orig_track.notes,
                    recon_track.notes,
                    tolerances
                )
                comparison.note_comparisons = note_comparisons
                
                # Count errors
                for nc in note_comparisons:
                    if nc.is_missing:
                        comparison.missing_notes += 1
                    elif nc.is_extra:
                        comparison.extra_notes += 1
                    else:
                        if abs(nc.start_diff) > tolerances.get('note_start_tick', 1):
                            comparison.timing_errors += 1
                        if abs(nc.velocity_diff) > tolerances.get('velocity_bin', 1):
                            comparison.velocity_errors += 1
            else:
                # Quick comparison
                comparison.missing_notes = max(0, len(orig_track.notes) - len(recon_track.notes))
                comparison.extra_notes = max(0, len(recon_track.notes) - len(orig_track.notes))
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _match_tracks(
        self,
        original_tracks: List[Instrument],
        reconstructed_tracks: List[Instrument]
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Match tracks between original and reconstructed MIDI.
        
        Args:
            original_tracks: Original instrument tracks
            reconstructed_tracks: Reconstructed instrument tracks
            
        Returns:
            List of (original_index, reconstructed_index) pairs
        """
        pairs = []
        used_recon = set()
        
        # Try to match by program and drum status first
        for i, orig_track in enumerate(original_tracks):
            best_match = None
            best_score = 0
            
            for j, recon_track in enumerate(reconstructed_tracks):
                if j in used_recon:
                    continue
                
                # Calculate similarity score
                score = 0
                if orig_track.is_drum == recon_track.is_drum:
                    score += 10
                if orig_track.program == recon_track.program:
                    score += 5
                if orig_track.name and recon_track.name:
                    if orig_track.name.lower() == recon_track.name.lower():
                        score += 3
                
                # Compare note count similarity
                if orig_track.notes and recon_track.notes:
                    note_ratio = min(len(orig_track.notes), len(recon_track.notes)) / \
                                max(len(orig_track.notes), len(recon_track.notes))
                    score += note_ratio * 2
                
                if score > best_score:
                    best_score = score
                    best_match = j
            
            if best_match is not None and best_score > 5:
                pairs.append((i, best_match))
                used_recon.add(best_match)
            else:
                pairs.append((i, None))  # No match found
        
        # Add any unmatched reconstructed tracks
        for j in range(len(reconstructed_tracks)):
            if j not in used_recon:
                pairs.append((None, j))
        
        return pairs
    
    def _compare_notes(
        self,
        original_notes: List[Note],
        reconstructed_notes: List[Note],
        tolerances: Dict[str, Union[int, float]]
    ) -> List[NoteComparison]:
        """
        Compare individual notes between tracks.
        
        Args:
            original_notes: Original track notes
            reconstructed_notes: Reconstructed track notes
            tolerances: Tolerance thresholds
            
        Returns:
            List of NoteComparison objects
        """
        comparisons = []
        
        # Sort notes by start time and pitch for matching
        orig_sorted = sorted(original_notes, key=lambda n: (n.start, n.pitch))
        recon_sorted = sorted(reconstructed_notes, key=lambda n: (n.start, n.pitch))
        
        # Create index for faster matching
        recon_index = self._build_note_index(recon_sorted)
        matched_recon = set()
        
        # Match each original note
        for orig_note in orig_sorted:
            # Find best matching reconstructed note
            best_match = self._find_best_note_match(
                orig_note,
                recon_index,
                matched_recon,
                tolerances
            )
            
            if best_match is not None:
                # Found a match
                recon_note = recon_sorted[best_match]
                matched_recon.add(best_match)
                
                comparison = NoteComparison(
                    original_note=orig_note,
                    reconstructed_note=recon_note,
                    pitch_match=(orig_note.pitch == recon_note.pitch),
                    velocity_diff=abs(orig_note.velocity - recon_note.velocity),
                    start_diff=abs(orig_note.start - recon_note.start),
                    duration_diff=abs((orig_note.end - orig_note.start) - 
                                    (recon_note.end - recon_note.start))
                )
            else:
                # Missing note
                comparison = NoteComparison(
                    original_note=orig_note,
                    reconstructed_note=None,
                    is_missing=True
                )
            
            comparisons.append(comparison)
        
        # Add extra notes (in reconstructed but not matched)
        for i, recon_note in enumerate(recon_sorted):
            if i not in matched_recon:
                comparison = NoteComparison(
                    original_note=recon_note,  # Use recon as placeholder
                    reconstructed_note=recon_note,
                    is_extra=True
                )
                comparisons.append(comparison)
        
        return comparisons
    
    def _build_note_index(self, notes: List[Note]) -> Dict[int, List[int]]:
        """
        Build index of notes by pitch for faster matching.
        
        Args:
            notes: List of notes to index
            
        Returns:
            Dictionary mapping pitch to list of note indices
        """
        index = defaultdict(list)
        for i, note in enumerate(notes):
            index[note.pitch].append(i)
        return dict(index)
    
    def _find_best_note_match(
        self,
        orig_note: Note,
        recon_index: Dict[int, List[int]],
        matched: set,
        tolerances: Dict[str, Union[int, float]]
    ) -> Optional[int]:
        """
        Find best matching reconstructed note for an original note.
        
        Args:
            orig_note: Original note to match
            recon_index: Index of reconstructed notes by pitch
            matched: Set of already matched reconstructed note indices
            tolerances: Tolerance thresholds
            
        Returns:
            Index of best matching note or None
        """
        # Get tolerance values
        start_tolerance = tolerances.get('note_start_tick', 1)
        
        # Look for notes with same pitch first
        candidates = []
        
        if orig_note.pitch in recon_index:
            for idx in recon_index[orig_note.pitch]:
                if idx not in matched:
                    candidates.append(idx)
        
        # If no same-pitch candidates, check nearby pitches (for pitch bend effects)
        if not candidates:
            for pitch_offset in [-1, 1]:
                alt_pitch = orig_note.pitch + pitch_offset
                if alt_pitch in recon_index:
                    for idx in recon_index[alt_pitch]:
                        if idx not in matched:
                            candidates.append(idx)
        
        if not candidates:
            return None
        
        # Find best match based on timing
        best_idx = None
        best_score = float('inf')
        
        for idx in candidates:
            # Calculate matching score (lower is better)
            # This is a simple distance metric
            time_diff = abs(orig_note.start - recon_index[orig_note.pitch][0])
            
            if time_diff <= start_tolerance * 10:  # Reasonable search window
                score = time_diff
                if score < best_score:
                    best_score = score
                    best_idx = idx
        
        return best_idx
    
    def _calculate_aggregate_metrics(self, metrics: RoundTripMetrics) -> None:
        """
        Calculate aggregate metrics from track comparisons.
        
        Args:
            metrics: RoundTripMetrics object to update
        """
        # Sum up totals from track comparisons
        for track_comp in metrics.track_comparisons:
            metrics.total_notes_original += track_comp.original_note_count
            metrics.total_notes_reconstructed += track_comp.reconstructed_note_count
            metrics.missing_notes += track_comp.missing_notes
            metrics.extra_notes += track_comp.extra_notes
            metrics.timing_errors += track_comp.timing_errors
            metrics.velocity_errors += track_comp.velocity_errors
            
            # Collect detailed statistics if available
            if track_comp.note_comparisons:
                start_diffs = []
                duration_diffs = []
                velocity_diffs = []
                
                for nc in track_comp.note_comparisons:
                    if not nc.is_missing and not nc.is_extra:
                        start_diffs.append(nc.start_diff)
                        duration_diffs.append(nc.duration_diff)
                        velocity_diffs.append(nc.velocity_diff)
                        
                        if nc.duration_diff > metrics.max_duration_diff:
                            metrics.max_duration_diff = nc.duration_diff
                        if nc.start_diff > metrics.max_start_diff:
                            metrics.max_start_diff = nc.start_diff
                        if nc.velocity_diff > metrics.max_velocity_diff:
                            metrics.max_velocity_diff = nc.velocity_diff
                
                # Calculate means
                if start_diffs:
                    metrics.mean_start_diff = np.mean(start_diffs)
                if duration_diffs:
                    metrics.mean_duration_diff = np.mean(duration_diffs)
                if velocity_diffs:
                    metrics.mean_velocity_diff = np.mean(velocity_diffs)
        
        # Calculate aggregate ratios and scores
        metrics.calculate_aggregates()
    
    def _apply_tolerance_checks(self, metrics: RoundTripMetrics) -> ValidationResult:
        """
        Apply tolerance thresholds to determine pass/fail.
        
        Args:
            metrics: Calculated metrics
            
        Returns:
            ValidationResult with pass/fail status
        """
        result = ValidationResult(is_valid=True)
        tolerances = self.validation_config.tolerances
        
        # Check missing notes ratio
        if metrics.missing_notes_ratio > tolerances.get('missing_notes_ratio', 0.01):
            result.add_error(f"Missing notes ratio {metrics.missing_notes_ratio:.3f} exceeds "
                           f"tolerance {tolerances['missing_notes_ratio']}")
        
        # Check extra notes ratio
        if metrics.extra_notes_ratio > tolerances.get('extra_notes_ratio', 0.01):
            result.add_error(f"Extra notes ratio {metrics.extra_notes_ratio:.3f} exceeds "
                           f"tolerance {tolerances['extra_notes_ratio']}")
        
        # Check timing accuracy
        if metrics.max_start_diff > tolerances.get('note_start_tick', 1):
            result.add_warning(f"Maximum start time difference {metrics.max_start_diff} exceeds "
                             f"tolerance {tolerances['note_start_tick']}")
        
        # Check duration accuracy
        if metrics.max_duration_diff > tolerances.get('note_duration', 2):
            result.add_warning(f"Maximum duration difference {metrics.max_duration_diff} exceeds "
                             f"tolerance {tolerances['note_duration']}")
        
        # Check overall quality score
        quality_threshold = self.validation_config.quality_threshold
        if metrics.overall_accuracy < quality_threshold:
            result.add_error(f"Overall accuracy {metrics.overall_accuracy:.3f} below "
                           f"quality threshold {quality_threshold}")
        
        # Additional warnings for degraded quality
        if metrics.timing_accuracy < 0.98:
            result.add_warning(f"Timing accuracy degraded: {metrics.timing_accuracy:.3f}")
        
        if metrics.velocity_accuracy < 0.95:
            result.add_warning(f"Velocity accuracy degraded: {metrics.velocity_accuracy:.3f}")
        
        return result
    
    def validate_with_fallback(
        self,
        midi: MidiFile,
        primary_strategy: str,
        fallback_strategies: Optional[List[str]] = None
    ) -> Tuple[ValidationResult, RoundTripMetrics, str]:
        """
        Validate with automatic fallback to alternative strategies.
        
        Args:
            midi: MIDI file to validate
            primary_strategy: Primary tokenization strategy
            fallback_strategies: Alternative strategies to try
            
        Returns:
            Tuple of (ValidationResult, RoundTripMetrics, successful_strategy)
        """
        if fallback_strategies is None:
            fallback_strategies = ["REMI", "TSD", "Structured"]
            # Remove primary from fallbacks
            fallback_strategies = [s for s in fallback_strategies if s != primary_strategy]
        
        # Try primary strategy first
        logger.info(f"Attempting validation with primary strategy: {primary_strategy}")
        result, metrics = self.validate_round_trip(midi, primary_strategy)
        
        if result.is_valid:
            return result, metrics, primary_strategy
        
        # Try fallback strategies
        logger.warning(f"Primary strategy {primary_strategy} failed, trying fallbacks")
        
        for strategy in fallback_strategies:
            logger.info(f"Attempting validation with fallback strategy: {strategy}")
            result, metrics = self.validate_round_trip(midi, strategy)
            
            if result.is_valid:
                result.add_warning(f"Used fallback strategy {strategy} after {primary_strategy} failed")
                return result, metrics, strategy
        
        # All strategies failed
        logger.error(f"All strategies failed validation for MIDI file")
        return result, metrics, primary_strategy
    
    def batch_validate(
        self,
        midi_files: List[MidiFile],
        strategy: Optional[str] = None,
        parallel: bool = False,
        stop_on_failure: bool = False
    ) -> List[Tuple[ValidationResult, RoundTripMetrics]]:
        """
        Validate multiple MIDI files in batch.
        
        Args:
            midi_files: List of MIDI files to validate
            strategy: Tokenization strategy (uses config default if None)
            parallel: Whether to use parallel processing
            stop_on_failure: Whether to stop on first validation failure
            
        Returns:
            List of (ValidationResult, RoundTripMetrics) tuples
        """
        results = []
        strategy = strategy or self.config.tokenization
        
        if parallel and len(midi_files) > 1:
            import concurrent.futures
            import multiprocessing
            
            max_workers = self.config.processing.max_workers or multiprocessing.cpu_count()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit validation tasks
                futures = []
                for midi in midi_files:
                    future = executor.submit(
                        self.validate_round_trip,
                        midi,
                        strategy,
                        None,  # track_infos
                        True   # detailed_report
                    )
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                        
                        if stop_on_failure and not result[0].is_valid:
                            logger.warning("Stopping batch validation due to failure")
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                            
                    except Exception as e:
                        logger.error(f"Batch validation error: {e}")
                        results.append((
                            ValidationResult(is_valid=False, errors=[str(e)]),
                            RoundTripMetrics()
                        ))
        else:
            # Sequential processing
            for i, midi in enumerate(midi_files):
                logger.info(f"Validating file {i+1}/{len(midi_files)}")
                result, metrics = self.validate_round_trip(midi, strategy)
                results.append((result, metrics))
                
                if stop_on_failure and not result.is_valid:
                    logger.warning(f"Stopping batch validation at file {i+1} due to failure")
                    break
        
        # Log summary
        valid_count = sum(1 for r, _ in results if r.is_valid)
        logger.info(f"Batch validation complete: {valid_count}/{len(results)} files passed")
        
        return results
    
    def validate_chunked_midi(
        self,
        chunks: List[MidiFile],
        strategy: Optional[str] = None,
        original_midi: Optional[MidiFile] = None
    ) -> Tuple[ValidationResult, List[RoundTripMetrics]]:
        """
        Validate chunked MIDI files from chunk_midi_file function.
        
        Args:
            chunks: List of MIDI chunks
            strategy: Tokenization strategy
            original_midi: Original MIDI for reference
            
        Returns:
            Tuple of (overall ValidationResult, list of chunk metrics)
        """
        strategy = strategy or self.config.tokenization
        overall_result = ValidationResult(is_valid=True)
        chunk_metrics = []
        
        logger.info(f"Validating {len(chunks)} MIDI chunks")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Validating chunk {i+1}/{len(chunks)}")
            
            # Validate individual chunk
            result, metrics = self.validate_round_trip(chunk, strategy)
            chunk_metrics.append(metrics)
            
            if not result.is_valid:
                overall_result.add_warning(f"Chunk {i+1} failed validation: {result.errors}")
            
            # Track chunk-specific issues
            if metrics.missing_notes_ratio > 0.05:  # Higher tolerance for chunks
                overall_result.add_warning(f"Chunk {i+1} has high missing notes ratio: "
                                         f"{metrics.missing_notes_ratio:.3f}")
        
        # If we have the original, compare aggregate statistics
        if original_midi:
            total_notes_chunks = sum(m.total_notes_original for m in chunk_metrics)
            orig_metadata = extract_metadata(original_midi)
            
            if abs(total_notes_chunks - orig_metadata.note_count) > orig_metadata.note_count * 0.02:
                overall_result.add_warning(
                    f"Total notes in chunks ({total_notes_chunks}) differs from "
                    f"original ({orig_metadata.note_count})"
                )
        
        # Determine overall validity
        if len([m for m in chunk_metrics if m.overall_accuracy < 0.95]) > len(chunks) * 0.2:
            overall_result.is_valid = False
            overall_result.errors.append("Too many chunks failed quality threshold")
        
        return overall_result, chunk_metrics


# ============================================================================
# Utility Functions
# ============================================================================

def round_trip_test(
    midi_path: Union[str, Path],
    strategy: str = "REMI",
    config: Optional[MidiParserConfig] = None
) -> Tuple[bool, RoundTripMetrics]:
    """
    Simple round-trip test function as specified in Section 5.
    
    Args:
        midi_path: Path to MIDI file
        strategy: Tokenization strategy
        config: Optional configuration
        
    Returns:
        Tuple of (success, metrics)
        
    Example:
        >>> success, metrics = round_trip_test("song.mid", "REMI")
        >>> if success:
        >>>     print(f"Round-trip passed with {metrics.overall_accuracy:.2%} accuracy")
    """
    from parser.core.midi_loader import load_midi_file
    
    # Load MIDI file
    midi = load_midi_file(midi_path)
    if midi is None:
        return False, RoundTripMetrics()
    
    # Create validator
    validator = RoundTripValidator(config)
    
    # Run validation
    result, metrics = validator.validate_round_trip(midi, strategy)
    
    return result.is_valid, metrics


def compare_midi_files(
    original_path: Union[str, Path],
    reconstructed_path: Union[str, Path],
    config: Optional[MidiParserConfig] = None
) -> RoundTripMetrics:
    """
    Compare two MIDI files directly without tokenization.
    
    Args:
        original_path: Path to original MIDI
        reconstructed_path: Path to reconstructed MIDI
        config: Optional configuration
        
    Returns:
        RoundTripMetrics with comparison results
    """
    from parser.core.midi_loader import load_midi_file
    
    original = load_midi_file(original_path)
    reconstructed = load_midi_file(reconstructed_path)
    
    if original is None or reconstructed is None:
        raise ValueError("Failed to load MIDI files for comparison")
    
    validator = RoundTripValidator(config)
    metrics = validator._compare_midi_files(
        original,
        reconstructed,
        "direct_comparison",
        detailed=True
    )
    
    return metrics


def generate_validation_report(
    metrics: RoundTripMetrics,
    output_format: str = "text"
) -> str:
    """
    Generate a human-readable validation report.
    
    Args:
        metrics: Round-trip metrics
        output_format: Format ("text", "markdown", "json")
        
    Returns:
        Formatted report string
    """
    if output_format == "json":
        import json
        return json.dumps(metrics.to_dict(), indent=2)
    
    elif output_format == "markdown":
        report = [
            "# Round-Trip Validation Report",
            "",
            f"**Strategy:** {metrics.tokenization_strategy}",
            f"**Token Count:** {metrics.token_count:,}",
            f"**Processing Time:** {metrics.processing_time:.2f}s",
            "",
            "## Note Statistics",
            f"- Original Notes: {metrics.total_notes_original:,}",
            f"- Reconstructed Notes: {metrics.total_notes_reconstructed:,}",
            f"- Missing Notes: {metrics.missing_notes} ({metrics.missing_notes_ratio:.2%})",
            f"- Extra Notes: {metrics.extra_notes} ({metrics.extra_notes_ratio:.2%})",
            "",
            "## Accuracy Metrics",
            f"- **Overall Accuracy:** {metrics.overall_accuracy:.2%}",
            f"- Timing Accuracy: {metrics.timing_accuracy:.2%}",
            f"- Velocity Accuracy: {metrics.velocity_accuracy:.2%}",
            "",
            "## Timing Statistics",
            f"- Mean Start Difference: {metrics.mean_start_diff:.2f} ticks",
            f"- Max Start Difference: {metrics.max_start_diff} ticks",
            f"- Mean Duration Difference: {metrics.mean_duration_diff:.2f} ticks",
            f"- Max Duration Difference: {metrics.max_duration_diff} ticks",
            "",
            "## Track Analysis",
        ]
        
        for track in metrics.track_comparisons:
            report.append(f"- **{track.track_name}**: {track.accuracy_score:.2%} accuracy "
                        f"({track.original_note_count} → {track.reconstructed_note_count} notes)")
        
        return "\n".join(report)
    
    else:  # text format
        report = [
            "=" * 60,
            "ROUND-TRIP VALIDATION REPORT",
            "=" * 60,
            f"Strategy: {metrics.tokenization_strategy}",
            f"Token Count: {metrics.token_count:,}",
            f"Processing Time: {metrics.processing_time:.2f}s",
            "-" * 60,
            "NOTE STATISTICS:",
            f"  Original Notes: {metrics.total_notes_original:,}",
            f"  Reconstructed: {metrics.total_notes_reconstructed:,}",
            f"  Missing: {metrics.missing_notes} ({metrics.missing_notes_ratio:.2%})",
            f"  Extra: {metrics.extra_notes} ({metrics.extra_notes_ratio:.2%})",
            "-" * 60,
            "ACCURACY METRICS:",
            f"  Overall: {metrics.overall_accuracy:.2%}",
            f"  Timing: {metrics.timing_accuracy:.2%}",
            f"  Velocity: {metrics.velocity_accuracy:.2%}",
            "-" * 60,
            "TIMING STATISTICS:",
            f"  Mean Start Diff: {metrics.mean_start_diff:.2f} ticks",
            f"  Max Start Diff: {metrics.max_start_diff} ticks",
            f"  Mean Duration Diff: {metrics.mean_duration_diff:.2f} ticks",
            f"  Max Duration Diff: {metrics.max_duration_diff} ticks",
            "=" * 60
        ]
        
        return "\n".join(report)


def validate_tokenization_strategy(
    strategy: str,
    test_midi_path: Union[str, Path],
    config: Optional[MidiParserConfig] = None
) -> Dict[str, Any]:
    """
    Comprehensive validation of a specific tokenization strategy.
    
    Args:
        strategy: Tokenization strategy to validate
        test_midi_path: Path to test MIDI file
        config: Optional configuration
        
    Returns:
        Dictionary with validation results and recommendations
    """
    from parser.core.midi_loader import load_midi_file
    from parser.core.tokenizer_manager import TokenizerManager
    
    results = {
        "strategy": strategy,
        "file": str(test_midi_path),
        "validation_passed": False,
        "metrics": None,
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Load test MIDI
        midi = load_midi_file(test_midi_path)
        if midi is None:
            results["issues"].append("Failed to load test MIDI file")
            return results
        
        # Create validator
        validator = RoundTripValidator(config)
        
        # Run validation
        validation_result, metrics = validator.validate_round_trip(midi, strategy)
        
        results["validation_passed"] = validation_result.is_valid
        results["metrics"] = metrics.to_dict()
        results["issues"] = validation_result.errors + validation_result.warnings
        
        # Generate recommendations
        if metrics.missing_notes_ratio > 0.01:
            results["recommendations"].append(
                "High missing notes ratio - consider adjusting beat resolution"
            )
        
        if metrics.timing_accuracy < 0.98:
            results["recommendations"].append(
                "Poor timing accuracy - check time signature handling"
            )
        
        if metrics.velocity_accuracy < 0.95:
            results["recommendations"].append(
                "Poor velocity accuracy - adjust velocity quantization bins"
            )
        
        if not validation_result.is_valid:
            results["recommendations"].append(
                f"Consider using alternative strategy (current: {strategy})"
            )
        
    except Exception as e:
        results["issues"].append(f"Validation error: {str(e)}")
        results["recommendations"].append("Check tokenizer configuration")
    
    return results


# Export main classes and functions
__all__ = [
    'RoundTripValidator',
    'RoundTripMetrics',
    'NoteComparison',
    'TrackComparison',
    'round_trip_test',
    'compare_midi_files',
    'generate_validation_report',
    'validate_tokenization_strategy',
]