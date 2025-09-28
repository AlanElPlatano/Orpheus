"""
Validation metrics and data classes for MIDI round-trip validation.

This module contains the core data structures for tracking validation results
and calculating aggregate metrics from round-trip conversion tests.
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from miditoolkit import Note


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
    
    def calculate_timing_stats(self, comparisons: List[NoteComparison]) -> None:
        """Calculate timing statistics from note comparisons."""
        start_diffs = []
        duration_diffs = []
        velocity_diffs = []
        
        for nc in comparisons:
            if not nc.is_missing and not nc.is_extra:
                start_diffs.append(nc.start_diff)
                duration_diffs.append(nc.duration_diff)
                velocity_diffs.append(nc.velocity_diff)
                
                # Update maximums
                if nc.duration_diff > self.max_duration_diff:
                    self.max_duration_diff = nc.duration_diff
                if nc.start_diff > self.max_start_diff:
                    self.max_start_diff = nc.start_diff
                if nc.velocity_diff > self.max_velocity_diff:
                    self.max_velocity_diff = nc.velocity_diff
        
        # Calculate means using built-in statistics
        if start_diffs:
            self.mean_start_diff = statistics.mean(start_diffs)
        if duration_diffs:
            self.mean_duration_diff = statistics.mean(duration_diffs)
        if velocity_diffs:
            self.mean_velocity_diff = statistics.mean(velocity_diffs)
    
    def calculate_aggregates(self) -> None:
        """Calculate aggregate metrics from detailed comparisons."""
        if self.total_notes_original > 0:
            self.missing_notes_ratio = self.missing_notes / self.total_notes_original
            self.extra_notes_ratio = self.extra_notes / max(self.total_notes_original, 1)
            
            self.timing_accuracy = 1.0 - (self.timing_errors / max(self.total_notes_original, 1))
            self.velocity_accuracy = 1.0 - (self.velocity_errors / max(self.total_notes_original, 1))
            
            # Overall accuracy
            total_errors = (self.missing_notes + self.extra_notes + 
                          self.timing_errors + self.velocity_errors + self.pitch_errors)
            max_possible_errors = self.total_notes_original * 4  # 4 aspects per note
            self.overall_accuracy = 1.0 - (total_errors / max(max_possible_errors, 1))
    
    def aggregate_from_tracks(self) -> None:
        """Aggregate metrics from track comparisons."""
        # Reset totals
        self.total_notes_original = 0
        self.total_notes_reconstructed = 0
        self.missing_notes = 0
        self.extra_notes = 0
        self.timing_errors = 0
        self.velocity_errors = 0
        
        # Collect all note comparisons for detailed statistics
        all_comparisons = []
        
        # Sum up totals from track comparisons
        for track_comp in self.track_comparisons:
            self.total_notes_original += track_comp.original_note_count
            self.total_notes_reconstructed += track_comp.reconstructed_note_count
            self.missing_notes += track_comp.missing_notes
            self.extra_notes += track_comp.extra_notes
            self.timing_errors += track_comp.timing_errors
            self.velocity_errors += track_comp.velocity_errors
            
            # Collect note comparisons for detailed stats
            all_comparisons.extend(track_comp.note_comparisons)
        
        # Calculate detailed timing statistics
        if all_comparisons:
            self.calculate_timing_stats(all_comparisons)
        
        # Calculate aggregate ratios and scores
        self.calculate_aggregates()
    
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
            },
            "tracks": len(self.track_comparisons)
        }


# Constants for validation
DEFAULT_VALIDATION_TOLERANCES = {
    "note_start_tick": 1,
    "note_duration": 2,
    "velocity_bin": 1,
    "missing_notes_ratio": 0.01,
    "extra_notes_ratio": 0.01,
    "tempo_bpm_diff": 1.0,
}

MAX_SEQ_LENGTH = 2048
VALIDATION_TIMEOUT_SECONDS = 60


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class TokenizationError(ValidationError):
    """Raised when tokenization fails."""
    pass


class DetokenizationError(ValidationError):
    """Raised when detokenization fails."""
    pass


class ComparisonError(ValidationError):
    """Raised when MIDI comparison fails."""
    pass