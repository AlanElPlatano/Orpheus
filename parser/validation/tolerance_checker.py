"""
Tolerance checking and validation logic for round-trip validation.

This module applies tolerance thresholds to validation metrics and determines
pass/fail status based on configurable quality criteria.
"""

import logging
from typing import Dict, Union, Optional
from parser.core.midi_loader import ValidationResult
from parser.config.defaults import ValidationConfig
from .validation_metrics import RoundTripMetrics, DEFAULT_VALIDATION_TOLERANCES

logger = logging.getLogger(__name__)


class ToleranceChecker:
    """
    Applies tolerance thresholds to determine validation pass/fail status.
    
    This class encapsulates all the logic for checking whether validation
    metrics meet acceptable quality standards based on configurable tolerances.
    """
    
    def __init__(self, validation_config: Optional[ValidationConfig] = None):
        """
        Initialize the tolerance checker.
        
        Args:
            validation_config: Configuration with tolerance settings
        """
        self.config = validation_config
        self.tolerances = getattr(validation_config, 'tolerances', DEFAULT_VALIDATION_TOLERANCES)
        self.quality_threshold = getattr(validation_config, 'quality_threshold', 0.95)
        
    def check_tolerances(self, metrics: RoundTripMetrics) -> ValidationResult:
        """
        Apply all tolerance checks to validation metrics.
        
        Args:
            metrics: Calculated validation metrics
            
        Returns:
            ValidationResult with pass/fail status and detailed messages
        """
        result = ValidationResult(is_valid=True)
        
        # Check note count tolerances
        self._check_note_tolerances(metrics, result)
        
        # Check timing tolerances
        self._check_timing_tolerances(metrics, result)
        
        # Check velocity tolerances
        self._check_velocity_tolerances(metrics, result)
        
        # Check overall quality threshold
        self._check_quality_threshold(metrics, result)
        
        # Check track-level issues
        self._check_track_tolerances(metrics, result)
        
        return result
    
    def _check_note_tolerances(self, metrics: RoundTripMetrics, result: ValidationResult) -> None:
        """Check note count related tolerances."""
        missing_threshold = self.tolerances.get('missing_notes_ratio', 0.01)
        extra_threshold = self.tolerances.get('extra_notes_ratio', 0.01)
        
        if metrics.missing_notes_ratio > missing_threshold:
            result.add_error(f"Missing notes ratio {metrics.missing_notes_ratio:.3f} exceeds "
                           f"tolerance {missing_threshold}")
        
        if metrics.extra_notes_ratio > extra_threshold:
            result.add_error(f"Extra notes ratio {metrics.extra_notes_ratio:.3f} exceeds "
                           f"tolerance {extra_threshold}")
        
        # Warning for moderate issues
        if metrics.missing_notes_ratio > missing_threshold * 0.5:
            result.add_warning(f"Elevated missing notes ratio: {metrics.missing_notes_ratio:.3f}")
        
        if metrics.extra_notes_ratio > extra_threshold * 0.5:
            result.add_warning(f"Elevated extra notes ratio: {metrics.extra_notes_ratio:.3f}")
    
    def _check_timing_tolerances(self, metrics: RoundTripMetrics, result: ValidationResult) -> None:
        """Check timing related tolerances."""
        start_tolerance = self.tolerances.get('note_start_tick', 1)
        duration_tolerance = self.tolerances.get('note_duration', 2)
        
        if metrics.max_start_diff > start_tolerance * 5:  # Allow some flexibility
            result.add_warning(f"Large timing differences detected. Max: {metrics.max_start_diff} ticks")
        
        if metrics.max_duration_diff > duration_tolerance * 5:
            result.add_warning(f"Large duration differences detected. Max: {metrics.max_duration_diff} ticks")
        
        # Check timing accuracy
        if metrics.timing_accuracy < 0.98:
            result.add_warning(f"Timing accuracy degraded: {metrics.timing_accuracy:.3f}")
        
        if metrics.timing_accuracy < 0.95:
            result.add_error(f"Timing accuracy critically low: {metrics.timing_accuracy:.3f}")
    
    def _check_velocity_tolerances(self, metrics: RoundTripMetrics, result: ValidationResult) -> None:
        """Check velocity related tolerances."""
        velocity_tolerance = self.tolerances.get('velocity_bin', 1)
        
        if metrics.max_velocity_diff > velocity_tolerance * 10:
            result.add_warning(f"Large velocity differences detected. Max: {metrics.max_velocity_diff}")
        
        if metrics.velocity_accuracy < 0.95:
            result.add_warning(f"Velocity accuracy degraded: {metrics.velocity_accuracy:.3f}")
        
        if metrics.velocity_accuracy < 0.90:
            result.add_error(f"Velocity accuracy critically low: {metrics.velocity_accuracy:.3f}")
    
    def _check_quality_threshold(self, metrics: RoundTripMetrics, result: ValidationResult) -> None:
        """Check overall quality threshold."""
        if metrics.overall_accuracy < self.quality_threshold:
            result.add_error(f"Overall accuracy {metrics.overall_accuracy:.3f} below "
                           f"quality threshold {self.quality_threshold}")
        
        # Warning for approaching threshold
        warning_threshold = self.quality_threshold + 0.02
        if metrics.overall_accuracy < warning_threshold:
            result.add_warning(f"Overall accuracy {metrics.overall_accuracy:.3f} approaching "
                             f"quality threshold {self.quality_threshold}")
    
    def _check_track_tolerances(self, metrics: RoundTripMetrics, result: ValidationResult) -> None:
        """Check track-level tolerance issues."""
        total_tracks = len(metrics.track_comparisons)
        if total_tracks == 0:
            return
        
        # Check for tracks with poor accuracy
        poor_tracks = [tc for tc in metrics.track_comparisons if tc.accuracy_score < 0.90]
        if len(poor_tracks) > total_tracks * 0.2:
            result.add_warning(f"{len(poor_tracks)}/{total_tracks} tracks have poor accuracy")
        
        # Check for completely failed tracks
        failed_tracks = [tc for tc in metrics.track_comparisons if tc.accuracy_score < 0.5]
        if failed_tracks:
            result.add_error(f"{len(failed_tracks)} tracks completely failed validation")
        
        # Check for missing tracks
        missing_tracks = [tc for tc in metrics.track_comparisons 
                         if tc.original_note_count > 0 and tc.reconstructed_note_count == 0]
        if missing_tracks:
            result.add_error(f"{len(missing_tracks)} tracks are completely missing")
    
    def get_tolerance_summary(self) -> Dict[str, Union[int, float]]:
        """Get current tolerance settings."""
        return self.tolerances.copy()
    
    def update_tolerance(self, key: str, value: Union[int, float]) -> None:
        """Update a specific tolerance value."""
        self.tolerances[key] = value
        logger.debug(f"Updated tolerance {key} to {value}")
    
    def is_strict_mode(self) -> bool:
        """Check if operating in strict validation mode."""
        return self.quality_threshold >= 0.98
    
    def set_quality_threshold(self, threshold: float) -> None:
        """Set the overall quality threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        self.quality_threshold = threshold
        logger.info(f"Quality threshold set to {threshold}")
