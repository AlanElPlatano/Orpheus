"""
Dynamic threshold adjustment based on content complexity and use case requirements.

This module provides adaptive threshold management for validation based on musical
complexity, use case requirements, and content-specific characteristics, enabling
more nuanced quality assessment.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from parser.config.defaults import ValidationConfig
from parser.core.track_analyzer import TrackInfo
from parser.core.midi_loader import MidiMetadata

logger = logging.getLogger(__name__)


class UseCase(Enum):
    """Enumeration of use cases with different quality requirements."""
    RESEARCH = "research"
    PRODUCTION = "production"
    EDUCATION = "education"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    ARCHIVAL = "archival"
    REAL_TIME = "real_time"
    DEMO = "demo"


class ComplexityLevel(Enum):
    """Enumeration of musical complexity levels."""
    MINIMAL = "minimal"      # Single melodic line, simple rhythm
    SIMPLE = "simple"        # Basic harmony, regular rhythm
    MODERATE = "moderate"    # Standard complexity
    COMPLEX = "complex"      # Rich harmony, varied rhythm
    VERY_COMPLEX = "very_complex"  # Orchestral, polyrhythmic


@dataclass
class ComplexityMetrics:
    """Metrics for measuring musical complexity."""
    note_density: float = 0.0  # Notes per second
    polyphony_level: float = 0.0  # Average simultaneous notes
    rhythmic_complexity: float = 0.0  # Rhythm variation measure
    harmonic_complexity: float = 0.0  # Harmonic richness
    dynamic_range: float = 0.0  # Velocity variation
    pitch_range: int = 0  # Pitch spread
    track_count: int = 0  # Number of tracks
    tempo_changes: int = 0  # Number of tempo changes
    time_signature_changes: int = 0  # Time signature variations
    overall_complexity: float = 0.0  # Combined score
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "note_density": round(self.note_density, 2),
            "polyphony_level": round(self.polyphony_level, 2),
            "rhythmic_complexity": round(self.rhythmic_complexity, 3),
            "harmonic_complexity": round(self.harmonic_complexity, 3),
            "dynamic_range": round(self.dynamic_range, 1),
            "pitch_range": self.pitch_range,
            "track_count": self.track_count,
            "tempo_changes": self.tempo_changes,
            "time_signature_changes": self.time_signature_changes,
            "overall_complexity": round(self.overall_complexity, 3),
            "complexity_level": self.complexity_level.value
        }


@dataclass
class AdaptiveThresholds:
    """Adaptive threshold values based on context."""
    # Note-level tolerances
    note_start_tick: int = 1
    note_duration: int = 2
    velocity_bin: int = 1
    
    # Track-level tolerances
    missing_notes_ratio: float = 0.01
    extra_notes_ratio: float = 0.01
    tempo_bpm_diff: float = 1.0
    
    # Quality thresholds
    overall_quality_threshold: float = 0.95
    timing_accuracy_threshold: float = 0.98
    velocity_accuracy_threshold: float = 0.95
    pitch_accuracy_threshold: float = 0.99
    
    # Feature preservation thresholds
    musical_fidelity_threshold: float = 0.90
    sequence_quality_threshold: float = 0.85
    statistical_similarity_threshold: float = 0.85
    
    # Complexity-adjusted factors
    complexity_adjustment_factor: float = 1.0
    use_case_strictness_factor: float = 1.0
    
    # Explanation for adjustments
    adjustment_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "note_tolerances": {
                "start_tick": self.note_start_tick,
                "duration": self.note_duration,
                "velocity_bin": self.velocity_bin
            },
            "track_tolerances": {
                "missing_notes_ratio": round(self.missing_notes_ratio, 4),
                "extra_notes_ratio": round(self.extra_notes_ratio, 4),
                "tempo_bpm_diff": round(self.tempo_bpm_diff, 2)
            },
            "quality_thresholds": {
                "overall": round(self.overall_quality_threshold, 3),
                "timing": round(self.timing_accuracy_threshold, 3),
                "velocity": round(self.velocity_accuracy_threshold, 3),
                "pitch": round(self.pitch_accuracy_threshold, 3)
            },
            "preservation_thresholds": {
                "musical_fidelity": round(self.musical_fidelity_threshold, 3),
                "sequence_quality": round(self.sequence_quality_threshold, 3),
                "statistical_similarity": round(self.statistical_similarity_threshold, 3)
            },
            "adjustment_factors": {
                "complexity": round(self.complexity_adjustment_factor, 2),
                "use_case": round(self.use_case_strictness_factor, 2)
            },
            "adjustment_reasons": self.adjustment_reasons
        }


@dataclass
class ThresholdRecommendation:
    """Recommendations for threshold adjustments."""
    recommended_thresholds: AdaptiveThresholds
    confidence_score: float = 0.0
    recommendation_reasons: List[str] = field(default_factory=list)
    alternative_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AdaptiveThresholdManager:
    """
    Manages dynamic threshold adjustment based on content and context.
    
    This class provides intelligent threshold adaptation based on musical
    complexity, use case requirements, and genre-specific characteristics.
    """
    
    # Use case threshold presets
    USE_CASE_PRESETS = {
        UseCase.RESEARCH: {
            "quality_multiplier": 1.0,    # Highest standards
            "tolerance_multiplier": 0.5,   # Strictest tolerances
            "description": "Research-grade precision required"
        },
        UseCase.PRODUCTION: {
            "quality_multiplier": 0.98,
            "tolerance_multiplier": 0.7,
            "description": "Production quality with some flexibility"
        },
        UseCase.EDUCATION: {
            "quality_multiplier": 0.95,
            "tolerance_multiplier": 1.0,
            "description": "Educational use with balanced requirements"
        },
        UseCase.GENERATION: {
            "quality_multiplier": 0.93,
            "tolerance_multiplier": 1.2,
            "description": "AI generation with creative flexibility"
        },
        UseCase.ANALYSIS: {
            "quality_multiplier": 0.96,
            "tolerance_multiplier": 0.8,
            "description": "Analysis requiring good preservation"
        },
        UseCase.ARCHIVAL: {
            "quality_multiplier": 0.99,
            "tolerance_multiplier": 0.6,
            "description": "Archival storage requiring high fidelity"
        },
        UseCase.REAL_TIME: {
            "quality_multiplier": 0.90,
            "tolerance_multiplier": 1.5,
            "description": "Real-time processing with relaxed constraints"
        },
        UseCase.DEMO: {
            "quality_multiplier": 0.85,
            "tolerance_multiplier": 2.0,
            "description": "Demo/prototype with minimal constraints"
        }
    }
    
    # Complexity adjustment factors
    COMPLEXITY_ADJUSTMENTS = {
        ComplexityLevel.MINIMAL: {
            "tolerance_multiplier": 0.5,   # Stricter for simple content
            "quality_multiplier": 1.05,    # Higher expectations
            "description": "Simple content should have near-perfect preservation"
        },
        ComplexityLevel.SIMPLE: {
            "tolerance_multiplier": 0.7,
            "quality_multiplier": 1.02,
            "description": "Basic content with high preservation expected"
        },
        ComplexityLevel.MODERATE: {
            "tolerance_multiplier": 1.0,
            "quality_multiplier": 1.0,
            "description": "Standard complexity with normal thresholds"
        },
        ComplexityLevel.COMPLEX: {
            "tolerance_multiplier": 1.3,
            "quality_multiplier": 0.97,
            "description": "Complex content allows more tolerance"
        },
        ComplexityLevel.VERY_COMPLEX: {
            "tolerance_multiplier": 1.6,
            "quality_multiplier": 0.94,
            "description": "Very complex content with relaxed thresholds"
        }
    }
    
    # Genre-specific adjustments
    GENRE_ADJUSTMENTS = {
        "classical": {
            "timing_strictness": 1.2,
            "pitch_strictness": 1.1,
            "dynamics_strictness": 1.3,
            "description": "Classical music requires precise dynamics and timing"
        },
        "jazz": {
            "timing_strictness": 0.8,
            "pitch_strictness": 1.0,
            "dynamics_strictness": 1.1,
            "description": "Jazz allows timing flexibility but needs good dynamics"
        },
        "electronic": {
            "timing_strictness": 1.5,
            "pitch_strictness": 1.2,
            "dynamics_strictness": 0.9,
            "description": "Electronic music needs precise timing and pitch"
        },
        "folk": {
            "timing_strictness": 0.9,
            "pitch_strictness": 0.95,
            "dynamics_strictness": 0.85,
            "description": "Folk music allows general flexibility"
        },
        "orchestral": {
            "timing_strictness": 1.1,
            "pitch_strictness": 1.0,
            "dynamics_strictness": 1.4,
            "description": "Orchestral music needs excellent dynamic range"
        },
        "custom": {
            "timing_strictness": 1.0,
            "pitch_strictness": 1.0,
            "dynamics_strictness": 0.3,
            "description": "Orchestral music needs excellent dynamic range"
        }
    }
    
    def __init__(self, base_config: Optional[ValidationConfig] = None):
        """
        Initialize the adaptive threshold manager.
        
        Args:
            base_config: Base validation configuration
        """
        self.base_config = base_config or ValidationConfig()
        self.current_thresholds = self._create_default_thresholds()
        
    def calculate_adaptive_thresholds(
        self,
        track_infos: List[TrackInfo],
        metadata: Optional[MidiMetadata] = None,
        use_case: UseCase = UseCase.PRODUCTION,
        genre: Optional[str] = None,
        custom_requirements: Optional[Dict[str, float]] = None
    ) -> AdaptiveThresholds:
        """
        Calculate adaptive thresholds based on content and context.
        
        Args:
            track_infos: Track analysis information
            metadata: MIDI metadata
            use_case: Target use case
            genre: Musical genre (optional)
            custom_requirements: Custom requirement overrides
            
        Returns:
            AdaptiveThresholds with calculated values
        """
        logger.info(f"Calculating adaptive thresholds for {use_case.value} use case")
        
        # Start with base thresholds
        thresholds = self._create_default_thresholds()
        
        # Calculate complexity metrics
        complexity = self._calculate_complexity(track_infos, metadata)
        
        # Apply use case adjustments
        self._apply_use_case_adjustments(thresholds, use_case, complexity)
        
        # Apply complexity adjustments
        self._apply_complexity_adjustments(thresholds, complexity)
        
        # Apply genre-specific adjustments if provided
        if genre:
            self._apply_genre_adjustments(thresholds, genre)
        
        # Apply custom requirements if provided
        if custom_requirements:
            self._apply_custom_requirements(thresholds, custom_requirements)
        
        # Validate and normalize thresholds
        self._validate_thresholds(thresholds)
        
        logger.info(f"Adaptive thresholds calculated with complexity level: {complexity.complexity_level.value}")
        
        return thresholds
    
    def recommend_thresholds(
        self,
        track_infos: List[TrackInfo],
        metadata: Optional[MidiMetadata] = None,
        target_accuracy: float = 0.95,
        processing_constraints: Optional[Dict[str, Any]] = None
    ) -> ThresholdRecommendation:
        """
        Recommend optimal thresholds based on analysis.
        
        Args:
            track_infos: Track analysis information
            metadata: MIDI metadata
            target_accuracy: Target accuracy level
            processing_constraints: Processing constraints (e.g., speed requirements)
            
        Returns:
            ThresholdRecommendation with suggestions
        """
        recommendation = ThresholdRecommendation(
            recommended_thresholds=self._create_default_thresholds()
        )
        
        # Analyze content
        complexity = self._calculate_complexity(track_infos, metadata)
        
        # Determine best use case based on target accuracy
        if target_accuracy >= 0.98:
            use_case = UseCase.RESEARCH
            recommendation.recommendation_reasons.append("Research-grade thresholds for high accuracy target")
        elif target_accuracy >= 0.95:
            use_case = UseCase.PRODUCTION
            recommendation.recommendation_reasons.append("Production thresholds for standard accuracy")
        elif target_accuracy >= 0.90:
            use_case = UseCase.GENERATION
            recommendation.recommendation_reasons.append("Generation thresholds for moderate accuracy")
        else:
            use_case = UseCase.DEMO
            recommendation.recommendation_reasons.append("Demo thresholds for relaxed accuracy")
        
        # Calculate recommended thresholds
        recommendation.recommended_thresholds = self.calculate_adaptive_thresholds(
            track_infos, metadata, use_case
        )
        
        # Add processing constraint adjustments
        if processing_constraints:
            self._adjust_for_processing_constraints(
                recommendation.recommended_thresholds,
                processing_constraints,
                recommendation
            )
        
        # Calculate confidence score
        recommendation.confidence_score = self._calculate_recommendation_confidence(
            complexity, use_case, target_accuracy
        )
        
        # Generate alternative suggestions
        recommendation.alternative_suggestions = self._generate_alternatives(
            complexity, target_accuracy
        )
        
        # Add warnings if necessary
        if complexity.overall_complexity > 0.8:
            recommendation.warnings.append("High complexity detected - consider relaxing thresholds")
        
        if target_accuracy > 0.98 and complexity.overall_complexity > 0.7:
            recommendation.warnings.append("Target accuracy may be difficult to achieve with complex content")
        
        return recommendation
    
    def _create_default_thresholds(self) -> AdaptiveThresholds:
        """Create default thresholds from base configuration."""
        tolerances = self.base_config.tolerances
        
        return AdaptiveThresholds(
            note_start_tick=tolerances.get('note_start_tick', 1),
            note_duration=tolerances.get('note_duration', 2),
            velocity_bin=tolerances.get('velocity_bin', 1),
            missing_notes_ratio=tolerances.get('missing_notes_ratio', 0.01),
            extra_notes_ratio=tolerances.get('extra_notes_ratio', 0.01),
            tempo_bpm_diff=tolerances.get('tempo_bpm_diff', 1.0),
            overall_quality_threshold=self.base_config.quality_threshold
        )
    
    def _calculate_complexity(
        self,
        track_infos: List[TrackInfo],
        metadata: Optional[MidiMetadata]
    ) -> ComplexityMetrics:
        """
        Calculate musical complexity metrics.
        
        Args:
            track_infos: Track analysis information
            metadata: MIDI metadata
            
        Returns:
            ComplexityMetrics with calculated values
        """
        complexity = ComplexityMetrics()
        
        if not track_infos:
            return complexity
        
        # Track count
        complexity.track_count = len(track_infos)
        
        # Calculate aggregate statistics
        total_notes = sum(t.statistics.total_notes for t in track_infos)
        
        # Note density (notes per second)
        if metadata and metadata.duration_seconds > 0:
            complexity.note_density = total_notes / metadata.duration_seconds
        
        # Polyphony level
        polyphony_values = [t.statistics.avg_polyphony for t in track_infos if t.statistics.avg_polyphony > 0]
        if polyphony_values:
            complexity.polyphony_level = statistics.mean(polyphony_values)
        
        # Rhythmic complexity (based on note duration variety)
        duration_varieties = []
        for track in track_infos:
            if track.statistics.total_notes > 0:
                # Use coefficient of variation as complexity measure
                if hasattr(track.statistics, 'duration_stats') and track.statistics.duration_stats:
                    duration_std = track.statistics.duration_stats.get('std', 0)
                    duration_mean = track.statistics.duration_stats.get('mean', 1)
                    if duration_mean > 0:
                        cv = duration_std / duration_mean
                        duration_varieties.append(cv)
        
        if duration_varieties:
            complexity.rhythmic_complexity = statistics.mean(duration_varieties)
        
        # Harmonic complexity (based on simultaneous note relationships)
        chord_tracks = [t for t in track_infos if t.type == "chord"]
        if chord_tracks:
            chord_densities = [t.statistics.avg_polyphony for t in chord_tracks]
            complexity.harmonic_complexity = statistics.mean(chord_densities) / 10  # Normalize
        
        # Dynamic range
        velocity_ranges = []
        for track in track_infos:
            if hasattr(track.statistics, 'velocity_stats') and track.statistics.velocity_stats:
                v_min = track.statistics.velocity_stats.get('min', 0)
                v_max = track.statistics.velocity_stats.get('max', 127)
                velocity_ranges.append(v_max - v_min)
        
        if velocity_ranges:
            complexity.dynamic_range = statistics.mean(velocity_ranges)
        
        # Pitch range
        pitch_ranges = []
        for track in track_infos:
            if hasattr(track.statistics, 'pitch_stats') and track.statistics.pitch_stats:
                p_min = track.statistics.pitch_stats.get('min', 0)
                p_max = track.statistics.pitch_stats.get('max', 127)
                pitch_ranges.append(p_max - p_min)
        
        if pitch_ranges:
            complexity.pitch_range = max(pitch_ranges)
        
        # Tempo and time signature changes
        if metadata:
            complexity.tempo_changes = len(metadata.tempo_changes) - 1  # Subtract initial tempo
            complexity.time_signature_changes = len(metadata.time_signatures) - 1
        
        # Calculate overall complexity score
        complexity.overall_complexity = self._calculate_overall_complexity(complexity)
        
        # Determine complexity level
        complexity.complexity_level = self._determine_complexity_level(complexity.overall_complexity)
        
        return complexity
    
    def _calculate_overall_complexity(self, metrics: ComplexityMetrics) -> float:
        """
        Calculate overall complexity score from individual metrics.
        
        Args:
            metrics: Complexity metrics
            
        Returns:
            Overall complexity score (0-1)
        """
        scores = []
        weights = []
        
        # Note density score (normalize to 0-1, assuming 10 notes/sec is very complex)
        density_score = min(1.0, metrics.note_density / 10)
        scores.append(density_score)
        weights.append(0.2)
        
        # Polyphony score (normalize, assuming 6+ is very complex)
        poly_score = min(1.0, metrics.polyphony_level / 6)
        scores.append(poly_score)
        weights.append(0.2)
        
        # Rhythmic complexity (already 0-1 range approximately)
        scores.append(min(1.0, metrics.rhythmic_complexity))
        weights.append(0.15)
        
        # Harmonic complexity (already normalized)
        scores.append(min(1.0, metrics.harmonic_complexity))
        weights.append(0.15)
        
        # Dynamic range score (normalize, 0-127 range)
        dynamic_score = metrics.dynamic_range / 127
        scores.append(dynamic_score)
        weights.append(0.1)
        
        # Pitch range score (normalize, assuming 60+ semitones is complex)
        pitch_score = min(1.0, metrics.pitch_range / 60)
        scores.append(pitch_score)
        weights.append(0.1)
        
        # Track count score (normalize, assuming 8+ tracks is complex)
        track_score = min(1.0, metrics.track_count / 8)
        scores.append(track_score)
        weights.append(0.05)
        
        # Tempo changes score (normalize, assuming 5+ changes is complex)
        tempo_score = min(1.0, metrics.tempo_changes / 5)
        scores.append(tempo_score)
        weights.append(0.05)
        
        # Calculate weighted average
        if sum(weights) > 0:
            overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            overall = 0.5
        
        return max(0.0, min(1.0, overall))
    
    def _determine_complexity_level(self, overall_complexity: float) -> ComplexityLevel:
        """
        Determine complexity level from overall score.
        
        Args:
            overall_complexity: Overall complexity score (0-1)
            
        Returns:
            ComplexityLevel enum value
        """
        if overall_complexity < 0.2:
            return ComplexityLevel.MINIMAL
        elif overall_complexity < 0.4:
            return ComplexityLevel.SIMPLE
        elif overall_complexity < 0.6:
            return ComplexityLevel.MODERATE
        elif overall_complexity < 0.8:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    def _apply_use_case_adjustments(
        self,
        thresholds: AdaptiveThresholds,
        use_case: UseCase,
        complexity: ComplexityMetrics
    ) -> None:
        """
        Apply use case specific adjustments to thresholds.
        
        Args:
            thresholds: Thresholds to adjust
            use_case: Target use case
            complexity: Complexity metrics
        """
        preset = self.USE_CASE_PRESETS.get(use_case, self.USE_CASE_PRESETS[UseCase.PRODUCTION])
        
        # Apply quality multiplier
        thresholds.overall_quality_threshold *= preset["quality_multiplier"]
        thresholds.timing_accuracy_threshold *= preset["quality_multiplier"]
        thresholds.velocity_accuracy_threshold *= preset["quality_multiplier"]
        thresholds.pitch_accuracy_threshold *= preset["quality_multiplier"]
        
        # Apply tolerance multiplier
        thresholds.note_start_tick = int(thresholds.note_start_tick * preset["tolerance_multiplier"])
        thresholds.note_duration = int(thresholds.note_duration * preset["tolerance_multiplier"])
        thresholds.velocity_bin = int(thresholds.velocity_bin * preset["tolerance_multiplier"])
        
        thresholds.missing_notes_ratio *= preset["tolerance_multiplier"]
        thresholds.extra_notes_ratio *= preset["tolerance_multiplier"]
        thresholds.tempo_bpm_diff *= preset["tolerance_multiplier"]
        
        # Update preservation thresholds
        thresholds.musical_fidelity_threshold *= preset["quality_multiplier"]
        thresholds.sequence_quality_threshold *= preset["quality_multiplier"]
        thresholds.statistical_similarity_threshold *= preset["quality_multiplier"]
        
        # Record adjustment
        thresholds.use_case_strictness_factor = 1.0 / preset["tolerance_multiplier"]
        thresholds.adjustment_reasons.append(f"Applied {use_case.value} use case: {preset['description']}")
    
    def _apply_complexity_adjustments(
        self,
        thresholds: AdaptiveThresholds,
        complexity: ComplexityMetrics
    ) -> None:
        """
        Apply complexity-based adjustments to thresholds.
        
        Args:
            thresholds: Thresholds to adjust
            complexity: Complexity metrics
        """
        adjustment = self.COMPLEXITY_ADJUSTMENTS.get(
            complexity.complexity_level,
            self.COMPLEXITY_ADJUSTMENTS[ComplexityLevel.MODERATE]
        )
        
        # Apply tolerance adjustments
        thresholds.note_start_tick = int(thresholds.note_start_tick * adjustment["tolerance_multiplier"])
        thresholds.note_duration = int(thresholds.note_duration * adjustment["tolerance_multiplier"])
        thresholds.velocity_bin = int(thresholds.velocity_bin * adjustment["tolerance_multiplier"])
        
        thresholds.missing_notes_ratio *= adjustment["tolerance_multiplier"]
        thresholds.extra_notes_ratio *= adjustment["tolerance_multiplier"]
        
        # Apply quality adjustments
        thresholds.overall_quality_threshold *= adjustment["quality_multiplier"]
        
        # Special adjustments for extreme complexity
        if complexity.overall_complexity > 0.9:
            thresholds.timing_accuracy_threshold *= 0.95
            thresholds.adjustment_reasons.append("Relaxed timing for extreme complexity")
        
        if complexity.polyphony_level > 5:
            thresholds.velocity_accuracy_threshold *= 0.97
            thresholds.adjustment_reasons.append("Relaxed velocity for high polyphony")
        
        if complexity.rhythmic_complexity > 0.7:
            thresholds.note_duration *= 1.5
            thresholds.adjustment_reasons.append("Increased duration tolerance for complex rhythm")
        
        # Record adjustment
        thresholds.complexity_adjustment_factor = adjustment["tolerance_multiplier"]
        thresholds.adjustment_reasons.append(
            f"Applied {complexity.complexity_level.value} complexity: {adjustment['description']}"
        )
    
    def _apply_genre_adjustments(
        self,
        thresholds: AdaptiveThresholds,
        genre: str
    ) -> None:
        """
        Apply genre-specific adjustments to thresholds.
        
        Args:
            thresholds: Thresholds to adjust
            genre: Musical genre
        """
        genre_lower = genre.lower()
        
        if genre_lower in self.GENRE_ADJUSTMENTS:
            adjustment = self.GENRE_ADJUSTMENTS[genre_lower]
            
            # Apply timing adjustments
            timing_factor = 1.0 / adjustment.get("timing_strictness", 1.0)
            thresholds.note_start_tick = int(thresholds.note_start_tick * timing_factor)
            thresholds.timing_accuracy_threshold *= adjustment.get("timing_strictness", 1.0)
            
            # Apply pitch adjustments
            pitch_factor = 1.0 / adjustment.get("pitch_strictness", 1.0)
            thresholds.pitch_accuracy_threshold *= adjustment.get("pitch_strictness", 1.0)
            
            # Apply dynamics adjustments
            dynamics_factor = 1.0 / adjustment.get("dynamics_strictness", 1.0)
            thresholds.velocity_bin = int(thresholds.velocity_bin * dynamics_factor)
            thresholds.velocity_accuracy_threshold *= adjustment.get("dynamics_strictness", 1.0)
            
            thresholds.adjustment_reasons.append(
                f"Applied {genre} genre adjustments: {adjustment['description']}"
            )
    
    def _apply_custom_requirements(
        self,
        thresholds: AdaptiveThresholds,
        requirements: Dict[str, float]
    ) -> None:
        """
        Apply custom requirement overrides.
        
        Args:
            thresholds: Thresholds to adjust
            requirements: Custom requirements dictionary
        """
        for key, value in requirements.items():
            if hasattr(thresholds, key):
                old_value = getattr(thresholds, key)
                setattr(thresholds, key, value)
                thresholds.adjustment_reasons.append(
                    f"Custom override: {key} changed from {old_value} to {value}"
                )
    
    def _validate_thresholds(self, thresholds: AdaptiveThresholds) -> None:
        """
        Validate and normalize threshold values.
        
        Args:
            thresholds: Thresholds to validate
        """
        # Ensure minimum values
        thresholds.note_start_tick = max(1, thresholds.note_start_tick)
        thresholds.note_duration = max(1, thresholds.note_duration)
        thresholds.velocity_bin = max(1, thresholds.velocity_bin)
        
        # Ensure ratio bounds
        thresholds.missing_notes_ratio = max(0.001, min(0.5, thresholds.missing_notes_ratio))
        thresholds.extra_notes_ratio = max(0.001, min(0.5, thresholds.extra_notes_ratio))
        
        # Ensure quality threshold bounds
        thresholds.overall_quality_threshold = max(0.5, min(1.0, thresholds.overall_quality_threshold))
        thresholds.timing_accuracy_threshold = max(0.5, min(1.0, thresholds.timing_accuracy_threshold))
        thresholds.velocity_accuracy_threshold = max(0.5, min(1.0, thresholds.velocity_accuracy_threshold))
        thresholds.pitch_accuracy_threshold = max(0.5, min(1.0, thresholds.pitch_accuracy_threshold))
        
        # Ensure preservation threshold bounds
        thresholds.musical_fidelity_threshold = max(0.5, min(1.0, thresholds.musical_fidelity_threshold))
        thresholds.sequence_quality_threshold = max(0.5, min(1.0, thresholds.sequence_quality_threshold))
        thresholds.statistical_similarity_threshold = max(0.5, min(1.0, thresholds.statistical_similarity_threshold))
    
    def _adjust_for_processing_constraints(
        self,
        thresholds: AdaptiveThresholds,
        constraints: Dict[str, Any],
        recommendation: ThresholdRecommendation
    ) -> None:
        """
        Adjust thresholds based on processing constraints.
        
        Args:
            thresholds: Thresholds to adjust
            constraints: Processing constraints
            recommendation: Recommendation object to update
        """
        if 'max_processing_time' in constraints:
            max_time = constraints['max_processing_time']
            if max_time < 1.0:  # Less than 1 second
                # Relax thresholds for speed
                thresholds.note_start_tick *= 2
                thresholds.note_duration *= 2
                thresholds.overall_quality_threshold *= 0.95
                recommendation.recommendation_reasons.append(
                    f"Relaxed thresholds for fast processing (<{max_time}s)"
                )
        
        if 'max_memory_mb' in constraints:
            max_memory = constraints['max_memory_mb']
            if max_memory < 100:  # Less than 100MB
                # Simplify processing
                thresholds.statistical_similarity_threshold *= 0.9
                recommendation.recommendation_reasons.append(
                    f"Simplified statistical requirements for memory constraint (<{max_memory}MB)"
                )
        
        if 'real_time' in constraints and constraints['real_time']:
            # Real-time processing adjustments
            thresholds.overall_quality_threshold *= 0.92
            thresholds.sequence_quality_threshold *= 0.88
            recommendation.recommendation_reasons.append(
                "Adjusted for real-time processing requirements"
            )
    
    def _calculate_recommendation_confidence(
        self,
        complexity: ComplexityMetrics,
        use_case: UseCase,
        target_accuracy: float
    ) -> float:
        """
        Calculate confidence score for recommendation.
        
        Args:
            complexity: Complexity metrics
            use_case: Selected use case
            target_accuracy: Target accuracy level
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Reduce confidence for extreme complexity
        if complexity.overall_complexity > 0.8:
            confidence *= 0.85
        elif complexity.overall_complexity < 0.2:
            confidence *= 0.95  # Very simple, high confidence
        
        # Reduce confidence for mismatched use case and target
        use_case_expected_accuracy = {
            UseCase.RESEARCH: 0.98,
            UseCase.PRODUCTION: 0.95,
            UseCase.ARCHIVAL: 0.97,
            UseCase.GENERATION: 0.92,
            UseCase.REAL_TIME: 0.88,
            UseCase.DEMO: 0.85
        }
        
        expected = use_case_expected_accuracy.get(use_case, 0.95)
        accuracy_mismatch = abs(target_accuracy - expected)
        confidence *= (1.0 - accuracy_mismatch)
        
        # Reduce confidence for very high target with complex content
        if target_accuracy > 0.98 and complexity.overall_complexity > 0.6:
            confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_alternatives(
        self,
        complexity: ComplexityMetrics,
        target_accuracy: float
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative threshold suggestions.
        
        Args:
            complexity: Complexity metrics
            target_accuracy: Target accuracy level
            
        Returns:
            List of alternative suggestions
        """
        alternatives = []
        
        # Suggest stricter alternative
        if target_accuracy < 0.98:
            alternatives.append({
                "name": "High Fidelity",
                "description": "Stricter thresholds for maximum quality",
                "use_case": UseCase.RESEARCH.value,
                "expected_accuracy": 0.98,
                "trade_offs": ["Slower processing", "May reject valid variations"]
            })
        
        # Suggest balanced alternative
        alternatives.append({
            "name": "Balanced",
            "description": "Balanced thresholds for general use",
            "use_case": UseCase.PRODUCTION.value,
            "expected_accuracy": 0.95,
            "trade_offs": ["Good quality/speed balance", "Suitable for most content"]
        })
        
        # Suggest relaxed alternative for complex content
        if complexity.overall_complexity > 0.6:
            alternatives.append({
                "name": "Complex Content Optimized",
                "description": "Relaxed thresholds for complex musical content",
                "use_case": UseCase.GENERATION.value,
                "expected_accuracy": 0.92,
                "trade_offs": ["Better handling of complexity", "Some quality compromise"]
            })
        
        # Suggest speed-optimized alternative
        alternatives.append({
            "name": "Speed Optimized",
            "description": "Relaxed thresholds for fast processing",
            "use_case": UseCase.REAL_TIME.value,
            "expected_accuracy": 0.88,
            "trade_offs": ["Fast processing", "Lower quality requirements"]
        })
        
        return alternatives
    
    def get_threshold_explanation(
        self,
        thresholds: AdaptiveThresholds
    ) -> Dict[str, str]:
        """
        Get human-readable explanations for threshold values.
        
        Args:
            thresholds: Threshold values to explain
            
        Returns:
            Dictionary of explanations
        """
        explanations = {
            "note_start_tick": f"Notes can start up to {thresholds.note_start_tick} ticks off",
            "note_duration": f"Note durations can vary by up to {thresholds.note_duration} ticks",
            "velocity_bin": f"Velocity can vary by up to {thresholds.velocity_bin} levels",
            "missing_notes_ratio": f"Up to {thresholds.missing_notes_ratio*100:.1f}% of notes can be missing",
            "extra_notes_ratio": f"Up to {thresholds.extra_notes_ratio*100:.1f}% extra notes allowed",
            "tempo_bpm_diff": f"Tempo can vary by up to {thresholds.tempo_bpm_diff} BPM",
            "overall_quality_threshold": f"Overall quality must be at least {thresholds.overall_quality_threshold*100:.1f}%",
            "timing_accuracy_threshold": f"Timing accuracy must be at least {thresholds.timing_accuracy_threshold*100:.1f}%",
            "velocity_accuracy_threshold": f"Velocity accuracy must be at least {thresholds.velocity_accuracy_threshold*100:.1f}%",
            "pitch_accuracy_threshold": f"Pitch accuracy must be at least {thresholds.pitch_accuracy_threshold*100:.1f}%"
        }
        
        return explanations
    
    def compare_thresholds(
        self,
        thresholds1: AdaptiveThresholds,
        thresholds2: AdaptiveThresholds
    ) -> Dict[str, Any]:
        """
        Compare two sets of thresholds.
        
        Args:
            thresholds1: First threshold set
            thresholds2: Second threshold set
            
        Returns:
            Comparison results
        """
        comparison = {
            "strictness_comparison": self._compare_strictness(thresholds1, thresholds2),
            "differences": {},
            "summary": ""
        }
        
        # Calculate differences
        for attr in ['note_start_tick', 'note_duration', 'velocity_bin',
                    'missing_notes_ratio', 'extra_notes_ratio', 'tempo_bpm_diff',
                    'overall_quality_threshold', 'timing_accuracy_threshold',
                    'velocity_accuracy_threshold', 'pitch_accuracy_threshold']:
            val1 = getattr(thresholds1, attr)
            val2 = getattr(thresholds2, attr)
            
            if val1 != val2:
                comparison["differences"][attr] = {
                    "threshold1": val1,
                    "threshold2": val2,
                    "difference": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
        
        # Generate summary
        if comparison["strictness_comparison"] > 0:
            comparison["summary"] = "Threshold set 1 is stricter"
        elif comparison["strictness_comparison"] < 0:
            comparison["summary"] = "Threshold set 2 is stricter"
        else:
            comparison["summary"] = "Threshold sets are equally strict"
        
        return comparison
    
    def _compare_strictness(
        self,
        thresholds1: AdaptiveThresholds,
        thresholds2: AdaptiveThresholds
    ) -> float:
        """
        Compare overall strictness of two threshold sets.
        
        Args:
            thresholds1: First threshold set
            thresholds2: Second threshold set
            
        Returns:
            Strictness comparison (-1 to 1, positive means thresholds1 is stricter)
        """
        strictness1 = 0.0
        strictness2 = 0.0
        
        # Lower tolerances = stricter
        strictness1 -= (thresholds1.note_start_tick + thresholds1.note_duration + 
                       thresholds1.velocity_bin) / 10
        strictness2 -= (thresholds2.note_start_tick + thresholds2.note_duration + 
                       thresholds2.velocity_bin) / 10
        
        # Lower ratios = stricter
        strictness1 -= (thresholds1.missing_notes_ratio + thresholds1.extra_notes_ratio) * 10
        strictness2 -= (thresholds2.missing_notes_ratio + thresholds2.extra_notes_ratio) * 10
        
        # Higher quality thresholds = stricter
        strictness1 += (thresholds1.overall_quality_threshold + 
                       thresholds1.timing_accuracy_threshold +
                       thresholds1.velocity_accuracy_threshold +
                       thresholds1.pitch_accuracy_threshold)
        strictness2 += (thresholds2.overall_quality_threshold + 
                       thresholds2.timing_accuracy_threshold +
                       thresholds2.velocity_accuracy_threshold +
                       thresholds2.pitch_accuracy_threshold)
        
        return strictness1 - strictness2
    
    def optimize_for_content(
        self,
        track_infos: List[TrackInfo],
        validation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AdaptiveThresholds:
        """
        Optimize thresholds based on content and historical validation results.
        
        Args:
            track_infos: Track analysis information
            validation_history: Previous validation results for similar content
            
        Returns:
            Optimized thresholds
        """
        # Calculate complexity
        complexity = self._calculate_complexity(track_infos, None)
        
        # Start with complexity-appropriate base
        if complexity.complexity_level == ComplexityLevel.MINIMAL:
            use_case = UseCase.RESEARCH
        elif complexity.complexity_level == ComplexityLevel.SIMPLE:
            use_case = UseCase.ARCHIVAL
        elif complexity.complexity_level == ComplexityLevel.MODERATE:
            use_case = UseCase.PRODUCTION
        elif complexity.complexity_level == ComplexityLevel.COMPLEX:
            use_case = UseCase.GENERATION
        else:
            use_case = UseCase.REAL_TIME
        
        # Get base thresholds
        thresholds = self.calculate_adaptive_thresholds(track_infos, None, use_case)
        
        # Apply learning from validation history
        if validation_history:
            self._apply_historical_learning(thresholds, validation_history, complexity)
        
        # Fine-tune based on specific content characteristics
        self._fine_tune_for_content(thresholds, track_infos)
        
        return thresholds
    
    def _apply_historical_learning(
        self,
        thresholds: AdaptiveThresholds,
        history: List[Dict[str, Any]],
        complexity: ComplexityMetrics
    ) -> None:
        """
        Apply learning from historical validation results.
        
        Args:
            thresholds: Thresholds to adjust
            history: Validation history
            complexity: Current complexity metrics
        """
        # Find similar complexity validations
        similar_validations = []
        for result in history:
            if 'complexity' in result:
                complexity_diff = abs(result['complexity'] - complexity.overall_complexity)
                if complexity_diff < 0.2:  # Similar complexity
                    similar_validations.append(result)
        
        if not similar_validations:
            return
        
        # Calculate average success rates
        success_rates = [v.get('accuracy', 0) for v in similar_validations]
        avg_success = statistics.mean(success_rates) if success_rates else 0
        
        # Adjust thresholds based on historical success
        if avg_success < 0.9:
            # Relax thresholds if historical accuracy is low
            relax_factor = 1.0 + (0.9 - avg_success)
            thresholds.note_start_tick = int(thresholds.note_start_tick * relax_factor)
            thresholds.note_duration = int(thresholds.note_duration * relax_factor)
            thresholds.missing_notes_ratio *= relax_factor
            thresholds.adjustment_reasons.append(
                f"Relaxed thresholds based on historical accuracy ({avg_success:.1%})"
            )
        elif avg_success > 0.98:
            # Tighten thresholds if historical accuracy is very high
            tighten_factor = 0.9
            thresholds.note_start_tick = max(1, int(thresholds.note_start_tick * tighten_factor))
            thresholds.note_duration = max(1, int(thresholds.note_duration * tighten_factor))
            thresholds.adjustment_reasons.append(
                f"Tightened thresholds based on excellent historical accuracy ({avg_success:.1%})"
            )
    
    def _fine_tune_for_content(
        self,
        thresholds: AdaptiveThresholds,
        track_infos: List[TrackInfo]
    ) -> None:
        """
        Fine-tune thresholds based on specific content characteristics.
        
        Args:
            thresholds: Thresholds to adjust
            track_infos: Track analysis information
        """
        # Check for specific track types
        has_drums = any(t.type == "drums" for t in track_infos)
        has_complex_chords = any(t.type == "chord" and t.statistics.avg_polyphony > 4 
                                for t in track_infos)
        has_fast_passages = any(t.statistics.note_density > 5 for t in track_infos 
                               if hasattr(t.statistics, 'note_density'))
        
        # Adjust for drums (timing critical, velocity less so)
        if has_drums:
            thresholds.note_start_tick = max(1, thresholds.note_start_tick - 1)
            thresholds.velocity_bin = thresholds.velocity_bin + 1
            thresholds.adjustment_reasons.append("Adjusted for drum track presence")
        
        # Adjust for complex chords
        if has_complex_chords:
            thresholds.note_duration = thresholds.note_duration + 1
            thresholds.missing_notes_ratio *= 1.2
            thresholds.adjustment_reasons.append("Relaxed for complex chord voicings")
        
        # Adjust for fast passages
        if has_fast_passages:
            thresholds.note_start_tick = thresholds.note_start_tick + 1
            thresholds.timing_accuracy_threshold *= 0.98
            thresholds.adjustment_reasons.append("Adjusted for fast passages")