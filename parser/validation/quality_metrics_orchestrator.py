"""
Quality metrics orchestrator for comprehensive MIDI validation.

This module coordinates all specialized quality analysis components to provide
a unified quality assessment, combining musical feature preservation, sequence
quality, statistical similarity, and adaptive threshold management into a
single comprehensive quality score.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from miditoolkit import MidiFile

from parser.config.defaults import MidiParserConfig, DEFAULT_CONFIG
from parser.core.track_analyzer import TrackInfo
from parser.core.tokenizer_manager import TokenizationResult
from parser.core.midi_loader import MidiMetadata

from .validation_metrics import RoundTripMetrics
from .musical_feature_analyzer import MusicalFeatureAnalyzer, MusicalFeatureMetrics
from .sequence_quality_analyzer import SequenceQualityAnalyzer, SequenceQualityMetrics
from .statistical_comparator import StatisticalComparator, StatisticalMetrics
from .adaptive_threshold_manager import (
    AdaptiveThresholdManager,
    AdaptiveThresholds,
    ComplexityLevel,
    UseCase
)

logger = logging.getLogger(__name__)


class QualityGate(Enum):
    """Quality gate levels for validation."""
    PRODUCTION = "production"      # Highest quality requirements
    STANDARD = "standard"          # Normal quality requirements
    PERMISSIVE = "permissive"      # Relaxed quality requirements
    EXPERIMENTAL = "experimental"  # Minimal quality requirements


@dataclass
class QualityRecommendation:
    """Actionable recommendations for quality improvement."""
    category: str
    severity: str  # "critical", "warning", "info"
    issue: str
    recommendation: str
    impact_score: float = 0.0  # 0-1, higher means more impact on quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "category": self.category,
            "severity": self.severity,
            "issue": self.issue,
            "recommendation": self.recommendation,
            "impact_score": round(self.impact_score, 3)
        }


@dataclass
class ComponentScores:
    """Individual component quality scores."""
    round_trip_accuracy: float = 0.0
    musical_fidelity: float = 0.0
    sequence_quality: float = 0.0
    statistical_similarity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "round_trip_accuracy": round(self.round_trip_accuracy, 4),
            "musical_fidelity": round(self.musical_fidelity, 4),
            "sequence_quality": round(self.sequence_quality, 4),
            "statistical_similarity": round(self.statistical_similarity, 4)
        }


@dataclass
class QualityBreakdown:
    """Detailed quality score breakdown."""
    component_scores: ComponentScores = field(default_factory=ComponentScores)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    category_scores: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    quality_factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "components": self.component_scores.to_dict(),
            "weighted": {k: round(v, 4) for k, v in self.weighted_scores.items()},
            "categories": {k: round(v, 4) for k, v in self.category_scores.items()},
            "confidence": round(self.confidence_score, 4),
            "factors": {k: round(v, 4) for k, v in self.quality_factors.items()}
        }


@dataclass
class ComprehensiveQualityMetrics:
    """
    Comprehensive quality metrics combining all analysis components.
    
    This is the main output of the quality orchestrator, containing the
    final quality score and all supporting metrics.
    """
    # Core quality score (0-1)
    overall_quality_score: float = 0.0
    
    # Quality gate assessment
    quality_gate_passed: bool = False
    quality_gate_level: QualityGate = QualityGate.STANDARD
    quality_gate_threshold: float = 0.0
    
    # Component metrics
    round_trip_metrics: Optional[RoundTripMetrics] = None
    musical_metrics: Optional[MusicalFeatureMetrics] = None
    sequence_metrics: Optional[SequenceQualityMetrics] = None
    statistical_metrics: Optional[StatisticalMetrics] = None
    
    # Adaptive thresholds
    adaptive_thresholds: Optional[AdaptiveThresholds] = None
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE
    
    # Quality breakdown
    quality_breakdown: QualityBreakdown = field(default_factory=QualityBreakdown)
    
    # Recommendations
    recommendations: List[QualityRecommendation] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    quality_warnings: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    analysis_completeness: float = 1.0  # Fraction of analyses completed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_quality_score": round(self.overall_quality_score, 4),
            "quality_gate": {
                "passed": self.quality_gate_passed,
                "level": self.quality_gate_level.value,
                "threshold": round(self.quality_gate_threshold, 3),
                "score_vs_threshold": round(self.overall_quality_score - self.quality_gate_threshold, 4)
            },
            "breakdown": self.quality_breakdown.to_dict(),
            "complexity_level": self.complexity_level.value,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "critical_issues": self.critical_issues,
            "warnings": self.quality_warnings,
            "analysis_completeness": round(self.analysis_completeness, 2),
            "processing_time": round(self.processing_time, 3)
        }


class QualityMetricsOrchestrator:
    """
    Main coordinator for comprehensive quality analysis.
    
    This class orchestrates all quality analysis components, combining their
    results into a unified quality assessment with actionable recommendations.
    """
    
    # Default quality gate thresholds
    QUALITY_GATE_THRESHOLDS = {
        QualityGate.PRODUCTION: 0.95,
        QualityGate.STANDARD: 0.90,
        QualityGate.PERMISSIVE: 0.80,
        QualityGate.EXPERIMENTAL: 0.60
    }
    
    # Default component weights for overall score
    DEFAULT_COMPONENT_WEIGHTS = {
        "round_trip": 0.30,
        "musical": 0.30,
        "sequence": 0.20,
        "statistical": 0.20
    }
    
    # Category weights for different use cases
    USE_CASE_WEIGHTS = {
        UseCase.RESEARCH: {
            "round_trip": 0.35,
            "musical": 0.25,
            "sequence": 0.20,
            "statistical": 0.20
        },
        UseCase.PRODUCTION: {
            "round_trip": 0.30,
            "musical": 0.30,
            "sequence": 0.20,
            "statistical": 0.20
        },
        UseCase.GENERATION: {
            "round_trip": 0.25,
            "musical": 0.35,
            "sequence": 0.25,
            "statistical": 0.15
        },
        UseCase.ARCHIVAL: {
            "round_trip": 0.40,
            "musical": 0.20,
            "sequence": 0.15,
            "statistical": 0.25
        }
    }
    
    def __init__(
        self,
        config: Optional[MidiParserConfig] = None,
        quality_gate: QualityGate = QualityGate.STANDARD
    ):
        """
        Initialize the quality metrics orchestrator.
        
        Args:
            config: Parser configuration
            quality_gate: Quality gate level to enforce
        """
        self.config = config or DEFAULT_CONFIG
        self.quality_gate = quality_gate
        
        # Initialize specialized analyzers
        self.musical_analyzer = MusicalFeatureAnalyzer(
            beat_resolution=self.config.tokenizer.beat_resolution
        )
        
        self.sequence_analyzer = SequenceQualityAnalyzer(
            tokenizer_config=self.config.tokenizer,
            strategy=self.config.tokenization
        )
        
        self.statistical_comparator = StatisticalComparator(
            confidence_level=0.95
        )
        
        self.threshold_manager = AdaptiveThresholdManager(
            base_config=self.config.validation
        )
        
        # Component weights (can be customized)
        self.component_weights = self.DEFAULT_COMPONENT_WEIGHTS.copy()
        
        logger.info(f"Quality orchestrator initialized with {quality_gate.value} gate")
    
    def perform_comprehensive_analysis(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        tokenization_result: Optional[TokenizationResult] = None,
        round_trip_metrics: Optional[RoundTripMetrics] = None,
        track_infos: Optional[List[TrackInfo]] = None,
        metadata: Optional[MidiMetadata] = None,
        use_case: UseCase = UseCase.PRODUCTION,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> ComprehensiveQualityMetrics:
        """
        Perform comprehensive quality analysis combining all components.
        
        This is the main entry point for quality assessment.
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file
            tokenization_result: Optional tokenization result for sequence analysis
            round_trip_metrics: Optional pre-calculated round-trip metrics
            track_infos: Optional track analysis information
            metadata: Optional MIDI metadata
            use_case: Target use case for weight adjustment
            custom_weights: Optional custom component weights
            
        Returns:
            ComprehensiveQualityMetrics with complete analysis
        """
        start_time = time.time()
        logger.info(f"Starting comprehensive quality analysis for {use_case.value} use case")
        
        metrics = ComprehensiveQualityMetrics(
            quality_gate_level=self.quality_gate,
            quality_gate_threshold=self.QUALITY_GATE_THRESHOLDS[self.quality_gate]
        )
        
        # Store round-trip metrics if provided
        metrics.round_trip_metrics = round_trip_metrics
        
        # Determine adaptive thresholds based on content
        if track_infos:
            metrics.adaptive_thresholds = self.threshold_manager.calculate_adaptive_thresholds(
                track_infos, metadata, use_case
            )
            
            # Extract complexity level
            complexity = self.threshold_manager._calculate_complexity(track_infos, metadata)
            metrics.complexity_level = complexity.complexity_level
        else:
            metrics.adaptive_thresholds = self.threshold_manager._create_default_thresholds()
            metrics.complexity_level = ComplexityLevel.MODERATE
        
        # Adjust component weights based on use case
        if custom_weights:
            self.component_weights = custom_weights
        else:
            self.component_weights = self.USE_CASE_WEIGHTS.get(
                use_case, self.DEFAULT_COMPONENT_WEIGHTS
            ).copy()
        
        # Track analysis completeness
        analyses_attempted = 0
        analyses_completed = 0
        
        # 1. Musical Feature Analysis
        try:
            analyses_attempted += 1
            metrics.musical_metrics = self.musical_analyzer.analyze_musical_preservation(
                original, reconstructed, track_infos, round_trip_metrics
            )
            metrics.quality_breakdown.component_scores.musical_fidelity = (
                metrics.musical_metrics.overall_musical_fidelity
            )
            analyses_completed += 1
            logger.info(f"Musical analysis complete: {metrics.musical_metrics.overall_musical_fidelity:.2%}")
        except Exception as e:
            logger.error(f"Musical feature analysis failed: {e}")
            metrics.quality_warnings.append(f"Musical analysis incomplete: {str(e)}")
        
        # 2. Sequence Quality Analysis
        if tokenization_result:
            try:
                analyses_attempted += 1
                original_note_count = sum(
                    len(inst.notes) for inst in original.instruments
                )
                metrics.sequence_metrics = self.sequence_analyzer.analyze_sequence_quality(
                    tokenization_result,
                    original_note_count,
                    tokenization_result.vocabulary
                )
                metrics.quality_breakdown.component_scores.sequence_quality = (
                    metrics.sequence_metrics.overall_sequence_quality
                )
                analyses_completed += 1
                logger.info(f"Sequence analysis complete: {metrics.sequence_metrics.overall_sequence_quality:.2%}")
            except Exception as e:
                logger.error(f"Sequence quality analysis failed: {e}")
                metrics.quality_warnings.append(f"Sequence analysis incomplete: {str(e)}")
        
        # 3. Statistical Comparison
        try:
            analyses_attempted += 1
            metrics.statistical_metrics = self.statistical_comparator.perform_statistical_comparison(
                original, reconstructed, round_trip_metrics
            )
            metrics.quality_breakdown.component_scores.statistical_similarity = (
                metrics.statistical_metrics.overall_statistical_similarity
            )
            analyses_completed += 1
            logger.info(f"Statistical analysis complete: {metrics.statistical_metrics.overall_statistical_similarity:.2%}")
        except Exception as e:
            logger.error(f"Statistical comparison failed: {e}")
            metrics.quality_warnings.append(f"Statistical analysis incomplete: {str(e)}")
        
        # 4. Round-trip accuracy (if provided)
        if round_trip_metrics:
            metrics.quality_breakdown.component_scores.round_trip_accuracy = (
                round_trip_metrics.overall_accuracy
            )
        
        # Calculate analysis completeness
        if analyses_attempted > 0:
            metrics.analysis_completeness = analyses_completed / analyses_attempted
        
        # Calculate overall quality score
        metrics.overall_quality_score = self._calculate_overall_score(
            metrics.quality_breakdown.component_scores,
            metrics.analysis_completeness
        )
        
        # Populate quality breakdown
        self._populate_quality_breakdown(metrics)
        
        # Check quality gate
        metrics.quality_gate_passed = (
            metrics.overall_quality_score >= metrics.quality_gate_threshold
        )
        
        # Generate recommendations
        metrics.recommendations = self._generate_recommendations(metrics)
        
        # Identify critical issues
        metrics.critical_issues = self._identify_critical_issues(metrics)
        
        # Collect all warnings
        metrics.quality_warnings.extend(self._collect_all_warnings(metrics))
        
        # Record processing time
        metrics.processing_time = time.time() - start_time
        
        logger.info(f"Comprehensive analysis complete. Overall quality: {metrics.overall_quality_score:.2%}")
        logger.info(f"Quality gate {'PASSED' if metrics.quality_gate_passed else 'FAILED'} "
                   f"({self.quality_gate.value}: {metrics.quality_gate_threshold:.2%})")
        
        return metrics
    
    def _calculate_overall_score(
        self,
        component_scores: ComponentScores,
        completeness: float
    ) -> float:
        """
        Calculate weighted overall quality score.
        
        Args:
            component_scores: Individual component scores
            completeness: Analysis completeness factor
            
        Returns:
            Overall quality score (0-1)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Add weighted scores for available components
        if component_scores.round_trip_accuracy > 0:
            weighted_sum += component_scores.round_trip_accuracy * self.component_weights["round_trip"]
            total_weight += self.component_weights["round_trip"]
        
        if component_scores.musical_fidelity > 0:
            weighted_sum += component_scores.musical_fidelity * self.component_weights["musical"]
            total_weight += self.component_weights["musical"]
        
        if component_scores.sequence_quality > 0:
            weighted_sum += component_scores.sequence_quality * self.component_weights["sequence"]
            total_weight += self.component_weights["sequence"]
        
        if component_scores.statistical_similarity > 0:
            weighted_sum += component_scores.statistical_similarity * self.component_weights["statistical"]
            total_weight += self.component_weights["statistical"]
        
        if total_weight == 0:
            return 0.0
        
        # Calculate base score
        base_score = weighted_sum / total_weight
        
        # Apply completeness penalty (minor penalty for incomplete analysis)
        completeness_factor = 0.95 + 0.05 * completeness  # 95-100% of score based on completeness
        
        overall_score = base_score * completeness_factor
        
        return max(0.0, min(1.0, overall_score))
    
    def _populate_quality_breakdown(self, metrics: ComprehensiveQualityMetrics) -> None:
        """
        Populate detailed quality breakdown.
        
        Args:
            metrics: Metrics object to populate
        """
        breakdown = metrics.quality_breakdown
        
        # Store weighted scores
        scores = metrics.quality_breakdown.component_scores
        breakdown.weighted_scores = {
            "round_trip": scores.round_trip_accuracy * self.component_weights.get("round_trip", 0),
            "musical": scores.musical_fidelity * self.component_weights.get("musical", 0),
            "sequence": scores.sequence_quality * self.component_weights.get("sequence", 0),
            "statistical": scores.statistical_similarity * self.component_weights.get("statistical", 0)
        }
        
        # Calculate category scores
        breakdown.category_scores = {
            "accuracy": (scores.round_trip_accuracy * 0.6 + scores.statistical_similarity * 0.4),
            "musicality": (scores.musical_fidelity * 0.7 + scores.sequence_quality * 0.3),
            "technical": (scores.sequence_quality * 0.5 + scores.statistical_similarity * 0.5),
            "preservation": (scores.round_trip_accuracy * 0.5 + scores.musical_fidelity * 0.5)
        }
        
        # Calculate confidence score
        breakdown.confidence_score = self._calculate_confidence_score(metrics)
        
        # Extract quality factors
        breakdown.quality_factors = self._extract_quality_factors(metrics)
    
    def _calculate_confidence_score(self, metrics: ComprehensiveQualityMetrics) -> float:
        """
        Calculate confidence in the quality assessment.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Reduce confidence for incomplete analysis
        confidence *= metrics.analysis_completeness
        
        # Reduce confidence for extreme complexity
        if metrics.complexity_level == ComplexityLevel.VERY_COMPLEX:
            confidence *= 0.9
        elif metrics.complexity_level == ComplexityLevel.MINIMAL:
            confidence *= 0.95
        
        # Reduce confidence for conflicting component scores
        scores = [
            metrics.quality_breakdown.component_scores.round_trip_accuracy,
            metrics.quality_breakdown.component_scores.musical_fidelity,
            metrics.quality_breakdown.component_scores.sequence_quality,
            metrics.quality_breakdown.component_scores.statistical_similarity
        ]
        valid_scores = [s for s in scores if s > 0]
        
        if len(valid_scores) > 1:
            import statistics
            score_variance = statistics.stdev(valid_scores)
            if score_variance > 0.2:  # High variance indicates disagreement
                confidence *= (1.0 - score_variance)
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_quality_factors(self, metrics: ComprehensiveQualityMetrics) -> Dict[str, float]:
        """
        Extract key quality factors from detailed metrics.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            Dictionary of quality factors
        """
        factors = {}
        
        # Extract from musical metrics
        if metrics.musical_metrics:
            factors["pitch_preservation"] = metrics.musical_metrics.pitch_analysis.distribution_similarity
            factors["rhythm_preservation"] = metrics.musical_metrics.rhythm_analysis.rhythm_similarity
            factors["harmony_preservation"] = metrics.musical_metrics.harmonic_analysis.chord_similarity
        
        # Extract from sequence metrics
        if metrics.sequence_metrics:
            factors["vocabulary_efficiency"] = metrics.sequence_metrics.vocabulary_analysis.vocabulary_efficiency
            factors["sequence_coherence"] = metrics.sequence_metrics.coherence_analysis.structural_coherence
            factors["compression_efficiency"] = metrics.sequence_metrics.compression_analysis.encoding_efficiency
        
        # Extract from statistical metrics
        if metrics.statistical_metrics:
            factors["distribution_match"] = sum(
                comp.distribution_similarity_score 
                for comp in metrics.statistical_metrics.distribution_comparison.values()
            ) / max(len(metrics.statistical_metrics.distribution_comparison), 1)
            factors["correlation_strength"] = abs(metrics.statistical_metrics.correlation_analysis.pitch_correlation)
        
        # Extract from round-trip metrics
        if metrics.round_trip_metrics:
            factors["note_accuracy"] = 1.0 - metrics.round_trip_metrics.missing_notes_ratio
            factors["timing_accuracy"] = metrics.round_trip_metrics.timing_accuracy
            factors["velocity_accuracy"] = metrics.round_trip_metrics.velocity_accuracy
        
        return factors
    
    def _generate_recommendations(self, metrics: ComprehensiveQualityMetrics) -> List[QualityRecommendation]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            List of quality recommendations
        """
        recommendations = []
        
        # Round-trip recommendations
        if metrics.round_trip_metrics:
            if metrics.round_trip_metrics.missing_notes_ratio > 0.01:
                recommendations.append(QualityRecommendation(
                    category="round_trip",
                    severity="critical" if metrics.round_trip_metrics.missing_notes_ratio > 0.05 else "warning",
                    issue=f"High missing notes ratio: {metrics.round_trip_metrics.missing_notes_ratio:.2%}",
                    recommendation="Adjust tokenization parameters or try alternative strategy",
                    impact_score=min(1.0, metrics.round_trip_metrics.missing_notes_ratio * 10)
                ))
            
            if metrics.round_trip_metrics.timing_accuracy < 0.95:
                recommendations.append(QualityRecommendation(
                    category="round_trip",
                    severity="warning",
                    issue=f"Poor timing accuracy: {metrics.round_trip_metrics.timing_accuracy:.2%}",
                    recommendation="Increase beat resolution or check time signature handling",
                    impact_score=1.0 - metrics.round_trip_metrics.timing_accuracy
                ))
        
        # Musical recommendations
        if metrics.musical_metrics:
            for warning in metrics.musical_metrics.preservation_warnings:
                severity = "critical" if "poorly preserved" in warning.lower() else "warning"
                recommendations.append(QualityRecommendation(
                    category="musical",
                    severity=severity,
                    issue=warning,
                    recommendation=self._get_musical_recommendation(warning),
                    impact_score=0.7 if severity == "critical" else 0.4
                ))
        
        # Sequence recommendations
        if metrics.sequence_metrics:
            for suggestion in metrics.sequence_metrics.optimization_suggestions[:5]:  # Top 5
                recommendations.append(QualityRecommendation(
                    category="sequence",
                    severity="info",
                    issue="Sequence optimization opportunity",
                    recommendation=suggestion,
                    impact_score=0.3
                ))
        
        # Statistical recommendations
        if metrics.statistical_metrics:
            for warning in metrics.statistical_metrics.statistical_warnings[:3]:  # Top 3
                recommendations.append(QualityRecommendation(
                    category="statistical",
                    severity="warning",
                    issue=warning,
                    recommendation=self._get_statistical_recommendation(warning),
                    impact_score=0.5
                ))
        
        # Quality gate recommendations
        if not metrics.quality_gate_passed:
            deficit = metrics.quality_gate_threshold - metrics.overall_quality_score
            recommendations.append(QualityRecommendation(
                category="quality_gate",
                severity="critical",
                issue=f"Quality gate failed: {deficit:.2%} below threshold",
                recommendation=self._get_quality_gate_recommendation(metrics, deficit),
                impact_score=min(1.0, deficit * 2)
            ))
        
        # Sort by impact score (highest first)
        recommendations.sort(key=lambda r: r.impact_score, reverse=True)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _get_musical_recommendation(self, warning: str) -> str:
        """Get recommendation for musical warning."""
        warning_lower = warning.lower()
        
        if "pitch distribution" in warning_lower:
            return "Check pitch quantization and range settings in tokenizer configuration"
        elif "rhythm" in warning_lower:
            return "Adjust beat resolution or consider using a rhythm-focused tokenization strategy"
        elif "chord" in warning_lower or "harmony" in warning_lower:
            return "Enable chord tokens and consider using Structured tokenization for better harmony"
        elif "tempo" in warning_lower:
            return "Enable tempo tokens and verify tempo change handling"
        elif "dynamics" in warning_lower or "velocity" in warning_lower:
            return "Increase velocity quantization levels for better dynamic preservation"
        else:
            return "Review tokenization strategy and consider alternatives for better musical preservation"
    
    def _get_statistical_recommendation(self, warning: str) -> str:
        """Get recommendation for statistical warning."""
        warning_lower = warning.lower()
        
        if "distribution" in warning_lower:
            return "Consider adjusting quantization parameters to better match original distribution"
        elif "correlation" in warning_lower:
            return "Check for systematic biases in tokenization/detokenization process"
        elif "outlier" in warning_lower:
            return "Review handling of extreme values and edge cases"
        elif "trend" in warning_lower or "temporal" in warning_lower:
            return "Verify temporal structure preservation in tokenization strategy"
        else:
            return "Investigate statistical discrepancies and adjust tolerance thresholds if needed"
    
    def _get_quality_gate_recommendation(self, metrics: ComprehensiveQualityMetrics, deficit: float) -> str:
        """Get recommendation for quality gate failure."""
        # Find weakest component
        scores = metrics.quality_breakdown.component_scores
        components = {
            "round_trip": scores.round_trip_accuracy,
            "musical": scores.musical_fidelity,
            "sequence": scores.sequence_quality,
            "statistical": scores.statistical_similarity
        }
        
        weakest = min(components.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))
        
        if deficit < 0.05:
            return f"Minor improvements needed. Focus on {weakest[0]} component (currently {weakest[1]:.2%})"
        elif deficit < 0.10:
            return f"Moderate improvements required. Primary focus: {weakest[0]} component. Consider alternative tokenization strategy"
        else:
            return f"Significant improvements required. {weakest[0]} component critically low. Recommend complete reconfiguration"
    
    def _identify_critical_issues(self, metrics: ComprehensiveQualityMetrics) -> List[str]:
        """
        Identify critical issues that need immediate attention.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            List of critical issues
        """
        issues = []
        
        # Check component scores
        scores = metrics.quality_breakdown.component_scores
        
        if scores.round_trip_accuracy > 0 and scores.round_trip_accuracy < 0.80:
            issues.append(f"Critical: Round-trip accuracy below 80% ({scores.round_trip_accuracy:.1%})")
        
        if scores.musical_fidelity > 0 and scores.musical_fidelity < 0.70:
            issues.append(f"Critical: Musical fidelity severely degraded ({scores.musical_fidelity:.1%})")
        
        if scores.sequence_quality > 0 and scores.sequence_quality < 0.60:
            issues.append(f"Critical: Poor sequence quality ({scores.sequence_quality:.1%})")
        
        if scores.statistical_similarity > 0 and scores.statistical_similarity < 0.70:
            issues.append(f"Critical: Statistical properties not preserved ({scores.statistical_similarity:.1%})")
        
        # Check specific critical metrics
        if metrics.round_trip_metrics:
            if metrics.round_trip_metrics.missing_notes_ratio > 0.10:
                issues.append(f"Critical: {metrics.round_trip_metrics.missing_notes_ratio:.1%} of notes missing")
            
            if metrics.round_trip_metrics.timing_accuracy < 0.90:
                issues.append(f"Critical: Severe timing degradation ({metrics.round_trip_metrics.timing_accuracy:.1%})")
        
        # Check analysis completeness
        if metrics.analysis_completeness < 0.5:
            issues.append(f"Critical: Analysis only {metrics.analysis_completeness:.0%} complete")
        
        return issues
    
    def _collect_all_warnings(self, metrics: ComprehensiveQualityMetrics) -> List[str]:
        """
        Collect warnings from all components.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            Consolidated list of warnings
        """
        warnings = []
        
        # Collect from musical metrics
        if metrics.musical_metrics:
            warnings.extend(metrics.musical_metrics.preservation_warnings[:3])
        
        # Collect from sequence metrics
        if metrics.sequence_metrics:
            warnings.extend(metrics.sequence_metrics.quality_warnings[:3])
        
        # Collect from statistical metrics
        if metrics.statistical_metrics:
            warnings.extend(metrics.statistical_metrics.statistical_warnings[:3])
        
        # Add quality-specific warnings
        if metrics.overall_quality_score < 0.70:
            warnings.append("Overall quality below acceptable threshold")
        
        if metrics.quality_breakdown.confidence_score < 0.70:
            warnings.append("Low confidence in quality assessment - results may be unreliable")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_warnings = []
        for warning in warnings:
            if warning not in seen:
                seen.add(warning)
                unique_warnings.append(warning)
        
        return unique_warnings
    
    def set_quality_gate(self, gate: QualityGate) -> None:
        """
        Update the quality gate level.
        
        Args:
            gate: New quality gate level
        """
        self.quality_gate = gate
        logger.info(f"Quality gate set to {gate.value}")
    
    def set_component_weights(
        self,
        weights: Dict[str, float],
        normalize: bool = True
    ) -> None:
        """
        Set custom component weights for overall score calculation.
        
        Args:
            weights: Dictionary of component weights
            normalize: Whether to normalize weights to sum to 1.0
        """
        if normalize:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
        
        self.component_weights = weights
        logger.info(f"Component weights updated: {weights}")
    
    def get_quality_summary(
        self,
        metrics: ComprehensiveQualityMetrics
    ) -> Dict[str, Any]:
        """
        Generate a concise quality summary for reporting.
        
        Args:
            metrics: Comprehensive quality metrics
            
        Returns:
            Dictionary with summary information
        """
        return {
            "score": round(metrics.overall_quality_score, 3),
            "gate_passed": metrics.quality_gate_passed,
            "gate_level": metrics.quality_gate_level.value,
            "complexity": metrics.complexity_level.value,
            "confidence": round(metrics.quality_breakdown.confidence_score, 3),
            "components": {
                "round_trip": round(metrics.quality_breakdown.component_scores.round_trip_accuracy, 3),
                "musical": round(metrics.quality_breakdown.component_scores.musical_fidelity, 3),
                "sequence": round(metrics.quality_breakdown.component_scores.sequence_quality, 3),
                "statistical": round(metrics.quality_breakdown.component_scores.statistical_similarity, 3)
            },
            "critical_issues": len(metrics.critical_issues),
            "warnings": len(metrics.quality_warnings),
            "top_recommendation": metrics.recommendations[0].recommendation if metrics.recommendations else None
        }
    
    def calculate_quality_score(
        self,
        round_trip_metrics: RoundTripMetrics,
        tokenization_result: Optional[TokenizationResult] = None,
        original: Optional[MidiFile] = None,
        reconstructed: Optional[MidiFile] = None
    ) -> float:
        """
        Simplified method to calculate quality score from round-trip metrics.
        
        This method provides a quick quality assessment when full analysis
        is not needed.
        
        Args:
            round_trip_metrics: Round-trip validation metrics
            tokenization_result: Optional tokenization result
            original: Optional original MIDI file
            reconstructed: Optional reconstructed MIDI file
            
        Returns:
            Quality score (0-1)
        """
        # Start with round-trip accuracy
        base_score = round_trip_metrics.overall_accuracy
        
        # Apply modifiers based on specific metrics
        modifiers = []
        
        # Missing notes penalty
        if round_trip_metrics.missing_notes_ratio > 0.01:
            modifiers.append(1.0 - round_trip_metrics.missing_notes_ratio * 2)
        
        # Timing accuracy bonus/penalty
        if round_trip_metrics.timing_accuracy < 0.98:
            modifiers.append(round_trip_metrics.timing_accuracy)
        
        # Velocity accuracy modifier
        if round_trip_metrics.velocity_accuracy < 0.95:
            modifiers.append(round_trip_metrics.velocity_accuracy * 1.05)
        
        # Token efficiency modifier (if available)
        if tokenization_result:
            if hasattr(tokenization_result, 'sequence_length'):
                # Penalize very long sequences
                if tokenization_result.sequence_length > 2000:
                    modifiers.append(0.95)
                # Bonus for efficient encoding
                elif tokenization_result.sequence_length < 500:
                    modifiers.append(1.02)
        
        # Apply modifiers
        if modifiers:
            import statistics
            modifier = statistics.mean(modifiers)
            quality_score = base_score * modifier
        else:
            quality_score = base_score
        
        return max(0.0, min(1.0, quality_score))
    
    def enforce_quality_gate(
        self,
        quality_score: float,
        gate: Optional[QualityGate] = None
    ) -> Tuple[bool, str]:
        """
        Check if quality score passes the specified gate.
        
        Args:
            quality_score: Quality score to check
            gate: Quality gate to enforce (uses instance default if None)
            
        Returns:
            Tuple of (passed, message)
        """
        gate = gate or self.quality_gate
        threshold = self.QUALITY_GATE_THRESHOLDS[gate]
        
        passed = quality_score >= threshold
        
        if passed:
            margin = quality_score - threshold
            message = f"Quality gate {gate.value} PASSED (score: {quality_score:.3f}, margin: +{margin:.3f})"
        else:
            deficit = threshold - quality_score
            message = f"Quality gate {gate.value} FAILED (score: {quality_score:.3f}, deficit: -{deficit:.3f})"
        
        return passed, message
    
    def generate_quality_report(
        self,
        metrics: ComprehensiveQualityMetrics,
        format: str = "text"
    ) -> str:
        """
        Generate a formatted quality report.
        
        Args:
            metrics: Comprehensive quality metrics
            format: Report format ("text", "markdown", "json")
            
        Returns:
            Formatted report string
        """
        if format == "json":
            import json
            return json.dumps(metrics.to_dict(), indent=2)
        
        elif format == "markdown":
            return self._generate_markdown_report(metrics)
        
        else:  # text format
            return self._generate_text_report(metrics)
    
    def _generate_markdown_report(self, metrics: ComprehensiveQualityMetrics) -> str:
        """Generate markdown formatted quality report."""
        lines = [
            "# Comprehensive Quality Analysis Report",
            "",
            f"## Overall Quality Score: {metrics.overall_quality_score:.1%}",
            "",
            f"**Quality Gate:** {metrics.quality_gate_level.value.upper()}",
            f"**Status:** {'✅ PASSED' if metrics.quality_gate_passed else '❌ FAILED'}",
            f"**Threshold:** {metrics.quality_gate_threshold:.1%}",
            f"**Complexity Level:** {metrics.complexity_level.value}",
            f"**Analysis Confidence:** {metrics.quality_breakdown.confidence_score:.1%}",
            "",
            "## Component Scores",
            ""
        ]
        
        # Component scores table
        scores = metrics.quality_breakdown.component_scores
        lines.extend([
            "| Component | Score | Weight | Contribution |",
            "|-----------|-------|--------|--------------|"
        ])
        
        for comp, weight in self.component_weights.items():
            score_attr = f"{comp}_accuracy" if comp == "round_trip" else f"{comp}_fidelity" if comp == "musical" else f"{comp}_quality" if comp == "sequence" else f"{comp}_similarity"
            score = getattr(scores, score_attr.replace(comp + "_", comp == "round_trip" and "round_trip_" or comp == "musical" and "musical_" or comp == "sequence" and "sequence_" or "statistical_"))
            contribution = score * weight
            lines.append(f"| {comp.replace('_', ' ').title()} | {score:.1%} | {weight:.0%} | {contribution:.1%} |")
        
        # Category scores
        lines.extend([
            "",
            "## Category Scores",
            ""
        ])
        
        for category, score in metrics.quality_breakdown.category_scores.items():
            lines.append(f"- **{category.title()}:** {score:.1%}")
        
        # Critical issues
        if metrics.critical_issues:
            lines.extend([
                "",
                "## 🚨 Critical Issues",
                ""
            ])
            for issue in metrics.critical_issues:
                lines.append(f"- {issue}")
        
        # Recommendations
        if metrics.recommendations:
            lines.extend([
                "",
                "## 📋 Recommendations",
                ""
            ])
            for i, rec in enumerate(metrics.recommendations[:5], 1):
                icon = "🔴" if rec.severity == "critical" else "🟡" if rec.severity == "warning" else "ℹ️"
                lines.append(f"{i}. {icon} **{rec.category.title()}:** {rec.recommendation}")
                lines.append(f"   - *Issue:* {rec.issue}")
                lines.append(f"   - *Impact:* {rec.impact_score:.0%}")
                lines.append("")
        
        # Warnings
        if metrics.quality_warnings:
            lines.extend([
                "",
                "## ⚠️ Warnings",
                ""
            ])
            for warning in metrics.quality_warnings[:5]:
                lines.append(f"- {warning}")
        
        # Processing info
        lines.extend([
            "",
            "---",
            f"*Analysis completed in {metrics.processing_time:.2f}s*",
            f"*Analysis completeness: {metrics.analysis_completeness:.0%}*"
        ])
        
        return "\n".join(lines)
    
    def _generate_text_report(self, metrics: ComprehensiveQualityMetrics) -> str:
        """Generate plain text formatted quality report."""
        lines = [
            "=" * 60,
            "COMPREHENSIVE QUALITY ANALYSIS REPORT",
            "=" * 60,
            f"Overall Quality Score: {metrics.overall_quality_score:.1%}",
            f"Quality Gate: {metrics.quality_gate_level.value.upper()}",
            f"Status: {'PASSED' if metrics.quality_gate_passed else 'FAILED'}",
            f"Threshold: {metrics.quality_gate_threshold:.1%}",
            f"Complexity: {metrics.complexity_level.value}",
            f"Confidence: {metrics.quality_breakdown.confidence_score:.1%}",
            "-" * 60,
            "COMPONENT SCORES:",
        ]
        
        scores = metrics.quality_breakdown.component_scores
        lines.extend([
            f"  Round-trip Accuracy: {scores.round_trip_accuracy:.1%}",
            f"  Musical Fidelity: {scores.musical_fidelity:.1%}",
            f"  Sequence Quality: {scores.sequence_quality:.1%}",
            f"  Statistical Similarity: {scores.statistical_similarity:.1%}",
            "-" * 60,
            "CATEGORY SCORES:",
        ])
        
        for category, score in metrics.quality_breakdown.category_scores.items():
            lines.append(f"  {category.title()}: {score:.1%}")
        
        if metrics.critical_issues:
            lines.extend([
                "-" * 60,
                "CRITICAL ISSUES:",
            ])
            for issue in metrics.critical_issues:
                lines.append(f"  ! {issue}")
        
        if metrics.recommendations:
            lines.extend([
                "-" * 60,
                "TOP RECOMMENDATIONS:",
            ])
            for i, rec in enumerate(metrics.recommendations[:3], 1):
                lines.append(f"  {i}. {rec.recommendation}")
                lines.append(f"     Issue: {rec.issue}")
        
        lines.extend([
            "-" * 60,
            f"Analysis Time: {metrics.processing_time:.2f}s",
            f"Analysis Completeness: {metrics.analysis_completeness:.0%}",
            "=" * 60
        ])
        
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_quality_score(
    original: MidiFile,
    reconstructed: MidiFile,
    round_trip_metrics: RoundTripMetrics,
    config: Optional[MidiParserConfig] = None,
    quality_gate: QualityGate = QualityGate.STANDARD
) -> float:
    """
    Convenience function to calculate quality score.
    
    Args:
        original: Original MIDI file
        reconstructed: Reconstructed MIDI file
        round_trip_metrics: Round-trip validation metrics
        config: Optional parser configuration
        quality_gate: Quality gate level
        
    Returns:
        Quality score (0-1)
    """
    orchestrator = QualityMetricsOrchestrator(config, quality_gate)
    
    # Simplified calculation using round-trip metrics
    return orchestrator.calculate_quality_score(
        round_trip_metrics,
        original=original,
        reconstructed=reconstructed
    )


def perform_quality_analysis(
    original: MidiFile,
    reconstructed: MidiFile,
    tokenization_result: Optional[TokenizationResult] = None,
    round_trip_metrics: Optional[RoundTripMetrics] = None,
    track_infos: Optional[List[TrackInfo]] = None,
    config: Optional[MidiParserConfig] = None,
    quality_gate: QualityGate = QualityGate.STANDARD,
    use_case: UseCase = UseCase.PRODUCTION
) -> ComprehensiveQualityMetrics:
    """
    Convenience function to perform comprehensive quality analysis.
    
    Args:
        original: Original MIDI file
        reconstructed: Reconstructed MIDI file
        tokenization_result: Optional tokenization result
        round_trip_metrics: Optional round-trip metrics
        track_infos: Optional track information
        config: Optional parser configuration
        quality_gate: Quality gate level
        use_case: Target use case
        
    Returns:
        ComprehensiveQualityMetrics with full analysis
    """
    orchestrator = QualityMetricsOrchestrator(config, quality_gate)
    
    return orchestrator.perform_comprehensive_analysis(
        original=original,
        reconstructed=reconstructed,
        tokenization_result=tokenization_result,
        round_trip_metrics=round_trip_metrics,
        track_infos=track_infos,
        use_case=use_case
    )


# Export main classes and functions
__all__ = [
    'QualityMetricsOrchestrator',
    'ComprehensiveQualityMetrics',
    'QualityGate',
    'QualityRecommendation',
    'QualityBreakdown',
    'ComponentScores',
    'calculate_quality_score',
    'perform_quality_analysis'
]