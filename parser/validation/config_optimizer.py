"""
Configuration optimizer for intelligent parameter tuning.

This module provides intelligent configuration optimization based on validation
results and content analysis, offering auto-tuning recommendations, performance
optimization suggestions, and quality/speed trade-off analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import statistics

from parser.config.defaults import (
    MidiParserConfig,
    TokenizerConfig,
    ValidationConfig,
    ProcessingConfig,
    DEFAULT_CONFIG
)
from parser.validation.validation_metrics import RoundTripMetrics
from parser.validation.quality_metrics_orchestrator import ComprehensiveQualityMetrics
from parser.validation.adaptive_threshold_manager import ComplexityMetrics, ComplexityLevel
from parser.core.track_analyzer import TrackInfo

logger = logging.getLogger(__name__)


class OptimizationGoal(Enum):
    """Optimization goals for configuration tuning."""
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"
    MEMORY = "memory"
    THROUGHPUT = "throughput"


class OptimizationPriority(Enum):
    """Priority levels for optimization suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class OptimizationSuggestion:
    """Single configuration optimization suggestion."""
    parameter: str
    current_value: Any
    suggested_value: Any
    rationale: str
    priority: OptimizationPriority
    expected_improvement: float  # 0-1 scale
    trade_offs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "current": str(self.current_value),
            "suggested": str(self.suggested_value),
            "rationale": self.rationale,
            "priority": self.priority.value,
            "expected_improvement": round(self.expected_improvement, 2),
            "trade_offs": self.trade_offs
        }


@dataclass
class OptimizationResult:
    """Result of configuration optimization analysis."""
    optimization_goal: OptimizationGoal
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    optimized_config: Optional[MidiParserConfig] = None
    expected_quality_change: float = 0.0
    expected_speed_change: float = 0.0
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.optimization_goal.value,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "expected_changes": {
                "quality": round(self.expected_quality_change, 3),
                "speed": round(self.expected_speed_change, 3)
            },
            "confidence": round(self.confidence_score, 2)
        }


class ConfigurationOptimizer:
    """
    Intelligent configuration optimizer for validation and processing.
    
    Analyzes validation results and content characteristics to suggest
    optimal configuration parameters.
    """
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize configuration optimizer.
        
        Args:
            config: Base configuration to optimize from
        """
        self.config = config or DEFAULT_CONFIG
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("ConfigurationOptimizer initialized")
    
    def optimize_for_quality(
        self,
        validation_metrics: Optional[RoundTripMetrics] = None,
        quality_metrics: Optional[ComprehensiveQualityMetrics] = None,
        track_infos: Optional[List[TrackInfo]] = None
    ) -> OptimizationResult:
        """
        Optimize configuration for maximum quality.
        
        Args:
            validation_metrics: Round-trip validation metrics
            quality_metrics: Quality analysis metrics
            track_infos: Track analysis information
            
        Returns:
            OptimizationResult with quality-focused suggestions
        """
        logger.info("Optimizing configuration for quality")
        
        result = OptimizationResult(optimization_goal=OptimizationGoal.QUALITY)
        suggestions = []
        
        # Analyze validation results
        if validation_metrics:
            suggestions.extend(self._analyze_validation_quality(validation_metrics))
        
        # Analyze quality metrics
        if quality_metrics:
            suggestions.extend(self._analyze_quality_scores(quality_metrics))
        
        # Analyze content complexity
        if track_infos:
            suggestions.extend(self._analyze_content_requirements(track_infos))
        
        # Sort by priority and expected improvement
        suggestions.sort(
            key=lambda s: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[s.priority.value] * 
                s.expected_improvement
            ),
            reverse=True
        )
        
        result.suggestions = suggestions[:10]  # Top 10 suggestions
        result.confidence_score = self._calculate_confidence(suggestions, validation_metrics)
        
        # Estimate impact
        if suggestions:
            result.expected_quality_change = sum(s.expected_improvement for s in suggestions[:3]) / 3
            result.expected_speed_change = -0.1  # Quality usually costs speed
        
        # Generate optimized config
        result.optimized_config = self._apply_suggestions(suggestions[:5])
        
        return result
    
    def optimize_for_speed(
        self,
        processing_time: float,
        file_count: int = 1,
        track_infos: Optional[List[TrackInfo]] = None
    ) -> OptimizationResult:
        """
        Optimize configuration for faster processing.
        
        Args:
            processing_time: Current processing time in seconds
            file_count: Number of files processed
            track_infos: Track analysis information
            
        Returns:
            OptimizationResult with speed-focused suggestions
        """
        logger.info("Optimizing configuration for speed")
        
        result = OptimizationResult(optimization_goal=OptimizationGoal.SPEED)
        suggestions = []
        
        # Analyze processing performance
        avg_time = processing_time / max(file_count, 1)
        
        if avg_time > 10:  # Slow processing
            suggestions.extend(self._suggest_speed_improvements(avg_time))
        
        # Content-based optimizations
        if track_infos:
            suggestions.extend(self._suggest_complexity_reductions(track_infos))
        
        # Sort by expected improvement
        suggestions.sort(key=lambda s: s.expected_improvement, reverse=True)
        
        result.suggestions = suggestions[:10]
        result.confidence_score = 0.8  # Speed optimizations are generally reliable
        
        if suggestions:
            result.expected_speed_change = sum(s.expected_improvement for s in suggestions[:3]) / 3
            result.expected_quality_change = -0.05  # May reduce quality slightly
        
        result.optimized_config = self._apply_suggestions(suggestions[:5])
        
        return result
    
    def optimize_balanced(
        self,
        validation_metrics: Optional[RoundTripMetrics] = None,
        quality_metrics: Optional[ComprehensiveQualityMetrics] = None,
        processing_time: float = 0.0,
        track_infos: Optional[List[TrackInfo]] = None
    ) -> OptimizationResult:
        """
        Optimize configuration for balanced quality and speed.
        
        Args:
            validation_metrics: Validation metrics
            quality_metrics: Quality metrics
            processing_time: Processing time
            track_infos: Track information
            
        Returns:
            OptimizationResult with balanced suggestions
        """
        logger.info("Optimizing configuration for balanced performance")
        
        result = OptimizationResult(optimization_goal=OptimizationGoal.BALANCED)
        suggestions = []
        
        # Get quality suggestions
        if validation_metrics or quality_metrics:
            quality_suggestions = []
            if validation_metrics:
                quality_suggestions.extend(self._analyze_validation_quality(validation_metrics))
            if quality_metrics:
                quality_suggestions.extend(self._analyze_quality_scores(quality_metrics))
            
            # Keep only high-priority quality suggestions
            suggestions.extend([s for s in quality_suggestions if s.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]])
        
        # Get speed suggestions
        if processing_time > 5:
            speed_suggestions = self._suggest_speed_improvements(processing_time)
            # Keep only medium+ priority speed suggestions
            suggestions.extend([s for s in speed_suggestions if s.priority != OptimizationPriority.LOW])
        
        # Balance considerations
        suggestions.extend(self._suggest_balanced_parameters(track_infos))
        
        # Sort by balanced scoring
        suggestions.sort(
            key=lambda s: s.expected_improvement * 
                         (1.0 if s.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH] else 0.5),
            reverse=True
        )
        
        result.suggestions = suggestions[:10]
        result.confidence_score = 0.75
        
        if suggestions:
            result.expected_quality_change = 0.05
            result.expected_speed_change = 0.15
        
        result.optimized_config = self._apply_suggestions(suggestions[:5])
        
        return result
    
    def analyze_configuration(
        self,
        validation_metrics: Optional[RoundTripMetrics] = None,
        quality_metrics: Optional[ComprehensiveQualityMetrics] = None
    ) -> Dict[str, Any]:
        """
        Analyze current configuration effectiveness.
        
        Args:
            validation_metrics: Validation metrics
            quality_metrics: Quality metrics
            
        Returns:
            Configuration analysis report
        """
        analysis = {
            "configuration_health": "unknown",
            "issues": [],
            "strengths": [],
            "recommendations": []
        }
        
        # Tokenizer analysis
        tokenizer_issues = self._analyze_tokenizer_config()
        analysis["issues"].extend(tokenizer_issues)
        
        # Validation analysis
        if validation_metrics:
            if validation_metrics.overall_accuracy >= 0.95:
                analysis["strengths"].append("Excellent validation accuracy")
            elif validation_metrics.overall_accuracy < 0.85:
                analysis["issues"].append("Low validation accuracy")
                analysis["recommendations"].append("Consider adjusting tokenization strategy")
        
        # Quality analysis
        if quality_metrics:
            if quality_metrics.overall_quality_score >= 0.95:
                analysis["strengths"].append("High quality scores")
            elif quality_metrics.overall_quality_score < 0.80:
                analysis["issues"].append("Quality scores below target")
                analysis["recommendations"].append("Review quality thresholds and tokenizer settings")
        
        # Overall health
        if len(analysis["issues"]) == 0:
            analysis["configuration_health"] = "excellent"
        elif len(analysis["issues"]) <= 2:
            analysis["configuration_health"] = "good"
        elif len(analysis["issues"]) <= 4:
            analysis["configuration_health"] = "fair"
        else:
            analysis["configuration_health"] = "poor"
        
        return analysis
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def _analyze_validation_quality(self, metrics: RoundTripMetrics) -> List[OptimizationSuggestion]:
        """Analyze validation metrics for quality improvements."""
        suggestions = []
        
        # Missing notes issue
        if metrics.missing_notes_ratio > 0.02:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.beat_resolution",
                current_value=self.config.tokenizer.beat_resolution,
                suggested_value=min(8, self.config.tokenizer.beat_resolution * 2),
                rationale=f"High missing notes ratio ({metrics.missing_notes_ratio:.1%}), increase beat resolution",
                priority=OptimizationPriority.HIGH,
                expected_improvement=0.15,
                trade_offs=["Larger vocabulary", "Slower processing"]
            ))
        
        # Timing accuracy issue
        if metrics.timing_accuracy < 0.95:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.beat_resolution",
                current_value=self.config.tokenizer.beat_resolution,
                suggested_value=min(8, self.config.tokenizer.beat_resolution + 1),
                rationale=f"Poor timing accuracy ({metrics.timing_accuracy:.2%}), increase resolution",
                priority=OptimizationPriority.MEDIUM,
                expected_improvement=0.10,
                trade_offs=["Slightly larger vocabulary"]
            ))
        
        # Velocity accuracy issue
        if metrics.velocity_accuracy < 0.90:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.num_velocities",
                current_value=self.config.tokenizer.num_velocities,
                suggested_value=min(32, self.config.tokenizer.num_velocities * 2),
                rationale=f"Low velocity accuracy ({metrics.velocity_accuracy:.2%}), increase bins",
                priority=OptimizationPriority.MEDIUM,
                expected_improvement=0.12,
                trade_offs=["Larger vocabulary"]
            ))
        
        return suggestions
    
    def _analyze_quality_scores(self, metrics: ComprehensiveQualityMetrics) -> List[OptimizationSuggestion]:
        """Analyze quality metrics for improvements."""
        suggestions = []
        
        # Overall quality low
        if metrics.overall_quality_score < 0.85:
            suggestions.append(OptimizationSuggestion(
                parameter="validation.quality_threshold",
                current_value=self.config.validation.quality_threshold,
                suggested_value=0.80,
                rationale="Lower threshold to match actual performance",
                priority=OptimizationPriority.LOW,
                expected_improvement=0.05,
                trade_offs=["Accepts lower quality results"]
            ))
        
        # Musical fidelity issue
        if metrics.quality_breakdown.component_scores.musical_fidelity < 0.85:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.additional_tokens",
                current_value=self.config.tokenizer.additional_tokens,
                suggested_value={**self.config.tokenizer.additional_tokens, "Chord": True, "Tempo": True},
                rationale="Enable musical feature tokens for better preservation",
                priority=OptimizationPriority.HIGH,
                expected_improvement=0.15,
                trade_offs=["Larger vocabulary", "Slightly slower"]
            ))
        
        return suggestions
    
    def _analyze_content_requirements(self, track_infos: List[TrackInfo]) -> List[OptimizationSuggestion]:
        """Analyze content characteristics for optimization."""
        suggestions = []
        
        if not track_infos:
            return suggestions
        
        # Check complexity
        total_notes = sum(t.statistics.total_notes for t in track_infos)
        avg_polyphony = statistics.mean([t.statistics.avg_polyphony for t in track_infos if t.statistics.avg_polyphony > 0])
        
        # High complexity content
        if total_notes > 5000 or avg_polyphony > 4:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.max_seq_length",
                current_value=self.config.tokenizer.max_seq_length,
                suggested_value=4096,
                rationale="Complex content requires larger sequence length",
                priority=OptimizationPriority.MEDIUM,
                expected_improvement=0.10,
                trade_offs=["Higher memory usage"]
            ))
        
        # Check for specific track types
        has_drums = any(t.type == "drums" for t in track_infos)
        if has_drums and avg_polyphony > 3:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.beat_resolution",
                current_value=self.config.tokenizer.beat_resolution,
                suggested_value=min(8, self.config.tokenizer.beat_resolution + 2),
                rationale="Drum tracks with polyphony need higher resolution",
                priority=OptimizationPriority.MEDIUM,
                expected_improvement=0.12,
                trade_offs=["Larger vocabulary"]
            ))
        
        return suggestions
    
    def _suggest_speed_improvements(self, avg_time: float) -> List[OptimizationSuggestion]:
        """Suggest speed improvements."""
        suggestions = []
        
        # Enable parallel processing
        if not self.config.processing.parallel_processing:
            suggestions.append(OptimizationSuggestion(
                parameter="processing.parallel_processing",
                current_value=False,
                suggested_value=True,
                rationale="Enable parallel processing for faster batch operations",
                priority=OptimizationPriority.HIGH,
                expected_improvement=0.40,
                trade_offs=["Higher CPU usage"]
            ))
        
        # Reduce sequence length
        if self.config.tokenizer.max_seq_length > 2048:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.max_seq_length",
                current_value=self.config.tokenizer.max_seq_length,
                suggested_value=2048,
                rationale="Reduce max sequence length for faster processing",
                priority=OptimizationPriority.MEDIUM,
                expected_improvement=0.20,
                trade_offs=["May truncate long files"]
            ))
        
        # Reduce beat resolution if high
        if self.config.tokenizer.beat_resolution > 4:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.beat_resolution",
                current_value=self.config.tokenizer.beat_resolution,
                suggested_value=4,
                rationale="Lower beat resolution for faster tokenization",
                priority=OptimizationPriority.MEDIUM,
                expected_improvement=0.15,
                trade_offs=["Reduced timing precision"]
            ))
        
        return suggestions
    
    def _suggest_complexity_reductions(self, track_infos: List[TrackInfo]) -> List[OptimizationSuggestion]:
        """Suggest complexity reductions for speed."""
        suggestions = []
        
        # Many tracks
        if len(track_infos) > 8:
            suggestions.append(OptimizationSuggestion(
                parameter="processing.max_tracks",
                current_value=None,
                suggested_value=8,
                rationale="Limit track count for faster processing",
                priority=OptimizationPriority.LOW,
                expected_improvement=0.10,
                trade_offs=["Some tracks may be excluded"]
            ))
        
        return suggestions
    
    def _suggest_balanced_parameters(self, track_infos: Optional[List[TrackInfo]]) -> List[OptimizationSuggestion]:
        """Suggest balanced parameters."""
        suggestions = []
        
        # Balanced beat resolution
        if self.config.tokenizer.beat_resolution != 4:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.beat_resolution",
                current_value=self.config.tokenizer.beat_resolution,
                suggested_value=4,
                rationale="Standard resolution balances quality and speed",
                priority=OptimizationPriority.MEDIUM,
                expected_improvement=0.08,
                trade_offs=[]
            ))
        
        # Balanced velocity bins
        if self.config.tokenizer.num_velocities != 16:
            suggestions.append(OptimizationSuggestion(
                parameter="tokenizer.num_velocities",
                current_value=self.config.tokenizer.num_velocities,
                suggested_value=16,
                rationale="16 velocity bins provide good balance",
                priority=OptimizationPriority.LOW,
                expected_improvement=0.05,
                trade_offs=[]
            ))
        
        return suggestions
    
    def _analyze_tokenizer_config(self) -> List[str]:
        """Analyze tokenizer configuration for issues."""
        issues = []
        
        if self.config.tokenizer.beat_resolution > 8:
            issues.append("Very high beat resolution may cause performance issues")
        
        if self.config.tokenizer.num_velocities > 32:
            issues.append("High velocity quantization may be unnecessary")
        
        if self.config.tokenizer.max_seq_length < 512:
            issues.append("Low max sequence length may truncate content")
        
        return issues
    
    def _calculate_confidence(
        self,
        suggestions: List[OptimizationSuggestion],
        metrics: Optional[RoundTripMetrics]
    ) -> float:
        """Calculate confidence in suggestions."""
        if not suggestions:
            return 0.0
        
        # Base confidence on number and priority of suggestions
        confidence = 0.5
        
        # More high-priority suggestions = higher confidence
        high_priority = sum(1 for s in suggestions if s.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH])
        confidence += min(0.3, high_priority * 0.1)
        
        # Having metrics increases confidence
        if metrics:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _apply_suggestions(self, suggestions: List[OptimizationSuggestion]) -> MidiParserConfig:
        """Apply suggestions to create optimized config."""
        import copy
        optimized = copy.deepcopy(self.config)
        
        for suggestion in suggestions:
            parts = suggestion.parameter.split('.')
            
            if len(parts) == 2:
                section, param = parts
                section_obj = getattr(optimized, section, None)
                if section_obj and hasattr(section_obj, param):
                    setattr(section_obj, param, suggestion.suggested_value)
        
        return optimized
    
    def compare_configurations(
        self,
        config_a: MidiParserConfig,
        config_b: MidiParserConfig
    ) -> Dict[str, Any]:
        """Compare two configurations."""
        differences = []
        
        # Compare tokenizer settings
        if config_a.tokenizer.beat_resolution != config_b.tokenizer.beat_resolution:
            differences.append({
                "parameter": "tokenizer.beat_resolution",
                "config_a": config_a.tokenizer.beat_resolution,
                "config_b": config_b.tokenizer.beat_resolution
            })
        
        if config_a.tokenizer.num_velocities != config_b.tokenizer.num_velocities:
            differences.append({
                "parameter": "tokenizer.num_velocities",
                "config_a": config_a.tokenizer.num_velocities,
                "config_b": config_b.tokenizer.num_velocities
            })
        
        # Compare processing settings
        if config_a.processing.parallel_processing != config_b.processing.parallel_processing:
            differences.append({
                "parameter": "processing.parallel_processing",
                "config_a": config_a.processing.parallel_processing,
                "config_b": config_b.processing.parallel_processing
            })
        
        return {
            "differences": differences,
            "similarity": 1.0 - (len(differences) / 10)  # Rough similarity metric
        }


# Convenience functions

def optimize_configuration(
    config: MidiParserConfig,
    goal: OptimizationGoal = OptimizationGoal.BALANCED,
    validation_metrics: Optional[RoundTripMetrics] = None,
    quality_metrics: Optional[ComprehensiveQualityMetrics] = None
) -> OptimizationResult:
    """
    Optimize configuration for specified goal.
    
    Args:
        config: Current configuration
        goal: Optimization goal
        validation_metrics: Optional validation metrics
        quality_metrics: Optional quality metrics
        
    Returns:
        OptimizationResult with suggestions
    """
    optimizer = ConfigurationOptimizer(config)
    
    if goal == OptimizationGoal.QUALITY:
        return optimizer.optimize_for_quality(validation_metrics, quality_metrics)
    elif goal == OptimizationGoal.SPEED:
        return optimizer.optimize_for_speed(0.0)
    else:
        return optimizer.optimize_balanced(validation_metrics, quality_metrics)


# Export main classes
__all__ = [
    'ConfigurationOptimizer',
    'OptimizationResult',
    'OptimizationSuggestion',
    'OptimizationGoal',
    'OptimizationPriority',
    'optimize_configuration'
]
