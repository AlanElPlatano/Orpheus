"""
Validation pipeline orchestrator for comprehensive MIDI validation workflow.

This module provides high-level coordination of the complete validation workflow,
combining round-trip validation with quality analysis, managing pipeline state,
error recovery, and progress tracking.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
from pathlib import Path

from miditoolkit import MidiFile

from midi_parser.config.defaults import MidiParserConfig, DEFAULT_CONFIG
from midi_parser.core.midi_loader import MidiMetadata, ValidationResult, load_and_validate_midi
from midi_parser.core.track_analyzer import TrackInfo, analyze_tracks
from midi_parser.core.tokenizer_manager import TokenizationResult, tokenize_midi

from midi_parser.validation.round_trip_validator import RoundTripValidator
from midi_parser.validation.validation_metrics import RoundTripMetrics
from midi_parser.validation.quality_metrics_orchestrator import (
    QualityMetricsOrchestrator,
    ComprehensiveQualityMetrics,
    QualityGate,
    UseCase
)

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Validation pipeline execution stages."""
    INITIALIZATION = "initialization"
    MIDI_LOADING = "midi_loading"
    TRACK_ANALYSIS = "track_analysis"
    TOKENIZATION = "tokenization"
    ROUND_TRIP_VALIDATION = "round_trip_validation"
    QUALITY_ANALYSIS = "quality_analysis"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(Enum):
    """Overall pipeline status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    success: bool = True
    duration: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "stage": self.stage.value,
            "success": self.success,
            "duration": round(self.duration, 3),
            "error_message": self.error_message,
            "warnings": self.warnings,
            "has_data": self.data is not None
        }


@dataclass
class PipelineState:
    """Current state of the validation pipeline."""
    status: PipelineStatus = PipelineStatus.NOT_STARTED
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    completed_stages: List[PipelineStage] = field(default_factory=list)
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    progress_percentage: float = 0.0
    
    # Pipeline data
    midi_file: Optional[MidiFile] = None
    metadata: Optional[MidiMetadata] = None
    track_infos: Optional[List[TrackInfo]] = None
    tokenization_result: Optional[TokenizationResult] = None
    round_trip_metrics: Optional[RoundTripMetrics] = None
    quality_metrics: Optional[ComprehensiveQualityMetrics] = None
    
    @property
    def total_duration(self) -> float:
        """Calculate total pipeline duration."""
        if self.start_time == 0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """Check if pipeline is complete."""
        return self.status in [
            PipelineStatus.COMPLETED_SUCCESS,
            PipelineStatus.COMPLETED_WITH_WARNINGS,
            PipelineStatus.FAILED,
            PipelineStatus.CANCELLED
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "completed_stages": [s.value for s in self.completed_stages],
            "progress_percentage": round(self.progress_percentage, 1),
            "total_duration": round(self.total_duration, 3),
            "stage_results": {
                stage.value: result.to_dict() 
                for stage, result in self.stage_results.items()
            }
        }


@dataclass
class ValidationPipelineConfig:
    """Configuration for validation pipeline execution."""
    enable_round_trip: bool = True
    enable_quality_analysis: bool = True
    enable_detailed_metrics: bool = True
    quality_gate: QualityGate = QualityGate.STANDARD
    use_case: UseCase = UseCase.PRODUCTION
    continue_on_warnings: bool = True
    continue_on_stage_failure: bool = False
    max_stage_retries: int = 1
    stage_timeout: Optional[float] = None  # Seconds per stage
    
    # Progress callback
    progress_callback: Optional[Callable[[PipelineState], None]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_round_trip": self.enable_round_trip,
            "enable_quality_analysis": self.enable_quality_analysis,
            "enable_detailed_metrics": self.enable_detailed_metrics,
            "quality_gate": self.quality_gate.value,
            "use_case": self.use_case.value,
            "continue_on_warnings": self.continue_on_warnings,
            "continue_on_stage_failure": self.continue_on_stage_failure,
            "max_stage_retries": self.max_stage_retries,
            "stage_timeout": self.stage_timeout
        }


@dataclass
class ValidationPipelineResult:
    """Complete result from validation pipeline execution."""
    success: bool = False
    pipeline_state: Optional[PipelineState] = None
    validation_passed: bool = False
    quality_passed: bool = False
    overall_score: float = 0.0
    
    # Detailed results
    round_trip_result: Optional[Tuple[ValidationResult, RoundTripMetrics]] = None
    quality_metrics: Optional[ComprehensiveQualityMetrics] = None
    
    # Summary
    total_duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        result = {
            "success": self.success,
            "validation_passed": self.validation_passed,
            "quality_passed": self.quality_passed,
            "overall_score": round(self.overall_score, 4),
            "total_duration": round(self.total_duration, 3),
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }
        
        if self.pipeline_state:
            result["pipeline_state"] = self.pipeline_state.to_dict()
        
        if self.round_trip_result:
            validation_result, metrics = self.round_trip_result
            result["round_trip"] = {
                "valid": validation_result.is_valid,
                "accuracy": round(metrics.overall_accuracy, 4),
                "metrics": metrics.to_dict()
            }
        
        if self.quality_metrics:
            result["quality"] = self.quality_metrics.to_dict()
        
        return result


class ValidationPipelineOrchestrator:
    """
    High-level orchestrator for complete validation workflow.
    
    This class coordinates round-trip validation and quality analysis,
    managing pipeline state, error recovery, and progress tracking.
    """
    
    # Stage weights for progress calculation
    STAGE_WEIGHTS = {
        PipelineStage.INITIALIZATION: 0.05,
        PipelineStage.MIDI_LOADING: 0.10,
        PipelineStage.TRACK_ANALYSIS: 0.10,
        PipelineStage.TOKENIZATION: 0.15,
        PipelineStage.ROUND_TRIP_VALIDATION: 0.30,
        PipelineStage.QUALITY_ANALYSIS: 0.25,
        PipelineStage.REPORT_GENERATION: 0.05
    }
    
    def __init__(
        self,
        config: Optional[MidiParserConfig] = None,
        pipeline_config: Optional[ValidationPipelineConfig] = None
    ):
        """
        Initialize the validation pipeline orchestrator.
        
        Args:
            config: Parser configuration
            pipeline_config: Pipeline-specific configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.pipeline_config = pipeline_config or ValidationPipelineConfig()
        
        # Initialize validators
        self.round_trip_validator = RoundTripValidator(self.config)
        self.quality_orchestrator = QualityMetricsOrchestrator(
            self.config,
            self.pipeline_config.quality_gate
        )
        
        # Pipeline state
        self.state = PipelineState()
        
        logger.info(f"ValidationPipelineOrchestrator initialized with "
                   f"{self.pipeline_config.quality_gate.value} quality gate")
    
    def validate_pipeline(
        self,
        midi_path: Path,
        strategy: Optional[str] = None
    ) -> ValidationPipelineResult:
        """
        Execute complete validation pipeline on a MIDI file.
        
        This is the main entry point that orchestrates the entire workflow.
        
        Args:
            midi_path: Path to MIDI file
            strategy: Tokenization strategy (uses config default if None)
            
        Returns:
            ValidationPipelineResult with complete results
        """
        logger.info(f"Starting validation pipeline for {midi_path}")
        
        # Initialize result
        result = ValidationPipelineResult()
        
        # Reset state
        self._initialize_pipeline()
        
        try:
            # Stage 1: Initialize
            self._execute_stage(
                PipelineStage.INITIALIZATION,
                lambda: self._stage_initialize(midi_path, strategy)
            )
            
            # Stage 2: Load MIDI
            self._execute_stage(
                PipelineStage.MIDI_LOADING,
                lambda: self._stage_load_midi(midi_path)
            )
            
            # Stage 3: Analyze tracks
            self._execute_stage(
                PipelineStage.TRACK_ANALYSIS,
                lambda: self._stage_analyze_tracks()
            )
            
            # Stage 4: Tokenize (if round-trip enabled)
            if self.pipeline_config.enable_round_trip:
                self._execute_stage(
                    PipelineStage.TOKENIZATION,
                    lambda: self._stage_tokenize(strategy)
                )
                
                # Stage 5: Round-trip validation
                self._execute_stage(
                    PipelineStage.ROUND_TRIP_VALIDATION,
                    lambda: self._stage_round_trip_validation(strategy)
                )
            
            # Stage 6: Quality analysis (if enabled)
            if self.pipeline_config.enable_quality_analysis:
                self._execute_stage(
                    PipelineStage.QUALITY_ANALYSIS,
                    lambda: self._stage_quality_analysis()
                )
            
            # Stage 7: Generate report
            self._execute_stage(
                PipelineStage.REPORT_GENERATION,
                lambda: self._stage_generate_report()
            )
            
            # Mark completion
            self.state.status = PipelineStatus.COMPLETED_SUCCESS
            self.state.current_stage = PipelineStage.COMPLETED
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            self.state.status = PipelineStatus.FAILED
            self.state.current_stage = PipelineStage.FAILED
            result.errors.append(f"Pipeline failed: {str(e)}")
        
        finally:
            self.state.end_time = time.time()
        
        # Build final result
        result = self._build_final_result()
        
        logger.info(f"Pipeline completed: {result.success}, score: {result.overall_score:.2%}")
        
        return result
    
    def _initialize_pipeline(self) -> None:
        """Initialize pipeline state."""
        self.state = PipelineState(
            status=PipelineStatus.IN_PROGRESS,
            current_stage=PipelineStage.INITIALIZATION,
            start_time=time.time()
        )
        self._update_progress()
    
    def _execute_stage(
        self,
        stage: PipelineStage,
        stage_func: Callable[[], Any],
        retry_count: int = 0
    ) -> StageResult:
        """
        Execute a single pipeline stage with error handling.
        
        Args:
            stage: Pipeline stage to execute
            stage_func: Function to execute
            retry_count: Current retry attempt
            
        Returns:
            StageResult with execution results
        """
        self.state.current_stage = stage
        logger.info(f"Executing stage: {stage.value}")
        
        start_time = time.time()
        result = StageResult(stage=stage)
        
        try:
            # Execute stage function
            result.data = stage_func()
            result.success = True
            
        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}", exc_info=True)
            result.success = False
            result.error_message = str(e)
            
            # Retry logic
            if retry_count < self.pipeline_config.max_stage_retries:
                logger.info(f"Retrying stage {stage.value} (attempt {retry_count + 1})")
                return self._execute_stage(stage, stage_func, retry_count + 1)
            
            # Check if should continue
            if not self.pipeline_config.continue_on_stage_failure:
                raise
        
        finally:
            result.duration = time.time() - start_time
            self.state.stage_results[stage] = result
            
            if result.success:
                self.state.completed_stages.append(stage)
            
            self._update_progress()
            
            # Call progress callback if provided
            if self.pipeline_config.progress_callback:
                self.pipeline_config.progress_callback(self.state)
        
        return result
    
    def _update_progress(self) -> None:
        """Update pipeline progress percentage."""
        total_weight = sum(self.STAGE_WEIGHTS.values())
        completed_weight = sum(
            self.STAGE_WEIGHTS.get(stage, 0)
            for stage in self.state.completed_stages
        )
        
        # Add partial progress for current stage
        if self.state.current_stage in self.STAGE_WEIGHTS:
            current_weight = self.STAGE_WEIGHTS[self.state.current_stage]
            # Assume 50% through current stage
            completed_weight += current_weight * 0.5
        
        self.state.progress_percentage = (completed_weight / total_weight) * 100
    
    # =========================================================================
    # Stage Implementation Methods
    # =========================================================================
    
    def _stage_initialize(self, midi_path: Path, strategy: Optional[str]) -> Dict[str, Any]:
        """Initialize stage: validate inputs and setup."""
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        if not midi_path.suffix.lower() in ['.mid', '.midi']:
            raise ValueError(f"Not a MIDI file: {midi_path}")
        
        strategy = strategy or self.config.tokenization
        
        logger.info(f"Pipeline initialized for {midi_path.name} with strategy {strategy}")
        
        return {
            "midi_path": midi_path,
            "strategy": strategy,
            "config": self.config
        }
    
    def _stage_load_midi(self, midi_path: Path) -> Dict[str, Any]:
        """Load and validate MIDI file."""
        midi, metadata, validation = load_and_validate_midi(midi_path, self.config)
        
        if not validation.is_valid:
            if not self.pipeline_config.continue_on_warnings:
                raise ValueError(f"MIDI validation failed: {validation.errors}")
            logger.warning(f"MIDI validation warnings: {validation.warnings}")
        
        self.state.midi_file = midi
        self.state.metadata = metadata
        
        logger.info(f"MIDI loaded: {metadata.note_count} notes, "
                   f"{metadata.duration_seconds:.1f}s, {metadata.track_count} tracks")
        
        return {
            "midi": midi,
            "metadata": metadata,
            "validation": validation
        }
    
    def _stage_analyze_tracks(self) -> List[TrackInfo]:
        """Analyze MIDI tracks."""
        if not self.state.midi_file:
            raise ValueError("No MIDI file loaded")
        
        track_infos = analyze_tracks(self.state.midi_file, self.config)
        self.state.track_infos = track_infos
        
        logger.info(f"Analyzed {len(track_infos)} tracks")
        
        return track_infos
    
    def _stage_tokenize(self, strategy: Optional[str]) -> TokenizationResult:
        """Tokenize MIDI file."""
        if not self.state.midi_file:
            raise ValueError("No MIDI file loaded")
        
        result = tokenize_midi(
            self.state.midi_file,
            config=self.config,
            track_infos=self.state.track_infos,
            strategy=strategy,
            auto_select=False
        )
        
        if not result.success:
            raise ValueError(f"Tokenization failed: {result.error_message}")
        
        self.state.tokenization_result = result
        
        logger.info(f"Tokenized: {result.sequence_length} tokens, "
                   f"vocab size: {result.vocabulary_size}")
        
        return result
    
    def _stage_round_trip_validation(self, strategy: Optional[str]) -> Tuple[ValidationResult, RoundTripMetrics]:
        """Perform round-trip validation."""
        if not self.state.midi_file:
            raise ValueError("No MIDI file loaded")
        
        validation_result, metrics = self.round_trip_validator.validate_round_trip(
            self.state.midi_file,
            strategy=strategy,
            track_infos=self.state.track_infos,
            detailed_report=self.pipeline_config.enable_detailed_metrics
        )
        
        self.state.round_trip_metrics = metrics
        
        logger.info(f"Round-trip validation: {validation_result.is_valid}, "
                   f"accuracy: {metrics.overall_accuracy:.2%}")
        
        return validation_result, metrics
    
    def _stage_quality_analysis(self) -> ComprehensiveQualityMetrics:
        """Perform quality analysis."""
        if not self.state.midi_file:
            logger.warning("No original MIDI file for quality analysis")
            return ComprehensiveQualityMetrics()
        
        # Get the reconstructed MIDI from the tokenization/detokenization process
        reconstructed = None
        
        # If we have tokenization result, reconstruct the MIDI
        if self.state.tokenization_result and self.state.tokenization_result.tokens:
            try:
                # Get the strategy used for tokenization
                strategy = self.state.tokenization_result.tokenization_strategy
                
                # Create tokenizer to perform detokenization
                from midi_parser.core.tokenizer_manager import TokenizerManager
                tokenizer_manager = TokenizerManager(self.config)
                tokenizer = tokenizer_manager.create_tokenizer(strategy)
                
                # Detokenize the tokens back to MIDI
                tokens = self.state.tokenization_result.tokens
                
                # Handle different MidiTok API versions
                if hasattr(tokenizer, 'tokens_to_midi'):
                    reconstructed = tokenizer.tokens_to_midi(tokens)
                elif hasattr(tokenizer, 'detokenize'):
                    reconstructed = tokenizer.detokenize(tokens)
                else:
                    # Fallback for older versions
                    reconstructed = tokenizer(tokens, _=None)
                
                # Ensure we have a MidiFile object
                if not isinstance(reconstructed, MidiFile):
                    if hasattr(reconstructed, 'to_midi'):
                        reconstructed = reconstructed.to_midi()
                    else:
                        logger.error(f"Unexpected detokenization output type: {type(reconstructed)}")
                        reconstructed = None
                        
            except Exception as e:
                logger.error(f"Failed to reconstruct MIDI for quality analysis: {e}")
                reconstructed = None
        
        # Alternative: Try to get reconstructed MIDI from round-trip validator cache
        # The round-trip validator should have already performed this detokenization
        if reconstructed is None and hasattr(self.round_trip_validator, '_last_reconstructed_midi'):
            reconstructed = self.round_trip_validator._last_reconstructed_midi
            logger.info("Using cached reconstructed MIDI from round-trip validator")
        
        # If we still don't have a reconstructed MIDI, we can't do quality analysis
        if reconstructed is None:
            logger.warning("No reconstructed MIDI available for quality analysis")
            # Return basic metrics without comparison
            return ComprehensiveQualityMetrics(
                overall_quality_score=0.0,
                quality_gate_passed=False,
                critical_issues=["Unable to reconstruct MIDI for quality analysis"]
            )
        
        # Now perform the actual quality analysis with both original and reconstructed
        quality_metrics = self.quality_orchestrator.perform_comprehensive_analysis(
            original=self.state.midi_file,
            reconstructed=reconstructed,
            tokenization_result=self.state.tokenization_result,
            round_trip_metrics=self.state.round_trip_metrics,
            track_infos=self.state.track_infos,
            metadata=self.state.metadata,
            use_case=self.pipeline_config.use_case
        )
        
        self.state.quality_metrics = quality_metrics
        
        logger.info(f"Quality analysis: {quality_metrics.overall_quality_score:.2%}, "
                f"gate {'PASSED' if quality_metrics.quality_gate_passed else 'FAILED'}")
        
        return quality_metrics
    
    def _stage_generate_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        report = {
            "pipeline_status": self.state.status.value,
            "total_duration": self.state.total_duration,
            "stages_completed": len(self.state.completed_stages),
            "validation_config": self.pipeline_config.to_dict()
        }
        
        if self.state.round_trip_metrics:
            report["round_trip_accuracy"] = self.state.round_trip_metrics.overall_accuracy
        
        if self.state.quality_metrics:
            report["quality_score"] = self.state.quality_metrics.overall_quality_score
            report["quality_gate_passed"] = self.state.quality_metrics.quality_gate_passed
        
        logger.info("Validation report generated")
        
        return report
    
    def _build_final_result(self) -> ValidationPipelineResult:
        """Build final pipeline result."""
        result = ValidationPipelineResult()
        result.pipeline_state = self.state
        result.total_duration = self.state.total_duration
        
        # Determine success
        result.success = self.state.status == PipelineStatus.COMPLETED_SUCCESS
        
        # Extract validation results
        if PipelineStage.ROUND_TRIP_VALIDATION in self.state.stage_results:
            stage_result = self.state.stage_results[PipelineStage.ROUND_TRIP_VALIDATION]
            if stage_result.success and stage_result.data:
                result.round_trip_result = stage_result.data
                validation_result, metrics = stage_result.data
                result.validation_passed = validation_result.is_valid
        
        # Extract quality metrics
        if self.state.quality_metrics:
            result.quality_metrics = self.state.quality_metrics
            result.quality_passed = self.state.quality_metrics.quality_gate_passed
        
        # Calculate overall score
        scores = []
        if self.state.round_trip_metrics:
            scores.append(self.state.round_trip_metrics.overall_accuracy)
        if self.state.quality_metrics:
            scores.append(self.state.quality_metrics.overall_quality_score)
        
        result.overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Collect errors and warnings
        for stage_result in self.state.stage_results.values():
            if stage_result.error_message:
                result.errors.append(f"{stage_result.stage.value}: {stage_result.error_message}")
            result.warnings.extend(stage_result.warnings)
        
        # Add quality recommendations
        if self.state.quality_metrics:
            result.recommendations = [
                rec.recommendation 
                for rec in self.state.quality_metrics.recommendations[:5]
            ]
        
        return result
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "status": self.state.status.value,
            "current_stage": self.state.current_stage.value,
            "progress": self.state.progress_percentage,
            "duration": self.state.total_duration,
            "completed_stages": [s.value for s in self.state.completed_stages]
        }
    
    def cancel_pipeline(self) -> None:
        """Cancel pipeline execution."""
        logger.warning("Pipeline cancelled")
        self.state.status = PipelineStatus.CANCELLED
        self.state.end_time = time.time()
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state for reuse."""
        self.state = PipelineState()
        logger.info("Pipeline reset")


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_file_complete(
    midi_path: Path,
    config: Optional[MidiParserConfig] = None,
    quality_gate: QualityGate = QualityGate.STANDARD,
    use_case: UseCase = UseCase.PRODUCTION
) -> ValidationPipelineResult:
    """
    Convenience function for complete file validation.
    
    Args:
        midi_path: Path to MIDI file
        config: Parser configuration
        quality_gate: Quality gate level
        use_case: Target use case
        
    Returns:
        ValidationPipelineResult with complete results
    """
    pipeline_config = ValidationPipelineConfig(
        enable_round_trip=True,
        enable_quality_analysis=True,
        quality_gate=quality_gate,
        use_case=use_case
    )
    
    orchestrator = ValidationPipelineOrchestrator(config, pipeline_config)
    return orchestrator.validate_pipeline(midi_path)


# Export main classes
__all__ = [
    'ValidationPipelineOrchestrator',
    'ValidationPipelineConfig',
    'ValidationPipelineResult',
    'PipelineState',
    'PipelineStage',
    'PipelineStatus',
    'validate_file_complete'
]
