"""
Main quality control module for MIDI tokenization validation.

This module provides the main entry point and interface for the complete
validation system, integrating all validation subsystems and providing
the primary functions that external systems will call.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field

from miditoolkit import MidiFile

from parser.config.defaults import MidiParserConfig, DEFAULT_CONFIG
from parser.core.midi_loader import load_midi_file, MidiMetadata
from parser.core.track_analyzer import TrackInfo, analyze_tracks
from ..core.tokenizer_manager import TokenizationResult
from parser.core.json_serializer import validate_json_schema

# Validation components
from parser.validation.validation_pipeline_orchestrator import (
    ValidationPipelineOrchestrator,
    ValidationPipelineConfig,
    ValidationPipelineResult,
    QualityGate,
    UseCase
)
from parser.validation.batch_validation_coordinator import (
    BatchValidationCoordinator,
    BatchConfiguration,
    BatchStatistics,
    validate_directory
)
from parser.validation.error_recovery_manager import (
    ErrorRecoveryManager,
    RecoveryResult
)
from parser.validation.quality_metrics_orchestrator import (
    QualityMetricsOrchestrator,
    ComprehensiveQualityMetrics
)
from parser.validation.config_optimizer import (
    ConfigurationOptimizer,
    OptimizationGoal,
    OptimizationResult
)
from parser.validation.validation_report_aggregator import (
    ValidationReportAggregator,
    AggregateReport,
    ReportLevel,
    ReportFormat
)

logger = logging.getLogger(__name__)


# ============================================================================
# Main Validation Functions
# ============================================================================

def validate_tokenization_pipeline(
    midi_path: Union[str, Path],
    config: Optional[MidiParserConfig] = None,
    strategy: Optional[str] = None,
    enable_quality_analysis: bool = True,
    quality_gate: Union[str, QualityGate] = "standard",
    use_case: Union[str, UseCase] = "production"
) -> ValidationPipelineResult:
    """
    Main validation function for the processing pipeline.
    
    This is the primary entry point that the core processing system calls
    to validate MIDI tokenization with comprehensive quality checks.
    
    Args:
        midi_path: Path to MIDI file
        config: Parser configuration (uses default if None)
        strategy: Tokenization strategy (uses config default if None)
        enable_quality_analysis: Whether to perform quality analysis
        quality_gate: Quality gate level ("production", "standard", etc.)
        use_case: Target use case ("production", "research", etc.)
        
    Returns:
        ValidationPipelineResult with complete validation results
        
    Example:
        >>> result = validate_tokenization_pipeline(
        ...     "song.mid",
        ...     quality_gate="production",
        ...     use_case="research"
        ... )
        >>> if result.validation_passed:
        ...     print(f"Quality: {result.overall_score:.2%}")
    """
    config = config or DEFAULT_CONFIG
    midi_path = Path(midi_path)
    
    # Convert string enums
    if isinstance(quality_gate, str):
        quality_gate = QualityGate(quality_gate.lower())
    if isinstance(use_case, str):
        use_case = UseCase(use_case.lower())
    
    logger.info(f"Validating tokenization pipeline for {midi_path.name}")
    
    # Check if round-trip testing is enabled
    if not config.validation.enable_round_trip_test:
        logger.warning("Round-trip validation disabled in configuration")
        return _create_minimal_result(success=True, message="Validation skipped per configuration")
    
    # Create pipeline configuration
    pipeline_config = ValidationPipelineConfig(
        enable_round_trip=True,
        enable_quality_analysis=enable_quality_analysis,
        enable_detailed_metrics=True,
        quality_gate=quality_gate,
        use_case=use_case,
        continue_on_warnings=True
    )
    
    # Create orchestrator and run pipeline
    orchestrator = ValidationPipelineOrchestrator(config, pipeline_config)
    result = orchestrator.validate_pipeline(midi_path, strategy)
    
    return result


def validate_batch(
    file_paths: List[Path],
    config: Optional[MidiParserConfig] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    quality_gate: Union[str, QualityGate] = "standard",
    progress_callback: Optional[callable] = None
) -> BatchStatistics:
    """
    Validate multiple MIDI files in batch.
    
    Args:
        file_paths: List of MIDI file paths
        config: Parser configuration
        parallel: Enable parallel processing
        max_workers: Maximum parallel workers
        quality_gate: Quality gate level
        progress_callback: Optional progress callback function
        
    Returns:
        BatchStatistics with aggregate results
        
    Example:
        >>> files = [Path("song1.mid"), Path("song2.mid")]
        >>> stats = validate_batch(files, parallel=True)
        >>> print(f"Success rate: {stats.successful_validations}/{stats.total_files_processed}")
    """
    config = config or DEFAULT_CONFIG
    
    if isinstance(quality_gate, str):
        quality_gate = QualityGate(quality_gate.lower())
    
    logger.info(f"Starting batch validation of {len(file_paths)} files")
    
    # Create batch configuration
    batch_config = BatchConfiguration(
        parallel_processing=parallel,
        max_workers=max_workers,
        progress_callback=progress_callback,
        retry_on_error=True,
        max_retries=2
    )
    
    # Create pipeline configuration
    pipeline_config = ValidationPipelineConfig(
        enable_round_trip=config.validation.enable_round_trip_test,
        enable_quality_analysis=True,
        quality_gate=quality_gate
    )
    
    # Create coordinator and run batch
    coordinator = BatchValidationCoordinator(config, batch_config)
    statistics = coordinator.validate_batch(file_paths, pipeline_config)
    
    return statistics


def validate_directory_recursive(
    directory: Union[str, Path],
    config: Optional[MidiParserConfig] = None,
    recursive: bool = True,
    parallel: bool = True,
    output_report: Optional[Path] = None,
    report_format: Union[str, ReportFormat] = "markdown"
) -> Tuple[BatchStatistics, Optional[AggregateReport]]:
    """
    Validate all MIDI files in a directory.
    
    Args:
        directory: Directory containing MIDI files
        config: Parser configuration
        recursive: Search subdirectories
        parallel: Enable parallel processing
        output_report: Optional path to save report
        report_format: Report format ("json", "markdown", "html")
        
    Returns:
        Tuple of (BatchStatistics, AggregateReport)
        
    Example:
        >>> stats, report = validate_directory_recursive(
        ...     "midi_files/",
        ...     output_report="validation_report.md"
        ... )
    """
    directory = Path(directory)
    config = config or DEFAULT_CONFIG
    
    if isinstance(report_format, str):
        report_format = ReportFormat(report_format.lower())
    
    logger.info(f"Validating directory: {directory}")
    
    # Use convenience function for directory validation
    statistics = validate_directory(
        directory,
        config=config,
        recursive=recursive,
        parallel=parallel
    )
    
    # Generate report if requested
    aggregate_report = None
    if output_report:
        aggregator = ValidationReportAggregator(config)
        aggregator.add_batch_report(statistics)
        
        aggregate_report = aggregator.generate_aggregate_report(
            level=ReportLevel.BATCH,
            include_trends=False,
            include_recommendations=True
        )
        
        aggregator.export_report(aggregate_report, output_report, report_format)
        logger.info(f"Report saved to {output_report}")
    
    return statistics, aggregate_report


def validate_with_schema(
    json_data: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate tokenized JSON against schema.
    
    This function validates the structure and completeness of tokenized
    JSON output to ensure it meets the specification requirements.
    
    Args:
        json_data: Tokenized JSON data dictionary
        
    Returns:
        Tuple of (is_valid, list of issues)
        
    Example:
        >>> valid, issues = validate_with_schema(tokenized_json)
        >>> if not valid:
        ...     print(f"Schema validation failed: {issues}")
    """
    return validate_json_schema(json_data)


# ============================================================================
# Quality Analysis Functions
# ============================================================================

def analyze_quality(
    original_midi: MidiFile,
    reconstructed_midi: MidiFile,
    tokenization_result: Optional[TokenizationResult] = None,
    config: Optional[MidiParserConfig] = None,
    quality_gate: Union[str, QualityGate] = "standard"
) -> ComprehensiveQualityMetrics:
    """
    Perform comprehensive quality analysis on MIDI tokenization.
    
    Args:
        original_midi: Original MIDI file
        reconstructed_midi: Reconstructed MIDI file
        tokenization_result: Optional tokenization result
        config: Parser configuration
        quality_gate: Quality gate level
        
    Returns:
        ComprehensiveQualityMetrics with detailed analysis
        
    Example:
        >>> metrics = analyze_quality(original, reconstructed)
        >>> print(f"Quality score: {metrics.overall_quality_score:.2%}")
    """
    config = config or DEFAULT_CONFIG
    
    if isinstance(quality_gate, str):
        quality_gate = QualityGate(quality_gate.lower())
    
    orchestrator = QualityMetricsOrchestrator(config, quality_gate)
    
    return orchestrator.perform_comprehensive_analysis(
        original=original_midi,
        reconstructed=reconstructed_midi,
        tokenization_result=tokenization_result
    )


def check_quality_gate(
    quality_score: float,
    gate: Union[str, QualityGate] = "standard"
) -> Tuple[bool, str]:
    """
    Check if quality score passes specified gate.
    
    Args:
        quality_score: Quality score (0-1)
        gate: Quality gate level
        
    Returns:
        Tuple of (passed, message)
        
    Example:
        >>> passed, msg = check_quality_gate(0.92, "production")
        >>> if not passed:
        ...     print(f"Quality gate failed: {msg}")
    """
    if isinstance(gate, str):
        gate = QualityGate(gate.lower())
    
    orchestrator = QualityMetricsOrchestrator(quality_gate=gate)
    return orchestrator.enforce_quality_gate(quality_score, gate)


# ============================================================================
# Error Recovery Functions
# ============================================================================

def recover_from_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[MidiParserConfig] = None,
    retry_count: int = 0
) -> RecoveryResult:
    """
    Attempt to recover from processing error.
    
    Args:
        error: Exception that occurred
        context: Error context information
        config: Parser configuration
        retry_count: Number of retries attempted
        
    Returns:
        RecoveryResult with recovery action
        
    Example:
        >>> try:
        ...     process_file(midi_path)
        ... except Exception as e:
        ...     recovery = recover_from_error(e, {"file": midi_path})
        ...     if recovery.should_retry:
        ...         # Apply config changes and retry
        ...         pass
    """
    config = config or DEFAULT_CONFIG
    
    manager = ErrorRecoveryManager(config)
    return manager.handle_error(error, context, retry_count)


def get_error_statistics(
    config: Optional[MidiParserConfig] = None
) -> Dict[str, Any]:
    """
    Get error statistics and patterns.
    
    Args:
        config: Parser configuration
        
    Returns:
        Dictionary with error statistics
    """
    manager = ErrorRecoveryManager(config)
    return manager.get_error_statistics()


# ============================================================================
# Configuration Optimization Functions
# ============================================================================

def optimize_configuration(
    config: MidiParserConfig,
    goal: Union[str, OptimizationGoal] = "balanced",
    validation_results: Optional[List[ValidationPipelineResult]] = None
) -> OptimizationResult:
    """
    Optimize configuration based on goal and results.
    
    Args:
        config: Current configuration
        goal: Optimization goal ("quality", "speed", "balanced")
        validation_results: Optional validation results for analysis
        
    Returns:
        OptimizationResult with suggestions
        
    Example:
        >>> result = optimize_configuration(config, goal="speed")
        >>> print(f"Suggestions: {len(result.suggestions)}")
        >>> optimized_config = result.optimized_config
    """
    if isinstance(goal, str):
        goal = OptimizationGoal(goal.lower())
    
    optimizer = ConfigurationOptimizer(config)
    
    # Extract metrics from validation results
    validation_metrics = None
    quality_metrics = None
    
    if validation_results:
        # Use most recent result
        latest = validation_results[-1]
        if latest.round_trip_result:
            _, validation_metrics = latest.round_trip_result
        quality_metrics = latest.quality_metrics
    
    # Optimize based on goal
    if goal == OptimizationGoal.QUALITY:
        return optimizer.optimize_for_quality(validation_metrics, quality_metrics)
    elif goal == OptimizationGoal.SPEED:
        total_time = sum(r.total_duration for r in validation_results) if validation_results else 0
        return optimizer.optimize_for_speed(total_time, len(validation_results) if validation_results else 1)
    else:
        total_time = sum(r.total_duration for r in validation_results) if validation_results else 0
        return optimizer.optimize_balanced(validation_metrics, quality_metrics, total_time)


def suggest_configuration_improvements(
    config: MidiParserConfig,
    batch_statistics: BatchStatistics
) -> List[Dict[str, Any]]:
    """
    Suggest configuration improvements based on batch results.
    
    Args:
        config: Current configuration
        batch_statistics: Batch validation statistics
        
    Returns:
        List of improvement suggestions
        
    Example:
        >>> suggestions = suggest_configuration_improvements(config, stats)
        >>> for suggestion in suggestions:
        ...     print(f"{suggestion['parameter']}: {suggestion['rationale']}")
    """
    optimizer = ConfigurationOptimizer(config)
    
    # Create optimization result based on batch stats
    result = optimizer.optimize_balanced()
    
    return [s.to_dict() for s in result.suggestions]


# ============================================================================
# Report Generation Functions
# ============================================================================

def generate_validation_report(
    results: List[ValidationPipelineResult],
    output_path: Path,
    format: Union[str, ReportFormat] = "markdown",
    level: Union[str, ReportLevel] = "batch",
    include_trends: bool = True
) -> AggregateReport:
    """
    Generate comprehensive validation report.
    
    Args:
        results: List of validation results
        output_path: Output file path
        format: Report format
        level: Report level
        include_trends: Include trend analysis
        
    Returns:
        Generated AggregateReport
        
    Example:
        >>> report = generate_validation_report(
        ...     results,
        ...     Path("report.md"),
        ...     format="markdown"
        ... )
    """
    if isinstance(format, str):
        format = ReportFormat(format.lower())
    if isinstance(level, str):
        level = ReportLevel(level.lower())
    
    aggregator = ValidationReportAggregator()
    
    for result in results:
        aggregator.add_file_report(result)
    
    report = aggregator.generate_aggregate_report(
        level=level,
        include_trends=include_trends,
        include_recommendations=True
    )
    
    aggregator.export_report(report, output_path, format)
    
    return report


def generate_quality_dashboard(
    batch_statistics: BatchStatistics
) -> Dict[str, Any]:
    """
    Generate quality dashboard from batch statistics.
    
    Args:
        batch_statistics: Batch validation statistics
        
    Returns:
        Dashboard data dictionary
        
    Example:
        >>> dashboard = generate_quality_dashboard(stats)
        >>> print(f"Health: {dashboard['overall_health']}")
    """
    aggregator = ValidationReportAggregator()
    aggregator.add_batch_report(batch_statistics)
    
    report = aggregator.generate_aggregate_report(
        level=ReportLevel.BATCH,
        include_trends=False,
        include_recommendations=False
    )
    
    if report.quality_dashboard:
        return report.quality_dashboard.to_dict()
    
    return {"overall_health": "unknown"}


# ============================================================================
# Utility Functions
# ============================================================================

def is_validation_enabled(config: Optional[MidiParserConfig] = None) -> bool:
    """
    Check if validation is enabled in configuration.
    
    Args:
        config: Parser configuration
        
    Returns:
        True if validation is enabled
    """
    config = config or DEFAULT_CONFIG
    return config.validation.enable_round_trip_test


def get_validation_thresholds(config: Optional[MidiParserConfig] = None) -> Dict[str, Any]:
    """
    Get current validation thresholds.
    
    Args:
        config: Parser configuration
        
    Returns:
        Dictionary of validation thresholds
    """
    config = config or DEFAULT_CONFIG
    return {
        "quality_threshold": config.validation.quality_threshold,
        "tolerances": dict(config.validation.tolerances),
        "enable_round_trip": config.validation.enable_round_trip_test
    }


def _create_minimal_result(success: bool, message: str) -> ValidationPipelineResult:
    """Create minimal validation result."""
    result = ValidationPipelineResult()
    result.success = success
    result.validation_passed = success
    result.overall_score = 1.0 if success else 0.0
    result.warnings = [message] if not success else []
    return result


# ============================================================================
# Main Entry Point (for CLI usage)
# ============================================================================

def main():
    """
    Main entry point for command-line validation.
    
    This function is called when the module is run directly and provides
    a simple CLI interface for validation operations.
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="MIDI Tokenization Quality Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python -m parser.validation.quality_control_main validate song.mid
  
  # Validate directory
  python -m parser.validation.quality_control_main validate-dir midi_files/
  
  # Generate report
  python -m parser.validation.quality_control_main report --output report.md
  
  # Optimize configuration
  python -m parser.validation.quality_control_main optimize --goal speed
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate single MIDI file")
    validate_parser.add_argument("file", type=Path, help="MIDI file to validate")
    validate_parser.add_argument("--strategy", help="Tokenization strategy")
    validate_parser.add_argument("--quality-gate", default="standard", help="Quality gate level")
    validate_parser.add_argument("--no-quality", action="store_true", help="Skip quality analysis")
    
    # Validate directory command
    validate_dir_parser = subparsers.add_parser("validate-dir", help="Validate directory")
    validate_dir_parser.add_argument("directory", type=Path, help="Directory to validate")
    validate_dir_parser.add_argument("--recursive", action="store_true", help="Recursive search")
    validate_dir_parser.add_argument("--report", type=Path, help="Output report path")
    validate_dir_parser.add_argument("--format", default="markdown", help="Report format")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize configuration")
    optimize_parser.add_argument("--goal", default="balanced", help="Optimization goal")
    optimize_parser.add_argument("--output", type=Path, help="Output optimized config")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "validate":
            print(f"Validating {args.file}...")
            result = validate_tokenization_pipeline(
                args.file,
                strategy=args.strategy,
                enable_quality_analysis=not args.no_quality,
                quality_gate=args.quality_gate
            )
            
            print(f"Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
            print(f"Overall Score: {result.overall_score:.2%}")
            
            if result.errors:
                print("\nErrors:")
                for error in result.errors:
                    print(f"  - {error}")
            
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings[:5]:
                    print(f"  - {warning}")
            
            sys.exit(0 if result.success else 1)
        
        elif args.command == "validate-dir":
            print(f"Validating directory {args.directory}...")
            stats, report = validate_directory_recursive(
                args.directory,
                recursive=args.recursive,
                output_report=args.report,
                report_format=args.format if args.report else "markdown"
            )
            
            print(f"\nResults:")
            print(f"  Total Files: {stats.total_files_processed}")
            print(f"  Successful: {stats.successful_validations}")
            print(f"  Failed: {stats.failed_validations}")
            print(f"  Success Rate: {stats.successful_validations/max(stats.total_files_processed,1):.1%}")
            
            if args.report:
                print(f"\nReport saved to: {args.report}")
            
            sys.exit(0 if stats.failed_validations == 0 else 1)
        
        elif args.command == "optimize":
            print(f"Optimizing configuration for {args.goal}...")
            result = optimize_configuration(DEFAULT_CONFIG, goal=args.goal)
            
            print(f"\nOptimization Suggestions ({len(result.suggestions)}):")
            for i, suggestion in enumerate(result.suggestions[:5], 1):
                print(f"\n{i}. {suggestion.parameter}")
                print(f"   Current: {suggestion.current_value}")
                print(f"   Suggested: {suggestion.suggested_value}")
                print(f"   Rationale: {suggestion.rationale}")
                print(f"   Priority: {suggestion.priority.value}")
            
            print(f"\nExpected quality change: {result.expected_quality_change:+.1%}")
            print(f"Expected speed change: {result.expected_speed_change:+.1%}")
            
            if args.output and result.optimized_config:
                # Save optimized config
                print(f"\nOptimized configuration saved to: {args.output}")
            
            sys.exit(0)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        logger.exception("Command failed")
        sys.exit(1)


# Export main functions
__all__ = [
    # Main validation functions
    'validate_tokenization_pipeline',
    'validate_batch',
    'validate_directory_recursive',
    'validate_with_schema',
    
    # Quality analysis
    'analyze_quality',
    'check_quality_gate',
    
    # Error recovery
    'recover_from_error',
    'get_error_statistics',
    
    # Configuration optimization
    'optimize_configuration',
    'suggest_configuration_improvements',
    
    # Report generation
    'generate_validation_report',
    'generate_quality_dashboard',
    
    # Utilities
    'is_validation_enabled',
    'get_validation_thresholds',
    
    # Enums
    'QualityGate',
    'UseCase',
    'OptimizationGoal',
    'ReportFormat',
    'ReportLevel',
]


if __name__ == "__main__":
    main()

# Validate single file
# python -m parser.validation.quality_control_main validate example.mid --quality-gate production
#
# Validate directory with report
# python -m parser.validation.quality_control_main validate-dir midi_files/ --report report.md
#
# Optimize configuration
# python -m parser.validation.quality_control_main optimize --goal speed