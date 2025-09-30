"""
Report generation utilities for round-trip validation.

This module provides various report formats and utility functions for
analyzing and presenting validation results in human-readable formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from .validation_metrics import RoundTripMetrics


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
    if output_format.lower() == "json":
        return json.dumps(metrics.to_dict(), indent=2)
    
    elif output_format.lower() == "markdown":
        return _generate_markdown_report(metrics)
    
    else:  # text format
        return _generate_text_report(metrics)


def _generate_markdown_report(metrics: RoundTripMetrics) -> str:
    """Generate markdown formatted report."""
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


def _generate_text_report(metrics: RoundTripMetrics) -> str:
    """Generate plain text formatted report."""
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
    config: Optional[Any] = None
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
    from midi_parser.core.midi_loader import load_midi_file
    from .round_trip_validator import RoundTripValidator
    
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


def compare_midi_files(
    original_path: Union[str, Path],
    reconstructed_path: Union[str, Path],
    config: Optional[Any] = None
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
    from midi_parser.core.midi_loader import load_midi_file
    from .midi_comparator import MidiComparator
    from .note_matcher import NoteMatcher
    from .validation_metrics import DEFAULT_VALIDATION_TOLERANCES
    
    original = load_midi_file(original_path)
    reconstructed = load_midi_file(reconstructed_path)
    
    if original is None or reconstructed is None:
        raise ValueError("Failed to load MIDI files for comparison")
    
    note_matcher = NoteMatcher(DEFAULT_VALIDATION_TOLERANCES)
    comparator = MidiComparator(note_matcher)
    
    metrics = comparator.compare_files(
        original,
        reconstructed,
        "direct_comparison",
        detailed=True
    )
    
    return metrics


def round_trip_test(
    midi_path: Union[str, Path],
    strategy: str = "REMI",
    config: Optional[Any] = None
) -> tuple[bool, RoundTripMetrics]:
    """
    Simple round-trip test function.
    
    Args:
        midi_path: Path to MIDI file
        strategy: Tokenization strategy
        config: Optional configuration
        
    Returns:
        Tuple of (success, metrics)
    """
    from midi_parser.core.midi_loader import load_midi_file
    from .round_trip_validator import RoundTripValidator
    
    # Load MIDI file
    midi = load_midi_file(midi_path)
    if midi is None:
        return False, RoundTripMetrics()
    
    # Create validator
    validator = RoundTripValidator(config)
    
    # Run validation
    result, metrics = validator.validate_round_trip(midi, strategy)
    
    return result.is_valid, metrics


def generate_batch_summary(
    batch_stats: Dict[str, Union[int, float]],
    output_format: str = "text"
) -> str:
    """
    Generate summary report for batch validation results.
    
    Args:
        batch_stats: Statistics from batch validation
        output_format: Format ("text", "markdown", "json")
        
    Returns:
        Formatted summary report
    """
    if output_format.lower() == "json":
        return json.dumps(batch_stats, indent=2)
    
    elif output_format.lower() == "markdown":
        return _generate_batch_markdown(batch_stats)
    
    else:
        return _generate_batch_text(batch_stats)


def _generate_batch_markdown(stats: Dict[str, Union[int, float]]) -> str:
    """Generate markdown batch summary."""
    report = [
        "# Batch Validation Summary",
        "",
        f"**Total Files:** {stats.get('total_files', 0)}",
        f"**Success Rate:** {stats.get('success_rate', 0):.1%}",
        f"**Average Accuracy:** {stats.get('avg_accuracy', 0):.2%}",
        f"**Total Processing Time:** {stats.get('total_processing_time', 0):.1f}s",
        "",
        "## Detailed Statistics",
        f"- Successful Validations: {stats.get('successful_validations', 0)}",
        f"- Failed Validations: {stats.get('failed_validations', 0)}",
        f"- Average Processing Time: {stats.get('avg_processing_time', 0):.2f}s per file",
        f"- Total Tokens: {stats.get('total_tokens', 0):,}",
        f"- Average Tokens per File: {stats.get('avg_tokens_per_file', 0):.0f}",
        "",
        "## Quality Metrics",
        f"- Minimum Accuracy: {stats.get('min_accuracy', 0):.2%}",
        f"- Maximum Accuracy: {stats.get('max_accuracy', 0):.2%}",
        f"- Total Original Notes: {stats.get('total_notes_original', 0):,}",
        f"- Total Reconstructed Notes: {stats.get('total_notes_reconstructed', 0):,}",
    ]
    
    # Add error analysis if available
    if 'common_errors' in stats and stats['common_errors']:
        report.extend([
            "",
            "## Common Issues",
            "### Errors"
        ])
        for error, count in stats['common_errors'].items():
            report.append(f"- {error}: {count} occurrences")
    
    return "\n".join(report)


def _generate_batch_text(stats: Dict[str, Union[int, float]]) -> str:
    """Generate plain text batch summary."""
    report = [
        "=" * 50,
        "BATCH VALIDATION SUMMARY",
        "=" * 50,
        f"Total Files: {stats.get('total_files', 0)}",
        f"Success Rate: {stats.get('success_rate', 0):.1%}",
        f"Average Accuracy: {stats.get('avg_accuracy', 0):.2%}",
        "-" * 50,
        "PROCESSING STATISTICS:",
        f"  Successful: {stats.get('successful_validations', 0)}",
        f"  Failed: {stats.get('failed_validations', 0)}",
        f"  Avg Time: {stats.get('avg_processing_time', 0):.2f}s per file",
        f"  Total Time: {stats.get('total_processing_time', 0):.1f}s",
        "-" * 50,
        "QUALITY STATISTICS:",
        f"  Min Accuracy: {stats.get('min_accuracy', 0):.2%}",
        f"  Max Accuracy: {stats.get('max_accuracy', 0):.2%}",
        f"  Total Tokens: {stats.get('total_tokens', 0):,}",
        "=" * 50
    ]
    
    return "\n".join(report)


def export_detailed_results(
    results: List[tuple],
    output_path: Union[str, Path],
    include_metrics: bool = True
) -> None:
    """
    Export detailed validation results to JSON file.
    
    Args:
        results: List of (ValidationResult, RoundTripMetrics) tuples
        output_path: Path to output JSON file
        include_metrics: Whether to include detailed metrics
    """
    export_data = {
        "summary": {
            "total_files": len(results),
            "successful": sum(1 for r, _ in results if r.is_valid),
            "failed": sum(1 for r, _ in results if not r.is_valid)
        },
        "results": []
    }
    
    for i, (validation_result, metrics) in enumerate(results):
        result_data = {
            "file_index": i,
            "validation_passed": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings
        }
        
        if include_metrics:
            result_data["metrics"] = metrics.to_dict()
        
        export_data["results"].append(result_data)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)