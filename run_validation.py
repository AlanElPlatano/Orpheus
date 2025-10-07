#!/usr/bin/env python3

# python run_validation.py

"""
Complete validation example with all features including proper report generation
"""
from pathlib import Path
from midi_parser.validation.quality_control_main import (
    validate_tokenization_pipeline,
    validate_batch,
    validate_directory_recursive,
    optimize_configuration,
    generate_validation_report
)

def validate_single_file():
    """Test 1: Validate a single MIDI file"""
    print("=" * 60)
    print("TEST 1: Single File Validation")
    print("=" * 60)
    
    result = validate_tokenization_pipeline(
        midi_path=Path("source_midis/single/Fuerza Regida - OYE.mid"),
        strategy="REMI",
        enable_quality_analysis=True,
        quality_gate="permissive",
        use_case="production"
    )
    
    print(f"\n✓ Validation Status: {'PASSED' if result.validation_passed else 'FAILED'}")
    print(f"✓ Quality Score: {result.overall_score:.1%}")
    print(f"✓ Processing Time: {result.total_duration:.2f}s")
    
    # Show round-trip metrics
    if result.round_trip_result:
        _, metrics = result.round_trip_result
        print(f"\nRound-Trip Metrics:")
        print(f"  - Original notes: {metrics.total_notes_original}")
        print(f"  - Reconstructed notes: {metrics.total_notes_reconstructed}")
        print(f"  - Missing: {metrics.missing_notes} ({metrics.missing_notes_ratio:.1%})")
        print(f"  - Timing accuracy: {metrics.timing_accuracy:.1%}")
        print(f"  - Velocity accuracy: {metrics.velocity_accuracy:.1%}")
    
    # Show quality metrics
    if result.quality_metrics:
        print(f"\nQuality Metrics:")
        print(f"  - Overall quality: {result.quality_metrics.overall_quality_score:.1%}")
        print(f"  - Complexity: {result.quality_metrics.complexity_level.value}")
        print(f"  - Gate passed: {result.quality_metrics.quality_gate_passed}")
    
    # Show issues
    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for error in result.errors[:3]:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for warning in result.warnings[:3]:
            print(f"  - {warning}")
    
    # Show recommendations
    if result.recommendations:
        print(f"\n💡 Top Recommendations:")
        for rec in result.recommendations[:3]:
            print(f"  - {rec}")
    
    # Generate individual file report
    print("\n📄 Generating detailed report...")
    report = generate_validation_report(
        results=[result],
        output_path=Path("validation_reports/single_file_report.md"),
        format="markdown",
        level="file",
        include_trends=False
    )
    print(f"✓ Report saved to: validation_reports/single_file_report.md")
    
    # Also generate JSON version
    report_json = generate_validation_report(
        results=[result],
        output_path=Path("validation_reports/single_file_report.json"),
        format="json",
        level="file",
        include_trends=False
    )
    print(f"✓ JSON report saved to: validation_reports/single_file_report.json")
    
    return result


def validate_multiple_files():
    """Test 2: Batch validation with detailed report"""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Validation")
    print("=" * 60)
    
    # Get all MIDI files
    batch_dir = Path("source_midis/batch")
    midi_files = list(batch_dir.glob("*.mid")) + list(batch_dir.glob("*.midi"))
    
    if not midi_files:
        print("No MIDI files found in source_midis/batch")
        return None
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Run batch validation
    stats = validate_batch(
        file_paths=midi_files,
        parallel=True,
        max_workers=4,
        quality_gate="permissive"  # ✓ Changed to permissive
    )
    
    print(f"\nBatch Results:")
    print(f"  - Total files: {stats.total_files_processed}")
    print(f"  - Successful: {stats.successful_validations}")
    print(f"  - Failed: {stats.failed_validations}")
    print(f"  - Success rate: {stats.successful_validations/stats.total_files_processed:.1%}")
    print(f"  - Avg quality: {stats.average_quality_score:.1%}")
    print(f"  - Avg time per file: {stats.average_processing_time:.2f}s")
    
    # Show common errors
    if stats.error_types:
        print(f"\nCommon Errors:")
        for error_type, count in list(stats.error_types.items())[:5]:
            print(f"  - {error_type}: {count}")
    
    # Generate batch report with statistics
    print("\n📊 Generating batch report...")
    
    from midi_parser.validation.quality_control_main import generate_quality_dashboard
    dashboard = generate_quality_dashboard(stats)
    
    print(f"\n📈 Quality Dashboard:")
    print(f"  - Overall health: {dashboard.get('overall_health', 'unknown')}")
    
    # Save detailed batch statistics to JSON
    import json
    batch_report_path = Path("validation_reports/batch_statistics.json")
    with open(batch_report_path, 'w') as f:
        json.dump(stats.to_dict(), f, indent=2)
    print(f"✓ Batch statistics saved to: {batch_report_path}")
    
    return stats


def validate_directory_with_report():
    """Test 3: Directory validation with full comprehensive report"""
    print("\n" + "=" * 60)
    print("TEST 3: Directory Validation with Comprehensive Report")
    print("=" * 60)
    
    stats, report = validate_directory_recursive(
        directory=Path("source_midis/"),
        recursive=True,
        parallel=True,
        output_report=Path("validation_reports/directory_report.md"),
        report_format="markdown"
    )
    
    print(f"\nDirectory Validation Complete:")
    print(f"  - Total files: {stats.total_files_processed}")
    print(f"  - Success rate: {stats.successful_validations/stats.total_files_processed:.1%}")
    print(f"  - Markdown report saved to: validation_reports/directory_report.md")
    
    # Generate additional report formats
    if report:
        print("\n📄 Generating additional report formats...")
        
        # JSON format (for programmatic access)
        import json
        json_path = Path("validation_reports/directory_report.json")
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  ✓ JSON: {json_path}")
        
        # HTML format (for viewing in browser)
        from midi_parser.validation.validation_report_aggregator import (
            ValidationReportAggregator,
            ReportFormat
        )
        aggregator = ValidationReportAggregator()
        aggregator.add_batch_report(stats)
        
        html_report = aggregator.generate_aggregate_report(
            level="batch" if hasattr(report, 'level') else "summary",
            include_trends=True,
            include_recommendations=True
        )
        
        aggregator.export_report(
            html_report,
            Path("validation_reports/directory_report.html"),
            ReportFormat.HTML
        )
        print(f"  ✓ HTML: validation_reports/directory_report.html")
        
        # Text format (for console/logs)
        aggregator.export_report(
            html_report,
            Path("validation_reports/directory_report.txt"),
            ReportFormat.TEXT
        )
        print(f"  ✓ Text: validation_reports/directory_report.txt")
        
        # CSV format (for spreadsheets/analysis)
        aggregator.export_report(
            html_report,
            Path("validation_reports/directory_report.csv"),
            ReportFormat.CSV
        )
        print(f"  ✓ CSV: validation_reports/directory_report.csv")
    
    return stats, report


def generate_comparison_report():
    """Test 4: Generate comparison report across different REMI configurations"""
    print("\n" + "=" * 60)
    print("TEST 4: REMI Configuration Comparison")
    print("=" * 60)
    
    test_file = Path("source_midis/single/Fuerza Regida - OYE.mid")
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    # Test different configurations of REMI
    configs = [
        ("REMI-Standard", "permissive"),
        ("REMI-Standard", "standard"),
        ("REMI-Standard", "production"),
    ]
    results = []
    
    print(f"Testing file with {len(configs)} different configurations...")
    
    for strategy, gate in configs:
        print(f"\n  Testing with {strategy} (gate: {gate})...")
        try:
            result = validate_tokenization_pipeline(
                midi_path=test_file,
                strategy="REMI",
                enable_quality_analysis=True,
                quality_gate=gate
            )
            results.append(result)
            print(f"    ✓ Score: {result.overall_score:.1%}, Passed: {result.validation_passed}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    if not results:
        print("No successful validations to compare")
        return
    
    print("\n📊 Generating configuration comparison report...")
    
    report = generate_validation_report(
        results=results,
        output_path=Path("validation_reports/config_comparison.md"),
        format="markdown",
        level="batch",
        include_trends=True
    )
    
    print(f"✓ Comparison report saved to: validation_reports/config_comparison.md")
    
    print("\n📈 Configuration Comparison Summary:")
    print(f"{'Config':<25} {'Score':<10} {'Passed':<10} {'Time (s)':<10}")
    print("-" * 60)
    for i, result in enumerate(results):
        config_name = f"{configs[i][0]}-{configs[i][1]}"
        passed = "✓" if result.validation_passed else "✗"
        print(f"{config_name:<25} {result.overall_score:<10.1%} {passed:<10} {result.total_duration:<10.2f}")
    
    return results


def get_optimization_suggestions():
    """Test 5: Get configuration optimization suggestions"""
    print("\n" + "=" * 60)
    print("TEST 5: Configuration Optimization")
    print("=" * 60)
    
    from midi_parser.config.defaults import DEFAULT_CONFIG
    
    # Optimize for different goals
    for goal in ["quality", "speed", "balanced"]:
        print(f"\n--- Optimizing for: {goal.upper()} ---")
        
        result = optimize_configuration(
            config=DEFAULT_CONFIG,
            goal=goal
        )
        
        print(f"Expected quality change: {result.expected_quality_change:+.1%}")
        print(f"Expected speed change: {result.expected_speed_change:+.1%}")
        print(f"\nTop 3 Suggestions:")
        
        for i, suggestion in enumerate(result.suggestions[:3], 1):
            print(f"\n{i}. {suggestion.parameter}")
            print(f"   Current: {suggestion.current_value}")
            print(f"   Suggested: {suggestion.suggested_value}")
            print(f"   Reason: {suggestion.rationale}")
            print(f"   Priority: {suggestion.priority.value}")
    
    # Save optimization suggestions to file
    print("\n📄 Saving optimization suggestions...")
    import json
    
    for goal in ["quality", "speed", "balanced"]:
        result = optimize_configuration(config=DEFAULT_CONFIG, goal=goal)
        
        suggestions_file = Path(f"validation_reports/optimization_{goal}.json")
        with open(suggestions_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  ✓ {goal}: {suggestions_file}")


def main():
    """Run all validation tests with comprehensive reporting"""
    print("\n" + "🎵" * 30)
    print("MIDI VALIDATION SYSTEM - COMPREHENSIVE TEST WITH REPORTS")
    print("🎵" * 30)
    
    # Make sure directories exist
    Path("source_midis/single").mkdir(parents=True, exist_ok=True)
    Path("source_midis/batch").mkdir(parents=True, exist_ok=True)
    Path("validation_reports").mkdir(parents=True, exist_ok=True)
    
    try:
        # Run tests
        print("\n🔍 Running validation tests...\n")
        
        result1 = validate_single_file()
        result2 = validate_multiple_files()
        result3 = validate_directory_with_report()
        result4 = generate_comparison_report()
        get_optimization_suggestions()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
        print("\n📁 Reports generated in validation_reports/:")
        print("  - single_file_report.md/.json")
        print("  - optimization_quality/speed/balanced.json")
        
        print("\n💡 Next steps:")
        print("  1. Review validation_reports/single_file_report.md")
        print("  2. Check validation_reports/optimization_*.json for suggestions")
        print("  3. Adjust tolerances if timing accuracy is consistently ~94%")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have MIDI files in:")
        print("  - source_midis/single/")
        print("  - source_midis/batch/")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()