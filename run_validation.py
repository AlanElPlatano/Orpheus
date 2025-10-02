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
        midi_path=Path("midi_files/single/example.mid"),
        strategy="TSD",  # or "TSD", "Structured", etc.
        enable_quality_analysis=True,
        quality_gate="standard",
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
        include_trends=False  # Single file, no trends
    )
    print(f"✓ Report saved to: validation_reports/single_file_report.md")
    
    # Also generate JSON version for programmatic access
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
        parallel=True,  # Use parallel processing
        max_workers=4,  # Number of parallel workers
        quality_gate="standard"
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
    
    # You can also use the generate_quality_dashboard for a summary view
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
    
    # This function generates reports automatically
    stats, report = validate_directory_recursive(
        directory=Path("test_files/"),
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
    """Test 4: Generate comparison report across multiple validation runs"""
    print("\n" + "=" * 60)
    print("TEST 4: Comparison Report (Multiple Validation Runs)")
    print("=" * 60)
    
    # Validate with different strategies
    strategies = ["REMI", "TSD", "Structured"]
    results = []
    
    test_file = Path("midi_files/single/example.mid")
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    print(f"Testing file with {len(strategies)} different strategies...")
    
    for strategy in strategies:
        print(f"\n  Testing with {strategy}...")
        try:
            result = validate_tokenization_pipeline(
                midi_path=test_file,
                strategy=strategy,
                enable_quality_analysis=True,
                quality_gate="standard"
            )
            results.append(result)
            print(f"    ✓ Score: {result.overall_score:.1%}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    if not results:
        print("No successful validations to compare")
        return
    
    # Generate comprehensive comparison report
    print("\n📊 Generating strategy comparison report...")
    
    report = generate_validation_report(
        results=results,
        output_path=Path("validation_reports/strategy_comparison.md"),
        format="markdown",
        level="batch",
        include_trends=True
    )
    
    print(f"✓ Comparison report saved to: validation_reports/strategy_comparison.md")
    
    # Generate summary table
    print("\n📈 Strategy Comparison Summary:")
    print(f"{'Strategy':<15} {'Score':<10} {'Passed':<10} {'Time (s)':<10}")
    print("-" * 50)
    for i, result in enumerate(results):
        strategy_name = strategies[i] if i < len(strategies) else "Unknown"
        passed = "✓" if result.validation_passed else "✗"
        print(f"{strategy_name:<15} {result.overall_score:<10.1%} {passed:<10} {result.total_duration:<10.2f}")
    
    return results


def generate_campaign_report():
    """Test 5: Generate a campaign-level report with trends"""
    print("\n" + "=" * 60)
    print("TEST 5: Campaign Report with Trends")
    print("=" * 60)
    
    from midi_parser.validation.validation_report_aggregator import (
        ValidationReportAggregator,
        ReportLevel,
        ReportFormat
    )
    
    # Collect multiple batch statistics (simulating multiple validation runs)
    batch_dir = Path("test_files/batch_test")
    midi_files = list(batch_dir.glob("*.mid")) + list(batch_dir.glob("*.midi"))
    
    if not midi_files:
        print("No MIDI files found for campaign report")
        return
    
    print(f"Running campaign validation on {len(midi_files)} files...")
    
    # Split files into multiple batches to simulate campaign
    batch_size = max(2, len(midi_files) // 3)
    batches = []
    
    for i in range(0, len(midi_files), batch_size):
        batch = midi_files[i:i+batch_size]
        print(f"\n  Processing batch {len(batches)+1} ({len(batch)} files)...")
        
        stats = validate_batch(
            file_paths=batch,
            parallel=True,
            quality_gate="standard"
        )
        batches.append(stats)
        print(f"    ✓ Success rate: {stats.successful_validations/stats.total_files_processed:.1%}")
    
    # Generate campaign report
    print("\n📊 Generating campaign report with trend analysis...")
    
    aggregator = ValidationReportAggregator()
    
    # Add all batch reports
    for batch_stats in batches:
        aggregator.add_batch_report(batch_stats)
    
    # Generate comprehensive campaign report
    campaign_report = aggregator.generate_aggregate_report(
        level=ReportLevel.CAMPAIGN,
        include_trends=True,
        include_recommendations=True
    )
    
    # Export in multiple formats
    print("\n📄 Exporting campaign reports...")
    
    aggregator.export_report(
        campaign_report,
        Path("validation_reports/campaign_report.md"),
        ReportFormat.MARKDOWN
    )
    print(f"  ✓ Markdown: validation_reports/campaign_report.md")
    
    aggregator.export_report(
        campaign_report,
        Path("validation_reports/campaign_report.html"),
        ReportFormat.HTML
    )
    print(f"  ✓ HTML: validation_reports/campaign_report.html")
    
    aggregator.export_report(
        campaign_report,
        Path("validation_reports/campaign_report.json"),
        ReportFormat.JSON
    )
    print(f"  ✓ JSON: validation_reports/campaign_report.json")
    
    # Print campaign summary
    if campaign_report.quality_dashboard:
        dashboard = campaign_report.quality_dashboard
        print(f"\n📈 Campaign Summary:")
        print(f"  - Overall Health: {dashboard.overall_health.upper()}")
        print(f"  - Health Score: {dashboard.health_score:.1f}/100")
        print(f"  - Average Quality: {dashboard.average_quality:.1%}")
        print(f"  - Pass Rate: {dashboard.pass_rate:.1%}")
        
        if dashboard.quality_trend:
            trend = dashboard.quality_trend
            print(f"  - Quality Trend: {trend.direction} ({trend.change_rate:+.1f}%)")
    
    return campaign_report


def get_optimization_suggestions():
    """Test 6: Get configuration optimization suggestions"""
    print("\n" + "=" * 60)
    print("TEST 6: Configuration Optimization")
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
        result5 = generate_campaign_report()
        get_optimization_suggestions()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
        print("\n📁 Reports generated in validation_reports/:")
        print("  - single_file_report.md/.json")
        print("  - batch_statistics.json")
        print("  - directory_report.md/.json/.html/.txt/.csv")
        print("  - strategy_comparison.md")
        print("  - campaign_report.md/.html/.json")
        print("  - optimization_quality/speed/balanced.json")
        
        print("\n💡 Next steps:")
        print("  1. Open validation_reports/directory_report.html in your browser")
        print("  2. Review validation_reports/campaign_report.md for trends")
        print("  3. Check validation_reports/optimization_*.json for suggestions")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have MIDI files in:")
        print("  - test_files/single_test/")
        print("  - test_files/batch_test/")
        print("\nTip: Place at least one .mid file in each directory")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()