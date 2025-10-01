"""
Validation report aggregator for comprehensive reporting.

This module provides comprehensive validation reporting with detailed analysis,
actionable insights, multi-level reporting (file/batch/campaign), trend analysis,
configuration recommendations, and quality dashboards.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from collections import defaultdict, Counter
import statistics

from midi_parser.config.defaults import MidiParserConfig, DEFAULT_CONFIG
from midi_parser.validation.validation_pipeline_orchestrator import ValidationPipelineResult
from midi_parser.validation.batch_validation_coordinator import BatchStatistics
from midi_parser.validation.report_generator import generate_validation_report

logger = logging.getLogger(__name__)


class ReportLevel(Enum):
    """Levels of reporting granularity."""
    FILE = "file"
    BATCH = "batch"
    CAMPAIGN = "campaign"
    SUMMARY = "summary"


class ReportFormat(Enum):
    """Output formats for reports."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"


@dataclass
class TrendAnalysis:
    """Trend analysis over time or across batches."""
    trend_type: str  # "quality", "performance", "errors"
    direction: str  # "improving", "declining", "stable"
    change_rate: float  # Percentage change
    confidence: float  # Statistical confidence
    data_points: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.trend_type,
            "direction": self.direction,
            "change_rate": round(self.change_rate, 2),
            "confidence": round(self.confidence, 3),
            "data_points": len(self.data_points)
        }


@dataclass
class QualityDashboard:
    """Dashboard view of quality metrics."""
    overall_health: str  # "excellent", "good", "fair", "poor"
    health_score: float  # 0-100
    
    # Key metrics
    average_quality: float = 0.0
    quality_consistency: float = 0.0  # Standard deviation
    pass_rate: float = 0.0
    
    # Top issues
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Trends
    quality_trend: Optional[TrendAnalysis] = None
    performance_trend: Optional[TrendAnalysis] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_health": self.overall_health,
            "health_score": round(self.health_score, 1),
            "metrics": {
                "average_quality": round(self.average_quality, 3),
                "quality_consistency": round(self.quality_consistency, 3),
                "pass_rate": round(self.pass_rate, 3)
            },
            "issues": {
                "critical": self.critical_issues[:5],
                "warnings": self.warnings[:10],
                "recommendations": self.recommendations[:5]
            },
            "trends": {
                "quality": self.quality_trend.to_dict() if self.quality_trend else None,
                "performance": self.performance_trend.to_dict() if self.performance_trend else None
            }
        }


@dataclass
class ConfigurationRecommendation:
    """Configuration recommendation with justification."""
    parameter: str
    current_value: Any
    recommended_value: Any
    justification: str
    impact: str  # "high", "medium", "low"
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "current": self.current_value,
            "recommended": self.recommended_value,
            "justification": self.justification,
            "impact": self.impact,
            "confidence": round(self.confidence, 2)
        }


@dataclass
class AggregateReport:
    """Complete aggregate validation report."""
    report_id: str
    timestamp: str
    level: ReportLevel
    
    # Summary statistics
    total_files: int = 0
    total_batches: int = 0
    overall_success_rate: float = 0.0
    
    # Quality metrics
    quality_dashboard: Optional[QualityDashboard] = None
    
    # Detailed statistics
    file_reports: List[Dict[str, Any]] = field(default_factory=list)
    batch_statistics: Optional[BatchStatistics] = None
    
    # Analysis
    trend_analyses: List[TrendAnalysis] = field(default_factory=list)
    configuration_recommendations: List[ConfigurationRecommendation] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "summary": {
                "total_files": self.total_files,
                "total_batches": self.total_batches,
                "success_rate": round(self.overall_success_rate, 3)
            },
            "dashboard": self.quality_dashboard.to_dict() if self.quality_dashboard else None,
            "trends": [t.to_dict() for t in self.trend_analyses],
            "recommendations": [r.to_dict() for r in self.configuration_recommendations],
            "metadata": self.metadata
        }


class ValidationReportAggregator:
    """
    Aggregates validation reports with multi-level analysis and insights.
    
    This class provides comprehensive reporting capabilities including
    trend analysis, configuration recommendations, and quality dashboards.
    """
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize report aggregator.
        
        Args:
            config: Parser configuration
        """
        self.config = config or DEFAULT_CONFIG
        
        # Report storage
        self.file_reports: List[ValidationPipelineResult] = []
        self.batch_reports: List[BatchStatistics] = []
        self.historical_data: List[Dict[str, Any]] = []
        
        logger.info("ValidationReportAggregator initialized")
    
    def add_file_report(self, result: ValidationPipelineResult) -> None:
        """Add a single file validation report."""
        self.file_reports.append(result)
    
    def add_batch_report(self, statistics: BatchStatistics) -> None:
        """Add a batch validation report."""
        self.batch_reports.append(statistics)
    
    def add_historical_data(self, data: List[Dict[str, Any]]) -> None:
        """Add historical data for trend analysis."""
        self.historical_data.extend(data)
    
    def generate_aggregate_report(
        self,
        level: ReportLevel = ReportLevel.BATCH,
        include_trends: bool = True,
        include_recommendations: bool = True
    ) -> AggregateReport:
        """
        Generate comprehensive aggregate report.
        
        Args:
            level: Reporting level
            include_trends: Whether to include trend analysis
            include_recommendations: Whether to include recommendations
            
        Returns:
            AggregateReport with complete analysis
        """
        logger.info(f"Generating {level.value} level aggregate report")
        
        # Create report
        report = AggregateReport(
            report_id=self._generate_report_id(),
            timestamp=datetime.now().isoformat(),
            level=level
        )
        
        # Aggregate based on level
        if level == ReportLevel.FILE:
            self._aggregate_file_level(report)
        elif level == ReportLevel.BATCH:
            self._aggregate_batch_level(report)
        elif level == ReportLevel.CAMPAIGN:
            self._aggregate_campaign_level(report)
        else:  # SUMMARY
            self._aggregate_summary_level(report)
        
        # Generate quality dashboard
        report.quality_dashboard = self._generate_dashboard(report)
        
        # Analyze trends if requested
        if include_trends:
            report.trend_analyses = self._analyze_trends()
        
        # Generate recommendations if requested
        if include_recommendations:
            report.configuration_recommendations = self._generate_recommendations(report)
        
        # Add metadata
        report.metadata = self._generate_metadata()
        
        logger.info(f"Report generated: {report.report_id}")
        
        return report
    
    def _aggregate_file_level(self, report: AggregateReport) -> None:
        """Aggregate file-level reports."""
        report.total_files = len(self.file_reports)
        
        if not self.file_reports:
            return
        
        # Calculate success rate
        successful = sum(1 for r in self.file_reports if r.success)
        report.overall_success_rate = successful / len(self.file_reports)
        
        # Extract file report summaries
        for result in self.file_reports[:100]:  # Limit to 100 for performance
            summary = {
                "success": result.success,
                "validation_passed": result.validation_passed,
                "quality_passed": result.quality_passed,
                "overall_score": result.overall_score,
                "duration": result.total_duration,
                "errors": len(result.errors),
                "warnings": len(result.warnings)
            }
            report.file_reports.append(summary)
    
    def _aggregate_batch_level(self, report: AggregateReport) -> None:
        """Aggregate batch-level reports."""
        report.total_batches = len(self.batch_reports)
        
        if not self.batch_reports:
            self._aggregate_file_level(report)  # Fall back to file level
            return
        
        # Aggregate across batches
        total_files = sum(b.total_files_processed for b in self.batch_reports)
        total_successful = sum(b.successful_validations for b in self.batch_reports)
        
        report.total_files = total_files
        report.overall_success_rate = total_successful / max(total_files, 1)
        
        # Combine batch statistics
        if len(self.batch_reports) == 1:
            report.batch_statistics = self.batch_reports[0]
        else:
            # Merge multiple batch statistics
            report.batch_statistics = self._merge_batch_statistics(self.batch_reports)
    
    def _aggregate_campaign_level(self, report: AggregateReport) -> None:
        """Aggregate campaign-level reports (multiple batches over time)."""
        self._aggregate_batch_level(report)
        
        # Add campaign-specific analysis
        report.metadata["campaign_duration"] = self._calculate_campaign_duration()
        report.metadata["daily_throughput"] = self._calculate_daily_throughput()
        report.metadata["resource_utilization"] = self._calculate_resource_utilization()
    
    def _aggregate_summary_level(self, report: AggregateReport) -> None:
        """Generate executive summary."""
        self._aggregate_batch_level(report)
        
        # Keep only high-level metrics
        report.file_reports = []  # Clear detailed reports
        
        # Add summary insights
        report.metadata["executive_summary"] = self._generate_executive_summary()
    
    def _generate_dashboard(self, report: AggregateReport) -> QualityDashboard:
        """Generate quality dashboard."""
        dashboard = QualityDashboard(
            overall_health="unknown",
            health_score=0.0
        )
        
        # Calculate metrics from available data
        quality_scores = []
        
        # From file reports
        for file_report in self.file_reports:
            if file_report.overall_score > 0:
                quality_scores.append(file_report.overall_score)
        
        # From batch reports
        for batch in self.batch_reports:
            if batch.average_quality_score > 0:
                quality_scores.append(batch.average_quality_score)
        
        if quality_scores:
            dashboard.average_quality = statistics.mean(quality_scores)
            dashboard.quality_consistency = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            dashboard.pass_rate = report.overall_success_rate
            
            # Calculate health score
            health_score = (
                dashboard.average_quality * 40 +  # 40% weight on quality
                dashboard.pass_rate * 40 +         # 40% weight on pass rate
                (1 - dashboard.quality_consistency) * 20  # 20% weight on consistency
            ) * 100
            
            dashboard.health_score = health_score
            
            # Determine health status
            if health_score >= 90:
                dashboard.overall_health = "excellent"
            elif health_score >= 75:
                dashboard.overall_health = "good"
            elif health_score >= 60:
                dashboard.overall_health = "fair"
            else:
                dashboard.overall_health = "poor"
        
        # Collect issues
        dashboard.critical_issues = self._collect_critical_issues()
        dashboard.warnings = self._collect_warnings()
        dashboard.recommendations = self._collect_top_recommendations()
        
        return dashboard
    
    def _analyze_trends(self) -> List[TrendAnalysis]:
        """Analyze trends in validation results."""
        trends = []
        
        # Quality trend
        quality_trend = self._analyze_quality_trend()
        if quality_trend:
            trends.append(quality_trend)
        
        # Performance trend
        performance_trend = self._analyze_performance_trend()
        if performance_trend:
            trends.append(performance_trend)
        
        # Error trend
        error_trend = self._analyze_error_trend()
        if error_trend:
            trends.append(error_trend)
        
        return trends
    
    def _analyze_quality_trend(self) -> Optional[TrendAnalysis]:
        """Analyze quality score trends."""
        if len(self.batch_reports) < 2:
            return None
        
        # Extract quality scores over time
        data_points = []
        for i, batch in enumerate(self.batch_reports):
            if batch.average_quality_score > 0:
                # Use index as pseudo-timestamp
                data_points.append((float(i), batch.average_quality_score))
        
        if len(data_points) < 2:
            return None
        
        # Calculate trend
        first_half = [v for _, v in data_points[:len(data_points)//2]]
        second_half = [v for _, v in data_points[len(data_points)//2:]]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        change_rate = ((avg_second - avg_first) / avg_first) * 100 if avg_first > 0 else 0
        
        # Determine direction
        if change_rate > 5:
            direction = "improving"
        elif change_rate < -5:
            direction = "declining"
        else:
            direction = "stable"
        
        return TrendAnalysis(
            trend_type="quality",
            direction=direction,
            change_rate=change_rate,
            confidence=0.8 if len(data_points) > 5 else 0.5,
            data_points=data_points
        )
    
    def _analyze_performance_trend(self) -> Optional[TrendAnalysis]:
        """Analyze processing performance trends."""
        if len(self.batch_reports) < 2:
            return None
        
        # Extract processing times
        data_points = []
        for i, batch in enumerate(self.batch_reports):
            if batch.average_processing_time > 0:
                data_points.append((float(i), batch.average_processing_time))
        
        if len(data_points) < 2:
            return None
        
        # Calculate trend
        first_half = [v for _, v in data_points[:len(data_points)//2]]
        second_half = [v for _, v in data_points[len(data_points)//2:]]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        # For performance, negative change is good (faster)
        change_rate = ((avg_second - avg_first) / avg_first) * 100 if avg_first > 0 else 0
        
        if change_rate < -5:
            direction = "improving"
        elif change_rate > 5:
            direction = "declining"
        else:
            direction = "stable"
        
        return TrendAnalysis(
            trend_type="performance",
            direction=direction,
            change_rate=abs(change_rate),
            confidence=0.7,
            data_points=data_points
        )
    
    def _analyze_error_trend(self) -> Optional[TrendAnalysis]:
        """Analyze error rate trends."""
        if len(self.batch_reports) < 2:
            return None
        
        # Extract error rates
        data_points = []
        for i, batch in enumerate(self.batch_reports):
            error_rate = batch.failed_validations / max(batch.total_files_processed, 1)
            data_points.append((float(i), error_rate))
        
        if len(data_points) < 2:
            return None
        
        # Calculate trend
        first_half = [v for _, v in data_points[:len(data_points)//2]]
        second_half = [v for _, v in data_points[len(data_points)//2:]]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        change_rate = ((avg_second - avg_first) / max(avg_first, 0.01)) * 100
        
        # For errors, negative change is good
        if change_rate < -5:
            direction = "improving"
        elif change_rate > 5:
            direction = "declining"
        else:
            direction = "stable"
        
        return TrendAnalysis(
            trend_type="errors",
            direction=direction,
            change_rate=abs(change_rate),
            confidence=0.75,
            data_points=data_points
        )
    
    def _generate_recommendations(self, report: AggregateReport) -> List[ConfigurationRecommendation]:
        """Generate configuration recommendations."""
        recommendations = []
        
        # Analyze current performance
        if report.quality_dashboard:
            # Quality-based recommendations
            if report.quality_dashboard.average_quality < 0.9:
                recommendations.append(ConfigurationRecommendation(
                    parameter="validation.quality_threshold",
                    current_value=self.config.validation.quality_threshold,
                    recommended_value=0.85,
                    justification="Lower quality threshold to match actual performance",
                    impact="medium",
                    confidence=0.8
                ))
            
            # Consistency-based recommendations
            if report.quality_dashboard.quality_consistency > 0.2:
                recommendations.append(ConfigurationRecommendation(
                    parameter="validation.enable_round_trip_test",
                    current_value=self.config.validation.enable_round_trip_test,
                    recommended_value=True,
                    justification="High quality variance suggests need for stricter validation",
                    impact="high",
                    confidence=0.7
                ))
        
        # Error-based recommendations
        if self.batch_reports:
            latest_batch = self.batch_reports[-1]
            
            # Memory errors
            if latest_batch.error_types.get("memory_error", 0) > 0:
                recommendations.append(ConfigurationRecommendation(
                    parameter="processing.max_file_size_mb",
                    current_value=self.config.processing.max_file_size_mb,
                    recommended_value=5.0,
                    justification="Memory errors detected, reduce max file size",
                    impact="high",
                    confidence=0.9
                ))
                
                recommendations.append(ConfigurationRecommendation(
                    parameter="tokenizer.max_seq_length",
                    current_value=self.config.tokenizer.max_seq_length,
                    recommended_value=1024,
                    justification="Reduce sequence length to prevent memory issues",
                    impact="high",
                    confidence=0.85
                ))
            
            # Tokenization errors
            if latest_batch.error_types.get("tokenization_error", 0) > 2:
                recommendations.append(ConfigurationRecommendation(
                    parameter="tokenization",
                    current_value=self.config.tokenization,
                    recommended_value="REMI",
                    justification="Multiple tokenization failures, switch to more robust strategy",
                    impact="high",
                    confidence=0.8
                ))
        
        # Performance-based recommendations
        if report.batch_statistics and report.batch_statistics.average_processing_time > 30:
            recommendations.append(ConfigurationRecommendation(
                parameter="processing.parallel_processing",
                current_value=self.config.processing.parallel_processing,
                recommended_value=True,
                justification="Slow processing detected, enable parallelization",
                impact="high",
                confidence=0.9
            ))
        
        # Sort by impact and confidence
        recommendations.sort(key=lambda r: (
            {"high": 3, "medium": 2, "low": 1}.get(r.impact, 0) * r.confidence
        ), reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _merge_batch_statistics(self, batches: List[BatchStatistics]) -> BatchStatistics:
        """Merge multiple batch statistics."""
        merged = BatchStatistics()
        
        for batch in batches:
            merged.total_files_processed += batch.total_files_processed
            merged.successful_validations += batch.successful_validations
            merged.failed_validations += batch.failed_validations
            merged.skipped_files += batch.skipped_files
            merged.total_processing_time += batch.total_processing_time
            merged.total_tokens_processed += batch.total_tokens_processed
            
            # Merge error types
            for error_type, count in batch.error_types.items():
                merged.error_types[error_type] += count
            
            # Collect all issues
            merged.common_issues.extend(batch.common_issues)
        
        # Recalculate averages
        if merged.total_files_processed > 0:
            merged.average_processing_time = merged.total_processing_time / merged.total_files_processed
            merged.average_tokens_per_file = merged.total_tokens_processed / merged.total_files_processed
        
        # Recalculate quality metrics
        quality_scores = []
        for batch in batches:
            if batch.average_quality_score > 0:
                # Weight by number of files
                weight = batch.total_files_processed
                quality_scores.extend([batch.average_quality_score] * weight)
        
        if quality_scores:
            merged.average_quality_score = statistics.mean(quality_scores)
            merged.min_quality_score = min(b.min_quality_score for b in batches if b.min_quality_score < 1)
            merged.max_quality_score = max(b.max_quality_score for b in batches if b.max_quality_score > 0)
        
        # Merge quality distribution
        for batch in batches:
            for category, count in batch.quality_distribution.items():
                merged.quality_distribution[category] = merged.quality_distribution.get(category, 0) + count
        
        # Deduplicate and sort common issues
        issue_counter = Counter(merged.common_issues)
        merged.common_issues = [issue for issue, _ in issue_counter.most_common(10)]
        
        return merged
    
    def _collect_critical_issues(self) -> List[str]:
        """Collect critical issues from reports."""
        issues = []
        
        for report in self.file_reports[:50]:  # Sample
            if report.quality_metrics and report.quality_metrics.critical_issues:
                issues.extend(report.quality_metrics.critical_issues[:2])
        
        # Deduplicate and return top issues
        issue_counter = Counter(issues)
        return [issue for issue, _ in issue_counter.most_common(5)]
    
    def _collect_warnings(self) -> List[str]:
        """Collect warnings from reports."""
        warnings = []
        
        for report in self.file_reports[:50]:  # Sample
            warnings.extend(report.warnings[:2])
        
        for batch in self.batch_reports:
            warnings.extend(batch.common_issues[:5])
        
        # Deduplicate and return top warnings
        warning_counter = Counter(warnings)
        return [warning for warning, _ in warning_counter.most_common(10)]
    
    def _collect_top_recommendations(self) -> List[str]:
        """Collect top recommendations."""
        recommendations = []
        
        for report in self.file_reports[:20]:  # Sample
            recommendations.extend(report.recommendations[:1])
        
        # Deduplicate and return top recommendations
        rec_counter = Counter(recommendations)
        return [rec for rec, _ in rec_counter.most_common(5)]
    
    def _calculate_campaign_duration(self) -> float:
        """Calculate total campaign duration in hours."""
        if not self.batch_reports:
            return 0.0
        
        total_time = sum(b.total_processing_time for b in self.batch_reports)
        return total_time / 3600  # Convert to hours
    
    def _calculate_daily_throughput(self) -> float:
        """Calculate average files processed per day."""
        duration_hours = self._calculate_campaign_duration()
        if duration_hours == 0:
            return 0.0
        
        total_files = sum(b.total_files_processed for b in self.batch_reports)
        duration_days = duration_hours / 24
        
        return total_files / max(duration_days, 1)
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        utilization = {
            "cpu_efficiency": 0.0,
            "memory_efficiency": 0.0,
            "throughput_efficiency": 0.0
        }
        
        if self.batch_reports:
            # Estimate based on processing patterns
            avg_time = statistics.mean(b.average_processing_time for b in self.batch_reports)
            
            # CPU efficiency (lower time = higher efficiency)
            utilization["cpu_efficiency"] = min(1.0, 10.0 / max(avg_time, 1))
            
            # Memory efficiency (fewer errors = higher efficiency)
            total_memory_errors = sum(
                b.error_types.get("memory_error", 0) for b in self.batch_reports
            )
            total_files = sum(b.total_files_processed for b in self.batch_reports)
            utilization["memory_efficiency"] = 1.0 - (total_memory_errors / max(total_files, 1))
            
            # Throughput efficiency
            utilization["throughput_efficiency"] = min(1.0, self._calculate_daily_throughput() / 1000)
        
        return utilization
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary text."""
        if not self.batch_reports:
            return "No validation data available."
        
        total_files = sum(b.total_files_processed for b in self.batch_reports)
        success_rate = sum(b.successful_validations for b in self.batch_reports) / max(total_files, 1)
        avg_quality = statistics.mean(
            b.average_quality_score for b in self.batch_reports 
            if b.average_quality_score > 0
        ) if self.batch_reports else 0
        
        summary = f"""
Validation Campaign Summary:
- Processed {total_files:,} files across {len(self.batch_reports)} batches
- Overall success rate: {success_rate:.1%}
- Average quality score: {avg_quality:.2%}
- Total processing time: {self._calculate_campaign_duration():.1f} hours
- Daily throughput: {self._calculate_daily_throughput():.0f} files/day

Key Findings:
"""
        
        # Add trend summary
        quality_trend = self._analyze_quality_trend()
        if quality_trend:
            summary += f"- Quality trend: {quality_trend.direction} ({quality_trend.change_rate:+.1f}%)\n"
        
        performance_trend = self._analyze_performance_trend()
        if performance_trend:
            summary += f"- Performance trend: {performance_trend.direction}\n"
        
        # Add top issues
        critical_issues = self._collect_critical_issues()
        if critical_issues:
            summary += f"- Top critical issue: {critical_issues[0]}\n"
        
        return summary.strip()
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            "generator": "ValidationReportAggregator",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "tokenization": self.config.tokenization,
                "quality_threshold": self.config.validation.quality_threshold,
                "enable_round_trip": self.config.validation.enable_round_trip_test
            }
        }
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"report_{timestamp}"
    
    def export_report(
        self,
        report: AggregateReport,
        output_path: Path,
        format: ReportFormat = ReportFormat.JSON
    ) -> None:
        """
        Export report to file.
        
        Args:
            report: Report to export
            output_path: Output file path
            format: Export format
        """
        logger.info(f"Exporting report to {output_path} as {format.value}")
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add encoding='utf-8' to all file writes to handle emoji characters
        if format == ReportFormat.JSON:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2)
                
        elif format == ReportFormat.MARKDOWN:
            content = self._format_markdown(report)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif format == ReportFormat.HTML:
            content = self._format_html(report)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif format == ReportFormat.TEXT:
            content = self._format_text(report)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif format == ReportFormat.CSV:
            content = self._format_csv(report)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Report exported successfully to {output_path}")
    
    def _format_markdown(self, report: AggregateReport) -> str:
        """Format report as Markdown."""
        lines = [
            f"# Validation Report - {report.report_id}",
            f"*Generated: {report.timestamp}*",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Add dashboard if available
        if report.quality_dashboard:
            dashboard = report.quality_dashboard
            lines.extend([
                f"### Overall Health: {dashboard.overall_health.upper()} ({dashboard.health_score:.1f}/100)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Average Quality | {dashboard.average_quality:.2%} |",
                f"| Pass Rate | {dashboard.pass_rate:.2%} |",
                f"| Quality Consistency | {dashboard.quality_consistency:.3f} |",
                ""
            ])
            
            # Add critical issues
            if dashboard.critical_issues:
                lines.extend([
                    "### 🚨 Critical Issues",
                    ""
                ])
                for issue in dashboard.critical_issues:
                    lines.append(f"- {issue}")
                lines.append("")
        
        # Add summary statistics
        lines.extend([
            "## Summary Statistics",
            "",
            f"- **Total Files:** {report.total_files:,}",
            f"- **Total Batches:** {report.total_batches}",
            f"- **Success Rate:** {report.overall_success_rate:.1%}",
            ""
        ])
        
        # Add trends
        if report.trend_analyses:
            lines.extend([
                "## Trends",
                ""
            ])
            for trend in report.trend_analyses:
                icon = "📈" if trend.direction == "improving" else "📉" if trend.direction == "declining" else "📊"
                lines.append(f"### {icon} {trend.trend_type.title()} Trend: {trend.direction.upper()}")
                lines.append(f"- Change Rate: {trend.change_rate:+.1f}%")
                lines.append(f"- Confidence: {trend.confidence:.0%}")
                lines.append("")
        
        # Add recommendations
        if report.configuration_recommendations:
            lines.extend([
                "## Configuration Recommendations",
                ""
            ])
            for i, rec in enumerate(report.configuration_recommendations[:5], 1):
                impact_icon = "🔴" if rec.impact == "high" else "🟡" if rec.impact == "medium" else "🟢"
                lines.append(f"{i}. {impact_icon} **{rec.parameter}**")
                lines.append(f"   - Current: `{rec.current_value}`")
                lines.append(f"   - Recommended: `{rec.recommended_value}`")
                lines.append(f"   - Justification: {rec.justification}")
                lines.append(f"   - Confidence: {rec.confidence:.0%}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_html(self, report: AggregateReport) -> str:
        """Format report as HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Validation Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .health-excellent {{ color: #28a745; }}
        .health-good {{ color: #17a2b8; }}
        .health-fair {{ color: #ffc107; }}
        .health-poor {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric-card {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .trend-improving {{ color: green; }}
        .trend-declining {{ color: red; }}
        .trend-stable {{ color: gray; }}
        .recommendation {{ background: #e9ecef; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }}
        .impact-high {{ border-left-color: #dc3545; }}
        .impact-medium {{ border-left-color: #ffc107; }}
        .impact-low {{ border-left-color: #28a745; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Validation Report</h1>
        <p>Report ID: {report.report_id}</p>
        <p>Generated: {report.timestamp}</p>
    </div>
"""
        
        # Add dashboard
        if report.quality_dashboard:
            dashboard = report.quality_dashboard
            health_class = f"health-{dashboard.overall_health}"
            html += f"""
    <div class="metric-card">
        <h2>Quality Dashboard</h2>
        <h3 class="{health_class}">Overall Health: {dashboard.overall_health.upper()} ({dashboard.health_score:.1f}/100)</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average Quality</td><td>{dashboard.average_quality:.2%}</td></tr>
            <tr><td>Pass Rate</td><td>{dashboard.pass_rate:.2%}</td></tr>
            <tr><td>Quality Consistency</td><td>{dashboard.quality_consistency:.3f}</td></tr>
        </table>
    </div>
"""
        
        # Add summary
        html += f"""
    <div class="metric-card">
        <h2>Summary Statistics</h2>
        <ul>
            <li><strong>Total Files:</strong> {report.total_files:,}</li>
            <li><strong>Total Batches:</strong> {report.total_batches}</li>
            <li><strong>Success Rate:</strong> {report.overall_success_rate:.1%}</li>
        </ul>
    </div>
"""
        
        # Add trends
        if report.trend_analyses:
            html += '<div class="metric-card"><h2>Trends</h2>'
            for trend in report.trend_analyses:
                trend_class = f"trend-{trend.direction}"
                html += f"""
        <div>
            <h3 class="{trend_class}">{trend.trend_type.title()}: {trend.direction.upper()}</h3>
            <p>Change Rate: {trend.change_rate:+.1f}% | Confidence: {trend.confidence:.0%}</p>
        </div>
"""
            html += '</div>'
        
        # Add recommendations
        if report.configuration_recommendations:
            html += '<div class="metric-card"><h2>Recommendations</h2>'
            for rec in report.configuration_recommendations[:5]:
                impact_class = f"impact-{rec.impact}"
                html += f"""
        <div class="recommendation {impact_class}">
            <strong>{rec.parameter}</strong><br>
            Current: <code>{rec.current_value}</code> → Recommended: <code>{rec.recommended_value}</code><br>
            <em>{rec.justification}</em><br>
            <small>Confidence: {rec.confidence:.0%}</small>
        </div>
"""
            html += '</div>'
        
        html += """
</body>
</html>
"""
        return html
    
    def _format_text(self, report: AggregateReport) -> str:
        """Format report as plain text."""
        lines = [
            "=" * 70,
            f"VALIDATION REPORT - {report.report_id}",
            f"Generated: {report.timestamp}",
            "=" * 70,
            ""
        ]
        
        # Dashboard
        if report.quality_dashboard:
            dashboard = report.quality_dashboard
            lines.extend([
                "QUALITY DASHBOARD",
                "-" * 40,
                f"Overall Health: {dashboard.overall_health.upper()} ({dashboard.health_score:.1f}/100)",
                f"Average Quality: {dashboard.average_quality:.2%}",
                f"Pass Rate: {dashboard.pass_rate:.2%}",
                f"Quality Consistency: {dashboard.quality_consistency:.3f}",
                ""
            ])
            
            if dashboard.critical_issues:
                lines.extend([
                    "Critical Issues:",
                ])
                for issue in dashboard.critical_issues:
                    lines.append(f"  ! {issue}")
                lines.append("")
        
        # Summary
        lines.extend([
            "SUMMARY STATISTICS",
            "-" * 40,
            f"Total Files: {report.total_files:,}",
            f"Total Batches: {report.total_batches}",
            f"Success Rate: {report.overall_success_rate:.1%}",
            ""
        ])
        
        # Trends
        if report.trend_analyses:
            lines.extend([
                "TRENDS",
                "-" * 40
            ])
            for trend in report.trend_analyses:
                lines.append(f"{trend.trend_type.title()}: {trend.direction.upper()} ({trend.change_rate:+.1f}%)")
            lines.append("")
        
        # Recommendations
        if report.configuration_recommendations:
            lines.extend([
                "TOP RECOMMENDATIONS",
                "-" * 40
            ])
            for i, rec in enumerate(report.configuration_recommendations[:3], 1):
                lines.extend([
                    f"{i}. {rec.parameter}",
                    f"   Current: {rec.current_value} -> Recommended: {rec.recommended_value}",
                    f"   {rec.justification}",
                    ""
                ])
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def _format_csv(self, report: AggregateReport) -> str:
        """Format report as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["Report ID", report.report_id])
        writer.writerow(["Generated", report.timestamp])
        writer.writerow(["Level", report.level.value])
        writer.writerow([])
        
        # Write summary
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Files", report.total_files])
        writer.writerow(["Total Batches", report.total_batches])
        writer.writerow(["Success Rate", f"{report.overall_success_rate:.3f}"])
        
        if report.quality_dashboard:
            writer.writerow(["Overall Health", report.quality_dashboard.overall_health])
            writer.writerow(["Health Score", f"{report.quality_dashboard.health_score:.1f}"])
            writer.writerow(["Average Quality", f"{report.quality_dashboard.average_quality:.3f}"])
            writer.writerow(["Pass Rate", f"{report.quality_dashboard.pass_rate:.3f}"])
        
        writer.writerow([])
        
        # Write trends
        if report.trend_analyses:
            writer.writerow(["Trend Type", "Direction", "Change Rate", "Confidence"])
            for trend in report.trend_analyses:
                writer.writerow([
                    trend.trend_type,
                    trend.direction,
                    f"{trend.change_rate:.2f}",
                    f"{trend.confidence:.3f}"
                ])
            writer.writerow([])
        
        # Write recommendations
        if report.configuration_recommendations:
            writer.writerow(["Parameter", "Current", "Recommended", "Impact", "Confidence"])
            for rec in report.configuration_recommendations:
                writer.writerow([
                    rec.parameter,
                    str(rec.current_value),
                    str(rec.recommended_value),
                    rec.impact,
                    f"{rec.confidence:.2f}"
                ])
        
        return output.getvalue()


# Convenience functions

def create_aggregator(config: Optional[MidiParserConfig] = None) -> ValidationReportAggregator:
    """Create a report aggregator instance."""
    return ValidationReportAggregator(config)


def generate_campaign_report(
    batch_reports: List[BatchStatistics],
    output_path: Path,
    format: ReportFormat = ReportFormat.MARKDOWN
) -> AggregateReport:
    """
    Generate a campaign-level report from batch statistics.
    
    Args:
        batch_reports: List of batch statistics
        output_path: Output file path
        format: Export format
        
    Returns:
        Generated AggregateReport
    """
    aggregator = ValidationReportAggregator()
    
    for batch in batch_reports:
        aggregator.add_batch_report(batch)
    
    report = aggregator.generate_aggregate_report(
        level=ReportLevel.CAMPAIGN,
        include_trends=True,
        include_recommendations=True
    )
    
    aggregator.export_report(report, output_path, format)
    
    return report


# Export main classes
__all__ = [
    'ValidationReportAggregator',
    'AggregateReport',
    'QualityDashboard',
    'TrendAnalysis',
    'ConfigurationRecommendation',
    'ReportLevel',
    'ReportFormat',
    'create_aggregator',
    'generate_campaign_report'
]