"""
Batch validation coordinator for large-scale MIDI validation.

This module handles batch validation with parallel processing, progress monitoring,
aggregate statistics calculation, and batch optimization for comprehensive validation
campaigns.
"""

import logging
import time
import concurrent.futures
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from collections import defaultdict

from miditoolkit import MidiFile

from midi_parser.config.defaults import MidiParserConfig, ProcessingConfig, DEFAULT_CONFIG
from midi_parser.core.midi_loader import load_midi_file
from midi_parser.validation.validation_pipeline_orchestrator import (
    ValidationPipelineOrchestrator,
    ValidationPipelineConfig,
    ValidationPipelineResult,
    PipelineStatus
)
from midi_parser.validation.error_recovery_manager import ErrorRecoveryManager

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of batch processing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Single item in batch processing queue."""
    file_path: Path
    priority: int = 0
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[ValidationPipelineResult] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": str(self.file_path.name),
            "status": self.status.value,
            "processing_time": round(self.processing_time, 2),
            "retry_count": self.retry_count,
            "success": self.result.success if self.result else None,
            "error": self.error_message
        }


@dataclass
class BatchProgress:
    """Progress tracking for batch validation."""
    total_files: int = 0
    completed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    current_file: Optional[str] = None
    start_time: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.completed_files == 0:
            return 0.0
        return self.successful_files / self.completed_files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total_files,
            "completed": self.completed_files,
            "successful": self.successful_files,
            "failed": self.failed_files,
            "skipped": self.skipped_files,
            "progress_percentage": round(self.progress_percentage, 1),
            "success_rate": round(self.success_rate * 100, 1),
            "current_file": self.current_file,
            "elapsed_time": round(self.elapsed_time, 1),
            "estimated_remaining": round(self.estimated_remaining, 1)
        }


@dataclass
class BatchStatistics:
    """Aggregate statistics for batch validation."""
    # File statistics
    total_files_processed: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    skipped_files: int = 0
    
    # Quality metrics
    average_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    total_tokens_processed: int = 0
    average_tokens_per_file: float = 0.0
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    common_issues: List[str] = field(default_factory=list)
    
    # Strategy performance
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files": {
                "total": self.total_files_processed,
                "successful": self.successful_validations,
                "failed": self.failed_validations,
                "skipped": self.skipped_files,
                "success_rate": round(self.successful_validations / max(self.total_files_processed, 1) * 100, 1)
            },
            "quality": {
                "average": round(self.average_quality_score, 3),
                "min": round(self.min_quality_score, 3),
                "max": round(self.max_quality_score, 3),
                "distribution": self.quality_distribution
            },
            "performance": {
                "total_time": round(self.total_processing_time, 1),
                "average_time": round(self.average_processing_time, 2),
                "total_tokens": self.total_tokens_processed,
                "average_tokens": round(self.average_tokens_per_file, 0)
            },
            "errors": dict(self.error_types),
            "common_issues": self.common_issues[:10],
            "strategy_performance": {
                k: round(v, 3) for k, v in self.strategy_success_rates.items()
            }
        }


@dataclass
class BatchConfiguration:
    """Configuration for batch processing."""
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 10  # Files per chunk
    stop_on_failure: bool = False
    max_retries: int = 2
    retry_on_error: bool = True
    skip_corrupted: bool = True
    
    # Progress reporting
    progress_interval: float = 1.0  # Seconds between progress updates
    progress_callback: Optional[Callable[[BatchProgress], None]] = None
    
    # Optimization
    adaptive_optimization: bool = True
    optimize_after_files: int = 10  # Optimize configuration after N files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parallel_processing": self.parallel_processing,
            "max_workers": self.max_workers,
            "chunk_size": self.chunk_size,
            "stop_on_failure": self.stop_on_failure,
            "max_retries": self.max_retries,
            "retry_on_error": self.retry_on_error,
            "skip_corrupted": self.skip_corrupted,
            "adaptive_optimization": self.adaptive_optimization
        }


class BatchValidationCoordinator:
    """
    Coordinates large-scale batch validation with parallel processing.
    
    This class manages batch validation across multiple files with
    parallel processing, progress monitoring, and optimization.
    """
    
    def __init__(
        self,
        config: Optional[MidiParserConfig] = None,
        batch_config: Optional[BatchConfiguration] = None
    ):
        """
        Initialize batch validation coordinator.
        
        Args:
            config: Parser configuration
            batch_config: Batch processing configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.batch_config = batch_config or BatchConfiguration()
        
        # Set max workers if not specified
        if self.batch_config.max_workers is None:
            self.batch_config.max_workers = (
                self.config.processing.max_workers or multiprocessing.cpu_count()
            )
        
        # Initialize components
        self.error_manager = ErrorRecoveryManager(self.config)
        
        # Batch state
        self.batch_items: List[BatchItem] = []
        self.progress = BatchProgress()
        self.statistics = BatchStatistics()
        self._cancel_requested = False
        
        logger.info(f"BatchValidationCoordinator initialized with "
                   f"{self.batch_config.max_workers} workers")
    
    def validate_batch(
        self,
        file_paths: List[Path],
        pipeline_config: Optional[ValidationPipelineConfig] = None,
        strategy: Optional[str] = None
    ) -> BatchStatistics:
        """
        Validate a batch of MIDI files.
        
        Args:
            file_paths: List of MIDI file paths
            pipeline_config: Pipeline configuration
            strategy: Tokenization strategy
            
        Returns:
            BatchStatistics with aggregate results
        """
        logger.info(f"Starting batch validation of {len(file_paths)} files")
        
        # Initialize batch
        self._initialize_batch(file_paths)
        
        # Start progress tracking
        self.progress.start_time = time.time()
        
        try:
            if self.batch_config.parallel_processing and len(file_paths) > 1:
                results = self._process_parallel(pipeline_config, strategy)
            else:
                results = self._process_sequential(pipeline_config, strategy)
            
            # Calculate final statistics
            self._calculate_statistics(results)
            
        except KeyboardInterrupt:
            logger.warning("Batch validation cancelled by user")
            self._cancel_requested = True
        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            raise
        finally:
            self.progress.elapsed_time = time.time() - self.progress.start_time
        
        logger.info(f"Batch validation completed: {self.statistics.successful_validations}/"
                   f"{self.statistics.total_files_processed} successful")
        
        return self.statistics
    
    def _initialize_batch(self, file_paths: List[Path]) -> None:
        """Initialize batch items and progress."""
        self.batch_items = [
            BatchItem(file_path=path, priority=i)
            for i, path in enumerate(file_paths)
        ]
        
        self.progress = BatchProgress(total_files=len(file_paths))
        self.statistics = BatchStatistics()
        self._cancel_requested = False
    
    def _process_sequential(
        self,
        pipeline_config: Optional[ValidationPipelineConfig],
        strategy: Optional[str]
    ) -> List[BatchItem]:
        """Process files sequentially."""
        results = []
        
        for item in self.batch_items:
            if self._cancel_requested:
                break
            
            if self.batch_config.stop_on_failure and self.progress.failed_files > 0:
                logger.warning("Stopping batch due to failure")
                break
            
            # Update progress
            self.progress.current_file = item.file_path.name
            self._report_progress()
            
            # Process file
            result = self._process_single_file(item, pipeline_config, strategy)
            results.append(result)
            
            # Update progress
            self.progress.completed_files += 1
            if result.result and result.result.success:
                self.progress.successful_files += 1
            elif result.status == BatchStatus.FAILED:
                self.progress.failed_files += 1
            else:
                self.progress.skipped_files += 1
            
            # Optimize configuration if enabled
            if (self.batch_config.adaptive_optimization and 
                self.progress.completed_files % self.batch_config.optimize_after_files == 0):
                self._optimize_configuration()
        
        return results
    
    def _process_parallel(
        self,
        pipeline_config: Optional[ValidationPipelineConfig],
        strategy: Optional[str]
    ) -> List[BatchItem]:
        """Process files in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_config.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._process_single_file,
                    item,
                    pipeline_config,
                    strategy
                ): item
                for item in self.batch_items
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(futures):
                if self._cancel_requested:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    
                    # Update progress
                    self.progress.completed_files += 1
                    if result.result and result.result.success:
                        self.progress.successful_files += 1
                    elif result.status == BatchStatus.FAILED:
                        self.progress.failed_files += 1
                    else:
                        self.progress.skipped_files += 1
                    
                    self._report_progress()
                    
                    # Check stop condition
                    if self.batch_config.stop_on_failure and self.progress.failed_files > 0:
                        logger.warning("Stopping batch due to failure")
                        for f in futures:
                            f.cancel()
                        break
                    
                except concurrent.futures.TimeoutError:
                    item = futures[future]
                    logger.error(f"Timeout processing {item.file_path}")
                    item.status = BatchStatus.FAILED
                    item.error_message = "Processing timeout"
                    results.append(item)
                    self.progress.failed_files += 1
                    
                except Exception as e:
                    item = futures[future]
                    logger.error(f"Error processing {item.file_path}: {e}")
                    item.status = BatchStatus.FAILED
                    item.error_message = str(e)
                    results.append(item)
                    self.progress.failed_files += 1
        
        return results
    
    def _process_single_file(
        self,
        item: BatchItem,
        pipeline_config: Optional[ValidationPipelineConfig],
        strategy: Optional[str]
    ) -> BatchItem:
        """
        Process a single file with error recovery.
        
        Args:
            item: Batch item to process
            pipeline_config: Pipeline configuration
            strategy: Tokenization strategy
            
        Returns:
            Updated BatchItem with results
        """
        start_time = time.time()
        
        try:
            # Create pipeline orchestrator for this file
            orchestrator = ValidationPipelineOrchestrator(
                self.config,
                pipeline_config
            )
            
            # Run validation
            result = orchestrator.validate_pipeline(
                item.file_path,
                strategy
            )
            
            item.result = result
            item.status = BatchStatus.COMPLETED if result.success else BatchStatus.FAILED
            
        except Exception as e:
            logger.error(f"Error processing {item.file_path}: {e}")
            
            # Try error recovery
            if self.batch_config.retry_on_error and item.retry_count < self.batch_config.max_retries:
                recovery = self.error_manager.handle_error(
                    e,
                    {"file": str(item.file_path), "strategy": strategy},
                    item.retry_count
                )
                
                if recovery.should_retry:
                    # Update config and retry
                    item.retry_count += 1
                    logger.info(f"Retrying {item.file_path} (attempt {item.retry_count})")
                    
                    # Apply config changes
                    if recovery.config_changes:
                        for key, value in recovery.config_changes.items():
                            if hasattr(self.config, key):
                                setattr(self.config, key, value)
                    
                    # Recursive retry
                    return self._process_single_file(item, pipeline_config, strategy)
                    
                elif recovery.should_skip:
                    item.status = BatchStatus.FAILED
                    item.error_message = recovery.message
                    logger.info(f"Skipping {item.file_path}: {recovery.message}")
                else:
                    item.status = BatchStatus.FAILED
                    item.error_message = str(e)
            else:
                item.status = BatchStatus.FAILED
                item.error_message = str(e)
        
        finally:
            item.processing_time = time.time() - start_time
        
        return item
    
    def _calculate_statistics(self, results: List[BatchItem]) -> None:
        """Calculate aggregate statistics from results."""
        self.statistics.total_files_processed = len(results)
        
        quality_scores = []
        token_counts = []
        strategy_results = defaultdict(list)
        
        for item in results:
            if item.status == BatchStatus.COMPLETED and item.result:
                self.statistics.successful_validations += 1
                
                # Quality scores
                if item.result.overall_score > 0:
                    quality_scores.append(item.result.overall_score)
                    self.statistics.min_quality_score = min(
                        self.statistics.min_quality_score,
                        item.result.overall_score
                    )
                    self.statistics.max_quality_score = max(
                        self.statistics.max_quality_score,
                        item.result.overall_score
                    )
                
                # Token counts
                if item.result.round_trip_result:
                    _, metrics = item.result.round_trip_result
                    token_counts.append(metrics.token_count)
                
                # Strategy performance
                if item.result.pipeline_state:
                    strategy = item.result.pipeline_state.stage_results.get(
                        "tokenization", {}
                    ).get("data", {}).get("strategy", "unknown")
                    strategy_results[strategy].append(item.result.success)
                
                # Collect issues
                self.statistics.common_issues.extend(item.result.warnings[:2])
                
            elif item.status == BatchStatus.FAILED:
                self.statistics.failed_validations += 1
                
                # Track error types
                if item.error_message:
                    # Simple error classification
                    if "corrupt" in item.error_message.lower():
                        self.statistics.error_types["corrupted_file"] += 1
                    elif "memory" in item.error_message.lower():
                        self.statistics.error_types["memory_error"] += 1
                    elif "token" in item.error_message.lower():
                        self.statistics.error_types["tokenization_error"] += 1
                    else:
                        self.statistics.error_types["other"] += 1
            else:
                self.statistics.skipped_files += 1
        
        # Calculate averages
        if quality_scores:
            self.statistics.average_quality_score = sum(quality_scores) / len(quality_scores)
            
            # Quality distribution
            for score in quality_scores:
                if score >= 0.95:
                    bucket = "excellent"
                elif score >= 0.90:
                    bucket = "good"
                elif score >= 0.80:
                    bucket = "fair"
                else:
                    bucket = "poor"
                self.statistics.quality_distribution[bucket] = \
                    self.statistics.quality_distribution.get(bucket, 0) + 1
        
        if token_counts:
            self.statistics.total_tokens_processed = sum(token_counts)
            self.statistics.average_tokens_per_file = sum(token_counts) / len(token_counts)
        
        # Calculate strategy success rates
        for strategy, results_list in strategy_results.items():
            if results_list:
                self.statistics.strategy_success_rates[strategy] = \
                    sum(results_list) / len(results_list)
        
        # Performance metrics
        self.statistics.total_processing_time = sum(
            item.processing_time for item in results
        )
        if results:
            self.statistics.average_processing_time = \
                self.statistics.total_processing_time / len(results)
        
        # Sort common issues by frequency
        from collections import Counter
        issue_counts = Counter(self.statistics.common_issues)
        self.statistics.common_issues = [
            issue for issue, _ in issue_counts.most_common(10)
        ]
    
    def _optimize_configuration(self) -> None:
        """Optimize configuration based on results so far."""
        logger.info("Optimizing configuration based on batch results")
        
        # Get error statistics
        error_stats = self.error_manager.get_error_statistics()
        
        # Get configuration suggestions
        suggestions = self.error_manager.suggest_configuration_adjustments()
        
        # Apply suggestions
        for key, value in suggestions.items():
            if hasattr(self.config, key):
                logger.info(f"Adjusting {key} to {value}")
                setattr(self.config, key, value)
        
        # Adjust batch configuration based on performance
        if self.statistics.average_processing_time > 60:  # Slow processing
            if self.batch_config.chunk_size > 5:
                self.batch_config.chunk_size = 5
                logger.info("Reduced chunk size for slow processing")
        
        if self.statistics.failed_validations > self.statistics.successful_validations * 0.2:
            # High failure rate
            self.batch_config.max_retries = min(self.batch_config.max_retries + 1, 5)
            logger.info(f"Increased max retries to {self.batch_config.max_retries}")
    
    def _report_progress(self) -> None:
        """Report current progress."""
        # Update elapsed time
        self.progress.elapsed_time = time.time() - self.progress.start_time
        
        # Estimate remaining time
        if self.progress.completed_files > 0:
            avg_time_per_file = self.progress.elapsed_time / self.progress.completed_files
            remaining_files = self.progress.total_files - self.progress.completed_files
            self.progress.estimated_remaining = avg_time_per_file * remaining_files
        
        # Call callback if provided
        if self.batch_config.progress_callback:
            self.batch_config.progress_callback(self.progress)
        
        # Log progress
        if self.progress.completed_files % 10 == 0 or self.progress.completed_files == self.progress.total_files:
            logger.info(f"Progress: {self.progress.completed_files}/{self.progress.total_files} "
                       f"({self.progress.progress_percentage:.1f}%) - "
                       f"Success rate: {self.progress.success_rate:.1%}")
    
    def cancel_batch(self) -> None:
        """Cancel batch processing."""
        logger.warning("Batch validation cancellation requested")
        self._cancel_requested = True
    
    def get_failed_files(self) -> List[Path]:
        """Get list of files that failed validation."""
        return [
            item.file_path
            for item in self.batch_items
            if item.status == BatchStatus.FAILED
        ]
    
    def get_successful_files(self) -> List[Path]:
        """Get list of files that passed validation."""
        return [
            item.file_path
            for item in self.batch_items
            if item.status == BatchStatus.COMPLETED and item.result and item.result.success
        ]
    
    def generate_batch_report(self) -> Dict[str, Any]:
        """Generate comprehensive batch validation report."""
        return {
            "configuration": self.batch_config.to_dict(),
            "progress": self.progress.to_dict(),
            "statistics": self.statistics.to_dict(),
            "failed_files": [str(p) for p in self.get_failed_files()],
            "error_analysis": self.error_manager.get_error_statistics(),
            "recommendations": self.error_manager.suggest_configuration_adjustments()
        }


# Convenience functions

def validate_directory(
    directory: Path,
    config: Optional[MidiParserConfig] = None,
    recursive: bool = True,
    parallel: bool = True
) -> BatchStatistics:
    """
    Validate all MIDI files in a directory.
    
    Args:
        directory: Directory containing MIDI files
        config: Parser configuration
        recursive: Whether to search recursively
        parallel: Whether to use parallel processing
        
    Returns:
        BatchStatistics with results
    """
    # Find MIDI files
    if recursive:
        midi_files = list(directory.rglob("*.mid")) + list(directory.rglob("*.midi"))
    else:
        midi_files = list(directory.glob("*.mid")) + list(directory.glob("*.midi"))
    
    logger.info(f"Found {len(midi_files)} MIDI files in {directory}")
    
    # Create coordinator
    batch_config = BatchConfiguration(parallel_processing=parallel)
    coordinator = BatchValidationCoordinator(config, batch_config)
    
    # Run validation
    return coordinator.validate_batch(midi_files)


# Export main classes
__all__ = [
    'BatchValidationCoordinator',
    'BatchConfiguration',
    'BatchStatistics',
    'BatchProgress',
    'BatchItem',
    'validate_directory'
]