"""
Batch processing operations for round-trip validation.

This module handles validation of multiple MIDI files, chunked MIDI files,
and parallel processing with proper error handling and progress tracking.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from miditoolkit import MidiFile
from midi_parser.core.midi_loader import ValidationResult, extract_metadata
from .validation_metrics import RoundTripMetrics

logger = logging.getLogger(__name__)


class BatchValidator:
    """
    Handles batch validation operations and parallel processing.
    
    This class manages validation of multiple MIDI files with support for
    parallel processing, progress tracking, and comprehensive error handling.
    """
    
    def __init__(self, validator):
        """
        Initialize batch validator.
        
        Args:
            validator: RoundTripValidator instance to use for individual validations
        """
        self.validator = validator
        
    def validate_batch(
        self,
        midi_files: List[MidiFile],
        strategy: Optional[str] = None,
        parallel: bool = False,
        stop_on_failure: bool = False,
        max_workers: Optional[int] = None
    ) -> List[Tuple[ValidationResult, RoundTripMetrics]]:
        """
        Validate multiple MIDI files in batch.
        
        Args:
            midi_files: List of MIDI files to validate
            strategy: Tokenization strategy (uses config default if None)
            parallel: Whether to use parallel processing
            stop_on_failure: Whether to stop on first validation failure
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (ValidationResult, RoundTripMetrics) tuples
        """
        if not midi_files:
            return []
        
        strategy = strategy or self.validator.config.tokenization
        results = []
        
        logger.info(f"Starting batch validation of {len(midi_files)} files with {strategy}")
        start_time = time.time()
        
        if parallel and len(midi_files) > 1:
            results = self._validate_parallel(
                midi_files, strategy, stop_on_failure, max_workers
            )
        else:
            results = self._validate_sequential(
                midi_files, strategy, stop_on_failure
            )
        
        # Log summary
        elapsed = time.time() - start_time
        valid_count = sum(1 for r, _ in results if r.is_valid)
        logger.info(f"Batch validation complete: {valid_count}/{len(results)} files passed "
                   f"in {elapsed:.2f}s")
        
        return results
    
    def _validate_sequential(
        self,
        midi_files: List[MidiFile],
        strategy: str,
        stop_on_failure: bool
    ) -> List[Tuple[ValidationResult, RoundTripMetrics]]:
        """Sequential validation processing."""
        results = []
        
        for i, midi in enumerate(midi_files):
            logger.info(f"Validating file {i+1}/{len(midi_files)}")
            
            try:
                result, metrics = self.validator.validate_round_trip(midi, strategy)
                results.append((result, metrics))
                
                if stop_on_failure and not result.is_valid:
                    logger.warning(f"Stopping batch validation at file {i+1} due to failure")
                    break
                    
            except Exception as e:
                logger.error(f"Error validating file {i+1}: {e}")
                error_result = ValidationResult(is_valid=False, errors=[str(e)])
                results.append((error_result, RoundTripMetrics()))
                
                if stop_on_failure:
                    logger.warning(f"Stopping batch validation at file {i+1} due to error")
                    break
        
        return results
    
    def _validate_parallel(
        self,
        midi_files: List[MidiFile],
        strategy: str,
        stop_on_failure: bool,
        max_workers: Optional[int]
    ) -> List[Tuple[ValidationResult, RoundTripMetrics]]:
        """Parallel validation processing."""
        import concurrent.futures
        import multiprocessing
        
        if max_workers is None:
            max_workers = self.validator.config.processing.max_workers or multiprocessing.cpu_count()
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit validation tasks
            futures = []
            for i, midi in enumerate(midi_files):
                future = executor.submit(
                    self._safe_validate,
                    midi, strategy, i + 1
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                    
                    if stop_on_failure and not result[0].is_valid:
                        logger.warning("Stopping batch validation due to failure")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                        
                except concurrent.futures.TimeoutError:
                    logger.error("Validation timeout")
                    timeout_result = ValidationResult(is_valid=False, errors=["Validation timeout"])
                    results.append((timeout_result, RoundTripMetrics()))
                    
                except Exception as e:
                    logger.error(f"Parallel validation error: {e}")
                    error_result = ValidationResult(is_valid=False, errors=[str(e)])
                    results.append((error_result, RoundTripMetrics()))
        
        return results
    
    def _safe_validate(
        self,
        midi: MidiFile,
        strategy: str,
        file_num: int
    ) -> Tuple[ValidationResult, RoundTripMetrics]:
        """Thread-safe validation wrapper."""
        try:
            logger.debug(f"Validating file {file_num} in thread")
            return self.validator.validate_round_trip(midi, strategy)
        except Exception as e:
            logger.error(f"Error in thread validation for file {file_num}: {e}")
            return (
                ValidationResult(is_valid=False, errors=[str(e)]),
                RoundTripMetrics()
            )
    
    def validate_chunked_midi(
        self,
        chunks: List[MidiFile],
        strategy: Optional[str] = None,
        original_midi: Optional[MidiFile] = None
    ) -> Tuple[ValidationResult, List[RoundTripMetrics]]:
        """
        Validate chunked MIDI files from chunk_midi_file function.
        
        Args:
            chunks: List of MIDI chunks
            strategy: Tokenization strategy
            original_midi: Original MIDI for reference comparison
            
        Returns:
            Tuple of (overall ValidationResult, list of chunk metrics)
        """
        if not chunks:
            return ValidationResult(is_valid=False, errors=["No chunks provided"]), []
        
        strategy = strategy or self.validator.config.tokenization
        overall_result = ValidationResult(is_valid=True)
        chunk_metrics = []
        
        logger.info(f"Validating {len(chunks)} MIDI chunks")
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Validating chunk {i+1}/{len(chunks)}")
            
            try:
                # Validate individual chunk
                result, metrics = self.validator.validate_round_trip(chunk, strategy)
                chunk_metrics.append(metrics)
                
                if not result.is_valid:
                    overall_result.add_warning(f"Chunk {i+1} failed validation: {result.errors}")
                
                # Track chunk-specific issues with higher tolerance for chunks
                if metrics.missing_notes_ratio > 0.05:
                    overall_result.add_warning(f"Chunk {i+1} has high missing notes ratio: "
                                             f"{metrics.missing_notes_ratio:.3f}")
                                             
            except Exception as e:
                logger.error(f"Error validating chunk {i+1}: {e}")
                overall_result.add_error(f"Chunk {i+1} validation error: {str(e)}")
                chunk_metrics.append(RoundTripMetrics())
        
        # Compare against original if provided
        if original_midi and chunk_metrics:
            self._validate_chunk_completeness(original_midi, chunk_metrics, overall_result)
        
        # Determine overall validity
        failed_chunks = len([m for m in chunk_metrics if m.overall_accuracy < 0.95])
        if failed_chunks > len(chunks) * 0.2:
            overall_result.is_valid = False
            overall_result.add_error(f"Too many chunks failed quality threshold: {failed_chunks}/{len(chunks)}")
        
        return overall_result, chunk_metrics
    
    def _validate_chunk_completeness(
        self,
        original_midi: MidiFile,
        chunk_metrics: List[RoundTripMetrics],
        overall_result: ValidationResult
    ) -> None:
        """Validate that chunks preserve the original MIDI content."""
        try:
            orig_metadata = extract_metadata(original_midi)
            total_notes_chunks = sum(m.total_notes_original for m in chunk_metrics)
            
            # Check note count preservation
            note_diff_ratio = abs(total_notes_chunks - orig_metadata.note_count) / max(orig_metadata.note_count, 1)
            if note_diff_ratio > 0.02:  # 2% tolerance
                overall_result.add_warning(
                    f"Total notes in chunks ({total_notes_chunks}) differs significantly from "
                    f"original ({orig_metadata.note_count}). Difference: {note_diff_ratio:.1%}"
                )
            
            # Check for empty chunks
            empty_chunks = len([m for m in chunk_metrics if m.total_notes_original == 0])
            if empty_chunks > 0:
                overall_result.add_warning(f"{empty_chunks} chunks are empty")
                
        except Exception as e:
            logger.error(f"Error validating chunk completeness: {e}")
            overall_result.add_warning("Could not validate chunk completeness against original")
    
    def validate_with_fallback(
        self,
        midi: MidiFile,
        primary_strategy: str,
        fallback_strategies: Optional[List[str]] = None
    ) -> Tuple[ValidationResult, RoundTripMetrics, str]:
        """
        Validate with automatic fallback to alternative strategies.
        
        Args:
            midi: MIDI file to validate
            primary_strategy: Primary tokenization strategy
            fallback_strategies: Alternative strategies to try
            
        Returns:
            Tuple of (ValidationResult, RoundTripMetrics, successful_strategy)
        """
        if fallback_strategies is None:
            fallback_strategies = ["REMI", "TSD", "Structured"]
            fallback_strategies = [s for s in fallback_strategies if s != primary_strategy]
        
        # Try primary strategy first
        logger.info(f"Attempting validation with primary strategy: {primary_strategy}")
        try:
            result, metrics = self.validator.validate_round_trip(midi, primary_strategy)
            
            if result.is_valid:
                return result, metrics, primary_strategy
        except Exception as e:
            logger.warning(f"Primary strategy {primary_strategy} failed with error: {e}")
        
        # Try fallback strategies
        logger.warning(f"Primary strategy {primary_strategy} failed, trying fallbacks")
        
        for strategy in fallback_strategies:
            logger.info(f"Attempting validation with fallback strategy: {strategy}")
            try:
                result, metrics = self.validator.validate_round_trip(midi, strategy)
                
                if result.is_valid:
                    result.add_warning(f"Used fallback strategy {strategy} after {primary_strategy} failed")
                    return result, metrics, strategy
                    
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy} failed with error: {e}")
                continue
        
        # All strategies failed
        logger.error("All strategies failed validation for MIDI file")
        final_result = ValidationResult(is_valid=False, errors=["All tokenization strategies failed"])
        return final_result, RoundTripMetrics(), primary_strategy
    
    def get_batch_statistics(
        self,
        results: List[Tuple[ValidationResult, RoundTripMetrics]]
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate comprehensive statistics from batch validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            Dictionary with batch statistics
        """
        if not results:
            return {}
        
        valid_results = [r for r, m in results if r.is_valid]
        metrics_list = [m for r, m in results if r.is_valid]
        
        stats = {
            'total_files': len(results),
            'successful_validations': len(valid_results),
            'failed_validations': len(results) - len(valid_results),
            'success_rate': len(valid_results) / len(results),
            'total_processing_time': sum(m.processing_time for m in metrics_list),
            'avg_processing_time': sum(m.processing_time for m in metrics_list) / max(len(metrics_list), 1),
            'total_tokens': sum(m.token_count for m in metrics_list),
            'avg_tokens_per_file': sum(m.token_count for m in metrics_list) / max(len(metrics_list), 1),
            'total_notes_original': sum(m.total_notes_original for m in metrics_list),
            'total_notes_reconstructed': sum(m.total_notes_reconstructed for m in metrics_list),
            'avg_accuracy': sum(m.overall_accuracy for m in metrics_list) / max(len(metrics_list), 1),
            'min_accuracy': min((m.overall_accuracy for m in metrics_list), default=0.0),
            'max_accuracy': max((m.overall_accuracy for m in metrics_list), default=0.0),
        }
        
        if valid_results:
            error_counts = {}
            warning_counts = {}
            
            for result, _ in results:
                for error in result.errors:
                    error_counts[error] = error_counts.get(error, 0) + 1
                for warning in result.warnings:
                    warning_counts[warning] = warning_counts.get(warning, 0) + 1
            
            stats['common_errors'] = error_counts
            stats['common_warnings'] = warning_counts
        
        return stats